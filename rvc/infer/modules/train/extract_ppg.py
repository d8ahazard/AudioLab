import argparse
import gc
import os
import traceback
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from tqdm import tqdm

from handlers.config import model_path
from rvc.whisper.model import Whisper, ModelDimensions
from rvc.whisper.audio import load_audio, pad_or_trim, log_mel_spectrogram

# Ensure HF models cache in the same location
hf_dir = os.path.join(model_path, "hf")
os.makedirs(hf_dir, exist_ok=True)
os.environ["HF_HOME"] = hf_dir

try:
    from pyannote.audio import Pipeline
except ImportError:
    Pipeline = None
    print("pyannote.audio not installed. Diarization will not work.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--version", type=str, default="v2")
    parser.add_argument("--i_part", type=int, default=0)
    parser.add_argument("--n_part", type=int, default=1)
    # If you want to enable diarization via a CLI flag:
    parser.add_argument("--do_diarization", action="store_true",
                        help="Run speaker diarization after extracting PPGs.")
    return parser.parse_args()


def load_model(path) -> Whisper:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(path, map_location="cpu")
    dims = ModelDimensions(**checkpoint["dims"])
    print(dims)
    model = Whisper(dims)
    # the original whisper decoder is removed
    del model.decoder
    # truncate the encoder
    cut = len(model.encoder.blocks) // 4
    cut = -1 * cut
    del model.encoder.blocks[cut:]
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    model.half()
    model.to(device)
    return model


def pred_ppg(whisper, wav_path, ppg_path):
    """
    Extracts PPG features by running audio through
    the truncated encoder, then saves to .npy.
    """
    audio = load_audio(wav_path)
    audio_length = audio.shape[0]
    # The original code uses audio_length // 320 to match strides
    ppg_length = audio_length // 320

    # Pad or trim for fixed-size Whisper input
    audio = pad_or_trim(audio)

    # Prepare mel
    mel = log_mel_spectrogram(audio).half().to(whisper.device)

    with torch.no_grad():
        # Forward pass through the truncated encoder
        ppg = whisper.encoder(mel.unsqueeze(0)).squeeze().data.cpu().float().numpy()
        ppg = ppg[:ppg_length, ]  # [length, dim=1280]
        np.save(ppg_path, ppg, allow_pickle=False)


def extract_ppg_features(exp_dir, n_part=1, i_part=0, callback: Callable = None):
    """
    Iterates over a directory of .wav files in exp_dir/1_16k_wavs,
    extracts PPG features, and saves .npy files into exp_dir/4_ppg1280.
    Splits the file list into n_part parts, taking only the i_part subset.
    """
    input_wavs_path = Path(exp_dir) / "1_16k_wavs"
    out_path = Path(exp_dir) / "4_ppg1280"
    os.makedirs(out_path, exist_ok=True)
    files = sorted(list(Path(input_wavs_path).glob("**/*.wav")))[i_part::n_part]

    whisper_model_path = os.path.join(model_path, "rvc", "large-v3.pt")
    whisper = load_model(whisper_model_path)
    filtered_wav_paths = [f for f in files if
                          not os.path.exists(out_path / f.relative_to(input_wavs_path).with_suffix(".npy"))]
    for wav_path in filtered_wav_paths:
        if callback is not None:
            # TODO: Implement callback
            print("Implement callback for extract_ppg_features needed.")
        try:
            save_path = out_path / wav_path.relative_to(input_wavs_path).with_suffix(".npy")
            pred_ppg(whisper, wav_path, save_path)
        except:
            print(traceback.format_exc())


def diarize_and_create_speaker_mapping(exp_dir, callback: Callable = None):
    """
    Uses pyannote.audio's speaker diarization pipeline to determine the
    'dominant speaker' for each .wav file, then creates name2id.json
    mapping from 'basename' -> integer speaker ID.

    If your dataset truly has multiple speakers per file and you need to split
    them, you’d have to re-segment your audio and store them separately.
    This example lumps each file under only one 'dominant speaker' ID.
    """
    if Pipeline is None:
        print("pyannote.audio not available; skipping diarization.")
        return

    print("Initializing pyannote speaker diarization pipeline...")
    pipeline = Pipeline.from_pretrained("tensorlake/speaker-diarization-3.1")
    input_wavs_path = Path(exp_dir) / "1_16k_wavs"

    # Global speaker dictionary: speaker label -> speaker ID
    # Because the same speaker label might appear in multiple files,
    # we want consistent numbering across the entire dataset.
    global_speaker_map = {}
    global_speaker_idx = 0

    # The final mapping of file basename -> speaker ID
    spk_mapping_file = Path(exp_dir) / "spk_map.json"
    name2id = {}

    if os.path.exists(spk_mapping_file):
        with spk_mapping_file.open("r", encoding="utf-8") as f:
            name2id = json.load(f)

    wav_files = sorted(input_wavs_path.rglob("*.wav"))
    wav_files = [f for f in wav_files if f.stem not in name2id]

    for wav_path in wav_files:
        if callback is not None:
            # TODO: Implement callback
            print("Implement callback for diarize_and_create_speaker_mapping needed.")

        base_name = wav_path.stem

        # Apply the pipeline
        diarization = pipeline(wav_path)

        # Sum coverage for each speaker label
        coverage = {}
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            duration = segment.end - segment.start
            coverage[speaker] = coverage.get(speaker, 0) + duration

        if not coverage:
            # no speech at all? default to speaker id 0
            name2id[base_name] = 0
            continue

        # pick the speaker with the largest coverage
        primary_speaker = max(coverage, key=coverage.get)

        # if this speaker hasn't been assigned a global ID, assign it
        if primary_speaker not in global_speaker_map:
            global_speaker_map[primary_speaker] = global_speaker_idx
            global_speaker_idx += 1

        # store the file -> speaker ID
        name2id[base_name] = global_speaker_map[primary_speaker]

    # Write out the name2id file
    spk_mapping_file = Path(exp_dir) / "spk_map.json"
    with spk_mapping_file.open("w", encoding="utf-8") as f:
        json.dump(name2id, f, indent=4)
    # Delete pipeline
    del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Speaker mapping created at: {spk_mapping_file}")
