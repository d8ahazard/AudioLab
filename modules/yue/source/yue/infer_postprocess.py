from collections import Counter
import copy
import os

import numpy as np
import soundfile as sf
import torch
import torchaudio
from modules.yue.source.yue.codecmanipulator import CodecManipulator
from modules.yue.source.yue.models.soundstream_hubert_new import SoundStream
from modules.yue.source.yue.post_process_audio import replace_low_freq_with_energy_matched
from modules.yue.source.yue.vocoder import build_codec_model, process_audio

# convert audio tokens to audio
def save_audio(wav: torch.Tensor, path, sample_rate: int, rescale: bool = False):
    folder_path = os.path.dirname(path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    limit = 0.99
    max_val = wav.abs().max()
    wav = wav * min(limit / max_val, 1) if rescale else wav.clamp(-limit, limit)
    torchaudio.save(str(path), wav, sample_rate=sample_rate, encoding="PCM_S", bits_per_sample=16)

def post_process(
    codec_model: SoundStream,
    device: torch.device,
    input_tracks: list,
    output_base_name: str,
    output_dir: str,
    config_path: str,
    vocal_decoder_path: str,
    inst_decoder_path: str,
    rescale: bool,
    hq_audio: bool,
) -> tuple:
    # reconstruct tracks
    recons_output_dir = os.path.join(output_dir, "recons")
    recons_mix_dir = os.path.join(recons_output_dir, "mix")
    os.makedirs(recons_mix_dir, exist_ok=True)

    file_extension = ".wav" if hq_audio else ".mp3"

    track_type_names = ["vtrack", "itrack"]
    track_names = [f"{output_base_name}_{name}" for name in track_type_names]

    output_files = []

    tracks = []
    for itrack, track in enumerate(input_tracks):
        decodec_rlt = []
        decoded_waveform = codec_model.decode(torch.as_tensor(track.astype(np.int16), dtype=torch.long).unsqueeze(0).permute(1, 0, 2).to(device))
        decoded_waveform = decoded_waveform.cpu().squeeze(0)
        decodec_rlt.append(torch.as_tensor(decoded_waveform))
        decodec_rlt = torch.cat(decodec_rlt, dim=-1)
        save_path = os.path.join(recons_output_dir, track_names[itrack] + file_extension)
        tracks.append(save_path)
        save_audio(decodec_rlt, save_path, 16000)

    # mix tracks
    for inst_path in tracks:
        try:
            if (inst_path.endswith(".wav") or inst_path.endswith(".mp3")) and "itrack" in inst_path:
                # find pair
                vocal_path = inst_path.replace("itrack", "vtrack")
                if not os.path.exists(vocal_path):
                    continue
                # mix
                recons_mix = os.path.join(recons_mix_dir, os.path.basename(inst_path).replace("itrack", "mixed"))
                vocal_stem, sr = sf.read(inst_path)
                instrumental_stem, _ = sf.read(vocal_path)
                mix_stem = (vocal_stem + instrumental_stem) / 1
                sf.write(recons_mix, mix_stem, sr)
        except Exception as e:
            print(e)

    # vocoder to upsample audios
    vocal_decoder, inst_decoder = build_codec_model(config_path, vocal_decoder_path, inst_decoder_path)
    vocoder_output_dir = os.path.join(output_dir, "vocoder")
    vocoder_stems_dir = os.path.join(vocoder_output_dir, "stems")
    vocoder_mix_dir = os.path.join(vocoder_output_dir, "mix")
    os.makedirs(vocoder_mix_dir, exist_ok=True)
    os.makedirs(vocoder_stems_dir, exist_ok=True)

    decoders = [vocal_decoder, inst_decoder]
    processed_outputs = []
    for itrack, track in enumerate(tracks):
        output = process_audio(input_tracks[itrack], os.path.join(vocoder_stems_dir, track_names[itrack] + file_extension), rescale, device, decoders[itrack], codec_model)
        processed_outputs.append(output)

    # mix tracks
    try:
        mix_output = processed_outputs[0] + processed_outputs[1]
        vocoder_mix = os.path.join(vocoder_mix_dir, os.path.basename(recons_mix))
        save_audio(mix_output, vocoder_mix, 44100, rescale)
        print(f"Created mix: {vocoder_mix}")
    except RuntimeError as e:
        print(e)
        print(f"mix {vocoder_mix} failed! inst: {processed_outputs[0].shape}, vocal: {processed_outputs[1].shape}")

    # Post process
    mixed_output_name = os.path.join(output_dir, os.path.basename(recons_mix))
    replace_low_freq_with_energy_matched(
        a_file=recons_mix, b_file=vocoder_mix, c_file=mixed_output_name, cutoff_freq=5500.0  # 16kHz  # 48kHz
    )
    output_files.append(mixed_output_name)

    return tuple(output_files)

def fix_output(output):
    # Fix invalid codes (a dirty solution, which may harm the quality of audio)
    # We are trying to find better one
    fixed_output = copy.deepcopy(output)
    for i, line in enumerate(output):
        for j, element in enumerate(line):
            if element < 0 or element > 1023:
                counter = Counter(line)
                most_frequant = sorted(counter.items(), key=lambda x: x[1], reverse=True)[0][0]
                fixed_output[i, j] = most_frequant
    return fixed_output

def encode_stage1(input):
    codec_tool = CodecManipulator("xcodec", 0, 1)
    return codec_tool.ids2npy(input)

def encode_stage2(input):
    codec_tool = CodecManipulator("xcodec", 0, 8)
    return fix_output(codec_tool.ids2npy(input))
