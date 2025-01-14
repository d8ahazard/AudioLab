import os
import traceback
import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

import fairseq

from handlers.config import model_path

hf_dir = os.path.join(model_path, "hf")
os.makedirs(hf_dir, exist_ok=True)
# Set HF_HUB_CACHE_DIR to the model_path
os.environ["HF_HOME"] = hf_dir

# Set environment variables for MPS fallback
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=Path, required=True)
    parser.add_argument("--i_part", type=int, default=0)
    parser.add_argument("--n_part", type=int, default=1)
    return parser.parse_args()


def setup_device():
    """
    Decide whether we use CUDA, MPS (Apple Silicon), or CPU.
    """
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    return device


def readwave(wav_path, normalize=False):
    """
    Loads a 16k wav file as a Tensor of shape (1, num_samples).
    If stereo, it averages to mono.
    Optionally layer-normalizes the audio.
    """
    wav, sr = sf.read(wav_path)
    assert sr == 16000, f"Expected 16kHz sample rate, got {sr}."
    feats = torch.from_numpy(wav).float()
    if feats.dim() == 2:  # double channels => average to mono
        feats = feats.mean(-1)
    assert feats.dim() == 1, "Audio tensor must be 1D after any channel mixing."
    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    # Add batch dimension (B=1)
    feats = feats.view(1, -1)
    return feats


def extract_features(exp_dir, i_part=0, n_part=1, callback: Callable = None):
    """
    Uses fairseq to load a HuBERT model from local checkpoint
    (huBERT base .pt). Iterates over .wav files in exp_dir/1_16k_wavs,
    extracts features, and saves them in exp_dir/3_feature768 as .npy.
    Uses i_part/n_part to optionally split the .wav list among multiple workers.
    """
    device = setup_device()
    exp_dir = Path(exp_dir)

    log_file_path = exp_dir / "extract_f0_feature.log"
    with open(log_file_path, "a+") as f:
        def printt(message):
            print(message)
            f.write(f"{message}\n")
            f.flush()

        # Load local fairseq model file: e.g. models/hubert_base.pt
        local_model_path = Path(os.path.join(model_path, "rvc", "hubert_base.pt"))
        # local_model_path = Path(__file__).parents[3] / "models/hubert_base.pt"

        printt(f"Experiment directory: {exp_dir}")
        wav_source_dir = exp_dir / "1_16k_wavs"
        output_save_dir = exp_dir / "3_feature768"
        output_save_dir.mkdir(parents=True, exist_ok=True)

        printt(f"Load model(s) from {local_model_path}")
        if not local_model_path.exists():
            printt(
                f"Error: {local_model_path} does not exist. "
                "You may download it from: https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main"
            )
            return

        # Load fairseq model
        models, saved_cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [str(local_model_path)], suffix=""
        )
        model = models[0].to(device)
        printt(f"Model moved to {device}")
        # Use half precision on CUDA only
        if device not in ["mps", "cpu"]:
            model = model.half()
        model.eval()

        # Figure out which .wav files to process (round-robin split)
        todo = sorted(wav_source_dir.iterdir())[i_part::n_part]
        n_print = max(1, len(todo) // 10) if len(todo) > 0 else 1

        if len(todo) == 0:
            printt("no-feature-todo")
            return

        printt(f"all-feature-{len(todo)}")

        for idx, file in enumerate(todo):
            # TODO: Implement callback usage here
            if callback:
                print(f"Implement callback in extract_feature_print/extract_features: {callback}")
            try:
                if file.suffix.lower() != ".wav":
                    continue

                out_path = output_save_dir / file.with_suffix(".npy").name
                if out_path.exists():
                    continue

                # Load wav with or without normalization
                feats = readwave(file, normalize=saved_cfg.task.normalize)

                # Create a padding mask of all False (no actual padding)
                padding_mask = torch.zeros_like(feats, dtype=torch.bool)

                # Prepare inputs for fairseq's extract_features
                inputs = {
                    "source": feats.half().to(device) if device not in ["mps", "cpu"] else feats.to(device),
                    "padding_mask": padding_mask.to(device),
                    "output_layer": 12,  # which transformer layer to extract from
                }

                with torch.no_grad(), torch.autocast("cuda", enabled=(device == "cuda")):
                    logits = model.extract_features(**inputs)
                    feat_out = logits[0]  # the raw feature tensor

                # Convert to float CPU numpy
                feat_out = feat_out.squeeze(0).float().cpu().numpy()

                # Check for NaNs
                if np.isnan(feat_out).any():
                    printt(f"{file.name}-contains nan")
                    continue

                # Save .npy features
                np.save(out_path, feat_out, allow_pickle=False)

                # Print partial progress
                if idx % n_print == 0:
                    printt(f"all-{len(todo)},now-{idx},{file.name},{feat_out.shape}")

            except Exception:
                printt(traceback.format_exc())

        printt("all-feature-done")
