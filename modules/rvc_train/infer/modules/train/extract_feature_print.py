import os
import traceback
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from transformers import HubertModel, Wav2Vec2FeatureExtractor

# Set environment variables
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=Path, required=True)
    parser.add_argument("--i_part", type=int, default=0)
    parser.add_argument("--n_part", type=int, default=1)
    return parser.parse_args()


def setup_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    return device


def readwave(wav_path, normalize=False):
    wav, sr = sf.read(wav_path)
    assert sr == 16000
    feats = torch.from_numpy(wav).float()
    if feats.dim() == 2:  # double channels
        feats = feats.mean(-1)
    assert feats.dim() == 1, feats.dim()
    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    feats = feats.view(1, -1)
    return feats


def main():
    args = parse_args()
    device = setup_device()
    exp_dir = args.exp_dir
    i_part = args.i_part
    n_part = args.n_part

    log_file_path = exp_dir / "extract_f0_feature.log"
    with open(log_file_path, "a+") as f:
        def printt(message):
            print(message)
            f.write(f"{message}\n")
            f.flush()

        model_name = "facebook/hubert-base-ls960"
        printt(f"Experiment directory: {exp_dir}")

        wav_source_dir = exp_dir / "1_16k_wavs"
        output_save_dir = exp_dir / "3_feature768"
        Path(output_save_dir).mkdir(parents=True, exist_ok=True)

        printt(f"Loading model: {model_name}")
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        model = HubertModel.from_pretrained(model_name)
        model = model.to(device)
        printt(f"Model moved to {device}")

        if device not in ["mps", "cpu"]:
            model = model.half()
        model.eval()

        todo = sorted(list(wav_source_dir.iterdir()))[i_part::n_part]
        n = max(1, len(todo) // 10)  # Print at most ten times
        if len(todo) == 0:
            printt("no-feature-todo")
        else:
            printt(f"all-feature-{len(todo)}")
            for idx, file in enumerate(todo):
                try:
                    if file.suffix == ".wav":
                        wav_path = wav_source_dir / file.name
                        out_path = output_save_dir / file.name.replace("wav", "npy")

                        if out_path.exists():
                            continue

                        feats = readwave(wav_path, normalize=True)
                        inputs = feature_extractor(feats.numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
                        inputs = {key: tensor.to(device) for key, tensor in inputs.items()}

                        with torch.no_grad():
                            outputs = model(**inputs)
                            feats = outputs.last_hidden_state.squeeze(0).float().cpu().numpy()

                        if np.isnan(feats).sum() == 0:
                            np.save(out_path, feats, allow_pickle=False)
                        else:
                            printt(f"{file.name}-contains nan")
                        if idx % n == 0:
                            printt(f"all-{len(todo)},now-{idx},{file.name},{feats.shape}")
                except:
                    printt(traceback.format_exc())
            printt("all-feature-done")


if __name__ == "__main__":
    main()
