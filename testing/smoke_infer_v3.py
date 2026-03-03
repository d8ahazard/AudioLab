"""
RVC v3 inference smoke utilities (no fairseq).

Can be run as a script or imported by smoke_train_v3.py for end-to-end validation.
"""

from __future__ import annotations

import logging
import warnings

# Suppress noisy torchaudio FFmpeg extension DEBUG and autocast FutureWarnings
logging.getLogger("torio").setLevel(logging.WARNING)
logging.getLogger("torio._extension.utils").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.autocast.*")

import json
import os
import sys

import numpy as np


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def infer_v3_to_wav(
    project_name: str,
    out_wav: str,
    checkpoint_path: str | None = None,
    filelist_path: str | None = None,
    lyrics: str | None = None,
    source_audio_path: str | None = None,
) -> dict:
    import torch
    from scipy.io.wavfile import write as wavwrite

    from handlers.config import model_path, output_path
    from modules.rvc.utils import HParams
    from modules.rvc_v3.configs.v3_config import RVCV3Config
    from modules.rvc_v3.models.generator import RVCV3Generator
    from modules.rvc_v3.models.text_encoder import TextEncoder
    from modules.rvc.infer.lib.train.data_utils import TextAudioLoaderMultiNSFsid

    device = "cuda" if torch.cuda.is_available() else "cpu"

    v3_dir = os.path.join(model_path, "trained", "v3")
    default_base = os.path.join(v3_dir, project_name)
    safe_path = default_base + ".safetensors"
    pth_path = default_base + ".pth"
    if checkpoint_path:
        ckpt_path = checkpoint_path
    elif os.path.exists(safe_path):
        ckpt_path = safe_path
    else:
        ckpt_path = pth_path
    exp_dir = os.path.join(output_path, "voices", project_name)
    filelist = filelist_path or os.path.join(exp_dir, "filelist.txt")

    # Resolve path: try .safetensors first
    ckpt_base = ckpt_path
    if ckpt_base.endswith(".pth"):
        ckpt_base = ckpt_base[:-4]
    elif ckpt_base.endswith(".safetensors"):
        ckpt_base = ckpt_base[:-11]
    safe_candidate = ckpt_base + ".safetensors"
    pth_candidate = ckpt_base + ".pth"
    if os.path.exists(safe_candidate):
        ckpt_path = safe_candidate
    elif os.path.exists(pth_candidate):
        ckpt_path = pth_candidate
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing trained v3 checkpoint: {ckpt_path} (tried {safe_candidate}, {pth_candidate})")
    if not os.path.exists(filelist):
        raise FileNotFoundError(f"Missing filelist: {filelist}")

    # Load config + generator weights (safetensors or pth)
    from modules.rvc_v3.io.checkpoint_io import load_inference_model

    gen_sd, text_sd, meta = load_inference_model(ckpt_path, device="cpu")
    config = RVCV3Config.from_dict(meta.get("config", {}))

    gen = RVCV3Generator(
        spec_channels=config.spec_channels,
        segment_size=config.segment_size // config.hop_length,
        inter_channels=config.inter_channels,
        hidden_channels=config.hidden_channels,
        filter_channels=config.filter_channels,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        kernel_size=config.kernel_size,
        p_dropout=config.p_dropout,
        resblock=config.resblock,
        resblock_kernel_sizes=config.resblock_kernel_sizes,
        resblock_dilation_sizes=config.resblock_dilation_sizes,
        upsample_rates=config.upsample_rates,
        upsample_initial_channel=config.upsample_initial_channel,
        upsample_kernel_sizes=config.upsample_kernel_sizes,
        spk_embed_dim=config.spk_embed_dim,
        gin_channels=config.gin_channels,
        sr=config.sampling_rate,
        vocoder_type=config.vocoder_type,
        ppg_dim=config.get_content_feature_dim(),
        text_encoder_dim=config.text_encoder_dim,
        n_cross_attn_layers=config.n_cross_attn_layers,
    )

    gen.load_state_dict(gen_sd, strict=False)
    gen = gen.to(device).eval()

    # Text encoder for lyrics conditioning
    text_encoder = None
    text_features = None
    text_mask = None
    if text_sd:
        text_encoder = TextEncoder(
            vocab_size=200,
            d_model=config.text_encoder_dim,
            nhead=config.text_encoder_heads,
            num_layers=config.text_encoder_layers,
            dim_feedforward=config.text_encoder_ff_dim,
            dropout=config.text_dropout,
        )
        text_encoder.load_state_dict(text_sd, strict=False)
        text_encoder = text_encoder.to(device).eval()

    # Resolve lyrics: explicit, or transcribe source, or load from lyrics dir for first filelist item
    resolved_lyrics = lyrics
    if not resolved_lyrics and source_audio_path and os.path.isfile(source_audio_path) and text_encoder:
        try:
            from modules.rvc_v3.data_prep.transcriber import Transcriber
            transcriber = Transcriber(output_dir=exp_dir, model_size="base")
            stem = os.path.splitext(os.path.basename(source_audio_path))[0]
            out_json = os.path.join(exp_dir, "lyrics", f"{stem}_infer.json")
            os.makedirs(os.path.dirname(out_json), exist_ok=True)
            segs = transcriber.transcribe_file(source_audio_path, out_json, language=None, word_timestamps=True)
            if segs:
                resolved_lyrics = "[clean] " + " ".join(s.get("text", "") for s in segs)
        except Exception:
            pass
    if not resolved_lyrics:
        # Try lyrics from project dir (first filelist item stem)
        with open(filelist, "r", encoding="utf-8") as f:
            first_line = f.readline()
        if first_line:
            parts = first_line.strip().split("|")
            if parts:
                stem = os.path.splitext(os.path.basename(parts[0]))[0]
                lyrics_file = os.path.join(exp_dir, "lyrics", f"{stem}.json")
                if os.path.isfile(lyrics_file):
                    with open(lyrics_file, "r", encoding="utf-8") as lf:
                        data = json.load(lf)
                    full = data.get("full_text", "") or " ".join(s.get("text", "") for s in data.get("segments", []))
                    if full:
                        resolved_lyrics = "[clean] " + full.strip()

    if resolved_lyrics and text_encoder:
        try:
            from modules.rvc_v3.training.train_wrapper import SimplePhonemizer
            phonemizer = SimplePhonemizer()
            ids = phonemizer.phonemize(resolved_lyrics)
            text_tokens = torch.LongTensor(ids).unsqueeze(0).to(device)
            text_mask = torch.zeros_like(text_tokens, dtype=torch.bool)
            with torch.no_grad():
                text_features = text_encoder(text_tokens, text_mask)
        except Exception:
            pass

    # Build dataset config from standard RVC json so TextAudioLoaderMultiNSFsid can read filelist.
    cfg_path = os.path.join(ROOT_DIR, "modules", "rvc", "configs", "v3", "48k.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        hps = HParams(**json.load(f))
    hps.data.training_files = filelist

    ds = TextAudioLoaderMultiNSFsid(filelist, hps.data)
    spec, wav, phone, pitch, pitchf, sid = ds[0]

    phone = phone.unsqueeze(0).to(device)
    phone_lengths = torch.LongTensor([phone.size(1)]).to(device)
    pitch = pitch.unsqueeze(0).to(device)
    pitchf = pitchf.unsqueeze(0).to(device)
    sid = sid.to(device)

    with torch.no_grad():
        audio, x_mask, _ = gen.infer(
            phone, phone_lengths, pitch, pitchf, sid,
            text_features=text_features,
            text_mask=text_mask,
        )

    y = audio.squeeze().detach().cpu().float().numpy()
    y = np.clip(y, -1.0, 1.0)
    y_i16 = (y * 32767.0).astype(np.int16)
    wavwrite(out_wav, int(config.sampling_rate), y_i16)

    return {
        "project_name": project_name,
        "checkpoint_path": ckpt_path,
        "filelist_path": filelist,
        "output_wav": out_wav,
        "sample_rate": int(config.sampling_rate),
        "num_samples": int(y_i16.shape[0]),
        "duration_sec": float(y_i16.shape[0] / float(config.sampling_rate)),
        "device": device,
    }


def main() -> int:
    project_name = os.environ.get("SMOKE_PROJECT", "Maynard_Full_v3_smoke")
    from handlers.config import output_path
    exp_dir = os.path.join(output_path, "voices", project_name)
    out_wav = os.path.join(exp_dir, "smoke_v3_out.wav")
    info = infer_v3_to_wav(project_name=project_name, out_wav=out_wav)
    print(f"[infer] wrote {info['output_wav']} ({info['duration_sec']:.2f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

