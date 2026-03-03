#!/usr/bin/env python
"""
Convert RVC v3 .pth checkpoint to .safetensors + _config.json.

Usage:
    python scripts/convert_v3_pth_to_safetensors.py <input.pth> [--output <output_dir>]

When training finishes, run:
    python scripts/convert_v3_pth_to_safetensors.py models/trained/v3/Maynard_Full_v3_smoke.pth

This produces Maynard_Full_v3_smoke.safetensors and Maynard_Full_v3_smoke_config.json.
If {stem}.index exists alongside the input .pth, it is left unchanged (FAISS format).
The index will be used with the new safetensors model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert RVC v3 .pth to safetensors")
    parser.add_argument("input", type=str, help="Path to .pth checkpoint")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output directory (default: same as input)")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    if input_path.suffix.lower() != ".pth":
        print(f"Warning: Expected .pth file, got {input_path.suffix}", file=sys.stderr)

    output_dir = Path(args.output).resolve() if args.output else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    # Index handling
    index_path = input_path.parent / f"{stem}.index"
    if index_path.exists():
        print(f"Index found: {index_path} (unchanged, will be used with new safetensors model)")

    # Load .pth
    import torch
    ckpt = torch.load(input_path, map_location="cpu", weights_only=True)

    # Extract tensors and meta
    generator_sd = ckpt.get("generator", ckpt.get("model", {}))
    text_encoder_sd = ckpt.get("text_encoder", {})
    config = ckpt.get("config", {})
    version = ckpt.get("version", "v3")
    iteration = ckpt.get("iteration", 0)
    lr = ckpt.get("learning_rate", 0.0001)

    if not generator_sd:
        print("Error: No generator or model state dict in checkpoint", file=sys.stderr)
        return 1

    from safetensors.torch import save_file as safe_save_file

    def _flatten(sd, prefix):
        return {f"{prefix}{k}": v for k, v in sd.items()}

    tensors = {}
    tensors.update(_flatten(generator_sd, "generator."))
    tensors.update(_flatten(text_encoder_sd, "text_encoder."))

    safe_path = output_dir / f"{stem}.safetensors"
    config_path = output_dir / f"{stem}_config.json"

    safe_save_file(tensors, safe_path)
    meta = {"config": config, "version": version, "iteration": iteration, "learning_rate": lr}
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Validate conversion can be loaded back with expected structures.
    from modules.rvc_v3.io.checkpoint_io import load_inference_model
    gen_loaded, text_loaded, meta_loaded = load_inference_model(str(safe_path), device="cpu")
    if not gen_loaded:
        print("Error: converted safetensors missing generator state", file=sys.stderr)
        return 1
    if not isinstance(meta_loaded.get("config"), dict):
        print("Error: converted config metadata is invalid", file=sys.stderr)
        return 1
    if text_encoder_sd and not text_loaded:
        print("Error: converted safetensors missing text_encoder state", file=sys.stderr)
        return 1

    size_mb = safe_path.stat().st_size / (1024 * 1024)
    print(f"Converted: {safe_path} ({size_mb:.2f} MB)")
    print(f"Config: {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
