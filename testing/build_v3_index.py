"""
Build a V3-native retrieval index from project features.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from handlers.config import model_path, output_path
from modules.rvc_v3.configs.v3_config import RVCV3Config
from modules.rvc_v3.training.build_index import IndexBuilder


def main() -> int:
    parser = argparse.ArgumentParser(description="Build V3 retrieval index from project features")
    parser.add_argument("--features-project", type=str, default="Maynard_Full")
    parser.add_argument("--config-project", type=str, default="Maynard_Full_v3_smoke")
    parser.add_argument("--out-stem", type=str, default=None, help="Output stem path without extension")
    args = parser.parse_args()

    feature_project_dir = Path(output_path) / "voices" / args.features_project
    cfg_path = Path(output_path) / "voices" / args.config_project / "config_v3.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config_v3.json at {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = RVCV3Config.from_dict(json.load(f))

    out_stem = (
        Path(args.out_stem)
        if args.out_stem
        else (Path(model_path) / "trained" / "v3" / f"{args.config_project}_native_retrieval")
    )
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    builder = IndexBuilder(
        feature_dim=cfg.get_content_feature_dim(),
        index_type="IVF",
        n_clusters=256,
        use_gpu=True,
    )
    built = builder.build_from_project(
        project_dir=str(feature_project_dir),
        config=cfg,
        output_name=out_stem.stem,
    )
    built_path = Path(built)
    # If build_from_project wrote into project dir, copy stem files to trained/v3 destination.
    if built_path.parent.resolve() != out_stem.parent.resolve() or built_path.stem != out_stem.stem:
        for ext in [".index", ".npy", ".meta.npz"]:
            src = built_path.with_suffix(ext)
            if src.exists():
                dst = out_stem.with_suffix(ext)
                dst.write_bytes(src.read_bytes())
    print(f"[index] wrote {out_stem.with_suffix('.index')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

