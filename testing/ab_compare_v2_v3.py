"""
Deterministic V2 vs V3 clone A/B runner.

Runs both backends on the same input vocal file and writes objective stats JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from handlers.config import model_path
from modules.rvc.configs.config import Config
from modules.rvc.infer.modules.vc.pipeline import VC
from modules.rvc_v3.configs.v3_config import RVCV3Config
from modules.rvc_v3.inference.pipeline import RVCV3Pipeline
from modules.rvc_v3.io.checkpoint_io import load_inference_model


def _pick_input_vocal(project_dir: Path) -> Path:
    stems = project_dir / "stems"
    if stems.exists():
        vocals = sorted([p for p in stems.iterdir() if p.is_file() and "(Vocals)" in p.name])
        if vocals:
            return vocals[0]
    source_dir = project_dir / "source"
    if source_dir.exists():
        src = sorted([p for p in source_dir.iterdir() if p.is_file()])
        if src:
            return src[0]
    raise FileNotFoundError(f"No vocal/source wav found in {project_dir}")


def _wav_stats(path: Path) -> Dict[str, Any]:
    y, sr = sf.read(str(path), dtype="float32")
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if y.size == 0:
        return {"path": str(path), "sr": int(sr), "samples": 0}
    return {
        "path": str(path),
        "sr": int(sr),
        "samples": int(y.shape[0]),
        "duration_sec": float(y.shape[0] / sr),
        "peak_abs": float(np.max(np.abs(y))),
        "rms": float(np.sqrt(np.mean(np.square(y), dtype=np.float64))),
        "clipped_frac": float(np.mean(np.abs(y) > 0.99)),
    }


def _resolve_lyrics_text(project_dir: Path, input_wav: Path, lyrics_json: Optional[str]) -> Optional[str]:
    if lyrics_json:
        p = Path(lyrics_json)
        if not p.exists():
            raise FileNotFoundError(f"Lyrics JSON not found: {p}")
    else:
        # Primary location used by clone path:
        # <project>/cloned/<input_stem>_transcript.json
        p = project_dir / "cloned" / f"{input_wav.stem}_transcript.json"
        if not p.exists():
            # Fallback: first transcript-like JSON under cloned/
            cloned_dir = project_dir / "cloned"
            candidates = sorted(cloned_dir.glob("*_transcript.json")) if cloned_dir.exists() else []
            p = candidates[0] if candidates else None
    if p is None or not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    text = (data.get("full_text", "") or "").strip()
    if text:
        return text
    segs = data.get("segments", [])
    if isinstance(segs, list):
        parts = [str(s.get("text", "")).strip() for s in segs if isinstance(s, dict)]
        text = " ".join([t for t in parts if t]).strip()
        if text:
            return text
    return None


def _run_v2(vc: VC, in_wav: Path, out_dir: Path, v2_model: str, index_rate: float, pitch_shift: int) -> Path:
    run_project_dir = out_dir / f"v2_ir{index_rate:.2f}"
    run_project_dir.mkdir(parents=True, exist_ok=True)
    outs = vc.vc_multi(
        model=v2_model,
        sid=0,
        paths=[str(in_wav)],
        f0_up_key=pitch_shift,
        f0_method="rmvpe+",
        index_rate=index_rate,
        filter_radius=3,
        rms_mix_rate=0.9,
        protect=0.2,
        merge_type="median",
        crepe_hop_length=160,
        f0_autotune=False,
        rmvpe_onnx=False,
        clone_stereo=False,
        pitch_correction=False,
        pitch_correction_humanize=0.95,
        project_dir=str(run_project_dir),
        model_display_name=Path(v2_model).stem,
        callback=None,
        use_model_warmup=False,
        warmup_duration=5.0,
    )
    if not outs:
        raise RuntimeError("V2 clone produced no output")
    return Path(outs[0])


def _run_v3(
    v3: RVCV3Pipeline,
    in_wav: Path,
    out_dir: Path,
    index_rate: float,
    pitch_shift: int,
    lyrics: Optional[str],
) -> Path:
    out_file = out_dir / f"v3_ir{index_rate:.2f}" / "cloned" / f"{in_wav.stem}(Cloned)(v3_ir{index_rate:.2f}).wav"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    v3.convert(
        audio_path=str(in_wav),
        lyrics=lyrics,
        output_path=str(out_file),
        index_rate=index_rate,
        pitch_shift=pitch_shift,
        speaker_id=0,
    )
    if not out_file.exists():
        raise RuntimeError("V3 clone produced no output")
    return out_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic V2 vs V3 A/B clone comparison")
    parser.add_argument(
        "--project-dir",
        type=str,
        default=r"E:\dev\AudioLab\outputs\process\212_vox_maynard1_fixed_de8cb7bb",
    )
    parser.add_argument(
        "--v2-model",
        type=str,
        default=str(Path(model_path) / "trained" / "Maynard_Full_v190.pth"),
    )
    parser.add_argument(
        "--v3-model",
        type=str,
        default=str(Path(model_path) / "trained" / "v3" / "Maynard_Full_v3_smoke.safetensors"),
    )
    parser.add_argument(
        "--v3-index",
        type=str,
        default=str(Path(model_path) / "trained" / "v3" / "Maynard_Full_v3_smoke.index"),
    )
    parser.add_argument("--pitch-shift", type=int, default=0)
    parser.add_argument("--lyrics-json", type=str, default=None)
    parser.add_argument("--report", type=str, default=None)
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    out_dir = project_dir / "ab_compare_v2_v3"
    out_dir.mkdir(parents=True, exist_ok=True)

    input_wav = _pick_input_vocal(project_dir)
    lyrics_text = _resolve_lyrics_text(project_dir, input_wav, args.lyrics_json)

    vc = VC(Config(), True)
    vc.get_vc(args.v2_model)

    _, _, meta = load_inference_model(args.v3_model, device="cpu")
    cfg = RVCV3Config.from_dict(meta.get("config", {}))
    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "-1" and torch.cuda.is_available() else "cpu"
    v3 = RVCV3Pipeline(args.v3_model, cfg, device=device)
    if args.v3_index and Path(args.v3_index).exists():
        v3.load_retrieval_index(args.v3_index)

    runs: List[Dict[str, Any]] = []
    matrix: List[Tuple[str, float]] = [("v2", 0.0), ("v2", 0.75), ("v3", 0.0), ("v3", 0.75)]
    for backend, ir in matrix:
        if backend == "v2":
            out = _run_v2(vc, input_wav, out_dir, args.v2_model, ir, args.pitch_shift)
            debug = {}
        else:
            out = _run_v3(v3, input_wav, out_dir, ir, args.pitch_shift, lyrics=lyrics_text)
            debug = getattr(v3, "last_convert_debug", {})
        runs.append(
            {
                "backend": backend,
                "index_rate": ir,
                "input": str(input_wav),
                "output": str(out),
                "stats": _wav_stats(out),
                "debug": debug,
            }
        )

    report_path = Path(args.report) if args.report else (out_dir / "ab_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "project_dir": str(project_dir),
                "input_wav": str(input_wav),
                "lyrics_json": args.lyrics_json,
                "lyrics_used": bool(lyrics_text),
                "lyrics_chars": int(len(lyrics_text)) if lyrics_text else 0,
                "v2_model": args.v2_model,
                "v3_model": args.v3_model,
                "v3_index": args.v3_index if Path(args.v3_index).exists() else None,
                "runs": runs,
            },
            f,
            indent=2,
        )
    print(f"[ab] report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

