"""
RVC v3 training smoke test (Windows-friendly).

Avoids the Gradio UI pipeline (which imports fairseq) by directly reusing already
preprocessed RVC v2-style artifacts under an existing voice project, generating a
filelist.txt for a new smoke project, and invoking the v3 training wrapper.

This script now performs end-to-end verification:
1) train v3
2) assert artifacts
3) infer v3 output
4) infer v2 baseline output
5) compute objective similarity metrics + write A/B artifacts + JSON report
"""

from __future__ import annotations

import logging
import warnings

# Suppress noisy torchaudio FFmpeg extension DEBUG logs and autocast FutureWarnings
logging.getLogger("torio").setLevel(logging.WARNING)
logging.getLogger("torio._extension.utils").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.autocast.*")

import glob
import json
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, stft


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class TeeWriter:
    """Write to both stdout and a log file."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        self._file = open(log_path, "w", encoding="utf-8")

    def write(self, data: str):
        sys.__stdout__.write(data)
        self._file.write(data)
        self._file.flush()

    def flush(self):
        sys.__stdout__.flush()
        self._file.flush()

    def close(self):
        self._file.close()
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def _set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def _assert_file(path: str, label: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {label}: {path}")


def _assert_dir(path: str, label: str) -> None:
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Missing {label}: {path}")


def _ensure_mono(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return y.astype(np.float32)
    return np.mean(y, axis=1).astype(np.float32)


def _load_audio(path: str) -> tuple[np.ndarray, int]:
    y, sr = sf.read(path)
    return _ensure_mono(y), int(sr)


def _estimate_f0_series(y: np.ndarray, sr: int, hop: int = 256, frame: int = 1024) -> np.ndarray:
    fmin = 50.0
    fmax = 1100.0
    min_lag = max(1, int(sr / fmax))
    max_lag = max(min_lag + 1, int(sr / fmin))
    out = []
    for start in range(0, max(1, len(y) - frame), hop):
        w = y[start : start + frame]
        if w.shape[0] < frame:
            break
        win = np.hanning(frame).astype(np.float32)
        ww = w * win
        rms = float(np.sqrt(np.mean(np.square(ww), dtype=np.float64)))
        if rms < 1e-4:
            out.append(0.0)
            continue
        ac = np.correlate(ww, ww, mode="full")[frame - 1 :]
        ac[:min_lag] = 0.0
        search = ac[min_lag:max_lag]
        if search.size == 0:
            out.append(0.0)
            continue
        lag = int(np.argmax(search)) + min_lag
        out.append(float(sr / max(lag, 1)))
    return np.asarray(out, dtype=np.float32)


def _audio_similarity_metrics(v2_wav: str, v3_wav: str) -> Dict[str, Any]:
    v2_native, v2_sr_native = _load_audio(v2_wav)
    v3_native, v3_sr_native = _load_audio(v3_wav)
    if v2_native.size == 0 or v3_native.size == 0:
        raise RuntimeError("Empty audio encountered while computing quality metrics")

    v2 = resample_poly(v2_native, 16000, v2_sr_native).astype(np.float32)
    v3 = resample_poly(v3_native, 16000, v3_sr_native).astype(np.float32)
    min_len = min(v2.shape[0], v3.shape[0])
    v2a = v2[:min_len]
    v3a = v3[:min_len]

    n_fft = 1024
    hop = 256
    _, _, s2 = stft(v2a, fs=16000, nperseg=n_fft, noverlap=n_fft - hop, boundary=None)
    _, _, s3 = stft(v3a, fs=16000, nperseg=n_fft, noverlap=n_fft - hop, boundary=None)
    s2 = np.abs(s2)
    s3 = np.abs(s3)
    t = min(s2.shape[1], s3.shape[1])
    s2 = s2[:, :t]
    s3 = s3[:, :t]
    stft_l1 = float(np.mean(np.abs(s2 - s3)))
    stft_l1_norm = float(stft_l1 / (float(np.mean(np.abs(s2))) + 1e-8))

    f0_2 = _estimate_f0_series(v2a, sr=16000, hop=hop, frame=1024)
    f0_3 = _estimate_f0_series(v3a, sr=16000, hop=hop, frame=1024)
    vf = np.isfinite(f0_2) & np.isfinite(f0_3) & (f0_2 > 1.0) & (f0_3 > 1.0)
    if int(np.sum(vf)) >= 20:
        f0_corr = float(np.corrcoef(f0_2[vf], f0_3[vf])[0, 1])
    else:
        f0_corr = None

    rms2 = float(np.sqrt(np.mean(np.square(v2a), dtype=np.float64)))
    rms3 = float(np.sqrt(np.mean(np.square(v3a), dtype=np.float64)))
    rms_db_diff = float(abs(20.0 * np.log10(max(rms3, 1e-8) / max(rms2, 1e-8))))

    duration_v2 = float(v2_native.shape[0] / float(v2_sr_native))
    duration_v3 = float(v3_native.shape[0] / float(v3_sr_native))
    duration_ratio = float(duration_v3 / max(duration_v2, 1e-8))
    v3_peak = float(np.max(np.abs(v3_native)))

    return {
        "stft_l1": stft_l1,
        "stft_l1_norm": stft_l1_norm,
        "f0_corr": f0_corr,
        "rms_db_diff": rms_db_diff,
        "duration_v2_sec": duration_v2,
        "duration_v3_sec": duration_v3,
        "duration_ratio_v3_to_v2": duration_ratio,
        "v3_peak_abs": v3_peak,
    }


def _resolve_v2_model_path(model_name: str, model_root: str) -> str:
    if os.path.isfile(model_name):
        return model_name
    if model_name.lower().endswith(".pth"):
        candidate = os.path.join(model_root, "trained", model_name)
    else:
        candidate = os.path.join(model_root, "trained", f"{model_name}.pth")
    _assert_file(candidate, "v2 baseline model checkpoint")
    return candidate


def _discover_v2_reference_wav(model_name: str, output_root: str) -> Optional[str]:
    manual_ref = os.environ.get("SMOKE_V2_REFERENCE_WAV", "").strip()
    if manual_ref:
        return manual_ref if os.path.isfile(manual_ref) else None

    patterns = [
        os.path.join(output_root, "process", "**", "cloned", "*.wav"),
        os.path.join(output_root, "voices", "**", "cloned", "*.wav"),
    ]
    needle = model_name.lower().replace(".pth", "")
    candidates: list[str] = []
    for pattern in patterns:
        for wav_path in glob.glob(pattern, recursive=True):
            base = os.path.basename(wav_path).lower()
            if "(cloned)" in base and needle in base:
                candidates.append(wav_path)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _generate_v2_baseline_clone(
    model_path_pth: str,
    source_wav: str,
    exp_dir: str,
    model_display_name: str,
) -> Optional[str]:
    try:
        from modules.rvc.configs.config import Config
        from modules.rvc.infer.modules.vc.pipeline import VC
    except Exception:
        return None

    vc = VC(Config(), True)
    outputs = vc.vc_multi(
        model=model_path_pth,
        sid=0,
        paths=[source_wav],
        f0_up_key=0,
        f0_method="rmvpe+",
        index_rate=1.0,
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
        project_dir=exp_dir,
        model_display_name=model_display_name,
        callback=None,
        use_model_warmup=False,
        warmup_duration=0.0,
    )
    if not outputs:
        return None
    baseline = outputs[0]
    return baseline if os.path.isfile(baseline) else None


def main() -> int:
    import torch
    from handlers.config import model_path, output_path
    from modules.rvc.utils import HParams
    from modules.rvc_v3.training.train_wrapper import train_rvc_v3
    from testing.smoke_infer_v3 import infer_v3_to_wav

    project_name = os.environ.get("SMOKE_PROJECT", "Maynard_Full_v3_smoke")
    base_project = os.environ.get("SMOKE_BASE_PROJECT", "Maynard_Full")
    n_items_raw = os.environ.get("SMOKE_N_ITEMS", "200").strip()
    n_items = int(n_items_raw) if n_items_raw else 200
    seed = int(os.environ.get("SMOKE_SEED", "20260302"))

    epochs = int(os.environ.get("SMOKE_EPOCHS", "300"))
    run_background = os.environ.get("SMOKE_RUN_BACKGROUND", "").lower() in ("1", "true", "yes")
    timeout_hours = float(os.environ.get("SMOKE_TIMEOUT_HOURS", "0"))  # 0 = no timeout

    if run_background:
        exp_dir_preview = os.path.join(output_path, "voices", project_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(exp_dir_preview, f"smoke_train_{timestamp}.log")
        os.makedirs(exp_dir_preview, exist_ok=True)
        cmd = [sys.executable, "-m", "testing.smoke_train_v3"]
        env = os.environ.copy()
        env.pop("SMOKE_RUN_BACKGROUND", None)
        with open(log_path, "w", encoding="utf-8") as logf:
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=logf,
                stderr=subprocess.STDOUT,
                cwd=ROOT_DIR,
            )
        timeout_sec = int(timeout_hours * 3600) if timeout_hours > 0 else None
        print(f"[smoke] Running in background (PID {proc.pid})")
        print(f"[smoke] Log file: {log_path}")
        print(f"[smoke] Monitor: tail -f \"{log_path}\"")
        if timeout_sec:
            proc.wait(timeout=timeout_sec)
        else:
            proc.wait()
        return proc.returncode

    exp_dir_early = os.path.join(output_path, "voices", project_name)
    os.makedirs(exp_dir_early, exist_ok=True)
    log_path = os.environ.get("SMOKE_LOG_FILE")
    if not log_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(exp_dir_early, f"smoke_train_{timestamp}.log")
    tee = TeeWriter(log_path)
    sys.stdout = tee
    print(f"[smoke] Log: {log_path} (SMOKE_TIMEOUT_HOURS=0 for no limit)")
    print(f"[smoke] requested_n_items={n_items} (SMOKE_N_ITEMS={n_items_raw!r})")

    try:
        return _run_smoke_main(output_path, model_path, project_name, base_project, n_items, seed, epochs)
    finally:
        sys.stdout = sys.__stdout__
        tee.close()


def _run_smoke_main(
    output_path: str,
    model_path: str,
    project_name: str,
    base_project: str,
    n_items: int,
    seed: int,
    epochs: int,
) -> int:
    import torch
    from modules.rvc.utils import HParams
    from modules.rvc_v3.training.train_wrapper import train_rvc_v3
    from testing.smoke_infer_v3 import infer_v3_to_wav

    batch_size = int(os.environ.get("SMOKE_BATCH", "2"))
    v2_baseline_model = os.environ.get("SMOKE_V2_MODEL", "Maynard_Full_v190")

    max_stft_l1_norm = float(os.environ.get("SMOKE_MAX_STFT_L1_NORM", "1.10"))
    min_f0_corr = float(os.environ.get("SMOKE_MIN_F0_CORR", "0.15"))
    max_rms_db_diff = float(os.environ.get("SMOKE_MAX_RMS_DB_DIFF", "8.0"))
    min_duration_ratio = float(os.environ.get("SMOKE_MIN_DURATION_RATIO", "0.67"))
    max_duration_ratio = float(os.environ.get("SMOKE_MAX_DURATION_RATIO", "1.50"))
    min_v3_duration_sec = float(os.environ.get("SMOKE_MIN_V3_DURATION_SEC", "0.50"))

    _set_deterministic_seed(seed)

    base_dir = os.path.join(output_path, "voices", base_project)
    base_gt = os.path.join(base_dir, "0_gt_wavs")
    base_fea = os.path.join(base_dir, "3_feature768")
    base_f0 = os.path.join(base_dir, "2a_f0")
    base_f0nsf = os.path.join(base_dir, "2b-f0nsf")

    for d, label in (
        (base_gt, "base gt wavs"),
        (base_fea, "base feature dir"),
        (base_f0, "base f0 dir"),
        (base_f0nsf, "base f0nsf dir"),
    ):
        _assert_dir(d, label)

    def basenames_in(dir_path: str, suffix: str) -> set[str]:
        out = set()
        for p in glob.glob(os.path.join(dir_path, f"*{suffix}")):
            out.add(os.path.basename(p)[: -len(suffix)])
        return out

    # Intersect names that exist in all required dirs.
    gt_names = basenames_in(base_gt, ".wav")
    fea_names = basenames_in(base_fea, ".npy")
    f0_names = basenames_in(base_f0, ".wav.npy")
    f0nsf_names = basenames_in(base_f0nsf, ".wav.npy")
    all_names = sorted(gt_names & fea_names & f0_names & f0nsf_names)
    available_items = len(all_names)
    if available_items == 0:
        raise RuntimeError("No intersecting preprocessed items found to train on.")
    allow_partial = os.environ.get("SMOKE_ALLOW_PARTIAL_DATA", "0").strip().lower() in ("1", "true", "yes")
    if n_items > 0 and available_items < n_items and not allow_partial:
        raise RuntimeError(
            "Insufficient intersecting training items for requested smoke run: "
            f"requested={n_items}, available={available_items} "
            f"(gt={len(gt_names)}, feat={len(fea_names)}, f0={len(f0_names)}, f0nsf={len(f0nsf_names)}). "
            "Fix preprocessing coverage, reduce SMOKE_N_ITEMS intentionally, or set "
            "SMOKE_ALLOW_PARTIAL_DATA=1 to override."
        )
    names = all_names[:n_items] if n_items > 0 else all_names
    eval_name = names[0]
    eval_wav = os.path.join(base_gt, f"{eval_name}.wav")
    _assert_file(eval_wav, "evaluation source wav")

    # Create smoke project dir and filelist.txt pointing at base_project artifacts
    exp_dir = os.path.join(output_path, "voices", project_name)
    lyrics_dir = os.path.join(exp_dir, "lyrics")
    os.makedirs(lyrics_dir, exist_ok=True)

    # V3: Transcribe base project vocals for text conditioning (smoke lyrics)
    try:
        from modules.rvc_v3.data_prep.transcriber import Transcriber
        transcriber = Transcriber(output_dir=base_dir, model_size="base")
        for stem in names:
            wav_path = os.path.join(base_gt, f"{stem}.wav")
            out_path = os.path.join(lyrics_dir, f"{stem}.json")
            if not os.path.isfile(out_path):
                segs = transcriber.transcribe_file(
                    wav_path, out_path, language=None, word_timestamps=True
                )
                if segs:
                    print(f"[smoke] transcribed {stem} -> {len(segs)} segments")
    except Exception as tr_err:
        print(f"[smoke] WARNING: transcription skipped (training will run without lyrics): {tr_err}")
    os.makedirs(exp_dir, exist_ok=True)
    filelist = os.path.join(exp_dir, "filelist.txt")

    with open(filelist, "w", encoding="utf-8") as f:
        for n in names:
            wav = os.path.join(base_gt, f"{n}.wav")
            fea = os.path.join(base_fea, f"{n}.npy")
            f0 = os.path.join(base_f0, f"{n}.wav.npy")
            f0nsf = os.path.join(base_f0nsf, f"{n}.wav.npy")
            f.write(f"{wav}|{fea}|{f0}|{f0nsf}|0\n")

    print(f"[smoke] base_project={base_project}")
    print(f"[smoke] project={project_name}")
    print(
        f"[smoke] items={len(names)} (available={available_items}, requested={n_items}, "
        f"gt={len(gt_names)}, feat={len(fea_names)}, f0={len(f0_names)}, f0nsf={len(f0nsf_names)})"
    )
    print(f"[smoke] filelist={filelist}")
    print(f"[smoke] eval_wav={eval_wav}")
    print(f"[smoke] epochs={epochs} batch={batch_size} seed={seed}")

    # Load v2-style hparams from existing RVC config (v3/48k.json), then override for smoke.
    cfg_path = os.path.join(ROOT_DIR, "modules", "rvc", "configs", "v3", "48k.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    hparams = HParams(**cfg)
    hparams.name = project_name
    hparams.model_dir = hparams.experiment_dir = exp_dir
    hparams.sample_rate = "48k"
    hparams.if_f0 = 1
    hparams.data.training_files = filelist

    hparams.train.batch_size = batch_size
    hparams.train.epochs = epochs

    hparams.pretrainG = os.path.join(model_path, "rvc", "pretrained_v2", "f0G48k.pth")
    hparams.pretrainD = os.path.join(model_path, "rvc", "pretrained_v2", "f0D48k.pth")

    report: Dict[str, Any] = {
        "project_name": project_name,
        "base_project": base_project,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "n_items": len(names),
        "eval_wav": eval_wav,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "thresholds": {
            "max_stft_l1_norm": max_stft_l1_norm,
            "min_f0_corr": min_f0_corr,
            "max_rms_db_diff": max_rms_db_diff,
            "min_duration_ratio": min_duration_ratio,
            "max_duration_ratio": max_duration_ratio,
            "min_v3_duration_sec": min_v3_duration_sec,
        },
        "checks": {},
        "artifacts": {},
        "metrics": {},
        "pass": False,
    }

    train_rvc_v3(hparams, progress=None)

    v3_trained_dir = os.path.join(model_path, "trained", "v3")
    final_pth = os.path.join(v3_trained_dir, f"{project_name}.pth")
    final_safe = os.path.join(v3_trained_dir, f"{project_name}.safetensors")
    # Convert .pth to .safetensors if training produced .pth (e.g. in-progress run)
    if os.path.exists(final_pth) and not os.path.exists(final_safe):
        try:
            import subprocess
            conv_script = os.path.join(ROOT_DIR, "scripts", "convert_v3_pth_to_safetensors.py")
            subprocess.run(
                [sys.executable, conv_script, final_pth],
                check=True,
                cwd=ROOT_DIR,
            )
        except Exception as conv_err:
            print(f"[smoke] Warning: could not convert .pth to safetensors: {conv_err}")

    config_v3 = os.path.join(exp_dir, "config_v3.json")
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    final_v3_model = final_safe if os.path.exists(final_safe) else final_pth
    _assert_file(config_v3, "config_v3.json")
    _assert_dir(ckpt_dir, "checkpoint dir")
    _assert_file(final_v3_model, "final v3 model")
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "checkpoint_epoch_*.safetensors")))
    if not ckpts:
        ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "checkpoint_epoch_*.pt")))
    if not ckpts:
        raise RuntimeError("No epoch checkpoints found after training")

    report["checks"]["artifacts_ok"] = True
    report["artifacts"]["config_v3"] = config_v3
    report["artifacts"]["checkpoint_dir"] = ckpt_dir
    report["artifacts"]["checkpoint_count"] = len(ckpts)
    report["artifacts"]["final_v3_model"] = final_v3_model

    smoke_dir = os.path.join(exp_dir, "smoke_compare")
    os.makedirs(smoke_dir, exist_ok=True)

    v3_out = os.path.join(smoke_dir, "v3_clone.wav")
    infer_info = infer_v3_to_wav(
        project_name=project_name,
        out_wav=v3_out,
        checkpoint_path=final_v3_model,
        filelist_path=filelist,
        source_audio_path=eval_wav,
    )
    _assert_file(v3_out, "v3 smoke output wav")
    report["artifacts"]["v3_output_wav"] = v3_out
    report["checks"]["v3_infer_ok"] = True

    v2_model_path = _resolve_v2_model_path(v2_baseline_model, model_path)
    v2_model_name = os.path.splitext(os.path.basename(v2_model_path))[0]
    v2_ref = _generate_v2_baseline_clone(
        model_path_pth=v2_model_path,
        source_wav=eval_wav,
        exp_dir=exp_dir,
        model_display_name=v2_model_name,
    )
    if not v2_ref:
        v2_ref = _discover_v2_reference_wav(v2_model_name, output_path)
    if not v2_ref:
        raise RuntimeError(
            "Could not produce or discover a V2 cloned sample for baseline comparison. "
            "Set SMOKE_V2_REFERENCE_WAV to an existing cloned V2 wav."
        )
    _assert_file(v2_ref, "v2 baseline reference wav")

    v2_out = os.path.join(smoke_dir, "v2_clone.wav")
    shutil.copy2(v2_ref, v2_out)
    report["artifacts"]["v2_model"] = v2_model_path
    report["artifacts"]["v2_reference_source"] = v2_ref
    report["artifacts"]["v2_output_wav"] = v2_out
    report["checks"]["v2_infer_ok"] = True

    metrics = _audio_similarity_metrics(v2_out, v3_out)
    report["metrics"] = metrics

    f0_corr = metrics["f0_corr"]
    checks = {
        "duration_ratio_ok": min_duration_ratio <= metrics["duration_ratio_v3_to_v2"] <= max_duration_ratio,
        "v3_duration_ok": metrics["duration_v3_sec"] >= min_v3_duration_sec,
        "rms_ok": metrics["rms_db_diff"] <= max_rms_db_diff,
        "stft_ok": metrics["stft_l1_norm"] <= max_stft_l1_norm,
        "f0_corr_ok": (f0_corr is None) or (f0_corr >= min_f0_corr),
        "v3_peak_ok": metrics["v3_peak_abs"] > 1e-4 and metrics["v3_peak_abs"] <= 1.1,
    }
    report["checks"].update(checks)
    report["artifacts"]["ab_dir"] = smoke_dir
    report["artifacts"]["infer_info"] = infer_info

    report["pass"] = all(bool(v) for v in report["checks"].values())
    report_path = os.path.join(exp_dir, "smoke_v3_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[smoke] report={report_path}")
    print(f"[smoke] checks={report['checks']}")
    print(f"[smoke] metrics={report['metrics']}")

    if not report["pass"]:
        failed = [k for k, v in report["checks"].items() if not v]
        raise RuntimeError(f"Smoke validation failed checks: {failed}")

    print("[smoke] done (PASS)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[smoke] interrupted")
        raise
    except Exception as e:
        print(f"[smoke] ERROR: {e}")
        traceback.print_exc()
        raise

