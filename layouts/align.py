# align.py
# Sentence-first alignment of multiple target takes to a master, using WhisperX word timestamps.
# - Uses process_transcription_whisperx() (provided by your codebase) to generate word-level JSON.
# - Groups by sentence via inter-word gaps (default 0.4s), using MASTER as the reference grid.
# - Monotonic sentence matching (edit distance + duration penalty) => confidence scores.
# - Rebuilds each target on a blank timeline pinned to master sentence boundaries with a humanization window.
# - Exports aligned audio, full JSON report, and a colorful overlay PNG.
# - Wires directly into your Gradio UI.

import os
import re
import gc
import json
import time
import math
import shutil
import string
import logging
import subprocess
from typing import List, Dict, Tuple, Optional, Any, TYPE_CHECKING

import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import gradio as gr

# Your existing import: WhisperX wrapper
from layouts.transcribe import process_transcription_whisperx
from handlers.config import output_path
import hashlib
from modules.rtla.config import (
    SAMPLE_RATE as RTLA_SR,
    HOP_LENGTH as RTLA_HOP,
    DTW_WINDOW_SIZE as RTLA_WIN,
    FEATURES as RTLA_FEATURES,
    FRAME_RATE as RTLA_FR,
    CHUNK_SIZE as RTLA_CHUNK,
)
from modules.rtla.stream_processor import StreamProcessor as RTLAStrProc
from modules.rtla.oltw import OLTW
from modules.rtla.utils import make_path_strictly_monotonic
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --------------------------
# Text & token utils
# --------------------------

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)

def normalize_word(w: str) -> str:
    return w.strip().lower().translate(_PUNCT_TABLE)

def normalize_text(text: str) -> str:
    return " ".join(normalize_word(t) for t in text.split() if t.strip())

def levenshtein(a: List[str], b: List[str]) -> int:
    """Classic Levenshtein distance on token lists."""
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,      # deletion
                dp[i, j - 1] + 1,      # insertion
                dp[i - 1, j - 1] + cost  # substitution
            )
    return int(dp[n, m])

# --------------------------
# WhisperX result parsing
# --------------------------

def _is_transcript_json(path: str) -> bool:
    if not path.endswith(".json"):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        # WhisperX-like: has "segments" with "words"
        if isinstance(obj, dict) and "segments" in obj:
            segs = obj.get("segments", [])
            if segs and isinstance(segs[0], dict):
                if "words" in segs[0]:
                    return True
        return False
    except Exception:
        return False

def _map_outputs_by_basename(outputs: List[str]) -> Dict[str, List[str]]:
    """Group output files by the base audio filename (without extension)."""
    by_base: Dict[str, List[str]] = {}
    for p in outputs:
        base = os.path.basename(p)
        # try to backtrack to the original audio base name
        # e.g., mytrack.json / mytrack_words.json / etc.
        stem, _ = os.path.splitext(base)
        by_base.setdefault(stem, []).append(p)
    return by_base

def _pick_json_for_audio(all_outputs: List[str], audio_path: str) -> Optional[str]:
    """Find the most likely JSON transcript for a given audio file."""
    audio_base = os.path.splitext(os.path.basename(audio_path))[0]
    # First pass: any JSON whose stem starts with audio_base
    candidates = [p for p in all_outputs if p.endswith(".json") and audio_base in os.path.basename(p)]
    if candidates:
        # Prefer ones that contain "words" or "aligned"
        ranked = sorted(candidates, key=lambda p: (("words" not in p and "align" not in p), len(p)))
        for c in ranked:
            if _is_transcript_json(c):
                return c
        # fallback to any JSON
        return ranked[0]

    # Second pass: any WhisperX-like JSON at all, last modified nearest to this audio?
    all_jsons = [p for p in all_outputs if _is_transcript_json(p)]
    if not all_jsons:
        return None
    # heuristic: choose the latest JSON
    all_jsons.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return all_jsons[0]

def load_word_list(transcript_json_path: str) -> List[Dict[str, Any]]:
    """Return a flat list of words with start/end and text."""
    with open(transcript_json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    words: List[Dict[str, Any]] = []
    for seg in obj.get("segments", []):
        for w in seg.get("words", []) or []:
            if "start" in w and "end" in w and "word" in w:
                words.append({
                    "text": w["word"],
                    "start": float(w["start"]),
                    "end": float(w["end"]),
                    "score": float(w.get("score", 0.0))
                })
    # Ensure sorted
    words.sort(key=lambda x: (x["start"], x["end"]))
    return words

# --------------------------
# Sentence grouping
# --------------------------

def group_sentences_from_words(words: List[Dict[str, Any]],
                               gap_threshold_s: float = 0.4) -> List[Dict[str, Any]]:
    """
    Group words into sentence-like chunks (lines) by inter-word gap.
    Gap >= threshold => new sentence.
    """
    sentences = []
    if not words:
        return sentences

    current = {"words": [words[0]],
               "start": words[0]["start"],
               "end": words[0]["end"]}

    eps = 1e-3
    for prev, cur in zip(words, words[1:]):
        gap = float(cur["start"]) - float(prev["end"])
        # keep in same sentence if gap < threshold (with tiny epsilon tolerance)
        if gap > (gap_threshold_s - eps):
            # close current
            current["end"] = float(current["words"][-1]["end"])
            current["text"] = " ".join(w["text"] for w in current["words"])
            current["norm_text"] = normalize_text(current["text"])
            sentences.append(current)
            # start new
            current = {"words": [cur], "start": float(cur["start"]), "end": float(cur["end"])}
        else:
            current["words"].append(cur)

    # close last
    current["end"] = float(current["words"][-1]["end"])
    current["text"] = " ".join(w["text"] for w in current["words"])
    current["norm_text"] = normalize_text(current["text"])
    sentences.append(current)

    # Index
    for i, s in enumerate(sentences):
        s["idx"] = i
        s["duration"] = float(s["end"] - s["start"])
        s["tokens"] = [normalize_word(w["text"]) for w in s["words"] if w["text"].strip()]
    return sentences

# --------------------------
# Sentence matching (monotonic)
# --------------------------

def sentence_pair_cost(master_s: Dict[str, Any], target_s: Dict[str, Any]) -> Tuple[float, float]:
    """
    Return (cost, confidence) for aligning a master sentence to a target sentence.
    Cost is minimized by DP.
    Confidence is for reporting: higher is better [0..1].
    """
    mtoks, ttoks = master_s["tokens"], target_s["tokens"]
    if not mtoks and not ttoks:
        return (0.0, 1.0)
    # Edit distance similarity
    dist = levenshtein(mtoks, ttoks)
    max_len = max(1, max(len(mtoks), len(ttoks)))
    sim = 1.0 - (dist / max_len)  # 0..1

    # Duration penalty
    mdur, tdur = master_s["duration"], target_s["duration"]
    dur_diff = abs(mdur - tdur)
    # Convert to [0..1] score via exponential falloff
    dur_score = math.exp(-dur_diff / 0.5)  # half-second scale
    # Confidence is a mix
    confidence = 0.75 * sim + 0.25 * dur_score

    # DP cost (lower is better). Map confidence to a cost.
    cost = 1.0 - confidence
    return (float(cost), float(confidence))

def match_sentences_monotonic(master_sents: List[Dict[str, Any]],
                              target_sents: List[Dict[str, Any]],
                              gap_penalty: float = 0.6) -> List[Tuple[Optional[int], Optional[int], float]]:
    """
    Needleman-Wunsch style alignment with monotonicity.
    Returns list of (master_idx or None, target_idx or None, confidence).
    """
    M, N = len(master_sents), len(target_sents)
    # dp and backtrace
    dp = np.full((M + 1, N + 1), np.inf, dtype=np.float32)
    bt = np.zeros((M + 1, N + 1, 2), dtype=np.int16)  # backtrace: (-1,0) deletion, (0,-1) insertion, (-1,-1) match
    dp[0, 0] = 0.0
    for i in range(1, M + 1):
        dp[i, 0] = dp[i - 1, 0] + gap_penalty  # master deletion (unmatched master)
        bt[i, 0] = [-1, 0]
    for j in range(1, N + 1):
        dp[0, j] = dp[0, j - 1] + gap_penalty  # target insertion (unmatched target)
        bt[0, j] = [0, -1]

    conf_cache = {}
    for i in range(1, M + 1):
        for j in range(1, N + 1):
            pair = (i - 1, j - 1)
            if pair not in conf_cache:
                cost, conf = sentence_pair_cost(master_sents[i - 1], target_sents[j - 1])
                conf_cache[pair] = (cost, conf)
            cost, _ = conf_cache[pair]

            # match
            match_cost = dp[i - 1, j - 1] + cost
            # gap (skip master sentence)
            del_cost = dp[i - 1, j] + gap_penalty
            # gap (skip target sentence)
            ins_cost = dp[i, j - 1] + gap_penalty

            best = match_cost
            move = (-1, -1)
            if del_cost < best:
                best = del_cost
                move = (-1, 0)
            if ins_cost < best:
                best = ins_cost
                move = (0, -1)
            dp[i, j] = best
            bt[i, j] = move

    # Recover path
    i, j = M, N
    path: List[Tuple[Optional[int], Optional[int], float]] = []
    while i > 0 or j > 0:
        di, dj = bt[i, j]
        if di == -1 and dj == -1:
            # match
            _, conf = conf_cache[(i - 1, j - 1)]
            path.append((i - 1, j - 1, conf))
            i -= 1; j -= 1
        elif di == -1 and dj == 0:
            # master unmatched
            path.append((i - 1, None, 0.0))
            i -= 1
        elif di == 0 and dj == -1:
            # target unmatched
            path.append((None, j - 1, 0.0))
            j -= 1
        else:
            # should not happen
            break
    path.reverse()
    return path

# --------------------------
# Word-level mapping (per matched sentence)
# --------------------------

def align_words(master_words: List[str], target_words: List[str]) -> List[Tuple[Optional[int], Optional[int]]]:
    """
    Simple word alignment mapping indices (monotonic).
    Returns list of (m_idx or None, t_idx or None).
    """
    n, m = len(master_words), len(target_words)
    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    bt = np.zeros((n + 1, m + 1, 2), dtype=np.int16)
    dp[0, 0] = 0.0
    gap_penalty = 1.0
    for i in range(1, n + 1):
        dp[i, 0] = i * gap_penalty
        bt[i, 0] = [-1, 0]
    for j in range(1, m + 1):
        dp[0, j] = j * gap_penalty
        bt[0, j] = [0, -1]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0.0 if master_words[i - 1] == target_words[j - 1] else 1.0
            match = dp[i - 1, j - 1] + cost
            delete = dp[i - 1, j] + gap_penalty
            insert = dp[i, j - 1] + gap_penalty
            best = match; move = (-1, -1)
            if delete < best:
                best = delete; move = (-1, 0)
            if insert < best:
                best = insert; move = (0, -1)
            dp[i, j] = best
            bt[i, j] = move

    # Backtrace
    i, j = n, m
    out: List[Tuple[Optional[int], Optional[int]]] = []
    while i > 0 or j > 0:
        di, dj = bt[i, j]
        if di == -1 and dj == -1:
            out.append((i - 1, j - 1))
            i -= 1; j -= 1
        elif di == -1 and dj == 0:
            out.append((i - 1, None))
            i -= 1
        elif di == 0 and dj == -1:
            out.append((None, j - 1))
            j -= 1
    out.reverse()
    return out

# --------------------------
# Audio assembly
# --------------------------

def time_to_samples(t: float, sr: int) -> int:
    return max(0, int(round(t * sr)))

def crossfade_add(dst: np.ndarray, start: int, clip: np.ndarray, fade_ms: float, sr: int):
    """Place clip into dst at 'start' with symmetric crossfade (fade_ms)."""
    L = clip.shape[-1]
    end = start + L
    if end > dst.shape[-1]:
        L = dst.shape[-1] - start
        if L <= 0:
            return
        clip = clip[..., :L]
        end = start + L

    if fade_ms <= 0:
        dst[..., start:end] += clip
        return

    fade_samps = max(1, int(sr * (fade_ms / 1000.0)))
    fade_samps = min(fade_samps, L // 2)

    # Pre-existing region
    existing = dst[..., start:end].copy()
    # Build fades
    fade_in = np.linspace(0.0, 1.0, fade_samps, dtype=clip.dtype)
    sustain_len = L - 2 * fade_samps
    sustain = np.ones((max(sustain_len, 0),), dtype=clip.dtype)
    fade_out = np.linspace(1.0, 0.0, fade_samps, dtype=clip.dtype)

    env = np.concatenate([fade_in, sustain, fade_out]) if sustain_len > 0 else np.concatenate([fade_in, fade_out])
    env = env[:L]

    # Mix: new replaces existing where env is strong, but we still sum
    dst[..., start:end] = existing * (1.0 - env) + clip * env

def uniform_time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    """Librosa time-stretch wrapper (mono or multi-channel)."""
    # librosa.effects.time_stretch works on mono; loop channels if needed.
    if y.ndim == 1:
        return librosa.effects.time_stretch(y, rate=rate)
    else:
        chans = []
        for c in range(y.shape[0]):
            chans.append(librosa.effects.time_stretch(y[c], rate=rate))
        # pad to largest length
        maxL = max(ch.shape[-1] for ch in chans)
        out = np.zeros((y.shape[0], maxL), dtype=np.float32)
        for c, ch in enumerate(chans):
            out[c, :ch.shape[-1]] = ch
        return out

# --------------------------
# RTLA-based time warping
# --------------------------

def _compute_warp_path(master_audio_path: str, target_audio_path: str) -> Optional[np.ndarray]:
    """Compute strict monotonic warping path between master and target using RTLA (OLTW).
    Returns path as np.ndarray shape (2, N) in frame indices or None on failure.
    """
    try:
        sp = RTLAStrProc(sample_rate=RTLA_SR, chunk_size=RTLA_CHUNK, hop_length=RTLA_HOP, features=RTLA_FEATURES)
        oltw = OLTW(
            sp,
            ref_audio_path=master_audio_path,
            window_size=RTLA_WIN,
            sample_rate=RTLA_SR,
            hop_length=RTLA_HOP,
            max_run_count=3,
            metric="cosine",
            features=RTLA_FEATURES,
        )
        wp = oltw.run_offline(target_audio_path=target_audio_path)
        wp = make_path_strictly_monotonic(wp)
        return wp
    except Exception as e:
        logger.warning(f"RTLA warp path failed with features {RTLA_FEATURES}: {e}")
        # Fallback to chroma-only features
        try:
            sp = RTLAStrProc(sample_rate=RTLA_SR, chunk_size=RTLA_CHUNK, hop_length=RTLA_HOP, features=["chroma"]) 
            oltw = OLTW(
                sp,
                ref_audio_path=master_audio_path,
                window_size=RTLA_WIN,
                sample_rate=RTLA_SR,
                hop_length=RTLA_HOP,
                max_run_count=3,
                metric="cosine",
                features=["chroma"],
            )
            wp = oltw.run_offline(target_audio_path=target_audio_path)
            wp = make_path_strictly_monotonic(wp)
            return wp
        except Exception as e2:
            logger.error(f"RTLA warp path failed (fallback chroma): {e2}", exc_info=True)
            return None


def _time_warp_target_to_master(master_len: int, sr: int, target_y: np.ndarray, wp: np.ndarray) -> np.ndarray:
    """Build time-warped target signal aligned to master using warping path.
    - master_len: samples in master timeline
    - sr: output sample rate (master's sr)
    - target_y: target audio resampled to sr (1D)
    - wp: warping path (2, N) frame indices
    """
    if target_y.ndim > 1:
        target_y = target_y.squeeze()
    t_master = np.arange(master_len, dtype=np.float64) / float(sr)
    times_ref = wp[0].astype(np.float64) / float(RTLA_FR)
    times_tgt = wp[1].astype(np.float64) / float(RTLA_FR)
    # Map master time to target time via linear interpolation
    t_target = np.interp(t_master, times_ref, times_tgt, left=times_tgt[0], right=times_tgt[-1])
    pos = np.clip(t_target * float(sr), 0.0, float(len(target_y) - 1))
    x = np.arange(len(target_y), dtype=np.float64)
    aligned = np.interp(pos, x, target_y, left=0.0, right=0.0).astype(np.float32)
    return aligned

def _assemble_by_sentences_with_wp(master_sents: List[Dict[str, Any]],
                                   wp: np.ndarray,
                                   target_y: np.ndarray,
                                   sr: int,
                                   xfade_ms: float,
                                   master_len: int) -> Tuple[np.ndarray, List[Tuple[float, float, bool]]]:
    """Build aligned audio by stretching whole sentences using DTW timing mapping.
    For each master sentence [t0,t1], map to target times via wp, extract, uniformly time-stretch to fit,
    and place with crossfade. Avoids per-sample warping (which can cause pitch jitter).
    """
    if target_y.ndim > 1:
        target_y = target_y.squeeze()
    out = np.zeros(master_len, dtype=np.float32)
    placed_spans: List[Tuple[float, float, bool]] = []

    times_ref = wp[0].astype(np.float64) / float(RTLA_FR)
    times_tgt = wp[1].astype(np.float64) / float(RTLA_FR)

    for s in master_sents:
        m_start = float(s["start"]) if isinstance(s, dict) else float(s.start)
        m_end = float(s["end"]) if isinstance(s, dict) else float(s.end)
        if m_end <= m_start:
            continue
        start_samp = time_to_samples(m_start, sr)
        end_samp = time_to_samples(m_end, sr)
        needed = max(0, end_samp - start_samp)
        if needed <= 0:
            continue

        # Map master times to target times (linear interp over DTW path)
        t0 = float(np.interp(m_start, times_ref, times_tgt, left=times_tgt[0], right=times_tgt[-1]))
        t1 = float(np.interp(m_end, times_ref, times_tgt, left=times_tgt[-1], right=times_tgt[-1]))
        if not (t1 > t0):
            placed_spans.append((m_start, m_end, False))
            continue
        src0 = time_to_samples(t0, sr)
        src1 = time_to_samples(t1, sr)
        src0 = max(0, min(len(target_y) - 1, src0))
        src1 = max(0, min(len(target_y), src1))
        if src1 - src0 <= 1:
            placed_spans.append((m_start, m_end, False))
            continue

        clip = target_y[src0:src1].copy()
        # Uniform time-stretch to fit the master window exactly (phase-vocoder via librosa)
        rate = max(1e-4, clip.shape[-1] / float(needed))
        stretched = uniform_time_stretch(clip, rate=rate)
        if stretched.shape[-1] < needed:
            pad = needed - stretched.shape[-1]
            clip_fit = np.pad(stretched, (0, pad), mode="constant")
        else:
            clip_fit = stretched[:needed]

        crossfade_add(out, start_samp, clip_fit, fade_ms=xfade_ms, sr=sr)
        placed_spans.append((m_start, m_end, True))

    return out, placed_spans

# --------------------------
# Orchestration
# --------------------------

def build_sentences_for_audio(json_path: str,
                              sentence_gap_s: float) -> List[Dict[str, Any]]:
    words = load_word_list(json_path)
    return group_sentences_from_words(words, gap_threshold_s=sentence_gap_s)

def _load_audio_any(path: str, sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    y, srr = librosa.load(path, sr=sr, mono=True)  # mono for alignment; keep it simple
    y = y.astype(np.float32)
    return y, srr

def _find_jsons_for_files(outputs: List[str], files: List[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for ap in files:
        jp = _pick_json_for_audio(outputs, ap)
        if jp is None:
            raise RuntimeError(f"No transcript JSON found for {ap}")
        mapping[ap] = jp
    return mapping

# --------------------------
# Output directory helpers
# --------------------------

def _file_hash(path: str, chunk: int = 8192) -> str:
    h = hashlib.sha256()
    try:
        with open(path, 'rb') as f:
            while True:
                b = f.read(chunk)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()[:12]
    except Exception:
        return os.path.basename(path)

def _project_folder(master_audio_path: str, secondary_audio_paths: List[str]) -> Tuple[str, str]:
    master_name = os.path.splitext(os.path.basename(master_audio_path))[0]
    parts = [_file_hash(master_audio_path)] + [_file_hash(p) for p in sorted(secondary_audio_paths)]
    ph = hashlib.sha256("_".join(parts).encode()).hexdigest()[:8]
    folder = f"{master_name}_{ph}"
    root = os.path.join(output_path, "align", folder)
    return root, folder

def _sent_as_text(s: Dict[str, Any]) -> str:
    return s.get("text") or " ".join(w["text"] for w in s.get("words", []))

def align_secondary_to_master(master_audio_path: str,
                              secondary_audio_paths: List[str],
                              humanize_value: float,
                              progress: gr.Progress):

    # Output root
    out_root, folder_name = _project_folder(master_audio_path, secondary_audio_paths)
    os.makedirs(out_root, exist_ok=True)

    # 0) Ensure RTLA model is present if phoneme features are enabled
    try:
        from modules.rtla.config import CRNN_MODEL_PT, CRNN_MODEL_SAFE, CRNN_CONFIG_JSON, MODELS_DIR
        # Ensure rtla models directory exists
        os.makedirs(MODELS_DIR, exist_ok=True)
        # Try to copy from the attached project
        if not (os.path.exists(CRNN_MODEL_PT) or os.path.exists(CRNN_MODEL_SAFE)):
            guess_src = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                     "..", "real-time-lyrics-alignment", "model", "pretrained-model.pt")
            guess_src = os.path.normpath(guess_src)
            if os.path.exists(guess_src):
                shutil.copy2(guess_src, CRNN_MODEL_PT)
        # Convert to safetensors if only .pt is present
        if os.path.exists(CRNN_MODEL_PT) and not os.path.exists(CRNN_MODEL_SAFE):
            try:
                import subprocess, sys
                converter = os.path.join(os.path.dirname(__file__), "..", "modules", "rtla", "convert_model_to_safetensors.py")
                converter = os.path.normpath(converter)
                subprocess.run([sys.executable, converter], check=True)
            except Exception:
                logger.warning("CRNN .pt->.safetensors conversion failed; will use .pt loader.")
    except Exception:
        pass

    # 1) Transcribe with WhisperX (no speakers, with alignment)
    progress(0.0, "Transcribing with WhisperX (word timestamps, no speakers)...")
    files_to_transcribe = [master_audio_path] + secondary_audio_paths
    # Use your provided function
    summary, outputs = process_transcription_whisperx(
        audio_files=files_to_transcribe,
        model_size="large-v3",
        language="auto",
        align_output=True,
        assign_speakers=False,
        batch_size=8,
        compute_type="float16",
        return_char_alignments=False,
        progress=progress or gr.Progress(track_tqdm=True)
    )

    # 2) Build sentence grids
    # Increase the sentence gap slightly to form more coherent sentences
    sentence_gap_s = 0.6
    json_map = _find_jsons_for_files(outputs, files_to_transcribe)
    master_json = json_map[master_audio_path]
    master_sents = build_sentences_for_audio(master_json, sentence_gap_s)

    # 3) Load master audio (defines the global timeline)
    progress(0.15, "Loading master audio & building reference sentence grid...")
    master_y, sr = _load_audio_any(master_audio_path, sr=None)
    master_dur = len(master_y) / sr
    master_len = len(master_y)

    tolerance_ms = max(0.0, float(humanize_value) * 10.0)
    tolerance_s = tolerance_ms / 1000.0
    xfade_ms = 10.0

    all_export_files: List[str] = []
    # Save master track into output alignment folder for reference
    try:
        master_base = os.path.splitext(os.path.basename(master_audio_path))[0]
        master_copy_path = os.path.join(out_root, f"{master_base}__master.wav")
        sf.write(master_copy_path, master_y, sr)
        all_export_files.append(master_copy_path)
    except Exception as e:
        logger.warning(f"Failed to save master track copy: {e}")
    report_summaries: List[str] = []
    overlay_path: Optional[str] = None

    # 4) Iterate targets
    for idx, tgt_path in enumerate(secondary_audio_paths):
        progress(0.2 + 0.7 * (idx / max(1, len(secondary_audio_paths))), f"Aligning target {idx+1}/{len(secondary_audio_paths)}...")
        tgt_json = json_map[tgt_path]
        # Build target sentence list by binning target words into master windows
        tgt_words = load_word_list(tgt_json)
        tgt_sents = []
        for ms in master_sents:
            start = float(ms["start"])
            end = float(ms["end"])
            words_in = [w for w in tgt_words if (w["start"] >= start and w["end"] <= end)]
            text = " ".join(w["text"] for w in words_in)
            tgt_sents.append({
                "idx": ms["idx"],
                "start": start,
                "end": end,
                "duration": float(end - start),
                "words": words_in,
                "text": text,
                "norm_text": normalize_text(text),
                "tokens": [normalize_word(w["text"]) for w in words_in if w["text"].strip()]
            })

        # 1:1 sentence pairing using master as grid with confidence per sentence
        pairs = []
        for i in range(len(master_sents)):
            cost, conf = sentence_pair_cost(master_sents[i], tgt_sents[i])
            # Consider empty target sentence as unmatched (conf 0)
            if not tgt_sents[i]["words"]:
                pairs.append((i, None, 0.0))
            else:
                pairs.append((i, i, conf))

        # Word-level inside each matched sentence for reporting
        per_sentence_reports = []
        for m_idx, t_idx, conf in pairs:
            if m_idx is not None and t_idx is not None:
                m_words = [normalize_word(w["text"]) for w in master_sents[m_idx]["words"]]
                t_words = [normalize_word(w["text"]) for w in tgt_sents[t_idx]["words"]]
                word_map = align_words(m_words, t_words)
            else:
                word_map = []
            per_sentence_reports.append({
                "master_idx": m_idx,
                "target_idx": t_idx,
                "confidence": conf,
                "master_text": _sent_as_text(master_sents[m_idx]) if m_idx is not None else None,
                "target_text": _sent_as_text(tgt_sents[t_idx]) if t_idx is not None else None,
                "word_alignment": [{"m": mi, "t": ti} for (mi, ti) in word_map]
            })

        # Build aligned audio onto blank timeline
        tgt_y, tgt_sr = _load_audio_any(tgt_path, sr=sr)  # resample to master's sr
        # Prefer RTLA warp for constructing aligned audio
        wp = _compute_warp_path(master_audio_path, tgt_path)
        if wp is not None:
            # Use DTW only to estimate per-sentence source spans, then uniform-stretch sentences
            out, placed_spans = _assemble_by_sentences_with_wp(master_sents, wp, tgt_y, sr, xfade_ms, master_len)
        else:
            # Fallback to sentence/window-based assembly
            out = np.zeros(master_len, dtype=np.float32)
            placed_spans = []
            for m_idx, t_idx, conf in pairs:
                if m_idx is None:
                    continue
                m_start = float(master_sents[m_idx]["start"])
                m_end = float(master_sents[m_idx]["end"])
                m_dur = max(0.0, m_end - m_start)
                if m_dur <= 0.0:
                    continue
                start_samp = time_to_samples(m_start, sr)
                end_samp = time_to_samples(m_end, sr)
                if t_idx is None:
                    placed_spans.append((m_start, m_end, False))
                    continue
                t_start = float(tgt_sents[t_idx]["start"])
                t_end = float(tgt_sents[t_idx]["end"])
                t_dur = max(0.0, t_end - t_start)
                if t_dur <= 0.0:
                    placed_spans.append((m_start, m_end, False))
                    continue
                t0 = time_to_samples(t_start, sr)
                t1 = time_to_samples(t_end, sr)
                clip = tgt_y[t0:t1].copy()
                dur_diff = abs(m_dur - t_dur)
                if dur_diff <= tolerance_s:
                    needed = end_samp - start_samp
                    if clip.shape[-1] < needed:
                        pad = needed - clip.shape[-1]
                        clip = np.pad(clip, (0, pad), mode="constant")
                    else:
                        clip = clip[:needed]
                else:
                    rate = max(1e-4, clip.shape[-1] / max(1, (end_samp - start_samp)))
                    stretched = uniform_time_stretch(clip, rate=rate)
                    if stretched.shape[-1] < (end_samp - start_samp):
                        pad = (end_samp - start_samp) - stretched.shape[-1]
                        clip = np.pad(stretched, (0, pad), mode="constant")
                    else:
                        clip = stretched[:(end_samp - start_samp)]
                crossfade_add(out, start_samp, clip, fade_ms=xfade_ms, sr=sr)
                placed_spans.append((m_start, m_end, True))

        # Export aligned wav
        tgt_base = os.path.splitext(os.path.basename(tgt_path))[0]
        aligned_wav = os.path.join(out_root, f"{tgt_base}__aligned.wav")
        sf.write(aligned_wav, out, sr)
        all_export_files.append(aligned_wav)

        # Compose report JSON
        report = {
            "master_audio": master_audio_path,
            "target_audio": tgt_path,
            "sample_rate": sr,
            "humanize_ms": tolerance_ms,
            "sentence_gap_s": sentence_gap_s,
            "master_sentences": [{
                "idx": s["idx"], "start": s["start"], "end": s["end"],
                "text": _sent_as_text(s)
            } for s in master_sents],
            "target_sentences": [{
                "idx": s["idx"], "start": s["start"], "end": s["end"],
                "text": _sent_as_text(s)
            } for s in tgt_sents],
            "matches": per_sentence_reports
        }
        report_path = os.path.join(out_root, f"{tgt_base}__report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        all_export_files.append(report_path)

        # Human-readable summary line
        matched = sum(1 for m,t,_ in pairs if m is not None and t is not None)
        missed_m = sum(1 for m,t,_ in pairs if m is not None and t is None)
        missed_t = sum(1 for m,t,_ in pairs if m is None and t is not None)
        confs = [c for m,t,c in pairs if m is not None and t is not None]
        avg_conf = float(np.mean(confs)) if confs else 0.0
        report_summaries.append(
            f"[{tgt_base}] matched={matched}  master_gaps={missed_m}  target_extras={missed_t}  avg_conf={avg_conf:.3f}"
        )

        # Visualization (use the first target for the big overlay)
        if idx == 0:
            overlay_path = os.path.join(out_root, "alignment_visualization.png")
            _plot_overlay(master_y, out, sr, master_sents, pairs, overlay_path)

    # Build overall summary text
    summary_lines = [f"Output: {out_root}", summary]
    summary_lines.extend(report_summaries)
    summary_text = "\n".join(summary_lines)

    # Return for UI: summary, files, viz_path, first aligned audio (UI picks it separately)
    return summary_text, all_export_files, None, overlay_path

def _plot_overlay(master_y: np.ndarray, aligned_y: np.ndarray, sr: int,
                  master_sents: List[Dict[str, Any]],
                  pairs: List[Tuple[Optional[int], Optional[int], float]],
                  out_path: str):
    """Create a colorful overlay plot showing matched/unmatched master spans."""
    duration = len(master_y) / sr
    t = np.linspace(0, duration, len(master_y), endpoint=False)

    plt.figure(figsize=(16, 6))
    # Waveforms
    plt.plot(t, master_y, linewidth=0.7, label="Master", alpha=0.9, color="#3b82f6")  # blue
    plt.plot(t, aligned_y, linewidth=0.7, label="Aligned Target", alpha=0.7, color="#f59e0b")  # orange

    # Shaded spans
    for (m_idx, t_idx, conf) in pairs:
        if m_idx is None:
            continue
        s = master_sents[m_idx]
        c = "#22c55e" if t_idx is not None else "#ef4444"  # green matched, red unmatched
        alpha = 0.15 if t_idx is not None else 0.12
        plt.axvspan(s["start"], s["end"], color=c, alpha=alpha)

    plt.title("Alignment Overlay — Master vs. Aligned Target")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (norm.)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

# --------------------------
# UI wiring (your exact layout, with our function)
# --------------------------

def render(arg_handler_local):
    with gr.Blocks() as ui:
        gr.Markdown("# 🎚️ Align")
        gr.Markdown("Align multiple takes to a master using sentence-based matching with strict timing pins.")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📁 Inputs")
                master_file = gr.File(label="Master Track (single)", file_count="single", file_types=["audio"])
                secondary_files = gr.File(label="Secondary Tracks (multiple)", file_count="multiple", file_types=["audio"])
                humanize = gr.Slider(
                    label="Humanize",
                    minimum=0,
                    maximum=10,
                    step=1,
                    value=0,
                    info="Tolerance at sentence level (ms = value * 10) before stretching."
                )
                start_btn = gr.Button(value="🚀 Start Alignment", variant="primary")

            with gr.Column(scale=2):
                gr.Markdown("### 📊 Results")
                status = gr.Textbox(label="Status & Reports", value="Idle", lines=15, max_lines=30)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🎵 Aligned Audio Files")
                audio_outputs = gr.File(label="Download Aligned Tracks", file_count="multiple")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📈 Waveform Visualization")
                viz_output = gr.Image(label="Track Alignment Overlay", type="filepath")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🎧 Audio Preview")
                gr.Markdown("Preview the first aligned track below:")
                audio_preview = gr.Audio(label="Preview Aligned Audio", visible=True)

        def start(master, secondaries, h):
            if not master:
                return "⚠️ Please provide a master track.", [], None, None
            sec_list = secondaries or []
            if isinstance(sec_list, str):
                sec_list = [sec_list]
            sec_list = [s for s in sec_list if s and os.path.exists(s)]
            if not sec_list:
                return "⚠️ Please provide at least one secondary track.", [], None, None
            p = gr.Progress(track_tqdm=True)
            try:
                summary_text, all_files, _unused, viz_path = align_secondary_to_master(
                    master, sec_list, float(h), p
                )
                audio_files = [f for f in all_files if f.endswith('.wav') and '__aligned' in f]
                preview_audio = audio_files[0] if audio_files else None
                return summary_text, all_files, viz_path, preview_audio
            except Exception as e:
                logger.error(f"Alignment failed: {e}", exc_info=True)
                return f"❌ Error during alignment: {str(e)}", [], None, None

        start_btn.click(
            fn=start,
            inputs=[master_file, secondary_files, humanize],
            outputs=[status, audio_outputs, viz_output, audio_preview]
        )

    return ui

def register_descriptions(arg_handler_local):
    descriptions = {
        "master_file": "Upload the reference master track (the timing source).",
        "secondary_files": "Upload one or more secondary takes to align to the master.",
        "humanize": "Sentence-level tolerance before stretching (value * 10 ms).",
        "start_btn": "Start sentence-based alignment.",
        "audio_outputs": "Download aligned audio tracks and detailed reports.",
        "viz_output": "Waveform overlay. Matched/unmatched sections are highlighted.",
        "audio_preview": "Preview the first aligned track."
    }
    for elem_id, desc in descriptions.items():
        if hasattr(arg_handler_local, "register_description"):
            arg_handler_local.register_description("align", elem_id, desc)

def listen():
    return
