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
                                   master_len: int,
                                   master_y: Optional[np.ndarray] = None,
                                   max_nudge_ms: float = 0.0,
                                   is_empty_sentence: Optional[List[bool]] = None,
                                   overrides: Optional[List[Optional[Tuple[float, float]]]] = None) -> Tuple[np.ndarray, List[Tuple[float, float, bool]], List[Tuple[float, float]]]:
    """Build aligned audio by stretching whole sentences using DTW timing mapping.
    For each master sentence [t0,t1], map to target times via wp, extract, uniformly time-stretch to fit,
    and place with crossfade. Avoids per-sample warping (which can cause pitch jitter).
    """
    if target_y.ndim > 1:
        target_y = target_y.squeeze()
    out = np.zeros(master_len, dtype=np.float32)
    placed_spans: List[Tuple[float, float, bool]] = []
    mapped_times: List[Tuple[float, float]] = []

    times_ref = wp[0].astype(np.float64) / float(RTLA_FR)
    times_tgt = wp[1].astype(np.float64) / float(RTLA_FR)
    # Simple onset envelopes for optional micro-nudging
    def _env(y):
        y_abs = np.abs(y.astype(np.float32))
        win = max(1, int(sr * (10.0 / 1000.0)))
        k = np.ones(win, dtype=np.float32) / float(win)
        return np.convolve(y_abs, k, mode="same")
    master_env = _env(master_y) if (master_y is not None and max_nudge_ms > 0.0) else None
    target_env = _env(target_y) if max_nudge_ms > 0.0 else None

    # baseline amplitude to detect near-silence
    global_rms = float(np.sqrt(np.maximum(1e-12, np.mean(target_y.astype(np.float32) ** 2))))
    silence_thresh = max(1e-6, global_rms * 0.2)

    for idx_s, s in enumerate(master_sents):
        m_start = float(s["start"]) if isinstance(s, dict) else float(s.start)
        m_end = float(s["end"]) if isinstance(s, dict) else float(s.end)
        if m_end <= m_start:
            continue
        start_samp = time_to_samples(m_start, sr)
        end_samp = time_to_samples(m_end, sr)
        needed = max(0, end_samp - start_samp)
        if needed <= 0:
            continue

        # Map master times to target times (linear interp over DTW path) unless overridden
        if overrides is not None and idx_s < len(overrides) and overrides[idx_s] is not None:
            t0, t1 = overrides[idx_s]
        else:
            t0 = float(np.interp(m_start, times_ref, times_tgt, left=times_tgt[0], right=times_tgt[-1]))
            t1 = float(np.interp(m_end, times_ref, times_tgt, left=times_tgt[0], right=times_tgt[-1]))
        if not (t1 > t0):
            placed_spans.append((m_start, m_end, False))
            mapped_times.append((t0, t1))
            continue

        # Optional onset nudge: bounded by max_nudge_ms
        if master_env is not None and target_env is not None:
            win_ms = min(150.0, max(50.0, (m_end - m_start) * 1000.0 * 0.3))
            win_samps = max(1, int(sr * (win_ms / 1000.0)))
            nudge_samps = max(1, int(sr * (max_nudge_ms / 1000.0)))
            m0 = start_samp
            m1 = min(len(master_env), m0 + win_samps)
            ref_seg = master_env[m0:m1]
            if ref_seg.size >= 8:
                tgt_c = time_to_samples(t0, sr)
                t_lo = max(0, tgt_c - nudge_samps)
                t_hi = min(len(target_env), tgt_c + nudge_samps + win_samps)
                cand = target_env[t_lo:t_hi]
                best_off = 0
                best_score = -1.0
                step = max(1, int(sr * 0.001))
                max_off = min(nudge_samps * 2, max(0, cand.size - ref_seg.size))
                for off in range(0, max_off + 1, step):
                    seg = cand[off:off + ref_seg.size]
                    if seg.size != ref_seg.size:
                        break
                    denom = (np.linalg.norm(ref_seg) * np.linalg.norm(seg)) + 1e-8
                    score = float(np.dot(ref_seg, seg) / denom)
                    if score > best_score:
                        best_score = score
                        best_off = off
                signed = best_off - (tgt_c - t_lo)
                signed = np.clip(signed, -nudge_samps, nudge_samps)
                t0 = t0 + float(signed) / float(sr)
        src0 = time_to_samples(t0, sr)
        src1 = time_to_samples(t1, sr)
        src0 = max(0, min(len(target_y) - 1, src0))
        src1 = max(0, min(len(target_y), src1))
        if src1 - src0 <= 1:
            placed_spans.append((m_start, m_end, False))
            mapped_times.append((t0, t1))
            continue

        clip = target_y[src0:src1].copy()

        # If sentence had no words (by WhisperX), treat as unmatched initially (unless explicitly overridden)
        if (
            is_empty_sentence is not None and idx_s < len(is_empty_sentence) and is_empty_sentence[idx_s]
        ) and not (overrides is not None and idx_s < len(overrides) and overrides[idx_s] is not None):
            placed_spans.append((m_start, m_end, False))
            mapped_times.append((t0, t1))
            continue

        # If clip is near-silent, defer to auto-fill
        clip_rms = float(np.sqrt(np.maximum(1e-12, np.mean(clip ** 2))))
        if clip_rms < silence_thresh:
            placed_spans.append((m_start, m_end, False))
            mapped_times.append((t0, t1))
            continue
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
        mapped_times.append((t0, t1))

    return out, placed_spans, mapped_times

def _auto_fill_unmatched_segments(master_sents: List[Dict[str, Any]],
                                  placed_spans: List[Tuple[float, float, bool]],
                                  mapped_times: List[Tuple[float, float]],
                                  target_y: np.ndarray,
                                  sr: int,
                                  xfade_ms: float,
                                  out: np.ndarray,
                                  master_y: Optional[np.ndarray] = None) -> None:
    """Fill master sentences that were not placed by estimating target gaps between neighboring placed segments.
    Modifies 'out' in place.
    """
    if target_y.ndim > 1:
        target_y = target_y.squeeze()

    def _env(y: np.ndarray) -> np.ndarray:
        y_abs = np.abs(y.astype(np.float32))
        win = max(1, int(sr * (10.0 / 1000.0)))
        k = np.ones(win, dtype=np.float32) / float(win)
        return np.convolve(y_abs, k, mode="same")

    target_env = _env(target_y)
    master_env = _env(master_y) if master_y is not None else None
    n = len(master_sents)
    for i in range(n):
        m_start = float(master_sents[i]["start"]) if isinstance(master_sents[i], dict) else float(master_sents[i].start)
        m_end = float(master_sents[i]["end"]) if isinstance(master_sents[i], dict) else float(master_sents[i].end)
        if m_end <= m_start:
            continue
        if i < len(placed_spans) and placed_spans[i][2]:
            continue
        # Find neighbors
        jL = i - 1
        while jL >= 0 and not (jL < len(placed_spans) and placed_spans[jL][2]):
            jL -= 1
        jR = i + 1
        while jR < n and not (jR < len(placed_spans) and placed_spans[jR][2]):
            jR += 1
        # Determine candidate target window
        t_left = mapped_times[i][0]
        t_right = mapped_times[i][1]
        if jL >= 0:
            t_left = max(t_left, mapped_times[jL][1])
        if jR < n:
            t_right = min(t_right, mapped_times[jR][0])
        if not (t_right > t_left):
            # fallback to direct mapping
            t_left, t_right = mapped_times[i]
            if not (t_right > t_left):
                continue
        # Expand the search window slightly to give correlation more room
        t_mid = 0.5 * (t_left + t_right)
        t_span = max(0.02, (t_right - t_left))
        expand = min(1.0, 0.5 * t_span)
        t_left_s = max(0.0, t_left - expand)
        t_right_s = t_right + expand

        src0 = time_to_samples(t_left_s, sr)
        src1 = time_to_samples(t_right_s, sr)
        src0 = max(0, min(len(target_y) - 1, src0))
        src1 = max(0, min(len(target_y), src1))
        if src1 - src0 <= 1:
            continue
        start_samp = time_to_samples(m_start, sr)
        end_samp = time_to_samples(m_end, sr)
        needed = max(0, end_samp - start_samp)
        if needed <= 0:
            continue
        # If possible, select the best-matching window by correlation with master envelope
        best_seg = None
        if master_env is not None:
            m0 = start_samp
            m1 = min(len(master_env), end_samp)
            ref_env = master_env[m0:m1]
            if ref_env.size >= 8 and (src1 - src0) > needed:
                cand_env = target_env[src0:src1]
                step = max(1, int(sr * 0.005))  # 5 ms step
                best_score = -1.0
                best_off = 0
                for off in range(0, max(0, cand_env.size - (m1 - m0) - 1), step):
                    seg = cand_env[off:off + (m1 - m0)]
                    if seg.size != ref_env.size:
                        break
                    denom = (np.linalg.norm(ref_env) * np.linalg.norm(seg)) + 1e-8
                    score = float(np.dot(ref_env, seg) / denom)
                    if score > best_score:
                        best_score = score
                        best_off = off
                src_sel0 = src0 + best_off
                src_sel1 = min(len(target_y), src_sel0 + (m1 - m0))
                if src_sel1 - src_sel0 > 1:
                    best_seg = target_y[src_sel0:src_sel1].copy()
        # Fallback: choose highest-energy window of required length inside search band
        if best_seg is None:
            cand = target_y[src0:src1]
            if cand.shape[-1] <= needed:
                best_seg = cand.copy()
            else:
                # sliding RMS
                win = needed
                step = max(1, int(sr * 0.005))
                best_energy = -1.0
                best_off = 0
                for off in range(0, cand.shape[-1] - win, step):
                    seg = cand[off:off + win]
                    e = float(np.mean(seg.astype(np.float32) ** 2))
                    if e > best_energy:
                        best_energy = e
                        best_off = off
                best_seg = cand[best_off:best_off + win].copy()

        clip = best_seg
        rate = max(1e-4, clip.shape[-1] / float(needed))
        stretched = uniform_time_stretch(clip, rate=rate)
        if stretched.shape[-1] < needed:
            pad = needed - stretched.shape[-1]
            clip_fit = np.pad(stretched, (0, pad), mode="constant")
        else:
            clip_fit = stretched[:needed]
        crossfade_add(out, start_samp, clip_fit, fade_ms=xfade_ms, sr=sr)
        # mark as placed
        placed_spans[i] = (m_start, m_end, True)

# --------------------------
# Advanced analysis helpers
# --------------------------

def _analyze_master_for_advanced(master_audio_path: str,
                                 secondary_audio_paths: List[str],
                                 sentence_gap_s: float,
                                 progress: gr.Progress) -> Tuple[List[Dict[str, Any]], str, List[List[Any]], Dict[str, Dict[str, Any]], str]:
    """Transcribe and return sentence grid, table rows, and multi-target editor data."""
    progress(0.02, "Analyzing master: transcription with diarization...")
    summary, outputs = process_transcription_whisperx(
        audio_files=[master_audio_path] + secondary_audio_paths,
        model_size="large-v3",
        language="auto",
        align_output=True,
        assign_speakers=True,  # Enable for master only
        batch_size=4,
        compute_type="float16",
        return_char_alignments=False,
        progress=progress or gr.Progress(track_tqdm=True)
    )

    json_map = _find_jsons_for_files(outputs, [master_audio_path] + secondary_audio_paths)
    master_json = json_map[master_audio_path]
    master_sents = build_sentences_for_audio(master_json, sentence_gap_s)

    # Build table rows: idx, start, end, text, override_start, override_end, force_place
    rows: List[List[Any]] = []
    for s in master_sents:
        rows.append([
            int(s["idx"]),
            float(s["start"]),
            float(s["end"]),
            _sent_as_text(s),
            None,
            None,
            False,
        ])

    # Build multi-target editor data
    targets_data = {}
    for tgt_path in secondary_audio_paths:
        tgt_json = json_map.get(tgt_path, master_json)  # fallback to master if no transcript
        tgt_sents = build_sentences_for_audio(tgt_json, sentence_gap_s)
        # Align target sentences to master timeline
        aligned_tgt_sents = []
        for ms in master_sents:
            m_start = float(ms["start"])
            m_end = float(ms["end"])
            # Find words in target that fall within master sentence bounds
            tgt_words = load_word_list(tgt_json)
            words_in = [w for w in tgt_words if (w["start"] >= m_start and w["end"] <= m_end)]
            text = " ".join(w["text"] for w in words_in)
            aligned_tgt_sents.append({
                "idx": ms["idx"],
                "start": m_start,
                "end": m_end,
                "text": text,
                "words": words_in,
                "duration": float(m_end - m_start)
            })

        tgt_base = os.path.splitext(os.path.basename(tgt_path))[0]
        targets_data[tgt_base] = {
            "audio_path": tgt_path,
            "sentences": [{"idx": int(s["idx"]), "start": float(s["start"]), "end": float(s["end"]), "text": s["text"]} for s in aligned_tgt_sents],
            "duration": librosa.get_duration(path=tgt_path) if os.path.exists(tgt_path) else 0.0
        }

    # Serialize for JS editor
    editor_data_json = json.dumps({
        "master": {
            "sentences": [{"idx": int(s["idx"]), "start": float(s["start"]), "end": float(s["end"]), "text": _sent_as_text(s)} for s in master_sents],
            "duration": librosa.get_duration(path=master_audio_path) if os.path.exists(master_audio_path) else 0.0
        },
        "targets": targets_data
    })

    return master_sents, master_json, rows, targets_data, editor_data_json

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
                              progress: gr.Progress,
                              auto_fit_gaps: bool = False):

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
            empties = [len(ms.get("words", [])) == 0 for ms in master_sents]
            out, placed_spans, mapped_times = _assemble_by_sentences_with_wp(
                master_sents, wp, tgt_y, sr, xfade_ms, master_len,
                master_y=master_y, max_nudge_ms=tolerance_ms, is_empty_sentence=empties, overrides=None
            )
            if auto_fit_gaps:
                _auto_fill_unmatched_segments(master_sents, placed_spans, mapped_times, tgt_y, sr, xfade_ms, out, master_y=master_y)
        else:
            # Fallback to sentence/window-based assembly
            out = np.zeros(master_len, dtype=np.float32)
            placed_spans = []
            mapped_times: List[Tuple[float, float]] = []
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
                    mapped_times.append((m_start, m_end))
                    continue
                t_start = float(tgt_sents[t_idx]["start"])
                t_end = float(tgt_sents[t_idx]["end"])
                t_dur = max(0.0, t_end - t_start)
                if t_dur <= 0.0:
                    placed_spans.append((m_start, m_end, False))
                    mapped_times.append((m_start, m_end))
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
                mapped_times.append((t_start, t_end))

            if auto_fit_gaps:
                _auto_fill_unmatched_segments(master_sents, placed_spans, mapped_times, tgt_y, sr, xfade_ms, out, master_y=master_y)

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

        with gr.Tabs():
            with gr.TabItem("Basic"):
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
                        auto_fit = gr.Checkbox(label="Auto-fit unmatched segments", value=False)
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

                def start(master, secondaries, h, autofit):
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
                            master, sec_list, float(h), p, bool(autofit)
                        )
                        audio_files = [f for f in all_files if f.endswith('.wav') and '__aligned' in f]
                        preview_audio = audio_files[0] if audio_files else None
                        return summary_text, all_files, viz_path, preview_audio
                    except Exception as e:
                        logger.error(f"Alignment failed: {e}", exc_info=True)
                        return f"❌ Error during alignment: {str(e)}", [], None, None

                start_btn.click(
                    fn=start,
                    inputs=[master_file, secondary_files, humanize, auto_fit],
                    outputs=[status, audio_outputs, viz_output, audio_preview]
                )

            with gr.TabItem("Advanced"):
                # States to carry analysis between steps
                adv_master_json_state = gr.State()
                adv_master_sents_state = gr.State()
                adv_targets_data_state = gr.State()
                # Hidden channels for rich editor
                adv_master_audio_tag = gr.Audio(visible=False, elem_id="adv-master-audio")
                adv_editor_data = gr.Textbox(visible=False, elem_id="adv-editor-data")
                adv_editor_plan = gr.Textbox(visible=False, elem_id="adv-editor-plan")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🧠 Analyze & Edit")
                        adv_master_file = gr.File(label="Master Track (single)", file_count="single", file_types=["audio"])
                        adv_secondary_files = gr.File(label="Secondary Tracks (multiple)", file_count="multiple", file_types=["audio"])
                        adv_sentence_gap = gr.Slider(label="Sentence gap (s)", minimum=0.2, maximum=1.2, step=0.1, value=0.6)
                        analyze_btn = gr.Button(value="🔍 Analyze Master", variant="secondary")
                        gr.Markdown("🎵 **Master Lyrics** (read-only) and 🎯 **Target Tracks** (editable)")
                        gr.Markdown("📝 **Instructions:** Select a master sentence, then choose a target track and drag a region to map it.")
                        adv_humanize = gr.Slider(label="Humanize", minimum=0, maximum=10, step=1, value=0,
                                                 info="Tolerance at sentence level (ms = value * 10) before stretching.")
                        adv_auto_fit = gr.Checkbox(label="Auto-fit unmatched segments", value=True)
                        run_adv_btn = gr.Button(value="🚀 Run Advanced Alignment", variant="primary")

                    with gr.Column(scale=2):
                        gr.Markdown("### 📊 Results (Advanced)")
                        adv_status = gr.Textbox(label="Status & Reports", value="Idle", lines=15, max_lines=30)

                with gr.Row():
                    with gr.Column():
                        # Amazing custom JS-powered multi-target editor
                        adv_editor_html = gr.HTML(
                            value=(
                                """
<style>
#adv-editor-root {
  font-family: ui-sans-serif, system-ui;
  background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
  border-radius: 12px;
  padding: 16px;
  border: 1px solid #334155;
  box-shadow: 0 8px 32px rgba(0,0,0,0.3);
  max-height: 600px;
  overflow: hidden;
}

#adv-editor-root .toolbar {
  display: flex;
  gap: 12px;
  align-items: center;
  margin-bottom: 16px;
  padding-bottom: 12px;
  border-bottom: 1px solid #475569;
}

#adv-editor-root .btn {
  padding: 8px 16px;
  background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
  color: white;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-weight: 500;
  transition: all 0.2s ease;
  box-shadow: 0 2px 4px rgba(59,130,246,0.3);
}

#adv-editor-root .btn:hover {
  background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
  transform: translateY(-1px);
  box-shadow: 0 4px 8px rgba(59,130,246,0.4);
}

#adv-editor-root .btn:active {
  transform: translateY(0);
}

#adv-editor-root .btn.secondary {
  background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%);
}

#adv-editor-root .btn.secondary:hover {
  background: linear-gradient(135deg, #4b5563 0%, #374151 100%);
}

#adv-editor-root .btn.danger {
  background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
}

#adv-editor-root .btn.danger:hover {
  background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
}

#adv-editor-root .status {
  margin-left: auto;
  padding: 6px 12px;
  background: rgba(59,130,246,0.1);
  border-radius: 6px;
  color: #93c5fd;
  font-size: 0.875rem;
}

#adv-editor-root .main-content {
  display: flex;
  gap: 16px;
  height: 500px;
}

#adv-editor-root .waveform-section {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

#adv-editor-root .waveform-track {
  background: rgba(15,23,42,0.8);
  border-radius: 8px;
  padding: 12px;
  border: 1px solid #334155;
}

#adv-editor-root .waveform-track h4 {
  margin: 0 0 8px 0;
  color: #e2e8f0;
  font-size: 0.875rem;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 8px;
}

#adv-editor-root .waveform {
  height: 120px;
  border-radius: 4px;
  overflow: hidden;
}

#adv-editor-root .timeline {
  height: 20px;
  margin-top: 4px;
  background: rgba(51,65,85,0.5);
  border-radius: 4px;
}

#adv-editor-root .sentences-section {
  flex: 1;
  background: rgba(15,23,42,0.8);
  border-radius: 8px;
  padding: 12px;
  border: 1px solid #334155;
  overflow-y: auto;
}

#adv-editor-root .sentences-section h4 {
  margin: 0 0 12px 0;
  color: #e2e8f0;
  font-size: 0.875rem;
  font-weight: 600;
}

#adv-editor-root .sentence-group {
  margin-bottom: 16px;
  border: 1px solid #475569;
  border-radius: 6px;
  overflow: hidden;
}

#adv-editor-root .sentence-group-header {
  background: rgba(59,130,246,0.1);
  padding: 8px 12px;
  border-bottom: 1px solid #475569;
  font-weight: 600;
  color: #93c5fd;
  font-size: 0.875rem;
}

#adv-editor-root .sentence-row {
  display: flex;
  gap: 8px;
  align-items: center;
  padding: 8px 12px;
  background: rgba(30,41,59,0.5);
  border-bottom: 1px solid #475569;
}

#adv-editor-root .sentence-row:last-child {
  border-bottom: none;
}

#adv-editor-root .sentence-info {
  flex: 0 0 150px;
  color: #94a3b8;
  font-size: 0.75rem;
  font-family: monospace;
}

#adv-editor-root .sentence-text {
  flex: 1;
  background: rgba(51,65,85,0.5);
  border: 1px solid #475569;
  border-radius: 4px;
  padding: 6px 8px;
  color: #e2e8f0;
  font-size: 0.875rem;
}

#adv-editor-root .sentence-controls {
  display: flex;
  gap: 4px;
  align-items: center;
}

#adv-editor-root .target-select {
  background: rgba(51,65,85,0.8);
  border: 1px solid #475569;
  border-radius: 4px;
  padding: 4px 8px;
  color: #e2e8f0;
  font-size: 0.75rem;
  min-width: 120px;
}

#adv-editor-root .btn-small {
  padding: 4px 8px;
  font-size: 0.75rem;
}

#adv-editor-root .mapping-indicator {
  padding: 2px 6px;
  background: rgba(168,85,247,0.2);
  border: 1px solid #a855f7;
  border-radius: 3px;
  font-size: 0.75rem;
  color: #c084fc;
}

#adv-editor-root .no-mapping {
  text-align: center;
  padding: 20px;
  color: #64748b;
  font-style: italic;
}

#adv-editor-root .region-marker {
  background: rgba(34,197,94,0.3);
  border: 1px solid #22c55e;
}

#adv-editor-root .region-marker.override {
  background: rgba(168,85,247,0.3);
  border-color: #a855f7;
}
</style>
<div id="adv-editor-root">
  <div class="toolbar">
    <button class="btn" id="adv-ed-zoom-in">🔍 Zoom In</button>
    <button class="btn" id="adv-ed-zoom-out">🔍 Zoom Out</button>
    <button class="btn" id="adv-ed-play">▶️ Play/Pause</button>
    <button class="btn secondary" id="adv-ed-save">💾 Save All</button>
    <button class="btn danger" id="adv-ed-clear">🗑️ Clear All</button>
    <div class="status" id="adv-ed-status">Ready</div>
  </div>

  <div class="main-content">
    <div class="waveform-section">
      <div class="waveform-track">
        <h4>🎵 Master Track</h4>
        <div id="ws-master" class="waveform"></div>
        <div id="ws-master-timeline" class="timeline"></div>
      </div>
    </div>

    <div class="sentences-section">
      <h4>📝 Master Sentences & Target Mappings</h4>
      <div id="adv-ed-sentences"></div>
    </div>
  </div>
</div>
<script type="module">
import WaveSurfer from "https://cdn.jsdelivr.net/npm/wavesurfer.js@7/dist/wavesurfer.esm.js";
import Timeline from "https://cdn.jsdelivr.net/npm/wavesurfer.js@7/dist/plugins/timeline.esm.js";
import Regions from "https://cdn.jsdelivr.net/npm/wavesurfer.js@7/dist/plugins/regions.esm.js";

const q = (sel) => document.querySelector(sel);
const msg = (t) => { const el = q('#adv-ed-status'); if (el) el.textContent = t; };

function getTextboxValue(elemId){
  const host = document.getElementById(elemId);
  if(!host) return null;
  const ta = host.querySelector('textarea');
  return ta ? ta.value : null;
}
function setTextboxValue(elemId, val){
  const host = document.getElementById(elemId);
  if(!host) return;
  const ta = host.querySelector('textarea');
  if(ta){ ta.value = val; ta.dispatchEvent(new Event('input', { bubbles: true })); }
}

function getAudioSrc(elemId){
  const host = document.getElementById(elemId);
  if(!host) return null;
  const audio = host.querySelector('audio');
  return audio ? audio.src : null;
}

let master, masterRegs;
let editorData = null;
let plans = {};

function updateSentencesUI(){
  const root = q('#adv-ed-sentences');
  if(!root) return;

  root.innerHTML = '';

  const masterSentences = editorData?.master?.sentences || [];
  const targets = editorData?.targets || {};

  masterSentences.forEach(masterSentence => {
    const group = document.createElement('div');
    group.className = 'sentence-group';

    const header = document.createElement('div');
    header.className = 'sentence-group-header';
    header.textContent = `Master #${masterSentence.idx} [${masterSentence.start.toFixed(2)}–${masterSentence.end.toFixed(2)}]`;

    const row = document.createElement('div');
    row.className = 'sentence-row';

    const info = document.createElement('div');
    info.className = 'sentence-info';
    info.textContent = 'Master Text:';

    const text = document.createElement('input');
    text.className = 'sentence-text';
    text.type = 'text';
    text.value = masterSentence.text || '';
    text.readOnly = true; // Master text is read-only

    const controls = document.createElement('div');
    controls.className = 'sentence-controls';

    // Create dropdown for target selection
    const targetSelect = document.createElement('select');
    targetSelect.className = 'target-select';
    targetSelect.innerHTML = '<option value="">Select target...</option>';

    Object.keys(targets).forEach(targetName => {
      const option = document.createElement('option');
      option.value = targetName;
      option.textContent = `🎯 ${targetName}`;
      targetSelect.appendChild(option);
    });

    const mapBtn = document.createElement('button');
    mapBtn.className = 'btn btn-small';
    mapBtn.textContent = '🎯 Map Region';
    mapBtn.disabled = true;

    const clearBtn = document.createElement('button');
    clearBtn.className = 'btn btn-small secondary';
    clearBtn.textContent = '❌';
    clearBtn.style.display = 'none';

    // Show current mappings
    let hasMappings = false;
    Object.entries(targets).forEach(([targetName, targetData]) => {
      const targetPlan = plans[targetName] || {};
      if(targetPlan[masterSentence.idx]){
        hasMappings = true;
        const mapping = document.createElement('div');
        mapping.className = 'mapping-indicator';
        mapping.textContent = `${targetName}: [${targetPlan[masterSentence.idx].t0.toFixed(2)}–${targetPlan[masterSentence.idx].t1.toFixed(2)}]`;
        controls.appendChild(mapping);
      }
    });

    if(hasMappings){
      clearBtn.style.display = 'inline-block';
    }

    targetSelect.addEventListener('change', () => {
      mapBtn.disabled = !targetSelect.value;
    });

    mapBtn.addEventListener('click', () => {
      const selectedTarget = targetSelect.value;
      if(!selectedTarget) return;

      // For now, we'll use a placeholder - in a full implementation,
      // you'd have individual target waveforms or a unified interface
      msg(`Click and drag on the master waveform to select a region for ${selectedTarget}`);

      // Simplified: just store a placeholder mapping
      if(!plans[selectedTarget]) plans[selectedTarget] = {};
      plans[selectedTarget][masterSentence.idx] = { t0: masterSentence.start, t1: masterSentence.end };

      updateSentencesUI();
      msg(`Mapped to ${selectedTarget}`);
    });

    clearBtn.addEventListener('click', () => {
      Object.keys(targets).forEach(targetName => {
        if(plans[targetName] && plans[targetName][masterSentence.idx]){
          delete plans[targetName][masterSentence.idx];
          if(Object.keys(plans[targetName]).length === 0){
            delete plans[targetName];
          }
        }
      });
      updateSentencesUI();
      msg('Cleared all mappings');
    });

    controls.appendChild(targetSelect);
    controls.appendChild(mapBtn);
    controls.appendChild(clearBtn);

    row.appendChild(info);
    row.appendChild(text);
    row.appendChild(controls);

    group.appendChild(header);
    group.appendChild(row);
    root.appendChild(group);
  });

  if(masterSentences.length === 0){
    const noData = document.createElement('div');
    noData.className = 'no-mapping';
    noData.textContent = 'No sentences detected. Run analysis first.';
    root.appendChild(noData);
  }
}

function initWaves(){
  const msrc = getAudioSrc('adv-master-audio');
  if(!msrc){ msg('Waiting for master audio...'); return; }

  if(master){ master.destroy(); master = null; }

  master = WaveSurfer.create({
    container: '#ws-master',
    waveColor: '#60a5fa',
    progressColor: '#3b82f6',
    height: 120,
    responsive: true
  });
  masterRegs = master.registerPlugin(Regions.create({ dragSelection: { slop: 5 } }));
  master.registerPlugin(Timeline.create({ container: '#ws-master-timeline' }));

  master.load(msrc);

  // Draw master sentence regions
  master.on('ready', () => {
    masterRegs.clearRegions();
    (editorData?.master?.sentences || []).forEach(s => {
      masterRegs.addRegion({
        start: s.start,
        end: s.end,
        drag: false,
        resize: false,
        color: 'rgba(34,197,94,0.2)',
        content: `#${s.idx}`,
        contentEditable: false
      });
    });
  });

  // Handle region selection for mapping
  masterRegs.on('region-created', (region) => {
    msg(`Selected region [${region.start.toFixed(2)}–${region.end.toFixed(2)}]`);
    // In a full implementation, this would update the current mapping
  });
}

function ensureInit(){
  const raw = getTextboxValue('adv-editor-data');
  if(!raw){ setTimeout(ensureInit, 500); return; }
  try { editorData = JSON.parse(raw); } catch { editorData = { master: { sentences: [] }, targets: {} }; }

  updateSentencesUI();
  setTimeout(initWaves, 600);
}

// Toolbar events
q('#adv-ed-zoom-in')?.addEventListener('click', () => { master?.zoom(1.25); });
q('#adv-ed-zoom-out')?.addEventListener('click', () => { master?.zoom(0.8); });
q('#adv-ed-play')?.addEventListener('click', () => { master?.playPause(); });
q('#adv-ed-save')?.addEventListener('click', () => {
  const payload = { plans, master: editorData?.master, targets: editorData?.targets };
  setTextboxValue('adv-editor-plan', JSON.stringify(payload));
  msg('All edits saved!');
});
q('#adv-ed-clear')?.addEventListener('click', () => {
  plans = {};
  updateSentencesUI();
  msg('All edits cleared');
});

ensureInit();
</script>
"""
                            ),
                            elem_id="adv-editor-html"
                        )

                with gr.Row():
                    with gr.Column():
                        adv_audio_outputs = gr.File(label="Download Aligned Tracks", file_count="multiple")

                with gr.Row():
                    with gr.Column():
                        adv_viz_output = gr.Image(label="Track Alignment Overlay", type="filepath")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("Preview the first aligned track below:")
                        adv_audio_preview = gr.Audio(label="Preview Aligned Audio", visible=True)

                def _build_overrides(rows, num_sents: int) -> List[Optional[Tuple[float, float]]]:
                    ov: List[Optional[Tuple[float, float]]] = [None] * num_sents

                    # Handle different input types (list of lists, pandas DataFrame, or None)
                    if rows is None:
                        return ov

                    # Convert pandas DataFrame to list of lists if needed
                    if hasattr(rows, 'values'):  # pandas DataFrame
                        row_list = rows.values.tolist()
                    elif isinstance(rows, list):
                        row_list = rows
                    else:
                        return ov

                    for r in row_list:
                        try:
                            idx = int(r[0])
                        except Exception:
                            continue
                        if idx < 0 or idx >= num_sents:
                            continue
                        t0 = r[4]
                        t1 = r[5]
                        if t0 is None or t1 is None:
                            continue
                        try:
                            t0f = float(t0)
                            t1f = float(t1)
                            if t1f > t0f:
                                ov[idx] = (t0f, t1f)
                        except Exception:
                            pass
                    return ov

                def analyze(master, secondaries, gap_s):
                    if not master:
                        return [], None, None, None, None, None
                    sec_list = secondaries or []
                    if isinstance(sec_list, str):
                        sec_list = [sec_list]
                    sec_list = [s for s in sec_list if s and os.path.exists(s)]
                    if not sec_list:
                        return [], None, None, None, None, None

                    p = gr.Progress(track_tqdm=True)
                    sents, master_json, rows, targets_data, editor_data_json = _analyze_master_for_advanced(master, sec_list, float(gap_s), p)
                    return rows, master_json, sents, targets_data, editor_data_json, master

                analyze_btn.click(
                    fn=analyze,
                    inputs=[adv_master_file, adv_secondary_files, adv_sentence_gap],
                    outputs=[adv_master_json_state, adv_master_sents_state, adv_targets_data_state, adv_editor_data, adv_master_audio_tag]
                )

                def run_advanced(master, secondaries, h, autofit, master_json, master_sents, targets_data, plan_json):
                    if not master:
                        return "⚠️ Please provide a master track.", [], None, None
                    if master_sents is None or master_json is None:
                        return "⚠️ Please click 'Analyze Master' first.", [], None, None
                    sec_list = secondaries or []
                    if isinstance(sec_list, str):
                        sec_list = [sec_list]
                    sec_list = [s for s in sec_list if s and os.path.exists(s)]
                    if not sec_list:
                        return "⚠️ Please provide at least one secondary track.", [], None, None

                    sentence_gap_s = 0.6
                    # Ensure master sentences re-built from saved json for safety
                    try:
                        ms = build_sentences_for_audio(master_json, sentence_gap_s)
                    except Exception:
                        ms = master_sents

                    p = gr.Progress(track_tqdm=True)

                    out_root, folder_name = _project_folder(master, sec_list)
                    os.makedirs(out_root, exist_ok=True)

                    master_y, sr = _load_audio_any(master, sr=None)
                    master_len = len(master_y)
                    tolerance_ms = max(0.0, float(h) * 10.0)
                    xfade_ms = 10.0

                    overrides = [None] * len(ms)
                    # Use editor plan overrides if present
                    try:
                        if plan_json:
                            pj = json.loads(plan_json)
                            plans = pj.get("plans", {})
                            # Convert per-target plans into sentence overrides
                            for target_name, target_plans in plans.items():
                                for sentence_idx, mapping in target_plans.items():
                                    try:
                                        idx = int(sentence_idx)
                                        if idx < 0 or idx >= len(ms):
                                            continue
                                        t0 = float(mapping.get("t0", None))
                                        t1 = float(mapping.get("t1", None))
                                        if t0 is not None and t1 is not None and t1 > t0:
                                            overrides[idx] = (t0, t1)
                                    except Exception:
                                        pass
                    except Exception:
                        pass

                    all_export_files: List[str] = []
                    report_summaries: List[str] = []
                    overlay_path: Optional[str] = None

                    # Save master copy
                    try:
                        master_base = os.path.splitext(os.path.basename(master))[0]
                        master_copy_path = os.path.join(out_root, f"{master_base}__master.wav")
                        sf.write(master_copy_path, master_y, sr)
                        all_export_files.append(master_copy_path)
                    except Exception as e:
                        logger.warning(f"Failed to save master track copy: {e}")

                    # Transcription for targets (reuse whisperx wrapper for consistency)
                    summary, outputs = process_transcription_whisperx(
                        audio_files=sec_list,
                        model_size="large-v3",
                        language="auto",
                        align_output=True,
                        assign_speakers=False,
                        batch_size=8,
                        compute_type="float16",
                        return_char_alignments=False,
                        progress=p
                    )
                    json_map = _find_jsons_for_files(outputs, sec_list)

                    for idx, tgt_path in enumerate(sec_list):
                        tgt_json = json_map[tgt_path]
                        # Build target sentence list by binning target words into master windows (same as basic)
                        tgt_words = load_word_list(tgt_json)
                        tgt_sents = []
                        for ms_i in ms:
                            start = float(ms_i["start"])
                            end = float(ms_i["end"])
                            words_in = [w for w in tgt_words if (w["start"] >= start and w["end"] <= end)]
                            text = " ".join(w["text"] for w in words_in)
                            tgt_sents.append({
                                "idx": ms_i["idx"],
                                "start": start,
                                "end": end,
                                "duration": float(end - start),
                                "words": words_in,
                                "text": text,
                                "norm_text": normalize_text(text),
                                "tokens": [normalize_word(w["text"]) for w in words_in if w["text"].strip()]
                            })

                        per_sentence_reports = []
                        for i_m in range(len(ms)):
                            cost, conf = sentence_pair_cost(ms[i_m], tgt_sents[i_m])
                            if not tgt_sents[i_m]["words"] and not (overrides[i_m] is not None):
                                per_sentence_reports.append({
                                    "master_idx": i_m,
                                    "target_idx": None,
                                    "confidence": 0.0,
                                    "master_text": _sent_as_text(ms[i_m]),
                                    "target_text": None,
                                    "word_alignment": []
                                })
                            else:
                                m_words = [normalize_word(w["text"]) for w in ms[i_m]["words"]]
                                t_words = [normalize_word(w["text"]) for w in tgt_sents[i_m]["words"]]
                                word_map = align_words(m_words, t_words)
                                per_sentence_reports.append({
                                    "master_idx": i_m,
                                    "target_idx": i_m,
                                    "confidence": conf,
                                    "master_text": _sent_as_text(ms[i_m]),
                                    "target_text": _sent_as_text(tgt_sents[i_m]),
                                    "word_alignment": [{"m": mi, "t": ti} for (mi, ti) in word_map]
                                })

                        tgt_y, _ = _load_audio_any(tgt_path, sr=sr)
                        wp = _compute_warp_path(master, tgt_path)
                        if wp is not None:
                            empties = [len(x.get("words", [])) == 0 for x in ms]
                            out, placed_spans, mapped_times = _assemble_by_sentences_with_wp(
                                ms, wp, tgt_y, sr, xfade_ms, master_len,
                                master_y=master_y, max_nudge_ms=tolerance_ms, is_empty_sentence=empties, overrides=overrides
                            )
                            if autofit:
                                _auto_fill_unmatched_segments(ms, placed_spans, mapped_times, tgt_y, sr, xfade_ms, out, master_y=master_y)
                        else:
                            # Fallback identical to basic, with mapped_times for auto-fill
                            out = np.zeros(master_len, dtype=np.float32)
                            placed_spans = []
                            mapped_times: List[Tuple[float, float]] = []
                            for i_m in range(len(ms)):
                                m_start = float(ms[i_m]["start"])
                                m_end = float(ms[i_m]["end"])
                                start_samp = time_to_samples(m_start, sr)
                                end_samp = time_to_samples(m_end, sr)
                                if overrides[i_m] is not None:
                                    t0o, t1o = overrides[i_m]
                                    s0 = time_to_samples(t0o, sr)
                                    s1 = time_to_samples(t1o, sr)
                                    clip = tgt_y[s0:s1].copy()
                                else:
                                    t0 = time_to_samples(tgt_sents[i_m]["start"], sr)
                                    t1 = time_to_samples(tgt_sents[i_m]["end"], sr)
                                    clip = tgt_y[t0:t1].copy()
                                needed = max(0, end_samp - start_samp)
                                if needed <= 0:
                                    placed_spans.append((m_start, m_end, False))
                                    mapped_times.append((m_start, m_end))
                                    continue
                                if clip.shape[-1] < needed:
                                    pad = needed - clip.shape[-1]
                                    clip = np.pad(clip, (0, pad), mode="constant")
                                else:
                                    clip = clip[:needed]
                                crossfade_add(out, start_samp, clip, fade_ms=xfade_ms, sr=sr)
                                placed_spans.append((m_start, m_end, True))
                                if overrides[i_m] is not None:
                                    mapped_times.append(overrides[i_m])
                                else:
                                    mapped_times.append((tgt_sents[i_m]["start"], tgt_sents[i_m]["end"]))
                            if autofit:
                                _auto_fill_unmatched_segments(ms, placed_spans, mapped_times, tgt_y, sr, xfade_ms, out, master_y=master_y)

                        tgt_base = os.path.splitext(os.path.basename(tgt_path))[0]
                        aligned_wav = os.path.join(out_root, f"{tgt_base}__aligned.wav")
                        sf.write(aligned_wav, out, sr)
                        all_export_files.append(aligned_wav)

                        report = {
                            "master_audio": master,
                            "target_audio": tgt_path,
                            "sample_rate": sr,
                            "humanize_ms": tolerance_ms,
                            "sentence_gap_s": sentence_gap_s,
                            "master_sentences": [{
                                "idx": s["idx"], "start": s["start"], "end": s["end"],
                                "text": _sent_as_text(s)
                            } for s in ms],
                            "target_sentences": [{
                                "idx": s["idx"], "start": s["start"], "end": s["end"],
                                "text": _sent_as_text(s)
                            } for s in tgt_sents],
                            "matches": per_sentence_reports,
                            "overrides": [{"idx": i, "t0": (overrides[i][0] if overrides[i] else None), "t1": (overrides[i][1] if overrides[i] else None)} for i in range(len(overrides))]
                        }
                        report_path = os.path.join(out_root, f"{tgt_base}__report.json")
                        with open(report_path, "w", encoding="utf-8") as f:
                            json.dump(report, f, indent=2)
                        all_export_files.append(report_path)

                        matched = sum(1 for pr in per_sentence_reports if pr.get("target_idx") is not None)
                        missed = len(ms) - matched
                        confs = [float(pr.get("confidence", 0.0)) for pr in per_sentence_reports if pr.get("target_idx") is not None]
                        avg_conf = float(np.mean(confs)) if confs else 0.0
                        report_summaries.append(
                            f"[{tgt_base}] matched={matched}  unmatched={missed}  avg_conf={avg_conf:.3f}"
                        )

                        if idx == 0:
                            overlay_path = os.path.join(out_root, "alignment_visualization.png")
                            _plot_overlay(master_y, out, sr, ms, [(i, (i if pr.get('target_idx') is not None else None), float(pr.get('confidence', 0.0))) for i, pr in enumerate(per_sentence_reports)], overlay_path)

                    summary_lines = [f"Output: {out_root}"]
                    summary_lines.extend(report_summaries)
                    summary_text = "\n".join(summary_lines)

                    audio_files = [f for f in all_export_files if f.endswith('.wav') and '__aligned' in f]
                    preview_audio = audio_files[0] if audio_files else None
                    return summary_text, all_export_files, overlay_path, preview_audio

                run_adv_btn.click(
                    fn=run_advanced,
                    inputs=[adv_master_file, adv_secondary_files, adv_humanize, adv_auto_fit, adv_master_json_state, adv_master_sents_state, adv_targets_data_state, adv_editor_plan],
                    outputs=[adv_status, adv_audio_outputs, adv_viz_output, adv_audio_preview]
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
        "audio_preview": "Preview the first aligned track.",
        "adv_master_file": "Master track for Advanced tab.",
        "adv_secondary_files": "Secondary tracks for Advanced tab.",
        "adv_sentence_gap": "Controls sentence grouping window for analysis.",
        "analyze_btn": "Analyze master (transcribe, diarize, sentence grid).",
        "adv_table": "Editable sentences: text, optional overrides (seconds), force place.",
        "adv_humanize": "Tolerance before stretching in Advanced mode.",
        "adv_auto_fit": "Attempt to fill unmatched sentences using smart search.",
        "run_adv_btn": "Run alignment using overrides and advanced fitting.",
        "adv_status": "Reports for Advanced run.",
        "adv_audio_outputs": "Download aligned tracks from Advanced run.",
        "adv_viz_output": "Overlay visualization for Advanced run.",
        "adv_audio_preview": "Preview from Advanced run."
    }
    for elem_id, desc in descriptions.items():
        if hasattr(arg_handler_local, "register_description"):
            arg_handler_local.register_description("align", elem_id, desc)

def listen():
    return
