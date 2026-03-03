"""
V3 dataset components: loader and collate that add lyrics/text_tokens to the V2 pipeline.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from modules.rvc.infer.lib.train.data_utils import (
    TextAudioLoaderMultiNSFsid,
    TextAudioCollateMultiNSFsid,
)
from modules.rvc.infer.lib.train.utils import load_filepaths_and_text

logger = logging.getLogger(__name__)


def _load_lyrics_manifest() -> Dict[str, Dict[str, str]]:
    """Load optional smoke lyrics manifest (project -> stem -> lyrics)."""
    candidates = [
        Path(__file__).resolve().parent.parent.parent.parent / "testing" / "smoke_lyrics_manifest.json",
        Path.cwd() / "testing" / "smoke_lyrics_manifest.json",
    ]
    for p in candidates:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return {k: v for k, v in data.items() if isinstance(v, dict) and not k.startswith("_")}
            except Exception as e:
                logger.warning("Could not load lyrics manifest %s: %s", p, e)
    return {}


def _apply_simple_tagging(segments: list, full_text: str) -> str:
    """
    Apply simple advanced tagging for auto-transcribed lyrics.
    - If segments have no tags, prepend [clean] as default vocal style.
    - Detects repeated phrases (potential chorus) and tags with [clean].
    """
    if not segments and not full_text:
        return ""
    text = full_text or " ".join(s.get("text", "") for s in segments)
    text = (text or "").strip()
    if not text:
        return ""

    # Check if any segment already has tags
    has_tags = any(s.get("tags") for s in segments)
    if has_tags:
        return text

    # Default: prepend [clean] for neutral vocal style
    return "[clean] " + text


def _get_lyrics_for_stem(
    lyrics_dir: Path,
    stem: str,
    phonemizer,
    apply_tagging: bool = True,
    manifest_override: Optional[Dict[str, str]] = None,
    project_name: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load lyrics from lyrics_dir/<stem>.json or manifest, phonemize, return (text_tokens, text_mask).
    text_mask: False = valid token, True = padding (invalid).
    Returns (empty_tokens, empty_mask) if no lyrics or phonemizer unavailable.
    """
    text = None
    source_desc = f"{lyrics_dir}/{stem}.json"
    segments = []
    if manifest_override and project_name and stem in manifest_override:
        text = manifest_override.get(stem, "").strip()
    if not text:
        lyrics_file = lyrics_dir / f"{stem}.json"
        if not lyrics_file.exists():
            return torch.zeros(1, dtype=torch.long), torch.ones(1, dtype=torch.bool)

        if phonemizer is None:
            return torch.zeros(1, dtype=torch.long), torch.ones(1, dtype=torch.bool)

        try:
            with open(lyrics_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning("Failed to load lyrics %s: %s", lyrics_file, e)
            return torch.zeros(1, dtype=torch.long), torch.ones(1, dtype=torch.bool)

        segments = data.get("segments", [])
        full_text = data.get("full_text", "")

        if not segments and not full_text:
            return torch.zeros(1, dtype=torch.long), torch.ones(1, dtype=torch.bool)

        # Use full_text or concatenate segment texts
        if full_text:
            text = full_text
        else:
            text = " ".join(s.get("text", "") for s in segments)

    if apply_tagging:
        text = _apply_simple_tagging(segments, text)
    else:
        text = (text or "").strip()
    if not text:
        return torch.zeros(1, dtype=torch.long), torch.ones(1, dtype=torch.bool)

    try:
        # Support both Phonemizer (phonemize->strs, encode->ids) and SimplePhonemizer (phonemize->ids)
        try:
            phonemes = phonemizer.phonemize(text, preserve_tags=True)
        except TypeError:
            phonemes = phonemizer.phonemize(text)
        if phonemes and isinstance(phonemes[0], int):
            token_ids = phonemes
        elif hasattr(phonemizer, "encode"):
            token_ids = phonemizer.encode(phonemes)
        else:
            token_ids = []
    except Exception as e:
        logger.warning("Phonemization failed for %s: %s", source_desc, e)
        return torch.zeros(1, dtype=torch.long), torch.ones(1, dtype=torch.bool)

    if not token_ids:
        return torch.zeros(1, dtype=torch.long), torch.ones(1, dtype=torch.bool)

    tokens = torch.LongTensor(token_ids)
    mask = torch.zeros(len(token_ids), dtype=torch.bool)  # False = valid
    return tokens, mask


class TextAudioLoaderMultiNSFsidV3(TextAudioLoaderMultiNSFsid):
    """
    V2 loader extended to load per-stem lyrics and return text_tokens/text_mask.
    """

    def __init__(
        self,
        audiopaths_and_text: str,
        hparams,
        project_dir: str,
        phonemizer: Optional[Any] = None,
    ):
        super().__init__(audiopaths_and_text, hparams)
        self.project_dir = Path(project_dir)
        self.lyrics_dir = self.project_dir / "lyrics"
        self.phonemizer = phonemizer
        manifest_all = _load_lyrics_manifest()
        self.manifest_override = manifest_all.get(self.project_dir.name, {})

    def get_audio_text_pair(self, audiopath_and_text):
        result = super().get_audio_text_pair(audiopath_and_text)
        spec, wav, phone, pitch, pitchf, sid = result

        wav_path = audiopath_and_text[0]
        stem = Path(wav_path).stem
        text_tokens, text_mask = _get_lyrics_for_stem(
            self.lyrics_dir,
            stem,
            self.phonemizer,
            apply_tagging=True,
            manifest_override=self.manifest_override or None,
            project_name=self.project_dir.name,
        )

        return (spec, wav, phone, pitch, pitchf, sid, text_tokens, text_mask)


class TextAudioCollateMultiNSFsidV3(TextAudioCollateMultiNSFsid):
    """
    Collate that pads text_tokens/text_mask in addition to V2 fields.
    """

    def __init__(self, return_ids: bool = False):
        super().__init__(return_ids=return_ids)

    def __call__(self, batch):
        # Check if batch has V3 text (7-element tuples: ..., text_tokens, text_mask)
        has_text = len(batch) > 0 and len(batch[0]) >= 8

        if not has_text:
            # Fallback to standard V2 collate (drop any extra elements)
            v2_batch = [row[:6] for row in batch]
            return super().__call__(v2_batch) + (None, None)

        # Sort by spec length (phone length) descending
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].size(1) for x in batch]), dim=0, descending=True
        )

        batch_size = len(batch)
        max_spec_len = max(x[0].size(1) for x in batch)
        max_wave_len = max(x[1].size(1) for x in batch)
        max_phone_len = max(x[2].size(0) for x in batch)

        spec_padded = torch.FloatTensor(batch_size, batch[0][0].size(0), max_spec_len)
        wave_padded = torch.FloatTensor(batch_size, 1, max_wave_len)
        phone_padded = torch.FloatTensor(batch_size, max_phone_len, batch[0][2].shape[1])
        pitch_padded = torch.LongTensor(batch_size, max_phone_len)
        pitchf_padded = torch.FloatTensor(batch_size, max_phone_len)
        spec_lengths = torch.LongTensor(batch_size)
        wave_lengths = torch.LongTensor(batch_size)
        phone_lengths = torch.LongTensor(batch_size)
        sid = torch.LongTensor(batch_size)

        spec_padded.zero_()
        wave_padded.zero_()
        phone_padded.zero_()
        pitch_padded.zero_()
        pitchf_padded.zero_()

        # Text
        max_text_len = max(x[6].size(0) for x in batch)
        text_tokens_padded = torch.zeros(batch_size, max_text_len, dtype=torch.long)
        text_mask_padded = torch.ones(batch_size, max_text_len, dtype=torch.bool)

        for i, idx in enumerate(ids_sorted_decreasing):
            row = batch[idx]
            spec, wav, phone, pitch, pitchf, dv, text_tokens, text_mask = row[:8]

            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wave_padded[i, :, : wav.size(1)] = wav
            wave_lengths[i] = wav.size(1)

            phone_padded[i, : phone.size(0), :] = phone
            phone_lengths[i] = phone.size(0)

            pitch_padded[i, : pitch.size(0)] = pitch
            pitchf_padded[i, : pitchf.size(0)] = pitchf

            sid[i] = dv

            text_tokens_padded[i, : text_tokens.size(0)] = text_tokens
            text_mask_padded[i, : text_mask.size(0)] = text_mask

        return (
            phone_padded,
            phone_lengths,
            pitch_padded,
            pitchf_padded,
            spec_padded,
            spec_lengths,
            wave_padded,
            wave_lengths,
            sid,
            text_tokens_padded,
            text_mask_padded,
        )
