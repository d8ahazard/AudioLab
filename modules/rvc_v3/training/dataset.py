"""
Dataset class for RVC V3 training.

Loads audio, content features, pitch, and text/lyric tokens.
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import librosa

logger = logging.getLogger(__name__)


class RVCV3Dataset(Dataset):
    """
    Dataset for RVC V3 training.
    
    Loads:
    - Audio segments
    - Pre-extracted content features (HuBERT + Whisper)
    - Pitch contours (F0)
    - Text/lyric tokens with tags
    - Speaker IDs
    """
    
    def __init__(
        self,
        project_dir: str,
        config,
        phonemizer,
        split: str = "train",
        augment: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            project_dir: Path to project directory
            config: RVCV3Config object
            phonemizer: Phonemizer instance for text tokenization
            split: 'train' or 'val'
            augment: Whether to apply augmentation
        """
        self.project_dir = Path(project_dir)
        self.config = config
        self.phonemizer = phonemizer
        self.split = split
        self.augment = augment
        
        self.sr = config.sampling_rate
        self.hop_length = config.hop_length
        self.segment_size = config.segment_size
        
        # Paths
        self.vocals_dir = self.project_dir / config.vocals_dir
        self.features_dir = self.project_dir / config.features_dir
        self.lyrics_dir = self.project_dir / config.lyrics_dir
        
        # Load file list
        self.file_list = self._load_file_list()
        
        # Load annotated lyrics
        self.lyrics_data = self._load_lyrics()
        
        logger.info(
            f"RVCV3Dataset ({split}): {len(self.file_list)} files loaded"
        )
    
    def _load_file_list(self) -> List[str]:
        """Load list of training files."""
        # Look for a split file
        split_file = self.project_dir / f"{self.split}_files.txt"
        
        if split_file.exists():
            with open(split_file, 'r') as f:
                files = [line.strip() for line in f if line.strip()]
        else:
            # Auto-split: use all vocal files
            vocal_files = sorted(self.vocals_dir.glob("*.wav"))
            
            # Simple train/val split (90/10)
            if self.split == "train":
                files = [f.stem for f in vocal_files[:int(len(vocal_files) * 0.9)]]
            else:
                files = [f.stem for f in vocal_files[int(len(vocal_files) * 0.9):]]
        
        return files
    
    def _load_lyrics(self) -> Dict:
        """Load annotated lyrics."""
        lyrics_file = self.lyrics_dir / "annotated_lyrics.json"
        
        if not lyrics_file.exists():
            logger.warning(f"No annotated lyrics found at {lyrics_file}")
            return {}
        
        with open(lyrics_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.
        
        Returns:
            Dictionary containing:
            - audio: Audio segment (segment_size,)
            - content_features: Content features (T', feature_dim)
            - pitch: Coarse pitch (T',)
            - pitchf: Fine pitch (T',)
            - text_tokens: Text token IDs (text_len,)
            - text_mask: Text padding mask (text_len,)
            - speaker_id: Speaker ID (scalar)
        """
        file_stem = self.file_list[idx]
        
        # Load audio
        audio_path = self.vocals_dir / f"{file_stem}.wav"
        audio, sr = librosa.load(str(audio_path), sr=self.sr, mono=True)
        
        # Load pre-extracted features
        features_file = self.features_dir / f"{file_stem}.pt"
        if features_file.exists():
            features_data = torch.load(features_file, map_location='cpu', weights_only=True)
            content_features = features_data['content']  # (T', feature_dim)
            pitch = features_data['pitch']  # (T',)
            pitchf = features_data['pitchf']  # (T',)
        else:
            raise FileNotFoundError(f'Extract content and pitch before training: {features_file}')

        # Cached encoder frames (~50 Hz) and STFT frames have different clocks.
        # Align the entire utterance before taking a hop-aligned audio crop.
        n_frames = max(1, len(audio) // self.hop_length)
        content_features = F.interpolate(content_features.T[None], size=n_frames,
                                         mode='nearest')[0].T
        voiced = F.interpolate((pitchf > 0).float()[None,None], size=n_frames,
                               mode='nearest')[0,0].bool()
        pitchf = F.interpolate(pitchf.float()[None,None], size=n_frames,
                               mode='linear', align_corners=False)[0,0]
        pitchf = torch.where(voiced, pitchf, 0.)
        pitch = F.interpolate(pitch.float()[None,None], size=n_frames,
                              mode='nearest')[0,0].long()
        
        # Random segment extraction for training
        start = 0
        if len(audio) > self.segment_size:
            max_start = len(audio) - self.segment_size
            start = (random.randint(0, max_start // self.hop_length) * self.hop_length
                     if self.split == 'train' else 0)
            audio = audio[start:start + self.segment_size]
            
            # Adjust feature indices
            start_frame = start // self.hop_length
            end_frame = start_frame + (self.segment_size // self.hop_length)
            
            content_features = content_features[start_frame:end_frame]
            pitch = pitch[start_frame:end_frame]
            pitchf = pitchf[start_frame:end_frame]
        else:
            # Pad if too short
            pad_len = self.segment_size - len(audio)
            audio = np.pad(audio, (0, pad_len), mode='constant')

        text_tokens, text_mask = self._get_text_tokens(
            file_stem, start / self.sr, (start + self.segment_size) / self.sr)
        
        # Augmentation
        if self.augment and self.split == "train":
            audio = self._augment_audio(audio)
        
        # Convert to tensors
        audio = torch.FloatTensor(audio)
        
        # Ensure correct shapes
        if pitch.dim() == 0:
            pitch = pitch.unsqueeze(0)
        if pitchf.dim() == 0:
            pitchf = pitchf.unsqueeze(0)
        
        # Speaker ID (currently single speaker, ID = 0)
        speaker_id = torch.LongTensor([0])
        
        return {
            'audio': audio,
            'content_features': content_features,
            'pitch': pitch,
            'pitchf': pitchf,
            'text_tokens': text_tokens,
            'text_mask': text_mask,
            'speaker_id': speaker_id,
            'filename': file_stem
        }
    
    def _get_text_tokens(self, file_stem: str, start: float = 0.,
                         end: float = float('inf')) -> Tuple[torch.Tensor, torch.Tensor]:
        """Use only file-associated words overlapping the actual audio crop."""
        own_file = self.lyrics_dir / f'{file_stem}.json'
        data = json.loads(own_file.read_text(encoding='utf-8')) if own_file.exists() else self.lyrics_data
        segments = []
        for seg in data.get('segments', []):
            name = seg.get('file', seg.get('filename', data.get('file', data.get('filename'))))
            if not own_file.exists() and (not name or Path(name).stem != file_stem):
                continue  # Ambiguous project-wide text must not condition every training song.
            units = seg.get('words') or [seg]
            for unit in units:
                if 'start' not in unit or 'end' not in unit: continue
                if float(unit['start']) < end and float(unit['end']) > start:
                    segments.append(dict(text=unit.get('text', unit.get('word', '')),
                                         tags=seg.get('tags', [])))
        
        if not segments:
            # No lyrics - return empty
            return torch.zeros(1, dtype=torch.long), torch.ones(1, dtype=torch.bool)
        
        all_text = ' '.join([seg.get('text', '') for seg in segments])
        if getattr(self.config, 'text_tokenizer', 'legacy') == 'char_v1':
            from modules.rvc_v3.text_tokens import encode_char_v1
            ids=encode_char_v1(all_text)
            if not ids: return torch.zeros(1,dtype=torch.long),torch.ones(1,dtype=torch.bool)
            return torch.tensor(ids,dtype=torch.long),torch.zeros(len(ids),dtype=torch.bool)
        
        # Get tags
        all_tags = []
        for seg in segments:
            all_tags.extend(seg.get('tags', []))
        
        # Phonemize
        phonemes = self.phonemizer.phonemize(all_text, preserve_tags=True)
        
        # Add tags at the beginning
        if all_tags:
            tag_tokens = [f"[{tag}]" for tag in set(all_tags)]
            phonemes = tag_tokens + phonemes
        
        # Encode to IDs
        token_ids = self.phonemizer.encode(phonemes)
        if not token_ids:
            return torch.zeros(1, dtype=torch.long), torch.ones(1, dtype=torch.bool)
        
        # Create mask (all valid for now)
        mask = torch.zeros(len(token_ids), dtype=torch.bool)
        
        return torch.LongTensor(token_ids), mask
    
    def _augment_audio(self, audio: np.ndarray) -> np.ndarray:
        """Apply audio augmentation."""
        # Random gain
        if random.random() < 0.5:
            gain = random.uniform(0.8, 1.2)
            audio = audio * gain
        
        # Random noise
        if random.random() < 0.3:
            noise_level = random.uniform(0.0001, 0.001)
            noise = np.random.randn(len(audio)) * noise_level
            audio = audio + noise
        
        # Clip
        audio = np.clip(audio, -1.0, 1.0)
        
        return audio
    
    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate function for DataLoader.
        
        Handles variable-length sequences.
        """
        # Get batch size
        batch_size = len(batch)
        
        # Audio (already same length from segment extraction)
        audio = torch.stack([item['audio'] for item in batch])
        
        # Content features (may vary in length, pad)
        content_features = [item['content_features'] for item in batch]
        max_content_len = max(f.shape[0] for f in content_features)
        feature_dim = content_features[0].shape[1]
        
        padded_content = torch.zeros(batch_size, max_content_len, feature_dim)
        content_lengths = torch.LongTensor([f.shape[0] for f in content_features])
        
        for i, f in enumerate(content_features):
            padded_content[i, :f.shape[0], :] = f
        
        # Pitch (pad to max_content_len)
        pitch = [item['pitch'] for item in batch]
        pitchf = [item['pitchf'] for item in batch]
        
        padded_pitch = torch.zeros(batch_size, max_content_len)
        padded_pitchf = torch.zeros(batch_size, max_content_len)
        
        for i, (p, pf) in enumerate(zip(pitch, pitchf)):
            length = min(len(p), max_content_len)
            padded_pitch[i, :length] = p[:length]
            padded_pitchf[i, :length] = pf[:length]
        
        # Text tokens (pad to max length)
        text_tokens = [item['text_tokens'] for item in batch]
        max_text_len = max(t.shape[0] for t in text_tokens)
        
        padded_text = torch.zeros(batch_size, max_text_len, dtype=torch.long)
        text_mask = torch.ones(batch_size, max_text_len, dtype=torch.bool)
        
        for i, t in enumerate(text_tokens):
            padded_text[i, :t.shape[0]] = t
            text_mask[i, :t.shape[0]] = batch[i]['text_mask'][:t.shape[0]]
        
        # Speaker IDs
        speaker_ids = torch.stack([item['speaker_id'] for item in batch]).squeeze(1)
        
        return {
            'audio': audio,
            'content_features': padded_content,
            'content_lengths': content_lengths,
            'pitch': padded_pitch,
            'pitchf': padded_pitchf,
            'text_tokens': padded_text,
            'text_mask': text_mask,
            'speaker_ids': speaker_ids
        }

