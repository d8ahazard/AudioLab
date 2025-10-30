"""
Phoneme converter for RVC V3.

Converts text to phoneme sequences for training.
"""

import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Set

logger = logging.getLogger(__name__)


class Phonemizer:
    """
    Convert text to phoneme sequences.
    
    Uses phonemizer library or espeak-ng for text-to-phoneme conversion.
    Preserves style tags as special tokens.
    """
    
    # Special tokens
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    SPACE_TOKEN = "<SP>"
    
    def __init__(
        self,
        backend: str = "espeak",
        language: str = "en-us",
        output_dir: Optional[str] = None
    ):
        """
        Initialize the phonemizer.
        
        Args:
            backend: Phonemizer backend ('espeak', 'festival', 'segments')
            language: Language code (e.g., 'en-us', 'es', 'fr-fr')
            output_dir: Output directory for vocabularies
        """
        self.backend = backend
        self.language = language
        self.output_dir = Path(output_dir) if output_dir else None
        
        # Vocabulary
        self.phoneme_to_id: Dict[str, int] = {}
        self.id_to_phoneme: Dict[int, str] = {}
        self.tag_to_id: Dict[str, int] = {}
        self.id_to_tag: Dict[int, str] = {}
        
        # Initialize with special tokens
        self._init_vocab()
        
        # Check backend availability
        self._check_backend()
    
    def _init_vocab(self):
        """Initialize vocabulary with special tokens."""
        special_tokens = [
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN,
            self.SPACE_TOKEN,
        ]
        
        for token in special_tokens:
            idx = len(self.phoneme_to_id)
            self.phoneme_to_id[token] = idx
            self.id_to_phoneme[idx] = token
    
    def _check_backend(self):
        """Check if phonemizer backend is available."""
        if self.backend == "espeak":
            try:
                # Check if espeak-ng is available
                result = subprocess.run(
                    ["espeak-ng", "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    logger.info(f"espeak-ng available: {result.stdout.split()[2]}")
                    return True
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
            
            # Check if libs are available
            libs_path = Path("libs")
            if (libs_path / "libespeak-ng.dll").exists():
                logger.info("espeak-ng DLL found in libs/")
                return True
            
            logger.warning("espeak-ng not found, will try phonemizer library")
        
        try:
            import phonemizer
            logger.info(f"phonemizer library available (version {phonemizer.__version__})")
            return True
        except ImportError:
            logger.warning("phonemizer library not installed")
        
        return False
    
    def phonemize(self, text: str, preserve_tags: bool = True) -> List[str]:
        """
        Convert text to phonemes.
        
        Args:
            text: Input text
            preserve_tags: Whether to preserve [tag] markers
        
        Returns:
            List of phoneme strings
        """
        # Extract tags if preserve_tags is True
        tags = []
        if preserve_tags:
            tag_pattern = r'\[([^\]]+)\]'
            tags = re.findall(tag_pattern, text)
            # Remove tags from text temporarily
            text_no_tags = re.sub(tag_pattern, '', text)
        else:
            text_no_tags = text
        
        # Clean text
        text_no_tags = text_no_tags.strip()
        
        if not text_no_tags:
            return []
        
        # Phonemize using available backend
        try:
            phonemes = self._phonemize_with_library(text_no_tags)
        except Exception as e:
            logger.warning(f"Library phonemization failed: {e}, trying fallback")
            phonemes = self._phonemize_fallback(text_no_tags)
        
        # Add tags back as special tokens
        if preserve_tags and tags:
            # Insert tag tokens at the beginning
            tag_tokens = [f"[{tag}]" for tag in tags]
            phonemes = tag_tokens + phonemes
        
        return phonemes
    
    def _phonemize_with_library(self, text: str) -> List[str]:
        """Phonemize using phonemizer library."""
        try:
            from phonemizer import phonemize
            from phonemizer.backend import EspeakBackend
            
            # Phonemize
            phoneme_str = phonemize(
                text,
                language=self.language,
                backend=self.backend,
                strip=True,
                preserve_punctuation=False,
                with_stress=False
            )
            
            # Split into individual phonemes
            phonemes = phoneme_str.split()
            
            return phonemes
            
        except ImportError:
            raise ImportError("phonemizer library not installed")
    
    def _phonemize_fallback(self, text: str) -> List[str]:
        """
        Fallback phonemization using character-level or simple rules.
        
        This is a very basic fallback - in production, proper phonemizer should be used.
        """
        logger.warning("Using character-level fallback for phonemization")
        
        # Simple character-level tokenization
        # In a real implementation, you'd use a proper G2P model
        phonemes = []
        for char in text.lower():
            if char.isalpha():
                phonemes.append(char)
            elif char.isspace():
                phonemes.append(self.SPACE_TOKEN)
        
        return phonemes
    
    def build_vocabulary(self, phoneme_sequences: List[List[str]]) -> Dict[str, int]:
        """
        Build vocabulary from phoneme sequences.
        
        Args:
            phoneme_sequences: List of phoneme sequences
        
        Returns:
            Phoneme to ID mapping
        """
        # Collect all unique phonemes and tags
        phonemes = set()
        tags = set()
        
        for seq in phoneme_sequences:
            for token in seq:
                if token.startswith('[') and token.endswith(']'):
                    # Style tag
                    tags.add(token)
                else:
                    # Regular phoneme
                    phonemes.add(token)
        
        # Add phonemes to vocabulary
        for phoneme in sorted(phonemes):
            if phoneme not in self.phoneme_to_id:
                idx = len(self.phoneme_to_id)
                self.phoneme_to_id[phoneme] = idx
                self.id_to_phoneme[idx] = phoneme
        
        # Add tags to separate vocabulary
        for tag in sorted(tags):
            if tag not in self.tag_to_id:
                idx = len(self.tag_to_id)
                self.tag_to_id[tag] = idx
                self.id_to_tag[idx] = tag
        
        logger.info(f"Built vocabulary: {len(self.phoneme_to_id)} phonemes, {len(self.tag_to_id)} tags")
        
        return self.phoneme_to_id
    
    def encode(self, phonemes: List[str]) -> List[int]:
        """
        Encode phonemes to IDs.
        
        Args:
            phonemes: List of phoneme strings
        
        Returns:
            List of phoneme IDs
        """
        ids = []
        unk_id = self.phoneme_to_id.get(self.UNK_TOKEN, 1)
        
        for phoneme in phonemes:
            # Check if it's a tag
            if phoneme.startswith('[') and phoneme.endswith(']'):
                # Tags are encoded separately or as special phonemes
                if phoneme in self.phoneme_to_id:
                    ids.append(self.phoneme_to_id[phoneme])
                else:
                    # Add new tag to vocabulary
                    idx = len(self.phoneme_to_id)
                    self.phoneme_to_id[phoneme] = idx
                    self.id_to_phoneme[idx] = phoneme
                    ids.append(idx)
            else:
                # Regular phoneme
                ids.append(self.phoneme_to_id.get(phoneme, unk_id))
        
        return ids
    
    def decode(self, ids: List[int]) -> List[str]:
        """
        Decode IDs to phonemes.
        
        Args:
            ids: List of phoneme IDs
        
        Returns:
            List of phoneme strings
        """
        return [self.id_to_phoneme.get(i, self.UNK_TOKEN) for i in ids]
    
    def save_vocabulary(self, output_dir: Optional[str] = None):
        """Save vocabulary to JSON files."""
        save_dir = Path(output_dir) if output_dir else self.output_dir
        
        if save_dir is None:
            logger.warning("No output directory specified")
            return
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save phoneme vocabulary
        phoneme_vocab_file = save_dir / "phoneme_vocab.json"
        with open(phoneme_vocab_file, 'w', encoding='utf-8') as f:
            json.dump({
                'phoneme_to_id': self.phoneme_to_id,
                'id_to_phoneme': {int(k): v for k, v in self.id_to_phoneme.items()}
            }, f, indent=2, ensure_ascii=False)
        
        # Save tag vocabulary
        if self.tag_to_id:
            tag_vocab_file = save_dir / "tag_vocab.json"
            with open(tag_vocab_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'tag_to_id': self.tag_to_id,
                    'id_to_tag': {int(k): v for k, v in self.id_to_tag.items()}
                }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved vocabularies to {save_dir}")
    
    def load_vocabulary(self, input_dir: str):
        """Load vocabulary from JSON files."""
        input_path = Path(input_dir)
        
        # Load phoneme vocabulary
        phoneme_vocab_file = input_path / "phoneme_vocab.json"
        if phoneme_vocab_file.exists():
            with open(phoneme_vocab_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.phoneme_to_id = data['phoneme_to_id']
                self.id_to_phoneme = {int(k): v for k, v in data['id_to_phoneme'].items()}
        
        # Load tag vocabulary
        tag_vocab_file = input_path / "tag_vocab.json"
        if tag_vocab_file.exists():
            with open(tag_vocab_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.tag_to_id = data['tag_to_id']
                self.id_to_tag = {int(k): v for k, v in data['id_to_tag'].items()}
        
        logger.info(f"Loaded vocabularies from {input_dir}")
    
    def get_vocab_size(self) -> int:
        """Get the size of the phoneme vocabulary."""
        return len(self.phoneme_to_id)

