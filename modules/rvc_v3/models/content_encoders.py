"""
Dual content encoder system for RVC V3.

Combines HuBERT and Whisper encoders for robust content features.
"""

import logging
import os
from typing import Optional, Tuple

def _patch_tensorboard_no_tf():
    """
    Fix broken TensorBoard installs that try to import TensorFlow.

    `fairseq` imports `torch.utils.tensorboard.SummaryWriter`, which imports
    `tensorboard.compat.tf`. On this machine the installed `tensorboard` package
    is missing `tensorboard.compat.notf`, causing it to fall back to importing
    TensorFlow and crashing due to incompatible `ml_dtypes`.

    Providing a stub `tensorboard.compat.notf` makes TensorBoard use its
    `tensorflow_stub` backend and avoids importing TensorFlow entirely.
    """
    try:
        import sys
        import types

        sys.modules.setdefault("tensorboard.compat.notf", types.ModuleType("tensorboard.compat.notf"))
    except Exception:
        # If anything goes wrong, don't block import; worst case we hit the original error.
        pass


_patch_tensorboard_no_tf()

import torch
import torch.nn as nn
import torch.nn.functional as F
import fairseq
import numpy as np

logger = logging.getLogger(__name__)


class HuBERTEncoder:
    """
    Wrapper for HuBERT content encoder.
    
    Extracts speaker-independent acoustic features (768-D).
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        is_half: bool = False
    ):
        """
        Initialize HuBERT encoder.
        
        Args:
            model_path: Path to hubert_base.pt
            device: Device to run on
            is_half: Whether to use FP16
        """
        self.device = device
        self.is_half = is_half
        self.model = None
        self.model_path = model_path
        
        logger.info(f"Initializing HuBERT encoder from {model_path}")
        self._load_model()
    
    def _load_model(self):
        """Load HuBERT model."""
        if not os.path.exists(self.model_path):
            logger.warning(f"HuBERT model not found at {self.model_path}. Attempting automatic download...")
            try:
                from handlers.download import download_hubert_model
                self.model_path = download_hubert_model()
                logger.info(f"HuBERT model downloaded successfully to {self.model_path}")
            except Exception as e:
                raise FileNotFoundError(
                    f"HuBERT model not found: {self.model_path}. "
                    f"Automatic download failed: {str(e)}\n"
                    "Please manually download from https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt"
                )
        
        models, saved_cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [self.model_path],
            suffix="",
        )
        
        self.model = models[0]
        self.model = self.model.to(self.device)
        self.input_normalize = getattr(task, 'normalize', False)
        
        if self.is_half and self.device not in ["mps", "cpu"]:
            self.model = self.model.half()
        
        self.model.eval()
        logger.info("HuBERT model loaded successfully")
    
    @torch.no_grad()
    def extract_features(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract content features from audio.
        
        Args:
            audio: Audio tensor (1, T) at 16kHz
        
        Returns:
            Feature tensor (1, T', 768) where T' = T // 320
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # Ensure correct dtype
        if self.is_half and self.device not in ["mps", "cpu"]:
            audio = audio.half()
        else:
            audio = audio.float()
        
        audio = audio.to(self.device)
        
        if getattr(self, 'input_normalize', False):
            audio = torch.nn.functional.layer_norm(audio, audio.shape)
        
        # Create padding mask
        padding_mask = torch.zeros_like(audio, dtype=torch.bool, device=self.device)
        
        # Extract features
        inputs = {
            "source": audio,
            "padding_mask": padding_mask,
            "output_layer": 12,  # Use layer 12 for v2 (768-D)
        }
        
        logits = self.model.extract_features(**inputs)
        features = logits[0]  # (1, T', 768)
        
        return features


class WhisperEncoder:
    """
    Wrapper for Whisper content encoder.
    
    Extracts robust phonetic features (1024-D or 1280-D depending on model).
    """
    
    def __init__(
        self,
        model_name: str = "large-v3",
        device: str = "cuda",
        is_half: bool = False
    ):
        """
        Initialize Whisper encoder.
        
        Args:
            model_name: Whisper model name (base, small, medium, large, large-v3)
            device: Device to run on
            is_half: Whether to use FP16
        """
        self.device = device
        self.is_half = is_half
        self.model_name = model_name
        self.model = None
        
        logger.info(f"Initializing Whisper encoder: {model_name}")
        self._load_model()
    
    def _load_model(self):
        """Load Whisper model."""
        try:
            import whisper
            self.model = whisper.load_model(self.model_name, device=self.device)
            
            if self.is_half and self.device not in ["mps", "cpu"]:
                self.model = self.model.half()
            
            self.model.eval()
            logger.info(f"Whisper {self.model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper: {e}")
            raise
    
    @torch.no_grad()
    def extract_features(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract content features from audio.
        
        Args:
            audio: Audio tensor (1, T) at 16kHz
        
        Returns:
            Feature tensor (1, T', D) where D=1024 or 1280
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # Ensure correct dtype
        if self.is_half and self.device not in ["mps", "cpu"]:
            audio = audio.half()
        else:
            audio = audio.float()
        
        audio = audio.to(self.device)
        
        # Pad or trim to 30 seconds (whisper's expected input length)
        # For longer audio, we'll process in chunks
        max_length = 16000 * 30  # 30 seconds at 16kHz
        
        if audio.shape[1] > max_length:
            # Process in overlapping chunks and concatenate
            hop_length = max_length // 2
            chunks = []
            
            for i in range(0, audio.shape[1], hop_length):
                chunk = audio[:, i:i+max_length]
                if chunk.shape[1] < 16000:  # Skip very small chunks
                    continue
                
                # Pad last chunk if needed
                if chunk.shape[1] < max_length:
                    chunk = F.pad(chunk, (0, max_length - chunk.shape[1]))
                
                # Extract features for chunk
                chunk_features = self._extract_chunk_features(chunk)
                chunks.append(chunk_features)
            
            # Concatenate chunks
            features = torch.cat(chunks, dim=1)
        else:
            # Pad to expected length
            if audio.shape[1] < max_length:
                audio = F.pad(audio, (0, max_length - audio.shape[1]))
            
            features = self._extract_chunk_features(audio)
        
        return features
    
    def _extract_chunk_features(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract features from a single audio chunk."""
        # Convert to mel spectrogram (Whisper preprocessing)
        mel = self.model.encoder(audio)
        
        # mel shape: (batch, n_mels, time) -> need (batch, time, features)
        if mel.dim() == 3:
            mel = mel.transpose(1, 2)  # (batch, time, features)
        
        return mel


class DualContentEncoder(nn.Module):
    """
    Dual content encoder combining HuBERT and Whisper.
    
    Fuses features from both encoders for robust content representation.
    """
    
    def __init__(
        self,
        hubert_path: str,
        whisper_model: str = "large-v3",
        fusion_method: str = "concat",
        output_dim: Optional[int] = None,
        device: str = "cuda",
        is_half: bool = False
    ):
        """
        Initialize dual content encoder.
        
        Args:
            hubert_path: Path to HuBERT model
            whisper_model: Whisper model name
            fusion_method: How to fuse features ('concat', 'add', 'learned')
            output_dim: Output dimension (for PCA/learned projection)
            device: Device to run on
            is_half: Whether to use FP16
        """
        super().__init__()
        
        self.device = device
        self.is_half = is_half
        self.fusion_method = fusion_method
        self.output_dim = output_dim
        
        # Initialize encoders
        self.hubert_encoder = HuBERTEncoder(hubert_path, device, is_half)
        self.whisper_encoder = WhisperEncoder(whisper_model, device, is_half)
        
        # Determine feature dimensions
        self.hubert_dim = 768  # HuBERT base output
        self.whisper_dim = 1280 if "large" in whisper_model else 1024
        
        # Fusion layer
        if fusion_method == "concat":
            self.fused_dim = self.hubert_dim + self.whisper_dim
            if output_dim is not None:
                # Add projection layer
                self.projection = nn.Linear(self.fused_dim, output_dim)
                self.fused_dim = output_dim
            else:
                self.projection = None
        
        elif fusion_method == "add":
            # Need to project to same dimension first
            assert self.hubert_dim == self.whisper_dim, \
                "For 'add' fusion, dimensions must match"
            self.fused_dim = self.hubert_dim
            self.projection = None
        
        elif fusion_method == "learned":
            # Learned fusion with attention or weighted sum
            self.fusion_layer = nn.Linear(self.hubert_dim + self.whisper_dim, output_dim or 768)
            self.fused_dim = output_dim or 768
            self.projection = None
        
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        logger.info(
            f"Dual content encoder initialized: "
            f"fusion={fusion_method}, output_dim={self.fused_dim}"
        )
    
    @torch.no_grad()
    def extract_features(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract fused content features.
        
        Args:
            audio: Audio tensor (1, T) at 16kHz
        
        Returns:
            Fused feature tensor (1, T', D)
        """
        # Extract features from both encoders
        hubert_features = self.hubert_encoder.extract_features(audio)  # (1, T', 768)
        whisper_features = self.whisper_encoder.extract_features(audio)  # (1, T'', 1024/1280)
        
        # Align temporal dimensions via interpolation
        # HuBERT: 16000Hz -> ~50Hz (320 hop)
        # Whisper: varies but generally higher rate
        
        target_length = hubert_features.shape[1]
        
        if whisper_features.shape[1] != target_length:
            whisper_features = F.interpolate(
                whisper_features.transpose(1, 2),
                size=target_length,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        
        # Fuse features
        if self.fusion_method == "concat":
            fused = torch.cat([hubert_features, whisper_features], dim=-1)
            if self.projection is not None:
                fused = self.projection(fused)
        
        elif self.fusion_method == "add":
            fused = hubert_features + whisper_features
        
        elif self.fusion_method == "learned":
            combined = torch.cat([hubert_features, whisper_features], dim=-1)
            fused = self.fusion_layer(combined)
        
        return fused
    
    def get_output_dim(self) -> int:
        """Get the output feature dimension."""
        return self.fused_dim

