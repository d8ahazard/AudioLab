"""
Feature extraction for RVC V3 training.

Extracts content features (HuBERT + Whisper) and pitch (F0) from audio.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import torch
import librosa
from tqdm import tqdm

from modules.rvc_v3.models.content_encoders import DualContentEncoder, HuBERTEncoder
from handlers.config import model_path

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extract and cache features for training.
    """
    
    def __init__(
        self,
        config,
        device: str = "cuda",
        use_dual_encoder: bool = True
    ):
        """
        Initialize feature extractor.
        
        Args:
            config: RVCV3Config
            device: Device to run on
            use_dual_encoder: Whether to use dual encoder (HuBERT + Whisper)
        """
        self.config = config
        self.device = device
        self.use_dual_encoder = use_dual_encoder
        
        # Initialize content encoder
        hubert_path = os.path.join(model_path, "rvc", "hubert_base.pt")
        
        if use_dual_encoder:
            self.content_encoder = DualContentEncoder(
                hubert_path=hubert_path,
                whisper_model=config.whisper_model,
                fusion_method=config.fusion_method,
                output_dim=config.content_output_dim,
                device=device,
                is_half=False
            )
            logger.info("Using dual content encoder (HuBERT + Whisper)")
        else:
            self.content_encoder = HuBERTEncoder(
                model_path=hubert_path,
                device=device,
                is_half=False
            )
            logger.info("Using HuBERT content encoder only")
        
        # Initialize pitch extractor
        self._init_pitch_extractor()
    
    def _init_pitch_extractor(self):
        """Initialize pitch extraction model."""
        from modules.rvc.infer.lib.rmvpe import RMVPE
        
        rmvpe_path = os.path.join(model_path, "rvc", "rmvpe.pt")
        self.pitch_extractor = RMVPE(rmvpe_path, is_half=False, device=self.device)
        
        logger.info("RMVPE pitch extractor initialized")
    
    def extract_from_audio(
        self,
        audio_path: str,
        target_sr: int = 16000
    ) -> dict:
        """
        Extract features from a single audio file.
        
        Args:
            audio_path: Path to audio file
            target_sr: Target sample rate for content encoder (usually 16kHz)
        
        Returns:
            Dictionary with:
            - content: Content features (T, feature_dim)
            - pitch: Coarse pitch (T,)
            - pitchf: Fine pitch (T,)
        """
        # Load audio at target sample rate for encoder
        audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0)  # (1, T)
        
        # Extract content features
        with torch.no_grad():
            if self.use_dual_encoder:
                content_features = self.content_encoder.extract_features(audio_tensor)
            else:
                content_features = self.content_encoder.extract_features(audio_tensor)
        
        # Squeeze batch dimension
        content_features = content_features.squeeze(0).cpu()  # (T', feature_dim)
        
        # Extract pitch
        f0 = self.pitch_extractor.infer_from_audio(audio, thred=0.03)  # Raw F0
        
        # Convert to coarse pitch
        f0_coarse = self._coarse_f0(f0)
        
        # Align pitch to content features
        # HuBERT outputs ~50Hz (320 hop at 16kHz)
        # Need to match content feature length
        target_len = content_features.shape[0]
        
        if len(f0) != target_len:
            # Resample F0 to match
            f0 = self._resample_f0(f0, target_len)
            f0_coarse = self._resample_f0(f0_coarse, target_len)
        
        return {
            'content': content_features,
            'pitch': torch.FloatTensor(f0_coarse),
            'pitchf': torch.FloatTensor(f0)
        }
    
    def _coarse_f0(self, f0: np.ndarray) -> np.ndarray:
        """Convert F0 to coarse pitch bins."""
        f0_bin = 256
        f0_max = 1100.0
        f0_min = 50.0
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)
        
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (
            f0_bin - 2
        ) / (f0_mel_max - f0_mel_min) + 1
        
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
        f0_coarse = np.rint(f0_mel).astype(int)
        
        return f0_coarse
    
    def _resample_f0(self, f0: np.ndarray, target_len: int) -> np.ndarray:
        """Resample F0 to target length."""
        if len(f0) == target_len:
            return f0
        
        # Simple linear interpolation
        from scipy import interpolate
        
        x_old = np.linspace(0, 1, len(f0))
        x_new = np.linspace(0, 1, target_len)
        
        f_interp = interpolate.interp1d(x_old, f0, kind='linear', fill_value='extrapolate')
        f0_resampled = f_interp(x_new)
        
        return f0_resampled
    
    def extract_project(
        self,
        project_dir: str,
        callback: Optional[Callable] = None
    ):
        """
        Extract features for all audio files in a project.
        
        Args:
            project_dir: Path to project directory
            callback: Optional callback for progress updates
        """
        project_path = Path(project_dir)
        vocals_dir = project_path / self.config.vocals_dir
        features_dir = project_path / self.config.features_dir
        features_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all vocal files
        audio_files = list(vocals_dir.glob("*.wav"))
        
        logger.info(f"Extracting features for {len(audio_files)} files")
        
        for i, audio_file in enumerate(tqdm(audio_files, desc="Extracting features")):
            # Check if features already extracted
            features_file = features_dir / f"{audio_file.stem}.pt"
            
            if features_file.exists():
                logger.debug(f"Skipping {audio_file.name} (already extracted)")
                continue
            
            try:
                # Extract features
                features = self.extract_from_audio(str(audio_file))
                
                # Save features
                torch.save(features, features_file)
                
                if callback:
                    progress = (i + 1) / len(audio_files)
                    callback(progress, f"Extracted {audio_file.name}", len(audio_files))
            
            except Exception as e:
                logger.error(f"Failed to extract features from {audio_file}: {e}")
                continue
        
        logger.info(f"Feature extraction complete: {len(audio_files)} files")

