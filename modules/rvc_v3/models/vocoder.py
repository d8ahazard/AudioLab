"""
Stereo vocoder support for RVC V3.

Extends BigVGAN/HiFiGAN to support stereo output.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.rvc.lib.models_bigvgan import GeneratorBigVgan
from modules.rvc.lib.models import GeneratorNSF

logger = logging.getLogger(__name__)


class StereoVocoder(nn.Module):
    """
    Stereo vocoder wrapper.
    
    Extends mono vocoder to produce stereo output with optional
    spatial effects and channel-independent processing.
    """
    
    def __init__(
        self,
        base_vocoder_type: str = 'bigvgan',
        stereo_mode: str = 'dual',
        **vocoder_kwargs
    ):
        """
        Initialize stereo vocoder.
        
        Args:
            base_vocoder_type: Base vocoder type ('bigvgan', 'hifigan')
            stereo_mode: How to generate stereo:
                - 'dual': Two separate decoders for L/R
                - 'shared': Shared decoder with stereo head
                - 'mono': Mono output duplicated to stereo
            **vocoder_kwargs: Arguments for base vocoder
        """
        super().__init__()
        
        self.stereo_mode = stereo_mode
        self.base_vocoder_type = base_vocoder_type
        
        if stereo_mode == 'dual':
            # Two separate vocoders for left and right channels
            if base_vocoder_type == 'bigvgan':
                self.vocoder_left = GeneratorBigVgan(**vocoder_kwargs)
                self.vocoder_right = GeneratorBigVgan(**vocoder_kwargs)
            elif base_vocoder_type == 'hifigan':
                self.vocoder_left = GeneratorNSF(**vocoder_kwargs)
                self.vocoder_right = GeneratorNSF(**vocoder_kwargs)
            else:
                raise ValueError(f"Unknown vocoder type: {base_vocoder_type}")
            
            logger.info("StereoVocoder: dual mode with separate L/R decoders")
        
        elif stereo_mode == 'shared':
            # Shared decoder with stereo output head
            if base_vocoder_type == 'bigvgan':
                self.vocoder = GeneratorBigVgan(**vocoder_kwargs)
            elif base_vocoder_type == 'hifigan':
                self.vocoder = GeneratorNSF(**vocoder_kwargs)
            else:
                raise ValueError(f"Unknown vocoder type: {base_vocoder_type}")
            
            # Add stereo conversion layer
            # The vocoder outputs mono, we'll add a layer to convert to stereo
            self.stereo_head = nn.Conv1d(1, 2, kernel_size=7, padding=3)
            
            logger.info("StereoVocoder: shared mode with stereo head")
        
        elif stereo_mode == 'mono':
            # Simple mono vocoder, duplicate output
            if base_vocoder_type == 'bigvgan':
                self.vocoder = GeneratorBigVgan(**vocoder_kwargs)
            elif base_vocoder_type == 'hifigan':
                self.vocoder = GeneratorNSF(**vocoder_kwargs)
            else:
                raise ValueError(f"Unknown vocoder type: {base_vocoder_type}")
            
            logger.info("StereoVocoder: mono mode (duplicate to stereo)")
        
        else:
            raise ValueError(f"Unknown stereo mode: {stereo_mode}")
    
    def forward(
        self,
        x: torch.Tensor,
        f0: Optional[torch.Tensor] = None,
        g: Optional[torch.Tensor] = None,
        stereo_width: float = 1.0
    ) -> torch.Tensor:
        """
        Generate stereo audio.
        
        Args:
            x: Latent features (batch, channels, time)
            f0: F0 for NSF vocoder (batch, time)
            g: Speaker embedding (batch, channels, 1)
            stereo_width: Stereo width parameter (0=mono, 1=full stereo)
        
        Returns:
            Stereo audio (batch, 2, samples) or (batch, samples, 2)
        """
        if self.stereo_mode == 'dual':
            # Generate left and right channels separately
            if self.base_vocoder_type == 'bigvgan':
                audio_left = self.vocoder_left(x, g=g)
                audio_right = self.vocoder_right(x, g=g)
            else:  # hifigan/nsf
                audio_left = self.vocoder_left(x, f0, g=g)
                audio_right = self.vocoder_right(x, f0, g=g)
            
            # Stack to stereo
            audio = torch.stack([audio_left.squeeze(1), audio_right.squeeze(1)], dim=1)
            
            # Apply stereo width
            if stereo_width < 1.0:
                mid = (audio[:, 0:1] + audio[:, 1:2]) / 2
                side = audio - mid
                audio = mid + side * stereo_width
        
        elif self.stereo_mode == 'shared':
            # Generate mono then convert to stereo
            if self.base_vocoder_type == 'bigvgan':
                audio_mono = self.vocoder(x, g=g)
            else:  # hifigan/nsf
                audio_mono = self.vocoder(x, f0, g=g)
            
            # Convert to stereo using stereo head
            audio = self.stereo_head(audio_mono)  # (batch, 2, samples)
            
            # Apply stereo width
            if stereo_width < 1.0:
                mid = audio.mean(dim=1, keepdim=True)
                side = audio - mid
                audio = mid + side * stereo_width
        
        elif self.stereo_mode == 'mono':
            # Generate mono and duplicate
            if self.base_vocoder_type == 'bigvgan':
                audio_mono = self.vocoder(x, g=g)
            else:  # hifigan/nsf
                audio_mono = self.vocoder(x, f0, g=g)
            
            # Duplicate to stereo
            audio_mono = audio_mono.squeeze(1)  # (batch, samples)
            audio = audio_mono.unsqueeze(1).repeat(1, 2, 1)  # (batch, 2, samples)
        
        return audio
    
    def remove_weight_norm(self):
        """Remove weight normalization (for inference)."""
        if self.stereo_mode == 'dual':
            if hasattr(self.vocoder_left, 'remove_weight_norm'):
                self.vocoder_left.remove_weight_norm()
            if hasattr(self.vocoder_right, 'remove_weight_norm'):
                self.vocoder_right.remove_weight_norm()
        else:
            if hasattr(self.vocoder, 'remove_weight_norm'):
                self.vocoder.remove_weight_norm()


class MonoToStereoConverter(nn.Module):
    """
    Convert mono audio to pseudo-stereo using various techniques.
    
    Can be used as post-processing when model only produces mono.
    """
    
    def __init__(self, method: str = 'haas'):
        """
        Initialize converter.
        
        Args:
            method: Stereo conversion method:
                - 'duplicate': Simple duplication
                - 'haas': Haas effect (slight delay)
                - 'eq': EQ-based stereo widening
                - 'chorus': Chorus-like effect
        """
        super().__init__()
        self.method = method
    
    def forward(
        self,
        audio: torch.Tensor,
        width: float = 0.5
    ) -> torch.Tensor:
        """
        Convert mono to stereo.
        
        Args:
            audio: Mono audio (batch, samples) or (batch, 1, samples)
            width: Stereo width (0=mono, 1=full effect)
        
        Returns:
            Stereo audio (batch, 2, samples)
        """
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)  # Add channel dim
        
        if audio.shape[1] == 2:
            # Already stereo
            return audio
        
        # Squeeze channel if mono
        audio = audio.squeeze(1)  # (batch, samples)
        
        if self.method == 'duplicate':
            # Simple duplication
            stereo = audio.unsqueeze(1).repeat(1, 2, 1)
        
        elif self.method == 'haas':
            # Haas effect: slight delay on right channel
            delay_samples = int(width * 10)  # Up to 10 samples delay
            
            left = audio
            right = F.pad(audio, (delay_samples, 0))[:, :audio.shape[1]]
            
            stereo = torch.stack([left, right], dim=1)
        
        elif self.method == 'eq':
            # EQ-based: boost different frequencies on each channel
            # Simple implementation: high-pass left, low-pass right
            
            # Create simple filters (not ideal, but fast)
            kernel_size = 31
            low_pass = torch.hann_window(kernel_size, device=audio.device)
            low_pass = low_pass / low_pass.sum()
            
            # Apply convolution
            left = audio
            right = F.conv1d(
                audio.unsqueeze(1),
                low_pass.view(1, 1, -1),
                padding=kernel_size//2
            ).squeeze(1)
            
            # Mix based on width
            stereo = torch.stack([left, right], dim=1)
            mono = audio.unsqueeze(1).repeat(1, 2, 1)
            stereo = mono * (1 - width) + stereo * width
        
        elif self.method == 'chorus':
            # Chorus-like effect with pitch modulation
            # Simplified version
            left = audio
            
            # Slight pitch shift on right (very simple approximation)
            shift_ratio = 1.0 + width * 0.01
            right = F.interpolate(
                audio.unsqueeze(1),
                scale_factor=shift_ratio,
                mode='linear',
                align_corners=False
            ).squeeze(1)
            
            # Trim/pad to match length
            if right.shape[1] > audio.shape[1]:
                right = right[:, :audio.shape[1]]
            else:
                right = F.pad(right, (0, audio.shape[1] - right.shape[1]))
            
            stereo = torch.stack([left, right], dim=1)
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return stereo

