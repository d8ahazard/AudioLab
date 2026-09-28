"""
Configuration for RVC V3 training and inference.
"""

import json
import logging
import math
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class RVCV3Config:
    """Configuration for RVC V3 model and training."""
    
    # Model Architecture
    spec_channels: int = 1025  # Linear STFT bins for the posterior encoder
    segment_size: int = 16384  # 32 frames at the default 512-sample hop
    backbone: str = "v2_compatible"
    inter_channels: int = 192
    hidden_channels: int = 192
    filter_channels: int = 768
    n_heads: int = 2
    n_layers: int = 6
    kernel_size: int = 3
    p_dropout: float = 0.0
    
    # Vocoder
    resblock: str = "1"
    resblock_kernel_sizes: List[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: List[List[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )
    upsample_rates: List[int] = field(default_factory=lambda: [8, 8, 2, 2, 2])
    upsample_initial_channel: int = 512
    upsample_kernel_sizes: List[int] = field(default_factory=lambda: [16, 16, 4, 4, 4])
    use_spectral_norm: bool = False
    gin_channels: int = 256
    spk_embed_dim: int = 109
    vocoder_type: str = 'hifigan'  # 'bigvgan' or 'hifigan'
    
    # V3-specific: Content Encoders
    use_dual_encoder: bool = False  # Dual features require separately trained conditioning
    hubert_dim: int = 768
    whisper_model: str = "large-v3"
    whisper_dim: int = 1280
    fusion_method: str = "concat"  # 'concat', 'add', 'learned'
    content_output_dim: Optional[int] = None  # Output dim after fusion (None = no projection)
    
    # V3-specific: Text Encoder
    text_encoder_type: str = "transformer"  # 'transformer' or 'bilstm'
    text_tokenizer: str = "char_v1"
    text_encoder_dim: int = 256
    text_encoder_layers: int = 6
    text_encoder_heads: int = 8
    text_encoder_ff_dim: int = 1024
    text_dropout: float = 0.1
    
    # V3-specific: Cross-Attention
    n_cross_attn_layers: int = 2
    
    # V3-specific: Stereo Support
    stereo_mode: str = "mono"  # 'mono', 'shared', 'dual'
    
    # Audio Settings
    sampling_rate: int = 44100  # Target sample rate
    filter_length: int = 2048
    hop_length: int = 512  # Adjusted for 44.1kHz
    win_length: int = 2048
    n_mel_channels: int = 128
    mel_fmin: float = 0.0
    mel_fmax: float = 22050.0  # Nyquist for 44.1kHz
    
    # Training
    learning_rate: float = 0.0001
    betas: List[float] = field(default_factory=lambda: [0.8, 0.99])
    eps: float = 1e-9
    batch_size: int = 4
    fp16_run: bool = True
    lr_decay: float = 0.999
    init_lr_ratio: float = 1.0
    warmup_epochs: int = 0
    epochs: int = 20000
    total_steps: int = 10000000
    
    # Loss Weights
    c_mel: float = 1.0  # Mel reconstruction loss
    c_kl: float = 0.2  # KL divergence loss
    c_stft: float = 9.0  # Multi-resolution STFT loss
    
    # Logging
    log_interval: int = 200
    save_interval: int = 1000

    # Checkpointing
    checkpoint_interval: int = 25  # Save periodic checkpoint every N epochs
    checkpoint_keep_last: int = 2  # Keep last N periodic checkpoints (prune older)

    # Early stopping (plateau/uptrend)
    early_stop_plateau_patience: int = 20  # Epochs without improvement to trigger stop
    early_stop_uptrend_patience: int = 10  # Consecutive epochs with increasing mel to trigger stop
    early_stop_min_epochs: int = 100  # Minimum epochs before early stop can trigger
    
    # Seed
    seed: int = 1234
    
    # Paths (relative to project dir)
    raw_audio_dir: str = "raw"
    vocals_dir: str = "vocals"
    lyrics_dir: str = "lyrics"
    features_dir: str = "features"
    checkpoints_dir: str = "checkpoints"
    logs_dir: str = "logs"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RVCV3Config':
        """Create from dictionary."""
        # Filter out keys that aren't in the dataclass
        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        # Existing checkpoints use the original custom attention implementation.
        filtered_data.setdefault('backbone', 'legacy')
        filtered_data.setdefault('text_tokenizer', 'legacy' if filtered_data['backbone']=='legacy' else 'char_v1')
        return cls(**filtered_data)

    def validate_contract(self):
        if self.backbone not in {'legacy','v2_compatible'}:
            raise ValueError(f'Unknown V3 backbone: {self.backbone}')
        if self.text_tokenizer not in {'legacy','char_v1'}:
            raise ValueError(f'Unknown V3 text vocabulary: {self.text_tokenizer}')
        if self.segment_size % self.hop_length:
            raise ValueError('V3 training segment_size must contain whole spectrogram frames')
        if math.prod(self.upsample_rates) != self.hop_length:
            raise ValueError('V3 decoder upsampling must equal the spectrogram hop_length')
        if self.spec_channels != self.filter_length // 2 + 1:
            raise ValueError('V3 posterior requires linear spectrogram bins')
        if self.backbone == 'v2_compatible' and (self.use_dual_encoder or self.hubert_dim != 768):
            raise ValueError('The V2-compatible backbone requires single 768-D content features')
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Config saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'RVCV3Config':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        config = cls.from_dict(data)
        logger.info(f"Config loaded from {path}")
        return config
    
    def get_content_feature_dim(self) -> int:
        """Get the dimension of content features after fusion."""
        if not self.use_dual_encoder:
            return self.hubert_dim
        
        if self.content_output_dim is not None:
            return self.content_output_dim
        
        if self.fusion_method == "concat":
            return self.hubert_dim + self.whisper_dim
        elif self.fusion_method == "add":
            return self.hubert_dim
        else:  # learned
            return self.content_output_dim or 768


def load_config(path: str) -> RVCV3Config:
    """Load configuration from JSON file."""
    return RVCV3Config.load(path)


def save_config(config: RVCV3Config, path: str):
    """Save configuration to JSON file."""
    config.save(path)


# Default configurations for different sample rates
def get_default_config(sample_rate: int = 44100) -> RVCV3Config:
    """Get default configuration for a sample rate."""
    config = RVCV3Config()
    
    if sample_rate == 32000:
        config.upsample_rates = [10, 4, 2, 2, 2]
        config.upsample_kernel_sizes = [16, 8, 4, 4, 4]
        config.sampling_rate = 32000
        config.hop_length = 320
        config.mel_fmax = 16000.0
        config.segment_size = 12800
    elif sample_rate == 40000:
        config.upsample_rates = [10, 10, 2, 2]
        config.upsample_kernel_sizes = [16, 16, 4, 4]
        config.sampling_rate = 40000
        config.hop_length = 400
        config.mel_fmax = 20000.0
        config.segment_size = 16000
    elif sample_rate == 48000:
        config.upsample_rates = [12, 10, 2, 2]
        config.upsample_kernel_sizes = [24, 20, 4, 4]
        config.sampling_rate = 48000
        config.hop_length = 480
        config.mel_fmax = 24000.0
        config.segment_size = 19200
    else:  # 44100
        config.sampling_rate = 44100
        config.hop_length = 512
        config.mel_fmax = 22050.0
        config.segment_size = 16384
    
    return config

