"""
Separation profile definitions for vocal/instrumental separation.

Defines different quality presets for separation with their associated models,
weights, and processing parameters.
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Dict


class SeparationProfile(str, Enum):
    """Quality profiles for audio separation."""
    V1_STANDARD = "v1"
    V2_HIGH_QUALITY = "v2"


@dataclass
class ModelSpec:
    """Specification for a separation model."""
    id: str
    vocal_weight: float = 8.0
    inst_weight: float = 16.0
    kwargs: Optional[Dict] = None


# Model presets for each separation profile
# These models are from the native AudioSeparate project and provide
# the best quality for their respective profiles
MODEL_PRESETS: Dict[SeparationProfile, List[ModelSpec]] = {
    # v1 (Standard) - 2 models (fast, solid quality)
    # Uses a smaller ensemble for faster processing while maintaining good quality
    SeparationProfile.V1_STANDARD: [
        ModelSpec(
            id="vocals_mel_band_roformer.ckpt",
            vocal_weight=8.6,
            inst_weight=16.0
        ),
        ModelSpec(
            id="MDX23C-8KFFT-InstVoc_HQ.ckpt",
            vocal_weight=7.2,
            inst_weight=14.9
        ),
    ],
    
    # v2 (High Quality) - 5 models (slower, best quality)
    # Uses a larger ensemble with the newest high-fidelity models
    # for maximum fidelity, cleaner vocals, and fuller instrumentals
    SeparationProfile.V2_HIGH_QUALITY: [
        ModelSpec(
            id="model_bs_roformer_ep_368_sdr_12.9628.ckpt",
            vocal_weight=8.7,
            inst_weight=16.0
        ),
        ModelSpec(
            id="vocals_mel_band_roformer.ckpt",
            vocal_weight=8.6,
            inst_weight=16.0
        ),
        ModelSpec(
            id="Kim_Vocal_2.onnx",
            vocal_weight=8.5,
            inst_weight=16.0
        ),
        ModelSpec(
            id="melband_roformer_big_beta4.ckpt",
            vocal_weight=8.5,
            inst_weight=16.0
        ),
        ModelSpec(
            id="MDX23C-8KFFT-InstVoc_HQ.ckpt",
            vocal_weight=7.2,
            inst_weight=14.9
        ),
    ],
}


# Default parameters for each profile
PROFILE_DEFAULTS = {
    SeparationProfile.V1_STANDARD: {
        "ensemble_size": 2,
        "residual_fill_pct": 0.40,
        "bleed_guard_multiplier": 1.00,
    },
    SeparationProfile.V2_HIGH_QUALITY: {
        "ensemble_size": 5,
        "residual_fill_pct": 0.40,
        "bleed_guard_multiplier": 1.15,
    },
}


def get_profile_models(profile: SeparationProfile, ensemble_size: Optional[int] = None) -> List[ModelSpec]:
    """
    Get the list of models for a given profile.
    
    Args:
        profile: The separation profile to use
        ensemble_size: Optional override for the number of models to use
        
    Returns:
        List of ModelSpec objects for the profile
    """
    models = MODEL_PRESETS[profile]
    
    if ensemble_size is not None:
        # Limit to requested ensemble size
        models = models[:ensemble_size]
    
    return models


def get_profile_defaults(profile: SeparationProfile) -> Dict:
    """
    Get the default parameters for a given profile.
    
    Args:
        profile: The separation profile
        
    Returns:
        Dictionary of default parameters
    """
    return PROFILE_DEFAULTS[profile].copy()

