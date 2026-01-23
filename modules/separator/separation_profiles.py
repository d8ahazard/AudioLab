"""
Separation profile definitions for vocal/instrumental separation.

Defines different quality presets for separation with their associated models,
SDR performance metrics, and auto-calculated weights.

## Auto-Weighting System

Model weights are automatically calculated from published SDR (Signal-to-Distortion Ratio)
performance metrics using exponential scaling. This ensures:

1. Better models automatically get more weight in the ensemble
2. Adding new models only requires their SDR values - no manual weight tuning
3. Weights scale appropriately regardless of ensemble size
4. Vocal and instrumental weights are calculated independently, emphasizing each model's strengths

The algorithm uses temperature-controlled exponential scaling (similar to softmax):
- Higher SDR → exponentially higher weight
- Temperature parameter controls how much we emphasize differences
- Weights normalize to sum(models) * 8.0 for stable blending

Example: If BS Roformer has 17.0 dB inst_sdr and Big Beta 4 has 16.0 dB,
BS Roformer will get significantly more weight in the instrumental blend.

## Usage Example

To view calculated weights for debugging:
```python
from modules.separator.separation_profiles import SeparationProfile, print_model_weights
print_model_weights(SeparationProfile.V2_HIGH_QUALITY)
```

Output:
```
=== V2 Profile Weights ===
Model                                              Vocal SDR    Vocal Weight   Inst SDR     Inst Weight
---------------------------------------------------------------------------------------------------------
model_bs_roformer_ep_368_sdr_12.9628.ckpt          12.97        12.209         17.00        20.804
melband_roformer_big_beta4.ckpt                    12.90        10.614         16.00        2.815
MDX23C-8KFFT-InstVoc_HQ.ckpt                       11.80        1.176          15.00        0.381
---------------------------------------------------------------------------------------------------------
TOTALS                                                          24.000                      24.000
```
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
    """Specification for a separation model.
    
    SDR (Signal-to-Distortion Ratio) values should be provided from the model's
    published performance metrics. Weights will be auto-calculated from SDR if not provided.
    """
    id: str
    vocal_sdr: Optional[float] = None  # Vocal SDR in dB (e.g., 12.97)
    inst_sdr: Optional[float] = None   # Instrumental SDR in dB (e.g., 17.0)
    vocal_weight: Optional[float] = None  # Auto-calculated if None
    inst_weight: Optional[float] = None   # Auto-calculated if None
    kwargs: Optional[Dict] = None


# Model presets for each separation profile
# These models are from the native AudioSeparate project and provide
# the best quality for their respective profiles
MODEL_PRESETS: Dict[SeparationProfile, List[ModelSpec]] = {
    # v1 (Standard) - 2 models (fast, solid quality)
    # Uses a smaller ensemble for faster processing while maintaining good quality
    # Weights are auto-calculated from SDR performance metrics
    SeparationProfile.V1_STANDARD: [
        ModelSpec(
            id="vocals_mel_band_roformer.ckpt",
            vocal_sdr=12.3,  # Solid vocal separation
            inst_sdr=15.5    # Good instrumental preservation
        ),
        ModelSpec(
            id="MDX23C-8KFFT-InstVoc_HQ.ckpt",
            vocal_sdr=11.8,  # High-quality MDX architecture
            inst_sdr=15.0    # Strong instrumental performance
        ),
    ],
    
    # v2 (High Quality) - 3 models (optimized for speed and quality)
    # Uses the top-performing models from 2025:
    # - BS Roformer (ViperX): Best overall separation (12.97 dB vocal SDR, 17.0 dB instrumental SDR)
    # - Mel-Band Roformer Big Beta 4 (Unwa): Superior vocal clarity and harmonic handling (12.9 dB vocal SDR)
    # - MDX23C: Different architecture for ensemble diversity, reduces phase artifacts
    # Weights are auto-calculated from SDR performance metrics using exponential scaling
    SeparationProfile.V2_HIGH_QUALITY: [
        ModelSpec(
            id="model_bs_roformer_ep_368_sdr_12.9628.ckpt",
            vocal_sdr=12.97,  # Best overall, published metric
            inst_sdr=17.0     # Exceptional instrumental preservation
        ),
        ModelSpec(
            id="melband_roformer_big_beta4.ckpt",
            vocal_sdr=12.9,   # Top-tier vocal clarity
            inst_sdr=16.0     # Excellent instrumental quality
        ),
        ModelSpec(
            id="MDX23C-8KFFT-InstVoc_HQ.ckpt",
            vocal_sdr=11.8,   # Strong performance, different architecture
            inst_sdr=15.0     # Solid instrumental separation
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
        "ensemble_size": 3,
        "residual_fill_pct": 0.40,
        "bleed_guard_multiplier": 1.15,
    },
}


def auto_calculate_weights(models: List[ModelSpec], temperature: float = 0.5) -> List[ModelSpec]:
    """
    Automatically calculate ensemble weights based on SDR performance metrics.
    
    Uses exponential scaling (similar to softmax) to emphasize better-performing models
    while still giving weight to all models for ensemble diversity.
    
    Args:
        models: List of ModelSpec objects with vocal_sdr and inst_sdr values
        temperature: Controls emphasis on differences (0.1=strong emphasis, 1.0=mild emphasis)
                     Lower values make better models dominate more
    
    Returns:
        List of ModelSpec objects with calculated vocal_weight and inst_weight
    
    Algorithm:
        1. Extract SDR values for vocals and instrumentals separately
        2. Apply exponential scaling: exp(SDR / temperature)
        3. Normalize to sum to N * 8.0 (where N = number of models)
        4. This ensures weights are proportional to performance while maintaining
           reasonable absolute values for blending
    """
    import math
    
    if not models:
        return models
    
    # Extract SDR values
    vocal_sdrs = [m.vocal_sdr for m in models if m.vocal_sdr is not None]
    inst_sdrs = [m.inst_sdr for m in models if m.inst_sdr is not None]
    
    if not vocal_sdrs or not inst_sdrs:
        # If no SDR data, use default weights
        return models
    
    # Calculate exponential-scaled weights (softmax-like)
    def calc_weights(sdrs: List[float], temp: float) -> List[float]:
        # Apply temperature scaling and exponential
        exp_vals = [math.exp(sdr / temp) for sdr in sdrs]
        total = sum(exp_vals)
        # Normalize to sum to len(sdrs) * 8.0 (reasonable blending range)
        target_sum = len(sdrs) * 8.0
        return [(exp_val / total) * target_sum for exp_val in exp_vals]
    
    vocal_weights = calc_weights(vocal_sdrs, temperature)
    inst_weights = calc_weights(inst_sdrs, temperature)
    
    # Create new ModelSpec list with calculated weights
    updated_models = []
    for i, model in enumerate(models):
        updated_models.append(ModelSpec(
            id=model.id,
            vocal_sdr=model.vocal_sdr,
            inst_sdr=model.inst_sdr,
            vocal_weight=model.vocal_weight if model.vocal_weight is not None else vocal_weights[i],
            inst_weight=model.inst_weight if model.inst_weight is not None else inst_weights[i],
            kwargs=model.kwargs
        ))
    
    return updated_models


def get_profile_models(profile: SeparationProfile, ensemble_size: Optional[int] = None) -> List[ModelSpec]:
    """
    Get the list of models for a given profile with auto-calculated weights.
    
    Args:
        profile: The separation profile to use
        ensemble_size: Optional override for the number of models to use
        
    Returns:
        List of ModelSpec objects for the profile with calculated weights
    """
    models = MODEL_PRESETS[profile]
    
    if ensemble_size is not None:
        # Limit to requested ensemble size
        models = models[:ensemble_size]
    
    # Auto-calculate weights based on SDR if not manually specified
    models = auto_calculate_weights(models)
    
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


def print_model_weights(profile: SeparationProfile) -> None:
    """
    Debug utility: Print the auto-calculated weights for a profile.
    
    Useful for verifying that the auto-weighting system is working correctly
    and understanding how much emphasis each model gets.
    
    Args:
        profile: The separation profile to analyze
    """
    models = get_profile_models(profile)
    print(f"\n=== {profile.value.upper()} Profile Weights ===")
    print(f"{'Model':<50} {'Vocal SDR':<12} {'Vocal Weight':<14} {'Inst SDR':<12} {'Inst Weight':<14}")
    print("-" * 105)
    
    for model in models:
        model_name = model.id.split('/')[-1][:45]  # Truncate long names
        print(f"{model_name:<50} {model.vocal_sdr:<12.2f} {model.vocal_weight:<14.3f} "
              f"{model.inst_sdr:<12.2f} {model.inst_weight:<14.3f}")
    
    total_v = sum(m.vocal_weight for m in models)
    total_i = sum(m.inst_weight for m in models)
    print("-" * 105)
    print(f"{'TOTALS':<50} {'':<12} {total_v:<14.3f} {'':<12} {total_i:<14.3f}")
    print()

