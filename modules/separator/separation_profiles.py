"""Versioned separation recipes.

V1-V3 retain historical SDR-derived weights for compatibility; those numbers
are not a common benchmark and must not be used to rank new models. V4 is an
experimental three-model candidate with equal weights and no residual fill.
Promotion requires local listening evidence (see docs/separation-v4.md).
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Dict


class SeparationProfile(str, Enum):
    """Quality profiles for audio separation."""
    V1_STANDARD = "v1"
    V2_HIGH_QUALITY = "v2"
    V3_MAXIMUM = "v3"
    V4_EXPERIMENTAL = "v4"
    V4_CLEAN_INSTRUMENTAL = "v4"  # Compatibility alias for existing callers.
    HYBRID_CLEANED = "hybrid_cleaned"


class SeparationPreset(str, Enum):
    """Use-case specific presets for optimized separation."""
    KARAOKE = "karaoke"           # Optimized for clean instrumental with no vocal bleed
    REMIX = "remix"               # Balanced for DJ/remix use
    PODCAST = "podcast"           # Optimized for voice isolation
    ACAPPELLA = "acappella"       # Optimized for clean vocal extraction
    INSTRUMENTAL = "instrumental"  # Optimized for clean instrumental


@dataclass
class ModelSpec:
    """Specification for a separation model.
    
    SDR (Signal-to-Distortion Ratio) values should be provided from the model's
    published performance metrics. Weights will be auto-calculated from SDR if not provided.
    
    Attributes:
        id: Model filename or identifier
        vocal_sdr: Published vocal SDR in dB (higher = better vocal separation)
        inst_sdr: Published instrumental SDR in dB (higher = better instrumental preservation)
        vocal_weight: Override weight for vocal blending (auto-calculated if None)
        inst_weight: Override weight for instrumental blending (auto-calculated if None)
        description: Human-readable description of the model's strengths
        kwargs: Additional model-specific parameters
    """
    id: str
    vocal_sdr: Optional[float] = None
    inst_sdr: Optional[float] = None
    vocal_weight: Optional[float] = None
    inst_weight: Optional[float] = None
    description: Optional[str] = None
    kwargs: Optional[Dict] = None


# =============================================================================
# Model Definitions
# =============================================================================
# These models are sourced from the audio-separator project and represent
# the current state-of-the-art in music source separation.

# Core ensemble models
MODELS = {
    "resurrection_vocals": ModelSpec(
        id="bs_roformer_vocals_resurrection_unwa.ckpt", vocal_weight=1.0, inst_weight=1.0,
        description="V4 experimental: Unwa Resurrection vocals"),
    "beta7": ModelSpec(
        id="melband_roformer_big_beta7.ckpt", vocal_weight=1.0, inst_weight=1.0,
        description="V4 experimental: author-pinned Big Beta 7"),
    "becruily_inst": ModelSpec(
        id="mel_band_roformer_instrumental_becruily.ckpt", vocal_weight=1.0, inst_weight=1.0,
        description="V4 experimental: dedicated instrumental estimate"),
    # BS-RoFormer variants (Band-Split RoFormer) - Best overall quality
    "bs_roformer_ep368": ModelSpec(
        id="model_bs_roformer_ep_368_sdr_12.9628.ckpt",
        vocal_sdr=12.97,
        inst_sdr=17.0,
        description="Best overall separation, excellent for both vocals and instrumentals"
    ),
    
    # Mel-Band RoFormer variants
    "melband_big_beta4": ModelSpec(
        id="melband_roformer_big_beta4.ckpt",
        vocal_sdr=12.9,
        inst_sdr=16.0,
        description="Superior vocal clarity and harmonic handling"
    ),
    "melband_vocals": ModelSpec(
        id="vocals_mel_band_roformer.ckpt",
        vocal_sdr=12.3,
        inst_sdr=15.5,
        description="Balanced performance, good all-rounder"
    ),
    "melband_karaoke": ModelSpec(
        id="mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
        vocal_sdr=10.2,
        inst_sdr=16.8,
        description="Optimized for karaoke - minimal vocal bleed in instrumental"
    ),
    
    # MDX23C variants
    "mdx23c_instvoc_hq": ModelSpec(
        id="MDX23C-8KFFT-InstVoc_HQ.ckpt",
        vocal_sdr=11.8,
        inst_sdr=15.0,
        description="Different architecture for ensemble diversity, reduces phase artifacts"
    ),
    
    # Specialized models
    "kim_vocal_1": ModelSpec(
        id="Kim_Vocal_1.onnx",
        vocal_sdr=11.5,
        inst_sdr=14.5,
        description="Fast ONNX model for quick processing"
    ),
    "kim_vocal_2": ModelSpec(
        id="Kim_Vocal_2.onnx",
        vocal_sdr=11.6,
        inst_sdr=14.8,
        description="Improved Kim vocal model"
    ),
    "uvr_voc_ft": ModelSpec(
        id="UVR-MDX-NET-Voc_FT.onnx",
        vocal_sdr=11.2,
        inst_sdr=14.2,
        description="UVR fine-tuned vocal model"
    ),
}


# =============================================================================
# Profile Presets
# =============================================================================

MODEL_PRESETS: Dict[SeparationProfile, List[ModelSpec]] = {
    # V1 (Standard) - 2 models (fast, solid quality)
    # Best for quick processing where speed matters more than maximum quality
    SeparationProfile.V1_STANDARD: [
        MODELS["melband_vocals"],
        MODELS["mdx23c_instvoc_hq"],
    ],
    
    # V2 (High Quality) - 3 models (balanced speed and quality)
    # Default profile for most use cases
    SeparationProfile.V2_HIGH_QUALITY: [
        MODELS["bs_roformer_ep368"],
        MODELS["melband_big_beta4"],
        MODELS["mdx23c_instvoc_hq"],
    ],
    
    # V3 (Maximum Quality) - 5 models (best possible quality)
    # For professional use where quality is paramount
    SeparationProfile.V3_MAXIMUM: [
        MODELS["bs_roformer_ep368"],
        MODELS["melband_big_beta4"],
        MODELS["melband_vocals"],
        MODELS["mdx23c_instvoc_hq"],
        MODELS["kim_vocal_2"],
    ],

    # Unpromoted candidate: equal weights until local listening evidence selects a recipe.
    SeparationProfile.V4_CLEAN_INSTRUMENTAL: [
        MODELS["resurrection_vocals"],
        MODELS["beta7"],
        MODELS["becruily_inst"],
    ],
}


# =============================================================================
# Use-Case Presets
# =============================================================================

PRESET_CONFIGS: Dict[SeparationPreset, Dict] = {
    # Karaoke: Clean instrumental with minimal vocal bleed
    SeparationPreset.KARAOKE: {
        "models": [
            MODELS["melband_karaoke"],
            MODELS["bs_roformer_ep368"],
            MODELS["mdx23c_instvoc_hq"],
        ],
        "ensemble_size": 3,
        "residual_fill_pct": 0.50,  # Higher fill for cleaner instrumental
        "bleed_guard_multiplier": 1.30,  # Aggressive bleed prevention
        "description": "Optimized for karaoke - clean instrumentals with minimal vocal artifacts"
    },
    
    # Remix: Balanced for DJ/production use
    SeparationPreset.REMIX: {
        "models": [
            MODELS["bs_roformer_ep368"],
            MODELS["melband_big_beta4"],
        ],
        "ensemble_size": 2,
        "residual_fill_pct": 0.35,
        "bleed_guard_multiplier": 1.10,
        "description": "Balanced separation for remixing and DJ use"
    },
    
    # Podcast: Optimized for voice isolation from background
    SeparationPreset.PODCAST: {
        "models": [
            MODELS["bs_roformer_ep368"],
            MODELS["melband_big_beta4"],
            MODELS["kim_vocal_2"],
        ],
        "ensemble_size": 3,
        "residual_fill_pct": 0.25,  # Less fill to preserve voice detail
        "bleed_guard_multiplier": 1.00,
        "description": "Optimized for isolating speech from background noise/music"
    },
    
    # Acappella: Cleanest possible vocal extraction
    SeparationPreset.ACAPPELLA: {
        "models": [
            MODELS["bs_roformer_ep368"],
            MODELS["melband_big_beta4"],
            MODELS["melband_vocals"],
            MODELS["uvr_voc_ft"],
        ],
        "ensemble_size": 4,
        "residual_fill_pct": 0.20,  # Minimal fill for pure vocals
        "bleed_guard_multiplier": 0.90,  # Allow some instrumental to avoid vocal artifacts
        "description": "Maximum vocal clarity for acappella extraction"
    },
    
    # Instrumental: Cleanest possible instrumental
    SeparationPreset.INSTRUMENTAL: {
        "models": [
            MODELS["melband_karaoke"],
            MODELS["bs_roformer_ep368"],
            MODELS["melband_big_beta4"],
        ],
        "ensemble_size": 3,
        "residual_fill_pct": 0.55,  # High fill for clean instrumental
        "bleed_guard_multiplier": 1.40,  # Very aggressive bleed prevention
        "description": "Maximum instrumental purity for backing tracks"
    },
}


# =============================================================================
# Profile Defaults
# =============================================================================

PROFILE_DEFAULTS: Dict[SeparationProfile, Dict] = {
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
    SeparationProfile.V3_MAXIMUM: {
        "ensemble_size": 5,
        "residual_fill_pct": 0.35,
        "bleed_guard_multiplier": 1.20,
    },
    SeparationProfile.V4_CLEAN_INSTRUMENTAL: {
        "ensemble_size": 3,
        "residual_fill_pct": 0.0,
        "bleed_guard_multiplier": 1.0,
    },
}


# =============================================================================
# Helper Functions
# =============================================================================

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
    """
    import math
    
    if not models:
        return models
    
    # Extract SDR values
    vocal_sdrs = [m.vocal_sdr for m in models if m.vocal_sdr is not None]
    inst_sdrs = [m.inst_sdr for m in models if m.inst_sdr is not None]
    
    if len(vocal_sdrs) != len(models) or len(inst_sdrs) != len(models):
        # If no SDR data, use equal weights
        n = len(models)
        return [
            ModelSpec(
                id=m.id,
                vocal_sdr=m.vocal_sdr,
                inst_sdr=m.inst_sdr,
                vocal_weight=m.vocal_weight if m.vocal_weight is not None else 8.0,
                inst_weight=m.inst_weight if m.inst_weight is not None else 8.0,
                description=m.description,
                kwargs=m.kwargs
            )
            for m in models
        ]
    
    # Calculate exponential-scaled weights (softmax-like)
    def calc_weights(sdrs: List[float], temp: float) -> List[float]:
        # Apply temperature scaling and exponential
        exp_vals = [math.exp((sdr - max(sdrs)) / temp) for sdr in sdrs]
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
            description=model.description,
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
    if profile == SeparationProfile.HYBRID_CLEANED:
        if ensemble_size not in (None, 6):
            raise ValueError("Cleaned hybrid uses six core models (V2 vocals + V4 instrumental)")
        return get_profile_models(SeparationProfile.V2_HIGH_QUALITY) + get_profile_models(SeparationProfile.V4_EXPERIMENTAL)
    models = list(MODEL_PRESETS.get(profile, MODEL_PRESETS[SeparationProfile.V2_HIGH_QUALITY]))
    
    if profile == SeparationProfile.V4_EXPERIMENTAL and ensemble_size not in (None, 3):
        raise ValueError("V4 uses exactly three core models; quality controls inference effort")
    if ensemble_size is not None:
        # Limit to requested ensemble size
        models = models[:ensemble_size]
    
    # Auto-calculate weights based on SDR if not manually specified
    models = auto_calculate_weights(models)
    
    return models


def selected_models(options):
    profile = SeparationProfile(options.get("separation_profile", "hybrid_cleaned"))
    if profile == SeparationProfile.HYBRID_CLEANED:
        return get_profile_models(profile, options.get("ensemble_size"))
    preset = options.get("separation_preset")
    if preset and profile != SeparationProfile.V4_EXPERIMENTAL:
        models = get_preset_models(SeparationPreset(preset))
        size = options.get("ensemble_size")
        return models[:size] if size is not None else models
    return get_profile_models(profile, options.get("ensemble_size"))


def get_preset_models(preset: SeparationPreset) -> List[ModelSpec]:
    """
    Get the list of models for a use-case preset with auto-calculated weights.
    
    Args:
        preset: The use-case preset
        
    Returns:
        List of ModelSpec objects for the preset with calculated weights
    """
    config = PRESET_CONFIGS.get(preset)
    if not config:
        return get_profile_models(SeparationProfile.V2_HIGH_QUALITY)
    
    models = list(config["models"])
    ensemble_size = config.get("ensemble_size", len(models))
    models = models[:ensemble_size]
    
    return auto_calculate_weights(models)


def get_profile_defaults(profile: SeparationProfile) -> Dict:
    """
    Get the default parameters for a given profile.
    
    Args:
        profile: The separation profile
        
    Returns:
        Dictionary of default parameters
    """
    if profile == SeparationProfile.HYBRID_CLEANED:
        return {"ensemble_size": 6, "residual_fill_pct": 0., "bleed_guard_multiplier": 1.}
    return PROFILE_DEFAULTS.get(profile, PROFILE_DEFAULTS[SeparationProfile.V2_HIGH_QUALITY]).copy()


def get_preset_defaults(preset: SeparationPreset) -> Dict:
    """
    Get the default parameters for a use-case preset.
    
    Args:
        preset: The use-case preset
        
    Returns:
        Dictionary of default parameters
    """
    config = PRESET_CONFIGS.get(preset, {})
    return {
        "ensemble_size": config.get("ensemble_size", 3),
        "residual_fill_pct": config.get("residual_fill_pct", 0.40),
        "bleed_guard_multiplier": config.get("bleed_guard_multiplier", 1.15),
    }


def list_profiles() -> List[Dict]:
    """
    List all available separation profiles with their details.
    
    Returns:
        List of profile information dictionaries
    """
    profiles = []
    for profile in SeparationProfile:
        defaults = get_profile_defaults(profile)
        models = get_profile_models(profile)
        profiles.append({
            "id": profile.value,
            "name": profile.name.replace("_", " ").title(),
            "model_count": len(models),
            "defaults": defaults,
            "models": [m.id for m in models],
        })
    return profiles


def list_presets() -> List[Dict]:
    """
    List all available use-case presets with their details.
    
    Returns:
        List of preset information dictionaries
    """
    presets = []
    for preset in SeparationPreset:
        config = PRESET_CONFIGS.get(preset, {})
        models = get_preset_models(preset)
        presets.append({
            "id": preset.value,
            "name": preset.name.replace("_", " ").title(),
            "description": config.get("description", ""),
            "model_count": len(models),
            "defaults": get_preset_defaults(preset),
            "models": [m.id for m in models],
        })
    return presets


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
        v_sdr = model.vocal_sdr or 0.0
        i_sdr = model.inst_sdr or 0.0
        v_wt = model.vocal_weight or 0.0
        i_wt = model.inst_weight or 0.0
        print(f"{model_name:<50} {v_sdr:<12.2f} {v_wt:<14.3f} {i_sdr:<12.2f} {i_wt:<14.3f}")
    
    total_v = sum(m.vocal_weight or 0.0 for m in models)
    total_i = sum(m.inst_weight or 0.0 for m in models)
    print("-" * 105)
    print(f"{'TOTALS':<50} {'':<12} {total_v:<14.3f} {'':<12} {total_i:<14.3f}")
    print()


def print_preset_info(preset: SeparationPreset) -> None:
    """
    Debug utility: Print detailed information about a preset.
    
    Args:
        preset: The preset to analyze
    """
    config = PRESET_CONFIGS.get(preset, {})
    models = get_preset_models(preset)
    defaults = get_preset_defaults(preset)
    
    print(f"\n=== {preset.value.upper()} Preset ===")
    print(f"Description: {config.get('description', 'N/A')}")
    print(f"Ensemble Size: {defaults['ensemble_size']}")
    print(f"Residual Fill: {defaults['residual_fill_pct']:.0%}")
    print(f"Bleed Guard: {defaults['bleed_guard_multiplier']:.2f}x")
    print(f"\nModels ({len(models)}):")
    for model in models:
        print(f"  - {model.id}")
        if model.description:
            print(f"    {model.description}")
    print()
