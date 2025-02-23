import math
import logging
import numpy as np
import torch
import librosa
from collections import Counter

from handlers.spectrogram import F0Visualizer
from modules.rvc.infer.lib.audio import autotune_f0
from modules.rvc.pitch_extraction import FeatureExtractor

logger = logging.getLogger(__name__)
vis = F0Visualizer()


def group_pitch_shift_factors(time_axis, shift_factors, tolerance=0.02):
    """
    Group contiguous frames that share similar pitch shift factors.
    Returns a list of segments (start_time, end_time, median_shift) for processing.
    """
    groups = []
    if len(shift_factors) == 0:
        return groups
    start_idx = 0
    current = shift_factors[0]
    for i in range(1, len(shift_factors)):
        if abs(shift_factors[i] - current) > tolerance:
            median_shift = float(np.median(shift_factors[start_idx:i]))
            groups.append((time_axis[start_idx], time_axis[i - 1], median_shift))
            start_idx = i
            current = shift_factors[i]
    median_shift = float(np.median(shift_factors[start_idx:]))
    groups.append((time_axis[start_idx], time_axis[-1], median_shift))
    return groups


def detect_key(audio, sr):
    """
    Detects the key and scale of an audio signal using a Krumhansl–Schmuckler–style algorithm.
    Computes the chroma_stft, averages it over time, and then compares the result against rotated
    major and minor key profiles to choose the best match.

    Parameters:
      audio: np.array, the final corrected audio signal.
      sr: int, sample rate of the audio.

    Returns:
      best_key: The detected key (e.g. 'C', 'G#', etc.).
      best_scale: The detected scale ('major' or 'minor').
    """
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # Krumhansl–Schmuckler key profiles (from Krumhansl & Kessler, 1982)
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 2.88, 2.75])
    keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    best_corr = -np.inf
    best_key = None
    best_scale = None

    # For each possible key (by rotating the profile) check both major and minor correlations.
    for i in range(12):
        major_rot = np.roll(major_profile, i)
        minor_rot = np.roll(minor_profile, i)
        corr_major = np.corrcoef(chroma_mean, major_rot)[0, 1]
        corr_minor = np.corrcoef(chroma_mean, minor_rot)[0, 1]
        if corr_major > best_corr:
            best_corr = corr_major
            best_key = keys[i]
            best_scale = "major"
        if corr_minor > best_corr:
            best_corr = corr_minor
            best_key = keys[i]
            best_scale = "minor"

    logger.info(f"Detected Key: {best_key}, Scale: {best_scale} (corr={best_corr:.3f})")
    return best_key, best_scale


def process_channel(channel_audio, orig_sr, extractor, f0_method, strength=0.5, humanize=False):
    """
    Process a single channel using CPU-based operations.
      1. Resample the channel (without converting to mono) to 16000 Hz for pitch extraction.
      2. Extract f0 using the provided FeatureExtractor.
      3. Auto-tune the f0 and optionally add natural humanization.
      4. Compute per-frame pitch shift factors and segment the track accordingly.
      5. For each segment, apply CPU-based pitch shifting (via librosa) and blend the result.
      6. Detect key and scale using a Krumhansl–Schmuckler algorithm.
    """
    # Step 1: Resample for pitch extraction at 16000 Hz.
    target_extraction_sr = 16000
    audio_extraction = librosa.resample(channel_audio.astype(np.float32),
                                        orig_sr=orig_sr,
                                        target_sr=target_extraction_sr)

    # Step 2: Extract f0 using FeatureExtractor.
    f0_up_key = 0
    merge_type = "median"
    filter_radius = 3
    crepe_hop_length = 160
    f0_autotune_flag = False  # We'll perform autotune separately.
    f0_coarse, original_f0 = extractor.get_f0(
        x=audio_extraction,
        f0_up_key=f0_up_key,
        f0_method=f0_method,
        merge_type=merge_type,
        filter_radius=filter_radius,
        crepe_hop_length=crepe_hop_length,
        f0_autotune=f0_autotune_flag,
        f0_min=50,
        f0_max=1100
    )

    # Step 3: Auto-tune f0 and optionally humanize.
    if f0_autotune_flag:
        tuned_f0 = autotune_f0(original_f0)
    else:
        tuned_f0 = original_f0
    if humanize:
        cents_variation = np.random.uniform(-0.05, 0.05, size=tuned_f0.shape)
        tuned_f0 = tuned_f0 * (2 ** (cents_variation / 12))
    vis.add_f0(tuned_f0, "Auto-tuned f0")

    # Step 4: Compute per-frame pitch shift factors.
    with np.errstate(divide='ignore', invalid='ignore'):
        shift_factors = np.where(original_f0 > 1, tuned_f0 / original_f0, 1.0)
    hop_length = 160
    frame_duration = hop_length / target_extraction_sr
    time_axis = np.arange(len(shift_factors)) * frame_duration
    groups = group_pitch_shift_factors(time_axis, shift_factors)

    # Step 5: Process each segment using CPU-based librosa pitch shifting.
    corrected_audio = np.copy(channel_audio).astype(np.float32)
    total_samples = channel_audio.shape[0]
    for start_t, end_t, median_shift in groups:
        semitone_shift = 0.0 if median_shift <= 0 else 12 * math.log2(median_shift)
        start_sample = int(start_t * orig_sr)
        end_sample = int(end_t * orig_sr) + 1
        start_sample = max(0, start_sample)
        end_sample = min(total_samples, end_sample)
        if end_sample - start_sample < int(orig_sr * 0.02):
            continue  # Skip segments shorter than 20ms.

        # Convert segment to float64 for compatibility.
        segment = channel_audio[start_sample:end_sample].astype(np.float64)
        try:
            shifted_segment = librosa.effects.pitch_shift(segment, sr=orig_sr, n_steps=semitone_shift)
        except Exception as e:
            logger.error(f"Error pitch shifting segment [{start_sample}:{end_sample}]: {e}")
            continue

        # Cast the result back to float32 and blend.
        shifted_segment = shifted_segment.astype(np.float32)
        blended = (1.0 - strength) * channel_audio[start_sample:end_sample].astype(np.float32) \
                  + strength * shifted_segment
        corrected_audio[start_sample:end_sample] = blended

    # Step 6: Detect key and scale using the KS algorithm.
    detected_key, detected_scale = detect_key(corrected_audio, orig_sr)
    return corrected_audio, detected_key, detected_scale


def auto_tune_track(audio, tgt_sr, strength=0.5, humanize=False, f0_method="rmvpe", extractor_config=None):
    """
    Advanced auto-tune function that avoids heavy GPU usage by performing pitch shifting
    on the CPU via librosa, while still using advanced segmentation and f0 extraction.

    Parameters:
      audio: np.array with shape (num_samples,) or (num_channels, num_samples)
      tgt_sr: Original sample rate of the input track.
      strength: Blending factor between original and pitch-corrected audio.
      humanize: If True, introduces slight natural pitch variations.
      f0_method: String or list indicating the pitch extraction method (e.g., "rmvpe", "crepe", "hybrid", etc.)
      extractor_config: Optional configuration object for FeatureExtractor.

    Returns:
      corrected_audio: np.array with the same shape and sample rate as input.
      detected_key: Detected musical key.
      detected_scale: Detected musical scale.
    """
    logger.info("Starting advanced auto-tune process (CPU-based pitch shifting).")

    # If no configuration is provided, use a dummy config.
    if extractor_config is None:
        class DummyConfig:
            x_pad = 2
            x_query = 16
            x_center = 8
            x_max = 120
            is_half = False
            device = "cuda" if torch.cuda.is_available() else "cpu"

        extractor_config = DummyConfig()

    # Initialize FeatureExtractor with internal extraction SR fixed at 16000.
    extractor = FeatureExtractor(tgt_sr=tgt_sr, config=extractor_config, onnx=False)

    # Process each channel independently.
    if audio.ndim == 1:
        corrected_channel, detected_key, detected_scale = process_channel(audio, tgt_sr, extractor, f0_method, strength,
                                                                          humanize)
        corrected_audio = corrected_channel
    elif audio.ndim == 2:
        corrected_channels = []
        keys = []
        scales = []
        for ch in audio:
            corr, key, scale = process_channel(ch, tgt_sr, extractor, f0_method, strength, humanize)
            corrected_channels.append(corr)
            keys.append(key)
            scales.append(scale)
        corrected_audio = np.vstack(corrected_channels)
        # Choose the most common key/scale among channels.
        detected_key = Counter(keys).most_common(1)[0][0]
        detected_scale = Counter(scales).most_common(1)[0][0]
    else:
        logger.error("Audio input must be a 1D or 2D numpy array.")
        return audio, "N/A", "N/A"

    logger.info("Advanced auto-tune process complete (CPU-based).")
    return corrected_audio, detected_key, detected_scale
