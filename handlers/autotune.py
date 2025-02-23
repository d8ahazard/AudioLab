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
    Detects the key of an audio signal using its chromagram.
    Computes the chroma_stft, averages it over time, and selects the key corresponding
    to the maximum mean chroma value.

    Parameters:
      audio: np.array, the audio signal.
      sr: int, sample rate of the audio.

    Returns:
      estimated_key: The detected key as a string.
      scale: The detected scale (defaulted to "major").
    """
    import librosa  # in case not imported here
    chromagram = librosa.feature.chroma_stft(y=audio, sr=sr)
    mean_chroma = np.mean(chromagram, axis=1)
    chroma_to_key = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    estimated_key_index = np.argmax(mean_chroma)
    estimated_key = chroma_to_key[estimated_key_index]
    logger.info(f"Detected Key: {estimated_key}")
    return estimated_key, "major"


def process_channel(channel_audio, orig_sr, extractor, f0_method, strength=0.5, humanize=False):
    """
    Process a single channel using CPU-based operations.
      1. Resample the channel (without converting to mono) to 16000 Hz for pitch extraction.
      2. Extract f0 using the provided FeatureExtractor.
      3. Auto-tune the f0 and optionally add natural humanization.
      4. Compute per-frame pitch shift factors and segment the track accordingly.
      5. For each segment, apply CPU-based pitch shifting (via librosa) and blend the result.
      6. Perform key detection using a chromagram-based approach.
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

    # Step 3: Autotune f0 and optionally humanize.
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

    # Step 6: Detect key using the final corrected audio.
    detected_key, scale = detect_key(corrected_audio, orig_sr)
    return corrected_audio, detected_key, scale


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
      scale: Detected scale ("major" or "minor").
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
        corrected_channel, detected_key, scale = process_channel(audio, tgt_sr, extractor, f0_method, strength,
                                                                 humanize)
        corrected_audio = corrected_channel
    elif audio.ndim == 2:
        corrected_channels = []
        keys = []
        scales = []
        for ch in audio:
            corr, key, scl = process_channel(ch, tgt_sr, extractor, f0_method, strength, humanize)
            corrected_channels.append(corr)
            keys.append(key)
            scales.append(scl)
        corrected_audio = np.vstack(corrected_channels)
        detected_key = Counter(keys).most_common(1)[0][0]
        scale = Counter(scales).most_common(1)[0][0]
    else:
        logger.error("Audio input must be a 1D or 2D numpy array.")
        return audio, "N/A", "N/A"

    logger.info("Advanced auto-tune process complete (CPU-based).")
    return corrected_audio, detected_key, scale
