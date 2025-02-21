import numpy as np
import pyworld as pw

from modules.rvc.pitch_extraction import FeatureExtractor


def generate_allowed_midi(key='C', scale_type='major'):
    """
    Generates allowed MIDI note numbers (0-127) based on a given key and scale.
    """
    if scale_type == 'major':
        offsets = [0, 2, 4, 5, 7, 9, 11]
    elif scale_type == 'minor':
        offsets = [0, 2, 3, 5, 7, 8, 10]
    else:
        offsets = [0, 2, 4, 5, 7, 9, 11]
    note_to_semitone = {
        'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4,
        'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8,
        'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
    }
    root = note_to_semitone.get(key, 0)
    allowed_classes = [(root + offset) % 12 for offset in offsets]
    allowed = [m for m in range(0, 128) if m % 12 in allowed_classes]
    return allowed


def get_target_pitch(f, allowed_midi):
    """
    Converts a given frequency to MIDI, snaps it to the nearest allowed note, and converts back to Hz.
    """
    if f <= 0:
        return 0
    midi = 69 + 12 * np.log2(f / 440.0)
    snapped_midi = min(allowed_midi, key=lambda note: abs(note - midi))
    return 440.0 * (2 ** ((snapped_midi - 69) / 12.0))


def smooth_pitch(pitch, alpha):
    """
    Applies exponential smoothing to the pitch contour.

    Parameters:
      pitch : numpy array of pitch values.
      alpha : smoothing factor between 0 and 1. Higher values produce heavier smoothing.

    Returns:
      Smoothed pitch contour.
    """
    smoothed = np.copy(pitch)
    for i in range(1, len(pitch)):
        smoothed[i] = alpha * smoothed[i - 1] + (1 - alpha) * pitch[i]
    return smoothed


def autotune(signal, fs, key='C', scale_type='major', strength=1.0, humanize=0.0, f0_method='hybrid'):
    """
    Autotunes a vocal track by estimating pitch using FeatureExtractor.get_f0,
    snapping it to a musical scale, and synthesizing a corrected output.

    NOTE: This function expects a mono audio signal.

    Parameters:
      signal         : numpy array containing a mono audio signal.
      fs             : sample rate (e.g., 44100).
      key            : target musical key (e.g., 'C').
      scale_type     : 'major' or 'minor'.
      strength       : float in [0, 1], where 1.0 means full pitch correction.
      humanize       : float in [0, 1] controlling the smoothing of pitch transitions.
                       0 means no smoothing (more robotic), 1 means heavy smoothing (more natural).
      f0_method      : 'hybrid', 'crepe', 'rmvpe+', etc. (whatever is implemented in FeatureExtractor).

    Returns:
      y : numpy array of the autotuned audio.
    """

    # Ensure signal is double precision
    signal = np.asarray(signal, dtype=np.float64)

    # Minimal dummy config required by FeatureExtractor
    class DummyConfig:
        x_pad = 0.2
        x_query = 2.0
        x_center = 0.0
        x_max = 10.0
        is_half = False
        device = 'cpu'

    # Instantiate the feature extractor.
    fe = FeatureExtractor(fs, DummyConfig(), onnx=False)

    # Use our new get_f0 function for pitch extraction.
    # The user can choose 'hybrid', 'crepe', 'rmvpe+', etc.
    f0_coarse, f0_raw = fe.get_f0(
        x=signal,
        f0_up_key=0,
        f0_method=f0_method,
        merge_type="median",
        filter_radius=3,
        crepe_hop_length=160,
        f0_autotune=False,
        rmvpe_onnx=False,
        f0_min=50,
        f0_max=1100
    )

    # Compute a time axis based on the hop length used during extraction.
    hop_length = fe.window  # typically 160 samples at 16000Hz in the original; adapt if needed
    timeaxis = np.arange(len(f0_raw)) * hop_length / fs

    # --- Step 2: WORLD Analysis for spectral envelope & aperiodicity ---
    sp = pw.cheaptrick(signal, f0_raw, timeaxis, fs)
    ap = pw.d4c(signal, f0_raw, timeaxis, fs)

    # --- Step 3: Key/Scale Quantization ---
    allowed_midi = generate_allowed_midi(key, scale_type)
    target_f0 = np.array([get_target_pitch(f, allowed_midi) if f > 0 else 0 for f in f0_raw])

    # --- Step 4: Blending Pitch Correction ---
    corrected_f0 = (1 - strength) * f0_raw + strength * target_f0

    # --- Step 5: Humanize via smoothing (if desired) ---
    if humanize > 0:
        alpha = min(humanize * 0.9, 0.99)
        corrected_f0 = smooth_pitch(corrected_f0, alpha)

    # --- Step 6: Resynthesis using WORLD ---
    y = pw.synthesize(corrected_f0, sp, ap, fs)
    return y


def detect_key(signal, fs, f0_method):
    """
    Detects the musical key and scale (major/minor) from the given vocal track.

    This function extracts the f0 contour using FeatureExtractor.get_f0,
    filters voiced frames, converts them to MIDI note numbers, computes the
    histogram of pitch classes, and compares it against Krumhansl–Schmuckler key profiles.

    For key detection, stereo signals are averaged to mono.

    Returns:
      detected_key : string representing the key (e.g., 'C', 'G#', etc.)
      scale_type   : 'major' or 'minor'
    """
    # Average channels if stereo for key detection only
    if signal.ndim > 1:
        mono_signal = np.mean(signal, axis=1)
    else:
        mono_signal = signal
    # Ensure mono signal is double precision for consistency
    mono_signal = np.asarray(mono_signal, dtype=np.float64)

    # Minimal dummy config
    class DummyConfig:
        x_pad = 0.2
        x_query = 2.0
        x_center = 0.0
        x_max = 10.0
        is_half = False
        device = 'cpu'

    # Use your feature extractor to get pitch
    fe = FeatureExtractor(fs, DummyConfig(), onnx=False)
    _, f0_raw = fe.get_f0(
        x=mono_signal,
        f0_up_key=0,
        f0_method=f0_method,
        merge_type="median",
        filter_radius=3,
        crepe_hop_length=160,
        f0_autotune=False,
        rmvpe_onnx=False,
        f0_min=50,
        f0_max=1100
    )

    # Filter out unvoiced parts
    voiced = f0_raw[f0_raw > 0]
    if len(voiced) == 0:
        return 'C', 'major'

    # Convert to MIDI and then to pitch classes (0-11)
    midi_notes = 69 + 12 * np.log2(voiced / 440.0)
    pitch_classes = np.mod(np.round(midi_notes), 12).astype(int)

    # Create a histogram of pitch classes
    hist = np.zeros(12)
    for pc in pitch_classes:
        hist[pc] += 1
    if hist.sum() > 0:
        hist = hist / hist.sum()

    # Krumhansl-Schmuckler key profiles (normalized)
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    major_profile /= major_profile.sum()
    minor_profile /= minor_profile.sum()

    best_corr = -np.inf
    best_key = 0
    best_scale = 'major'
    # Evaluate all 12 possible keys for both major and minor
    for key_idx in range(12):
        major_rotated = np.roll(major_profile, key_idx)
        minor_rotated = np.roll(minor_profile, key_idx)
        corr_major = np.corrcoef(hist, major_rotated)[0, 1]
        corr_minor = np.corrcoef(hist, minor_rotated)[0, 1]
        if corr_major > best_corr:
            best_corr = corr_major
            best_key = key_idx
            best_scale = 'major'
        if corr_minor > best_corr:
            best_corr = corr_minor
            best_key = key_idx
            best_scale = 'minor'

    # Convert key index to note name (using sharps)
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    detected_key = note_names[best_key]
    return detected_key, best_scale


def auto_tune_track(signal, fs, strength=0.5, humanize=0.5, f0_method='hybrid'):
    """
    Wrapper that automatically detects the key/scale of the input vocal track and applies autotune.

    For key detection, if the input is stereo it is averaged to mono. However, if the input
    is stereo, each channel is autotuned separately to preserve stereo imaging.

    Parameters:
      signal         : numpy array containing a mono or stereo audio signal.
                       Expected shape is (n_samples,) for mono or (n_samples, n_channels) for stereo.
      fs             : sample rate (e.g., 44100).
      strength       : float in [0, 1] controlling the degree of pitch correction.
      humanize       : float in [0, 1] controlling the smoothing of pitch transitions.
      f0_method      : 'hybrid', 'crepe', 'rmvpe+', etc.
                       (Any method recognized by FeatureExtractor's get_f0).

    Returns:
      tuned_signal   : numpy array of the autotuned audio (same shape as input).
      detected_key   : detected musical key as a string.
      scale_type     : detected scale ('major' or 'minor').
    """
    detected_key, scale_type = detect_key(signal, fs, f0_method)

    # Process each channel separately if stereo; otherwise, process directly.
    if signal.ndim == 1:
        tuned_signal = autotune(signal, fs, key=detected_key, scale_type=scale_type,
                                strength=strength, humanize=humanize, f0_method=f0_method)
    else:
        tuned_channels = []
        for ch in range(signal.shape[1]):
            channel = signal[:, ch]
            tuned_channel = autotune(channel, fs, key=detected_key, scale_type=scale_type,
                                     strength=strength, humanize=humanize, f0_method=f0_method)
            tuned_channels.append(tuned_channel)
        tuned_signal = np.column_stack(tuned_channels)

    return tuned_signal, detected_key, scale_type
