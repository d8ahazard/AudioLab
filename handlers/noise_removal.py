import librosa
import numpy as np
import pyloudnorm as pyln  # added import for pyloudnorm


def restore_silence(original_audio, cloned_audio, sr_original, sr_cloned,
                    noise_floor=0.001, frame_size=1024):
    """
    Enhanced audio restoration with proper overlap-add processing and hybrid normalization.
    Fixed windowing implementation with correct reconstruction.
    Final output is loudness-normalized using pyloudnorm to match the original audio's integrated loudness.
    """

    # Helper function for overlap-add reconstruction
    def overlap_add(frames, hop_length):
        frame_size, n_frames = frames.shape
        output_length = hop_length * (n_frames - 1) + frame_size
        result = np.zeros(output_length, dtype=frames.dtype)
        for i in range(n_frames):
            start = i * hop_length
            result[start:start + frame_size] += frames[:, i]
        return result

    # Resample and match lengths
    if sr_cloned != sr_original:
        cloned_audio = librosa.resample(cloned_audio.T, sr_cloned, sr_original).T

    target_length = original_audio.shape[0]
    if cloned_audio.shape[0] > target_length:
        cloned_audio = cloned_audio[:target_length]
    else:
        cloned_audio = np.pad(cloned_audio, ((0, target_length - cloned_audio.shape[0]), (0, 0)),
                              mode='constant')

    # Convert to 32-bit for processing
    original = original_audio.astype(np.float32)
    cloned = cloned_audio.astype(np.float32)

    # Overlap-add parameters (50% overlap)
    hop_size = frame_size // 2
    window = np.hanning(frame_size)

    processed = np.zeros_like(cloned)

    for ch in range(cloned.shape[1]):
        channel_clone = cloned[:, ch]
        channel_orig = original[:, ch] if original.ndim > 1 else original

        # Create windowed frames with 50% overlap
        frames = librosa.util.frame(channel_clone, frame_length=frame_size, hop_length=hop_size)
        orig_frames = librosa.util.frame(channel_orig, frame_length=frame_size, hop_length=hop_size)
        n_frames = frames.shape[1]

        # Apply window to frames
        windowed_frames = frames * window[:, None]
        processed_frames = np.zeros_like(windowed_frames)

        for i in range(n_frames):
            # Original frame analysis
            orig_frame = orig_frames[:, i] * window
            orig_rms = np.sqrt(np.mean(orig_frame ** 2))
            orig_peak = np.max(np.abs(orig_frame))

            # Clone frame analysis
            clone_frame = windowed_frames[:, i]
            clone_rms = np.sqrt(np.mean(clone_frame ** 2))
            clone_peak = np.max(np.abs(clone_frame))

            if orig_rms < noise_floor:
                processed_frames[:, i] = 0
            else:
                # Hybrid scaling with RMS/peak balance
                rms_scale = orig_rms / (clone_rms + 1e-8)
                peak_scale = orig_peak / (clone_peak + 1e-8)
                combined_scale = 0.7 * rms_scale + 0.3 * peak_scale

                # Frequency-domain processing
                frame_fft = np.fft.rfft(clone_frame)
                magnitudes = np.abs(frame_fft)
                angles = np.angle(frame_fft)

                # Noise reduction (spectral subtraction)
                noise_estimate = np.median(magnitudes)
                magnitudes = np.clip(magnitudes - noise_estimate, 0, None)

                processed_frame = np.fft.irfft(magnitudes * np.exp(1j * angles))
                processed_frame *= combined_scale
                processed_frames[:, i] = processed_frame

        # Overlap-add reconstruction using custom implementation
        processed_ch = overlap_add(processed_frames, hop_size)

        # Match length precisely
        if len(processed_ch) > target_length:
            processed_ch = processed_ch[:target_length]
        else:
            processed_ch = np.pad(processed_ch, (0, target_length - len(processed_ch)))

        processed[:, ch] = processed_ch

    # Loudness normalization using pyloudnorm
    meter = pyln.Meter(sr_original)
    orig_loudness = meter.integrated_loudness(original)
    proc_loudness = meter.integrated_loudness(processed)
    normalized_audio = pyln.normalize.loudness(processed, proc_loudness, orig_loudness)

    return normalized_audio
