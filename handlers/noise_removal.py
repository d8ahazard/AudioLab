import traceback

import librosa
import numpy as np
from scipy import signal


def restore_silence(
        original_audio,
        processed,
        silence_threshold=0.001,
        frame_size=512,
        smoothing_kernel_size=1  # Set to 1 to disable smoothing.
):
    """
    Restores silence and forces the processed audio's dynamic envelope to match
    the original audio on a per-frame basis. In frames where the original audio is
    nearly silent, the output is forced to zero. Otherwise, each frame of the processed
    audio is scaled by the exact ratio of the original RMS to the processed RMS.

    This function works for mono or multi-channel audio. If the reference (original_audio)
    is mono but processed is multi-channel, the reference is tiled appropriately.

    Parameters:
        original_audio (np.ndarray): Original audio array of shape (n_samples,) or (n_samples, channels).
        processed (np.ndarray): Processed audio array to be scaled (must have same number of time samples).
        silence_threshold (float): Maximum absolute amplitude in a frame to be considered silent.
        frame_size (int): Number of samples per frame.
        smoothing_kernel_size (int): Size of moving-average kernel (in frames) to smooth the gain factors.
                                     Set to 1 to disable smoothing.

    Returns:
        np.ndarray: Processed audio after applying per-frame gain adjustments.
    """
    # Ensure both arrays have the same number of time samples.
    L = original_audio.shape[0]
    L_proc = processed.shape[0]
    if L_proc < L:
        pad_width = ((0, L - L_proc),) + ((0, 0),) * (original_audio.ndim - 1)
        processed = np.pad(processed, pad_width, mode='constant')
    elif L_proc > L:
        processed = processed[:L]

    # If the reference is mono but processed is multi-channel, tile the reference.
    if original_audio.ndim < processed.ndim:
        original_audio = np.tile(original_audio[:, np.newaxis], (1, processed.shape[1]))

    # Pad so that the length is a multiple of frame_size.
    n_frames = int(np.ceil(L / frame_size))
    pad_total = n_frames * frame_size - L
    if pad_total > 0:
        pad_width = ((0, pad_total),) + ((0, 0),) * (original_audio.ndim - 1)
        orig_padded = np.pad(original_audio, pad_width, mode='constant')
        proc_padded = np.pad(processed, pad_width, mode='constant')
    else:
        orig_padded = original_audio
        proc_padded = processed

    # Reshape to frames: shape (n_frames, frame_size, [channels...])
    new_shape = (n_frames, frame_size) + orig_padded.shape[1:]
    orig_frames = orig_padded.reshape(new_shape)
    proc_frames = proc_padded.reshape(new_shape)

    # Compute per-frame metrics.
    if original_audio.ndim == 1:
        max_vals = np.max(np.abs(orig_frames), axis=1)  # (n_frames,)
        orig_rms = np.sqrt(np.mean(orig_frames ** 2, axis=1))
        proc_rms = np.sqrt(np.mean(proc_frames ** 2, axis=1))
    else:
        axes = (1,) + tuple(range(2, orig_frames.ndim))
        max_vals = np.max(np.abs(orig_frames), axis=axes)
        orig_rms = np.sqrt(np.mean(orig_frames ** 2, axis=axes))
        proc_rms = np.sqrt(np.mean(proc_frames ** 2, axis=axes))

    # Compute per-frame scaling factors (exact ratio).
    scales = orig_rms / (proc_rms + 1e-8)
    # For frames that are nearly silent in the original, force gain to 0.
    scales[max_vals < silence_threshold] = 0.0

    # Optional smoothing of scales over frames.
    if smoothing_kernel_size > 1:
        kernel = np.ones(smoothing_kernel_size) / smoothing_kernel_size
        scales = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=0, arr=scales)

    # Broadcast scales to the frame shape and apply.
    scales = scales.reshape((n_frames,) + (1,) * (orig_frames.ndim - 1))
    proc_scaled = proc_frames * scales

    # Reshape back to the original time axis and remove padding.
    processed_scaled = proc_scaled.reshape(orig_padded.shape)[:L]
    return processed_scaled


def restore_silence_deepseek(original_audio, cloned_audio, sr_original, sr_cloned,
                             noise_floor=0.001, frame_size=1024):
    """
    Enhanced audio restoration with proper overlap-add processing and hybrid normalization.
    Fixed windowing implementation with correct reconstruction.
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

    # Final volume matching
    orig_max = np.max(np.abs(original))
    processed_max = np.max(np.abs(processed))
    return processed * (orig_max / (processed_max + 1e-8))


def restore_silence_gemini(original_audio, cloned_audio, original_sr, cloned_sr, noise_floor=0.001):
    """
    Processes an RVC cloned audio track to match the original.

    Args:
        original_audio_path: Path to the original stereo audio file.
        cloned_audio_path: Path to the RVC cloned stereo audio file.
        noise_floor: The noise floor level to apply.

    Returns:
        The processed cloned audio as a numpy array and the sample rate.
        Returns None if there's an error.
    """
    try:
        # 1. Load audio files
        # original_audio, original_sr = librosa.load(original_audio_path, sr=None, mono=False)  # Load in stereo
        # cloned_audio, cloned_sr = librosa.load(cloned_audio_path, sr=None, mono=False)  # Load in stereo

        if original_audio.ndim != 2 or cloned_audio.ndim != 2:
            raise ValueError("Audio files must be stereo.")

        # Resample if necessary
        if original_sr != cloned_sr:
            cloned_audio = librosa.resample(cloned_audio, cloned_sr, original_sr, axis=1)  # Resample both channels
            cloned_sr = original_sr

        # Match lengths (pad or trim)
        original_len = original_audio.shape[1]
        cloned_len = cloned_audio.shape[1]

        if cloned_len < original_len:
            padding = np.zeros((2, original_len - cloned_len))  # Stereo padding
            cloned_audio = np.concatenate([cloned_audio, padding], axis=1)
        elif cloned_len > original_len:
            cloned_audio = cloned_audio[:, :original_len]  # Trim

        # 2. Apply noise floor
        cloned_audio = np.where(np.abs(cloned_audio) < noise_floor, 0, cloned_audio)

        # 3. Volume scaling (more robust)
        def scale_channel(original_chan, cloned_chan):
            original_peak = np.max(np.abs(original_chan))
            cloned_peak = np.max(np.abs(cloned_chan))

            if original_peak == 0:  # Avoid division by zero
                scale_factor = 1.0  # Or potentially 0.0 if you want to silence it
            elif cloned_peak == 0:
                scale_factor = 0.0 if original_peak != 0 else 1.0
            else:
                scale_factor = original_peak / cloned_peak

            cloned_chan = cloned_chan * scale_factor
            return cloned_chan

        for chan in range(2):  # Process both channels
            cloned_audio[chan] = scale_channel(original_audio[chan], cloned_audio[chan])

        return cloned_audio

    except Exception as e:
        print(f"Error processing audio: {e}")
        traceback.print_exc()
        return None


def restore_silence_claude(y_orig, y_clone, sr_orig, sr_clone, noise_floor=0.001):
    """
    Process an RVC cloned audio track to match characteristics of the original.

    Args:
        y_orig (numpy.ndarray): Original audio data (stereo)
        y_clone (numpy.ndarray): Cloned audio data (stereo)
        sr_orig (float): Original sample rate
        sr_clone (float): Clone sample rate
        noise_floor (float): Threshold for noise removal (default: 0.001)

    Returns:
        numpy.ndarray: Processed audio data
        float: Sample rate
    """
    # Ensure stereo format
    if y_orig.ndim == 1:
        y_orig = np.stack([y_orig, y_orig])
    if y_clone.ndim == 1:
        y_clone = np.stack([y_clone, y_clone])

    # Match lengths
    if len(y_orig[0]) > len(y_clone[0]):
        # Pad cloned audio with silence
        padding = len(y_orig[0]) - len(y_clone[0])
        y_clone = np.pad(y_clone, ((0, 0), (0, padding)), mode='constant')
    else:
        # Trim cloned audio
        y_clone = y_clone[:, :len(y_orig[0])]

    # Apply noise floor to cloned audio
    noise_mask = np.abs(y_clone) < noise_floor
    y_clone[noise_mask] = 0

    def get_active_segments(audio, window_size=2048):
        # Compute RMS energy and get the first row since librosa.feature.rms returns 2D array
        energy = librosa.feature.rms(y=audio, frame_length=window_size)[0]
        # Create a mask the same length as the audio by repeating the energy values
        hop_length = window_size // 4  # Default hop_length in librosa.feature.rms
        mask = np.repeat(energy > noise_floor, hop_length)
        # Pad or trim to match original length
        if len(mask) < len(audio):
            mask = np.pad(mask, (0, len(audio) - len(mask)), mode='edge')
        else:
            mask = mask[:len(audio)]
        return mask

    # Process each channel separately
    for channel in range(2):
        orig_active = get_active_segments(y_orig[channel])
        clone_active = get_active_segments(y_clone[channel])

        # Only process if we have active segments
        if np.any(orig_active) and np.any(clone_active):
            # Get volume statistics for active segments
            orig_max = np.max(np.abs(y_orig[channel][orig_active]))
            orig_min = np.min(np.abs(y_orig[channel][orig_active][y_orig[channel][orig_active] > noise_floor]))
            clone_max = np.max(np.abs(y_clone[channel][clone_active]))
            clone_min = np.min(np.abs(y_clone[channel][clone_active][y_clone[channel][clone_active] > noise_floor]))

            # Calculate scaling factors
            scale_max = orig_max / clone_max if clone_max > 0 else 1
            scale_min = orig_min / clone_min if clone_min > 0 else 1

            # Apply smooth volume adjustment using sigmoid-like function
            def smooth_scale(x, scale_min, scale_max):
                if clone_max == clone_min:
                    return x * scale_max
                normalized = (np.abs(x) - clone_min) / (clone_max - clone_min)
                scale = scale_min + (scale_max - scale_min) * normalized
                return np.sign(x) * np.abs(x) * scale

            # Only apply scaling to active segments
            y_clone[channel][clone_active] = smooth_scale(
                y_clone[channel][clone_active],
                scale_min,
                scale_max
            )

    # Apply mild smoothing to prevent artifacts
    window_length = 1024
    window = signal.hann(window_length)
    for channel in range(2):
        y_clone[channel] = signal.convolve(
            y_clone[channel],
            window / window.sum(),
            mode='same'
        )

    return y_clone


def restore_silence_chatgpt(original_audio: np.ndarray,
                            cloned_audio: np.ndarray,
                            sr_original: int,
                            sr_cloned: int,
                            noise_floor: float = 0.001) -> np.ndarray:
    """
    Process a cloned audio track to match an original track by ensuring:
      1. The cloned track is resampled (if needed) and its length is padded/trimmed
         to match the original.
      2. Noise below the threshold is removed.
      3. The dynamic range (min/max of the singing portions) of the cloned track is
         linearly adjusted to match that of the original.
      4. Any portion where the original is silent is forced to silence in the cloned output.

    Parameters:
      original_audio (np.ndarray): Original audio track with shape (n_samples, channels).
      cloned_audio (np.ndarray): Cloned audio track with shape (m_samples, channels).
      sr_original (int): Sample rate of the original audio.
      sr_cloned (int): Sample rate of the cloned audio.
      noise_floor (float): Threshold below which audio is considered silent.

    Returns:
      np.ndarray: The processed cloned audio track.
    """
    # Step 1: Resample cloned audio if the sample rates differ.
    if sr_cloned != sr_original:
        num_channels = cloned_audio.shape[1] if cloned_audio.ndim > 1 else 1
        resampled_channels = []
        for ch in range(num_channels):
            channel_data = cloned_audio[:, ch] if num_channels > 1 else cloned_audio
            resampled = librosa.resample(channel_data, orig_sr=sr_cloned, target_sr=sr_original)
            resampled_channels.append(resampled)
        cloned_audio = (np.stack(resampled_channels, axis=-1)
                        if num_channels > 1 else resampled_channels[0])

    # Ensure cloned_audio is a 2D array (n_samples, channels)
    if cloned_audio.ndim == 1:
        cloned_audio = np.expand_dims(cloned_audio, axis=-1)

    # Step 1b: Adjust length of cloned_audio to match original_audio.
    orig_length = original_audio.shape[0]
    clone_length = cloned_audio.shape[0]
    if clone_length < orig_length:
        pad_width = orig_length - clone_length
        padding = np.zeros((pad_width, cloned_audio.shape[1]), dtype=cloned_audio.dtype)
        cloned_audio = np.concatenate([cloned_audio, padding], axis=0)
    elif clone_length > orig_length:
        cloned_audio = cloned_audio[:orig_length, :]

    # Step 2: Apply noise floor to remove low-level noise.
    cloned_audio = np.where(np.abs(cloned_audio) < noise_floor, 0, cloned_audio)

    # Step 3: Dynamic range adjustment based on singing parts.
    processed_cloned = np.copy(cloned_audio)
    if original_audio.ndim == 1:
        original_audio = np.expand_dims(original_audio, axis=-1)

    num_channels = original_audio.shape[1]
    for ch in range(num_channels):
        # Identify "singing" parts in the original track.
        singing_mask = np.abs(original_audio[:, ch]) > noise_floor
        if np.sum(singing_mask) == 0:
            continue

        orig_singing = original_audio[singing_mask, ch]
        orig_min, orig_max = np.min(orig_singing), np.max(orig_singing)

        clone_singing = processed_cloned[singing_mask, ch]
        clone_min, clone_max = np.min(clone_singing), np.max(clone_singing)

        if np.isclose(clone_max, clone_min):
            continue

        scale_factor = (orig_max - orig_min) / (clone_max - clone_min)
        offset = orig_min - clone_min * scale_factor

        processed_cloned[singing_mask, ch] = processed_cloned[singing_mask, ch] * scale_factor + offset

    # Step 4: Force silence in the cloned track wherever the original is silent.
    # This is done on a per-sample, per-channel basis.
    silent_mask = np.abs(original_audio) < noise_floor
    processed_cloned[silent_mask] = 0

    return processed_cloned
