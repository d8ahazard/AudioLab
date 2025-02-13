import numpy as np


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
