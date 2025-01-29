import numpy as np


def restore_silence(original_audio, processed, silence_threshold=0.001, frame_size=512):
    """
    Restores silence and ensures that processed audio does not exceed the original audio's volume,
    on a frame-by-frame basis.

    Args:
        original_audio (numpy.ndarray): The original audio array.
        processed (numpy.ndarray): The processed audio array.
        silence_threshold (float): Amplitude below which a frame is considered silence.
        frame_size (int): Number of samples per frame for processing.

    Returns:
        numpy.ndarray: The processed audio with silences restored and volume capped to match the original.
    """
    new_opt = np.zeros_like(original_audio)
    processed_len = len(processed)
    og_len = len(original_audio)

    # Handle any potential length mismatch by aligning lengths
    if processed_len != og_len:
        min_len = min(processed_len, og_len)
        original_audio = original_audio[:min_len]
        processed = processed[:min_len]
        new_opt = new_opt[:min_len]
        og_len = min_len

    # Process audio in frames
    for start in range(0, og_len, frame_size):
        end = min(start + frame_size, og_len)
        og_frame = original_audio[start:end]
        proc_frame = processed[start:end]

        # If the original frame is effectively silent, keep it silent
        if np.max(np.abs(og_frame)) < silence_threshold:
            new_opt[start:end] = 0.0
        else:
            og_rms = np.sqrt(np.mean(og_frame ** 2)) + 1e-8
            proc_rms = np.sqrt(np.mean(proc_frame ** 2)) + 1e-8

            # Only scale down if the processed frame's RMS exceeds the original's RMS
            if proc_rms > og_rms:
                scale = og_rms / proc_rms
                new_opt[start:end] = proc_frame * scale
            else:
                new_opt[start:end] = proc_frame

    return new_opt

