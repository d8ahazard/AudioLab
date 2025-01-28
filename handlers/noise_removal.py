import numpy as np


def restore_silence(original_audio, processed, silence_threshold=0.001, frame_size=512):
    """
    Restores silence from the original audio to the processed audio.

    Args:
        original_audio (numpy.ndarray): The original audio array.
        processed (numpy.ndarray): The processed audio array.
        silence_threshold (float): Amplitude below which a frame is considered silence.
        frame_size (int): Number of samples per frame for silence detection.

    Returns:
        numpy.ndarray: The processed audio with silences restored.
    """
    new_opt = np.zeros_like(original_audio)
    processed_len = len(processed)
    og_len = len(original_audio)

    print(f"Processed len: {processed_len}, Original len: {og_len}")

    # Handle any potential length mismatch by aligning lengths
    if processed_len != og_len:
        min_len = min(processed_len, og_len)
        original_audio = original_audio[:min_len]
        processed = processed[:min_len]

    # Process audio in frames
    for start in range(0, og_len, frame_size):
        end = min(start + frame_size, og_len)
        og_frame = original_audio[start:end]

        # Check if the original frame is silence
        if np.max(np.abs(og_frame)) < silence_threshold:
            # If silent, zero out the processed frame
            new_opt[start:end] = 0
        else:
            # Otherwise, use the processed audio
            new_opt[start:end] = processed[start:end]

    return new_opt
