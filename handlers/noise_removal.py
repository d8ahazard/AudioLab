import numpy as np


def restore_silence(original_audio, processed, silence_threshold=0.001, frame_size=512):
    import numpy as np

    new_opt = np.zeros_like(original_audio)
    processed_len = len(processed)
    og_len = len(original_audio)

    # Fix length mismatch by padding or trimming 'processed'
    if processed_len < og_len:
        processed = np.pad(processed, (0, og_len - processed_len), mode='constant')
    elif processed_len > og_len:
        processed = processed[:og_len]

    for start in range(0, og_len, frame_size):
        end = min(start + frame_size, og_len)
        og_frame = original_audio[start:end]
        proc_frame = processed[start:end]

        if np.max(np.abs(og_frame)) < silence_threshold:
            new_opt[start:end] = 0.0
        # else:
        #     og_rms = np.sqrt(np.mean(og_frame ** 2)) + 1e-8
        #     proc_rms = np.sqrt(np.mean(proc_frame ** 2)) + 1e-8
        #     scale = og_rms / proc_rms
        #     new_opt[start:end] = proc_frame * scale

    return new_opt
