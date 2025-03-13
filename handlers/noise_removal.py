import librosa
import numpy as np
import pyloudnorm as pyln  # added import for pyloudnorm


def restore_silence(original_audio, cloned_audio, sr_original, sr_cloned,
                    silence_threshold=0.002, window_size=1024, hop_length=512):
    """
    Restores the amplitude profile of the original audio to the cloned audio.
    Preserves silence regions from the original audio and matches volume levels.
    
    Args:
        original_audio: The original audio numpy array (can be mono or stereo)
        cloned_audio: The cloned/processed audio numpy array
        sr_original: Sample rate of the original audio
        sr_cloned: Sample rate of the cloned audio
        silence_threshold: RMS threshold below which audio is considered silent
        window_size: Size of the analysis window in samples
        hop_length: Hop length between windows in samples
    
    Returns:
        Processed audio with restored silence and matched volume
    """
    # Ensure consistent dimensions
    if original_audio.ndim == 1:
        original_audio = original_audio.reshape(-1, 1)
    if cloned_audio.ndim == 1:
        cloned_audio = cloned_audio.reshape(-1, 1)
    
    # Resample if needed
    if sr_cloned != sr_original:
        cloned_audio = librosa.resample(
            cloned_audio.T, orig_sr=sr_cloned, target_sr=sr_original
        ).T
    
    # Match lengths
    target_length = original_audio.shape[0]
    if cloned_audio.shape[0] > target_length:
        cloned_audio = cloned_audio[:target_length]
    else:
        pad_width = ((0, target_length - cloned_audio.shape[0]), (0, 0))
        cloned_audio = np.pad(cloned_audio, pad_width, mode='constant')
    
    # Create output array
    result = np.zeros_like(cloned_audio, dtype=np.float32)
    
    # Get number of channels
    n_channels = original_audio.shape[1]
    
    # Create window function for analysis
    window = np.hanning(window_size)
    
    # Process each channel
    for ch in range(n_channels):
        orig_channel = original_audio[:, ch]
        clone_channel = cloned_audio[:, ch]
        
        # Calculate frames
        n_frames = 1 + (len(orig_channel) - window_size) // hop_length
        
        # Create envelope arrays
        orig_env = np.zeros(len(orig_channel))
        gain_env = np.zeros(len(orig_channel))
        mask_env = np.zeros(len(orig_channel))
        
        # Process each frame
        for i in range(n_frames):
            start = i * hop_length
            end = start + window_size
            
            if end > len(orig_channel):
                break
                
            # Get frame data
            orig_frame = orig_channel[start:end] * window
            clone_frame = clone_channel[start:end] * window
            
            # Calculate RMS
            orig_rms = np.sqrt(np.mean(orig_frame ** 2) + 1e-8)
            clone_rms = np.sqrt(np.mean(clone_frame ** 2) + 1e-8)
            
            # Determine if this is a silent region
            is_silent = orig_rms < silence_threshold
            
            # Calculate gain factor (1.0 = no change)
            gain = 1.0 if is_silent else min(orig_rms / (clone_rms + 1e-8), 10.0)
            
            # Fill envelope at this position
            orig_env[start:end] += orig_rms * window
            gain_env[start:end] += gain * window
            mask_env[start:end] += (0.0 if is_silent else 1.0) * window
        
        # Normalize the envelope weights
        weight_sum = np.zeros_like(orig_env)
        for i in range(n_frames):
            start = i * hop_length
            end = start + window_size
            if end > len(weight_sum):
                break
            weight_sum[start:end] += window
        
        # Avoid division by zero
        nonzero_indices = weight_sum > 1e-8
        orig_env[nonzero_indices] /= weight_sum[nonzero_indices]
        gain_env[nonzero_indices] /= weight_sum[nonzero_indices]
        mask_env[nonzero_indices] /= weight_sum[nonzero_indices]
        
        # Apply gain and silence mask
        result[:, ch] = clone_channel * gain_env * mask_env
        
    # Apply final loudness matching
    try:
        # Global loudness normalization as a final step
        meter = pyln.Meter(sr_original)
        orig_loudness = meter.integrated_loudness(original_audio)
        if orig_loudness < -70:  # If original is basically silent
            pass  # Skip loudness normalization
        else:
            proc_loudness = meter.integrated_loudness(result)
            if proc_loudness < -70:  # Very quiet processed audio
                pass  # Skip to avoid extreme gain
            else:
                result = pyln.normalize.loudness(result, proc_loudness, orig_loudness)
    except Exception as e:
        # Fallback if pyloudnorm fails
        orig_rms = np.sqrt(np.mean(original_audio**2))
        result_rms = np.sqrt(np.mean(result**2))
        if result_rms > 1e-8:  # Avoid division by zero
            result = result * (orig_rms / result_rms)
    
    # Safety limit to prevent clipping
    peak = np.max(np.abs(result))
    if peak > 0.98:
        result = result * (0.98 / peak)
        
    return result
