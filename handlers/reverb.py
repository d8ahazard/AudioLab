import json

import numpy as np
import soundfile as sf
from pydub import AudioSegment

################################################################################
# GLOBAL OPTIONS
################################################################################

# If you'd like some crossfeed between channels in apply_reverb, adjust this:
CROSS_FEED = 0.3  # 0.0 means no crossfeed; 0.3+ is partial blending


################################################################################
# AUDIO I/O
################################################################################

def read_audio(file_path):
    """
    Reads an audio file (any format supported by pydub) into a numpy array of
    shape (num_frames, num_channels), normalized to -1..1.
    """
    audio = AudioSegment.from_file(file_path)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)

    num_channels = audio.channels
    if num_channels > 1:
        # Reshape from (num_frames * num_channels,) to (num_frames, num_channels)
        samples = samples.reshape((-1, num_channels))

    # Normalize to -1..1 (same for mono or multi-channel)
    samples /= np.iinfo(audio.array_type).max
    return samples, audio.frame_rate


def save_params_to_file(params, output_path):
    """
    Saves reverb parameters (dictionary) to a JSON file.
    """
    with open(output_path, "w") as file:
        json.dump(params, file, indent=2)


def load_params_from_file(param_path):
    """
    Loads reverb parameters (dictionary) from a JSON file.
    """
    with open(param_path, "r") as file:
        params = json.load(file)
    return params


################################################################################
# EXTRACTION
################################################################################

def extract_reverb(
        dry_path,
        wet_path,
        param_output_path,
        tail_start_offset=0.05,
        decay_drop_db=30.0,
        fallback_decay_time=0.5,
        min_decay_time=0.01
):
    """
    Estimates basic reverb parameters from a 'dry' vocal track and a 'wet' (reverb-only) track.
    Uses an FFT-based cross-correlation for faster alignment on large files.
    Includes partial-drop measurement to avoid overly long decay times for small-room reverbs.
    """
    import numpy as np

    # -------------------------------------------------------------------------
    # HELPER: FFT-based cross-correlation
    # -------------------------------------------------------------------------
    def fft_xcorr(a, b):
        """
        Returns the full cross-correlation array of signals a and b using an FFT-based approach.
        Correlation length = len(a) + len(b) - 1.
        """
        N = len(a) + len(b) - 1
        # Next power-of-two size for zero-padding
        N_fft = 1 << (N - 1).bit_length()

        A = np.fft.rfft(a, n=N_fft)
        B = np.fft.rfft(b, n=N_fft)
        corr_full = np.fft.irfft(A * np.conjugate(B), n=N_fft)
        # Truncate to the valid "full" length
        return corr_full[:N]

    # -------------------------------------------------------------------------
    # READ & TRUNCATE
    # -------------------------------------------------------------------------
    dry_signal, sr = read_audio(dry_path)
    wet_signal, wet_sr = read_audio(wet_path)
    if sr != wet_sr:
        raise ValueError("Dry and wet sample rates must match.")

    min_len = min(len(dry_signal), len(wet_signal))
    dry_signal = dry_signal[:min_len]
    wet_signal = wet_signal[:min_len]

    # -------------------------------------------------------------------------
    # STEP 1: Find pre-delay using cross-correlation in mono (FFT-based)
    # -------------------------------------------------------------------------
    def to_mono(x):
        return np.mean(x, axis=1) if x.ndim == 2 else x

    dry_mono = to_mono(dry_signal)
    wet_mono = to_mono(wet_signal)

    # FFT cross-correlation
    corr = fft_xcorr(wet_mono, dry_mono)
    # best_shift is argmax(corr) - (len(dry_mono)-1)
    best_shift = np.argmax(corr) - (len(dry_mono) - 1)
    if best_shift < 0:
        best_shift = 0
    pre_delay_samples = best_shift
    pre_delay_sec = pre_delay_samples / sr

    # -------------------------------------------------------------------------
    # STEP 2: Measure the tail for decay time
    # -------------------------------------------------------------------------
    tail_start = pre_delay_samples + int(tail_start_offset * sr)
    if tail_start >= len(wet_signal):
        tail_start = len(wet_signal) - 1

    tail = wet_signal[tail_start:]
    eps = 1e-10

    # amplitude envelope (RMS across channels)
    tail_energy = np.sqrt(np.sum(tail ** 2, axis=1)) + eps
    env_db = 20.0 * np.log10(tail_energy)

    max_db = np.max(env_db)
    min_db = np.min(env_db)
    actual_drop_db = max_db - min_db

    if actual_drop_db < 2.0:
        # Very little decay found; treat it as minimal
        decay_time = min_decay_time
    else:
        target_db = max_db - decay_drop_db
        if target_db < min_db:
            # The tail never drops that far -> measure partial drop
            idx_min = np.argmin(env_db)
            t_drop = idx_min / sr  # time from tail_start
            decay_time = t_drop * (60.0 / actual_drop_db)
            if decay_time < min_decay_time:
                decay_time = min_decay_time
        else:
            # We can measure the time it takes to reach target_db
            decay_idx = np.where(env_db <= target_db)[0]
            if len(decay_idx) == 0:
                # Could not find a point below target_db
                decay_time = fallback_decay_time
            else:
                t_drop = decay_idx[0] / sr
                decay_time = t_drop * (60.0 / decay_drop_db)

    decay_time = max(decay_time, min_decay_time)

    # -------------------------------------------------------------------------
    # STEP 3: Estimate Wet/Dry ratio
    # -------------------------------------------------------------------------
    dry_rms = np.sqrt(np.mean(dry_signal ** 2) + eps)
    wet_rms = np.sqrt(np.mean(wet_signal ** 2) + eps)
    wet_level = wet_rms / dry_rms

    # -------------------------------------------------------------------------
    # STEP 4: Save the parameters
    # -------------------------------------------------------------------------
    params = {
        "sample_rate": sr,
        "pre_delay": float(pre_delay_sec),
        "decay_time": float(decay_time),
        "wet_level": float(wet_level),
        "config": {
            "tail_start_offset": tail_start_offset,
            "decay_drop_db": decay_drop_db,
            "fallback_decay_time": fallback_decay_time,
            "min_decay_time": min_decay_time
        }
    }

    save_params_to_file(params, param_output_path)
    print(f"Extracted parameters saved to {param_output_path}: {params}")
    return param_output_path


################################################################################
# APPLY REVERB (MULTI-CHANNEL)
################################################################################

def apply_reverb(dry_path, param_path, output_path):
    """
    Loads reverb parameters from JSON, then applies a simple multi-channel feedback
    delay line with optional crossfeed to approximate that reverb.

    - If CROSS_FEED is 0.0, each channel is processed independently.
    - If CROSS_FEED > 0, each channel's feedback will include a fraction
      of the other channels' delayed signals (for a "stereo" spread).
    """
    dry_signal, sr = read_audio(dry_path)
    params = load_params_from_file(param_path)

    pd_sec = params["pre_delay"]
    decay_time = params["decay_time"]
    wet_level = params["wet_level"]

    if decay_time <= 0.01:
        decay_time = 0.01

    pd_samples = int(pd_sec * sr)
    feedback = np.exp(-3.0 * np.log(10) / (decay_time * sr))

    num_in_frames, num_channels = dry_signal.shape
    tail_length = int(sr * decay_time * 2)
    output_length = num_in_frames + pd_samples + tail_length

    out_signal = np.zeros((output_length, num_channels), dtype=np.float32)
    delay_line = np.zeros((output_length, num_channels), dtype=np.float32)

    for n in range(num_in_frames):
        for c in range(num_channels):
            x = dry_signal[n, c]
            read_idx = n - pd_samples
            if read_idx >= 0:
                sum_delayed = delay_line[read_idx, c]
                for c2 in range(num_channels):
                    if c2 != c:
                        sum_delayed += CROSS_FEED * delay_line[read_idx, c2]

                current_out = x + feedback * sum_delayed
                out_signal[n, c] += current_out
                delay_line[n, c] = current_out
            else:
                out_signal[n, c] += x

    # Construct final signal with wet/dry mix
    dry_extended = np.zeros((output_length, num_channels), dtype=np.float32)
    dry_extended[:num_in_frames, :] = dry_signal
    final_signal = dry_extended + wet_level * (out_signal - dry_extended)

    # Clip to -1..1
    final_signal = np.clip(final_signal, -1.0, 1.0)

    sf.write(output_path, final_signal, sr)
    print(f"Saved reverb-applied file to: {output_path}")
    return output_path


################################################################################
# TEST EXTRACTION
################################################################################

def test_extraction(isolated_vocal_path, isolated_reverb_path, original_wet_path, output_dir):
    """
    Runs multiple reverb-parameter extractions using the separated reverb track (wet-only),
    applies those parameters to the isolated vocal track, and compares the
    final generated wet track to the original wet track (vocals + reverb together).

    This way:
      - Extraction is based on the pure 'isolated_reverb' signal (best for analyzing reverb).
      - We measure MSE between the newly created wet mix and the original wet mix.
    """
    import os
    import numpy as np

    # Limit how many seconds of audio we compare for MSE (just for speed)
    MAX_COMPARE_DURATION_SEC = 5.0

    # We'll sweep through a few parameter configurations
    param_sweep = [
        {"tail_start_offset": 0.03, "decay_drop_db": 20.0, "fallback_decay_time": 0.3, "min_decay_time": 0.01},
        {"tail_start_offset": 0.05, "decay_drop_db": 30.0, "fallback_decay_time": 0.4, "min_decay_time": 0.01},
        {"tail_start_offset": 0.08, "decay_drop_db": 40.0, "fallback_decay_time": 0.5, "min_decay_time": 0.01},
        {"tail_start_offset": 0.10, "decay_drop_db": 35.0, "fallback_decay_time": 0.6, "min_decay_time": 0.01},
    ]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1) Read the isolated vocal track (dry), the separated reverb track, and the original wet track
    isolated_vocal, sr_vocal = read_audio(isolated_vocal_path)
    isolated_reverb, sr_reverb = read_audio(isolated_reverb_path)
    original_wet, sr_orig = read_audio(original_wet_path)

    # All must share the same sample rate
    if not (sr_vocal == sr_reverb == sr_orig):
        raise ValueError("Sample rates do not match among vocal, separated reverb, and original wet track.")

    # Truncate them all to the same minimum length (just in case)
    min_len = min(len(isolated_vocal), len(isolated_reverb), len(original_wet))
    isolated_vocal = isolated_vocal[:min_len]
    isolated_reverb = isolated_reverb[:min_len]
    original_wet = original_wet[:min_len]

    results = []

    # 2) For each set of parameters in our sweep:
    for i, config in enumerate(param_sweep):
        param_file = os.path.join(output_dir, f"params_{i}.json")

        # Extract reverb parameters from the DRY + WET-ONLY pair
        extract_reverb(
            dry_path=isolated_vocal_path,
            wet_path=isolated_reverb_path,
            param_output_path=param_file,
            tail_start_offset=config["tail_start_offset"],
            decay_drop_db=config["decay_drop_db"],
            fallback_decay_time=config["fallback_decay_time"],
            min_decay_time=config["min_decay_time"]
        )

        # 3) Apply reverb to the isolated vocal
        out_file = os.path.join(output_dir, f"test_reverb_{i}.wav")
        apply_reverb(isolated_vocal_path, param_file, out_file)

        # 4) Read the newly generated wet track and compare to the ORIGINAL wet track
        generated_wet, sr_gen = read_audio(out_file)
        if sr_gen != sr_vocal:
            raise ValueError("Generated track sample rate differs from the original vocal rate.")

        # Truncate to match the original wet track
        max_len = min(len(generated_wet), len(original_wet))
        generated_wet = generated_wet[:max_len]
        reference_wet = original_wet[:max_len]

        # Optionally limit the comparison to a certain duration for speed
        compare_len = int(MAX_COMPARE_DURATION_SEC * sr_vocal)
        compare_len = min(compare_len, max_len)
        generated_wet_seg = generated_wet[:compare_len]
        reference_wet_seg = reference_wet[:compare_len]

        # Convert to mono for cross-correlation + MSE
        def to_mono(x):
            return np.mean(x, axis=1) if x.ndim == 2 else x

        gen_mono = to_mono(generated_wet_seg)
        ref_mono = to_mono(reference_wet_seg)

        # If there's almost no data, just do raw MSE
        if len(gen_mono) < 2 or len(ref_mono) < 2:
            mse = np.mean((gen_mono - ref_mono[:len(gen_mono)]) ** 2)
        else:
            # Cross-correlate to align them (small alignment differences)
            corr = np.correlate(gen_mono, ref_mono, mode='full')
            best_shift = np.argmax(corr) - (len(ref_mono) - 1)

            if best_shift > 0:
                gen_aligned = gen_mono[best_shift:]
                ref_aligned = ref_mono[:len(gen_aligned)]
            else:
                shift_abs = abs(best_shift)
                ref_aligned = ref_mono[shift_abs:]
                gen_aligned = gen_mono[:len(ref_aligned)]

            mse = np.mean((gen_aligned - ref_aligned) ** 2)

        results.append((config, mse))
        print(f"Config {i}: {config}, MSE={mse:.6f} (full wet comparison)")

    # 5) Pick the best config
    best_config, best_mse = min(results, key=lambda x: x[1])
    print("\n=== Test Extraction Complete ===")
    print(f"Best config: {best_config} with MSE={best_mse:.6f}")
    print("All results:")
    for config, mse in results:
        print(f"  {config} => MSE={mse:.6f}")
