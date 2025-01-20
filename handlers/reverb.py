import numpy as np
import soundfile as sf
from pydub import AudioSegment
import json


def read_audio(file_path):
    """
    Reads an audio file (any format supported by pydub) and converts it to a numpy array.
    Normalizes samples to the range -1 to 1.
    """
    audio = AudioSegment.from_file(file_path)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    samples /= np.iinfo(audio.array_type).max  # Normalize to -1..1
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


def extract_reverb(dry_path, wet_path, param_output_path):
    """
    Estimates basic reverb parameters from a 'dry' vocal track
    and a 'wet' (reverb-only) track, then saves them to JSON.

    For demonstration:
      - Pre-delay (seconds)
      - Decay time (approx RT60)
      - Wet level (relative amplitude)
    """
    dry_signal, sr = read_audio(dry_path)
    wet_signal, wet_sr = read_audio(wet_path)

    if sr != wet_sr:
        raise ValueError("Dry and wet sample rates must match.")

    # Truncate to the same length
    min_len = min(len(dry_signal), len(wet_signal))
    dry_signal = dry_signal[:min_len]
    wet_signal = wet_signal[:min_len]

    # --- 1) Estimate Pre-delay ---
    # Find the peak in the dry signal (vocal transient),
    # then see where a similar peak appears in the wet track.
    dry_peak_idx = np.argmax(np.abs(dry_signal))
    wet_peak_idx = np.argmax(np.abs(wet_signal))
    pre_delay_samples = wet_peak_idx - dry_peak_idx
    # If negative or small, clamp to zero
    pre_delay_samples = max(pre_delay_samples, 0)
    pre_delay_sec = pre_delay_samples / sr

    # --- 2) Estimate RT60 (decay time) ---
    # We'll do a crude estimate by looking at the tail's envelope decay.
    #  a) Identify region after the main transient
    tail_start = wet_peak_idx + int(0.05 * sr)  # 50 ms after the wet peak
    if tail_start >= len(wet_signal):
        tail_start = len(wet_signal) - 1
    tail = wet_signal[tail_start:]

    #  b) Compute the envelope (RMS or absolute)
    eps = 1e-10
    env = np.abs(tail) + eps
    env_db = 20 * np.log10(env)

    #  c) We only want to measure the portion that decays (avoid the noise floor).
    #     Let’s find the max in that region, then see how long it takes to drop ~30 dB
    #     from that max (then double it for RT60 if we measure 30 dB).
    max_db = np.max(env_db)
    target_db = max_db - 30.0  # measure time to drop 30 dB
    if target_db < np.min(env_db):
        # If the tail doesn't drop 30 dB from the max, fallback
        decay_time = 0.5
    else:
        # Find where env_db first goes below target_db
        decay_idx = np.where(env_db <= target_db)[0]
        if len(decay_idx) == 0:
            # Didn't find a drop of 30 dB
            decay_time = 0.5
        else:
            t_30 = decay_idx[0] / sr  # seconds from tail_start
            # RT60 ~ 2x the time for 30 dB drop
            decay_time = t_30 * 2.0

    # --- 3) Estimate Wet/Dry ratio ---
    # We can approximate by comparing overall RMS of wet track to dry track (naive).
    dry_rms = np.sqrt(np.mean(dry_signal ** 2) + eps)
    wet_rms = np.sqrt(np.mean(wet_signal ** 2) + eps)
    wet_level = wet_rms / dry_rms  # ratio

    # Package our parameters
    params = {
        "sample_rate": sr,
        "pre_delay": float(pre_delay_sec),
        "decay_time": float(decay_time),
        "wet_level": float(wet_level)
    }

    # Save them
    save_params_to_file(params, param_output_path)
    print(f"Extracted parameters: {params}")
    return param_output_path


def apply_reverb(dry_path, param_path, output_path):
    """
    Loads reverb parameters from JSON, then applies a simple
    feedback delay network (FDN) or similar structure to
    approximate that reverb on a new dry signal.
    """
    dry_signal, sr = read_audio(dry_path)
    params = load_params_from_file(param_path)

    # Unpack parameters
    pd_sec = params["pre_delay"]
    decay_time = params["decay_time"]
    wet_level = params["wet_level"]
    # We'll do a very naive example with a single feedback delay line for demonstration

    # Convert pre-delay to samples
    pd_samples = int(pd_sec * sr)

    # Derive a feedback coefficient from the decay time (rough approximation)
    # Example: feedback = exp(-3 ln(10) / (decay_time * sr))
    # (which is e^(-time_for_60dB_decay / length_in_samples)).
    if decay_time <= 0.01:
        decay_time = 0.01
    feedback = np.exp(-3.0 * np.log(10) / (decay_time * sr))

    # Prepare output array a bit longer for reverb tail
    output_length = len(dry_signal) + pd_samples + int(sr * decay_time * 2)
    out_signal = np.zeros(output_length, dtype=np.float32)

    # A single feedback delay line approach
    delay_line = np.zeros(output_length, dtype=np.float32)

    # Process each sample
    for n in range(len(dry_signal)):
        x = dry_signal[n]
        # The delayed read index
        read_idx = n - pd_samples
        if read_idx >= 0:
            # Output is the sum of the input + feedback-delayed signal
            delayed_val = delay_line[read_idx]
            current_out = x + (feedback * delayed_val)

            # Write to output and delay line
            out_signal[n] += current_out  # direct + reverb
            delay_line[n] = current_out  # store for future feedback
        else:
            # No delay yet
            out_signal[n] += x

    # Mix in a simple wet/dry ratio.
    # We'll interpret 'wet_level' as how much to scale the reverb portion
    # relative to the dry signal. In a more complex scenario, you'd do separate
    # signals for "dry" and "wet" and then combine with a ratio. This is simplified.
    #
    # We'll do: final = dry + wet_level * (out_signal - dry_signal)
    # But we need to align lengths. Let's just do a direct approach:
    dry_extended = np.zeros_like(out_signal, dtype=np.float32)
    dry_extended[:len(dry_signal)] = dry_signal
    final_signal = dry_extended + wet_level * (out_signal - dry_extended)

    # Clip to -1..1
    final_signal = np.clip(final_signal, -1.0, 1.0)

    # Save as a WAV file
    sf.write(output_path, final_signal, sr)
    print(f"Saved reverb-applied file to: {output_path}")
    return output_path
