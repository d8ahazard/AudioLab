from scipy.optimize import curve_fit
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import soundfile as sf
from audio_separator.separator import Separator
from pydub import AudioSegment
from scipy.signal import fftconvolve

from handlers.config import output_path

logger = logging.getLogger(__name__)


################################################################################
# AUDIO I/O
################################################################################

def read_audio(file_path):
    """
    Reads an audio file into a numpy array (normalized to -1..1).
    """
    logger.info(f"Reading audio file: {file_path}")
    audio = AudioSegment.from_file(file_path)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)

    num_channels = audio.channels
    if num_channels > 1:
        samples = samples.reshape((-1, num_channels))

    samples /= np.iinfo(audio.array_type).max
    return samples, audio.frame_rate


def save_params_to_file(params, output_path):
    with open(output_path, "w") as file:
        json.dump(params, file, indent=2)
    logger.info(f"Saved parameters to file: {output_path}")


def load_params_from_file(param_path):
    with open(param_path, "r") as file:
        return json.load(file)


################################################################################
# HELPER FUNCTIONS
################################################################################

def to_mono(signal):
    return np.mean(signal, axis=1) if signal.ndim == 2 else signal


def fft_xcorr(a, b):
    """
    Computes cross-correlation using FFT for fast signal alignment.
    """
    N = len(a) + len(b) - 1
    N_fft = 1 << (N - 1).bit_length()

    A = np.fft.rfft(a, n=N_fft)
    B = np.fft.rfft(b, n=N_fft)
    corr_full = np.fft.irfft(A * np.conjugate(B), n=N_fft)

    return corr_full[:N]


def estimate_rt60(signal, sr, curve_fit_maxfev=5000):
    """
    Estimates RT60 decay time by fitting an exponential decay to the RMS envelope.
    Now accepts a curve_fit_maxfev parameter for additional flexibility.
    """
    eps = 1e-10
    if signal.ndim == 2:
        env = np.sqrt(np.sum(signal ** 2, axis=1)) + eps
    else:
        env = np.abs(signal) + eps

    env_db = 20.0 * np.log10(env)
    time = np.linspace(0, len(env) / sr, len(env))

    def exp_decay(x, a, b, c):
        return a * np.exp(-b * x) + c

    popt, _ = curve_fit(exp_decay, time, env_db, maxfev=curve_fit_maxfev)
    decay_time = 3.0 / popt[1] if popt[1] != 0 else 0.5
    logger.info(f"Estimated decay time: {decay_time} sec (using maxfev={curve_fit_maxfev})")
    return max(decay_time, 0.01)


def wiener_deconvolution(signal, filter_kernel, epsilon=1e-6):
    """
    Performs Wiener deconvolution for better stability.
    """
    H = np.fft.rfft(filter_kernel, len(signal))
    Y = np.fft.rfft(signal)
    H_conj = np.conjugate(H)

    power_H = np.abs(H) ** 2
    deconvolved = np.fft.irfft((H_conj * Y) / (power_H + epsilon))

    return deconvolved


################################################################################
# EXTRACT REVERB PARAMETERS
################################################################################

def extract_reverb(dry_path, wet_path, param_output_path, wiener_epsilon=1e-6, curve_fit_maxfev=5000):
    """
    Extracts impulse response and reverb parameters using Wiener Deconvolution.
    Additional extraction settings can be passed to tune the extraction process.
    """
    logger.info(f"Extracting reverb from dry: {dry_path} and wet: {wet_path}")
    dry_signal, sr = read_audio(dry_path)
    wet_signal, wet_sr = read_audio(wet_path)

    if sr != wet_sr:
        raise ValueError("Dry and wet sample rates must match.")

    dry_mono = to_mono(dry_signal)
    wet_mono = to_mono(wet_signal)

    # Estimate pre-delay using FFT cross-correlation
    logger.info("Estimating pre-delay...")
    corr = fft_xcorr(wet_mono, dry_mono)
    best_shift = np.argmax(corr) - (len(dry_mono) - 1)
    best_shift = max(best_shift, 0)
    pre_delay_sec = best_shift / sr
    logger.info(f"Estimated pre-delay: {pre_delay_sec} sec")

    # Estimate RT60 decay time with adjustable curve_fit_maxfev
    logger.info("Estimating RT60 decay time...")
    decay_time = estimate_rt60(wet_signal, sr, curve_fit_maxfev=curve_fit_maxfev)

    # Extract impulse response using Wiener deconvolution with adjustable epsilon
    logger.info(f"Performing Wiener deconvolution (epsilon={wiener_epsilon})...")
    impulse_response = wiener_deconvolution(wet_mono, dry_mono, epsilon=wiener_epsilon)
    impulse_response = impulse_response[:int(sr * 2)]  # Limit IR length to 2 sec

    # Additional parameters:
    early_samples = int(0.05 * sr)
    early_energy = np.sum(np.square(impulse_response[:early_samples]))
    total_energy = np.sum(np.square(impulse_response)) + 1e-10
    early_reflection_ratio = early_energy / total_energy
    late_energy = total_energy - early_energy
    late_reverb_ratio = late_energy / total_energy

    envelope = np.abs(impulse_response)
    diffusion = float(np.var(envelope))

    fft_ir = np.abs(np.fft.rfft(impulse_response))
    freqs = np.fft.rfftfreq(len(impulse_response), d=1.0 / sr)
    spectral_centroid = float(np.sum(freqs * fft_ir) / (np.sum(fft_ir) + 1e-10))

    params = {
        "sample_rate": sr,
        "pre_delay": float(pre_delay_sec),
        "decay_time": float(decay_time),
        "early_reflection_ratio": early_reflection_ratio,
        "late_reverb_ratio": late_reverb_ratio,
        "diffusion": diffusion,
        "spectral_centroid": spectral_centroid,
        "impulse_response": impulse_response.tolist()
    }

    save_params_to_file(params, param_output_path)
    logger.info(f"Extracted parameters saved: {param_output_path}")
    return param_output_path


################################################################################
# APPLY REVERB USING FFT CONVOLUTION
################################################################################

def apply_reverb(dry_path, param_path, output_path):
    """
    Applies convolution reverb using extracted impulse response while maintaining stereo.
    """
    dry_signal, sr = read_audio(dry_path)
    params = load_params_from_file(param_path)

    impulse_response = np.array(params["impulse_response"])
    pre_delay_samples = int(params["pre_delay"] * sr)

    # Check if dry_signal is stereo
    num_channels = 1 if dry_signal.ndim == 1 else dry_signal.shape[1]

    if num_channels > 1:
        wet_signals = []
        for ch in range(num_channels):
            ch_dry = dry_signal[:, ch]
            ch_wet = fftconvolve(ch_dry, impulse_response, mode='full')
            ch_wet = np.pad(ch_wet, (pre_delay_samples, 0))
            wet_signals.append(ch_wet[:len(ch_dry)])
        wet_signal = np.stack(wet_signals, axis=1)
    else:
        dry_signal = to_mono(dry_signal)
        wet_signal = fftconvolve(dry_signal, impulse_response, mode='full')
        wet_signal = np.pad(wet_signal, (pre_delay_samples, 0))[:len(dry_signal)]

    final_signal = dry_signal + 0.7 * wet_signal
    final_signal = np.clip(final_signal, -1.0, 1.0)

    sf.write(output_path, final_signal, sr)
    return output_path


################################################################################
# PROCESS SONG
################################################################################

def process_song(dry_path, wet_path, output_dir):
    """
    Runs extraction and reverb application on a full song.
    """
    param_file = os.path.join(output_dir, "reverb_params.json")
    output_file = os.path.join(output_dir, "reverb_applied.wav")

    extract_reverb(dry_path, wet_path, param_file)
    apply_reverb(dry_path, param_file, output_file)

    return output_file


def generate_ir(sr, pre_delay, decay_time, early_reflection_ratio, diffusion, spectral_centroid, length=2.0):
    """
    Generates a synthetic impulse response for reverb simulation.

    This version creates a more realistic IR by:
      - Adding a pre-delay.
      - Generating an early reflection section (with several random impulses)
        whose energy is normalized to match early_reflection_ratio.
      - Creating an exponential decay tail (with added noise for diffusion)
        that is low-pass filtered to roughly adjust the spectral centroid.
      - The tail energy is normalized to be (1 - early_reflection_ratio).

    Args:
        sr (int): Sample rate.
        pre_delay (float): Pre-delay in seconds.
        decay_time (float): Decay time for the exponential tail.
        early_reflection_ratio (float): Fraction of total energy in early reflections.
        diffusion (float): Noise amplitude factor for diffusion.
        spectral_centroid (float): Desired spectral centroid (Hz) of the IR.
        length (float): Total length of the impulse response in seconds.

    Returns:
        np.ndarray: The generated impulse response.
    """
    total_samples = int(sr * length)
    pre_delay_samples = int(pre_delay * sr)
    early_length_samples = int(0.05 * sr)  # 50ms early reflection window
    tail_length_samples = total_samples - pre_delay_samples - early_length_samples

    ir = np.zeros(total_samples, dtype=np.float32)

    # --- Early Reflections ---
    n_impulses = np.random.randint(3, 7)
    early_ir = np.zeros(early_length_samples, dtype=np.float32)
    for _ in range(n_impulses):
        pos = np.random.randint(0, early_length_samples)
        amp = np.random.uniform(0.1, 1.0)
        early_ir[pos] += amp
    # Normalize early_ir energy to match early_reflection_ratio
    current_energy = np.sum(early_ir ** 2)
    if current_energy > 0:
        scale = np.sqrt(early_reflection_ratio / current_energy)
        early_ir *= scale
    ir[pre_delay_samples:pre_delay_samples + early_length_samples] = early_ir

    # --- Late Reverb Tail ---
    t = np.linspace(0, tail_length_samples / sr, tail_length_samples, endpoint=False)
    tail_ir = np.exp(-t / decay_time)
    # Add diffusion noise
    noise = diffusion * np.random.randn(tail_length_samples)
    tail_ir += noise
    # Apply a simple lowpass filter to roughly adjust spectral centroid.
    # (This is an approximation: higher 'spectral_centroid' means less lowpass filtering.)
    alpha = max(0, (spectral_centroid - 4000) / 10000)
    if alpha > 0:
        kernel_size = max(1, int(sr * 0.005))  # 5ms kernel
        kernel = np.exp(-np.linspace(0, kernel_size - 1, kernel_size) / (alpha * kernel_size))
        kernel /= np.sum(kernel)
        tail_ir = np.convolve(tail_ir, kernel, mode='same')
    # Normalize tail energy to be (1 - early_reflection_ratio)
    tail_energy = np.sum(tail_ir ** 2)
    if tail_energy > 0:
        scale_tail = np.sqrt((1 - early_reflection_ratio) / tail_energy)
        tail_ir *= scale_tail
    ir[pre_delay_samples + early_length_samples:] = tail_ir

    # Normalize IR amplitude (optional)
    max_amp = np.max(np.abs(ir))
    if max_amp > 0:
        ir /= max_amp

    return ir


################################################################################
# TESTING
################################################################################

def test_reverb(dry_file):
    """
    Tests the reverb extraction process by simulating various reverb conditions.

    Workflow:
      1. Apply all of the reverb settings (with realistic parameters) to the test track.
      2. Separate all of the simulated tracks.
      3. For each setting, iterate over a set of extraction parameter candidates
         (different Wiener epsilon and curve_fit settings) until the best extraction
         (lowest error relative to ground truth for all parameters) is found.
         On each iteration, the candidate grid is updated around the best candidate
         from the previous iteration.
      4. Save the results and extraction settings to a JSON file.

    The simulation now uses the following reverb parameters:
      - pre_delay, decay_time, early_reflection_ratio, diffusion, and spectral_centroid.
    The late_reverb_ratio is expected to be ~ (1 - early_reflection_ratio).

    Args:
        dry_file (str): Path to the dry input track.

    Returns:
        list: A list of dictionaries with the results for each reverb setting.
    """
    logger.info("Starting reverb simulation for all settings.")
    reverb_settings = [
        {"name": "small_room", "pre_delay": 0.005, "decay_time": 0.6,
         "early_reflection_ratio": 0.3, "diffusion": 0.005, "spectral_centroid": 5000},
        {"name": "large_room", "pre_delay": 0.02, "decay_time": 1.2,
         "early_reflection_ratio": 0.4, "diffusion": 0.01, "spectral_centroid": 4000},
        {"name": "delay", "pre_delay": 0.1, "decay_time": 0.8,
         "early_reflection_ratio": 0.2, "diffusion": 0.007, "spectral_centroid": 5500},
        {"name": "concert_hall", "pre_delay": 0.03, "decay_time": 2.5,
         "early_reflection_ratio": 0.5, "diffusion": 0.015, "spectral_centroid": 3000},
        {"name": "plate", "pre_delay": 0.0, "decay_time": 1.0,
         "early_reflection_ratio": 0.25, "diffusion": 0.008, "spectral_centroid": 6000},
    ]

    test_output_dir = os.path.join(output_path, "temp_test_reverb")
    os.makedirs(test_output_dir, exist_ok=True)

    dry_signal, sr = read_audio(dry_file)
    simulated_files = {}

    # 1. Apply all of the reverb settings to generate simulated tracks.
    for setting in reverb_settings:
        logger.info(f"Simulating reverb for setting: {setting['name']}")
        ir = generate_ir(sr, pre_delay=setting["pre_delay"],
                         decay_time=setting["decay_time"],
                         early_reflection_ratio=setting["early_reflection_ratio"],
                         diffusion=setting["diffusion"],
                         spectral_centroid=setting["spectral_centroid"],
                         length=2.0)
        if dry_signal.ndim == 1:
            convolved = fftconvolve(dry_signal, ir, mode='full')
            convolved = convolved[:len(dry_signal)]
        else:
            channels = []
            for ch in range(dry_signal.shape[1]):
                conv_ch = fftconvolve(dry_signal[:, ch], ir, mode='full')
                conv_ch = conv_ch[:len(dry_signal)]
                channels.append(conv_ch)
            convolved = np.stack(channels, axis=1)
        simulated_track = dry_signal + 0.7 * convolved
        simulated_track = np.clip(simulated_track, -1.0, 1.0)
        simulated_file = os.path.join(test_output_dir, f"{setting['name']}_simulated.wav")
        sf.write(simulated_file, simulated_track, sr)
        simulated_files[setting["name"]] = simulated_file
        logger.info(f"Simulated track for setting {setting['name']} saved: {simulated_file}")

    # 2. Separate all of the simulated tracks.
    logger.info("Simulated tracks generated. Starting separation for all tracks.")
    separated_files = {}
    for setting in reverb_settings:
        simulated_file = simulated_files[setting["name"]]
        separator = Separator(output_dir=test_output_dir)
        separator.load_model("Reverb_HQ_By_FoxJoy.onnx")
        separated = separator.separate(simulated_file)
        dry_sep = None
        wet_sep = None
        for f in separated:
            full_path = os.path.join(test_output_dir, f)
            if "No Reverb" in f:
                dry_sep = full_path
            else:
                wet_sep = full_path
        if not dry_sep or not wet_sep:
            logger.error(f"Separation failed for setting {setting['name']}")
            continue
        separated_files[setting["name"]] = {"dry": dry_sep, "wet": wet_sep}
        logger.info(f"Separation for setting {setting['name']} completed: Dry: {dry_sep}, Wet: {wet_sep}")

    # 3. Iterate over extraction settings for each separated track.
    # Start with an initial candidate grid.
    initial_candidates = [
        {"wiener_epsilon": 1e-6, "curve_fit_maxfev": 5000},
        {"wiener_epsilon": 1e-6, "curve_fit_maxfev": 10000},
        {"wiener_epsilon": 1e-5, "curve_fit_maxfev": 5000},
        {"wiener_epsilon": 1e-5, "curve_fit_maxfev": 10000},
        {"wiener_epsilon": 1e-7, "curve_fit_maxfev": 5000},
    ]
    max_iterations = 10
    tolerance = 0.05  # overall tolerance for total relative error

    results = []
    for setting in reverb_settings:
        if setting["name"] not in separated_files:
            continue
        sep_files = separated_files[setting["name"]]
        logger.info(f"Starting extraction parameter optimization for setting {setting['name']}")

        # Initialize candidate grid and global best values.
        current_candidates = initial_candidates
        global_best_error = float("inf")
        global_best_candidate = None
        global_best_extracted_params = None
        best_iteration = 0
        iteration = 0

        while iteration < max_iterations:
            iteration_best_error = float("inf")
            iteration_best_candidate = None
            iteration_best_extracted_params = None

            for idx, candidate in enumerate(current_candidates):
                param_file = os.path.join(test_output_dir, f"{setting['name']}_iter{iteration}_cand{idx}.json")
                extract_reverb(sep_files["dry"], sep_files["wet"], param_file,
                               wiener_epsilon=candidate["wiener_epsilon"],
                               curve_fit_maxfev=candidate["curve_fit_maxfev"])
                extracted_params = load_params_from_file(param_file)

                # Compute relative errors for all parameters.
                err_pre_delay = abs(extracted_params["pre_delay"] - setting["pre_delay"])
                rel_err_pre_delay = err_pre_delay / (setting["pre_delay"] if setting["pre_delay"] > 0 else 1)

                err_decay = abs(extracted_params["decay_time"] - setting["decay_time"])
                rel_err_decay = err_decay / setting["decay_time"]

                err_early = abs(extracted_params["early_reflection_ratio"] - setting["early_reflection_ratio"])
                rel_err_early = err_early / (
                    setting["early_reflection_ratio"] if setting["early_reflection_ratio"] > 0 else 1)

                expected_late = 1 - setting["early_reflection_ratio"]
                err_late = abs(extracted_params["late_reverb_ratio"] - expected_late)
                rel_err_late = err_late / (expected_late if expected_late > 0 else 1)

                err_diffusion = abs(extracted_params["diffusion"] - setting["diffusion"])
                rel_err_diffusion = err_diffusion / (setting["diffusion"] if setting["diffusion"] > 0 else 1)

                err_centroid = abs(extracted_params["spectral_centroid"] - setting["spectral_centroid"])
                rel_err_centroid = err_centroid / (
                    setting["spectral_centroid"] if setting["spectral_centroid"] > 0 else 1)

                total_error = (rel_err_pre_delay + rel_err_decay + rel_err_early +
                               rel_err_late + rel_err_diffusion + rel_err_centroid)
                logger.info(f"Setting {setting['name']}, Iteration {iteration}, Candidate {idx}: "
                            f"wiener_epsilon={candidate['wiener_epsilon']}, curve_fit_maxfev={candidate['curve_fit_maxfev']} -> "
                            f"pre_delay error: {rel_err_pre_delay:.3f}, decay_time error: {rel_err_decay:.3f}, "
                            f"early_refl error: {rel_err_early:.3f}, late_reverb error: {rel_err_late:.3f}, "
                            f"diffusion error: {rel_err_diffusion:.3f}, spectral_centroid error: {rel_err_centroid:.3f} "
                            f"--> total error: {total_error:.3f}")

                if total_error < iteration_best_error:
                    iteration_best_error = total_error
                    iteration_best_candidate = candidate
                    iteration_best_extracted_params = extracted_params

            # If improvement is seen in this iteration, update the global best.
            if iteration_best_error < global_best_error:
                global_best_error = iteration_best_error
                global_best_candidate = iteration_best_candidate
                global_best_extracted_params = iteration_best_extracted_params
                best_iteration = iteration
            else:
                # If no improvement in this iteration, break out.
                logger.info(f"No improvement in iteration {iteration} for setting {setting['name']}.")
                break

            # Check if the error meets our tolerance.
            if global_best_error <= tolerance:
                logger.info(f"Optimal extraction parameters found for setting {setting['name']} "
                            f"with total error {global_best_error:.3f}")
                break

            # Update candidate grid around the current best candidate.
            new_candidates = []
            base_epsilon = global_best_candidate["wiener_epsilon"]
            base_maxfev = global_best_candidate["curve_fit_maxfev"]
            for factor_epsilon in [0.8, 1.0, 1.2]:
                for factor_maxfev in [0.8, 1.0, 1.2]:
                    new_candidate = {
                        "wiener_epsilon": base_epsilon * factor_epsilon,
                        "curve_fit_maxfev": int(base_maxfev * factor_maxfev)
                    }
                    new_candidates.append(new_candidate)
            current_candidates = new_candidates
            iteration += 1

        results.append({
            "setting": setting,
            "best_extracted_params": global_best_extracted_params,
            "iterations": best_iteration,
            "best_candidate": global_best_candidate,
            "total_error": global_best_error
        })
        logger.info(f"Final extraction for setting {setting['name']}: {global_best_extracted_params}")

    # 4. Save the overall test results.
    results_file = os.path.join(test_output_dir, "test_reverb_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Test reverb completed. Results saved to {results_file}")
    return results


def batch_process(songs):
    """
    Runs processing in parallel for multiple songs.
    """
    with ProcessPoolExecutor() as executor:
        results = executor.map(lambda song: process_song(*song), songs)
    return list(results)
