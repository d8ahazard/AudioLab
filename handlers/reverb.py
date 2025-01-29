import json
import os
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, correlate
from scipy.optimize import curve_fit
from pydub import AudioSegment
from concurrent.futures import ProcessPoolExecutor
import logging

logger = logging.getLogger(__name__)

################################################################################
# AUDIO I/O
################################################################################

def read_audio(file_path):
    """
    Reads an audio file into a numpy array (normalized to -1..1).
    """
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


def estimate_rt60(signal, sr):
    """
    Estimates RT60 decay time by fitting an exponential decay to the RMS envelope.
    """
    eps = 1e-10
    env = np.sqrt(np.sum(signal ** 2, axis=1)) + eps
    env_db = 20.0 * np.log10(env)

    def exp_decay(x, a, b, c):
        return a * np.exp(-b * x) + c

    time = np.linspace(0, len(env) / sr, len(env))
    popt, _ = curve_fit(exp_decay, time, env_db, maxfev=5000)

    decay_time = 3.0 / popt[1] if popt[1] != 0 else 0.5
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

def extract_reverb(dry_path, wet_path, param_output_path):
    """
    Extracts impulse response and reverb parameters using Wiener Deconvolution.
    """
    dry_signal, sr = read_audio(dry_path)
    wet_signal, wet_sr = read_audio(wet_path)

    if sr != wet_sr:
        raise ValueError("Dry and wet sample rates must match.")

    dry_mono = to_mono(dry_signal)
    wet_mono = to_mono(wet_signal)

    # Estimate pre-delay using FFT cross-correlation
    corr = fft_xcorr(wet_mono, dry_mono)
    best_shift = np.argmax(corr) - (len(dry_mono) - 1)
    best_shift = max(best_shift, 0)
    pre_delay_sec = best_shift / sr

    # Estimate RT60 decay time
    decay_time = estimate_rt60(wet_signal, sr)

    # Extract impulse response using Wiener deconvolution
    impulse_response = wiener_deconvolution(wet_mono, dry_mono)
    impulse_response = impulse_response[:int(sr * 2)]  # Limit IR length to 2 sec

    params = {
        "sample_rate": sr,
        "pre_delay": float(pre_delay_sec),
        "decay_time": float(decay_time),
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

    # Apply convolution to each channel separately if stereo
    if num_channels > 1:
        wet_signals = []
        for ch in range(num_channels):
            ch_dry = dry_signal[:, ch]
            ch_wet = fftconvolve(ch_dry, impulse_response, mode='full')
            ch_wet = np.pad(ch_wet, (pre_delay_samples, 0))  # Add pre-delay
            wet_signals.append(ch_wet[:len(ch_dry)])  # Truncate to match original length
        wet_signal = np.stack(wet_signals, axis=1)  # Re-stack into stereo
    else:
        # Mono case
        dry_signal = to_mono(dry_signal)  # Ensure mono
        wet_signal = fftconvolve(dry_signal, impulse_response, mode='full')
        wet_signal = np.pad(wet_signal, (pre_delay_samples, 0))[:len(dry_signal)]

    # Mix dry and wet signals with a blending factor
    final_signal = dry_signal + 0.7 * wet_signal
    final_signal = np.clip(final_signal, -1.0, 1.0)  # Avoid clipping

    sf.write(output_path, final_signal, sr)
    return output_path



################################################################################
# TESTING
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


def batch_process(songs):
    """
    Runs processing in parallel for multiple songs.
    """
    with ProcessPoolExecutor() as executor:
        results = executor.map(lambda song: process_song(*song), songs)
    return list(results)
