import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve
from pydub import AudioSegment


def read_audio(file_path):
    """
    Reads an audio file (any format supported by pydub) and converts it to a numpy array.
    """
    audio = AudioSegment.from_file(file_path)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    samples /= np.iinfo(audio.array_type).max  # Normalize to -1 to 1
    return samples, audio.frame_rate


def save_ir_to_file(ir, output_path):
    """
    Saves the impulse response (IR) to a JSON file.
    """
    import json
    with open(output_path, "w") as file:
        json.dump(ir.tolist(), file)


def load_ir_from_file(ir_path):
    """
    Loads the impulse response (IR) from a JSON file.
    """
    import json
    with open(ir_path, "r") as file:
        ir = json.load(file)
    return np.array(ir, dtype=np.float32)


def extract_ir(dry_path, wet_path, ir_output_path):
    """
    Extracts the impulse response (IR) from dry and wet audio files and saves it to a file.
    """
    dry_signal, dry_sr = read_audio(dry_path)
    wet_signal, wet_sr = read_audio(wet_path)

    if dry_sr != wet_sr:
        raise ValueError("Sample rates of dry and wet signals must match.")

    # Ensure both signals have the same length
    min_length = min(len(dry_signal), len(wet_signal))
    dry_signal = dry_signal[:min_length]
    wet_signal = wet_signal[:min_length]

    # Perform deconvolution in the frequency domain
    dry_fft = np.fft.rfft(dry_signal)
    wet_fft = np.fft.rfft(wet_signal)

    # Avoid division by zero
    dry_fft[dry_fft == 0] = 1e-10

    ir_fft = wet_fft / dry_fft
    ir = np.fft.irfft(ir_fft)

    # Save the impulse response to a file
    save_ir_to_file(ir, ir_output_path)


def apply_reverb(dry_path, ir_path, output_path):
    """
    Applies reverb (based on the IR) to a dry audio file and saves the result.
    """
    dry_signal, dry_sr = read_audio(dry_path)
    ir = load_ir_from_file(ir_path)

    # Apply the IR to the dry signal
    wet_signal = fftconvolve(dry_signal, ir, mode='full')

    # Ensure the output doesn't exceed the original range
    wet_signal = np.clip(wet_signal, -1.0, 1.0)

    # Save the result as a WAV file
    sf.write(output_path, wet_signal, dry_sr)