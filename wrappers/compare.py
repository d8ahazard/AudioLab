import hashlib
import os
from typing import Any, List, Callable, Dict

import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
from scipy.signal import resample, stft

from handlers.config import output_path
from util.data_classes import ProjectFiles
from wrappers.base_wrapper import BaseWrapper


def compute_file_hash(filepath: str, chunk_size: int = 65536) -> str:
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            hasher.update(data)
    return hasher.hexdigest()


def generate_output_filename(file1: str, file2: str, output_folder: str) -> str:
    hash1 = compute_file_hash(file1)[:6]  # shorten the hash for readability
    hash2 = compute_file_hash(file2)[:6]
    base1 = os.path.splitext(os.path.basename(file1))[0]
    base2 = os.path.splitext(os.path.basename(file2))[0]

    return os.path.join(
        output_folder,
        f"{base1}_VS_{base2}_{hash1}_{hash2}_comparison.png"
    )


class Compare(BaseWrapper):
    title = "Compare"
    priority = 1000000

    def process_audio(self, inputs: List[ProjectFiles], callback=None, **kwargs: Dict[str, Any]) -> List[ProjectFiles]:
        """
        Compare two unique audio files and produce:
         - Time-domain waveforms
         - Absolute difference waveform
         - Spectrogram-based difference visualization
        """
        # Filter and ensure exactly two unique files
        pj_outputs = []
        for project in inputs:
            outputs = []
            input_files = project.last_outputs
            inputs, _ = self.filter_inputs(input_files, "audio")

            if len(inputs) >= 2:
                msg = f"Expected exactly 2 unique audio files, got {len(inputs)}."
                if callback:
                    callback(0, msg, 1)
                return []

            unique_files = [project.src_file, inputs[0]]

            # Prepare output
            output_folder = os.path.join(project.project_dir, "comparisons")
            os.makedirs(output_folder, exist_ok=True)
            output_file = generate_output_filename(unique_files[0], unique_files[1], output_folder)

            total_steps = 5
            current_step = 0
            if callback:
                callback(0, "Starting audio comparison with spectrogram difference", total_steps)

            sample_rate = 44100
            waveforms = []

            # 1) Load + resample both
            for file in unique_files:
                audio = AudioSegment.from_file(file)
                # Normalization factor for e.g. 16-bit PCM:
                divisor = float(2 ** (8 * audio.sample_width - 1))
                samples = np.array(audio.get_array_of_samples()) / divisor

                # Resample to common sample_rate
                resampled = resample(samples, int(len(samples) * sample_rate / audio.frame_rate))
                waveforms.append(resampled)

                current_step += 1
                if callback:
                    callback(current_step, f"Loaded/resampled {os.path.basename(file)}", total_steps)

            # 2) Align to min length
            min_len = min(len(w) for w in waveforms)
            waveforms = [w[:min_len] for w in waveforms]

            # 3) RMS-normalize each track for fair loudness comparison
            for i in range(len(waveforms)):
                rms = np.sqrt(np.mean(waveforms[i]**2))
                if rms > 1e-9:
                    waveforms[i] /= rms

            # 4) Compute absolute difference in time domain
            differences = np.abs(waveforms[0] - waveforms[1])

            current_step += 1
            if callback:
                callback(current_step, "Calculated differences", total_steps)

            # --- SPECTROGRAM DIFFERENCE ---
            # Compute the magnitude STFT of each waveform
            f1, t1, Zxx1 = stft(waveforms[0], fs=sample_rate, nperseg=2048, noverlap=1024)
            f2, t2, Zxx2 = stft(waveforms[1], fs=sample_rate, nperseg=2048, noverlap=1024)

            # Convert to magnitude
            spec1 = np.abs(Zxx1)
            spec2 = np.abs(Zxx2)

            # For plotting, we might align their time axes
            min_t_len = min(spec1.shape[1], spec2.shape[1])
            spec1 = spec1[:, :min_t_len]
            spec2 = spec2[:, :min_t_len]
            t_combined = t1[:min_t_len]

            # Take difference or ratio in the frequency domain
            spec_diff = np.abs(spec1 - spec2)

            # 5) Plot results
            plt.figure(figsize=(15, 10))

            # Top row: waveforms
            plt.subplot(4, 1, 1)
            plt.plot(waveforms[0], label="Track 1 (RMS=1)", alpha=0.6)
            plt.plot(waveforms[1], label="Track 2 (RMS=1)", alpha=0.6, linestyle='dashed')
            plt.title("Waveforms (RMS-Normalized)")
            plt.legend()

            # 2nd row: absolute difference
            plt.subplot(4, 1, 2)
            plt.plot(differences, color='red', label="|Track1 - Track2|")
            plt.title("Time-Domain Differences")
            plt.legend()

            # 3rd row: spectrogram difference for Track 1
            plt.subplot(4, 1, 3)
            plt.title("Spectrogram Track 1 (Magnitude)")
            plt.pcolormesh(t_combined, f1, 20*np.log10(spec1 + 1e-9), cmap='viridis', shading='auto')
            plt.ylabel("Frequency [Hz]")
            plt.xlabel("Time [sec]")

            # 4th row: difference spectrogram
            plt.subplot(4, 1, 4)
            plt.title("Spectrogram Difference (|Spec1 - Spec2|)")
            plt.pcolormesh(t_combined, f1, 20*np.log10(spec_diff + 1e-9), cmap='magma', shading='auto')
            plt.ylabel("Frequency [Hz]")
            plt.xlabel("Time [sec]")

            plt.tight_layout()
            plt.savefig(output_file)
            plt.close()

            current_step += 1
            if callback:
                callback(current_step, f"Saved visualization to {output_file}", total_steps)
            project.add_output("comparison", output_file)
            pj_outputs.append(project)

        return pj_outputs

    def register_api_endpoint(self, api) -> Any:
        pass
