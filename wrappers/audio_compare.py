import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, List, Callable, Dict
from pydub import AudioSegment
from scipy.signal import resample
import cv2

from handlers.config import output_path
from wrappers.base_wrapper import BaseWrapper


class AudioCompare(BaseWrapper):
    priority = 1000000

    def process_audio(self, inputs: List[str], callback: Callable = None, **kwargs: Dict[str, Any]) -> List[str]:
        output_folder = os.path.join(output_path, "comparisons")
        os.makedirs(output_folder, exist_ok=True)

        # Load and normalize audio tracks
        waveforms = []
        sample_rate = 44100  # Resample to a common sample rate
        for file in inputs:
            audio = AudioSegment.from_file(file)
            samples = np.array(audio.get_array_of_samples()) / (2 ** 15)
            resampled = resample(samples, int(len(samples) * sample_rate / audio.frame_rate))
            waveforms.append(resampled)

        # Determine the shortest waveform to align lengths
        min_length = min(len(w) for w in waveforms)
        waveforms = [w[:min_length] for w in waveforms]

        # Generate difference waveform
        differences = np.abs(waveforms[0] - waveforms[1]) if len(waveforms) > 1 else waveforms[0]

        # Visualization
        plt.figure(figsize=(15, 8))

        # Plot original waveforms
        plt.subplot(3, 1, 1)
        plt.plot(waveforms[0], label="Track 1", alpha=0.7)
        plt.plot(waveforms[1], label="Track 2", alpha=0.7, linestyle='dashed') if len(waveforms) > 1 else None
        plt.title("Original Waveforms")
        plt.legend()

        # Plot difference
        plt.subplot(3, 1, 2)
        plt.plot(differences, label="Differences", color='red')
        plt.title("Differences")
        plt.legend()

        # Combined visualization
        plt.subplot(3, 1, 3)
        combined_image = self.generate_combined_image(waveforms[0], waveforms[1] if len(waveforms) > 1 else None)
        # plt.imshow(combined_image, cmap='viridis', aspect='auto')
        # plt.title("Combined Visualization")

        output_file = os.path.join(output_folder, "comparison.png")
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

        return [output_file]

    def generate_combined_image(self, waveform1: np.ndarray, waveform2: np.ndarray = None) -> np.ndarray:
        """Generate a visual representation combining both waveforms and their differences."""
        height, width = 400, len(waveform1)
        canvas = np.zeros((height, width), dtype=np.float32)

        # Normalize waveforms for visualization
        waveform1 = np.interp(waveform1, (waveform1.min(), waveform1.max()), (0, height // 2))
        canvas[:height // 2, :len(waveform1)] = waveform1

        if waveform2 is not None:
            waveform2 = np.interp(waveform2, (waveform2.min(), waveform2.max()), (height // 2, height))
            canvas[height // 2:, :len(waveform2)] = waveform2

        # Enhance differences with filtering
        if waveform2 is not None:
            differences = np.abs(waveform1 - waveform2)
            diff_layer = cv2.GaussianBlur(differences.astype(np.float32), (0, 0), sigmaX=3)
            canvas += diff_layer

        return canvas

    def register_api_endpoint(self, api) -> Any:
        pass
