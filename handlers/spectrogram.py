import math

import matplotlib.pyplot as plt
import numpy as np


# A simple metaclass for singleton behavior.
class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


# Helper function to convert frequency (Hz) to a note name.
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def hz_to_note_name(frequency):
    """
    Converts a frequency in Hz to a note name string (e.g. A4).
    Returns an empty string if frequency is zero or negative.
    """
    if frequency <= 0:
        return ""
    # MIDI note conversion: 69 is A4 (440 Hz)
    midi = 69 + 12 * math.log2(frequency / 440.0)
    midi_int = int(round(midi))
    note = NOTE_NAMES[midi_int % 12]
    octave = (midi_int // 12) - 1
    return f"{note}{octave}"


class F0Visualizer(metaclass=SingletonMeta):
    def __init__(self):
        # We'll store the data as a list of (description, f0_array) tuples.
        self.f0_data = []

    def add_f0(self, f0, description):
        """
        Add an f0 computation along with a description.

        Parameters:
          f0: numpy array containing the f0 values.
          description: a string describing the computation method.
        """
        self.f0_data.append((description, f0))

    def clear(self):
        """Clear all stored f0 computations."""
        self.f0_data.clear()

    def visualize(self, output_path, sr=None, hop_length=256):
        """
        Visualize all stored f0 computations in separate subplots.
        Each plot shows the f0 over time (frame index or seconds if sr is provided)
        and annotates a few sample frames with note names.

        Parameters:
          output_path: The file path (including filename and extension) to save the image.
          sr: Optional; the sample rate to convert frame indices to seconds.
          hop_length: Optional; the number of samples between successive frames.
                      Used to convert frame index to seconds when sr is provided.

        Returns:
          output_path: The path where the image was saved.
        """
        num_plots = len(self.f0_data)
        if num_plots == 0:
            print("No f0 computations to visualize.")
            return None

        fig, axs = plt.subplots(num_plots, 1, figsize=(10, 3 * num_plots), squeeze=False)
        for idx, (desc, f0) in enumerate(self.f0_data):
            ax = axs[idx, 0]
            if sr is not None:
                # Multiply the frame indices by hop_length / sr to convert to seconds.
                time_axis = np.arange(len(f0)) * (hop_length / sr)
                ax.set_xlabel("Seconds")
            else:
                time_axis = np.arange(len(f0))
                ax.set_xlabel("Frame")
            ax.plot(time_axis, f0, label=desc, color='blue', linewidth=1.5)
            ax.set_title(desc)
            ax.set_ylabel("Frequency (Hz)")
            # Annotate a few sample points with their note names.
            sample_indices = np.linspace(0, len(f0) - 1, num=5, dtype=int)
            for i in sample_indices:
                freq = f0[i]
                note = hz_to_note_name(freq)
                if note:
                    ax.text(time_axis[i], freq, note, fontsize=8, color='red',
                            verticalalignment='bottom', horizontalalignment='center')
            ax.legend()
            ax.grid(True)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)
        return output_path
