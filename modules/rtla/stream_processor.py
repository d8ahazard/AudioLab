import time
import numpy as np
from typing import List

from .config import CHUNK_SIZE, HOP_LENGTH
from .utils import process_chroma, process_mel, process_phonemes


class StreamProcessor:
    def __init__(self, sample_rate: int, chunk_size: int, hop_length: int, features: List[str]):
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.features = features
        self.last_chunk = None
        self.index = 0

        self.feature_chunks: List[np.ndarray] = []

    def _process_feature(self, y, timestamp=None):
        if self.last_chunk is None:
            y = np.concatenate((np.zeros(self.hop_length), y))
        else:
            y = np.concatenate((self.last_chunk, y))

        y_feature = None
        for feature in self.features:
            if feature == "chroma":
                y_chroma = process_chroma(y)
                y_feature = y_chroma if y_feature is None else np.vstack((y_feature, y_chroma))
            elif feature == "mel":
                y_mel = process_mel(y)
                y_feature = y_mel if y_feature is None else np.vstack((y_feature, y_mel))
            elif feature == "phoneme":
                y_phoneme = process_phonemes(y)
                y_feature = y_phoneme if y_feature is None else np.vstack((y_feature, y_phoneme))

        current_chunk = {
            "timestamp": timestamp if timestamp else time.time(),
            "feature": y_feature[:, -int(self.chunk_size / self.hop_length) :],
        }
        self.feature_chunks.append(current_chunk["feature"])
        self.last_chunk = y[-self.hop_length :]
        self.index += 1

    def mock_stream(self, file_path: str):
        import librosa

        duration = int(librosa.get_duration(path=file_path))
        audio_y, _ = librosa.load(file_path, sr=self.sample_rate)
        padded_audio = np.concatenate((audio_y, np.zeros(duration * 2 * self.sample_rate)))
        trimmed_audio = padded_audio[: len(padded_audio) - (len(padded_audio) % self.chunk_size)]
        while trimmed_audio.any():
            audio_chunk = trimmed_audio[: self.chunk_size]
            self._process_feature(audio_chunk, time.time())
            trimmed_audio = trimmed_audio[self.chunk_size :]
            # index is incremented in _process_feature

        additional_padding_size = duration * 2 * self.sample_rate
        while additional_padding_size > 0:
            self._process_feature(np.zeros(self.chunk_size), time.time())
            additional_padding_size -= self.chunk_size


