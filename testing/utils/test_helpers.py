"""
Common test helper functions and utilities for AudioLab testing framework.
"""

import os
import json
import tempfile
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import numpy as np
import torch
import librosa

# Setup logging
logger = logging.getLogger(__name__)


class TestConfig:
    """Test configuration manager."""

    def __init__(self, config_path: str = "testing/config/test_settings.json"):
        self.config_path = Path(config_path)
        self._config = None

    def load(self) -> Dict[str, Any]:
        """Load test configuration from file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            self._config = json.load(f)

        return self._config

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        if self._config is None:
            self.load()

        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """Set configuration value."""
        if self._config is None:
            self.load()

        keys = key.split('.')
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value


class TemporaryDirectoryManager:
    """Context manager for temporary directories with automatic cleanup."""

    def __init__(self, prefix: str = "audiolab_test_"):
        self.prefix = prefix
        self.temp_dirs: List[str] = []

    def create_temp_dir(self) -> str:
        """Create a temporary directory and track it for cleanup."""
        temp_dir = tempfile.mkdtemp(prefix=self.prefix)
        self.temp_dirs.append(temp_dir)
        return temp_dir

    def cleanup(self):
        """Clean up all created temporary directories."""
        for temp_dir in self.temp_dirs:
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    logger.debug(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {temp_dir}: {e}")

        self.temp_dirs.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


def create_test_audio_file(
    duration: float = 2.0,
    sample_rate: int = 22050,
    frequency: float = 440.0,
    output_path: Optional[str] = None,
    format: str = "wav"
) -> str:
    """
    Create a test audio file with a sine wave.

    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        frequency: Frequency of the sine wave in Hz
        output_path: Output file path (optional)
        format: Audio format ('wav', 'mp3', 'flac')

    Returns:
        Path to the created audio file
    """
    if output_path is None:
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, f"test_audio_{frequency}Hz.{format}")

    # Generate sine wave
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = 0.5 * np.sin(2 * np.pi * frequency * t).astype(np.float32)

    # Save audio file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if format == "wav":
        import soundfile as sf
        sf.write(output_path, audio_data, sample_rate)
    else:
        # For other formats, save as WAV first then convert if needed
        import soundfile as sf
        temp_wav = output_path.replace(f".{format}", "_temp.wav")
        sf.write(temp_wav, audio_data, sample_rate)

        # Convert to target format (simplified - would need additional libraries)
        if format != "wav":
            logger.warning(f"Format conversion to {format} not fully implemented")

    return output_path


def load_test_audio(audio_path: str) -> np.ndarray:
    """Load audio file for testing."""
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        return audio, sr
    except Exception as e:
        logger.error(f"Failed to load audio file {audio_path}: {e}")
        raise


def compare_audio_arrays(
    original: np.ndarray,
    processed: np.ndarray,
    tolerance: float = 0.1,
    method: str = "rmse"
) -> bool:
    """
    Compare two audio arrays for similarity.

    Args:
        original: Original audio array
        processed: Processed audio array
        tolerance: Tolerance for similarity check
        method: Comparison method ('rmse', 'snr', 'correlation')

    Returns:
        True if arrays are similar within tolerance
    """
    if len(original) != len(processed):
        logger.warning(f"Audio arrays have different lengths: {len(original)} vs {len(processed)}")
        return False

    if method == "rmse":
        # Root Mean Square Error
        rmse = np.sqrt(np.mean((original - processed) ** 2))
        max_amplitude = max(np.max(np.abs(original)), np.max(np.abs(processed)))
        if max_amplitude == 0:
            return rmse == 0
        normalized_rmse = rmse / max_amplitude
        return normalized_rmse <= tolerance

    elif method == "snr":
        # Signal-to-Noise Ratio
        signal_power = np.mean(original ** 2)
        if signal_power == 0:
            return np.mean(processed ** 2) == 0

        noise_power = np.mean((original - processed) ** 2)
        if noise_power == 0:
            return True

        snr = 10 * np.log10(signal_power / noise_power)
        return snr >= tolerance

    elif method == "correlation":
        # Pearson correlation coefficient
        if np.std(original) == 0 or np.std(processed) == 0:
            return np.allclose(original, processed)

        corr_matrix = np.corrcoef(original, processed)
        correlation = corr_matrix[0, 1]
        return correlation >= (1 - tolerance)

    else:
        raise ValueError(f"Unknown comparison method: {method}")


def mock_model_inference(input_data: Any, model_type: str) -> Any:
    """
    Mock model inference for testing purposes.

    Args:
        input_data: Input data for the model
        model_type: Type of model ('tts', 'rvc', 'music_gen', etc.)

    Returns:
        Mocked model output
    """
    logger.info(f"Mocking {model_type} model inference")

    if model_type in ['tts', 'zonos', 'chatterbox', 'dia']:
        # Mock TTS output - return audio-like data
        if isinstance(input_data, str):
            # Text input - estimate duration based on text length
            duration = max(1.0, len(input_data) * 0.1)  # Rough estimate
        else:
            # Audio input - use original duration
            duration = len(input_data) / 22050 if hasattr(input_data, '__len__') else 2.0

        sample_rate = 22050
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        mock_audio = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        return mock_audio

    elif model_type in ['rvc', 'voice_conversion']:
        # Mock voice conversion - return similar audio with slight modifications
        if hasattr(input_data, '__len__'):
            # Modify pitch slightly
            mock_output = input_data * 0.95  # Slight volume/pitch change
            return mock_output
        else:
            return input_data

    elif model_type in ['music_gen', 'yue', 'stable_audio']:
        # Mock music generation
        duration = 10.0  # Standard music clip duration
        sample_rate = 22050
        t = np.linspace(0, duration, int(sample_rate * duration), False)

        # Generate multi-harmonic content
        frequencies = [220, 330, 440, 660]  # A3, E4, A4, E5
        mock_audio = np.zeros_like(t)

        for i, freq in enumerate(frequencies):
            mock_audio += 0.2 * np.sin(2 * np.pi * freq * t + i * np.pi / 4)

        return mock_audio.astype(np.float32)

    elif model_type in ['separator', 'stem_separator']:
        # Mock audio separation - return multiple stems
        if hasattr(input_data, '__len__'):
            # Split into stems (vocals, drums, bass, other)
            length = len(input_data)
            stems = {
                'vocals': input_data * 0.8,
                'drums': input_data * 0.3,
                'bass': input_data * 0.4,
                'other': input_data * 0.2
            }
            return stems

    else:
        logger.warning(f"Unknown model type: {model_type}")
        return input_data


def skip_if_no_gpu():
    """Skip test if GPU is not available."""
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("GPU not available")


def skip_if_no_torch():
    """Skip test if PyTorch is not available."""
    try:
        import torch
    except ImportError:
        import pytest
        pytest.skip("PyTorch not available")


def get_test_data_path(filename: str) -> str:
    """Get path to test data file."""
    config = TestConfig()
    test_data_root = config.get("test_environment.test_data_root", "testing/config")
    return os.path.join(test_data_root, filename)


def validate_test_environment():
    """Validate that the test environment is properly set up."""
    required_paths = [
        "testing/config",
        "testing/unit",
        "testing/integration",
        "testing/utils"
    ]

    missing_paths = []
    for path in required_paths:
        if not os.path.exists(path):
            missing_paths.append(path)

    if missing_paths:
        raise EnvironmentError(f"Test environment not properly set up. Missing: {missing_paths}")


# Global test configuration instance
test_config = TestConfig()
temp_dir_manager = TemporaryDirectoryManager()
