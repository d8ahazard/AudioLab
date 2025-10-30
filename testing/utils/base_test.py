"""
Base test classes and common functionality for AudioLab testing framework.
"""

import os
import unittest
import tempfile
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from unittest.mock import Mock, patch, MagicMock

from .test_helpers import (
    TestConfig, TemporaryDirectoryManager, mock_model_inference,
    create_test_audio_file, get_test_data_path
)
from .audio_validation import audio_validator

logger = logging.getLogger(__name__)


class AudioLabBaseTest(unittest.TestCase):
    """Base class for all AudioLab tests providing common functionality."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        # Initialize test configuration
        self.config = TestConfig()

        # Initialize temporary directory manager
        self.temp_manager = TemporaryDirectoryManager()

        # Create test directories
        self.test_dir = self.temp_manager.create_temp_dir()
        self.output_dir = os.path.join(self.test_dir, 'output')
        self.temp_dir = os.path.join(self.test_dir, 'temp')

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

        # Create standard test audio files
        self.test_audio_files = self._create_test_audio_files()

        # Setup logging for this test
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

    def tearDown(self):
        """Clean up test fixtures."""
        super().tearDown()
        self.temp_manager.cleanup()

    def _create_test_audio_files(self) -> Dict[str, str]:
        """Create standard test audio files for testing."""
        files = {}

        # Clean speech samples
        files['speech_clean_male'] = create_test_audio_file(
            duration=2.0, frequency=150,
            output_path=os.path.join(self.test_dir, 'speech_clean_male.wav')
        )
        files['speech_clean_female'] = create_test_audio_file(
            duration=2.0, frequency=220,
            output_path=os.path.join(self.test_dir, 'speech_clean_female.wav')
        )

        # Music samples
        files['music_classical'] = create_test_audio_file(
            duration=3.0, frequency=440,
            output_path=os.path.join(self.test_dir, 'music_classical.wav')
        )

        # Noise sample
        files['noise_white'] = create_test_audio_file(
            duration=2.0, frequency=1000,
            output_path=os.path.join(self.test_dir, 'noise_white.wav')
        )

        # Mixed audio
        files['speech_with_music'] = create_test_audio_file(
            duration=3.0, frequency=200,
            output_path=os.path.join(self.test_dir, 'speech_with_music.wav')
        )

        return files

    def create_test_audio(self, duration: float = 2.0, frequency: float = 440.0,
                         output_path: Optional[str] = None) -> str:
        """Create a test audio file."""
        if output_path is None:
            output_path = os.path.join(self.temp_dir, f'test_audio_{frequency}Hz.wav')

        return create_test_audio_file(
            duration=duration,
            frequency=frequency,
            output_path=output_path
        )

    def validate_audio_output(self, output_path: str, expected_duration: Optional[float] = None,
                            expected_sample_rate: Optional[int] = None) -> Dict[str, Any]:
        """Validate that an audio output file is correct."""
        return audio_validator.validate_audio_file(
            output_path,
            expected_sample_rate=expected_sample_rate,
            expected_duration=expected_duration
        )

    def compare_audio_similarity(self, original_path: str, processed_path: str,
                               tolerance_snr: Optional[float] = None) -> Dict[str, Any]:
        """Compare two audio files for similarity."""
        return audio_validator.validate_audio_similarity(
            original_path,
            processed_path,
            tolerance_snr=tolerance_snr
        )

    def mock_model(self, model_type: str, **kwargs):
        """Create a mocked model for testing."""
        return AudioLabModelMock(model_type, **kwargs)

    def skip_if_no_gpu(self):
        """Skip test if GPU is not available."""
        try:
            import torch
            if not torch.cuda.is_available():
                self.skipTest("GPU not available")
        except ImportError:
            self.skipTest("PyTorch not available")

    def skip_if_no_torch(self):
        """Skip test if PyTorch is not available."""
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch not available")

    def assert_file_exists(self, file_path: str, message: str = None):
        """Assert that a file exists."""
        if message is None:
            message = f"File should exist: {file_path}"

        self.assertTrue(os.path.exists(file_path), message)

        if not os.path.exists(file_path):
            # List directory contents for debugging
            directory = os.path.dirname(file_path)
            if os.path.exists(directory):
                files = os.listdir(directory)
                self.fail(f"{message}. Directory contents: {files}")
            else:
                self.fail(f"{message}. Directory does not exist: {directory}")

    def assert_file_not_exists(self, file_path: str, message: str = None):
        """Assert that a file does not exist."""
        if message is None:
            message = f"File should not exist: {file_path}"

        self.assertFalse(os.path.exists(file_path), message)


class AudioLabModelMock:
    """Mock model for testing AudioLab components."""

    def __init__(self, model_type: str, **kwargs):
        self.model_type = model_type
        self.config = kwargs

    def __call__(self, *args, **kwargs):
        """Mock model inference."""
        return mock_model_inference(args[0] if args else "test_input", self.model_type)

    def generate(self, *args, **kwargs):
        """Mock generation method."""
        return {'audio': mock_model_inference(args[0] if args else "test_input", self.model_type)}

    def infer(self, *args, **kwargs):
        """Mock inference method."""
        return mock_model_inference(args[0] if args else "test_input", self.model_type)

    def train(self, *args, **kwargs):
        """Mock training method."""
        return True

    def save(self, *args, **kwargs):
        """Mock save method."""
        return True

    def load(self, *args, **kwargs):
        """Mock load method."""
        return self


class LayoutTestMixin:
    """Mixin for testing layout modules."""

    def setUp(self):
        """Setup for layout tests."""
        super().setUp()

        # Mock Gradio components for layout testing
        self.mock_gradio_components()

    def mock_gradio_components(self):
        """Mock Gradio components for testing."""
        # Mock common Gradio components
        gradio_mocks = [
            'gradio.Button',
            'gradio.Audio',
            'gradio.Textbox',
            'gradio.Dropdown',
            'gradio.Slider',
            'gradio.Progress',
            'gradio.Tab',
            'gradio.Tabs',
            'gradio.Row',
            'gradio.Column',
            'gradio.Accordion'
        ]

        for component in gradio_mocks:
            mock_component = Mock()
            mock_component.return_value = Mock()
            setattr(self, component.split('.')[-1].lower(), mock_component)

    def mock_layout_dependencies(self, layout_name: str):
        """Mock dependencies for a specific layout."""
        if layout_name == 'rvc_train':
            # Mock RVC-specific dependencies
            with patch('modules.rvc.infer.modules.train.preprocess.preprocess_trainset'), \
                 patch('modules.rvc.infer.modules.train.extract.extract_f0_rmvpe.extract_f0_features_rmvpe'), \
                 patch('modules.rvc.infer.modules.train.train.train_main'):
                pass
        elif layout_name == 'tts':
            # Mock TTS-specific dependencies
            with patch('modules.zonos.model'), \
                 patch('chatterbox.tts.ChatterboxTTS'), \
                 patch('modules.diatts.dia.model'):
                pass


class ModuleTestMixin:
    """Mixin for testing core modules."""

    def mock_module_dependencies(self, module_name: str):
        """Mock dependencies for a specific module."""
        if module_name == 'rvc':
            # Mock RVC module dependencies
            with patch('librosa.load'), \
                 patch('torch.load'), \
                 patch('torchaudio.load'):
                pass
        elif module_name == 'zonos':
            # Mock Zonos module dependencies
            with patch('modules.zonos.conditioning.supported_language_codes'), \
                 patch('modules.zonos.model'):
                pass


class AudioProcessingTestMixin:
    """Mixin for testing audio processing functionality."""

    def validate_processing_output(self, input_path: str, output_path: str,
                                 expected_changes: Dict[str, Any] = None):
        """Validate audio processing output."""
        # Validate output file exists and is proper audio
        validation = self.validate_audio_output(output_path)
        self.assertTrue(validation['valid'], f"Invalid output audio: {validation.get('error', 'Unknown error')}")

        # Compare with input if expected changes are specified
        if expected_changes:
            metrics = audio_validator.calculate_audio_metrics(input_path, output_path)

            for metric, expected_value in expected_changes.items():
                if metric in metrics:
                    actual_value = metrics[metric]
                    self.assertAlmostEqual(
                        actual_value, expected_value, delta=0.1,
                        msg=f"{metric} mismatch: expected {expected_value}, got {actual_value}"
                    )

    def test_audio_processing_chain(self, processor_chain: List[str],
                                  input_audio: str, expected_outputs: int = 1):
        """Test a chain of audio processors."""
        # This would be implemented by specific test classes
        # that know how to invoke the processing chain
        pass


class PerformanceTestMixin:
    """Mixin for performance testing."""

    def measure_execution_time(self, func, *args, **kwargs) -> float:
        """Measure function execution time."""
        import time
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        return end_time - start_time

    def assert_performance_threshold(self, execution_time: float,
                                   max_time: float, operation: str = "operation"):
        """Assert that execution time is within acceptable threshold."""
        self.assertLessEqual(
            execution_time, max_time,
            f"{operation} took {execution_time".3f"}s, exceeds threshold of {max_time".3f"}s"
        )

    def test_memory_usage(self, func, *args, max_memory_mb: float = 100, **kwargs):
        """Test that function doesn't exceed memory threshold."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        try:
            func(*args, **kwargs)

            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = final_memory - initial_memory

            self.assertLessEqual(
                memory_used, max_memory_mb,
                f"Function used {memory_used".1f"}MB, exceeds threshold of {max_memory_mb}MB"
            )
        except Exception as e:
            # Clean up on error
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = final_memory - initial_memory
            self.fail(f"Function failed and used {memory_used".1f"}MB memory: {e}")


def create_test_dataset(dataset_size: int = 10, output_dir: Optional[str] = None) -> str:
    """Create a test dataset for training/testing."""
    if output_dir is None:
        temp_dir = tempfile.gettempdir()
        output_dir = os.path.join(temp_dir, f'test_dataset_{dataset_size}')

    os.makedirs(output_dir, exist_ok=True)

    # Create test audio files
    for i in range(dataset_size):
        audio_path = os.path.join(output_dir, f'audio_{i"03d"}.wav')
        create_test_audio_file(
            duration=2.0 + (i * 0.1),  # Varying durations
            frequency=200 + (i * 20),  # Varying frequencies
            output_path=audio_path
        )

    # Create metadata file
    metadata_path = os.path.join(output_dir, 'metadata.csv')
    with open(metadata_path, 'w') as f:
        f.write("filename,text,duration\\n")
        for i in range(dataset_size):
            f.write(f"audio_{i"03d"}.wav,Test utterance {i},{2.0 + (i * 0.1):.1f}\\n")

    return output_dir


def create_mock_config(model_type: str = "rvc", **overrides) -> Dict[str, Any]:
    """Create a mock configuration for testing."""
    base_configs = {
        'rvc': {
            'sample_rate': 22050,
            'hop_length': 256,
            'f0_min': 50,
            'f0_max': 1100,
            'model_version': 'v1'
        },
        'tts': {
            'sample_rate': 22050,
            'language': 'en',
            'emotion': 'neutral',
            'speed': 1.0
        },
        'music_gen': {
            'sample_rate': 22050,
            'duration': 10.0,
            'genre': 'classical',
            'tempo': 120
        }
    }

    config = base_configs.get(model_type, {})
    config.update(overrides)
    return config


# Utility functions for test data
def get_test_audio_path(filename: str) -> str:
    """Get path to a test audio file."""
    return get_test_data_path(os.path.join('audio_samples', filename))


def load_test_audio(audio_path: str) -> Any:
    """Load test audio file."""
    try:
        import librosa
        return librosa.load(audio_path, sr=None)
    except ImportError:
        # Fallback if librosa not available
        return None, None
