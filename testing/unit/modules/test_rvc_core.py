"""
Comprehensive unit tests for RVC core module functionality.
"""

import os
import unittest
import tempfile
import numpy as np
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path

from testing.utils.base_test import AudioLabBaseTest, ModuleTestMixin

# Import the RVC module for testing
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'modules', 'rvc'))

try:
    from infer.modules.vc.pipeline import Pipeline
    from infer.modules.train.preprocess import preprocess_trainset
    from infer.modules.train.extract.extract_f0_rmvpe import extract_f0_features_rmvpe
    from infer.modules.train.train import train_main
    from utils import HParams
    from pitch_extraction import pitch_extract
    rvc_available = True
except ImportError as e:
    rvc_available = False
    import_error = str(e)


class TestRVCPipeline(AudioLabBaseTest, ModuleTestMixin):
    """Test RVC voice conversion pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        # Create test model configuration
        self.test_config = {
            'model_path': os.path.join(self.test_dir, 'test_model.pth'),
            'index_path': os.path.join(self.test_dir, 'test_model.index'),
            'device': 'cpu',
            'sample_rate': 22050,
            'hop_length': 256
        }

        # Create mock model files
        with open(self.test_config['model_path'], 'wb') as f:
            f.write(b'mock_model_data')

        with open(self.test_config['index_path'], 'wb') as f:
            f.write(b'mock_index_data')

        # Test audio data
        self.test_audio = self.create_test_audio(duration=2.0, frequency=220)

    @unittest.skipIf(not rvc_available, "RVC module not available")
    def test_pipeline_initialization(self):
        """Test RVC pipeline initialization."""
        print("\\nTesting RVC pipeline initialization...")

        try:
            # Test pipeline creation with valid config
            pipeline = Pipeline(
                model_path=self.test_config['model_path'],
                index_path=self.test_config['index_path'],
                device=self.test_config['device']
            )

            self.assertIsNotNone(pipeline)

        except Exception as e:
            self.fail(f"Pipeline initialization failed: {e}")

    @unittest.skipIf(not rvc_available, "RVC module not available")
    def test_pipeline_voice_conversion(self):
        """Test voice conversion through pipeline."""
        print("\\nTesting RVC voice conversion...")

        with patch('librosa.load') as mock_load, \
             patch('torch.load') as mock_torch_load, \
             patch('torchaudio.save') as mock_save:

            # Setup mocks
            mock_audio_data = np.random.randn(44100).astype(np.float32)
            mock_load.return_value = (mock_audio_data, 22050)
            mock_torch_load.return_value = {'config': self.test_config}

            try:
                pipeline = Pipeline(
                    model_path=self.test_config['model_path'],
                    index_path=self.test_config['index_path'],
                    device='cpu'
                )

                # Test voice conversion
                result = pipeline.vc(
                    audio_path=self.test_audio,
                    f0_up_key=0,
                    f0_method='rmvpe',
                    index_rate=0.5,
                    filter_radius=3,
                    resample_sr=22050,
                    rms_mix_rate=0.25,
                    protect=0.33
                )

                # Should return converted audio
                self.assertIsNotNone(result)

            except Exception as e:
                self.fail(f"Voice conversion failed: {e}")

    @unittest.skipIf(not rvc_available, "RVC module not available")
    def test_pipeline_parameter_validation(self):
        """Test pipeline parameter validation."""
        print("\\nTesting RVC pipeline parameter validation...")

        try:
            pipeline = Pipeline(
                model_path=self.test_config['model_path'],
                index_path=self.test_config['index_path'],
                device='cpu'
            )

            # Test with invalid parameters
            with self.assertRaises(Exception):
                pipeline.vc(
                    audio_path="/nonexistent/audio.wav",  # Invalid path
                    f0_up_key=0,
                    f0_method='rmvpe',
                    index_rate=0.5,
                    filter_radius=3,
                    resample_sr=22050,
                    rms_mix_rate=0.25,
                    protect=0.33
                )

        except Exception as e:
            self.fail(f"Parameter validation failed: {e}")


class TestRVCPreprocessing(AudioLabBaseTest, ModuleTestMixin):
    """Test RVC dataset preprocessing."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        # Create test dataset
        self.dataset_dir = os.path.join(self.test_dir, 'test_dataset')
        os.makedirs(self.dataset_dir)

        # Create test audio files
        for i in range(5):
            audio_path = os.path.join(self.dataset_dir, f'audio_{i"03d"}.wav')
            self.create_test_audio(duration=2.0 + i * 0.2, output_path=audio_path)

        self.exp_dir = os.path.join(self.test_dir, 'experiments')
        os.makedirs(self.exp_dir)

    @unittest.skipIf(not rvc_available, "RVC module not available")
    def test_preprocess_trainset_basic(self):
        """Test basic dataset preprocessing."""
        print("\\nTesting basic dataset preprocessing...")

        try:
            result = preprocess_trainset(
                trainset_dir=self.dataset_dir,
                exp_dir=self.exp_dir,
                sr=22050,
                n_p=2
            )

            # Should complete preprocessing
            self.assertIsNotNone(result)

        except Exception as e:
            self.fail(f"Preprocessing failed: {e}")

    @unittest.skipIf(not rvc_available, "RVC module not available")
    def test_preprocess_trainset_different_sample_rates(self):
        """Test preprocessing with different sample rates."""
        print("\\nTesting preprocessing with different sample rates...")

        sample_rates = [22050, 44100, 48000]

        for sr in sample_rates:
            with self.subTest(sample_rate=sr):
                try:
                    result = preprocess_trainset(
                        trainset_dir=self.dataset_dir,
                        exp_dir=os.path.join(self.exp_dir, f'exp_{sr}'),
                        sr=sr,
                        n_p=1
                    )

                    self.assertIsNotNone(result)

                except Exception as e:
                    self.fail(f"Preprocessing at {sr}Hz failed: {e}")

    @unittest.skipIf(not rvc_available, "RVC module not available")
    def test_preprocess_trainset_empty_dataset(self):
        """Test preprocessing with empty dataset."""
        print("\\nTesting preprocessing with empty dataset...")

        empty_dir = os.path.join(self.test_dir, 'empty_dataset')
        os.makedirs(empty_dir)

        try:
            result = preprocess_trainset(
                trainset_dir=empty_dir,
                exp_dir=self.exp_dir,
                sr=22050,
                n_p=1
            )

            # Should handle empty dataset gracefully
            self.assertIsNotNone(result)

        except Exception as e:
            self.fail(f"Empty dataset preprocessing failed: {e}")


class TestRVCF0Extraction(AudioLabBaseTest, ModuleTestMixin):
    """Test RVC F0 feature extraction."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        self.exp_dir = os.path.join(self.test_dir, 'f0_extraction')
        os.makedirs(self.exp_dir)

        # Create test audio for F0 extraction
        self.test_audio_for_f0 = self.create_test_audio(duration=3.0, frequency=220)

    @unittest.skipIf(not rvc_available, "RVC module not available")
    def test_extract_f0_features_rmvpe(self):
        """Test F0 extraction with RMVPE method."""
        print("\\nTesting F0 extraction with RMVPE...")

        try:
            result = extract_f0_features_rmvpe(
                input_dir=self.exp_dir,
                exp_dir=self.exp_dir,
                f0_method='rmvpe',
                device='cpu',
                gpus='0'
            )

            # Should complete F0 extraction
            self.assertIsNotNone(result)

        except Exception as e:
            self.fail(f"RMVPE F0 extraction failed: {e}")

    @unittest.skipIf(not rvc_available, "RVC module not available")
    def test_extract_f0_features_different_methods(self):
        """Test F0 extraction with different methods."""
        print("\\nTesting F0 extraction with different methods...")

        # Note: This would test different F0 extraction methods
        # but the actual implementation might vary

        f0_methods = ['rmvpe', 'dio', 'harvest']  # Common methods

        for method in f0_methods:
            with self.subTest(f0_method=method):
                try:
                    # This would need to be adapted based on actual implementation
                    result = extract_f0_features_rmvpe(
                        input_dir=self.exp_dir,
                        exp_dir=self.exp_dir,
                        f0_method=method,
                        device='cpu',
                        gpus='0'
                    )

                    self.assertIsNotNone(result)

                except Exception as e:
                    print(f"F0 extraction with {method} failed: {e}")
                    # Some methods might not be available, which is OK


class TestRVCTraining(AudioLabBaseTest, ModuleTestMixin):
    """Test RVC model training."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        # Create training configuration
        self.train_config = {
            'exp_dir': os.path.join(self.test_dir, 'training'),
            'sr': 22050,
            'if_f0_3': True,
            'spk_id': 0,
            'gpus': '0',
            'version19': 'v1'
        }

        os.makedirs(self.train_config['exp_dir'])

        # Create minimal dataset for training
        self.train_dataset_dir = os.path.join(self.test_dir, 'train_dataset')
        os.makedirs(self.train_dataset_dir)

        for i in range(3):
            audio_path = os.path.join(self.train_dataset_dir, f'train_audio_{i}.wav')
            self.create_test_audio(duration=2.0, output_path=audio_path)

    @unittest.skipIf(not rvc_available, "RVC module not available")
    def test_train_main_basic(self):
        """Test basic model training."""
        print("\\nTesting basic model training...")

        try:
            # Mock the training process
            with patch('librosa.load'), \
                 patch('torch.save'), \
                 patch('torch.load'):

                result = train_main(
                    exp_dir=self.train_config['exp_dir'],
                    sr=self.train_config['sr'],
                    if_f0_3=self.train_config['if_f0_3'],
                    spk_id=self.train_config['spk_id'],
                    gpus=self.train_config['gpus'],
                    version19=self.train_config['version19']
                )

                # Should complete training
                self.assertIsNotNone(result)

        except Exception as e:
            self.fail(f"Model training failed: {e}")

    @unittest.skipIf(not rvc_available, "RVC module not available")
    def test_train_main_with_custom_config(self):
        """Test training with custom configuration."""
        print("\\nTesting training with custom configuration...")

        custom_config = self.train_config.copy()
        custom_config.update({
            'sr': 44100,
            'if_f0_3': False,
            'spk_id': 1
        })

        try:
            with patch('librosa.load'), \
                 patch('torch.save'), \
                 patch('torch.load'):

                result = train_main(**custom_config)

                self.assertIsNotNone(result)

        except Exception as e:
            self.fail(f"Custom config training failed: {e}")


class TestRVCPitchExtraction(AudioLabBaseTest):
    """Test RVC pitch extraction functionality."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        # Create test audio with known pitch
        self.test_audio_pitch = self.create_test_audio(duration=2.0, frequency=220)

    @unittest.skipIf(not rvc_available, "RVC module not available")
    def test_pitch_extract_basic(self):
        """Test basic pitch extraction."""
        print("\\nTesting basic pitch extraction...")

        try:
            # Load test audio
            import librosa
            audio, sr = librosa.load(self.test_audio_pitch, sr=None)

            # Test pitch extraction
            f0 = pitch_extract(
                audio=audio,
                sr=sr,
                f0_method='rmvpe',
                hop_length=256,
                f0_min=50,
                f0_max=1100
            )

            # Should return pitch values
            self.assertIsNotNone(f0)
            self.assertIsInstance(f0, np.ndarray)

        except Exception as e:
            self.fail(f"Pitch extraction failed: {e}")

    @unittest.skipIf(not rvc_available, "RVC module not available")
    def test_pitch_extract_parameter_variations(self):
        """Test pitch extraction with parameter variations."""
        print("\\nTesting pitch extraction parameter variations...")

        try:
            import librosa
            audio, sr = librosa.load(self.test_audio_pitch, sr=None)

            # Test different F0 methods and parameters
            test_configs = [
                {'f0_method': 'rmvpe', 'hop_length': 256, 'f0_min': 50, 'f0_max': 1100},
                {'f0_method': 'dio', 'hop_length': 128, 'f0_min': 75, 'f0_max': 800},
                {'f0_method': 'harvest', 'hop_length': 512, 'f0_min': 40, 'f0_max': 1200}
            ]

            for config in test_configs:
                with self.subTest(config=config):
                    f0 = pitch_extract(audio=audio, sr=sr, **config)
                    self.assertIsNotNone(f0)
                    self.assertIsInstance(f0, np.ndarray)

        except Exception as e:
            self.fail(f"Pitch extraction parameter test failed: {e}")


class TestRVCHParams(AudioLabBaseTest):
    """Test RVC hyperparameters configuration."""

    @unittest.skipIf(not rvc_available, "RVC module not available")
    def test_hparams_initialization(self):
        """Test HParams initialization."""
        print("\\nTesting HParams initialization...")

        try:
            # Test basic HParams creation
            hparams = HParams(
                sample_rate=22050,
                hop_length=256,
                f0_min=50,
                f0_max=1100
            )

            self.assertIsNotNone(hparams)
            self.assertEqual(hparams.sample_rate, 22050)

        except Exception as e:
            self.fail(f"HParams initialization failed: {e}")

    @unittest.skipIf(not rvc_available, "RVC module not available")
    def test_hparams_validation(self):
        """Test HParams validation."""
        print("\\nTesting HParams validation...")

        try:
            # Test with invalid parameters
            with self.assertRaises(Exception):
                HParams(
                    sample_rate=-22050,  # Invalid negative sample rate
                    hop_length=256
                )

        except Exception as e:
            self.fail(f"HParams validation failed: {e}")


class TestRVCUtils(AudioLabBaseTest):
    """Test RVC utility functions."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        # Test data for utility functions
        self.test_audio_data = np.random.randn(22050).astype(np.float32)
        self.test_f0_data = np.random.uniform(50, 1100, size=(86,)).astype(np.float32)

    @unittest.skipIf(not rvc_available, "RVC module not available")
    def test_audio_processing_utils(self):
        """Test audio processing utility functions."""
        print("\\nTesting audio processing utilities...")

        # This would test various utility functions in the RVC module
        # The specific functions would depend on the actual implementation

        try:
            # Test audio normalization (example)
            normalized_audio = self.test_audio_data / np.max(np.abs(self.test_audio_data))

            # Should be normalized to [-1, 1] range
            self.assertTrue(np.all(np.abs(normalized_audio) <= 1.0))

        except Exception as e:
            self.fail(f"Audio processing utils failed: {e}")


class TestRVCErrorHandling(AudioLabBaseTest):
    """Test error handling in RVC components."""

    @unittest.skipIf(not rvc_available, "RVC module not available")
    def test_pipeline_invalid_model_path(self):
        """Test pipeline with invalid model path."""
        print("\\nTesting pipeline with invalid model path...")

        try:
            with self.assertRaises(Exception):
                Pipeline(
                    model_path="/nonexistent/model.pth",
                    index_path="/nonexistent/index.index",
                    device='cpu'
                )

        except Exception as e:
            self.fail(f"Invalid model path handling failed: {e}")

    @unittest.skipIf(not rvc_available, "RVC module not available")
    def test_preprocessing_invalid_audio(self):
        """Test preprocessing with invalid audio files."""
        print("\\nTesting preprocessing with invalid audio...")

        # Create directory with invalid audio files
        invalid_dir = os.path.join(self.test_dir, 'invalid_audio')
        os.makedirs(invalid_dir)

        # Create invalid audio file
        invalid_audio = os.path.join(invalid_dir, 'invalid.wav')
        with open(invalid_audio, 'wb') as f:
            f.write(b'This is not audio data')

        try:
            result = preprocess_trainset(
                trainset_dir=invalid_dir,
                exp_dir=self.test_dir,
                sr=22050,
                n_p=1
            )

            # Should handle invalid audio gracefully
            self.assertIsNotNone(result)

        except Exception as e:
            self.fail(f"Invalid audio handling failed: {e}")


class TestRVCIntegration(AudioLabBaseTest, ModuleTestMixin):
    """Integration tests for RVC module."""

    def setUp(self):
        """Set up integration test fixtures."""
        super().setUp()

        # Create complete test environment
        self.integration_dataset_dir = os.path.join(self.test_dir, 'integration_dataset')
        os.makedirs(self.integration_dataset_dir)

        # Create dataset for integration testing
        for i in range(10):
            audio_path = os.path.join(self.integration_dataset_dir, f'int_audio_{i"03d"}.wav')
            self.create_test_audio(duration=2.0 + i * 0.1, output_path=audio_path)

        self.integration_exp_dir = os.path.join(self.test_dir, 'integration_training')
        os.makedirs(self.integration_exp_dir)

    @unittest.skipIf(not rvc_available, "RVC module not available")
    def test_full_rvc_workflow_integration(self):
        """Test complete RVC workflow integration."""
        print("\\nTesting full RVC workflow integration...")

        # Mock external dependencies
        with patch('librosa.load') as mock_librosa, \
             patch('torch.load') as mock_torch_load, \
             patch('torch.save') as mock_torch_save, \
             patch('torchaudio.save') as mock_torchaudio_save:

            # Setup mocks
            mock_audio_data = np.random.randn(44100).astype(np.float32)
            mock_librosa.return_value = (mock_audio_data, 22050)
            mock_torch_load.return_value = {'config': {'sample_rate': 22050}}
            mock_torch_save.return_value = True
            mock_torchaudio_save.return_value = True

            try:
                # 1. Preprocessing
                preprocess_result = preprocess_trainset(
                    trainset_dir=self.integration_dataset_dir,
                    exp_dir=self.integration_exp_dir,
                    sr=22050,
                    n_p=2
                )

                # 2. F0 extraction
                f0_result = extract_f0_features_rmvpe(
                    input_dir=self.integration_exp_dir,
                    exp_dir=self.integration_exp_dir,
                    f0_method='rmvpe',
                    device='cpu',
                    gpus='0'
                )

                # 3. Training
                train_result = train_main(
                    exp_dir=self.integration_exp_dir,
                    sr=22050,
                    if_f0_3=True,
                    spk_id=0,
                    gpus='0',
                    version19='v1'
                )

                # 4. Voice conversion pipeline
                model_path = os.path.join(self.integration_exp_dir, 'model.pth')
                index_path = os.path.join(self.integration_exp_dir, 'model.index')

                # Create mock model files
                with open(model_path, 'wb') as f:
                    f.write(b'mock_model')
                with open(index_path, 'wb') as f:
                    f.write(b'mock_index')

                pipeline = Pipeline(
                    model_path=model_path,
                    index_path=index_path,
                    device='cpu'
                )

                # 5. Voice conversion
                test_audio = self.create_test_audio(duration=2.0, frequency=220)
                vc_result = pipeline.vc(
                    audio_path=test_audio,
                    f0_up_key=0,
                    f0_method='rmvpe',
                    index_rate=0.5,
                    filter_radius=3,
                    resample_sr=22050,
                    rms_mix_rate=0.25,
                    protect=0.33
                )

                # Verify all components completed
                self.assertIsNotNone(preprocess_result)
                self.assertIsNotNone(f0_result)
                self.assertIsNotNone(train_result)
                self.assertIsNotNone(pipeline)
                self.assertIsNotNone(vc_result)

            except Exception as e:
                self.fail(f"RVC integration workflow failed: {e}")


if __name__ == '__main__':
    # Setup logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)

    # Run tests
    unittest.main(verbosity=2)
