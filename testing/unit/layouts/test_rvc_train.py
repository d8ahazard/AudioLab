"""
Unit tests for RVC training layout functionality.
"""

import os
import unittest
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from testing.utils.test_helpers import (
    TestConfig, TemporaryDirectoryManager, mock_model_inference, create_test_audio_file
)
from testing.utils.audio_validation import audio_validator

# Import the layout module for testing
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'layouts'))

try:
    from rvc_train import (
        preprocess_dataset, extract_f0_feature, click_train, train_index,
        validate_model_and_index, separate_vocal, get_pretrained_models
    )
    rvc_train_available = True
except ImportError as e:
    rvc_train_available = False
    import_error = str(e)


class TestRVCTrainingLayout(unittest.TestCase):
    """Test cases for RVC training layout functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = TestConfig()
        self.temp_manager = TemporaryDirectoryManager()

        # Create test directories
        self.test_dir = self.temp_manager.create_temp_dir()
        self.dataset_dir = os.path.join(self.test_dir, 'dataset')
        self.exp_dir = os.path.join(self.test_dir, 'experiments')

        os.makedirs(self.dataset_dir)
        os.makedirs(self.exp_dir)

        # Create test audio files
        self.test_audio_files = []
        for i in range(3):
            audio_path = create_test_audio_file(
                duration=2.0,
                output_path=os.path.join(self.dataset_dir, f'test_audio_{i}.wav')
            )
            self.test_audio_files.append(audio_path)

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_manager.cleanup()

    @unittest.skipIf(not rvc_train_available, "RVC train module not available")
    def test_preprocess_dataset_basic(self):
        """Test basic dataset preprocessing functionality."""
        print("\nTesting preprocess_dataset...")

        # Mock the progress callback
        progress_mock = Mock()

        try:
            # This would normally require actual dataset files and models
            # For testing, we'll mock most of the functionality
            with patch('modules.rvc.infer.modules.train.preprocess.preprocess_trainset') as mock_preprocess:
                mock_preprocess.return_value = True

                # Test with minimal parameters
                result = preprocess_dataset(
                    trainset_dir=self.dataset_dir,
                    exp_dir=self.exp_dir,
                    sr=22050,
                    n_p=1,
                    progress=progress_mock
                )

                # The function should complete without errors
                self.assertIsNotNone(result)

        except Exception as e:
            self.fail(f"preprocess_dataset failed: {e}")

    @unittest.skipIf(not rvc_train_available, "RVC train module not available")
    def test_extract_f0_feature_basic(self):
        """Test F0 feature extraction functionality."""
        print("\nTesting extract_f0_feature...")

        progress_mock = Mock()

        try:
            # Mock the F0 extraction functions
            with patch('modules.rvc.infer.modules.train.extract.extract_f0_rmvpe.extract_f0_features_rmvpe') as mock_extract:
                mock_extract.return_value = True

                result = extract_f0_feature(
                    num_processors=1,
                    extract_method="rmvpe",
                    use_pitch_guidance=False,
                    exp_dir=self.exp_dir,
                    project_version="v1",
                    gpus_rmvpe="0",
                    progress=progress_mock
                )

                self.assertIsNotNone(result)

        except Exception as e:
            self.fail(f"extract_f0_feature failed: {e}")

    @unittest.skipIf(not rvc_train_available, "RVC train module not available")
    def test_validate_model_and_index_basic(self):
        """Test model and index validation."""
        print("\nTesting validate_model_and_index...")

        try:
            # Test with non-existent model (should handle gracefully)
            result = validate_model_and_index(
                project_name="non_existent_model",
                model_version="v1"
            )

            # Should return some validation result
            self.assertIsNotNone(result)

        except Exception as e:
            self.fail(f"validate_model_and_index failed: {e}")

    def test_separate_vocal_basic(self):
        """Test vocal separation functionality."""
        print("\nTesting separate_vocal...")

        if not rvc_train_available:
            self.skipTest("RVC train module not available")

        progress_mock = Mock()

        try:
            # Mock the separation process
            with patch('wrappers.separate.Separate') as mock_separate:
                mock_instance = Mock()
                mock_instance.separate.return_value = self.test_audio_files
                mock_separate.return_value = mock_instance

                result = separate_vocal(
                    audio_files=self.test_audio_files,
                    progress=progress_mock
                )

                # Should return separated files
                self.assertIsInstance(result, list)

        except Exception as e:
            self.fail(f"separate_vocal failed: {e}")

    def test_get_pretrained_models_basic(self):
        """Test pretrained model retrieval."""
        print("\nTesting get_pretrained_models...")

        if not rvc_train_available:
            self.skipTest("RVC train module not available")

        try:
            # Test with basic parameters
            result = get_pretrained_models(
                path_str="test_path",
                f0_str="test_f0",
                sr2=22050
            )

            # Should handle the request gracefully
            self.assertIsNotNone(result)

        except Exception as e:
            self.fail(f"get_pretrained_models failed: {e}")

    def test_click_train_mocked(self):
        """Test training workflow with mocked dependencies."""
        print("\nTesting click_train (mocked)...")

        if not rvc_train_available:
            self.skipTest("RVC train module not available")

        progress_mock = Mock()

        try:
            # Mock all the heavy dependencies
            with patch('modules.rvc.infer.modules.train.train.train_main') as mock_train, \
                 patch('modules.rvc.infer.modules.train.preprocess.preprocess_trainset') as mock_preprocess, \
                 patch('modules.rvc.infer.modules.train.extract.extract_f0_rmvpe.extract_f0_features_rmvpe') as mock_f0:

                # Setup mocks
                mock_train.return_value = True
                mock_preprocess.return_value = True
                mock_f0.return_value = True

                # Test training initiation
                result = click_train(
                    exp_dir=self.exp_dir,
                    sr=22050,
                    if_f0_3=True,
                    spk_id=0,
                    gpus="0",
                    version19="v1",
                    progress=progress_mock
                )

                # Should complete the training workflow
                self.assertIsNotNone(result)

        except Exception as e:
            self.fail(f"click_train failed: {e}")

    def test_integration_workflow(self):
        """Test complete RVC training workflow integration."""
        print("\nTesting RVC training workflow integration...")

        if not rvc_train_available:
            self.skipTest("RVC train module not available")

        # This would be a more comprehensive integration test
        # that exercises the full pipeline with mocked components

        # For now, we'll test the workflow coordination
        try:
            # Test that the main functions can be called in sequence
            # without major errors

            # 1. Preprocessing
            progress_mock = Mock()
            with patch('modules.rvc.infer.modules.train.preprocess.preprocess_trainset') as mock_preprocess:
                mock_preprocess.return_value = True
                preprocess_result = preprocess_dataset(
                    trainset_dir=self.dataset_dir,
                    exp_dir=self.exp_dir,
                    sr=22050,
                    n_p=1,
                    progress=progress_mock
                )

            # 2. F0 extraction
            with patch('modules.rvc.infer.modules.train.extract.extract_f0_rmvpe.extract_f0_features_rmvpe') as mock_f0:
                mock_f0.return_value = True
                f0_result = extract_f0_feature(
                    num_processors=1,
                    extract_method="rmvpe",
                    use_pitch_guidance=False,
                    exp_dir=self.exp_dir,
                    project_version="v1",
                    gpus_rmvpe="0",
                    progress=progress_mock
                )

            # 3. Training
            with patch('modules.rvc.infer.modules.train.train.train_main') as mock_train:
                mock_train.return_value = True
                train_result = click_train(
                    exp_dir=self.exp_dir,
                    sr=22050,
                    if_f0_3=True,
                    spk_id=0,
                    gpus="0",
                    version19="v1",
                    progress=progress_mock
                )

            # All functions should complete successfully
            self.assertIsNotNone(preprocess_result)
            self.assertIsNotNone(f0_result)
            self.assertIsNotNone(train_result)

        except Exception as e:
            self.fail(f"Integration workflow failed: {e}")


class TestRVCTrainingEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions for RVC training."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = TestConfig()
        self.temp_manager = TemporaryDirectoryManager()
        self.test_dir = self.temp_manager.create_temp_dir()

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_manager.cleanup()

    @unittest.skipIf(not rvc_train_available, "RVC train module not available")
    def test_invalid_audio_files(self):
        """Test handling of invalid audio files."""
        print("\nTesting invalid audio file handling...")

        # Create invalid audio file
        invalid_audio_path = os.path.join(self.test_dir, 'invalid.wav')

        # Write some non-audio data
        with open(invalid_audio_path, 'wb') as f:
            f.write(b'This is not audio data')

        progress_mock = Mock()

        try:
            # Should handle gracefully without crashing
            with patch('modules.rvc.infer.modules.train.preprocess.preprocess_trainset') as mock_preprocess:
                mock_preprocess.side_effect = Exception("Invalid audio format")

                result = preprocess_dataset(
                    trainset_dir=self.test_dir,
                    exp_dir=self.test_dir,
                    sr=22050,
                    n_p=1,
                    progress=progress_mock
                )

                # Should handle the error appropriately
                self.assertIsNotNone(result)

        except Exception as e:
            self.fail(f"Invalid audio handling failed: {e}")

    @unittest.skipIf(not rvc_train_available, "RVC train module not available")
    def test_empty_dataset(self):
        """Test handling of empty dataset."""
        print("\nTesting empty dataset handling...")

        empty_dir = os.path.join(self.test_dir, 'empty_dataset')
        os.makedirs(empty_dir)

        progress_mock = Mock()

        try:
            # Should handle empty dataset gracefully
            result = preprocess_dataset(
                trainset_dir=empty_dir,
                exp_dir=self.test_dir,
                sr=22050,
                n_p=1,
                progress=progress_mock
            )

            # Should return some result (likely indicating empty dataset)
            self.assertIsNotNone(result)

        except Exception as e:
            self.fail(f"Empty dataset handling failed: {e}")


if __name__ == '__main__':
    # Setup logging for tests
    logging.basicConfig(level=logging.INFO)

    # Run tests
    unittest.main(verbosity=2)
