"""
Comprehensive unit tests for RVC training layout functionality.
"""

import os
import unittest
import tempfile
import shutil
import json
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path

from testing.utils.base_test import AudioLabBaseTest, LayoutTestMixin, create_test_dataset
from testing.utils.audio_validation import audio_validator

# Import the layout module for testing
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'layouts'))

try:
    from rvc_train import (
        preprocess_dataset, extract_f0_feature, click_train, train_index,
        validate_model_and_index, separate_vocal, get_pretrained_models,
        change_sr2, change_version19, change_f0, change_f0_method,
        preprocess_dataset, extract_f0_feature, click_train
    )
    rvc_train_available = True
except ImportError as e:
    rvc_train_available = False
    import_error = str(e)


class TestRVCTrainingPreprocessDataset(AudioLabBaseTest, LayoutTestMixin):
    """Test dataset preprocessing functionality."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        # Create test dataset
        self.dataset_dir = create_test_dataset(5, os.path.join(self.test_dir, 'test_dataset'))
        self.exp_dir = os.path.join(self.test_dir, 'experiments')
        os.makedirs(self.exp_dir)

    @unittest.skipIf(not rvc_train_available, "RVC train module not available")
    def test_preprocess_dataset_valid_input(self):
        """Test preprocessing with valid dataset."""
        print("\\nTesting preprocess_dataset with valid input...")

        progress_mock = Mock()

        with patch('modules.rvc.infer.modules.train.preprocess.preprocess_trainset') as mock_preprocess:
            mock_preprocess.return_value = True

            result = preprocess_dataset(
                trainset_dir=self.dataset_dir,
                exp_dir=self.exp_dir,
                sr=22050,
                n_p=2,
                progress=progress_mock
            )

            # Verify the preprocessing function was called
            mock_preprocess.assert_called_once()
            self.assertIsNotNone(result)

    @unittest.skipIf(not rvc_train_available, "RVC train module not available")
    def test_preprocess_dataset_empty_dataset(self):
        """Test preprocessing with empty dataset."""
        print("\\nTesting preprocess_dataset with empty dataset...")

        empty_dir = os.path.join(self.test_dir, 'empty_dataset')
        os.makedirs(empty_dir)

        progress_mock = Mock()

        with patch('modules.rvc.infer.modules.train.preprocess.preprocess_trainset') as mock_preprocess:
            mock_preprocess.return_value = True

            result = preprocess_dataset(
                trainset_dir=empty_dir,
                exp_dir=self.exp_dir,
                sr=22050,
                n_p=1,
                progress=progress_mock
            )

            # Should handle empty dataset gracefully
            self.assertIsNotNone(result)

    @unittest.skipIf(not rvc_train_available, "RVC train module not available")
    def test_preprocess_dataset_invalid_sample_rate(self):
        """Test preprocessing with invalid sample rate."""
        print("\\nTesting preprocess_dataset with invalid sample rate...")

        progress_mock = Mock()

        with patch('modules.rvc.infer.modules.train.preprocess.preprocess_trainset') as mock_preprocess:
            mock_preprocess.side_effect = Exception("Invalid sample rate")

            with self.assertRaises(Exception):
                preprocess_dataset(
                    trainset_dir=self.dataset_dir,
                    exp_dir=self.exp_dir,
                    sr=999999,  # Invalid sample rate
                    n_p=1,
                    progress=progress_mock
                )


class TestRVCTrainingF0Extraction(AudioLabBaseTest, LayoutTestMixin):
    """Test F0 feature extraction functionality."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.exp_dir = os.path.join(self.test_dir, 'experiments')
        os.makedirs(self.exp_dir)

    @unittest.skipIf(not rvc_train_available, "RVC train module not available")
    def test_extract_f0_feature_rmvpe(self):
        """Test F0 extraction with RMVPE method."""
        print("\\nTesting F0 extraction with RMVPE...")

        progress_mock = Mock()

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

            mock_extract.assert_called_once()
            self.assertIsNotNone(result)

    @unittest.skipIf(not rvc_train_available, "RVC train module not available")
    def test_extract_f0_feature_dio(self):
        """Test F0 extraction with DIO method."""
        print("\\nTesting F0 extraction with DIO...")

        progress_mock = Mock()

        with patch('modules.rvc.infer.modules.train.extract.extract_f0_print.extract_f0_features') as mock_extract:
            mock_extract.return_value = True

            result = extract_f0_feature(
                num_processors=1,
                extract_method="dio",
                use_pitch_guidance=False,
                exp_dir=self.exp_dir,
                project_version="v1",
                gpus_rmvpe="0",
                progress=progress_mock
            )

            mock_extract.assert_called_once()
            self.assertIsNotNone(result)

    @unittest.skipIf(not rvc_train_available, "RVC train module not available")
    def test_extract_f0_feature_with_guidance(self):
        """Test F0 extraction with pitch guidance."""
        print("\\nTesting F0 extraction with pitch guidance...")

        progress_mock = Mock()

        with patch('modules.rvc.infer.modules.train.extract.extract_f0_rmvpe.extract_f0_features_rmvpe') as mock_extract:
            mock_extract.return_value = True

            result = extract_f0_feature(
                num_processors=2,
                extract_method="rmvpe",
                use_pitch_guidance=True,
                exp_dir=self.exp_dir,
                project_version="v1",
                gpus_rmvpe="0,1",
                progress=progress_mock
            )

            self.assertIsNotNone(result)


class TestRVCTrainingWorkflow(AudioLabBaseTest, LayoutTestMixin):
    """Test complete RVC training workflow."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        # Create test dataset and experiment directory
        self.dataset_dir = create_test_dataset(10, os.path.join(self.test_dir, 'full_dataset'))
        self.exp_dir = os.path.join(self.test_dir, 'full_training')
        os.makedirs(self.exp_dir)

    @unittest.skipIf(not rvc_train_available, "RVC train module not available")
    def test_click_train_basic_workflow(self):
        """Test basic training workflow."""
        print("\\nTesting basic training workflow...")

        progress_mock = Mock()

        with patch('modules.rvc.infer.modules.train.train.train_main') as mock_train_main:
            mock_train_main.return_value = True

            result = click_train(
                exp_dir=self.exp_dir,
                sr=22050,
                if_f0_3=True,
                spk_id=0,
                gpus="0",
                version19="v1",
                progress=progress_mock
            )

            # Should complete training workflow
            self.assertIsNotNone(result)

    @unittest.skipIf(not rvc_train_available, "RVC train module not available")
    def test_click_train_with_custom_parameters(self):
        """Test training with custom parameters."""
        print("\\nTesting training with custom parameters...")

        progress_mock = Mock()

        with patch('modules.rvc.infer.modules.train.train.train_main') as mock_train_main:
            mock_train_main.return_value = True

            result = click_train(
                exp_dir=self.exp_dir,
                sr=44100,
                if_f0_3=False,
                spk_id=1,
                gpus="0,1",
                version19="v2",
                progress=progress_mock
            )

            self.assertIsNotNone(result)

    @unittest.skipIf(not rvc_train_available, "RVC train module not available")
    def test_train_index_creation(self):
        """Test index training after model training."""
        print("\\nTesting index training...")

        progress_mock = Mock()

        with patch('faiss.read_index'), \
             patch('faiss.write_index'):

            result = train_index(
                project_name="test_model",
                model_version="v1"
            )

            self.assertIsNotNone(result)


class TestRVCTrainingValidation(AudioLabBaseTest):
    """Test model and dataset validation."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        # Create mock model files
        self.model_dir = os.path.join(self.test_dir, 'models')
        os.makedirs(self.model_dir)

    @unittest.skipIf(not rvc_train_available, "RVC train module not available")
    def test_validate_model_and_index_valid(self):
        """Test validation of valid model and index."""
        print("\\nTesting model and index validation...")

        # Create mock model file
        model_path = os.path.join(self.model_dir, 'test_model_v1.pth')
        with open(model_path, 'wb') as f:
            f.write(b'mock_model_data')

        result = validate_model_and_index(
            project_name="test_model",
            model_version="v1"
        )

        self.assertIsNotNone(result)

    @unittest.skipIf(not rvc_train_available, "RVC train module not available")
    def test_validate_model_and_index_missing_files(self):
        """Test validation with missing model files."""
        print("\\nTesting validation with missing files...")

        result = validate_model_and_index(
            project_name="nonexistent_model",
            model_version="v1"
        )

        # Should handle missing files gracefully
        self.assertIsNotNone(result)


class TestRVCTrainingUtils(AudioLabBaseTest):
    """Test utility functions in RVC training."""

    @unittest.skipIf(not rvc_train_available, "RVC train module not available")
    def test_change_sr2_functionality(self):
        """Test sample rate change functionality."""
        print("\\nTesting sample rate change utility...")

        # Test different sample rate configurations
        test_cases = [
            (22050, True, "v1"),
            (44100, False, "v2"),
            (48000, True, "v1")
        ]

        for sr2, if_f0_3, version19 in test_cases:
            result = change_sr2(sr2, if_f0_3, version19)
            self.assertIsNotNone(result)

    @unittest.skipIf(not rvc_train_available, "RVC train module not available")
    def test_change_version19_functionality(self):
        """Test version change functionality."""
        print("\\nTesting version change utility...")

        test_cases = [
            (22050, True, "v1"),
            (22050, False, "v2"),
            (44100, True, "v1")
        ]

        for sr2, if_f0_3, version19 in test_cases:
            result = change_version19(sr2, if_f0_3, version19)
            self.assertIsNotNone(result)

    @unittest.skipIf(not rvc_train_available, "RVC train module not available")
    def test_change_f0_functionality(self):
        """Test F0 parameter change functionality."""
        print("\\nTesting F0 parameter change utility...")

        test_cases = [
            (True, 22050, "v1"),
            (False, 22050, "v2"),
            (True, 44100, "v1")
        ]

        for if_f0_3, sr2, version19 in test_cases:
            result = change_f0(if_f0_3, sr2, version19)
            self.assertIsNotNone(result)

    @unittest.skipIf(not rvc_train_available, "RVC train module not available")
    def test_change_f0_method_functionality(self):
        """Test F0 method change functionality."""
        print("\\nTesting F0 method change utility...")

        methods = ["rmvpe", "dio", "harvest", "crepe"]

        for method in methods:
            result = change_f0_method(method)
            self.assertIsNotNone(result)


class TestRVCTrainingVocalSeparation(AudioLabBaseTest):
    """Test vocal separation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        # Create test audio with mixed content
        self.mixed_audio = self.create_test_audio(duration=3.0, frequency=300)

    @unittest.skipIf(not rvc_train_available, "RVC train module not available")
    def test_separate_vocal_basic(self):
        """Test basic vocal separation."""
        print("\\nTesting vocal separation...")

        progress_mock = Mock()

        with patch('wrappers.separate.Separate') as mock_separate:
            mock_instance = Mock()
            mock_instance.separate.return_value = [self.mixed_audio]
            mock_separate.return_value = mock_instance

            result = separate_vocal(
                audio_files=[self.mixed_audio],
                progress=progress_mock
            )

            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 1)

    @unittest.skipIf(not rvc_train_available, "RVC train module not available")
    def test_separate_vocal_multiple_files(self):
        """Test vocal separation with multiple files."""
        print("\\nTesting vocal separation with multiple files...")

        # Create multiple test files
        audio_files = [
            self.create_test_audio(duration=2.0, frequency=200),
            self.create_test_audio(duration=3.0, frequency=300),
            self.create_test_audio(duration=2.5, frequency=250)
        ]

        progress_mock = Mock()

        with patch('wrappers.separate.Separate') as mock_separate:
            mock_instance = Mock()
            mock_instance.separate.return_value = audio_files
            mock_separate.return_value = mock_instance

            result = separate_vocal(
                audio_files=audio_files,
                progress=progress_mock
            )

            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 3)


class TestRVCTrainingErrorHandling(AudioLabBaseTest):
    """Test error handling in RVC training components."""

    @unittest.skipIf(not rvc_train_available, "RVC train module not available")
    def test_preprocess_dataset_permission_error(self):
        """Test preprocessing with permission error."""
        print("\\nTesting preprocessing permission error handling...")

        # Create a read-only directory
        readonly_dir = os.path.join(self.test_dir, 'readonly')
        os.makedirs(readonly_dir)
        os.chmod(readonly_dir, 0o444)  # Read-only

        progress_mock = Mock()

        try:
            with patch('modules.rvc.infer.modules.train.preprocess.preprocess_trainset') as mock_preprocess:
                mock_preprocess.side_effect = PermissionError("Permission denied")

                with self.assertRaises(PermissionError):
                    preprocess_dataset(
                        trainset_dir=readonly_dir,
                        exp_dir=self.test_dir,
                        sr=22050,
                        n_p=1,
                        progress=progress_mock
                    )
        finally:
            # Restore permissions for cleanup
            os.chmod(readonly_dir, 0o755)

    @unittest.skipIf(not rvc_train_available, "RVC train module not available")
    def test_extract_f0_feature_gpu_error(self):
        """Test F0 extraction with GPU error."""
        print("\\nTesting F0 extraction GPU error handling...")

        progress_mock = Mock()

        with patch('modules.rvc.infer.modules.train.extract.extract_f0_rmvpe.extract_f0_features_rmvpe') as mock_extract:
            mock_extract.side_effect = RuntimeError("CUDA error")

            with self.assertRaises(RuntimeError):
                extract_f0_feature(
                    num_processors=1,
                    extract_method="rmvpe",
                    use_pitch_guidance=False,
                    exp_dir=self.test_dir,
                    project_version="v1",
                    gpus_rmvpe="0",
                    progress=progress_mock
                )


class TestRVCTrainingIntegration(AudioLabBaseTest, LayoutTestMixin):
    """Integration tests for RVC training workflow."""

    def setUp(self):
        """Set up integration test fixtures."""
        super().setUp()

        # Create full test environment
        self.dataset_dir = create_test_dataset(20, os.path.join(self.test_dir, 'integration_dataset'))
        self.exp_dir = os.path.join(self.test_dir, 'integration_training')
        os.makedirs(self.exp_dir)

    @unittest.skipIf(not rvc_train_available, "RVC train module not available")
    def test_full_training_pipeline_integration(self):
        """Test complete training pipeline integration."""
        print("\\nTesting full training pipeline integration...")

        progress_mock = Mock()

        # Mock all the major components
        with patch('modules.rvc.infer.modules.train.preprocess.preprocess_trainset') as mock_preprocess, \
             patch('modules.rvc.infer.modules.train.extract.extract_f0_rmvpe.extract_f0_features_rmvpe') as mock_f0, \
             patch('modules.rvc.infer.modules.train.train.train_main') as mock_train, \
             patch('faiss.read_index') as mock_faiss_read, \
             patch('faiss.write_index') as mock_faiss_write:

            # Setup mocks
            mock_preprocess.return_value = True
            mock_f0.return_value = True
            mock_train.return_value = True
            mock_faiss_read.return_value = Mock()
            mock_faiss_write.return_value = True

            # Execute full pipeline
            # 1. Preprocessing
            preprocess_result = preprocess_dataset(
                trainset_dir=self.dataset_dir,
                exp_dir=self.exp_dir,
                sr=22050,
                n_p=2,
                progress=progress_mock
            )

            # 2. F0 extraction
            f0_result = extract_f0_feature(
                num_processors=2,
                extract_method="rmvpe",
                use_pitch_guidance=True,
                exp_dir=self.exp_dir,
                project_version="v1",
                gpus_rmvpe="0,1",
                progress=progress_mock
            )

            # 3. Training
            train_result = click_train(
                exp_dir=self.exp_dir,
                sr=22050,
                if_f0_3=True,
                spk_id=0,
                gpus="0,1",
                version19="v1",
                progress=progress_mock
            )

            # 4. Index training
            index_result = train_index(
                project_name="integration_test_model",
                model_version="v1"
            )

            # Verify all components completed
            self.assertIsNotNone(preprocess_result)
            self.assertIsNotNone(f0_result)
            self.assertIsNotNone(train_result)
            self.assertIsNotNone(index_result)

            # Verify all mocks were called
            mock_preprocess.assert_called_once()
            mock_f0.assert_called_once()
            mock_train.assert_called_once()

    @unittest.skipIf(not rvc_train_available, "RVC train module not available")
    def test_training_pipeline_error_recovery(self):
        """Test error recovery in training pipeline."""
        print("\\nTesting training pipeline error recovery...")

        progress_mock = Mock()

        # Test with preprocessing failure
        with patch('modules.rvc.infer.modules.train.preprocess.preprocess_trainset') as mock_preprocess:
            mock_preprocess.side_effect = Exception("Preprocessing failed")

            with self.assertRaises(Exception):
                preprocess_dataset(
                    trainset_dir=self.dataset_dir,
                    exp_dir=self.exp_dir,
                    sr=22050,
                    n_p=1,
                    progress=progress_mock
                )

            # Verify progress was updated about the error
            # (This would depend on the actual implementation)


if __name__ == '__main__':
    # Setup logging for tests
    logging.basicConfig(level=logging.INFO)

    # Run tests
    unittest.main(verbosity=2)
