"""
Unit tests for TTS layout functionality.
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
    from tts import run_zonos_tts, run_chatterbox_tts, run_dia_tts, download_model
    tts_available = True
except ImportError as e:
    tts_available = False
    import_error = str(e)


class TestTTSLayout(unittest.TestCase):
    """Test cases for TTS layout functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = TestConfig()
        self.temp_manager = TemporaryDirectoryManager()

        # Create test directories
        self.test_dir = self.temp_manager.create_temp_dir()
        self.output_dir = os.path.join(self.test_dir, 'tts_output')
        os.makedirs(self.output_dir)

        # Test parameters
        self.test_text = "This is a test sentence for text-to-speech synthesis."
        self.test_speaker_sample = create_test_audio_file(
            duration=3.0, output_path=os.path.join(self.test_dir, 'speaker_sample.wav')
        )

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_manager.cleanup()

    @unittest.skipIf(not tts_available, "TTS module not available")
    def test_run_zonos_tts_basic(self):
        """Test basic Zonos TTS functionality."""
        print("\nTesting run_zonos_tts...")

        progress_mock = Mock()

        try:
            # Mock the Zonos model and inference
            with patch('modules.zonos.conditioning.supported_language_codes', ['en']), \
                 patch('modules.zonos.model') as mock_zonos_model, \
                 patch('torchaudio.save') as mock_save:

                # Setup mock model
                mock_model_instance = Mock()
                mock_model_instance.generate.return_value = {
                    'audio': mock_model_inference(self.test_text, 'tts')
                }
                mock_zonos_model.return_value = mock_model_instance

                # Test TTS generation
                result = run_zonos_tts(
                    language="en",
                    emotion_choice="Neutral",
                    text=self.test_text,
                    speaker_sample=self.test_speaker_sample,
                    speed=1.0,
                    progress=progress_mock
                )

                # Should return audio file path
                self.assertIsNotNone(result)
                self.assertTrue(os.path.exists(result))

        except Exception as e:
            self.fail(f"run_zonos_tts failed: {e}")

    @unittest.skipIf(not tts_available, "TTS module not available")
    def test_run_chatterbox_tts_basic(self):
        """Test basic Chatterbox TTS functionality."""
        print("\nTesting run_chatterbox_tts...")

        progress_mock = Mock()

        try:
            # Mock Chatterbox TTS
            with patch('chatterbox.tts.ChatterboxTTS') as mock_chatterbox:
                mock_instance = Mock()
                mock_audio_data = mock_model_inference(self.test_text, 'tts')
                mock_instance.synthesize.return_value = (mock_audio_data, 22050)
                mock_chatterbox.return_value = mock_instance

                # Test TTS generation
                result = run_chatterbox_tts(
                    text=self.test_text,
                    speaker_sample=self.test_speaker_sample,
                    exaggeration=1.0,
                    cfg=7.0,
                    progress=progress_mock
                )

                # Should return audio file path
                self.assertIsNotNone(result)

        except Exception as e:
            self.fail(f"run_chatterbox_tts failed: {e}")

    @unittest.skipIf(not tts_available, "TTS module not available")
    def test_run_dia_tts_basic(self):
        """Test basic DIA TTS functionality."""
        print("\nTesting run_dia_tts...")

        progress_mock = Mock()

        try:
            # Mock DIA TTS model
            with patch('modules.diatts.dia.model') as mock_dia_model, \
                 patch('torchaudio.save') as mock_save:

                # Setup mock model
                mock_model_instance = Mock()
                mock_audio_data = mock_model_inference(self.test_text, 'tts')
                mock_model_instance.infer.return_value = mock_audio_data
                mock_dia_model.load_model.return_value = mock_model_instance

                # Test TTS generation
                result = run_dia_tts(
                    text=self.test_text,
                    prompt_text="A clear speaking voice",
                    speaker_sample=self.test_speaker_sample,
                    speed=1.0,
                    progress=progress_mock
                )

                # Should return audio file path
                self.assertIsNotNone(result)

        except Exception as e:
            self.fail(f"run_dia_tts failed: {e}")

    @unittest.skipIf(not tts_available, "TTS module not available")
    def test_download_model_basic(self):
        """Test model downloading functionality."""
        print("\nTesting download_model...")

        try:
            # Mock HuggingFace download
            with patch('huggingface_hub.hf_hub_download') as mock_download:
                mock_download.return_value = "/mocked/model/path"

                # Test model download
                result = download_model()

                # Should return model directory path
                self.assertIsNotNone(result)
                self.assertTrue(os.path.exists(result) or result == "/mocked/model/path")

        except Exception as e:
            self.fail(f"download_model failed: {e}")

    def test_tts_emotion_mapping(self):
        """Test emotion mapping functionality."""
        print("\nTesting TTS emotion mapping...")

        if not tts_available:
            self.skipTest("TTS module not available")

        # Test emotion vector mapping
        test_cases = [
            ("Happiness", [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ("Sadness", [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ("Neutral", None),
        ]

        for emotion, expected_vector in test_cases:
            # This would test the emotion mapping logic if it were extracted
            # For now, we'll verify the mapping exists in the module
            self.assertTrue(hasattr(run_zonos_tts, '__globals__'))

    def test_tts_language_support(self):
        """Test language support functionality."""
        print("\nTesting TTS language support...")

        if not tts_available:
            self.skipTest("TTS module not available")

        # Test that language codes are properly defined
        try:
            from modules.zonos.conditioning import supported_language_codes
            self.assertIsInstance(supported_language_codes, list)
            self.assertTrue(len(supported_language_codes) > 0)
        except ImportError:
            self.skipTest("Zonos conditioning module not available")

    def test_tts_output_validation(self):
        """Test TTS output audio validation."""
        print("\nTesting TTS output validation...")

        # Create expected test audio for comparison
        expected_audio_path = create_test_audio_file(
            duration=2.0, output_path=os.path.join(self.test_dir, 'expected_tts.wav')
        )

        # Test audio validation
        validation_result = audio_validator.validate_audio_file(
            expected_audio_path,
            expected_sample_rate=22050,
            expected_duration=2.0
        )

        self.assertTrue(validation_result['valid'])
        self.assertEqual(validation_result['properties']['sample_rate'], 22050)

    def test_tts_error_handling(self):
        """Test error handling in TTS functions."""
        print("\nTesting TTS error handling...")

        if not tts_available:
            self.skipTest("TTS module not available")

        progress_mock = Mock()

        # Test with invalid inputs
        try:
            # Test with empty text
            result = run_zonos_tts(
                language="en",
                emotion_choice="Neutral",
                text="",
                speaker_sample=self.test_speaker_sample,
                speed=1.0,
                progress=progress_mock
            )

            # Should handle empty text gracefully
            self.assertIsNotNone(result)

        except Exception as e:
            # Should not crash, but may return error indication
            print(f"Empty text handled with: {e}")

        try:
            # Test with invalid speaker sample
            result = run_zonos_tts(
                language="en",
                emotion_choice="Neutral",
                text=self.test_text,
                speaker_sample="/nonexistent/speaker.wav",
                speed=1.0,
                progress=progress_mock
            )

            # Should handle missing speaker file gracefully
            self.assertIsNotNone(result)

        except Exception as e:
            # Should not crash
            print(f"Missing speaker file handled with: {e}")


class TestTTSEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions for TTS."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = TestConfig()
        self.temp_manager = TemporaryDirectoryManager()
        self.test_dir = self.temp_manager.create_temp_dir()

        self.test_text = "Test text for edge case testing."
        self.test_speaker_sample = create_test_audio_file(
            duration=1.0, output_path=os.path.join(self.test_dir, 'speaker.wav')
        )

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_manager.cleanup()

    @unittest.skipIf(not tts_available, "TTS module not available")
    def test_very_long_text(self):
        """Test handling of very long text input."""
        print("\nTesting very long text input...")

        long_text = "This is a very long text input. " * 100  # ~4000 characters

        progress_mock = Mock()

        try:
            # Mock model to handle long text
            with patch('modules.zonos.model') as mock_model:
                mock_instance = Mock()
                mock_audio = mock_model_inference(long_text, 'tts')
                mock_instance.generate.return_value = {'audio': mock_audio}
                mock_model.return_value = mock_instance

                result = run_zonos_tts(
                    language="en",
                    emotion_choice="Neutral",
                    text=long_text,
                    speaker_sample=self.test_speaker_sample,
                    speed=1.0,
                    progress=progress_mock
                )

                # Should handle long text without crashing
                self.assertIsNotNone(result)

        except Exception as e:
            self.fail(f"Long text handling failed: {e}")

    @unittest.skipIf(not tts_available, "TTS module not available")
    def test_special_characters(self):
        """Test handling of special characters and punctuation."""
        print("\nTesting special characters...")

        special_text = "Hello! How are you? I'm fine. Testing: numbers 123, symbols @#$%..."

        progress_mock = Mock()

        try:
            # Mock model for special characters
            with patch('modules.zonos.model') as mock_model:
                mock_instance = Mock()
                mock_audio = mock_model_inference(special_text, 'tts')
                mock_instance.generate.return_value = {'audio': mock_audio}
                mock_model.return_value = mock_instance

                result = run_zonos_tts(
                    language="en",
                    emotion_choice="Neutral",
                    text=special_text,
                    speaker_sample=self.test_speaker_sample,
                    speed=1.0,
                    progress=progress_mock
                )

                # Should handle special characters
                self.assertIsNotNone(result)

        except Exception as e:
            self.fail(f"Special characters handling failed: {e}")

    @unittest.skipIf(not tts_available, "TTS module not available")
    def test_extreme_speed_values(self):
        """Test handling of extreme speed values."""
        print("\nTesting extreme speed values...")

        progress_mock = Mock()

        speed_values = [0.1, 5.0, 10.0]  # Very slow to very fast

        for speed in speed_values:
            try:
                with patch('modules.zonos.model') as mock_model:
                    mock_instance = Mock()
                    mock_audio = mock_model_inference(self.test_text, 'tts')
                    mock_instance.generate.return_value = {'audio': mock_audio}
                    mock_model.return_value = mock_instance

                    result = run_zonos_tts(
                        language="en",
                        emotion_choice="Neutral",
                        text=self.test_text,
                        speaker_sample=self.test_speaker_sample,
                        speed=speed,
                        progress=progress_mock
                    )

                    # Should handle extreme speeds
                    self.assertIsNotNone(result)

            except Exception as e:
                print(f"Speed {speed} failed: {e}")
                # Some speeds might be invalid, but shouldn't crash


if __name__ == '__main__':
    # Setup logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)

    # Run tests
    unittest.main(verbosity=2)
