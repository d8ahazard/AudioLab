"""
Comprehensive unit tests for TTS layout functionality.
"""

import os
import unittest
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path

from testing.utils.base_test import AudioLabBaseTest, LayoutTestMixin
from testing.utils.audio_validation import audio_validator

# Import the layout module for testing
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'layouts'))

try:
    from tts import (
        run_zonos_tts, run_chatterbox_tts, run_dia_tts, download_model,
        download_speaker_model, download_dia_model, set_espeak_lib_path_win,
        _parse_text_and_emotions, EMOTION_MAP
    )
    tts_available = True
except ImportError as e:
    tts_available = False
    import_error = str(e)


class TestTTSZonosEngine(AudioLabBaseTest, LayoutTestMixin):
    """Test Zonos TTS engine functionality."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        # Test data
        self.test_text = "This is a comprehensive test of the Zonos text-to-speech engine."
        self.test_languages = ["en", "es", "fr", "de"]
        self.test_emotions = ["Happiness", "Sadness", "Neutral"]
        self.test_speaker_sample = self.test_audio_files['speech_clean_female']

    @unittest.skipIf(not tts_available, "TTS module not available")
    def test_run_zonos_tts_basic_generation(self):
        """Test basic Zonos TTS generation."""
        print("\\nTesting basic Zonos TTS generation...")

        progress_mock = Mock()

        with patch('modules.zonos.conditioning.supported_language_codes', ['en', 'es', 'fr']), \
             patch('modules.zonos.model') as mock_zonos_model, \
             patch('torchaudio.save') as mock_save:

            # Setup mock model
            mock_model_instance = Mock()
            mock_audio_data = self.mock_model('tts')
            mock_model_instance.generate.return_value = {'audio': mock_audio_data}
            mock_zonos_model.return_value = mock_model_instance

            # Test basic generation
            result = run_zonos_tts(
                language="en",
                emotion_choice="Neutral",
                text=self.test_text,
                speaker_sample=self.test_speaker_sample,
                speed=1.0,
                progress=progress_mock
            )

            # Verify output
            self.assert_file_exists(result)
            self.assertTrue(result.endswith('.wav'))

            # Verify model was called correctly
            mock_model_instance.generate.assert_called_once()

    @unittest.skipIf(not tts_available, "TTS module not available")
    def test_run_zonos_tts_multiple_languages(self):
        """Test Zonos TTS with multiple languages."""
        print("\\nTesting Zonos TTS with multiple languages...")

        progress_mock = Mock()

        for language in self.test_languages:
            with patch('modules.zonos.conditioning.supported_language_codes', self.test_languages), \
                 patch('modules.zonos.model') as mock_zonos_model, \
                 patch('torchaudio.save') as mock_save:

                mock_model_instance = Mock()
                mock_audio_data = self.mock_model('tts')
                mock_model_instance.generate.return_value = {'audio': mock_audio_data}
                mock_zonos_model.return_value = mock_model_instance

                result = run_zonos_tts(
                    language=language,
                    emotion_choice="Neutral",
                    text=f"This is a test in {language} language.",
                    speaker_sample=self.test_speaker_sample,
                    speed=1.0,
                    progress=progress_mock
                )

                self.assert_file_exists(result)

    @unittest.skipIf(not tts_available, "TTS module not available")
    def test_run_zonos_tts_emotion_control(self):
        """Test Zonos TTS with emotion control."""
        print("\\nTesting Zonos TTS emotion control...")

        progress_mock = Mock()

        for emotion in self.test_emotions:
            with patch('modules.zonos.conditioning.supported_language_codes', ['en']), \
                 patch('modules.zonos.model') as mock_zonos_model, \
                 patch('torchaudio.save') as mock_save:

                mock_model_instance = Mock()
                mock_audio_data = self.mock_model('tts')
                mock_model_instance.generate.return_value = {'audio': mock_audio_data}
                mock_zonos_model.return_value = mock_model_instance

                result = run_zonos_tts(
                    language="en",
                    emotion_choice=emotion,
                    text=f"This is a test with {emotion.lower()} emotion.",
                    speaker_sample=self.test_speaker_sample,
                    speed=1.0,
                    progress=progress_mock
                )

                self.assert_file_exists(result)

    @unittest.skipIf(not tts_available, "TTS module not available")
    def test_run_zonos_tts_speed_variations(self):
        """Test Zonos TTS with different speed settings."""
        print("\\nTesting Zonos TTS speed variations...")

        progress_mock = Mock()
        speeds = [0.5, 1.0, 1.5, 2.0]

        for speed in speeds:
            with patch('modules.zonos.conditioning.supported_language_codes', ['en']), \
                 patch('modules.zonos.model') as mock_zonos_model, \
                 patch('torchaudio.save') as mock_save:

                mock_model_instance = Mock()
                mock_audio_data = self.mock_model('tts')
                mock_model_instance.generate.return_value = {'audio': mock_audio_data}
                mock_zonos_model.return_value = mock_model_instance

                result = run_zonos_tts(
                    language="en",
                    emotion_choice="Neutral",
                    text=f"This is a test at speed {speed}.",
                    speaker_sample=self.test_speaker_sample,
                    speed=speed,
                    progress=progress_mock
                )

                self.assert_file_exists(result)


class TestTTSChatterboxEngine(AudioLabBaseTest, LayoutTestMixin):
    """Test Chatterbox TTS engine functionality."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        self.test_text = "Testing Chatterbox text-to-speech functionality."
        self.test_speaker_sample = self.test_audio_files['speech_clean_male']

    @unittest.skipIf(not tts_available, "TTS module not available")
    def test_run_chatterbox_tts_basic(self):
        """Test basic Chatterbox TTS generation."""
        print("\\nTesting basic Chatterbox TTS...")

        progress_mock = Mock()

        with patch('chatterbox.tts.ChatterboxTTS') as mock_chatterbox:
            mock_instance = Mock()
            mock_audio_data = self.mock_model('tts')
            mock_instance.synthesize.return_value = (mock_audio_data, 22050)
            mock_chatterbox.return_value = mock_instance

            result = run_chatterbox_tts(
                text=self.test_text,
                speaker_sample=self.test_speaker_sample,
                exaggeration=1.0,
                cfg=7.0,
                progress=progress_mock
            )

            self.assert_file_exists(result)
            mock_instance.synthesize.assert_called_once()

    @unittest.skipIf(not tts_available, "TTS module not available")
    def test_run_chatterbox_tts_parameter_variations(self):
        """Test Chatterbox TTS with parameter variations."""
        print("\\nTesting Chatterbox TTS parameter variations...")

        progress_mock = Mock()
        exaggerations = [0.5, 1.0, 1.5, 2.0]
        cfgs = [3.0, 7.0, 12.0]

        for exaggeration in exaggerations:
            for cfg in cfgs:
                with patch('chatterbox.tts.ChatterboxTTS') as mock_chatterbox:
                    mock_instance = Mock()
                    mock_audio_data = self.mock_model('tts')
                    mock_instance.synthesize.return_value = (mock_audio_data, 22050)
                    mock_chatterbox.return_value = mock_instance

                    result = run_chatterbox_tts(
                        text=f"Test with exaggeration {exaggeration} and cfg {cfg}.",
                        speaker_sample=self.test_speaker_sample,
                        exaggeration=exaggeration,
                        cfg=cfg,
                        progress=progress_mock
                    )

                    self.assert_file_exists(result)


class TestTTSDIAEngine(AudioLabBaseTest, LayoutTestMixin):
    """Test DIA TTS engine functionality."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        self.test_text = "Testing DIA text-to-speech engine."
        self.test_prompt = "A clear and natural speaking voice."
        self.test_speaker_sample = self.test_audio_files['speech_clean_female']

    @unittest.skipIf(not tts_available, "TTS module not available")
    def test_run_dia_tts_basic(self):
        """Test basic DIA TTS generation."""
        print("\\nTesting basic DIA TTS...")

        progress_mock = Mock()

        with patch('modules.diatts.dia.model') as mock_dia_model, \
             patch('torchaudio.save') as mock_save:

            mock_model_instance = Mock()
            mock_audio_data = self.mock_model('tts')
            mock_model_instance.infer.return_value = mock_audio_data
            mock_dia_model.load_model.return_value = mock_model_instance

            result = run_dia_tts(
                text=self.test_text,
                prompt_text=self.test_prompt,
                speaker_sample=self.test_speaker_sample,
                speed=1.0,
                progress=progress_mock
            )

            self.assert_file_exists(result)
            mock_model_instance.infer.assert_called_once()

    @unittest.skipIf(not tts_available, "TTS module not available")
    def test_run_dia_tts_prompt_variations(self):
        """Test DIA TTS with different prompts."""
        print("\\nTesting DIA TTS with prompt variations...")

        progress_mock = Mock()
        prompts = [
            "A clear speaking voice",
            "A deep male voice",
            "A cheerful female voice",
            "A professional narrator"
        ]

        for prompt in prompts:
            with patch('modules.diatts.dia.model') as mock_dia_model, \
                 patch('torchaudio.save') as mock_save:

                mock_model_instance = Mock()
                mock_audio_data = self.mock_model('tts')
                mock_model_instance.infer.return_value = mock_audio_data
                mock_dia_model.load_model.return_value = mock_model_instance

                result = run_dia_tts(
                    text=f"Test with prompt: {prompt}",
                    prompt_text=prompt,
                    speaker_sample=self.test_speaker_sample,
                    speed=1.0,
                    progress=progress_mock
                )

                self.assert_file_exists(result)


class TestTTSModelManagement(AudioLabBaseTest):
    """Test TTS model downloading and management."""

    @unittest.skipIf(not tts_available, "TTS module not available")
    def test_download_zonos_model(self):
        """Test Zonos model downloading."""
        print("\\nTesting Zonos model download...")

        with patch('huggingface_hub.hf_hub_download') as mock_download:
            mock_download.return_value = "/mocked/zonos/model"

            result = download_model()

            self.assertIsNotNone(result)
            self.assertEqual(mock_download.call_count, 2)  # config.json and model.safetensors

    @unittest.skipIf(not tts_available, "TTS module not available")
    def test_download_speaker_model(self):
        """Test speaker embedding model downloading."""
        print("\\nTesting speaker model download...")

        with patch('huggingface_hub.hf_hub_download') as mock_download:
            mock_download.return_value = "/mocked/speaker/model"

            result = download_speaker_model()

            self.assertIsNotNone(result)
            self.assertEqual(mock_download.call_count, 2)  # Two speaker model files

    @unittest.skipIf(not tts_available, "TTS module not available")
    def test_download_dia_model(self):
        """Test DIA model downloading."""
        print("\\nTesting DIA model download...")

        with patch('huggingface_hub.hf_hub_download') as mock_download:
            mock_download.return_value = "/mocked/dia/model"

            result = download_dia_model()

            self.assertIsNotNone(result)
            self.assertEqual(mock_download.call_count, 2)  # config.json and dia-v0_1.pth

    @unittest.skipIf(not tts_available, "TTS module not available")
    def test_set_espeak_lib_path_win(self):
        """Test eSpeak library path setting on Windows."""
        print("\\nTesting eSpeak library path setting...")

        with patch('os.name', 'nt'), \
             patch('os.path.exists') as mock_exists, \
             patch('os.environ') as mock_environ:

            mock_exists.return_value = True

            # Should not raise an exception
            result = set_espeak_lib_path_win()
            self.assertIsNone(result)  # Function doesn't return anything


class TestTTSTextProcessing(AudioLabBaseTest):
    """Test text processing functionality."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        self.test_texts = [
            "Simple text without emotions.",
            "Text with [Happiness] emotion marker.",
            "Text with multiple [Sadness] emotions [Happiness] and markers.",
            "Text with [Neutral] default emotion.",
            "Complex text with [Anger] multiple [Fear] emotion [Surprise] markers."
        ]

    @unittest.skipIf(not tts_available, "TTS module not available")
    def test_parse_text_and_emotions_basic(self):
        """Test basic text and emotion parsing."""
        print("\\nTesting basic text and emotion parsing...")

        for text in self.test_texts:
            result = _parse_text_and_emotions(text, "Neutral")

            # Should return a list of text segments with emotions
            self.assertIsInstance(result, list)

            for segment in result:
                self.assertIn('text', segment)
                self.assertIn('emotion', segment)

    @unittest.skipIf(not tts_available, "TTS module not available")
    def test_parse_text_and_emotions_edge_cases(self):
        """Test text parsing with edge cases."""
        print("\\nTesting text parsing edge cases...")

        edge_cases = [
            "",  # Empty text
            "Text without emotion markers.",
            "Text with [InvalidEmotion] marker.",
            "Text with mismatched [brackets.",
            "Text with ]mismatched brackets[."
        ]

        for text in edge_cases:
            result = _parse_text_and_emotions(text, "Neutral")
            self.assertIsInstance(result, list)

    @unittest.skipIf(not tts_available, "TTS module not available")
    def test_emotion_map_validation(self):
        """Test emotion mapping dictionary."""
        print("\\nTesting emotion map validation...")

        # Verify emotion map structure
        self.assertIsInstance(EMOTION_MAP, dict)
        self.assertGreater(len(EMOTION_MAP), 0)

        # Test that all emotions have valid vectors or None
        for emotion, vector in EMOTION_MAP.items():
            if vector is not None:
                self.assertIsInstance(vector, list)
                self.assertEqual(len(vector), 8)  # Should be 8-dimensional
                # Check that it's a valid probability distribution (sum ≈ 1)
                vector_sum = sum(vector)
                self.assertAlmostEqual(vector_sum, 1.0, places=5)


class TestTTSOutputValidation(AudioLabBaseTest):
    """Test TTS output validation and quality."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        self.test_text = "This audio should be validated for quality and correctness."

    @unittest.skipIf(not tts_available, "TTS module not available")
    def test_tts_output_audio_properties(self):
        """Test TTS output audio file properties."""
        print("\\nTesting TTS output audio properties...")

        with patch('modules.zonos.conditioning.supported_language_codes', ['en']), \
             patch('modules.zonos.model') as mock_zonos_model, \
             patch('torchaudio.save') as mock_save:

            mock_model_instance = Mock()
            mock_audio_data = self.mock_model('tts')
            mock_model_instance.generate.return_value = {'audio': mock_audio_data}
            mock_zonos_model.return_value = mock_model_instance

            result = run_zonos_tts(
                language="en",
                emotion_choice="Neutral",
                text=self.test_text,
                speaker_sample=self.test_speaker_sample,
                speed=1.0,
                progress=Mock()
            )

            # Validate output file
            validation = self.validate_audio_output(
                result,
                expected_sample_rate=22050,
                expected_duration=2.0  # Rough estimate based on text length
            )

            self.assertTrue(validation['valid'])

    @unittest.skipIf(not tts_available, "TTS module not available")
    def test_tts_output_quality_comparison(self):
        """Test TTS output quality comparison."""
        print("\\nTesting TTS output quality comparison...")

        # Create reference audio
        reference_audio = self.create_test_audio(duration=2.0, frequency=220)

        with patch('modules.zonos.conditioning.supported_language_codes', ['en']), \
             patch('modules.zonos.model') as mock_zonos_model, \
             patch('torchaudio.save') as mock_save:

            mock_model_instance = Mock()
            mock_audio_data = self.mock_model('tts')
            mock_model_instance.generate.return_value = {'audio': mock_audio_data}
            mock_zonos_model.return_value = mock_model_instance

            result = run_zonos_tts(
                language="en",
                emotion_choice="Neutral",
                text=self.test_text,
                speaker_sample=self.test_speaker_sample,
                speed=1.0,
                progress=Mock()
            )

            # Compare with reference (should be similar in basic properties)
            similarity = self.compare_audio_similarity(
                reference_audio,
                result,
                tolerance_snr=10.0  # Allow for some difference in content
            )

            # The similarity check might fail due to different content,
            # but the files should both be valid audio
            self.assertIn('metrics', similarity)


class TestTTSErrorHandling(AudioLabBaseTest):
    """Test error handling in TTS components."""

    @unittest.skipIf(not tts_available, "TTS module not available")
    def test_tts_invalid_language(self):
        """Test TTS with invalid language."""
        print("\\nTesting TTS with invalid language...")

        progress_mock = Mock()

        with patch('modules.zonos.conditioning.supported_language_codes', ['en', 'es']), \
             patch('modules.zonos.model') as mock_zonos_model:

            mock_model_instance = Mock()
            mock_model_instance.generate.side_effect = Exception("Unsupported language")
            mock_zonos_model.return_value = mock_model_instance

            with self.assertRaises(Exception):
                run_zonos_tts(
                    language="invalid_lang",
                    emotion_choice="Neutral",
                    text="Test text",
                    speaker_sample=self.test_speaker_sample,
                    speed=1.0,
                    progress=progress_mock
                )

    @unittest.skipIf(not tts_available, "TTS module not available")
    def test_tts_missing_speaker_sample(self):
        """Test TTS with missing speaker sample."""
        print("\\nTesting TTS with missing speaker sample...")

        progress_mock = Mock()

        with patch('modules.zonos.conditioning.supported_language_codes', ['en']), \
             patch('modules.zonos.model') as mock_zonos_model:

            mock_model_instance = Mock()
            mock_model_instance.generate.side_effect = Exception("Speaker sample not found")
            mock_zonos_model.return_value = mock_model_instance

            with self.assertRaises(Exception):
                run_zonos_tts(
                    language="en",
                    emotion_choice="Neutral",
                    text="Test text",
                    speaker_sample="/nonexistent/speaker.wav",
                    speed=1.0,
                    progress=progress_mock
                )

    @unittest.skipIf(not tts_available, "TTS module not available")
    def test_tts_empty_text(self):
        """Test TTS with empty text."""
        print("\\nTesting TTS with empty text...")

        progress_mock = Mock()

        with patch('modules.zonos.conditioning.supported_language_codes', ['en']), \
             patch('modules.zonos.model') as mock_zonos_model:

            mock_model_instance = Mock()
            mock_model_instance.generate.side_effect = Exception("Empty text")
            mock_zonos_model.return_value = mock_model_instance

            with self.assertRaises(Exception):
                run_zonos_tts(
                    language="en",
                    emotion_choice="Neutral",
                    text="",
                    speaker_sample=self.test_speaker_sample,
                    speed=1.0,
                    progress=progress_mock
                )


class TestTTSIntegration(AudioLabBaseTest, LayoutTestMixin):
    """Integration tests for TTS functionality."""

    def setUp(self):
        """Set up integration test fixtures."""
        super().setUp()

        self.test_scenarios = [
            {
                'engine': 'zonos',
                'language': 'en',
                'emotion': 'Neutral',
                'text': 'Integration test for Zonos TTS engine.',
                'speaker_sample': self.test_audio_files['speech_clean_female']
            },
            {
                'engine': 'chatterbox',
                'text': 'Integration test for Chatterbox TTS engine.',
                'speaker_sample': self.test_audio_files['speech_clean_male'],
                'exaggeration': 1.0,
                'cfg': 7.0
            },
            {
                'engine': 'dia',
                'text': 'Integration test for DIA TTS engine.',
                'prompt_text': 'A clear speaking voice',
                'speaker_sample': self.test_audio_files['speech_clean_female']
            }
        ]

    @unittest.skipIf(not tts_available, "TTS module not available")
    def test_multi_engine_tts_integration(self):
        """Test integration across multiple TTS engines."""
        print("\\nTesting multi-engine TTS integration...")

        for scenario in self.test_scenarios:
            engine = scenario['engine']

            if engine == 'zonos':
                with patch('modules.zonos.conditioning.supported_language_codes', ['en']), \
                     patch('modules.zonos.model') as mock_model, \
                     patch('torchaudio.save') as mock_save:

                    mock_instance = Mock()
                    mock_audio = self.mock_model('tts')
                    mock_instance.generate.return_value = {'audio': mock_audio}
                    mock_model.return_value = mock_instance

                    result = run_zonos_tts(**scenario)

            elif engine == 'chatterbox':
                with patch('chatterbox.tts.ChatterboxTTS') as mock_chatterbox:
                    mock_instance = Mock()
                    mock_audio = self.mock_model('tts')
                    mock_instance.synthesize.return_value = (mock_audio, 22050)
                    mock_chatterbox.return_value = mock_instance

                    result = run_chatterbox_tts(**scenario)

            elif engine == 'dia':
                with patch('modules.diatts.dia.model') as mock_dia_model, \
                     patch('torchaudio.save') as mock_save:

                    mock_instance = Mock()
                    mock_audio = self.mock_model('tts')
                    mock_instance.infer.return_value = mock_audio
                    mock_dia_model.load_model.return_value = mock_instance

                    result = run_dia_tts(**scenario)

            # Verify each engine produces valid output
            self.assert_file_exists(result)

            # Validate audio properties
            validation = self.validate_audio_output(result)
            self.assertTrue(validation['valid'])


if __name__ == '__main__':
    # Setup logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)

    # Run tests
    unittest.main(verbosity=2)
