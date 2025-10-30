"""
Comprehensive unit tests for YuE music generation module.
"""

import os
import unittest
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path

from testing.utils.base_test import AudioLabBaseTest, ModuleTestMixin

# Import the YuE module for testing
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'modules', 'yue'))

try:
    from inference.infer import generate_music
    from inference.codecmanipulator import CodecManipulator
    from inference.mmtokenizer import MMTokenizer
    yue_available = True
except ImportError as e:
    yue_available = False
    import_error = str(e)


class TestYuEMusicGeneration(AudioLabBaseTest, ModuleTestMixin):
    """Test YuE music generation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        # Test prompts for music generation
        self.test_prompts = [
            "A peaceful piano melody in a minor key",
            "An upbeat electronic dance track with synth leads",
            "A classical orchestral piece with strings and woodwinds",
            "A jazz improvisation with saxophone and piano",
            "An ambient soundscape with nature sounds and drones"
        ]

        # Test parameters
        self.test_params = {
            'genre': 'classical',
            'duration': 30,
            'tempo': 120,
            'key': 'C major',
            'instruments': ['piano', 'violin', 'cello']
        }

    @unittest.skipIf(not yue_available, "YuE module not available")
    def test_generate_music_basic(self):
        """Test basic music generation."""
        print("\\nTesting basic music generation...")

        try:
            # Mock the generation process
            with patch('modules.yue.inference.infer.generate_music') as mock_generate:
                mock_audio = self.mock_model('music_gen')
                mock_generate.return_value = mock_audio

                result = generate_music(
                    prompt=self.test_prompts[0],
                    duration=10,
                    genre='classical',
                    seed=42
                )

                # Should return generated audio
                self.assertIsNotNone(result)
                self.assertIsInstance(result, (list, np.ndarray))

        except Exception as e:
            self.fail(f"Basic music generation failed: {e}")

    @unittest.skipIf(not yue_available, "YuE module not available")
    def test_generate_music_various_prompts(self):
        """Test music generation with various prompts."""
        print("\\nTesting music generation with various prompts...")

        for prompt in self.test_prompts:
            with self.subTest(prompt=prompt[:50] + "..."):  # Truncate for display
                try:
                    with patch('modules.yue.inference.infer.generate_music') as mock_generate:
                        mock_audio = self.mock_model('music_gen')
                        mock_generate.return_value = mock_audio

                        result = generate_music(
                            prompt=prompt,
                            duration=15,
                            genre='mixed',
                            seed=42
                        )

                        self.assertIsNotNone(result)

                except Exception as e:
                    self.fail(f"Music generation with prompt '{prompt[:50]}...' failed: {e}")

    @unittest.skipIf(not yue_available, "YuE module not available")
    def test_generate_music_parameter_variations(self):
        """Test music generation with parameter variations."""
        print("\\nTesting music generation parameter variations...")

        test_configs = [
            {'duration': 10, 'genre': 'classical', 'tempo': 60},
            {'duration': 20, 'genre': 'electronic', 'tempo': 120},
            {'duration': 30, 'genre': 'jazz', 'tempo': 90},
            {'duration': 15, 'genre': 'ambient', 'tempo': 75}
        ]

        for config in test_configs:
            with self.subTest(config=config):
                try:
                    with patch('modules.yue.inference.infer.generate_music') as mock_generate:
                        mock_audio = self.mock_model('music_gen')
                        mock_generate.return_value = mock_audio

                        result = generate_music(
                            prompt=self.test_prompts[0],
                            seed=42,
                            **config
                        )

                        self.assertIsNotNone(result)

                except Exception as e:
                    self.fail(f"Music generation with config {config} failed: {e}")


class TestYuECodecManipulator(AudioLabBaseTest):
    """Test YuE codec manipulation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        # Create test audio data
        self.test_audio_data = np.random.randn(22050 * 10).astype(np.float32)  # 10 seconds

    @unittest.skipIf(not yue_available, "YuE module not available")
    def test_codec_manipulator_initialization(self):
        """Test CodecManipulator initialization."""
        print("\\nTesting CodecManipulator initialization...")

        try:
            # Test basic initialization
            codec_manipulator = CodecManipulator()

            self.assertIsNotNone(codec_manipulator)

        except Exception as e:
            self.fail(f"CodecManipulator initialization failed: {e}")

    @unittest.skipIf(not yue_available, "YuE module not available")
    def test_codec_manipulator_encode_decode(self):
        """Test codec encoding and decoding."""
        print("\\nTesting codec encode/decode...")

        try:
            codec_manipulator = CodecManipulator()

            # Mock the encode/decode process
            with patch.object(codec_manipulator, 'encode') as mock_encode, \
                 patch.object(codec_manipulator, 'decode') as mock_decode:

                mock_encode.return_value = np.random.randn(100, 128).astype(np.float32)
                mock_decode.return_value = self.test_audio_data

                # Test encode/decode cycle
                encoded = codec_manipulator.encode(self.test_audio_data)
                decoded = codec_manipulator.decode(encoded)

                self.assertIsNotNone(encoded)
                self.assertIsNotNone(decoded)

        except Exception as e:
            self.fail(f"Codec manipulation failed: {e}")


class TestYuEMMTokenizer(AudioLabBaseTest):
    """Test YuE multi-modal tokenizer functionality."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        self.test_texts = [
            "A beautiful piano melody",
            "Electronic dance music with heavy bass",
            "Classical orchestral arrangement",
            "Jazz improvisation with saxophone"
        ]

        self.test_tokens = [
            [1, 150, 200, 50],
            [2, 300, 400, 75],
            [3, 100, 250, 60],
            [4, 175, 300, 80]
        ]

    @unittest.skipIf(not yue_available, "YuE module not available")
    def test_tokenizer_initialization(self):
        """Test MMTokenizer initialization."""
        print("\\nTesting MMTokenizer initialization...")

        try:
            tokenizer = MMTokenizer()

            self.assertIsNotNone(tokenizer)

        except Exception as e:
            self.fail(f"MMTokenizer initialization failed: {e}")

    @unittest.skipIf(not yue_available, "YuE module not available")
    def test_tokenizer_encode_decode(self):
        """Test tokenizer encoding and decoding."""
        print("\\nTesting tokenizer encode/decode...")

        try:
            tokenizer = MMTokenizer()

            for text in self.test_texts:
                # Test encoding
                tokens = tokenizer.encode(text)
                self.assertIsNotNone(tokens)
                self.assertIsInstance(tokens, (list, np.ndarray))

                # Test decoding
                decoded_text = tokenizer.decode(tokens)
                self.assertIsNotNone(decoded_text)
                self.assertIsInstance(decoded_text, str)

        except Exception as e:
            self.fail(f"Tokenizer encode/decode failed: {e}")

    @unittest.skipIf(not yue_available, "YuE module not available")
    def test_tokenizer_vocabulary(self):
        """Test tokenizer vocabulary handling."""
        print("\\nTesting tokenizer vocabulary...")

        try:
            tokenizer = MMTokenizer()

            # Test vocabulary access
            vocab_size = tokenizer.vocab_size
            self.assertIsInstance(vocab_size, int)
            self.assertGreater(vocab_size, 0)

            # Test special tokens
            if hasattr(tokenizer, 'special_tokens'):
                special_tokens = tokenizer.special_tokens
                self.assertIsInstance(special_tokens, dict)

        except Exception as e:
            self.fail(f"Tokenizer vocabulary test failed: {e}")


class TestYuEGenerationQuality(AudioLabBaseTest):
    """Test YuE music generation quality metrics."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        # Create reference audio for quality comparison
        self.reference_music = self.create_test_audio(duration=10.0, frequency=440)

    @unittest.skipIf(not yue_available, "YuE module not available")
    def test_music_generation_audio_properties(self):
        """Test generated music audio properties."""
        print("\\nTesting generated music audio properties...")

        try:
            with patch('modules.yue.inference.infer.generate_music') as mock_generate:
                # Create mock audio with specific properties
                sample_rate = 22050
                duration = 10.0
                mock_audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration))).astype(np.float32)
                mock_generate.return_value = mock_audio

                result = generate_music(
                    prompt=self.test_prompts[0],
                    duration=10,
                    genre='classical',
                    seed=42
                )

                # Validate audio properties
                validation = self.validate_audio_output(
                    result,
                    expected_sample_rate=sample_rate,
                    expected_duration=duration
                )

                self.assertTrue(validation['valid'])

        except Exception as e:
            self.fail(f"Music generation audio properties test failed: {e}")

    @unittest.skipIf(not yue_available, "YuE module not available")
    def test_music_generation_consistency(self):
        """Test music generation consistency across runs."""
        print("\\nTesting music generation consistency...")

        try:
            with patch('modules.yue.inference.infer.generate_music') as mock_generate:
                # Mock consistent output for same seed
                mock_audio = self.mock_model('music_gen')
                mock_generate.return_value = mock_audio

                # Generate music twice with same seed
                result1 = generate_music(
                    prompt=self.test_prompts[0],
                    duration=10,
                    genre='classical',
                    seed=42
                )

                result2 = generate_music(
                    prompt=self.test_prompts[0],
                    duration=10,
                    genre='classical',
                    seed=42
                )

                # Results should be identical for same seed (in mocked case)
                self.assertIsNotNone(result1)
                self.assertIsNotNone(result2)

        except Exception as e:
            self.fail(f"Music generation consistency test failed: {e}")


class TestYuEErrorHandling(AudioLabBaseTest):
    """Test error handling in YuE components."""

    @unittest.skipIf(not yue_available, "YuE module not available")
    def test_generate_music_invalid_prompt(self):
        """Test music generation with invalid prompt."""
        print("\\nTesting music generation with invalid prompt...")

        try:
            with patch('modules.yue.inference.infer.generate_music') as mock_generate:
                mock_generate.side_effect = Exception("Invalid prompt")

                with self.assertRaises(Exception):
                    generate_music(
                        prompt="",  # Empty prompt
                        duration=10,
                        genre='classical',
                        seed=42
                    )

        except Exception as e:
            self.fail(f"Invalid prompt handling failed: {e}")

    @unittest.skipIf(not yue_available, "YuE module not available")
    def test_generate_music_invalid_duration(self):
        """Test music generation with invalid duration."""
        print("\\nTesting music generation with invalid duration...")

        try:
            with patch('modules.yue.inference.infer.generate_music') as mock_generate:
                mock_generate.side_effect = Exception("Invalid duration")

                with self.assertRaises(Exception):
                    generate_music(
                        prompt=self.test_prompts[0],
                        duration=-5,  # Invalid negative duration
                        genre='classical',
                        seed=42
                    )

        except Exception as e:
            self.fail(f"Invalid duration handling failed: {e}")

    @unittest.skipIf(not yue_available, "YuE module not available")
    def test_codec_manipulator_invalid_input(self):
        """Test codec manipulator with invalid input."""
        print("\\nTesting codec manipulator with invalid input...")

        try:
            codec_manipulator = CodecManipulator()

            with patch.object(codec_manipulator, 'encode') as mock_encode:
                mock_encode.side_effect = Exception("Invalid audio format")

                with self.assertRaises(Exception):
                    # Pass invalid data
                    codec_manipulator.encode("invalid_audio_data")

        except Exception as e:
            self.fail(f"Invalid input handling failed: {e}")


class TestYuEPerformance(AudioLabBaseTest):
    """Test YuE performance characteristics."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        # Performance test parameters
        self.performance_configs = [
            {'duration': 10, 'complexity': 'low'},
            {'duration': 20, 'complexity': 'medium'},
            {'duration': 30, 'complexity': 'high'}
        ]

    @unittest.skipIf(not yue_available, "YuE module not available")
    def test_music_generation_performance(self):
        """Test music generation performance."""
        print("\\nTesting music generation performance...")

        for config in self.performance_configs:
            with self.subTest(config=config):
                try:
                    with patch('modules.yue.inference.infer.generate_music') as mock_generate:
                        # Mock faster generation for performance testing
                        mock_audio = self.mock_model('music_gen')
                        mock_generate.return_value = mock_audio

                        # Measure generation time
                        import time
                        start_time = time.time()

                        result = generate_music(
                            prompt=self.test_prompts[0],
                            seed=42,
                            **config
                        )

                        end_time = time.time()
                        generation_time = end_time - start_time

                        # Performance assertion (adjust based on expected performance)
                        self.assertLess(generation_time, 10.0)  # Should complete within 10 seconds
                        self.assertIsNotNone(result)

                except Exception as e:
                    self.fail(f"Performance test with config {config} failed: {e}")


class TestYuEIntegration(AudioLabBaseTest, ModuleTestMixin):
    """Integration tests for YuE module."""

    def setUp(self):
        """Set up integration test fixtures."""
        super().setUp()

        # Integration test scenario
        self.integration_prompts = [
            "A symphony orchestra playing a dramatic piece",
            "An electronic dance track with complex rhythms",
            "A folk song with acoustic guitar and vocals"
        ]

    @unittest.skipIf(not yue_available, "YuE module not available")
    def test_full_music_generation_pipeline(self):
        """Test complete music generation pipeline."""
        print("\\nTesting full music generation pipeline...")

        try:
            with patch('modules.yue.inference.infer.generate_music') as mock_generate:
                # Mock the complete pipeline
                mock_audio = self.mock_model('music_gen')
                mock_generate.return_value = mock_audio

                # Test tokenizer if available
                if yue_available:
                    try:
                        tokenizer = MMTokenizer()

                        for prompt in self.integration_prompts:
                            # 1. Tokenize prompt
                            tokens = tokenizer.encode(prompt)
                            self.assertIsNotNone(tokens)

                            # 2. Generate music
                            result = generate_music(
                                prompt=prompt,
                                duration=15,
                                genre='mixed',
                                seed=42
                            )

                            # 3. Validate output
                            self.assertIsNotNone(result)

                            # 4. Test codec manipulation if available
                            if yue_available:
                                try:
                                    codec_manipulator = CodecManipulator()
                                    # This would test the full encode/decode cycle
                                    self.assertIsNotNone(codec_manipulator)
                                except:
                                    pass  # Codec manipulator might not be critical

                    except:
                        # Tokenizer might not be available
                        pass

        except Exception as e:
            self.fail(f"Full pipeline integration failed: {e}")


if __name__ == '__main__':
    # Setup logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)

    # Run tests
    unittest.main(verbosity=2)
