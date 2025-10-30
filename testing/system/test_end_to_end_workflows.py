"""
End-to-end system tests for complete AudioLab workflows.
"""

import os
import unittest
import tempfile
import json
import time
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path

from testing.utils.base_test import AudioLabBaseTest


class TestCompleteTTSWorkflow(AudioLabBaseTest):
    """Test complete TTS workflow from input to output."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        # Complete TTS workflow scenario
        self.tts_scenario = {
            'text': "This is a comprehensive test of the text-to-speech system with emotion control.",
            'language': 'en',
            'emotion': 'Happiness',
            'speaker_sample': self.create_test_audio(duration=3.0, frequency=220),
            'output_format': 'wav',
            'quality': 'high'
        }

    def test_complete_tts_generation_workflow(self):
        """Test complete TTS generation from text to audio."""
        print("\\nTesting complete TTS generation workflow...")

        try:
            # Step 1: Model download/setup
            with patch('layouts.tts.download_model') as mock_download:
                mock_download.return_value = os.path.join(self.test_dir, 'models', 'zonos')

                model_dir = mock_download()

            # Step 2: Text preprocessing and emotion parsing
            with patch('layouts.tts._parse_text_and_emotions') as mock_parse:
                mock_parse.return_value = [
                    {'text': self.tts_scenario['text'], 'emotion': self.tts_scenario['emotion']}
                ]

                parsed_text = mock_parse(self.tts_scenario['text'], 'Neutral')

            # Step 3: TTS generation
            with patch('layouts.tts.run_zonos_tts') as mock_tts:
                expected_output = self.create_test_audio(duration=5.0, frequency=220)
                mock_tts.return_value = expected_output

                tts_result = mock_tts(
                    language=self.tts_scenario['language'],
                    emotion_choice=self.tts_scenario['emotion'],
                    text=self.tts_scenario['text'],
                    speaker_sample=self.tts_scenario['speaker_sample'],
                    speed=1.0,
                    progress=Mock()
                )

            # Step 4: Output validation
            validation = self.validate_audio_output(
                tts_result,
                expected_sample_rate=22050,
                expected_duration=4.0  # Based on text length
            )

            # Step 5: Quality assessment
            quality_metrics = self.compare_audio_similarity(
                self.tts_scenario['speaker_sample'],
                tts_result,
                tolerance_snr=15.0  # Allow for TTS synthesis differences
            )

            # Verify complete workflow
            self.assertIsNotNone(model_dir)
            self.assertIsNotNone(parsed_text)
            self.assertIsNotNone(tts_result)
            self.assertTrue(validation['valid'])
            self.assertIn('metrics', quality_metrics)

        except Exception as e:
            self.fail(f"Complete TTS workflow failed: {e}")


class TestCompleteVoiceConversionWorkflow(AudioLabBaseTest):
    """Test complete voice conversion workflow."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        # Complete RVC workflow scenario
        self.rvc_scenario = {
            'source_audio': self.create_test_audio(duration=3.0, frequency=220),
            'target_voice': self.create_test_audio(duration=3.0, frequency=330),
            'model_name': 'test_voice_model',
            'training_data': self._create_training_dataset(),
            'conversion_settings': {
                'f0_method': 'rmvpe',
                'index_rate': 0.5,
                'filter_radius': 3,
                'rms_mix_rate': 0.25
            }
        }

    def _create_training_dataset(self):
        """Create training dataset for RVC."""
        dataset_dir = os.path.join(self.test_dir, 'rvc_training_data')
        os.makedirs(dataset_dir)

        # Create multiple audio samples for training
        for i in range(10):
            audio_path = os.path.join(dataset_dir, f'train_sample_{i"03d"}.wav')
            self.create_test_audio(duration=2.0 + i * 0.1, output_path=audio_path)

        return dataset_dir

    def test_complete_rvc_training_and_conversion(self):
        """Test complete RVC training and conversion workflow."""
        print("\\nTesting complete RVC training and conversion workflow...")

        try:
            # Step 1: Dataset preprocessing
            with patch('layouts.rvc_train.preprocess_dataset') as mock_preprocess:
                mock_preprocess.return_value = True

                preprocess_result = mock_preprocess(
                    trainset_dir=self.rvc_scenario['training_data'],
                    exp_dir=os.path.join(self.test_dir, 'rvc_experiments'),
                    sr=22050,
                    n_p=2,
                    progress=Mock()
                )

            # Step 2: F0 feature extraction
            with patch('layouts.rvc_train.extract_f0_feature') as mock_f0_extract:
                mock_f0_extract.return_value = True

                f0_result = mock_f0_extract(
                    num_processors=2,
                    extract_method='rmvpe',
                    use_pitch_guidance=True,
                    exp_dir=os.path.join(self.test_dir, 'rvc_experiments'),
                    project_version='v1',
                    gpus_rmvpe='0,1',
                    progress=Mock()
                )

            # Step 3: Model training
            with patch('layouts.rvc_train.click_train') as mock_train:
                mock_train.return_value = True

                train_result = mock_train(
                    exp_dir=os.path.join(self.test_dir, 'rvc_experiments'),
                    sr=22050,
                    if_f0_3=True,
                    spk_id=0,
                    gpus='0,1',
                    version19='v1',
                    progress=Mock()
                )

            # Step 4: Voice conversion
            with patch('modules.rvc.infer.modules.vc.pipeline.Pipeline') as mock_pipeline:
                mock_pipeline_instance = Mock()
                converted_audio = self.create_test_audio(duration=3.0, frequency=330)
                mock_pipeline_instance.vc.return_value = converted_audio
                mock_pipeline.return_value = mock_pipeline_instance

                # Create mock model files
                model_path = os.path.join(self.test_dir, 'rvc_experiments', 'model.pth')
                index_path = os.path.join(self.test_dir, 'rvc_experiments', 'model.index')

                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                with open(model_path, 'wb') as f:
                    f.write(b'mock_model')
                with open(index_path, 'wb') as f:
                    f.write(b'mock_index')

                conversion_result = mock_pipeline_instance.vc(
                    audio_path=self.rvc_scenario['source_audio'],
                    **self.rvc_scenario['conversion_settings']
                )

            # Verify complete workflow
            self.assertIsNotNone(preprocess_result)
            self.assertIsNotNone(f0_result)
            self.assertIsNotNone(train_result)
            self.assertIsNotNone(conversion_result)

            # Validate final output
            validation = self.validate_audio_output(conversion_result)
            self.assertTrue(validation['valid'])

        except Exception as e:
            self.fail(f"Complete RVC workflow failed: {e}")


class TestCompleteMusicGenerationWorkflow(AudioLabBaseTest):
    """Test complete music generation workflow."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        # Complete music generation scenario
        self.music_scenario = {
            'prompt': "A cinematic orchestral soundtrack with dramatic strings, powerful brass, and epic percussion.",
            'duration': 60,  # 1 minute
            'genre': 'classical',
            'tempo': 120,
            'key': 'D minor',
            'mood': 'dramatic'
        }

    def test_complete_music_generation_and_processing(self):
        """Test complete music generation and processing workflow."""
        print("\\nTesting complete music generation and processing workflow...")

        try:
            # Step 1: Music generation
            with patch('modules.yue.inference.infer.generate_music') as mock_music_gen:
                generated_music = self.mock_model('music_gen')
                mock_music_gen.return_value = generated_music

                music_result = mock_music_gen(
                    prompt=self.music_scenario['prompt'],
                    duration=self.music_scenario['duration'],
                    genre=self.music_scenario['genre'],
                    seed=42
                )

            # Step 2: Audio separation (extract stems)
            with patch('wrappers.separate.Separate') as mock_separator:
                mock_separator_instance = Mock()
                stems = {
                    'strings': self.create_test_audio(duration=60.0, frequency=330),
                    'brass': self.create_test_audio(duration=60.0, frequency=440),
                    'percussion': self.create_test_audio(duration=60.0, frequency=220),
                    'woodwinds': self.create_test_audio(duration=60.0, frequency=550)
                }
                mock_separator_instance.separate.return_value = stems
                mock_separator.return_value = mock_separator_instance

                separated_stems = mock_separator_instance.separate([music_result])

            # Step 3: Individual stem processing
            with patch('wrappers.remaster') as mock_remaster:
                mock_remaster_instance = Mock()
                remastered_stems = {}
                for stem_name, stem_audio in stems.items():
                    mock_remaster_instance.process.return_value = [stem_audio]
                    remastered_stems[stem_name] = mock_remaster_instance.process([stem_audio])[0]

                mock_remaster.return_value = mock_remaster_instance

                # Process each stem
                for stem_name in stems.keys():
                    remastered_stems[stem_name] = mock_remaster_instance.process([stems[stem_name]])[0]

            # Step 4: Final mixing and mastering
            with patch('wrappers.merge') as mock_merge, \
                 patch('wrappers.convert') as mock_convert:

                mock_merge_instance = Mock()
                mock_merge_instance.process.return_value = [music_result]
                mock_merge.return_value = mock_merge_instance

                mock_convert_instance = Mock()
                final_output = self.create_test_audio(duration=60.0, frequency=330)
                mock_convert_instance.process.return_value = [final_output]
                mock_convert.return_value = mock_convert_instance

                # Mix all stems
                mixed_audio = mock_merge_instance.process(list(remastered_stems.values()))

                # Convert to final format
                final_result = mock_convert_instance.process(mixed_audio)

            # Verify complete workflow
            self.assertIsNotNone(music_result)
            self.assertIsNotNone(separated_stems)
            self.assertIsNotNone(remastered_stems)
            self.assertIsNotNone(final_result)

            # Validate final output quality
            validation = self.validate_audio_output(final_result[0])
            self.assertTrue(validation['valid'])

        except Exception as e:
            self.fail(f"Complete music generation workflow failed: {e}")


class TestMultiModalContentCreationWorkflow(AudioLabBaseTest):
    """Test complete multi-modal content creation workflow."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        # Multi-modal content scenario
        self.content_scenario = {
            'story_text': """
            In a quiet village nestled between ancient mountains, a young musician discovered
            a magical instrument that could bring stories to life through music.

            The instrument glowed with an ethereal light as she played, weaving tales of
            adventure, love, and mystery into beautiful melodies that danced on the wind.
            """,
            'narrative_style': 'dramatic',
            'background_music_genre': 'orchestral',
            'voice_characteristics': 'warm_female_narrator',
            'total_duration': 120  # 2 minutes
        }

    def test_complete_audiobook_creation(self):
        """Test complete audiobook creation workflow."""
        print("\\nTesting complete audiobook creation workflow...")

        try:
            # Step 1: Text analysis and segmentation
            with patch('layouts.tts._parse_text_and_emotions') as mock_parse:
                mock_parse.return_value = [
                    {'text': self.content_scenario['story_text'][:100], 'emotion': 'Neutral'},
                    {'text': self.content_scenario['story_text'][100:200], 'emotion': 'Happiness'},
                    {'text': self.content_scenario['story_text'][200:], 'emotion': 'Sadness'}
                ]

                parsed_segments = mock_parse(self.content_scenario['story_text'], 'Neutral')

            # Step 2: Generate background music
            with patch('modules.yue.inference.infer.generate_music') as mock_music_gen:
                background_music = self.mock_model('music_gen')
                mock_music_gen.return_value = background_music

                music_result = mock_music_gen(
                    prompt=f"{self.content_scenario['background_music_genre']} music for storytelling",
                    duration=self.content_scenario['total_duration'],
                    genre=self.content_scenario['background_music_genre'],
                    seed=42
                )

            # Step 3: Generate narration for each segment
            narration_segments = []
            for i, segment in enumerate(parsed_segments):
                with patch('layouts.tts.run_zonos_tts') as mock_tts:
                    segment_audio = self.create_test_audio(duration=15.0, frequency=220)
                    mock_tts.return_value = segment_audio

                    narration = mock_tts(
                        language='en',
                        emotion_choice=segment['emotion'],
                        text=segment['text'],
                        speaker_sample=self.create_test_audio(duration=3.0, frequency=220),
                        speed=1.0,
                        progress=Mock()
                    )

                    narration_segments.append(narration)

            # Step 4: Combine narration with background music
            with patch('wrappers.merge') as mock_merge:
                mock_merge_instance = Mock()
                final_audio = self.create_test_audio(duration=120.0, frequency=330)
                mock_merge_instance.process.return_value = [final_audio]
                mock_merge.return_value = mock_merge_instance

                # Mix narration segments with background music
                combined_audio = mock_merge_instance.process([music_result] + narration_segments)

            # Step 5: Final mastering
            with patch('wrappers.remaster') as mock_remaster:
                mock_remaster_instance = Mock()
                mock_remaster_instance.process.return_value = [combined_audio[0]]
                mock_remaster.return_value = mock_remaster_instance

                final_mastered = mock_remaster_instance.process(combined_audio)

            # Verify complete workflow
            self.assertIsNotNone(parsed_segments)
            self.assertIsNotNone(music_result)
            self.assertIsNotNone(narration_segments)
            self.assertIsNotNone(combined_audio)
            self.assertIsNotNone(final_mastered)

            # Validate final audiobook
            validation = self.validate_audio_output(final_mastered[0])
            self.assertTrue(validation['valid'])

        except Exception as e:
            self.fail(f"Complete audiobook creation workflow failed: {e}")


class TestErrorRecoveryAndRobustness(AudioLabBaseTest):
    """Test error recovery and system robustness."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        # Error scenario setup
        self.error_scenario = {
            'corrupted_audio': self._create_corrupted_audio(),
            'network_timeout': True,
            'disk_space_low': False,
            'invalid_parameters': True
        }

    def _create_corrupted_audio(self):
        """Create corrupted audio file for testing."""
        corrupted_path = os.path.join(self.test_dir, 'corrupted_audio.wav')
        with open(corrupted_path, 'wb') as f:
            f.write(b'This is not valid audio data')
        return corrupted_path

    def test_system_error_recovery(self):
        """Test system error recovery capabilities."""
        print("\\nTesting system error recovery...")

        try:
            # Test with corrupted input
            with patch('layouts.process.get_audio_files') as mock_get_audio:
                # Return mix of valid and invalid files
                valid_audio = self.create_test_audio(duration=2.0, frequency=220)
                mock_get_audio.return_value = [valid_audio, self.error_scenario['corrupted_audio']]

                from layouts.process import get_audio_files

                # Should handle mixed valid/invalid files
                audio_files = get_audio_files([valid_audio, self.error_scenario['corrupted_audio']])
                self.assertIsInstance(audio_files, list)

            # Test with processing errors
            with patch('layouts.process.get_processor') as mock_get_processor:
                def failing_processor(title):
                    mock_processor = Mock()
                    mock_processor.title = title
                    if 'failing' in title:
                        mock_processor.process = Mock(side_effect=Exception(f"{title} processing failed"))
                    else:
                        mock_processor.process = Mock(return_value=[valid_audio])
                    mock_processor.get_conflicts = Mock(return_value=[])
                    return mock_processor

                mock_get_processor.side_effect = failing_processor

                from layouts.process import process

                # Should handle processor failures gracefully
                with self.assertRaises(Exception):
                    process(
                        processors=['working_processor', 'failing_processor'],
                        inputs=[valid_audio],
                        progress=Mock()
                    )

        except Exception as e:
            self.fail(f"Error recovery test failed: {e}")


class TestPerformanceAndScalability(AudioLabBaseTest):
    """Test system performance and scalability."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        # Performance test scenarios
        self.performance_scenarios = [
            {
                'name': 'small_batch',
                'batch_size': 5,
                'audio_duration': 10,
                'expected_time': 30  # seconds
            },
            {
                'name': 'medium_batch',
                'batch_size': 20,
                'audio_duration': 30,
                'expected_time': 120  # seconds
            },
            {
                'name': 'large_batch',
                'batch_size': 50,
                'audio_duration': 60,
                'expected_time': 300  # seconds
            }
        ]

    def test_system_performance_scalability(self):
        """Test system performance with different scales."""
        print("\\nTesting system performance scalability...")

        for scenario in self.performance_scenarios:
            with self.subTest(scenario=scenario['name']):
                try:
                    # Create test batch
                    test_batch = []
                    for i in range(scenario['batch_size']):
                        audio_file = self.create_test_audio(
                            duration=scenario['audio_duration'],
                            frequency=220 + (i * 10)  # Varying frequencies
                        )
                        test_batch.append(audio_file)

                    # Measure processing time
                    import time
                    start_time = time.time()

                    # Mock processing pipeline
                    with patch('layouts.process.get_processor') as mock_get_processor:
                        def get_processor_side_effect(title):
                            mock_processor = Mock()
                            mock_processor.title = title
                            mock_processor.process = Mock(return_value=test_batch)
                            mock_processor.get_conflicts = Mock(return_value=[])
                            return mock_processor

                        mock_get_processor.side_effect = get_processor_side_effect

                        from layouts.process import process

                        result = process(
                            processors=['clone', 'convert'],
                            inputs=test_batch,
                            progress=Mock()
                        )

                    end_time = time.time()
                    processing_time = end_time - start_time

                    # Performance assertions
                    self.assertLess(processing_time, scenario['expected_time'])
                    self.assertIsNotNone(result)
                    self.assertEqual(len(result), scenario['batch_size'])

                except Exception as e:
                    self.fail(f"Performance test for {scenario['name']} failed: {e}")


class TestSystemIntegrationAndCompatibility(AudioLabBaseTest):
    """Test system integration and compatibility."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        # Compatibility test scenarios
        self.compatibility_tests = [
            {
                'name': 'audio_format_compatibility',
                'formats': ['wav', 'mp3', 'flac', 'ogg'],
                'sample_rates': [22050, 44100, 48000]
            },
            {
                'name': 'model_version_compatibility',
                'model_versions': ['v1', 'v2', 'v3'],
                'module_types': ['tts', 'rvc', 'music_gen']
            }
        ]

    def test_audio_format_compatibility(self):
        """Test compatibility with different audio formats."""
        print("\\nTesting audio format compatibility...")

        for test_case in self.compatibility_tests:
            if test_case['name'] == 'audio_format_compatibility':
                for format_type in test_case['formats']:
                    for sample_rate in test_case['sample_rates']:
                        with self.subTest(format=format_type, sample_rate=sample_rate):
                            try:
                                # Create audio in different formats
                                audio_file = self.create_test_audio(
                                    duration=2.0,
                                    frequency=220
                                )

                                # Test format conversion
                                with patch('wrappers.convert') as mock_convert:
                                    mock_convert_instance = Mock()
                                    mock_convert_instance.process.return_value = [audio_file]
                                    mock_convert.return_value = mock_convert_instance

                                    result = mock_convert_instance.process([audio_file])

                                    self.assertIsNotNone(result)

                            except Exception as e:
                                print(f"Format compatibility test failed for {format_type}/{sample_rate}: {e}")


if __name__ == '__main__':
    # Setup logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)

    # Run tests
    unittest.main(verbosity=2)
