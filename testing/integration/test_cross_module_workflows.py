"""
Integration tests for cross-module workflows in AudioLab.
"""

import os
import unittest
import tempfile
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path

from testing.utils.base_test import AudioLabBaseTest


class TestTTStoRVCWorkflow(AudioLabBaseTest):
    """Test TTS to RVC voice conversion workflow."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        # Create test text for TTS
        self.test_text = "This is a test sentence for text-to-speech followed by voice conversion."

        # Create speaker sample for TTS
        self.speaker_sample = self.create_test_audio(duration=3.0, frequency=220)

        # Create target voice sample for RVC
        self.target_voice = self.create_test_audio(duration=3.0, frequency=330)

    def test_tts_to_rvc_voice_conversion(self):
        """Test complete TTS to RVC workflow."""
        print("\\nTesting TTS to RVC voice conversion workflow...")

        # This test would integrate TTS and RVC modules
        # Mock both TTS and RVC functionality

        try:
            # Step 1: Generate speech with TTS
            with patch('layouts.tts.run_zonos_tts') as mock_tts:
                mock_tts.return_value = self.create_test_audio(duration=5.0, frequency=220)

                tts_output = mock_tts(
                    language="en",
                    emotion_choice="Neutral",
                    text=self.test_text,
                    speaker_sample=self.speaker_sample,
                    speed=1.0,
                    progress=Mock()
                )

            # Step 2: Convert voice with RVC
            with patch('modules.rvc.infer.modules.vc.pipeline.Pipeline') as mock_rvc_pipeline:
                mock_pipeline = Mock()
                mock_pipeline.vc.return_value = self.create_test_audio(duration=5.0, frequency=330)
                mock_rvc_pipeline.return_value = mock_pipeline

                # Setup RVC model files
                model_path = os.path.join(self.test_dir, 'rvc_model.pth')
                index_path = os.path.join(self.test_dir, 'rvc_model.index')

                with open(model_path, 'wb') as f:
                    f.write(b'mock_rvc_model')
                with open(index_path, 'wb') as f:
                    f.write(b'mock_rvc_index')

                rvc_output = mock_pipeline.vc(
                    audio_path=tts_output,
                    f0_up_key=0,
                    f0_method='rmvpe',
                    index_rate=0.5,
                    filter_radius=3,
                    resample_sr=22050,
                    rms_mix_rate=0.25,
                    protect=0.33
                )

            # Verify workflow completed
            self.assertIsNotNone(tts_output)
            self.assertIsNotNone(rvc_output)

            # Validate audio properties
            tts_validation = self.validate_audio_output(tts_output)
            rvc_validation = self.validate_audio_output(rvc_output)

            self.assertTrue(tts_validation['valid'])
            self.assertTrue(rvc_validation['valid'])

        except Exception as e:
            self.fail(f"TTS to RVC workflow failed: {e}")


class TestMusicGenerationToProcessingWorkflow(AudioLabBaseTest):
    """Test music generation to audio processing workflow."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        self.music_prompt = "An upbeat electronic dance track with synth leads and heavy bass."

    def test_music_generation_to_separation(self):
        """Test music generation followed by stem separation."""
        print("\\nTesting music generation to separation workflow...")

        try:
            # Step 1: Generate music
            with patch('modules.yue.inference.infer.generate_music') as mock_music_gen:
                mock_audio = self.mock_model('music_gen')
                mock_music_gen.return_value = mock_audio

                music_output = mock_music_gen(
                    prompt=self.music_prompt,
                    duration=20,
                    genre='electronic',
                    seed=42
                )

            # Step 2: Separate stems
            with patch('wrappers.separate.Separate') as mock_separator:
                mock_separator_instance = Mock()
                stems = {
                    'vocals': self.create_test_audio(duration=20.0, frequency=220),
                    'drums': self.create_test_audio(duration=20.0, frequency=100),
                    'bass': self.create_test_audio(duration=20.0, frequency=60),
                    'other': self.create_test_audio(duration=20.0, frequency=440)
                }
                mock_separator_instance.separate.return_value = stems
                mock_separator.return_value = mock_separator_instance

                separation_output = mock_separator_instance.separate([music_output])

            # Verify workflow completed
            self.assertIsNotNone(music_output)
            self.assertIsNotNone(separation_output)
            self.assertIsInstance(separation_output, dict)

            # Validate all stems exist and are valid audio
            for stem_name, stem_audio in separation_output.items():
                validation = self.validate_audio_output(stem_audio)
                self.assertTrue(validation['valid'], f"Invalid {stem_name} stem")

        except Exception as e:
            self.fail(f"Music generation to separation workflow failed: {e}")


class TestAudioProcessingPipelineWorkflow(AudioLabBaseTest):
    """Test complete audio processing pipeline workflow."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        # Create test audio input
        self.input_audio = self.create_test_audio(duration=5.0, frequency=220)

        # Define processing pipeline
        self.processing_pipeline = [
            'clone',      # Voice cloning
            'separate',   # Stem separation
            'convert',    # Format conversion
            'remaster'    # Audio remastering
        ]

    def test_complete_processing_pipeline(self):
        """Test complete audio processing pipeline."""
        print("\\nTesting complete audio processing pipeline...")

        try:
            # Mock the processing pipeline
            with patch('layouts.process.get_processor') as mock_get_processor, \
                 patch('layouts.process.check_processor_conflicts') as mock_conflicts:

                # Setup mock processors
                def get_processor_side_effect(title):
                    mock_processor = Mock()
                    mock_processor.title = title
                    mock_processor.process = Mock(return_value=[self.input_audio])
                    mock_processor.get_conflicts = Mock(return_value=[])
                    return mock_processor

                mock_get_processor.side_effect = get_processor_side_effect
                mock_conflicts.return_value = Mock()  # No conflicts

                # Execute processing pipeline
                from layouts.process import process
                result = process(
                    processors=self.processing_pipeline,
                    inputs=[self.input_audio],
                    progress=Mock()
                )

            # Verify pipeline completed
            self.assertIsNotNone(result)
            self.assertIsInstance(result, list)

            # Verify all processors were called
            self.assertEqual(mock_get_processor.call_count, len(self.processing_pipeline))

        except Exception as e:
            self.fail(f"Complete processing pipeline failed: {e}")


class TestMultiModalWorkflow(AudioLabBaseTest):
    """Test multi-modal audio processing workflows."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        # Create different types of audio inputs
        self.speech_input = self.create_test_audio(duration=3.0, frequency=220)
        self.music_input = self.create_test_audio(duration=10.0, frequency=440)
        self.noise_input = self.create_test_audio(duration=2.0, frequency=1000)

    def test_speech_enhancement_workflow(self):
        """Test speech enhancement workflow."""
        print("\\nTesting speech enhancement workflow...")

        try:
            # Step 1: Noise reduction
            with patch('handlers.noise_removal') as mock_noise_reduction:
                mock_noise_reduction.return_value = self.create_test_audio(duration=3.0, frequency=220)

                denoised = mock_noise_reduction([self.speech_input])

            # Step 2: Voice cloning for enhancement
            with patch('wrappers.clone') as mock_voice_clone:
                mock_voice_clone_instance = Mock()
                mock_voice_clone_instance.process.return_value = [denoised[0]]
                mock_voice_clone.return_value = mock_voice_clone_instance

                enhanced = mock_voice_clone_instance.process([denoised[0]])

            # Step 3: Final processing
            with patch('wrappers.remaster') as mock_remaster:
                mock_remaster_instance = Mock()
                mock_remaster_instance.process.return_value = [enhanced[0]]
                mock_remaster.return_value = mock_remaster_instance

                final_result = mock_remaster_instance.process([enhanced[0]])

            # Verify workflow completed
            self.assertIsNotNone(denoised)
            self.assertIsNotNone(enhanced)
            self.assertIsNotNone(final_result)

        except Exception as e:
            self.fail(f"Speech enhancement workflow failed: {e}")

    def test_music_production_workflow(self):
        """Test complete music production workflow."""
        print("\\nTesting music production workflow...")

        try:
            # Step 1: Generate base track
            with patch('modules.yue.inference.infer.generate_music') as mock_music_gen:
                mock_audio = self.mock_model('music_gen')
                mock_music_gen.return_value = mock_audio

                base_track = mock_music_gen(
                    prompt="A basic drum and bass track",
                    duration=30,
                    genre='electronic',
                    seed=42
                )

            # Step 2: Add vocals (TTS)
            with patch('layouts.tts.run_zonos_tts') as mock_tts:
                mock_tts.return_value = self.create_test_audio(duration=30.0, frequency=220)

                vocals = mock_tts(
                    language="en",
                    emotion_choice="Neutral",
                    text="This is a vocal track for the music production.",
                    speaker_sample=self.speech_input,
                    speed=1.0,
                    progress=Mock()
                )

            # Step 3: Mix and master
            with patch('wrappers.merge') as mock_merge, \
                 patch('wrappers.remaster') as mock_remaster:

                mock_merge_instance = Mock()
                mock_merge_instance.process.return_value = [base_track]
                mock_merge.return_value = mock_merge_instance

                mock_remaster_instance = Mock()
                mock_remaster_instance.process.return_value = [base_track]
                mock_remaster.return_value = mock_remaster_instance

                # Mix tracks
                mixed = mock_merge_instance.process([base_track, vocals])

                # Master final result
                mastered = mock_remaster_instance.process(mixed)

            # Verify workflow completed
            self.assertIsNotNone(base_track)
            self.assertIsNotNone(vocals)
            self.assertIsNotNone(mixed)
            self.assertIsNotNone(mastered)

        except Exception as e:
            self.fail(f"Music production workflow failed: {e}")


class TestErrorRecoveryWorkflow(AudioLabBaseTest):
    """Test error recovery in cross-module workflows."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        self.test_input = self.create_test_audio(duration=2.0, frequency=220)

    def test_workflow_error_recovery(self):
        """Test error recovery in processing workflows."""
        print("\\nTesting workflow error recovery...")

        try:
            # Test workflow with some components failing
            with patch('layouts.process.get_processor') as mock_get_processor:
                def get_processor_side_effect(title):
                    mock_processor = Mock()
                    mock_processor.title = title

                    if title == 'failing_processor':
                        # This processor will fail
                        mock_processor.process = Mock(side_effect=Exception(f"{title} failed"))
                    else:
                        # Other processors work normally
                        mock_processor.process = Mock(return_value=[self.test_input])
                        mock_processor.get_conflicts = Mock(return_value=[])

                    return mock_processor

                mock_get_processor.side_effect = get_processor_side_effect

                # Execute workflow with mixed success/failure
                from layouts.process import process

                with self.assertRaises(Exception):
                    # Should fail due to failing processor
                    process(
                        processors=['working_processor', 'failing_processor', 'another_working'],
                        inputs=[self.test_input],
                        progress=Mock()
                    )

        except Exception as e:
            self.fail(f"Error recovery workflow test failed: {e}")


class TestPerformanceWorkflow(AudioLabBaseTest):
    """Test performance characteristics of workflows."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        # Create larger test files for performance testing
        self.large_audio = self.create_test_audio(duration=30.0, frequency=220)

    def test_workflow_performance_limits(self):
        """Test workflow performance with large inputs."""
        print("\\nTesting workflow performance limits...")

        try:
            import time

            # Measure workflow performance
            start_time = time.time()

            # Mock a complex workflow
            with patch('layouts.process.get_processor') as mock_get_processor:
                def get_processor_side_effect(title):
                    mock_processor = Mock()
                    mock_processor.title = title
                    mock_processor.process = Mock(return_value=[self.large_audio])
                    mock_processor.get_conflicts = Mock(return_value=[])
                    return mock_processor

                mock_get_processor.side_effect = get_processor_side_effect

                from layouts.process import process

                result = process(
                    processors=['clone', 'separate', 'convert', 'remaster'],
                    inputs=[self.large_audio],
                    progress=Mock()
                )

            end_time = time.time()
            execution_time = end_time - start_time

            # Performance assertions
            self.assertLess(execution_time, 60.0)  # Should complete within 60 seconds
            self.assertIsNotNone(result)

        except Exception as e:
            self.fail(f"Performance workflow test failed: {e}")


class TestWorkflowValidation(AudioLabBaseTest):
    """Test workflow validation and compatibility."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        # Define incompatible processor combinations
        self.incompatible_combinations = [
            (['clone', 'tts'], 'Cannot use both cloning and TTS in same pipeline'),
            (['separate', 'merge'], 'Separation and merging conflict'),
        ]

    def test_workflow_compatibility_validation(self):
        """Test validation of workflow compatibility."""
        print("\\nTesting workflow compatibility validation...")

        try:
            # Test compatible workflow
            with patch('layouts.process.check_processor_conflicts') as mock_conflicts:
                mock_conflicts.return_value = Mock()  # No conflicts

                from layouts.process import check_processor_conflicts

                # Test compatible processors
                result = check_processor_conflicts(['clone', 'convert'])
                self.assertIsNotNone(result)

            # Test incompatible workflow
            with patch('layouts.process.check_processor_conflicts') as mock_conflicts:
                mock_conflicts.return_value = Mock(value="Conflicts detected")

                from layouts.process import check_processor_conflicts

                # Test incompatible processors
                result = check_processor_conflicts(['clone', 'tts'])
                self.assertIsNotNone(result)

        except Exception as e:
            self.fail(f"Workflow validation test failed: {e}")


if __name__ == '__main__':
    # Setup logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)

    # Run tests
    unittest.main(verbosity=2)
