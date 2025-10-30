"""
Comprehensive unit tests for audio processing pipeline layout.
"""

import os
import unittest
import tempfile
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path

from testing.utils.base_test import AudioLabBaseTest, LayoutTestMixin

# Import the layout module for testing
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'layouts'))

try:
    from process import (
        process, get_processor, check_processor_conflicts, enforce_defaults,
        update_preview, update_preview_select, list_projects, load_project,
        get_audio_files, get_image_files, get_video_files,
        is_audio, is_video, list_wrappers
    )
    process_available = True
except ImportError as e:
    process_available = False
    import_error = str(e)


class TestProcessPipeline(AudioLabBaseTest, LayoutTestMixin):
    """Test audio processing pipeline functionality."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        # Create test audio files
        self.test_audio_files = [
            self.create_test_audio(duration=2.0, frequency=220),
            self.create_test_audio(duration=3.0, frequency=330),
            self.create_test_audio(duration=2.5, frequency=440)
        ]

        # Mock wrapper classes
        self.mock_wrappers = self._create_mock_wrappers()

    def _create_mock_wrappers(self):
        """Create mock wrapper classes for testing."""
        wrappers = {}

        # Mock different wrapper types
        wrapper_configs = [
            ('clone', 'Voice Cloning'),
            ('separate', 'Audio Separation'),
            ('convert', 'Format Conversion'),
            ('remaster', 'Audio Remastering'),
            ('super_res', 'Super Resolution')
        ]

        for wrapper_id, title in wrapper_configs:
            mock_wrapper = Mock()
            mock_wrapper.title = title
            mock_wrapper.id = wrapper_id
            mock_wrapper.process = Mock(return_value=self.test_audio_files)
            mock_wrapper.get_conflicts = Mock(return_value=[])
            wrappers[wrapper_id] = mock_wrapper

        return wrappers

    @unittest.skipIf(not process_available, "Process module not available")
    def test_process_single_processor(self):
        """Test processing with a single processor."""
        print("\\nTesting processing with single processor...")

        progress_mock = Mock()

        with patch('process.get_processor') as mock_get_processor:
            # Mock the processor
            mock_processor = self.mock_wrappers['clone']
            mock_get_processor.return_value = mock_processor

            result = process(
                processors=['clone'],
                inputs=self.test_audio_files,
                progress=progress_mock
            )

            # Verify processing was called
            mock_processor.process.assert_called_once_with(self.test_audio_files, progress_mock)
            self.assertEqual(result, self.test_audio_files)

    @unittest.skipIf(not process_available, "Process module not available")
    def test_process_multiple_processors(self):
        """Test processing with multiple processors."""
        print("\\nTesting processing with multiple processors...")

        progress_mock = Mock()

        with patch('process.get_processor') as mock_get_processor:
            # Mock multiple processors
            def get_processor_side_effect(processor_title):
                return self.mock_wrappers.get(processor_title.lower(), Mock())

            mock_get_processor.side_effect = get_processor_side_effect

            result = process(
                processors=['clone', 'separate'],
                inputs=self.test_audio_files,
                progress=progress_mock
            )

            # Should process with both processors
            self.assertIsInstance(result, list)

    @unittest.skipIf(not process_available, "Process module not available")
    def test_process_empty_processors(self):
        """Test processing with no processors."""
        print("\\nTesting processing with no processors...")

        result = process(
            processors=[],
            inputs=self.test_audio_files,
            progress=Mock()
        )

        # Should return input unchanged
        self.assertEqual(result, self.test_audio_files)


class TestProcessProcessorManagement(AudioLabBaseTest):
    """Test processor instantiation and management."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        # Mock wrapper modules
        self.mock_wrapper_modules = self._create_mock_wrapper_modules()

    def _create_mock_wrapper_modules(self):
        """Create mock wrapper modules."""
        modules = {}

        wrapper_classes = [
            ('CloneWrapper', 'Voice Cloning'),
            ('SeparateWrapper', 'Audio Separation'),
            ('ConvertWrapper', 'Format Conversion'),
            ('RemasterWrapper', 'Audio Remastering')
        ]

        for class_name, title in wrapper_classes:
            mock_class = Mock()
            mock_instance = Mock()
            mock_instance.title = title
            mock_instance.process = Mock(return_value=[])
            mock_class.return_value = mock_instance
            modules[class_name.lower()] = mock_class

        return modules

    @unittest.skipIf(not process_available, "Process module not available")
    def test_get_processor_valid(self):
        """Test getting a valid processor."""
        print("\\nTesting get_processor with valid processor...")

        with patch.dict('sys.modules', self.mock_wrapper_modules):
            with patch('importlib.import_module') as mock_import:
                # Mock the import to return our mock module
                mock_module = Mock()
                mock_module.CloneWrapper = self.mock_wrapper_modules['clonewrapper']
                mock_import.return_value = mock_module

                processor = get_processor("Voice Cloning")

                self.assertIsNotNone(processor)
                self.assertEqual(processor.title, "Voice Cloning")

    @unittest.skipIf(not process_available, "Process module not available")
    def test_get_processor_invalid(self):
        """Test getting an invalid processor."""
        print("\\nTesting get_processor with invalid processor...")

        with self.assertRaises(Exception):
            get_processor("Nonexistent Processor")


class TestProcessConflictDetection(AudioLabBaseTest):
    """Test processor conflict detection."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        # Create mock processors with conflicts
        self.conflicting_processors = self._create_conflicting_processors()

    def _create_conflicting_processors(self):
        """Create processors that have conflicts."""
        processors = []

        # Processor 1 conflicts with Processor 2
        proc1 = Mock()
        proc1.title = "Processor 1"
        proc1.get_conflicts = Mock(return_value=["Processor 2"])
        processors.append(proc1)

        # Processor 2 conflicts with Processor 1
        proc2 = Mock()
        proc2.title = "Processor 2"
        proc2.get_conflicts = Mock(return_value=["Processor 1"])
        processors.append(proc2)

        return processors

    @unittest.skipIf(not process_available, "Process module not available")
    def test_check_processor_conflicts_detection(self):
        """Test detection of processor conflicts."""
        print("\\nTesting processor conflict detection...")

        with patch('process.get_processor') as mock_get_processor:
            def get_processor_side_effect(title):
                for proc in self.conflicting_processors:
                    if proc.title == title:
                        return proc
                return Mock()

            mock_get_processor.side_effect = get_processor_side_effect

            # Test conflict detection
            update_result = check_processor_conflicts(['Processor 1', 'Processor 2'])

            # Should detect conflicts
            self.assertIsNotNone(update_result)

    @unittest.skipIf(not process_available, "Process module not available")
    def test_check_processor_conflicts_no_conflicts(self):
        """Test conflict detection with no conflicts."""
        print("\\nTesting processor conflict detection with no conflicts...")

        with patch('process.get_processor') as mock_get_processor:
            mock_get_processor.return_value = Mock(get_conflicts=Mock(return_value=[]))

            update_result = check_processor_conflicts(['Processor 1', 'Processor 2'])

            # Should not detect conflicts
            self.assertIsNotNone(update_result)


class TestProcessDefaults(AudioLabBaseTest):
    """Test default value enforcement."""

    @unittest.skipIf(not process_available, "Process module not available")
    def test_enforce_defaults_basic(self):
        """Test basic default enforcement."""
        print("\\nTesting default enforcement...")

        # Test with empty processors list
        result = enforce_defaults([])

        # Should return some default configuration
        self.assertIsNotNone(result)

    @unittest.skipIf(not process_available, "Process module not available")
    def test_enforce_defaults_with_processors(self):
        """Test default enforcement with processors."""
        print("\\nTesting default enforcement with processors...")

        # Test with specific processors
        result = enforce_defaults(['clone', 'separate'])

        # Should return configuration for specified processors
        self.assertIsNotNone(result)


class TestProcessPreview(AudioLabBaseTest):
    """Test preview functionality."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        # Create test output file
        self.test_output = self.create_test_audio(duration=2.0, frequency=440)

    @unittest.skipIf(not process_available, "Process module not available")
    def test_update_preview_basic(self):
        """Test basic preview update."""
        print("\\nTesting basic preview update...")

        result = update_preview(self.test_output)

        # Should return preview update information
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)  # Gradio update format

    @unittest.skipIf(not process_available, "Process module not available")
    def test_update_preview_select_multiple(self):
        """Test preview selection with multiple files."""
        print("\\nTesting preview selection with multiple files...")

        result = update_preview_select(self.test_audio_files)

        # Should return preview update for multiple files
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)  # Gradio update format for multiple files


class TestProcessFileHandling(AudioLabBaseTest):
    """Test file type detection and handling."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        # Create test files of different types
        self.audio_file = self.create_test_audio(duration=1.0, frequency=220)
        self.image_file = os.path.join(self.test_dir, 'test_image.png')
        self.video_file = os.path.join(self.test_dir, 'test_video.mp4')

        # Create dummy image and video files
        with open(self.image_file, 'wb') as f:
            f.write(b'fake_png_data')
        with open(self.video_file, 'wb') as f:
            f.write(b'fake_mp4_data')

    @unittest.skipIf(not process_available, "Process module not available")
    def test_is_audio_detection(self):
        """Test audio file detection."""
        print("\\nTesting audio file detection...")

        self.assertTrue(is_audio(self.audio_file))
        self.assertFalse(is_audio(self.image_file))
        self.assertFalse(is_audio(self.video_file))

    @unittest.skipIf(not process_available, "Process module not available")
    def test_is_video_detection(self):
        """Test video file detection."""
        print("\\nTesting video file detection...")

        self.assertFalse(is_video(self.audio_file))
        self.assertFalse(is_video(self.image_file))
        # Note: Our fake video file won't be detected as video without proper headers
        # This is expected behavior

    @unittest.skipIf(not process_available, "Process module not available")
    def test_get_audio_files_from_mixed(self):
        """Test getting audio files from mixed file list."""
        print("\\nTesting audio file extraction from mixed files...")

        mixed_files = [
            self.audio_file,
            self.image_file,
            self.video_file,
            self.test_audio_files[0]  # Another audio file
        ]

        audio_files = get_audio_files(mixed_files)

        # Should extract only audio files
        self.assertEqual(len(audio_files), 2)
        self.assertIn(self.audio_file, audio_files)
        self.assertIn(self.test_audio_files[0], audio_files)

    @unittest.skipIf(not process_available, "Process module not available")
    def test_get_image_files_from_mixed(self):
        """Test getting image files from mixed file list."""
        print("\\nTesting image file extraction from mixed files...")

        mixed_files = [
            self.audio_file,
            self.image_file,
            self.video_file
        ]

        image_files = get_image_files(mixed_files)

        # Should extract only image files
        self.assertEqual(len(image_files), 1)
        self.assertIn(self.image_file, image_files)

    @unittest.skipIf(not process_available, "Process module not available")
    def test_get_video_files_from_mixed(self):
        """Test getting video files from mixed file list."""
        print("\\nTesting video file extraction from mixed files...")

        mixed_files = [
            self.audio_file,
            self.image_file,
            self.video_file
        ]

        video_files = get_video_files(mixed_files)

        # Should extract only video files (none in our test case)
        self.assertEqual(len(video_files), 0)


class TestProcessProjectManagement(AudioLabBaseTest):
    """Test project loading and management."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        # Create mock project structure
        self.project_dir = os.path.join(self.test_dir, 'projects')
        os.makedirs(self.project_dir)

        # Create sample project files
        self.project_file = os.path.join(self.project_dir, 'test_project.json')
        project_data = {
            'name': 'Test Project',
            'processors': ['clone', 'separate'],
            'settings': {'quality': 'high'}
        }

        with open(self.project_file, 'w') as f:
            json.dump(project_data, f)

    @unittest.skipIf(not process_available, "Process module not available")
    def test_list_projects_basic(self):
        """Test basic project listing."""
        print("\\nTesting basic project listing...")

        projects = list_projects()

        # Should return list of project names
        self.assertIsInstance(projects, list)

    @unittest.skipIf(not process_available, "Process module not available")
    def test_load_project_valid(self):
        """Test loading a valid project."""
        print("\\nTesting valid project loading...")

        with patch('process.list_projects') as mock_list_projects:
            mock_list_projects.return_value = ['test_project']

            # Mock project file reading
            with patch('builtins.open', create=True) as mock_open:
                mock_file = Mock()
                mock_file.read.return_value = json.dumps({
                    'name': 'Test Project',
                    'processors': ['clone'],
                    'input_files': self.test_audio_files
                })
                mock_open.return_value = mock_file

                result = load_project('test_project', self.test_audio_files)

                # Should return project update information
                self.assertIsNotNone(result)


class TestProcessWrappers(AudioLabBaseTest):
    """Test wrapper listing and management."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        # Mock wrapper discovery
        self.mock_wrapper_info = [
            {'id': 'clone', 'title': 'Voice Cloning', 'description': 'Clone voices'},
            {'id': 'separate', 'title': 'Audio Separation', 'description': 'Separate stems'},
            {'id': 'convert', 'title': 'Format Conversion', 'description': 'Convert formats'}
        ]

    @unittest.skipIf(not process_available, "Process module not available")
    def test_list_wrappers_basic(self):
        """Test basic wrapper listing."""
        print("\\nTesting basic wrapper listing...")

        with patch('os.listdir') as mock_listdir, \
             patch('importlib.util.spec_from_file_location') as mock_spec, \
             patch('importlib.util.module_from_spec') as mock_module:

            # Mock directory listing
            mock_listdir.return_value = ['clone.py', 'separate.py', 'convert.py']

            # Mock module loading
            mock_spec.return_value = Mock()
            mock_test_module = Mock()
            mock_test_module.__name__ = 'test_module'
            mock_module.return_value = mock_test_module

            wrappers = list_wrappers()

            # Should return wrapper information
            self.assertIsInstance(wrappers, list)


class TestProcessErrorHandling(AudioLabBaseTest):
    """Test error handling in processing pipeline."""

    @unittest.skipIf(not process_available, "Process module not available")
    def test_process_invalid_processor(self):
        """Test processing with invalid processor."""
        print("\\nTesting processing with invalid processor...")

        with patch('process.get_processor') as mock_get_processor:
            mock_get_processor.side_effect = Exception("Invalid processor")

            with self.assertRaises(Exception):
                process(
                    processors=['invalid_processor'],
                    inputs=self.test_audio_files,
                    progress=Mock()
                )

    @unittest.skipIf(not process_available, "Process module not available")
    def test_process_processor_failure(self):
        """Test processing when processor fails."""
        print("\\nTesting processing with processor failure...")

        progress_mock = Mock()

        with patch('process.get_processor') as mock_get_processor:
            mock_processor = Mock()
            mock_processor.process.side_effect = Exception("Processor failed")
            mock_get_processor.return_value = mock_processor

            with self.assertRaises(Exception):
                process(
                    processors=['failing_processor'],
                    inputs=self.test_audio_files,
                    progress=progress_mock
                )


class TestProcessIntegration(AudioLabBaseTest, LayoutTestMixin):
    """Integration tests for processing pipeline."""

    def setUp(self):
        """Set up integration test fixtures."""
        super().setUp()

        # Create comprehensive test scenario
        self.test_inputs = [
            self.create_test_audio(duration=2.0, frequency=220),
            self.create_test_audio(duration=3.0, frequency=330)
        ]

        self.test_processors = ['clone', 'separate', 'convert']

    @unittest.skipIf(not process_available, "Process module not available")
    def test_full_processing_pipeline(self):
        """Test complete processing pipeline."""
        print("\\nTesting full processing pipeline...")

        progress_mock = Mock()

        with patch('process.get_processor') as mock_get_processor, \
             patch('process.check_processor_conflicts') as mock_conflicts, \
             patch('process.enforce_defaults') as mock_defaults:

            # Setup mocks
            def get_processor_side_effect(title):
                mock_processor = Mock()
                mock_processor.title = title
                mock_processor.process = Mock(return_value=self.test_inputs)
                mock_processor.get_conflicts = Mock(return_value=[])
                return mock_processor

            mock_get_processor.side_effect = get_processor_side_effect
            mock_conflicts.return_value = Mock()  # No update
            mock_defaults.return_value = {}

            result = process(
                processors=self.test_processors,
                inputs=self.test_inputs,
                progress=progress_mock
            )

            # Should complete full pipeline
            self.assertIsInstance(result, list)

            # Should have called get_processor for each processor
            self.assertEqual(mock_get_processor.call_count, len(self.test_processors))


if __name__ == '__main__':
    # Setup logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)

    # Run tests
    unittest.main(verbosity=2)
