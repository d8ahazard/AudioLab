#!/usr/bin/env python3
"""
Simple verification script for the AudioLab testing framework.
"""

import os
import sys
import tempfile
import unittest

# Add testing directory to path
sys.path.insert(0, 'testing')

def test_basic_imports():
    """Test that basic imports work."""
    print("Testing basic imports...")

    try:
        from testing.utils.test_helpers import TestConfig, TemporaryDirectoryManager
        from testing.utils.audio_validation import audio_validator
        from testing.utils.base_test import AudioLabBaseTest

        print("✓ All basic imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_base_test_class():
    """Test the base test class functionality."""
    print("\\nTesting base test class...")

    try:
        from testing.utils.base_test import AudioLabBaseTest

        class TestVerification(AudioLabBaseTest):
            def test_initialization(self):
                self.assertIsNotNone(self.config)
                self.assertIsNotNone(self.temp_manager)
                self.assertTrue(len(self.test_audio_files) > 0)

            def test_audio_creation(self):
                test_audio = self.create_test_audio(duration=1.0, frequency=440)
                self.assert_file_exists(test_audio)

                validation = self.validate_audio_output(test_audio)
                self.assertTrue(validation['valid'])

        # Run the test
        suite = unittest.TestLoader().loadTestsFromTestCase(TestVerification)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        if result.wasSuccessful():
            print("✓ Base test class working correctly")
            return True
        else:
            print(f"✗ Base test class has {len(result.failures)} failures and {len(result.errors)} errors")
            return False

    except Exception as e:
        print(f"✗ Base test class test failed: {e}")
        return False

def test_audio_validation():
    """Test audio validation utilities."""
    print("\\nTesting audio validation...")

    try:
        from testing.utils.audio_validation import audio_validator
        from testing.utils.test_helpers import create_test_audio_file

        # Create test audio
        test_audio = create_test_audio_file(duration=2.0, frequency=440)

        # Test validation
        validation = audio_validator.validate_audio_file(
            test_audio,
            expected_sample_rate=22050,
            expected_duration=2.0
        )

        if validation['valid']:
            print("✓ Audio validation working correctly")
            return True
        else:
            print(f"✗ Audio validation failed: {validation.get('error', 'Unknown error')}")
            return False

    except Exception as e:
        print(f"✗ Audio validation test failed: {e}")
        return False

def main():
    """Main verification function."""
    print("=" * 60)
    print("AudioLab Testing Framework Verification")
    print("=" * 60)

    tests = [
        test_basic_imports,
        test_base_test_class,
        test_audio_validation
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)

    print("\\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"✓ All {total} tests passed!")
        print("\\n🎉 AudioLab Testing Framework is ready for use!")
        return 0
    else:
        print(f"✗ {passed}/{total} tests passed")
        print("\\n⚠️  Some issues need to be resolved before using the framework")
        return 1

if __name__ == "__main__":
    sys.exit(main())
