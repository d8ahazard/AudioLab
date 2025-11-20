"""
Unit tests for RVC V3 content encoders.

Tests HuBERTEncoder, WhisperEncoder, and DualContentEncoder.
"""

import os
import sys
import unittest
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from modules.rvc_v3.models.content_encoders import HuBERTEncoder
from handlers.config import model_path


class TestHuBERTEncoder(unittest.TestCase):
    """Test HuBERT content encoder."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.hubert_path = os.path.join(model_path, "rvc", "hubert_base.pt")
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def test_hubert_loads(self):
        """Test that HuBERT model loads successfully."""
        try:
            encoder = HuBERTEncoder(
                model_path=self.hubert_path,
                device=self.device,
                is_half=False
            )
            self.assertIsNotNone(encoder.model)
            print("✓ HuBERT model loaded successfully")
        except FileNotFoundError as e:
            # If HuBERT not found, test should trigger download
            print(f"HuBERT model not found, attempting download...")
            encoder = HuBERTEncoder(
                model_path=self.hubert_path,
                device=self.device,
                is_half=False
            )
            self.assertIsNotNone(encoder.model)
            print("✓ HuBERT model downloaded and loaded successfully")
    
    def test_feature_extraction_shape(self):
        """Test that feature extraction produces correct shape."""
        encoder = HuBERTEncoder(
            model_path=self.hubert_path,
            device=self.device,
            is_half=False
        )
        
        # Create dummy audio (1 second at 16kHz)
        audio = torch.randn(1, 16000).to(self.device)
        
        # Extract features
        features = encoder.extract_features(audio)
        
        # Check shape: (1, T', 768)
        self.assertEqual(len(features.shape), 3)
        self.assertEqual(features.shape[0], 1)  # Batch size
        self.assertEqual(features.shape[2], 768)  # Feature dimension
        
        # Check time dimension (should be ~50 frames for 1 second)
        expected_frames = 16000 // 320  # Hop length is 320
        self.assertGreater(features.shape[1], 40)
        self.assertLess(features.shape[1], 60)
        
        print(f"✓ Feature shape: {features.shape}")
    
    def test_feature_extraction_no_nan(self):
        """Test that features don't contain NaN or Inf."""
        encoder = HuBERTEncoder(
            model_path=self.hubert_path,
            device=self.device,
            is_half=False
        )
        
        # Create dummy audio
        audio = torch.randn(1, 16000).to(self.device)
        
        # Extract features
        features = encoder.extract_features(audio)
        
        # Check for NaN/Inf
        self.assertFalse(torch.isnan(features).any().item())
        self.assertFalse(torch.isinf(features).any().item())
        
        print("✓ Features contain no NaN or Inf values")
    
    def test_feature_extraction_deterministic(self):
        """Test that feature extraction is deterministic."""
        encoder = HuBERTEncoder(
            model_path=self.hubert_path,
            device=self.device,
            is_half=False
        )
        
        # Create dummy audio
        audio = torch.randn(1, 16000).to(self.device)
        
        # Extract features twice
        features1 = encoder.extract_features(audio)
        features2 = encoder.extract_features(audio)
        
        # Should be identical
        self.assertTrue(torch.allclose(features1, features2, rtol=1e-5))
        
        print("✓ Feature extraction is deterministic")
    
    def test_batch_processing(self):
        """Test batch processing of multiple audio samples."""
        encoder = HuBERTEncoder(
            model_path=self.hubert_path,
            device=self.device,
            is_half=False
        )
        
        # Create batch of dummy audio
        batch_size = 4
        audio = torch.randn(batch_size, 16000).to(self.device)
        
        # Extract features
        features = encoder.extract_features(audio)
        
        # Check batch dimension
        self.assertEqual(features.shape[0], batch_size)
        self.assertEqual(features.shape[2], 768)
        
        print(f"✓ Batch processing works: {features.shape}")
    
    def test_variable_length_audio(self):
        """Test with different audio lengths."""
        encoder = HuBERTEncoder(
            model_path=self.hubert_path,
            device=self.device,
            is_half=False
        )
        
        # Test different lengths
        lengths = [8000, 16000, 32000]  # 0.5s, 1s, 2s
        
        for length in lengths:
            audio = torch.randn(1, length).to(self.device)
            features = encoder.extract_features(audio)
            
            # Check that time dimension scales appropriately
            expected_frames = length // 320
            self.assertAlmostEqual(features.shape[1], expected_frames, delta=5)
            
            print(f"✓ Audio length {length} -> {features.shape[1]} frames")


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestHuBERTEncoder)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    success = run_tests()
    sys.exit(0 if success else 1)

