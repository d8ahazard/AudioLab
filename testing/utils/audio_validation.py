"""
Audio validation utilities for testing AudioLab components.
"""

import os
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

try:
    import librosa
    import soundfile as sf
    librosa_available = True
except ImportError:
    librosa_available = False

try:
    from pesq import pesq
    pesq_available = True
except ImportError:
    pesq_available = False

try:
    from pystoi import stoi
    stoi_available = True
except ImportError:
    stoi_available = False

from .test_helpers import test_config, load_test_audio

logger = logging.getLogger(__name__)


class AudioValidator:
    """Comprehensive audio validation utilities."""

    def __init__(self):
        self.config = test_config
        if not librosa_available:
            logger.warning("Librosa not available - some audio validation features disabled")

    def validate_audio_file(
        self,
        audio_path: Union[str, Path],
        expected_sample_rate: Optional[int] = None,
        expected_duration: Optional[float] = None,
        expected_channels: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Validate an audio file against expected properties.

        Args:
            audio_path: Path to audio file
            expected_sample_rate: Expected sample rate in Hz
            expected_duration: Expected duration in seconds
            expected_channels: Expected number of channels

        Returns:
            Dictionary with validation results
        """
        if not os.path.exists(audio_path):
            return {
                'valid': False,
                'error': f'File does not exist: {audio_path}',
                'properties': {}
            }

        try:
            # Load audio properties
            if librosa_available:
                audio, sr = librosa.load(str(audio_path), sr=None)
                duration = len(audio) / sr
                channels = 1 if audio.ndim == 1 else audio.shape[0]
            else:
                # Fallback to basic file info
                import wave
                with wave.open(str(audio_path), 'rb') as wav_file:
                    sr = wav_file.getframerate()
                    duration = wav_file.getnframes() / sr
                    channels = wav_file.getnchannels()

            properties = {
                'sample_rate': sr,
                'duration': duration,
                'channels': channels,
                'file_size': os.path.getsize(audio_path)
            }

            # Validate against expectations
            validation_errors = []

            if expected_sample_rate and abs(sr - expected_sample_rate) > 100:
                validation_errors.append(
                    f'Sample rate mismatch: expected {expected_sample_rate}, got {sr}'
                )

            if expected_duration:
                duration_tolerance = 0.1  # 100ms tolerance
                if abs(duration - expected_duration) > duration_tolerance:
                    validation_errors.append(
                        f'Duration mismatch: expected {expected_duration:.2f}s, got {duration:.2f}s'
                    )

            if expected_channels and channels != expected_channels:
                validation_errors.append(
                    f'Channels mismatch: expected {expected_channels}, got {channels}'
                )

            return {
                'valid': len(validation_errors) == 0,
                'errors': validation_errors,
                'properties': properties
            }

        except Exception as e:
            return {
                'valid': False,
                'error': f'Failed to read audio file: {str(e)}',
                'properties': {}
            }

    def calculate_audio_metrics(
        self,
        original_path: Union[str, Path],
        processed_path: Union[str, Path],
        metrics: List[str] = None
    ) -> Dict[str, float]:
        """
        Calculate audio quality metrics between original and processed audio.

        Args:
            original_path: Path to original audio file
            processed_path: Path to processed audio file
            metrics: List of metrics to calculate

        Returns:
            Dictionary with calculated metrics
        """
        if metrics is None:
            metrics = ['snr', 'rmse', 'pesq', 'stoi']

        results = {}

        try:
            # Load audio files
            if librosa_available:
                orig_audio, orig_sr = librosa.load(str(original_path), sr=None)
                proc_audio, proc_sr = librosa.load(str(processed_path), sr=None)

                # Resample to same sample rate if needed
                if orig_sr != proc_sr:
                    logger.warning(f"Sample rates differ: {orig_sr} vs {proc_sr}. Resampling...")
                    if orig_sr > proc_sr:
                        orig_audio = librosa.resample(orig_audio, orig_sr, proc_sr)
                    else:
                        proc_audio = librosa.resample(proc_audio, proc_sr, orig_sr)
                    target_sr = min(orig_sr, proc_sr)
                else:
                    target_sr = orig_sr

                # Ensure same length
                min_length = min(len(orig_audio), len(proc_audio))
                orig_audio = orig_audio[:min_length]
                proc_audio = proc_audio[:min_length]

                # Calculate metrics
                for metric in metrics:
                    if metric.lower() == 'snr':
                        results['snr_db'] = self._calculate_snr(orig_audio, proc_audio)
                    elif metric.lower() == 'rmse':
                        results['rmse'] = self._calculate_rmse(orig_audio, proc_audio)
                    elif metric.lower() == 'pesq' and pesq_available:
                        results['pesq'] = self._calculate_pesq(orig_audio, proc_audio, target_sr)
                    elif metric.lower() == 'stoi' and stoi_available:
                        results['stoi'] = self._calculate_stoi(orig_audio, proc_audio, target_sr)

            else:
                logger.warning("Librosa not available - using basic metrics only")
                results['file_size_ratio'] = self._calculate_file_size_ratio(
                    str(original_path), str(processed_path)
                )

        except Exception as e:
            logger.error(f"Failed to calculate audio metrics: {e}")
            results['error'] = str(e)

        return results

    def _calculate_snr(self, original: np.ndarray, processed: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio in dB."""
        noise = original - processed
        signal_power = np.mean(original ** 2)
        noise_power = np.mean(noise ** 2)

        if noise_power == 0:
            return float('inf')

        snr = 10 * np.log10(signal_power / noise_power)
        return float(snr)

    def _calculate_rmse(self, original: np.ndarray, processed: np.ndarray) -> float:
        """Calculate Root Mean Square Error."""
        return float(np.sqrt(np.mean((original - processed) ** 2)))

    def _calculate_pesq(self, original: np.ndarray, processed: np.ndarray, sample_rate: int) -> float:
        """Calculate PESQ (Perceptual Evaluation of Speech Quality)."""
        try:
            # Normalize audio to [-1, 1] range for PESQ
            orig_norm = original / (np.max(np.abs(original)) + 1e-10)
            proc_norm = processed / (np.max(np.abs(processed)) + 1e-10)

            # PESQ expects 16-bit integer range for best results
            pesq_score = pesq(sample_rate, orig_norm, proc_norm, 'wb')
            return float(pesq_score)
        except Exception as e:
            logger.warning(f"PESQ calculation failed: {e}")
            return 0.0

    def _calculate_stoi(self, original: np.ndarray, processed: np.ndarray, sample_rate: int) -> float:
        """Calculate STOI (Short-Time Objective Intelligibility)."""
        try:
            # Normalize audio
            orig_norm = original / (np.max(np.abs(original)) + 1e-10)
            proc_norm = processed / (np.max(np.abs(processed)) + 1e-10)

            stoi_score = stoi(orig_norm, proc_norm, sample_rate, extended=False)
            return float(stoi_score)
        except Exception as e:
            logger.warning(f"STOI calculation failed: {e}")
            return 0.0

    def _calculate_file_size_ratio(self, original_path: str, processed_path: str) -> float:
        """Calculate ratio of processed file size to original file size."""
        try:
            orig_size = os.path.getsize(original_path)
            proc_size = os.path.getsize(processed_path)
            return proc_size / orig_size if orig_size > 0 else 0.0
        except Exception:
            return 0.0

    def validate_audio_similarity(
        self,
        original_path: Union[str, Path],
        processed_path: Union[str, Path],
        tolerance_snr: float = None,
        tolerance_rmse: float = None,
        tolerance_pesq: float = None,
        tolerance_stoi: float = None
    ) -> Dict[str, Any]:
        """
        Validate that processed audio is similar to original within tolerances.

        Returns:
            Dictionary with validation results and metrics
        """
        # Get default tolerances from config
        if tolerance_snr is None:
            tolerance_snr = self.config.get("audio_validation.tolerance_snr_db", 20.0)
        if tolerance_rmse is None:
            tolerance_rmse = self.config.get("audio_validation.tolerance_rmse", 0.1)
        if tolerance_pesq is None:
            tolerance_pesq = self.config.get("audio_validation.tolerance_pesq", 2.5)
        if tolerance_stoi is None:
            tolerance_stoi = self.config.get("audio_validation.tolerance_stoi", 0.8)

        # Calculate metrics
        metrics = self.calculate_audio_metrics(original_path, processed_path)

        if 'error' in metrics:
            return {
                'similar': False,
                'error': metrics['error'],
                'metrics': metrics
            }

        # Check against tolerances
        validation_results = {}

        if 'snr_db' in metrics:
            validation_results['snr_valid'] = metrics['snr_db'] >= tolerance_snr
        else:
            validation_results['snr_valid'] = True  # Skip if not calculated

        if 'rmse' in metrics:
            validation_results['rmse_valid'] = metrics['rmse'] <= tolerance_rmse
        else:
            validation_results['rmse_valid'] = True

        if 'pesq' in metrics:
            validation_results['pesq_valid'] = metrics['pesq'] >= tolerance_pesq
        else:
            validation_results['pesq_valid'] = True

        if 'stoi' in metrics:
            validation_results['stoi_valid'] = metrics['stoi'] >= tolerance_stoi
        else:
            validation_results['stoi_valid'] = True

        # Overall similarity
        overall_valid = all(validation_results.values())

        return {
            'similar': overall_valid,
            'validation_results': validation_results,
            'metrics': metrics,
            'tolerances': {
                'snr_db': tolerance_snr,
                'rmse': tolerance_rmse,
                'pesq': tolerance_pesq,
                'stoi': tolerance_stoi
            }
        }


def create_audio_test_samples(output_dir: str = None) -> Dict[str, str]:
    """
    Create a set of standard test audio samples.

    Args:
        output_dir: Directory to save test samples (optional)

    Returns:
        Dictionary mapping sample names to file paths
    """
    if output_dir is None:
        output_dir = "testing/config/audio_samples"

    os.makedirs(output_dir, exist_ok=True)

    from .test_helpers import create_test_audio_file

    samples = {}

    # Clean speech samples
    samples['speech_clean_male'] = create_test_audio_file(
        duration=3.0, frequency=150, output_path=os.path.join(output_dir, 'speech_clean_male.wav')
    )
    samples['speech_clean_female'] = create_test_audio_file(
        duration=3.0, frequency=220, output_path=os.path.join(output_dir, 'speech_clean_female.wav')
    )

    # Music samples
    samples['music_classical'] = create_test_audio_file(
        duration=5.0, frequency=440, output_path=os.path.join(output_dir, 'music_classical.wav')
    )
    samples['music_rock'] = create_test_audio_file(
        duration=4.0, frequency=330, output_path=os.path.join(output_dir, 'music_rock.wav')
    )

    # Noise samples
    samples['noise_white'] = create_test_audio_file(
        duration=2.0, frequency=1000, output_path=os.path.join(output_dir, 'noise_white.wav')
    )

    # Mixed samples (speech + music)
    samples['speech_with_music'] = create_test_audio_file(
        duration=4.0, frequency=200, output_path=os.path.join(output_dir, 'speech_with_music.wav')
    )

    logger.info(f"Created {len(samples)} test audio samples in {output_dir}")
    return samples


# Global validator instance
audio_validator = AudioValidator()
