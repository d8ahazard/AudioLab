"""
Quality metrics for evaluating RVC models.

Implements objective metrics for comparing voice conversion quality:
- MCD (Mel Cepstral Distortion)
- Pitch Accuracy
- Speaker Similarity
- Optional: WER (Word Error Rate)
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torchaudio
import librosa
from scipy import spatial

logger = logging.getLogger(__name__)


def load_audio(audio_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load audio file and resample to target sample rate.
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate
        
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    return audio, sr


def compute_mcd(audio1_path: str, audio2_path: str, sr: int = 16000) -> float:
    """
    Compute Mel Cepstral Distortion between two audio files.
    
    Lower MCD indicates better similarity.
    
    Args:
        audio1_path: Path to first audio file (e.g., converted)
        audio2_path: Path to second audio file (e.g., target)
        sr: Sample rate
        
    Returns:
        MCD value in dB
    """
    try:
        # Load audio
        audio1, _ = load_audio(audio1_path, sr)
        audio2, _ = load_audio(audio2_path, sr)
        
        # Ensure same length
        min_len = min(len(audio1), len(audio2))
        audio1 = audio1[:min_len]
        audio2 = audio2[:min_len]
        
        # Extract mel-frequency cepstral coefficients (MFCC)
        n_mfcc = 13
        mfcc1 = librosa.feature.mfcc(y=audio1, sr=sr, n_mfcc=n_mfcc)
        mfcc2 = librosa.feature.mfcc(y=audio2, sr=sr, n_mfcc=n_mfcc)
        
        # Ensure same number of frames
        min_frames = min(mfcc1.shape[1], mfcc2.shape[1])
        mfcc1 = mfcc1[:, :min_frames]
        mfcc2 = mfcc2[:, :min_frames]
        
        # Compute MCD (excluding 0th coefficient)
        mfcc1 = mfcc1[1:, :]  # Exclude energy coefficient
        mfcc2 = mfcc2[1:, :]
        
        # MCD formula: (10 / ln(10)) * sqrt(2 * sum((c1 - c2)^2))
        diff = mfcc1 - mfcc2
        mcd = (10.0 / np.log(10)) * np.sqrt(2 * np.mean(np.sum(diff ** 2, axis=0)))
        
        return float(mcd)
        
    except Exception as e:
        logger.error(f"Error computing MCD: {e}")
        return float('inf')


def compute_pitch_accuracy(audio1_path: str, audio2_path: str, sr: int = 16000) -> Dict[str, float]:
    """
    Compute pitch accuracy between two audio files.
    
    Measures how well the pitch contour is preserved.
    
    Args:
        audio1_path: Path to first audio file (e.g., source or converted)
        audio2_path: Path to second audio file (e.g., converted or source)
        sr: Sample rate
        
    Returns:
        Dictionary with pitch metrics:
        - rmse: Root mean square error of F0
        - correlation: Correlation coefficient
        - voiced_frames: Percentage of frames with detected pitch
    """
    try:
        # Load audio
        audio1, _ = load_audio(audio1_path, sr)
        audio2, _ = load_audio(audio2_path, sr)
        
        # Extract pitch using librosa
        f0_1, voiced_flag1, voiced_probs1 = librosa.pyin(
            audio1,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        
        f0_2, voiced_flag2, voiced_probs2 = librosa.pyin(
            audio2,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        
        # Align lengths
        min_len = min(len(f0_1), len(f0_2))
        f0_1 = f0_1[:min_len]
        f0_2 = f0_2[:min_len]
        voiced_flag1 = voiced_flag1[:min_len]
        voiced_flag2 = voiced_flag2[:min_len]
        
        # Only compare voiced frames
        both_voiced = voiced_flag1 & voiced_flag2
        
        if np.sum(both_voiced) < 10:
            logger.warning("Too few voiced frames for pitch comparison")
            return {
                'rmse': float('inf'),
                'correlation': 0.0,
                'voiced_frames_pct': 0.0
            }
        
        f0_1_voiced = f0_1[both_voiced]
        f0_2_voiced = f0_2[both_voiced]
        
        # Compute RMSE
        rmse = np.sqrt(np.mean((f0_1_voiced - f0_2_voiced) ** 2))
        
        # Compute correlation
        correlation = np.corrcoef(f0_1_voiced, f0_2_voiced)[0, 1]
        
        # Percentage of voiced frames
        voiced_pct = 100.0 * np.sum(both_voiced) / len(both_voiced)
        
        return {
            'rmse': float(rmse),
            'correlation': float(correlation),
            'voiced_frames_pct': float(voiced_pct)
        }
        
    except Exception as e:
        logger.error(f"Error computing pitch accuracy: {e}")
        return {
            'rmse': float('inf'),
            'correlation': 0.0,
            'voiced_frames_pct': 0.0
        }


def compute_speaker_similarity(
    audio1_path: str,
    audio2_path: str,
    model_name: str = "speechbrain/spkrec-ecapa-voxceleb"
) -> float:
    """
    Compute speaker similarity using pretrained speaker verification model.
    
    Higher similarity indicates better voice conversion.
    
    Args:
        audio1_path: Path to first audio file (e.g., converted)
        audio2_path: Path to second audio file (e.g., target)
        model_name: Name of speaker verification model
        
    Returns:
        Cosine similarity (0 to 1, higher is better)
    """
    try:
        from speechbrain.pretrained import EncoderClassifier
        
        # Load speaker verification model
        classifier = EncoderClassifier.from_hparams(
            source=model_name,
            savedir="models/speechbrain"
        )
        
        # Extract embeddings
        embedding1 = classifier.encode_batch(torch.tensor(load_audio(audio1_path)[0]).unsqueeze(0))
        embedding2 = classifier.encode_batch(torch.tensor(load_audio(audio2_path)[0]).unsqueeze(0))
        
        # Compute cosine similarity
        similarity = 1 - spatial.distance.cosine(
            embedding1.squeeze().cpu().numpy(),
            embedding2.squeeze().cpu().numpy()
        )
        
        return float(similarity)
        
    except ImportError:
        logger.warning("SpeechBrain not installed, skipping speaker similarity")
        return 0.0
    except Exception as e:
        logger.error(f"Error computing speaker similarity: {e}")
        return 0.0


def compute_wer(audio_path: str, reference_text: str) -> float:
    """
    Compute Word Error Rate using ASR.
    
    Lower WER indicates better intelligibility.
    
    Args:
        audio_path: Path to audio file
        reference_text: Reference transcription
        
    Returns:
        WER as a percentage (0-100)
    """
    try:
        import whisper
        from jiwer import wer
        
        # Load Whisper model
        model = whisper.load_model("base")
        
        # Transcribe
        result = model.transcribe(audio_path)
        hypothesis = result['text'].strip().lower()
        reference = reference_text.strip().lower()
        
        # Compute WER
        error_rate = wer(reference, hypothesis) * 100
        
        return float(error_rate)
        
    except ImportError:
        logger.warning("Whisper or jiwer not installed, skipping WER")
        return 0.0
    except Exception as e:
        logger.error(f"Error computing WER: {e}")
        return 0.0


def evaluate_conversion(
    converted_audio: str,
    target_audio: str,
    source_audio: Optional[str] = None,
    reference_text: Optional[str] = None
) -> Dict[str, any]:
    """
    Comprehensive evaluation of voice conversion.
    
    Args:
        converted_audio: Path to converted audio
        target_audio: Path to target speaker audio
        source_audio: Optional path to source audio for pitch comparison
        reference_text: Optional reference text for WER
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info(f"Evaluating conversion...")
    logger.info(f"  Converted: {converted_audio}")
    logger.info(f"  Target: {target_audio}")
    
    results = {}
    
    # MCD: converted vs target (lower is better)
    logger.info("Computing MCD...")
    results['mcd'] = compute_mcd(converted_audio, target_audio)
    logger.info(f"  MCD: {results['mcd']:.2f} dB")
    
    # Pitch accuracy
    if source_audio:
        logger.info("Computing pitch accuracy...")
        pitch_metrics = compute_pitch_accuracy(source_audio, converted_audio)
        results['pitch_rmse'] = pitch_metrics['rmse']
        results['pitch_correlation'] = pitch_metrics['correlation']
        results['voiced_frames_pct'] = pitch_metrics['voiced_frames_pct']
        logger.info(f"  Pitch RMSE: {results['pitch_rmse']:.2f} Hz")
        logger.info(f"  Pitch correlation: {results['pitch_correlation']:.3f}")
        logger.info(f"  Voiced frames: {results['voiced_frames_pct']:.1f}%")
    
    # Speaker similarity: converted vs target (higher is better)
    logger.info("Computing speaker similarity...")
    results['speaker_similarity_target'] = compute_speaker_similarity(converted_audio, target_audio)
    logger.info(f"  Similarity to target: {results['speaker_similarity_target']:.3f}")
    
    # Speaker similarity: converted vs source (should be lower)
    if source_audio:
        results['speaker_similarity_source'] = compute_speaker_similarity(converted_audio, source_audio)
        logger.info(f"  Similarity to source: {results['speaker_similarity_source']:.3f}")
    
    # WER
    if reference_text:
        logger.info("Computing WER...")
        results['wer'] = compute_wer(converted_audio, reference_text)
        logger.info(f"  WER: {results['wer']:.1f}%")
    
    # Composite score (normalized)
    # MCD: normalize to 0-1 (assuming typical range 5-15 dB)
    mcd_norm = 1.0 - min(max((results['mcd'] - 5) / 10, 0), 1)
    
    # Speaker similarity: already 0-1
    spk_sim = results.get('speaker_similarity_target', 0)
    
    # Composite: weighted average
    results['composite_score'] = 0.5 * mcd_norm + 0.5 * spk_sim
    logger.info(f"\nComposite score: {results['composite_score']:.3f}")
    
    return results


def compare_models(
    model1_converted: str,
    model2_converted: str,
    target_audio: str,
    model1_name: str = "Model 1",
    model2_name: str = "Model 2"
) -> Dict[str, any]:
    """
    Compare two models' conversion results.
    
    Args:
        model1_converted: Path to model 1 converted audio
        model2_converted: Path to model 2 converted audio
        target_audio: Path to target audio
        model1_name: Name of first model
        model2_name: Name of second model
        
    Returns:
        Comparison results
    """
    logger.info(f"\nComparing {model1_name} vs {model2_name}")
    logger.info("=" * 60)
    
    # Evaluate both
    results1 = evaluate_conversion(model1_converted, target_audio)
    results2 = evaluate_conversion(model2_converted, target_audio)
    
    # Compare
    comparison = {
        'model1_name': model1_name,
        'model2_name': model2_name,
        'model1_results': results1,
        'model2_results': results2,
        'improvements': {}
    }
    
    # Calculate improvements
    for metric in ['mcd', 'speaker_similarity_target', 'composite_score']:
        if metric in results1 and metric in results2:
            if metric == 'mcd':
                # Lower is better for MCD
                improvement = (results1[metric] - results2[metric]) / results1[metric] * 100
            else:
                # Higher is better for others
                improvement = (results2[metric] - results1[metric]) / max(results1[metric], 0.001) * 100
            
            comparison['improvements'][metric] = improvement
    
    # Print comparison
    logger.info(f"\nResults:")
    logger.info(f"{'Metric':<30} {model1_name:<15} {model2_name:<15} {'Improvement':<15}")
    logger.info("-" * 80)
    logger.info(f"{'MCD (dB)':<30} {results1.get('mcd', 0):<15.2f} {results2.get('mcd', 0):<15.2f} {comparison['improvements'].get('mcd', 0):>+14.1f}%")
    logger.info(f"{'Speaker Similarity':<30} {results1.get('speaker_similarity_target', 0):<15.3f} {results2.get('speaker_similarity_target', 0):<15.3f} {comparison['improvements'].get('speaker_similarity_target', 0):>+14.1f}%")
    logger.info(f"{'Composite Score':<30} {results1.get('composite_score', 0):<15.3f} {results2.get('composite_score', 0):<15.3f} {comparison['improvements'].get('composite_score', 0):>+14.1f}%")
    
    if comparison['improvements'].get('composite_score', 0) > 0:
        logger.info(f"\n✓ {model2_name} is better overall!")
    else:
        logger.info(f"\n✓ {model1_name} is better overall!")
    
    return comparison

