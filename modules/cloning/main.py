"""
Main module to coordinate voice cloning and speaker separation for AudioLab.

This module provides a unified interface to the various voice cloning
and speaker separation methods available in AudioLab.
"""

import os
import logging
from typing import List, Optional, Dict, Any

from . import openvoice
from . import tts
from . import speaker_separation

logger = logging.getLogger(__name__)

        
def clone_voice_openvoice(
    target_file: str, 
    source_file: str, 
    output_dir: str, 
    strength: float = 0.5
) -> Optional[str]:
    """
    Clone a voice using OpenVoice.
    
    Args:
        target_file: Path to the target speaker audio file (whose voice will be modified)
        source_file: Path to the source speaker audio file (voice to clone)
        output_dir: Directory to save the output
        strength: Tone color conversion strength (0.0-1.0)
        
    Returns:
        Path to the generated audio file or None if failed
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    base_name = os.path.splitext(os.path.basename(target_file))[0]
    source_base = os.path.splitext(os.path.basename(source_file))[0]
    output_file = os.path.join(output_dir, f"{base_name}_cloned_openvoice_{source_base}.wav")
    
    return openvoice.clone_voice(
        target_speaker_path=target_file,
        source_speaker_path=source_file,
        output_path=output_file,
        tau=strength
    )


def clone_voice_tts(
    target_file: str, 
    source_file: str, 
    output_dir: str,
    custom_text: Optional[str] = None
) -> Optional[str]:
    """
    Clone a voice using TTS.
    
    Args:
        target_file: Path to the target speaker audio file (whose voice will be modified)
        source_file: Path to the source speaker audio file (voice to clone)
        output_dir: Directory to save the output
        custom_text: Optional custom text to synthesize instead of extracting from target_file
        
    Returns:
        Path to the generated audio file or None if failed
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    base_name = os.path.splitext(os.path.basename(target_file))[0]
    source_base = os.path.splitext(os.path.basename(source_file))[0]
    output_file = os.path.join(output_dir, f"{base_name}_cloned_tts_{source_base}.wav")
    
    return tts.clone_voice(
        target_speaker_path=target_file,
        source_speaker_path=source_file,
        output_path=output_file,
        text=custom_text
    )


def transcribe_audio(audio_file: str, output_dir: str) -> List[str]:
    """
    Process an audio file to identify different speakers.
    
    Args:
        audio_file: Path to the input audio file
        output_dir: Directory to save speaker information
        
    Returns:
        List of file paths containing speaker timestamps
    """
    return speaker_separation.transcribe_audio(audio_file, output_dir)


def separate_speakers(audio_file: str, output_dir: str) -> List[str]:
    """
    Separate different speakers in an audio file.
    
    Args:
        audio_file: Path to the input audio file
        output_dir: Directory to save separated audio files
        
    Returns:
        List of paths to separated speaker audio files
    """
    # First transcribe to get speaker segments
    speaker_files = transcribe_audio(audio_file, output_dir)
    
    # Then separate the speakers
    return speaker_separation.separate_speakers(audio_file, speaker_files, output_dir)


def choose_speaker(audio_file: str, output_dir: str, speaker_idx: int = 0) -> Optional[str]:
    """
    Extract a specific speaker from an audio file.
    
    Args:
        audio_file: Path to the input audio file
        output_dir: Directory to save output
        speaker_idx: Index of the speaker to extract (starting from 0)
        
    Returns:
        Path to the extracted speaker audio file or None if failed
    """
    # Separate all speakers
    speaker_files = separate_speakers(audio_file, output_dir)
    
    # Check if the requested speaker index is valid
    if not speaker_files or speaker_idx >= len(speaker_files):
        logger.error(f"Speaker index {speaker_idx} is out of range (found {len(speaker_files)} speakers)")
        return None
        
    return speaker_files[speaker_idx]


def cleanup():
    """Clean up all resources used by the cloning modules."""
    openvoice.cleanup()
    tts.cleanup()
    speaker_separation.cleanup()
    logger.info("All cloning resources cleaned up") 