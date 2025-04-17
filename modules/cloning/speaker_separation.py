"""
Speaker separation module for AudioLab.

This module provides functionality for separating different speakers
in an audio file using pyannote.audio and other speaker diarization techniques.
"""

import os
import gc
import logging
import torch
from typing import List, Dict, Tuple
from pydub import AudioSegment

from handlers.config import model_path
from modules.cloning.model_utils import load_pyannote_pipeline

logger = logging.getLogger(__name__)

# Global model reference
diarization_pipeline = None


def initialize_diarization():
    """
    Initialize the speaker diarization pipeline.
    This is done lazily on first use to avoid unnecessary imports and memory usage.
    """
    global diarization_pipeline
    
    if diarization_pipeline is not None:
        # Already initialized
        return True
    
    try:
        logger.info("Loading speaker diarization model")
        
        # Default model path
        model_id = "pyannote/speaker-diarization-3.1"
        
        # Look for HF auth token
        hub_token = os.environ.get("HF_TOKEN", None)
        
        # Initialize the pipeline using our custom function
        diarization_pipeline = load_pyannote_pipeline(model_id, hub_token)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            diarization_pipeline = diarization_pipeline.to(torch.device("cuda"))
            
        logger.info("Speaker diarization model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize speaker diarization model: {str(e)}")
        return False


def transcribe_audio(audio_file: str, output_dir: str) -> List[str]:
    """
    Process an audio file to identify different speakers.
    
    Args:
        audio_file: Path to the input audio file
        output_dir: Directory to save speaker information
        
    Returns:
        List of file paths containing speaker timestamps
    """
    try:
        # Initialize the diarization pipeline if needed
        if not initialize_diarization():
            logger.error("Failed to initialize speaker diarization")
            return []
            
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
            
        # Run the pipeline on the audio file
        logger.info(f"Processing audio file for speaker diarization: {audio_file}")
        diarization = diarization_pipeline(audio_file)
        
        # Extract speaker segments
        speakers = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker not in speakers:
                speakers[speaker] = []
            speakers[speaker].append((turn.start, turn.end))
            logger.debug(f"Speaker {speaker}: {turn.start:.1f}s - {turn.end:.1f}s")
        
        # Save speaker segments to files
        speaker_files = []
        for speaker in speakers:
            speaker_file = os.path.join(output_dir, f"speaker_{speaker}.txt")
            logger.info(f"Saving speaker {speaker} segments to {speaker_file}")
            
            with open(speaker_file, "w") as f:
                for start, end in speakers[speaker]:
                    f.write(f"start={start}s|stop={end}s\n")
                    
            speaker_files.append(speaker_file)
            
        return speaker_files
        
    except Exception as e:
        logger.error(f"Error in speaker diarization: {str(e)}")
        return []
    finally:
        # Clean up memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def separate_speakers(audio_file: str, speaker_files: List[str], output_dir: str) -> List[str]:
    """
    Separate individual speakers from an audio file based on diarization results.
    
    Args:
        audio_file: Path to the original audio file
        speaker_files: List of files containing speaker timestamps
        output_dir: Directory to save separated audio files
        
    Returns:
        List of paths to separated speaker audio files
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the original audio
        logger.info(f"Loading original audio: {audio_file}")
        original_audio = AudioSegment.from_file(audio_file)
        
        output_files = []
        
        # Process each speaker
        for speaker_idx, speaker_file in enumerate(speaker_files):
            logger.info(f"Processing speaker {speaker_idx} from {speaker_file}")
            
            # Read the speaker intervals
            intervals = []
            with open(speaker_file, 'r') as file:
                for line in file.readlines():
                    parts = line.strip().split('|')
                    start = float(parts[0].split('=')[1].rstrip('s')) * 1000  # Convert to milliseconds
                    end = float(parts[1].split('=')[1].rstrip('s')) * 1000
                    intervals.append((start, end))
            
            # Create a silent audio segment of the same length as the original
            speaker_audio = AudioSegment.silent(duration=len(original_audio))
            
            # Add each segment of this speaker's audio
            for start, end in intervals:
                speaker_audio = speaker_audio.overlay(original_audio[start:end], position=start)
            
            # Save the isolated speaker audio
            output_path = os.path.join(output_dir, f"speaker_{speaker_idx}.wav")
            speaker_audio.export(output_path, format='wav')
            output_files.append(output_path)
            
            logger.info(f"Saved separated audio for speaker {speaker_idx} to {output_path}")
            
        return output_files
        
    except Exception as e:
        logger.error(f"Error separating speakers: {str(e)}")
        return []


def get_speaker_segments(audio_file: str, output_dir: str) -> Dict[str, List[Tuple[float, float]]]:
    """
    Get the segments for each speaker in an audio file.
    
    Args:
        audio_file: Path to the input audio file
        output_dir: Directory to save speaker information
        
    Returns:
        Dictionary mapping speaker IDs to lists of (start, end) time tuples
    """
    try:
        # Initialize the diarization pipeline if needed
        if not initialize_diarization():
            logger.error("Failed to initialize speaker diarization")
            return {}
            
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
            
        # Run the pipeline on the audio file
        logger.info(f"Processing audio file for speaker diarization: {audio_file}")
        diarization = diarization_pipeline(audio_file)
        
        # Extract speaker segments
        speakers = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker not in speakers:
                speakers[speaker] = []
            speakers[speaker].append((turn.start, turn.end))
            
        return speakers
        
    except Exception as e:
        logger.error(f"Error getting speaker segments: {str(e)}")
        return {}


def cleanup():
    """Release resources used by speaker diarization."""
    global diarization_pipeline
    
    if diarization_pipeline is not None:
        # Move model to CPU before deletion helps free GPU memory
        if torch.cuda.is_available():
            try:
                diarization_pipeline = diarization_pipeline.to('cpu')
            except:
                pass
        
        del diarization_pipeline
        diarization_pipeline = None
        
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    logger.info("Speaker diarization resources cleaned up") 