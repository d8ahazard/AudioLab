"""
OpenVoice voice cloning implementation for AudioLab.

This module provides access to the OpenVoice voice cloning method,
which allows transferring the vocal timbre of a reference speaker to a target speech.
"""

import os
import gc
import logging
import shutil
import torch
import traceback
import librosa
import numpy as np
from typing import Optional, List, Tuple
from openvoice_cli.api import ToneColorConverter
import openvoice_cli.se_extractor as se_extractor
from pydub import AudioSegment

from handlers.config import model_path, output_path
from modules.cloning.model_utils import initialize_openvoice_model

logger = logging.getLogger(__name__)

# Will be initialized when needed
tone_color_converter = None

def split_audio_into_chunks(audio_path: str, chunk_duration_seconds: float = 10.0, base_temp_dir: str = None) -> Tuple[str, List[str]]:
    """
    Split audio file into smaller chunks for processing while maintaining exact timing.

    Args:
        audio_path: Path to the audio file to split
        chunk_duration_seconds: Duration of each chunk in seconds
        base_temp_dir: Base temp directory (project dir) for scoped chunking

    Returns:
        Tuple of (temp_directory_path, list_of_chunk_paths)
    """
    try:
        # Create unique temp directory for chunks
        import hashlib
        audio_name = os.path.basename(audio_path).rsplit('.', 1)[0]
        audio_hash = hashlib.md5(open(audio_path, 'rb').read()).hexdigest()[:8]
        # Determine project-scoped temp directory if provided
        if base_temp_dir:
            temp_dir = os.path.join(base_temp_dir, "temp", f"openvoice_chunks_{audio_name}_{audio_hash}")
        else:
            temp_dir = os.path.join(output_path, "temp", f"openvoice_chunks_{audio_name}_{audio_hash}")
        os.makedirs(temp_dir, exist_ok=True)

        # Load audio and get precise information
        audio = AudioSegment.from_file(audio_path)
        audio_duration = audio.duration_seconds
        sample_rate = audio.frame_rate
        num_channels = audio.channels

        logger.info(f"Splitting audio into {chunk_duration_seconds}s chunks (duration: {audio_duration:.2f}s, sample_rate: {sample_rate}Hz, channels: {num_channels})")

        # Calculate chunk boundaries based on time, not samples
        chunk_paths = []
        num_chunks = int(np.ceil(audio_duration / chunk_duration_seconds))

        for i in range(num_chunks):
            # Calculate time boundaries for this chunk
            start_time_ms = i * chunk_duration_seconds * 1000
            end_time_ms = min((i + 1) * chunk_duration_seconds * 1000, audio_duration * 1000)

            # Extract chunk with precise time boundaries
            chunk = audio[int(start_time_ms):int(end_time_ms)]

            chunk_path = os.path.join(temp_dir, f"chunk_{i:03d}.wav")
            chunk.export(chunk_path, format='wav')
            chunk_paths.append(chunk_path)

        logger.info(f"Created {len(chunk_paths)} chunks in {temp_dir}")
        return temp_dir, chunk_paths

    except Exception as e:
        logger.error(f"Error splitting audio into chunks: {e}")
        return None, []

def concatenate_audio_chunks(chunk_paths: List[str], output_path: str) -> bool:
    """
    Concatenate processed audio chunks back into a single file with perfect alignment.

    Args:
        chunk_paths: List of processed chunk file paths in order
        output_path: Path to save the concatenated audio

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Concatenating {len(chunk_paths)} processed chunks with perfect alignment")

        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Remove existing output file if it exists (Windows file locking issue)
        if os.path.exists(output_path):
            try:
                # Wait a moment for any file handles to be released
                import time
                time.sleep(0.1)
                os.remove(output_path)
                logger.info(f"Removed existing output file: {output_path}")
            except Exception as e:
                logger.warning(f"Could not remove existing output file: {e}")
                # Try renaming it first then deleting
                try:
                    backup_path = output_path + ".old"
                    if os.path.exists(backup_path):
                        os.remove(backup_path)
                    os.rename(output_path, backup_path)
                    os.remove(backup_path)
                    logger.info(f"Removed output file via rename workaround")
                except Exception as e2:
                    logger.error(f"Could not remove output file even with workaround: {e2}")

        # Load and concatenate all chunks without any gaps
        combined_audio = AudioSegment.empty()

        for i, chunk_path in enumerate(chunk_paths):
            if os.path.exists(chunk_path):
                chunk_audio = AudioSegment.from_file(chunk_path)
                # Concatenate directly without silence - maintain exact timing
                combined_audio += chunk_audio
                logger.debug(f"Added chunk {i+1}/{len(chunk_paths)}: {len(chunk_audio)} samples")
                # Release the chunk audio reference to free memory
                del chunk_audio

        # Export final audio with explicit file closing
        with open(output_path, 'wb') as f:
            combined_audio.export(f, format='wav')
        
        logger.info(f"Successfully concatenated audio to {output_path} (duration: {combined_audio.duration_seconds:.2f}s)")

        return True

    except Exception as e:
        logger.error(f"Error concatenating audio chunks: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def cleanup_temp_files(temp_dir: str):
    """Clean up temporary files and directories."""
    try:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        logger.warning(f"Error cleaning up temp files: {e}")

def initialize_openvoice():
    """
    Initialize the OpenVoice converter and dependencies.
    This is done lazily on first use to avoid unnecessary imports and memory usage.
    """
    global tone_color_converter
    
    if tone_color_converter is not None:
        # Already initialized
        return True
    
    try:
        # Import OpenVoice only when needed
        
        logger.info("Loading OpenVoice dependencies")
        
        # Get the model paths using our custom function
        config_path, checkpoint_path = initialize_openvoice_model()
            
        # Initialize models
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing OpenVoice converter on {device}")
        # Extract enable_watermark from kwargs before passing to parent
        enable_watermark = False
        tone_color_converter = ToneColorConverter(config_path, device=device)
        # Set watermark setting after initialization
        if not enable_watermark:
            tone_color_converter.watermark_model = None
        tone_color_converter.load_ckpt(checkpoint_path)
        
        logger.info("OpenVoice initialized successfully")
        return True
    
    except Exception as e:
        logger.error(f"Failed to initialize OpenVoice: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


def clone_voice_chunked(target_speaker_path: str, source_speaker_path: str, output_path: str, tau: float = 0.5, chunk_duration: float = 10.0, temp_dir: str = None) -> Optional[str]:
    """
    Clone a voice using OpenVoice with chunking for large audio files.

    Args:
        target_speaker_path: Path to the target speaker audio file (whose voice will be modified)
        source_speaker_path: Path to the source speaker audio file (voice to clone)
        output_path: Path to save the resulting audio
        tau: Tone color conversion strength (0.0-1.0)
        chunk_duration: Duration of each chunk in seconds

    Returns:
        Path to the generated audio file or None if failed
    """
    created_temp_dir = None
    processed_chunks = []

    try:
        # Initialize OpenVoice if needed
        if not initialize_openvoice():
            logger.error("Failed to initialize OpenVoice")
            return None

        # Check if audio file is small enough to process directly
        try:
            audio = AudioSegment.from_file(target_speaker_path)
            if audio.duration_seconds <= chunk_duration:
                logger.info(f"Audio is short ({audio.duration_seconds:.2f}s), processing directly")
                return _clone_voice_direct(target_speaker_path, source_speaker_path, output_path, tau)
        except Exception as e:
            logger.warning(f"Could not check audio duration: {e}")

        # Split audio into chunks
        logger.info(f"Splitting audio into {chunk_duration}s chunks for chunked processing")
        created_temp_dir, chunk_paths = split_audio_into_chunks(target_speaker_path, chunk_duration, base_temp_dir=temp_dir)

        if not chunk_paths:
            logger.error("Failed to split audio into chunks")
            return None

        # Extract speaker embeddings once (they don't change per chunk)
        logger.info(f"Extracting speaker embedding for source: {source_speaker_path}")
        source_se, _ = se_extractor.get_se(source_speaker_path, tone_color_converter, target_dir=output_path, vad=True)

        logger.info(f"Extracting speaker embedding for target: {target_speaker_path}")
        target_se, _ = se_extractor.get_se(target_speaker_path, tone_color_converter, target_dir=output_path, vad=True)

        # Process each chunk
        for i, chunk_path in enumerate(chunk_paths):
            logger.info(f"Processing chunk {i+1}/{len(chunk_paths)}: {os.path.basename(chunk_path)}")

            # Create output path for this chunk
            chunk_output_path = chunk_path.replace('.wav', '_processed.wav')

            try:
                # Run the tone color converter on this chunk
                tone_color_converter.convert(
                    audio_src_path=chunk_path,
                    src_se=target_se,  # Use target speaker embedding as source for conversion (voice to convert FROM)
                    tgt_se=source_se,  # Use source speaker embedding as target (voice to convert TO)
                    output_path=chunk_output_path,
                    tau=tau,
                )

                processed_chunks.append(chunk_output_path)
                logger.info(f"Successfully processed chunk {i+1}/{len(chunk_paths)}")

            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {e}")
                # Continue with other chunks even if one fails
                continue

        if not processed_chunks:
            logger.error("No chunks were successfully processed")
            return None

        # Concatenate processed chunks
        logger.info(f"Concatenating {len(processed_chunks)} processed chunks")
        success = concatenate_audio_chunks(processed_chunks, output_path)

        if success:
            logger.info(f"Voice cloning complete: {output_path}")
            return output_path
        else:
            logger.error("Failed to concatenate processed chunks")
            return None

    except Exception as e:
        logger.error(f"Error in chunked OpenVoice cloning: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None
    finally:
        # Clean up temporary files
        if created_temp_dir:
            cleanup_temp_files(created_temp_dir)
        # Clean up processed chunks
        for chunk_path in processed_chunks:
            try:
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
            except:
                pass

        # Clean up memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def clone_voice(target_speaker_path: str, source_speaker_path: str, output_path: str, tau: float = 0.5, chunk_duration: float = 10.0, temp_dir: str = None) -> Optional[str]:
    """
    Clone a voice using OpenVoice with automatic chunking for large files.

    Args:
        target_speaker_path: Path to the target speaker audio file (whose voice will be modified)
        source_speaker_path: Path to the source speaker audio file (voice to clone)
        output_path: Path to save the resulting audio
        tau: Tone color conversion strength (0.0-1.0)
        chunk_duration: Duration of each chunk in seconds for large files

    Returns:
        Path to the generated audio file or None if failed
    """
    try:
        # Check if audio file is small enough to process directly
        try:
            audio = AudioSegment.from_file(target_speaker_path)
            if audio.duration_seconds <= chunk_duration:
                logger.info(f"Audio is short ({audio.duration_seconds:.2f}s), processing directly")
                return _clone_voice_direct(target_speaker_path, source_speaker_path, output_path, tau)
        except Exception as e:
            logger.warning(f"Could not check audio duration: {e}")

        # For larger files, use chunked processing
        logger.info(f"Audio appears to be large, using chunked processing with {chunk_duration}s chunks")
        return clone_voice_chunked(target_speaker_path, source_speaker_path, output_path, tau, chunk_duration, temp_dir=temp_dir)

    except Exception as e:
        logger.error(f"Error in OpenVoice cloning: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None


def _clone_voice_direct(target_speaker_path: str, source_speaker_path: str, output_path: str, tau: float = 0.5) -> Optional[str]:
    """
    Clone a voice using OpenVoice.
    
    Args:
        target_speaker_path: Path to the target speaker audio file (whose voice will be modified)
        source_speaker_path: Path to the source speaker audio file (voice to clone)
        output_path: Path to save the resulting audio
        tau: Tone color conversion strength (0.0-1.0)
        
    Returns:
        Path to the generated audio file or None if failed
    """
    try:
        # Initialize OpenVoice if needed
        if not initialize_openvoice():
            logger.error("Failed to initialize OpenVoice")
            return None
            
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        logger.info(f"Extracting speaker embedding for source: {source_speaker_path}")
        source_se, _ = se_extractor.get_se(source_speaker_path, tone_color_converter, target_dir=output_path, vad=True)

        logger.info(f"Extracting speaker embedding for target: {target_speaker_path}")
        target_se, _ = se_extractor.get_se(target_speaker_path, tone_color_converter, target_dir=output_path, vad=True)
        
        # Run the tone color converter
        logger.info(f"Converting voice with tau={tau}")
        tone_color_converter.convert(
            audio_src_path=target_speaker_path,
            src_se=target_se,
            tgt_se=source_se,
            output_path=output_path,
            tau=tau,
        )
        
        # Clean up temporary files
        processed_path = os.path.join(os.path.dirname(output_path), "processed")
        if os.path.exists(processed_path):
            shutil.rmtree(processed_path)
            
        logger.info(f"Voice cloning complete: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error in OpenVoice cloning: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None
    finally:
        # Clean up memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def cleanup():
    """Release resources used by OpenVoice."""
    global tone_color_converter

    if tone_color_converter is not None:
        del tone_color_converter
        tone_color_converter = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("OpenVoice resources cleaned up")