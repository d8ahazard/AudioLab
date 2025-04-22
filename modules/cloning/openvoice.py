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
from typing import Optional
from openvoice_cli.api import ToneColorConverter
import openvoice_cli.se_extractor as se_extractor
        
from handlers.config import model_path
from modules.cloning.model_utils import initialize_openvoice_model

logger = logging.getLogger(__name__)

# Will be initialized when needed
tone_color_converter = None

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
        tone_color_converter = ToneColorConverter(config_path, device=device)
        tone_color_converter.load_ckpt(checkpoint_path)
        
        logger.info("OpenVoice initialized successfully")
        return True
    
    except Exception as e:
        logger.error(f"Failed to initialize OpenVoice: {str(e)}")
        return False


def clone_voice(target_speaker_path: str, source_speaker_path: str, output_path: str, tau: float = 0.5) -> Optional[str]:
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
        source_se, _ = se_extractor.get_se(source_speaker_path, tone_color_converter, vad=True)
        
        logger.info(f"Extracting speaker embedding for target: {target_speaker_path}")
        target_se, _ = se_extractor.get_se(target_speaker_path, tone_color_converter, vad=True)
        
        # Run the tone color converter
        logger.info(f"Converting voice with tau={tau}")
        tone_color_converter.convert(
            audio_src_path=target_speaker_path,
            src_se=source_se,
            tgt_se=target_se,
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