"""
Text-to-Speech based voice cloning implementation for AudioLab.

This module provides access to TTS-based voice cloning method.
"""

import os
import gc
import logging
import torch
import tempfile
import numpy as np
import soundfile as sf
from typing import Optional, Tuple

from handlers.config import model_path
from modules.cloning.model_utils import get_huggingface_model_path

logger = logging.getLogger(__name__)

# Will be initialized when needed
tts_model = None
vocoder = None
tts_is_loaded = False


def initialize_tts():
    """
    Initialize the TTS models.
    This is done lazily on first use to avoid unnecessary imports and memory usage.
    """
    global tts_model, vocoder, tts_is_loaded
    
    if tts_is_loaded:
        return True
    
    try:
        # Import TTS modules only when needed
        logger.info("Loading TTS dependencies")
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        
        # Setup model paths
        model_name = "coqui/XTTS-v2"
        xtts_path = get_huggingface_model_path(model_name, "tts")
        
        logger.info(f"Loading TTS model from {xtts_path}")
        
        # Initialize models
        config = XttsConfig()
        config.load_json(os.path.join(xtts_path, "config.json"))
        
        # Adjust paths to be absolute
        if not os.path.isabs(config.model_args.speaker_encoder_model_path):
            config.model_args.speaker_encoder_model_path = os.path.join(
                xtts_path, config.model_args.speaker_encoder_model_path
            )
            
        if not os.path.isabs(config.model_args.checkpoint_file):
            config.model_args.checkpoint_file = os.path.join(
                xtts_path, config.model_args.checkpoint_file
            )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Load the model
        tts_model = Xtts.init_from_config(config)
        tts_model.load_checkpoint(config, config.model_args.checkpoint_file, eval=True)
        tts_model.to(device)
        
        # No need for separate vocoder in XTTS v2
        vocoder = None
        
        tts_is_loaded = True
        logger.info("TTS loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize TTS: {str(e)}")
        return False


def extract_speech_to_text(audio_path: str, max_duration: int = 120) -> str:
    """
    Extract text from an audio file using a speech recognition model.
    
    Args:
        audio_path: Path to the audio file
        max_duration: Maximum duration in seconds to process
        
    Returns:
        Extracted text
    """
    try:
        import whisper
        
        # Load a small model for quick transcription
        model = whisper.load_model("base")
        
        # Transcribe audio
        logger.info(f"Transcribing audio: {audio_path}")
        result = model.transcribe(audio_path, fp16=torch.cuda.is_available())
        
        logger.info(f"Transcription complete. Text length: {len(result['text'])}")
        return result["text"]
        
    except Exception as e:
        logger.error(f"Error in speech-to-text extraction: {e}")
        return ""


def clone_voice(target_speaker_path: str, source_speaker_path: str, output_path: str, text: Optional[str] = None) -> Optional[str]:
    """
    Clone a voice using TTS.
    
    Args:
        target_speaker_path: Path to the target speaker audio file (whose voice will be modified or text will be used)
        source_speaker_path: Path to the source speaker audio file (voice to clone)
        output_path: Path to save the resulting audio
        text: Optional text to synthesize. If None, text will be extracted from target_speaker_path
        
    Returns:
        Path to the generated audio file or None if failed
    """
    try:
        # Initialize TTS if needed
        if not initialize_tts():
            logger.error("Failed to initialize TTS")
            return None
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        # If no text is provided, extract it from the target audio
        synthesize_text = text
        if synthesize_text is None:
            logger.info(f"Extracting speech from target audio: {target_speaker_path}")
            synthesize_text = extract_speech_to_text(target_speaker_path)
            
        if not synthesize_text:
            logger.error("No text to synthesize")
            return None
            
        # Generate audio with voice cloning
        logger.info("Generating TTS audio with voice cloning")
        
        # Get speaker embedding from source speaker
        gpt_cond_latent, speaker_embedding = tts_model.get_conditioning_latents(
            audio_path=source_speaker_path
        )
        
        # Determine language (simple heuristic, can be improved)
        language = "en"
        
        # Generate audio with the provided text and speaker embedding
        logger.info(f"Synthesizing text with cloned voice: {synthesize_text[:50]}...")
        model_output = tts_model.inference(
            text=synthesize_text,
            language=language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=0.75,
        )
        
        # Save the audio file
        output_sample_rate = tts_model.output_sample_rate
        sf.write(output_path, model_output, output_sample_rate)
        
        logger.info(f"TTS cloning complete: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error in TTS cloning: {e}")
        return None
    finally:
        # Clean up memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def cleanup():
    """Release resources used by TTS."""
    global tts_model, vocoder, tts_is_loaded
    
    if tts_model is not None:
        del tts_model
        tts_model = None
        
    if vocoder is not None:
        del vocoder
        vocoder = None
        
    tts_is_loaded = False
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    logger.info("TTS resources cleaned up") 