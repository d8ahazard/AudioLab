"""
Orpheus TTS Model handler for AudioLab.
"""

import os
import logging
import time
import numpy as np
import torch
import wave
from typing import List, Dict, Optional, Union, Tuple, Generator
from pathlib import Path

from handlers.config import model_path, output_path

logger = logging.getLogger("ADLB.Orpheus")

class OrpheusModel:
    """
    Wrapper for the Orpheus TTS model.
    """
    AVAILABLE_VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
    AVAILABLE_EMOTIONS = ["", "happy", "sad", "angry", "scared", "disgusted", "surprised"]
    
    def __init__(self, model_name: str = "canopylabs/orpheus-tts-0.1-finetune-prod"):
        """
        Initialize the Orpheus TTS model.
        
        Args:
            model_name: The name or path of the model to use. Either a Hugging Face model ID
                       or a local path to a model directory.
        """
        self.model_name = model_name
        self.model = None
        self.model_dir = os.path.join(model_path, "orpheus")
        self.repo_dir = os.path.join(self.model_dir, "Orpheus-TTS")
        os.makedirs(self.model_dir, exist_ok=True)
    
    def load_model(self):
        """
        Load the Orpheus TTS model.
        """
        try:
            from orpheus_tts import OrpheusModel as OrpheusTTS
        except ImportError as e:
            if "vllm._C" in str(e):
                logger.error("The vllm package is not properly installed. This is a dependency of orpheus_tts.")
                logger.error("Please make sure you have properly installed vllm==0.7.3 with the correct CUDA version.")
                logger.error("Try running the setup script again, or install manually: pip install vllm==0.7.3")
                raise ImportError("The vllm package is not properly installed. Please run the setup script again.") from e
            else:
                logger.error("orpheus_tts package not installed. Please run the setup script first.")
                logger.error(f"Import error details: {e}")
                raise ImportError("orpheus_tts package not installed. Please run the setup script first.") from e
        
        if self.model is not None:
            logger.info("Model already loaded")
            return self.model
        
        logger.info(f"Loading Orpheus TTS model: {self.model_name}")
        
        # If the model_name is a local path, use it directly
        if os.path.exists(self.model_name):
            model_path_or_name = self.model_name
        else:
            # For HF models, we'll store them in our model directory instead of the HF cache
            model_path_or_name = self.model_name
            
            # Set environment variables to force models into our directories
            os.environ["HF_HOME"] = self.model_dir
            os.environ["TRANSFORMERS_CACHE"] = os.path.join(self.model_dir, "transformers")
        
        try:
            with torch.cuda.device(0):
                self.model = OrpheusTTS(model_name=model_path_or_name)
            logger.info("Model loaded successfully")
            return self.model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def generate_speech(self, prompt: str, voice: str, emotion: str = "", 
                       temperature: float = 0.7, top_p: float = 0.9, 
                       repetition_penalty: float = 1.1) -> Generator[bytes, None, None]:
        """
        Generate speech from text using the Orpheus TTS model.
        
        Args:
            prompt: The text to synthesize
            voice: The voice to use
            emotion: Optional emotion tag to apply
            temperature: Sampling temperature (higher = more random)
            top_p: Top-p sampling parameter (higher = more diverse)
            repetition_penalty: Repetition penalty (higher = less repetition)
            
        Returns:
            Generator yielding audio chunks as bytes
        """
        if self.model is None:
            self.load_model()
        
        if voice not in self.AVAILABLE_VOICES:
            logger.warning(f"Voice {voice} not recognized, using default voice tara")
            voice = "tara"
            
        if emotion and emotion not in self.AVAILABLE_EMOTIONS:
            logger.warning(f"Emotion {emotion} not recognized, ignoring")
            emotion = ""
            
        # Format prompt with voice and emotion if provided
        formatted_prompt = f"{voice}: {prompt}"
        if emotion:
            formatted_prompt = f"<{emotion}>{formatted_prompt}"
            
        logger.info(f"Generating speech with voice: {voice}, emotion: {emotion or 'none'}")
        
        try:
            # Generate speech and yield audio chunks
            for audio_chunk in self.model.generate_speech(
                prompt=formatted_prompt,
                voice=voice,  # This is redundant since we already formatted the prompt, but keeping for clarity
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            ):
                yield audio_chunk
        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            raise
    
    def generate_speech_to_file(self, prompt: str, voice: str, emotion: str = "", 
                              output_file: str = None, **kwargs) -> str:
        """
        Generate speech from text and save to a file.
        
        Args:
            prompt: The text to synthesize
            voice: The voice to use
            emotion: Optional emotion tag to apply
            output_file: Path to save the generated audio (if None, will create one)
            **kwargs: Additional parameters for speech generation
            
        Returns:
            Path to the generated audio file
        """
        if output_file is None:
            # Create a unique filename in the output directory
            os.makedirs(os.path.join(output_path, "orpheus"), exist_ok=True)
            timestamp = int(time.time())
            voice_str = voice.lower()
            output_file = os.path.join(output_path, "orpheus", f"orpheus_{voice_str}_{timestamp}.wav")
        
        try:
            with wave.open(output_file, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                
                for audio_chunk in self.generate_speech(prompt, voice, emotion, **kwargs):
                    wf.writeframes(audio_chunk)
                    
            logger.info(f"Audio saved to {output_file}")
            return output_file
        except Exception as e:
            logger.error(f"Error saving audio to file: {e}")
            raise
    
    def list_available_voices(self) -> List[str]:
        """Return the list of available voices."""
        return self.AVAILABLE_VOICES
    
    def list_available_emotions(self) -> List[str]:
        """Return the list of available emotions."""
        return self.AVAILABLE_EMOTIONS