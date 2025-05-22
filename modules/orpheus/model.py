"""
Orpheus TTS model implementation for AudioLab.
"""

import logging
import os
import time
import wave
from pathlib import Path
from typing import Optional, List, Generator, Union

import torch
import numpy as np

# Try to import OrpheusTTSModel, but provide a fallback if not installed
try:
    from orpheus_tts import OrpheusModel as OrpheusTTSModel
    ORPHEUS_TTS_AVAILABLE = True
except ImportError:
    ORPHEUS_TTS_AVAILABLE = False
    
from handlers.config import output_path

logger = logging.getLogger("ADLB.Orpheus.Model")

# Available voices and emotion tags
AVAILABLE_VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
EMOTION_TAGS = {
    "laugh": "<laugh>",
    "chuckle": "<chuckle>",
    "sigh": "<sigh>",
    "cough": "<cough>",
    "sniffle": "<sniffle>",
    "groan": "<groan>",
    "yawn": "<yawn>",
    "gasp": "<gasp>"
}

class OrpheusModel:
    """
    Wrapper for the Orpheus TTS model.
    """
    
    def __init__(self, model_name: str = "canopylabs/orpheus-tts-0.1-finetune-prod"):
        """
        Initialize the Orpheus TTS model.
        
        Args:
            model_name: The name or path of the model to use
        """
        if not ORPHEUS_TTS_AVAILABLE:
            logger.error("orpheus_tts package is not installed. Please install it with 'pip install orpheus-tts'")
            raise ImportError("orpheus_tts package is not installed. Please install it with 'pip install orpheus-tts'")
            
        self.model_name = model_name
        self.model = None
        self.output_dir = os.path.join(output_path, "orpheus")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize model lazily to avoid loading it until needed
        
    def load_model(self):
        """Load the model if not already loaded"""
        if self.model is None:
            logger.info(f"Loading Orpheus TTS model: {self.model_name}")
            try:
                self.model = OrpheusTTSModel(
                    model_name=self.model_name
                )
                logger.info("Orpheus TTS model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading Orpheus TTS model: {e}")
                raise
    
    def generate_speech(
        self,
        prompt: str,
        voice: str = "tara",
        emotion: str = "",
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1
    ) -> Generator[bytes, None, None]:
        """
        Generate speech from text, returning audio chunks.
        
        Args:
            prompt: The text to convert to speech
            voice: The voice to use
            emotion: Optional emotion tag to apply
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            repetition_penalty: Repetition penalty parameter
            
        Returns:
            Generator yielding audio chunks
        """
        self.load_model()
        
        # Apply emotion tag if provided
        if emotion and emotion in EMOTION_TAGS:
            # Simple approach: add emotion tag at the beginning of the prompt
            prompt = f"{EMOTION_TAGS[emotion]} {prompt}"
        
        return self.model.generate_speech(
            prompt=prompt,
            voice=voice,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )
    
    def generate_speech_to_file(
        self,
        prompt: str,
        voice: str = "tara",
        emotion: str = "",
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        output_file: Optional[str] = None
    ) -> str:
        """
        Generate speech from text and save to a WAV file.
        
        Args:
            prompt: The text to convert to speech
            voice: The voice to use
            emotion: Optional emotion tag to apply
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            repetition_penalty: Repetition penalty parameter
            output_file: Optional path to save the output file
            
        Returns:
            Path to the generated audio file
        """
        self.load_model()
        
        # Generate a filename if not provided
        if output_file is None:
            timestamp = int(time.time())
            output_file = os.path.join(self.output_dir, f"orpheus_{voice}_{timestamp}.wav")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Generate speech
        syn_tokens = self.generate_speech(
            prompt=prompt,
            voice=voice,
            emotion=emotion,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )
        
        # Save to WAV file
        with wave.open(output_file, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            
            for audio_chunk in syn_tokens:
                wf.writeframes(audio_chunk)
        
        logger.info(f"Speech generated and saved to {output_file}")
        return output_file
    
    def apply_emotion_tag(self, text: str, emotion: str) -> str:
        """
        Apply an emotion tag to text.
        
        Args:
            text: Input text
            emotion: Emotion to apply
            
        Returns:
            Text with emotion tag
        """
        if emotion and emotion in EMOTION_TAGS:
            return f"{EMOTION_TAGS[emotion]} {text}"
        return text 