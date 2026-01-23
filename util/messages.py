"""
AudioLab User-Friendly Progress Messages
=========================================

Centralized message definitions for consistent, user-friendly progress updates
across all processing operations. These replace technical messages with
clear, helpful descriptions of what's happening.
"""

from typing import Dict


# =============================================================================
# Audio Separation Messages
# =============================================================================

SEPARATION_MESSAGES = {
    # Stage messages
    "init": "Preparing audio separation...",
    "loading_models": "Loading AI models...",
    "analyzing": "Analyzing audio characteristics...",
    
    # Ensemble processing
    "ensemble_start": "Starting multi-model separation for best quality...",
    "ensemble_model": "Separating with model {model_num} of {total_models}...",
    "ensemble_blending": "Blending results from all models...",
    "ensemble_complete": "Multi-model separation complete",
    
    # Vocal processing
    "vocals_extracting": "Extracting vocals from audio...",
    "vocals_complete": "Vocals successfully isolated",
    "bg_vocals_start": "Separating background vocals...",
    "bg_vocals_complete": "Background vocals isolated",
    
    # Instrumental processing
    "instrumental_extracting": "Extracting instrumental track...",
    "instrumental_complete": "Instrumental track ready",
    
    # Multi-stem
    "multistem_start": "Separating into individual instrument tracks...",
    "drums_extracting": "Isolating drums...",
    "drums_detailed": "Separating drum components (kick, snare, hi-hat)...",
    "bass_extracting": "Isolating bass...",
    "guitar_extracting": "Isolating guitar...",
    "piano_extracting": "Isolating piano/keys...",
    "woodwinds_extracting": "Isolating woodwind instruments...",
    "other_extracting": "Isolating other instruments...",
    "multistem_complete": "All stems separated successfully",
    
    # Transformations
    "transform_start": "Applying audio enhancements...",
    "reverb_removing": "Removing reverb from {target}...",
    "echo_removing": "Removing echo/delay from {target}...",
    "noise_removing": "Removing background noise from {target}...",
    "crowd_removing": "Removing crowd noise from {target}...",
    "transform_complete": "Audio enhancements applied",
    
    # Saving
    "saving_stems": "Saving separated audio files...",
    "saving_complete": "All files saved successfully",
    
    # Errors
    "error_model_load": "Error loading separation model",
    "error_processing": "Error during audio processing",
    "error_saving": "Error saving output files",
}


# =============================================================================
# Voice Cloning Messages
# =============================================================================

CLONE_MESSAGES = {
    "init": "Preparing voice conversion...",
    "loading_model": "Loading voice model '{model_name}'...",
    "extracting_features": "Analyzing vocal characteristics...",
    "converting": "Converting voice...",
    "pitch_shifting": "Adjusting pitch by {semitones} semitones...",
    "blending": "Blending converted vocals...",
    "complete": "Voice conversion complete",
    "error": "Voice conversion failed",
}


# =============================================================================
# TTS Messages
# =============================================================================

TTS_MESSAGES = {
    "init": "Preparing text-to-speech...",
    "loading_model": "Loading speech synthesis model...",
    "processing_text": "Processing input text...",
    "generating": "Generating speech...",
    "generating_chunk": "Generating audio chunk {current} of {total}...",
    "applying_effects": "Applying audio effects...",
    "complete": "Speech generated successfully",
    "error": "Speech generation failed",
}


# =============================================================================
# Music Generation Messages
# =============================================================================

MUSIC_MESSAGES = {
    "init": "Preparing music generation...",
    "loading_model": "Loading music model...",
    "processing_lyrics": "Processing lyrics...",
    "generating": "Generating music...",
    "generating_section": "Generating section {current} of {total}...",
    "finalizing": "Finalizing audio...",
    "complete": "Music generated successfully",
    "error": "Music generation failed",
}


# =============================================================================
# Training Messages
# =============================================================================

TRAINING_MESSAGES = {
    "init": "Preparing training environment...",
    "loading_data": "Loading training data...",
    "preprocessing": "Preprocessing audio files ({current}/{total})...",
    "extracting_features": "Extracting audio features...",
    "training_start": "Starting model training...",
    "training_epoch": "Training epoch {epoch}/{total_epochs}...",
    "training_step": "Training step {step}/{total_steps} (Loss: {loss:.4f})...",
    "validating": "Running validation...",
    "saving_checkpoint": "Saving checkpoint...",
    "training_complete": "Training completed successfully",
    "error": "Training error occurred",
}


# =============================================================================
# Transcription Messages
# =============================================================================

TRANSCRIBE_MESSAGES = {
    "init": "Preparing transcription...",
    "loading_model": "Loading speech recognition model...",
    "transcribing": "Transcribing audio...",
    "transcribing_file": "Transcribing file {current} of {total}...",
    "aligning": "Aligning words with timestamps...",
    "diarizing": "Identifying speakers...",
    "formatting": "Formatting output...",
    "complete": "Transcription complete",
    "error": "Transcription failed",
}


# =============================================================================
# General Processing Messages
# =============================================================================

PROCESS_MESSAGES = {
    "starting": "Starting processing pipeline...",
    "processor_start": "Running {processor_name}...",
    "processor_complete": "{processor_name} completed",
    "pipeline_complete": "All processing completed in {time:.1f}s",
    "loading_files": "Loading input files...",
    "saving_files": "Saving output files...",
    "error": "Processing error: {error}",
}


# =============================================================================
# Utility Functions
# =============================================================================

def get_message(category: str, key: str, **kwargs) -> str:
    """
    Get a user-friendly message with optional formatting.
    
    Args:
        category: Message category (e.g., 'separation', 'clone', 'tts')
        key: Message key within the category
        **kwargs: Format arguments for the message
    
    Returns:
        Formatted message string
    
    Example:
        msg = get_message('separation', 'ensemble_model', model_num=2, total_models=5)
        # Returns: "Separating with model 2 of 5..."
    """
    categories = {
        "separation": SEPARATION_MESSAGES,
        "clone": CLONE_MESSAGES,
        "tts": TTS_MESSAGES,
        "music": MUSIC_MESSAGES,
        "training": TRAINING_MESSAGES,
        "transcribe": TRANSCRIBE_MESSAGES,
        "process": PROCESS_MESSAGES,
    }
    
    messages = categories.get(category, {})
    template = messages.get(key, f"Processing ({key})...")
    
    try:
        return template.format(**kwargs)
    except KeyError:
        return template


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def format_file_size(bytes_size: int) -> str:
    """Format bytes into human-readable file size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f} TB"
