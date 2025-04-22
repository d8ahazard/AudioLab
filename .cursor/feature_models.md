# AudioLab Feature and Model Summary

This document provides an overview of the key features, AI models, and technologies used in each module of the AudioLab project.

## Process Tab Wrappers

### Separate (`wrappers/separate.py`)
- **Purpose**: Split audio into components (vocals, instruments, drums, bass)
- **Models**: Uses pre-trained source separation models
- **Key Parameters**:
  - Model selection (UVR, MDX, Demucs)
  - Stem options (vocals, instruments, bass, drums)
  - Processing options (overlap, shifts, GPU acceleration)

### Clone (`wrappers/clone.py`)
- **Purpose**: Voice cloning and manipulation
- **Models**: Uses RVC (Retrieval-based Voice Conversion)
- **Key Parameters**:
  - Voice model selection
  - Pitch shifting
  - Protection settings
  - Voice characteristics (clarity, similarity)

### Remaster (`wrappers/remaster.py`)
- **Purpose**: Enhance audio quality
- **Technologies**: Audio processing algorithms for normalization, EQ, compression
- **Key Parameters**:
  - Enhancement level
  - Processing mode
  - Target loudness

### Super-Res (`wrappers/super_res.py`)
- **Purpose**: Audio super-resolution to improve detail
- **Models**: Neural networks for audio enhancement
- **Key Parameters**:
  - Resolution factor
  - Model selection
  - Noise reduction

### Merge (`wrappers/merge.py`)
- **Purpose**: Combine separate audio tracks
- **Technologies**: Audio mixing algorithms
- **Key Parameters**:
  - Track volumes
  - Pan settings
  - Timing adjustments

### Export/Convert (`wrappers/export.py`, `wrappers/convert.py`)
- **Purpose**: Create final output files and change formats
- **Technologies**: Audio encoding libraries
- **Key Parameters**:
  - Output format
  - Quality settings
  - Metadata

## Standalone Feature Modules

### DiffRhythm (`modules/diffrythm/`)
- **Purpose**: End-to-end full-length song generation
- **Models**: Latent diffusion-based music generation models
- **Key Features**:
  - Generates songs up to 4:45 in length
  - Supports lyrics via LRC files
  - Style control via text or audio reference
  - Two model options: base (95s) and full (285s)

### Orpheus TTS (`modules/orpheus/`)
- **Purpose**: High-quality text-to-speech
- **Models**: LLM-powered speech synthesis
- **Key Features**:
  - Natural intonation and prosody
  - Voice cloning with minimal data
  - Multiple voice styles per model
  - Emotion and speaking style control

### Stable Audio (`modules/stable_audio/`)
- **Purpose**: Text-to-audio generation
- **Models**: StabilityAI's Stable Audio model
- **Key Features**:
  - Generate sound effects and ambient audio
  - Control generation with customizable parameters
  - Create multiple variations
  - Audio up to 47 seconds in length

### WaveTransfer (`modules/wavetransfer/`)
- **Purpose**: Multi-instrument timbre transfer
- **Models**: Diffusion models for audio transformation
- **Key Features**:
  - Transform sound characteristics between instruments
  - Two-step training process (main model + schedule network)
  - Chunked processing for longer files
  - Project-based workflow

### Transcribe (`layouts/transcribe.py`)
- **Purpose**: Audio transcription with speaker diarization
- **Models**: WhisperX for speech recognition
- **Key Features**:
  - High-accuracy transcription
  - Speaker identification
  - Word-level timestamps
  - Multiple output formats (JSON, text)

## Model Dependencies and Requirements

### RVC
- **Dependencies**: PyTorch, TorchAudio, CUDA
- **Requirements**: ~4GB VRAM for inference, 8GB+ for training
- **Models Location**: `/models/rvc/`

### DiffRhythm
- **Dependencies**: PyTorch, Diffusers, Accelerate
- **Requirements**: 8GB VRAM for base model, 16GB+ for full model
- **Models Location**: `/models/diffrythm/`

### Stable Audio
- **Dependencies**: Diffusers, StabilityAI models
- **Requirements**: 8GB+ VRAM
- **Models Location**: `/models/stable_audio/`

### WaveTransfer
- **Dependencies**: PyTorch, TorchAudio, Diffusers
- **Requirements**: 8GB+ VRAM, more for longer files
- **Models Location**: `/models/wavetransfer/`

### Orpheus
- **Dependencies**: PyTorch, HuggingFace Transformers
- **Requirements**: 8GB+ VRAM, depends on model size
- **Models Location**: `/models/orpheus/`

### Audio Separation
- **Dependencies**: PyTorch, TorchAudio
- **Requirements**: Varies by model (4-8GB VRAM)
- **Models Location**: `/models/separator/` 