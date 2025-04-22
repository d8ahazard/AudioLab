# AudioLab Module Dependencies

This document outlines key dependencies and relationships between modules in the AudioLab project to help AI agents understand how components interact.

## Core Module Relationships

### Process Tab Workflow
1. **Input files** → **Separate** (splits audio components)
2. **Separate outputs** → **Clone/Remaster/Super-Res** (transformation)
3. **Transformation outputs** → **Merge** (combines tracks)
4. **Merged outputs** → **Export/Convert** (final outputs)

### Inter-Process Communication
- **Clone** → **Merge**: Share pitch shift parameters
- **Separate** → **Clone**: Provides vocal tracks for voice cloning
- **TTS/Orpheus** → **Process**: Send generated audio to Process tab

### UI Dependencies
- Each layout module depends on the ArgHandler from `handlers/args.py`
- Tabs communicate using ArgHandler to retrieve elements from other tabs
- Most tabs include a "Send to Process" button to send outputs to Process tab

## Critical Dependencies

### Core Dependencies
- **BaseWrapper** (`wrappers/base_wrapper.py`): All wrappers depend on this class
- **AudioTrack** (`util/audio_track.py`): Core audio data handling used across most modules
- **ProjectFiles** (`util/data_classes.py`): Manages file operations across processing chain
- **ArgHandler** (`handlers/args.py`): Manages UI descriptions and inter-tab communication

### Feature Dependencies
- **RVC** depends on PyTorch and specific CUDA versions
- **Stable Audio** depends on Diffusers library
- **Separator** depends on specific pre-trained models
- **DiffRhythm** requires high VRAM GPU for full-length song generation
- **WaveTransfer** depends on diffusion models and chunked processing

## Extension Points

### Adding New Wrappers
New processing modules can be added by creating wrapper classes in `/wrappers/` that:
- Inherit from `BaseWrapper`
- Implement required methods: `process_audio()`, `register_api_endpoint()`, `render_options()`, `validate_args()`
- Set appropriate `title`, `priority`, `default`, and `required` properties
- Define `allowed_kwargs` with parameter metadata

### Adding New Features
New standalone features can be added by:
1. Creating a module in `/modules/your_feature/`
2. Creating a layout in `/layouts/your_feature.py`
3. Updating `main.py` to include the new tab
4. Adding API endpoints in the module's API registration function

## Configuration Dependencies
- Command-line arguments in `main.py`
- Configuration settings in `handlers/config.py`
- Environment variables for model paths and caching 