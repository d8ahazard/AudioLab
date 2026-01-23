# Changelog

All notable changes to AudioLab will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-01-23

### Added

#### Dependency Management
- Modern `pyproject.toml` configuration with proper metadata and optional dependency groups
- Split requirements files for better modularity:
  - `requirements-core.txt` - Core dependencies
  - `requirements-cuda.txt` - GPU/CUDA packages
  - `requirements-wheels.txt` - Custom wheel packages
  - `requirements-dev.txt` - Development tools
- Added `rich` library for enhanced terminal output
- Added `ruff` and `mypy` for code quality

#### Installation
- New unified `install.py` installer with:
  - Automatic OS detection (Windows/Linux/macOS)
  - GPU capability detection (CUDA version, VRAM)
  - Dependency group selection
  - espeak-ng auto-installation
  - Installation verification
- Docker support with `Dockerfile` and `docker-compose.yml`
- CPU-only Docker image (`Dockerfile.cpu`)
- `.dockerignore` for optimized builds

#### Progress Reporting
- New `ProgressReporter` class with:
  - WebSocket broadcasting for real-time updates
  - ETA calculation with exponential moving average
  - Sub-task progress tracking
  - Cancellation support
  - Thread-safe operations
- New `WebSocketProgressManager` for client connections
- User-friendly progress message templates in `util/messages.py`

#### Separation Models
- V3 separation profile with latest 2025 models
- Use-case specific presets: Karaoke, Remix, Podcast
- Improved SDR-based automatic weight calculation

#### Documentation
- Completely rewritten README with professional tone
- Comprehensive CHANGELOG (this file)
- Improved inline code documentation

### Changed

#### Dependencies
- Updated `wandb` from 0.15.4 to >=0.17.2
- Updated `protobuf` from 3.20.0 to >=4.25.0
- Updated `pytorch_lightning` to >=2.5.1
- Updated `transformers` to >=4.51.3
- Updated `gradio` to >=5.29.0
- Consolidated duplicate `py3langid` entries

#### Progress Messages
- Replaced technical progress messages with user-friendly descriptions
- Standardized message format across all processing operations
- Added context-aware messages for different processing stages

#### Code Quality
- Added type hints to core utility modules
- Standardized logging format
- Improved error messages with troubleshooting hints

### Removed

#### Deprecated Dependencies
- Removed `cog` - Unused Replicate deployment tool
- Removed `metrics` - Vague unused package
- Removed `progressbar` - Replaced by `tqdm`
- Removed `wave` - Unnecessary (standard library)
- Removed `dataset` - Duplicate of `datasets`

#### Code Cleanup
- Removed debug functions from `stem_separator.py`
- Removed unused imports across modules
- Removed redundant code blocks

### Fixed

- Fixed CUDA version inconsistencies (standardized on 12.4)
- Fixed duplicate package installations in setup scripts
- Fixed progress tracking accuracy in ensemble separation
- Improved error handling in voice cloning pipeline

---

## [1.5.0] - 2025-12-15

### Added

- RVC V3 support with improved voice quality
- ACE-Step music generation integration
- WaveTransfer timbre transfer module
- DIA TTS engine for realistic dialogue
- Chatterbox TTS integration
- Real-time audio alignment (RTLA) module
- Video input support with automatic audio extraction

### Changed

- Improved separation quality with updated BS-RoFormer model
- Enhanced RVC training with better loss tracking
- Updated Gradio to 5.x with new components
- Improved API documentation with Swagger UI

### Fixed

- Memory leaks in long-running separation operations
- Race conditions in multi-file processing
- Audio format compatibility issues

---

## [1.4.0] - 2025-09-01

### Added

- Zonos TTS with emotional expression control
- YuE music generation with lyrics support
- Stable Audio text-to-audio generation
- Background vocal separation
- Advanced drum separation (kick, snare, hi-hat, cymbals)
- Woodwind instrument separation
- Impulse response extraction for reverb

### Changed

- Refactored separation pipeline for better performance
- Improved caching system for processed files
- Updated model weights for better quality

---

## [1.3.0] - 2025-06-01

### Added

- WhisperX transcription with speaker diarization
- DAW export (Ableton Live and Reaper)
- Audio remastering with reference matching
- Super-resolution audio enhancement

### Changed

- Improved RVC training interface
- Better progress tracking in UI
- Updated separation models

---

## [1.2.0] - 2025-03-01

### Added

- Multi-file batch processing
- Project management system
- URL download support (YouTube, etc.)
- API endpoints for all major features

### Changed

- Refactored to modular wrapper architecture
- Improved error handling and recovery
- Better memory management for large files

---

## [1.1.0] - 2024-12-01

### Added

- Coqui TTS integration
- Voice cloning with OpenVoice
- Noise removal post-processing
- Echo/reverb removal

### Changed

- Improved separation quality
- Better UI organization
- Faster model loading

---

## [1.0.0] - 2024-09-01

### Added

- Initial release
- Audio separation (vocals, instrumental)
- RVC voice cloning
- Basic TTS functionality
- Web UI with Gradio
- REST API with FastAPI

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| 2.0.0 | 2026-01-23 | Major overhaul: dependencies, Docker, progress system |
| 1.5.0 | 2025-12-15 | RVC V3, ACE-Step, DIA TTS, video support |
| 1.4.0 | 2025-09-01 | Zonos TTS, YuE music, advanced separation |
| 1.3.0 | 2025-06-01 | Transcription, DAW export, remastering |
| 1.2.0 | 2025-03-01 | Batch processing, API, project management |
| 1.1.0 | 2024-12-01 | Coqui TTS, noise removal |
| 1.0.0 | 2024-09-01 | Initial release |
