# AudioLab Project Structure

This document provides a high-level overview of the AudioLab project structure to help AI agents navigate and understand the codebase.

## Key Directories

### Core Application
- **main.py**: Application entry point that configures Gradio UI and FastAPI server
- **api.py**: Defines FastAPI application and API endpoints
- **requirements.txt**: Python dependencies

### Organizational Directories
- **/layouts/**: UI layouts for different application tabs (Gradio interfaces)
- **/wrappers/**: Processing modules for the Process tab following the Wrapper pattern
- **/modules/**: Core functional modules of the application
- **/handlers/**: Handler modules for specific audio operations
- **/util/**: Utility classes and functions used across the application

### Resources and Assets
- **/models/**: Storage for machine learning models
- **/res/**: Resource files (images, icons, etc.)
- **/css/**: CSS stylesheets for the web interface
- **/js/**: JavaScript files for the web interface

### Runtime and User Data
- **/outputs/**: Directory for processed output files
- **/temp_uploads/**: Temporary directory for uploaded files
- **/venv/**: Python virtual environment

## Key Components and Patterns

### Wrapper Pattern
Audio processing modules in the Process tab follow the Wrapper pattern:
- All inherit from `BaseWrapper` in `wrappers/base_wrapper.py`
- Sorted by priority (100-499) for processing order:
  - Input (100-199): Separate, etc.
  - Transformation (200-299): Clone, Remaster, etc.
  - Combination (300-399): Merge, etc.
  - Export (400-499): Export, Convert, etc.

### Module Pattern
Standalone features with dedicated tabs:
- Implemented in `/modules/{feature_name}/`
- UI defined in `/layouts/{feature_name}.py`
- Each has `render()`, `register_descriptions()`, and `listen()` functions

### Data Classes
- **AudioTrack** (`util/audio_track.py`): Core class for audio data handling
- **ProjectFiles** (`util/data_classes.py`): Manages files for processing projects

### API Structure
- Each module registers its own API endpoints
- Follows RESTful design with endpoints like `/api/v1/{feature}/...` 