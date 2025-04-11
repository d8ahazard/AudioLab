# AudioLab - Project Structure and Documentation

AudioLab is an open-source audio processing application focused on voice-cloning, audio separation, and audio manipulation. This document provides an overview of the project's structure, describes the purpose of key directories and files, and explains the shared classes and design patterns.

## Project Overview

AudioLab is a modular Python application built with FastAPI and Gradio, offering a web-based interface for audio processing. It provides various features including voice cloning, text-to-speech, audio separation, remastering, and more.

## Directory Structure

### Root Directory

- **main.py**: The entry point of the application. Sets up logging, configures the Gradio UI, and starts the FastAPI server.
- **api.py**: Defines the FastAPI application and API endpoints for processing audio files.
- **requirements.txt**: Lists the Python dependencies required for the project.
- **setup.bat/setup.sh**: Setup scripts for Windows and Unix systems to configure the environment.
- **README.md**: Project documentation and user guide.
- **CONTRIBUTING.md**: Guidelines for contributing to the project.
- **LICENSE**: MIT license file.

### Key Directories

#### `/handlers`

Contains handler modules that process specific types of audio operations:

- **noise_removal.py**: Handles noise reduction from audio files.
- **spectrogram.py**: Generates and manipulates spectrograms.
- **autotune.py**: Implements auto-tuning for vocals.
- **tts.py**: Text-to-speech handler.
- **harmony.py**: Handles harmony generation and manipulation.
- **ableton.py**: Exports audio to Ableton format.
- **reaper.py**: Exports audio to Reaper format.
- **download.py**: Handles downloading of resources.
- **args.py**: Manages command-line arguments and UI parameter descriptions.
- **config.py**: Centralized configuration settings.

#### `/wrappers`

Defines wrapper classes that encapsulate audio processing modules for the Process tab:

- **base_wrapper.py**: Base abstract class defining the interface for all wrappers.
- **separate.py**: Handles audio source separation.
- **clone.py**: Voice cloning functionality.
- **remaster.py**: Audio remastering capabilities.
- **super_res.py**: Audio super-resolution enhancement.
- **merge.py**: Merges separate audio tracks.
- **export.py**: Handles exporting to various formats.
- **convert.py**: Audio format conversion.

#### `/modules`

Contains the core functional modules of the application:

- **rvc/**: Retrieval-based Voice Conversion module.
- **separator/**: Audio separation module.
- **yue/**: Music generation module.
- **zonos/**: High-quality TTS module.
- **stable_audio/**: Text-to-audio generation using StabilityAI's Stable Audio model.
- **diffrythm/**: End-to-end full-length song generation using DiffRhythm.

#### `/layouts`

Defines the Gradio UI layouts for different sections of the application:

- **process.py**: UI for the main processing interface.
- **rvc_train.py**: UI for voice model training.
- **music.py**: UI for music generation.
- **tts.py**: UI for text-to-speech.
- **zonos.py**: UI for the Zonos TTS system.
- **stable_audio.py**: UI for text-to-audio generation with StabilityAI's Stable Audio.
- **orpheus.py**: UI for Orpheus TTS functionality.
- **transcribe.py**: UI for audio transcription with speaker diarization.

#### `/util`

Utility classes and functions used across the application:

- **audio_track.py**: Implements the AudioTrack class for handling audio data.
- **data_classes.py**: Defines data structures like ProjectFiles for managing file operations.

#### Other Directories

- **/js**: JavaScript files for the web interface.
- **/css**: CSS stylesheets for the web interface.
- **/models**: Contains machine learning models.
- **/tts_models**: TTS-specific models.
- **/res**: Resource files (images, icons, etc.).
- **/outputs**: Directory where processed output files are stored.
- **/temp_uploads**: Temporary directory for uploaded files.
- **/logs**: Application logs.

## Key Classes and Design Patterns

### Wrapper Pattern

The application uses a wrapper pattern to encapsulate different audio processing modules in the Process tab:

- **BaseWrapper**: An abstract base class that defines the interface for all wrappers. Each wrapper implements:
  - `process_audio()`: Processes audio files and returns the results.
  - `register_api_endpoint()`: Registers FastAPI endpoints for the wrapper.
  - `render_options()`: Renders UI options in Gradio.
  - `validate_args()`: Validates input arguments.

#### Wrapper Structure and Organization

The Process tab uses multiple wrappers that can be chained together to create an audio processing pipeline:

1. **Wrapper Discovery System**:
   - Wrappers are automatically discovered at runtime from the `/wrappers` directory.
   - Each wrapper must inherit from `BaseWrapper` and implement the required methods.
   - Wrappers are sorted and displayed based on their `priority` attribute.

2. **Wrapper Properties**:
   - `title`: User-friendly name displayed in the UI.
   - `priority`: Determines the order of display and processing (lower numbers come first).
   - `default`: Whether the wrapper is enabled by default.
   - `required`: Whether the wrapper must always be included in processing chains.
   - `allowed_kwargs`: Dictionary of parameters that can be passed to the wrapper.

3. **Wrapper Categories (by priority order)**:
   - **Input Processors** (priority 100-199): Prepare and manipulate input files.
      - Separate: Splits audio into components (vocals, instruments, etc.)
   - **Transformation Processors** (priority 200-299): Transform audio content.
      - Clone: Voice cloning and manipulation.
      - Transcribe: Converts speech to text.
      - Remaster: Enhances audio quality.
      - Super Resolution: Improves audio detail.
   - **Combination Processors** (priority 300-399): Combine multiple audio files.
      - Merge: Combines separate audio tracks into one.
   - **Export Processors** (priority 400-499): Final output handlers.
      - Export: Creates final output files.
      - Convert: Changes file format.
      - Compare: Compares original and processed audio.

4. **Parameter Structure**:
   - Each wrapper defines `allowed_kwargs` with parameter metadata:
     - `type`: The parameter data type (str, int, float, bool, etc.).
     - `default`: Default value for the parameter.
     - `min/max`: Range limits for numeric parameters.
     - `choices`: List of allowed values for selection parameters.
     - `render`: Whether to show the parameter in the UI.
     - `required`: Whether the parameter is required.

### Module Pattern

For standalone features with their own tabs, the application uses a module pattern:

- Each feature has a dedicated module in the `/modules` directory.
- Modules don't inherit from BaseWrapper as they aren't part of the Process tab.
- Each module has a corresponding layout file in the `/layouts` directory.

### Layout Structure Pattern

Each layout module in the `/layouts` directory follows a consistent pattern:

1. **Required Functions**:
   - `render(arg_handler: ArgHandler)`: Creates and returns the UI elements.
   - `register_descriptions(arg_handler: ArgHandler)`: Registers tooltips and descriptions for UI elements.
   - `listen()`: Sets up event listeners for inter-tab communication.

2. **Standard UI Structure**:
   - Most tabs use a three-column layout:
     - Left column: Input parameters and options
     - Middle column: Secondary parameters or configuration
     - Right column: Action buttons at the top followed by output displays

3. **Global Elements**:
   - Each layout defines global variables for UI elements that need to be accessed by event listeners.
   - These globals (like `SEND_TO_PROCESS_BUTTON`) are used to connect tabs with each other.

4. **UI Element Requirements**:
   - Every UI element should have:
     - `elem_id` and `key` attributes with a prefix matching the tab name
     - `elem_classes="hintitem"` for tooltip functionality
     - Descriptions registered via `register_descriptions()`

5. **Inter-Tab Communication**:
   - Tabs communicate with each other using the `ArgHandler` class.
   - `arg_handler.get_element("main", "element_name")` retrieves elements from other tabs.
   - Most tabs include a "Send to Process" button to send outputs to the Process tab.

6. **Example Implementation**:
   ```python
   # Global variables for inter-tab communication
   SEND_TO_PROCESS_BUTTON = None
   OUTPUT_ELEMENT = None
   
   def render(arg_handler: ArgHandler):
       global SEND_TO_PROCESS_BUTTON, OUTPUT_ELEMENT
       
       # Three-column layout
       with gr.Row():
           with gr.Column():  # Left column - Inputs
               # UI elements for input
           
           with gr.Column():  # Middle column - Parameters
               # UI elements for parameters
           
           with gr.Column():  # Right column - Actions & Output
               # Action buttons at the top
               with gr.Group():
                   with gr.Row():
                       generate_btn = gr.Button("Generate", variant="primary")
                       SEND_TO_PROCESS_BUTTON = gr.Button("Send to Process")
               
               # Output displays
               OUTPUT_ELEMENT = gr.Audio(label="Output")
   
   def register_descriptions(arg_handler: ArgHandler):
       descriptions = {
           "element_id": "Description for this element",
       }
       for elem_id, description in descriptions.items():
           arg_handler.register_description("tab_prefix", elem_id, description)
   
   def listen():
       process_inputs = arg_handler.get_element("main", "process_inputs")
       if process_inputs and SEND_TO_PROCESS_BUTTON:
           SEND_TO_PROCESS_BUTTON.click(
               fn=send_to_process,
               inputs=[OUTPUT_ELEMENT, process_inputs],
               outputs=[process_inputs]
           )
   ```

### Data Structure

- **ProjectFiles**: Manages files associated with a processing project:
  - Keeps track of input/output files.
  - Organizes files in a structured directory hierarchy.
  - Handles file hashing and organization.

### UI Framework

The application uses Gradio to create a web-based UI:
- Uses a tabbed interface for different functionalities (Process, Train, Music, TTS, Zonos, Sound Forge).
- Each tab is defined in a separate layout file.
- ArgHandler manages descriptions and documentation for UI elements.

### API Structure

The application provides a modular REST API built with FastAPI:

1. **Modular API Design**:
   - Each layout module registers its own API endpoints through a `register_api_endpoints` function.
   - The main `api.py` automatically discovers and registers all endpoints from each module.
   - Each module is responsible for implementing and documenting its own endpoints.

2. **Core API Endpoints**:
   - **/api/v1/process/{processor}**: Endpoints for individual processing wrappers.
   - **/api/v1/process/multi**: Chain multiple processors together in a sequence.
   - **/api/v1/transcribe**: Transcribe audio files with speaker diarization.
   - **/api/v1/tts/generate**: Generate speech from text with various models.
   - **/api/v1/tts/models**: List available TTS models.
   - **/api/v1/music/generate**: Generate music with lyrics and genre prompts.
   - **/api/v1/orpheus/generate**: Generate speech using Orpheus TTS.
   - **/api/v1/orpheus/finetune**: Finetune Orpheus on custom voices.
   - **/api/v1/stable-audio/generate**: Generate audio from text using Stable Audio.
   - **/api/v1/rvc/train**: Train RVC voice models.
   - **/api/v1/rvc/models**: List available voice models.
   - **/api/v1/diffrythm/generate**: Generate full-length songs with DiffRhythm.

3. **API Documentation**:
   - Interactive documentation available at **/docs** using Swagger UI.
   - Alternative documentation at **/redoc** for a different presentation.
   - Programmatic schema access at **/openapi.json**.

4. **Parameter Validation**:
   - Uses Pydantic models for request validation.
   - Properly documents all parameters with types and constraints.
   - Returns helpful error messages for invalid requests.

5. **Asynchronous Processing**:
   - Long-running tasks (training, fine-tuning) run in the background.
   - Job status endpoints for checking progress of background tasks.
   - File download endpoints for retrieving generated content.

## Processing Flow

### Process Tab Workflow

The Process tab implements a flexible audio processing pipeline:

1. **Input Selection**:
   - Users upload audio files through the UI or provide URLs.
   - Files are saved to a temporary directory or loaded from existing projects.
   - Input files are displayed with preview capabilities for both audio and images.

2. **Processor Selection and Configuration**:
   - Users select which processors to apply from the checkbox list.
   - Required processors are automatically included and cannot be removed.
   - Each selected processor reveals its settings accordion when applicable.
   - Users configure each processor's parameters through the UI components.

3. **Processing Chain Execution**:
   - When the user clicks "Start Processing", the system:
     1. Creates a `ProjectFiles` object for each input file to track transformations.
     2. Processes inputs sequentially through each selected wrapper in priority order.
     3. Each wrapper receives the output of the previous wrapper as its input.
     4. Special parameters like pitch shift are passed between related processors (Clone → Merge).
     5. Progress updates are displayed during processing.

4. **Output Handling**:
   - Processed files are saved to the outputs directory with organized structure.
   - Output files are displayed with preview capabilities.
   - Users can select specific outputs for preview using the dropdown.
   - Output files can be downloaded or sent to other tabs for further processing.

5. **Error Handling**:
   - If a processor encounters an error, the chain stops at that point.
   - Error messages are logged and displayed to the user.
   - The system keeps all outputs generated before the error.

### Path Management and File Organization

The Process tab implements a sophisticated path management system:

1. **Path Mapping**:
   - Maintains global dictionaries mapping full paths to filenames and vice versa.
   - Handles duplicate filenames by appending a numeric suffix.
   - Uses the mapping to show user-friendly filenames in the UI while keeping track of full paths.

2. **Project Structure**:
   - Creates dedicated project folders in the outputs directory.
   - Organizes files in a structured hierarchy with source and output subdirectories.
   - Allows loading existing projects to continue previous work.

3. **File Type Handling**:
   - Automatically detects audio files (wav, mp3, flac) and image files.
   - Shows appropriate previews based on file type.
   - Tracks file relationships through the processing chain using `ProjectFiles`.

### Inter-Process Communication

Special considerations for processor communication:

1. **Parameter Sharing**:
   - Some processors need to share information (e.g., Clone → Merge for pitch shift).
   - The application keeps track of certain parameters and passes them between processors.

2. **Processor Dependencies**:
   - Some processors function correctly only when preceded by specific processors.
   - The UI enforces required processors and preferred ordering through the priority system.

## Extensibility

The project is designed to be extensible:
1. New processing modules can be added by creating new wrapper classes in `/wrappers`.
   - To add a new wrapper:
     - Create a new Python file in the `/wrappers` directory.
     - Define a class that inherits from `BaseWrapper`.
     - Implement required methods: `process_audio()`, `register_api_endpoint()`, `render_options()`.
     - Set appropriate `title`, `priority`, `default`, and `required` properties.
     - Define `allowed_kwargs` with properly documented parameters.
   - The wrapper will be automatically discovered and added to the Process tab.
   - Consider priority values to ensure your wrapper appears in the appropriate position.

2. New standalone features can be added by creating modules in `/modules` and layouts in `/layouts`.

### Adding New Features

To add a new standalone feature with its own tab:

1. **Create Module**:
   - Create a new directory in `/modules/your_feature/`
   - Implement the core functionality in the module
   - Add an `__init__.py` file to export the main class

2. **Create Layout**:
   - Create a new file in `/layouts/your_feature.py`
   - Implement the required functions: `render()`, `register_descriptions()`, and `listen()`
   - Follow the three-column layout pattern
   - Add proper element IDs and classes for tooltip support

3. **Update main.py**:
   - Import the layout functions
   - Register descriptions with the ArgHandler
   - Add a new tab in the UI
   - Set up event listeners

## Dependencies

Key dependencies include:
- FastAPI: For the REST API.
- Gradio: For the web UI.
- PyTorch/TorchAudio: For audio processing and ML models.
- Diffusers: For AI models like Stable Audio.
- Various audio processing libraries as indicated in requirements.txt.

## Configuration

The application is configured through:
- Command-line arguments (specified in main.py).
- Configuration settings in handlers/config.py.
- Environment variables for certain models and caching.

## Feature Highlights

### Sound Forge (StabilityAI's Stable Audio)

The Sound Forge tab provides text-to-audio generation capabilities using StabilityAI's Stable Audio model:

- Generate high-quality sound effects and ambient audio from text descriptions.
- Customize generation parameters like duration, inference steps, and guidance scale.
- Create multiple variations of the same prompt.
- Control generation with negative prompts to avoid unwanted characteristics.
- Generate audio up to 47 seconds in length.

### Transcribe

The Transcribe tab provides advanced audio transcription capabilities using WhisperX:

- Convert speech to text with high accuracy across multiple languages.
- Automatically identify and label different speakers in audio (speaker diarization).
- Create precise word-level timestamps for perfect alignment with audio.
- Support batch processing of multiple audio files.
- Generate both JSON (with detailed metadata) and human-readable text outputs.
- Control transcription parameters like language detection, alignment, and speaker assignment.

### DiffRhythm

The DiffRhythm module provides end-to-end full-length song generation capabilities:

- Generate full-length songs (up to 4m45s) from text prompts and style references.
- Support for generating music with lyrics using timestamped LRC files.
- Create songs in various styles using style prompts or reference audio.
- High-quality stereo audio generation at 44.1kHz.
- Blazingly fast generation compared to other music generation models.
- Chunked decoding option to optimize memory usage on consumer GPUs.
- Support for both text-to-music and pure music generation.
- Memory-efficient implementation based on the latent diffusion architecture.

The DiffRhythm module provides a dedicated UI interface with the following features:

- **Generation Interface**: A user-friendly interface for song generation with:
  - Model selection between base (95s) and full (285s) models
  - Chunked decoding option for optimized memory usage
  - Style prompt input for text-based style control
  - Reference audio upload for audio-based style control
  - LRC format lyrics entry with timestamp support
  - One-click generation with progress tracking
  - Audio preview and download capabilities
  - Send to Process tab for further audio processing

- **Training Information**: Comprehensive documentation about model training with:
  - Dataset preparation requirements and formats
  - Training resource recommendations
  - Future feature roadmap
  
- **API Integration**: Complete REST API with endpoint `/api/v1/diffrythm/generate` supporting:
  - Text prompt style descriptions
  - LRC format lyrics input
  - Audio reference file uploads
  - Model selection
  - Memory optimization options

The module seamlessly integrates with the rest of the AudioLab ecosystem, allowing generated songs to be further processed with other tools like audio separation, voice cloning, and remastering. 