from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, create_model
from typing import List, Dict, Any, Optional
import tempfile
import os
import shutil
from pathlib import Path
import importlib
import inspect
import logging
from util.data_classes import ProjectFiles
from wrappers.base_wrapper import BaseWrapper, TypedInput

# Initialize logger
logger = logging.getLogger(__name__)

# Add direct import for the transcribe functionality
import layouts.transcribe as transcribe_module
import layouts.music as music_module
import layouts.orpheus as orpheus_module
import layouts.process as process_module
import layouts.rvc_train as rvc_train_module
import layouts.stable_audio as stable_audio_module
import layouts.tts as tts_module
import layouts.diffrythm as diffrythm_module

app = FastAPI(
    title="AudioLab API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    description="""
    # AudioLab API

    AudioLab is an open-source audio processing application focused on voice-cloning, audio separation, and audio manipulation.
    This API provides access to all the core functionalities of AudioLab.

    ## Key Features

    - **Audio Separation**: Split audio into vocals, instruments, and other components
    - **Voice Cloning**: Clone vocals using RVC voice models
    - **Text-to-Speech**: Generate speech from text using various TTS models including Zonos and Orpheus
    - **Audio Remastering**: Clean up and enhance audio quality
    - **Audio Super-Resolution**: Improve audio detail and quality
    - **Audio Transcription**: Transcribe speech to text with speaker diarization
    - **Music Generation**: Generate music from text prompts and style parameters
    - **Full-Length Song Generation**: Generate complete songs with DiffRhythm
    - **Format Conversion**: Convert between audio formats with customizable settings

    ## API Structure

    The API is organized into several categories:
    
    - `/api/v1/process/{processor}`: Individual processor endpoints
    - `/api/v1/process/multi`: Process audio through multiple processors in sequence
    - `/api/v1/tts/*`: Text-to-speech endpoints
    - `/api/v1/transcribe`: Transcription endpoint
    - `/api/v1/music/*`: Music generation endpoints
    - `/api/v1/orpheus/*`: Orpheus TTS endpoints
    - `/api/v1/stable-audio/*`: Stable Audio endpoints
    - `/api/v1/rvc/*`: RVC voice model endpoints
    - `/api/v1/diffrythm/*`: DiffRhythm song generation endpoints
    
    For each processor endpoint, you can get detailed documentation on available parameters 
    from the `/api/v1/docs` endpoint.
    """,
    terms_of_service="https://example.com/terms/",
    contact={
        "name": "AudioLab Support",
        "url": "https://github.com/d8ahazard/audiolab/issues",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# Create a temporary directory for file uploads
TEMP_DIR = Path("temp_uploads")
TEMP_DIR.mkdir(exist_ok=True)

def create_pydantic_model_from_wrapper(wrapper: BaseWrapper) -> BaseModel:
    """Create a Pydantic model from a wrapper's allowed_kwargs"""
    fields = {}
    for key, value in wrapper.allowed_kwargs.items():
        field_type = value.type
        if value.field.default == ...:
            field_type = Optional[field_type]
        fields[key] = (field_type, value.field)
    
    model_name = f"{wrapper.__class__.__name__}Model"
    return create_model(model_name, **fields)

def get_all_wrappers():
    """Dynamically load all wrapper classes"""
    wrappers = {}
    wrappers_dir = Path("wrappers")
    for file in wrappers_dir.glob("*.py"):
        if file.name == "base_wrapper.py" or file.name.startswith("_"):
            continue
        
        module_name = f"wrappers.{file.stem}"
        module = importlib.import_module(module_name)
        
        for name, obj in module.__dict__.items():
            if (inspect.isclass(obj) and 
                issubclass(obj, BaseWrapper) and 
                obj is not BaseWrapper):
                wrapper = obj()
                wrappers[wrapper.title.lower().replace(" ", "_")] = wrapper
    
    return wrappers

WRAPPERS = get_all_wrappers()

class ProcessRequest(BaseModel):
    processors: List[str]
    settings: Dict[str, Dict[str, Any]]

@app.post("/api/v1/process/multi")
async def process_multi(
    request: ProcessRequest,
    files: List[UploadFile] = File(...)
):
    """
    Process multiple audio files through a chain of processors.
    
    This endpoint allows you to apply multiple audio processors in sequence to a set of input files.
    The output of each processor is fed as input to the next processor in the chain.
    
    ## Example Request
    
    ```python
    import requests
    
    url = "http://localhost:7860/api/v1/process/multi"
    
    # Audio file to process
    files = [
        ('files', ('vocals.wav', open('vocals.wav', 'rb'), 'audio/wav'))
    ]
    
    # Processor chain configuration
    json_data = {
        "processors": ["separate", "clone", "merge"],
        "settings": {
            "separate": {
                "vocals_only": True,
                "separate_bg_vocals": True,
                "reverb_removal": "Main Vocals"
            },
            "clone": {
                "selected_voice": "my_voice_model",
                "pitch_shift": 2,
                "pitch_extraction_method": "rmvpe+"
            },
            "merge": {
                "prevent_clipping": True
            }
        }
    }
    
    response = requests.post(url, files=files, json=json_data)
    ```
    
    ## Available Processors
    
    Use the `/api/v1/docs` endpoint to get a list of all available processors and their parameters.
    
    ## Common Processor Chains
    
    1. **Voice Cloning Pipeline**: `separate` → `clone` → `merge`
    2. **Audio Clean-up**: `separate` → `remaster` → `merge`
    3. **Enhanced Audio**: `separate` → `super_res` → `merge`
    
    ## Response
    
    The API returns the processed audio files as attachments.
    """
    try:
        # Create a temporary directory for this request
        with tempfile.TemporaryDirectory(dir=TEMP_DIR) as temp_dir:
            # Save uploaded files
            input_files = []
            for file in files:
                file_path = Path(temp_dir) / file.filename
                with file_path.open("wb") as f:
                    shutil.copyfileobj(file.file, f)
                input_files.append(ProjectFiles(str(file_path)))
            
            # Process through each wrapper
            current_files = input_files
            for processor_name in request.processors:
                if processor_name not in WRAPPERS:
                    raise HTTPException(status_code=400, detail=f"Unknown processor: {processor_name}")
                
                wrapper = WRAPPERS[processor_name]
                settings = request.settings.get(processor_name, {})
                
                # Validate settings against wrapper's allowed_kwargs
                if not wrapper.validate_args(**settings):
                    raise HTTPException(status_code=400, detail=f"Invalid settings for processor: {processor_name}")
                
                # Process files
                current_files = wrapper.process_audio(current_files, **settings)
            
            # Return the processed files
            output_files = []
            for project in current_files:
                for output in project.last_outputs:
                    output_path = Path(output)
                    if output_path.exists():
                        output_files.append(FileResponse(output))
            
            return output_files

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Create individual endpoints for each wrapper
for name, wrapper in WRAPPERS.items():
    model = create_pydantic_model_from_wrapper(wrapper)
    
    @app.post(f"/api/v1/process/{name}")
    async def process_single(
        files: List[UploadFile] = File(...),
        settings: model = None,
        wrapper=wrapper
    ):
        """
        Process audio files through a single processor
        """
        try:
            with tempfile.TemporaryDirectory(dir=TEMP_DIR) as temp_dir:
                # Save uploaded files
                input_files = []
                for file in files:
                    file_path = Path(temp_dir) / file.filename
                    with file_path.open("wb") as f:
                        shutil.copyfileobj(file.file, f)
                    input_files.append(ProjectFiles(str(file_path)))
                
                # Process files
                settings_dict = settings.dict() if settings else {}
                processed_files = wrapper.process_audio(input_files, **settings_dict)
                
                # Return the processed files
                output_files = []
                for project in processed_files:
                    for output in project.last_outputs:
                        output_path = Path(output)
                        if output_path.exists():
                            output_files.append(FileResponse(output))
                
                return output_files

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

# Add OpenAPI documentation
@app.get("/api/v1/docs")
async def get_documentation():
    """
    Get API documentation for all available processors.
    
    This endpoint returns detailed information about all available audio processors,
    including their parameters, descriptions, and default values.
    
    ## Example Request
    
    ```python
    import requests
    
    url = "http://localhost:7860/api/v1/docs"
    response = requests.get(url)
    processors = response.json()
    
    # Get info about a specific processor
    clone_info = processors["clone"]
    print(f"Clone processor: {clone_info['description']}")
    print("Parameters:")
    for param_name, param_info in clone_info['parameters'].items():
        print(f"  - {param_name}: {param_info['description']}")
    ```
    
    ## Response Format
    
    ```json
    {
        "processor_name": {
            "title": "Human-readable title",
            "description": "Detailed description",
            "priority": 1,
            "parameters": {
                "param_name": {
                    "description": "Parameter description",
                    "type": "Parameter type (str, int, float, bool)",
                    "default": "Default value",
                    "required": true/false
                }
            }
        }
    }
    ```
    """
    docs = {}
    for name, wrapper in WRAPPERS.items():
        docs[name] = {
            "title": wrapper.title,
            "description": wrapper.description,
            "priority": wrapper.priority,
            "parameters": {
                k: {
                    "description": v.description,
                    "type": str(v.type),
                    "default": v.field.default if v.field.default != ... else None,
                    "required": v.required
                }
                for k, v in wrapper.allowed_kwargs.items()
            }
        }
    return docs

def register_all_api_endpoints():
    """
    Register all API endpoints from each layout module
    """
    modules = [
        transcribe_module,
        music_module,
        orpheus_module,
        process_module,
        rvc_train_module,
        stable_audio_module,
        tts_module,
        diffrythm_module
    ]
    
    for module in modules:
        try:
            if hasattr(module, "register_api_endpoints"):
                logger.info(f"Registering API endpoints from {module.__name__}")
                module.register_api_endpoints(app)
            else:
                logger.warning(f"Module {module.__name__} does not have register_api_endpoints function")
        except Exception as e:
            logger.error(f"Error registering API endpoints from {module.__name__}: {e}")

# Register all API endpoints
register_all_api_endpoints() 