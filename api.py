import importlib
import inspect
import logging
import shutil
import tempfile
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, create_model

from util.data_classes import ProjectFiles
from wrappers.base_wrapper import BaseWrapper

# Import register_api_endpoints functions from layout modules
from layouts.orpheus import register_api_endpoints as register_orpheus_endpoints
from layouts.diffrythm import register_api_endpoints as register_diffrythm_endpoints
from layouts.music import register_api_endpoints as register_music_endpoints
from layouts.process import register_api_endpoints as register_process_endpoints
from layouts.rvc_train import register_api_endpoints as register_rvc_endpoints
from layouts.stable_audio import register_api_endpoints as register_stable_audio_endpoints
from layouts.tts import register_api_endpoints as register_tts_endpoints
from layouts.transcribe import register_api_endpoints as register_transcribe_endpoints

# Initialize logger
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AudioLab API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    favicon_path="/res/favicon.ico",
    description="""
    # AudioLab API

    AudioLab is an open-source audio processing application focused on voice-cloning, audio separation, and audio manipulation.
    This API provides access to all the core functionalities of AudioLab.

    ## Overview

    - **Audio Processing**: Separation, cloning, enhancement, and format conversion
    - **Voice Synthesis**: Multiple TTS engines including Orpheus for emotional synthesis
    - **Music Generation**: Complete song generation with YuE, DiffRhythm, and Stable Audio
    - **Training**: Custom voice models with RVC and DiffRhythm
    - **Utilities**: Transcription, project management, and multi-processing

    For detailed documentation on each endpoint's parameters and responses,
    explore the sections below.
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
    openapi_tags=[
        {
            "name": "Audio Processing",
            "description": """
            Core audio processing functionality including:
            - Audio separation (vocals, instruments, etc.)
            - Voice cloning with RVC
            - Audio enhancement and remastering
            - Format conversion and export
            """
        },
        {
            "name": "RVC",
            "description": """
            RVC voice cloning system:
            - Voice model training
            - Model management and download
            - Training job monitoring
            """
        },
        {
            "name": "Orpheus TTS",
            "description": """
            Emotional text-to-speech synthesis:
            - Speech generation with emotions
            - Custom voice training
            - Voice model management
            """
        },
        {
            "name": "Standard TTS",
            "description": """
            Traditional text-to-speech synthesis:
            - Multiple TTS models and voices
            - Language and speaker selection
            - Voice cloning support
            """
        },
        {
            "name": "YuE Music",
            "description": """
            Music generation with YuE:
            - Text-to-music generation
            - Style transfer and control
            - Reference audio support
            """
        },
        {
            "name": "DiffRhythm",
            "description": """
            Complete song generation with lyrics:
            - Song generation with LRC lyrics
            - Custom model training
            - Project and file management
            - Training job monitoring
            """
        },
        {
            "name": "Stable Audio",
            "description": """
            Music generation with Stable Audio:
            - Text-to-music generation
            - High-quality synthesis
            - Style control
            """
        },
        {
            "name": "Transcription",
            "description": """
            Speech transcription functionality:
            - Multi-speaker transcription
            - Speaker diarization
            - Multiple output formats (TXT, JSON, LRC)
            """
        },
        {
            "name": "Multi-Processing",
            "description": """
            Advanced processing pipelines:
            - Chain multiple processors
            - Custom processing workflows
            - Batch processing support
            """
        }
    ]
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a temporary directory for file uploads
TEMP_DIR = Path("temp_uploads")
TEMP_DIR.mkdir(exist_ok=True)

# Register API endpoints from layout modules
register_orpheus_endpoints(app)
register_diffrythm_endpoints(app)
register_music_endpoints(app)
register_process_endpoints(app)
register_rvc_endpoints(app)
register_stable_audio_endpoints(app)
register_tts_endpoints(app)
register_transcribe_endpoints(app)

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
    files: List[Dict[str, str]]  # Array of {filename, content} objects


@app.post("/api/v1/process/multi", tags=["Multi-Processing"])
async def process_multi(
        request: ProcessRequest = Body(...)
):
    """
    Process multiple audio files through a chain of processors

    This endpoint allows you to chain multiple audio processors together,
    passing the output of one processor as the input to the next.

    ## Request Body

    ```json
    {
      "processors": ["separate", "clone", "merge"],
      "settings": {
        "separate": {
          "vocals_only": true
        },
        "clone": {
          "selected_voice": "my_model"
        }
      },
      "files": [
        {
          "filename": "audio.wav",
          "content": "base64_encoded_file_content..."
        }
      ]
    }
    ```

    ## Parameters

    - **processors**: Array of processor names to apply in sequence
    - **settings**: Object containing settings for each processor
    - **files**: Array of file objects, each containing:
      - **filename**: Name of the file (with extension)
      - **content**: Base64-encoded file content

    ## Response

    ```json
    {
      "files": [
        {
          "filename": "processed_output.wav",
          "content": "base64_encoded_file_content..."
        }
      ]
    }
    ```
    """
    try:
        # Create a temporary directory for this request
        with tempfile.TemporaryDirectory(dir=TEMP_DIR) as temp_dir:
            # Save uploaded files
            input_files = []
            for file_data in request.files:
                file_path = Path(temp_dir) / file_data["filename"]
                
                # Decode base64
                try:
                    file_content = base64.b64decode(file_data["content"])
                except Exception as e:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Invalid base64 content for file {file_data['filename']}: {str(e)}"
                    )
                
                with file_path.open("wb") as f:
                    f.write(file_content)
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
            response_files = []
            for project in current_files:
                for output in project.last_outputs:
                    output_path = Path(output)
                    if output_path.exists():
                        # Read file and encode as base64
                        with open(output_path, "rb") as f:
                            file_content = base64.b64encode(f.read()).decode("utf-8")
                            
                        response_files.append({
                            "filename": output_path.name,
                            "content": file_content
                        })

            return {"files": response_files}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Create individual endpoints for each wrapper
for name, wrapper in WRAPPERS.items():
    wrapper.register_api_endpoint(app)


# Add OpenAPI documentation
@app.get("/api/v1/docs", tags=["API"])
async def get_documentation():
    """
    Get comprehensive API documentation for all available processors
    
    Returns detailed information about all available audio processors,
    including their parameters, descriptions, and default values.
    """
    docs = {}
    for name, wrapper in WRAPPERS.items():
        parameters = {}
        for k, v in wrapper.allowed_kwargs.items():
            # Extract range constraints
            range_info = {}
            for meta in v.field.metadata:
                if hasattr(meta, 'ge') and meta.ge is not None:
                    range_info['minimum'] = meta.ge
                if hasattr(meta, 'le') and meta.le is not None:
                    range_info['maximum'] = meta.le
                    
            # Get choices if available
            choices = None
            if hasattr(v.field, 'json_schema_extra') and v.field.json_schema_extra:
                if 'enum' in v.field.json_schema_extra:
                    choices = v.field.json_schema_extra['enum']
            
            parameters[k] = {
                "description": v.description,
                "type": str(v.type.__name__),
                "default": v.field.default if v.field.default != ... else None,
                "required": v.required,
                "gradio_type": v.gradio_type,
                "range": range_info if range_info else None,
                "choices": choices
            }
            
        docs[name] = {
            "title": wrapper.title,
            "description": wrapper.description,
            "priority": wrapper.priority,
            "parameters": parameters,
            "endpoint": f"/api/v1/process/{name.lower()}"
        }
    
    return {
        "version": "1.0.0",
        "description": "AudioLab API provides access to audio processing, voice cloning, and music generation capabilities.",
        "processors": docs,
        "multi_processing": {
            "endpoint": "/api/v1/process/multi",
            "description": "Chain multiple processors together in a sequence",
            "example": {
                "processors": ["separate", "clone", "merge"],
                "settings": {
                    "separate": {"vocals_only": True},
                    "clone": {"selected_voice": "model_name"}
                }
            }
        }
    }
