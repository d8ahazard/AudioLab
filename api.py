import importlib
import inspect
import logging
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
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


@app.post("/api/v1/process/multi", tags=["Multi-Processing"])
async def process_multi(
        request: ProcessRequest,
        files: List[UploadFile] = File(...)
):
    """
    Process multiple audio files through a chain of processors
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


    @app.post(f"/api/v1/process/{name}", tags=["Audio Processing"])
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
@app.get("/api/v1/docs", tags=["Multi-Processing"])
async def get_documentation():
    """
    Get API documentation for all available processors
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
