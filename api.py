import importlib
import inspect
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, create_model

from layouts.music import register_api_endpoints as register_music_endpoints
# Import register_api_endpoints functions from layout modules
from layouts.orpheus import register_api_endpoints as register_orpheus_endpoints
from layouts.process import register_api_endpoints as register_process_endpoints
from layouts.rvc_train import register_api_endpoints as register_rvc_endpoints
from layouts.stable_audio import register_api_endpoints as register_stable_audio_endpoints
from layouts.transcribe import register_api_endpoints as register_transcribe_endpoints
from layouts.tts import register_api_endpoints as register_tts_endpoints
from modules.acestep.api import register_api_endpoints as register_acestep_endpoints
from wrappers.base_wrapper import BaseWrapper

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
    - **Music Generation**: Complete song generation with YuE, DiffRhythm, ACE-Step, and Stable Audio
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
            "name": "Stable Audio",
            "description": """
            Music generation with Stable Audio:
            - Text-to-music generation
            - High-quality synthesis
            - Style control
            """
        },
        {
            "name": "ACE-Step",
            "description": """
            ACE-Step music generation:
            - Fast high-quality music generation
            - Lyrics support with vocal singing
            - LoRA-based specialized models
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
register_music_endpoints(app)
register_process_endpoints(app)
register_rvc_endpoints(app)
register_stable_audio_endpoints(app)
register_tts_endpoints(app)
register_transcribe_endpoints(app)
register_acestep_endpoints(app)

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



# Create individual endpoints for each wrapper
for name, wrapper in WRAPPERS.items():
    wrapper.register_api_endpoint(app)
