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
from util.data_classes import ProjectFiles
from wrappers.base_wrapper import BaseWrapper, TypedInput

app = FastAPI(
    title="AudioLab API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
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