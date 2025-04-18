import base64
import logging
import os
import re
import subprocess
import tempfile
from abc import abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Union, Tuple, Callable

import gradio as gr
import pydantic
from annotated_types import Ge, Le
from fastapi import HTTPException

from handlers.args import ArgHandler
from util.data_classes import ProjectFiles

logger = logging.getLogger(__name__)
if pydantic.__version__.startswith("1."):
    PYDANTIC_V2 = False
else:
    PYDANTIC_V2 = True


class TypedInput:
    def __init__(self, default: Any = ...,
                 description: str = None,
                 ge: float = None,
                 le: float = None,
                 step: float = None,
                 min_length: int = None,
                 max_length: int = None,
                 regex: str = None,
                 choices: List[Union[str, int]] = None,
                 type: type = None,
                 gradio_type: str = None,
                 render: bool = True,
                 required: bool = False,
                 refresh: Callable = None,
                 on_change: Callable = None,
                 on_click: Callable = None,
                 on_select: Callable = None,
                 controls: List[str] = None,
                 group_name: str = None,
                 ):
        field_kwargs = {
            "default": default,
            "description": description,
            "ge": ge,
            "le": le,
            "min_length": min_length,
            "max_length": max_length,
            "step": step,
        }

        if PYDANTIC_V2:
            field_kwargs["pattern"] = regex
            if choices:
                field_kwargs["json_schema_extra"] = {"enum": choices}
        else:
            field_kwargs["regex"] = regex
            field_kwargs["enum"] = choices

        field = pydantic.Field(**field_kwargs)
        self.type = type
        self.field = field
        self.render = render
        self.required = required
        self.refresh = refresh
        self.description = description
        self.choices = choices
        self.on_change = on_change
        self.on_click = on_click
        self.on_select = on_select
        self.controls = controls
        self.gradio_type = gradio_type if gradio_type else self.pick_gradio_type()
        self.group_name = group_name  # New parameter for accordion grouping

    def pick_gradio_type(self):
        if self.type == bool:
            return "Checkbox"
        if self.type == str:
            return "Text"
        if self.type == int or self.type == float:
            try:
                if self.field.ge is not None and self.field.le is not None:
                    return "Slider"
            except:
                pass
        if self.type == float:
            return "Number"
        if self.type == list:
            return "Textfield"
        # If we have choices, return a dropdown
        if self.choices:
            return "Dropdown"
        return "Text"


class BaseWrapper:
    _instance = None
    priority = 1000
    allowed_kwargs = {}
    description = "Base Wrapper"
    default = False
    required = False
    hidden_groups = []

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BaseWrapper, cls).__new__(cls)
            cls._instance.arg_handler = ArgHandler()
            class_name = cls.__name__
            cls._instance.title = ' '.join(
                word.capitalize() for word in re.sub(r'(?<!^)(?=[A-Z])', '_', class_name).split('_'))

        return cls._instance

    def validate_args(self, **kwargs: Dict[str, Any]) -> bool:
        """
        Validate the provided arguments.
        """
        filtered_kwargs = {}
        for key, value in kwargs.items():
            if key in self.allowed_kwargs:
                filtered_kwargs[key] = value
        for arg, value in self.allowed_kwargs.items():
            if value.field.required and arg not in filtered_kwargs or not filtered_kwargs[arg]:
                return False
        return True

    @abstractmethod
    def process_audio(self, inputs: List[ProjectFiles], callback=None, **kwargs: Dict[str, Any]) -> List[ProjectFiles]:
        pass

    @staticmethod
    def extract_audio_from_video(video_file: str) -> str:
        """
        Extract audio from a video file.
        
        Args:
            video_file: Path to the video file
            
        Returns:
            Path to the extracted audio file
        """
        # Get the directory and filename without extension
        video_dir = os.path.dirname(video_file)
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        
        # Create audio output path
        audio_file = os.path.join(video_dir, f"{video_name}_audio.wav")
        
        try:
            # Use ffmpeg to extract audio
            cmd = [
                "ffmpeg", "-y", "-i", video_file, 
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # PCM 16-bit
                "-ar", "44100",  # 44.1kHz sample rate
                "-ac", "2",  # Stereo
                audio_file
            ]
            
            result = subprocess.run(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Failed to extract audio from video: {result.stderr}")
                return None
                
            logger.info(f"Successfully extracted audio from video to {audio_file}")
            return audio_file
            
        except Exception as e:
            logger.error(f"Error extracting audio from video: {str(e)}")
            return None
            
    @staticmethod
    def recombine_audio_with_video(video_file: str, audio_file: str, output_file: str = None) -> str:
        """
        Recombine processed audio with the original video.
        
        Args:
            video_file: Path to the original video file
            audio_file: Path to the processed audio file
            output_file: Path to save the output video (default: auto-generated)
            
        Returns:
            Path to the output video file
        """
        # If output file is not specified, create one
        if not output_file:
            video_dir = os.path.dirname(video_file)
            video_name = os.path.splitext(os.path.basename(video_file))[0]
            audio_name = os.path.splitext(os.path.basename(audio_file))[0]
            output_file = os.path.join(video_dir, f"{video_name}_with_{audio_name}.mp4")
        
        # Validate input files
        if not os.path.exists(video_file):
            logger.error(f"Video file not found: {video_file}")
            return None
            
        if not os.path.exists(audio_file):
            logger.error(f"Audio file not found: {audio_file}")
            return None
            
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Use ffmpeg to combine video and audio
            cmd = [
                "ffmpeg", "-y",
                "-i", video_file,  # Video input
                "-i", audio_file,  # Audio input
                "-c:v", "copy",  # Copy video codec
                "-map", "0:v:0",  # Map video from first input
                "-map", "1:a:0",  # Map audio from second input
                "-shortest",  # End when shortest input ends
                output_file
            ]
            
            result = subprocess.run(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Failed to recombine audio with video: {result.stderr}")
                return None
                
            logger.info(f"Successfully recombined audio with video to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error recombining audio with video: {str(e)}")
            return None

    def register_api_endpoint(self, api) -> Any:
        """
        Register FastAPI endpoint for this wrapper.
        
        This method creates a standardized JSON API endpoint for the wrapper.
        Subclasses can override this method for custom behavior, but should
        follow the pattern of accepting only JSON requests.
        
        Args:
            api: FastAPI application instance
            
        Returns:
            The registered endpoint route
        """
        from fastapi import Body

        # Create Pydantic models for API documentation
        FileData, JsonRequest = self.create_json_models()
        
        # Extract wrapper name for endpoint path
        endpoint_name = self.title.lower().replace(" ", "_")
        
        # Create detailed description from wrapper metadata
        description = f"""
        {self.description}
        
        This endpoint processes audio files using the {self.title} processor.
        
        ## Request Body
        
        ```json
        {{
          "files": [
            {{
              "filename": "audio.wav",
              "content": "base64_encoded_file_content..."
            }}
          ],
          "settings": {{
        """
        
        # Add settings documentation from allowed_kwargs
        for key, value in self.allowed_kwargs.items():
            default_val = "null" if value.field.default is None else value.field.default
            if value.field.default is ...:
                default_val = "required"
            
            # Format description based on field type
            if value.type == bool:
                description += f"    \"{key}\": true|false,  // {value.description} Default: {default_val}\n"
            elif value.type == int:
                description += f"    \"{key}\": 0,  // {value.description} Default: {default_val}\n"
            elif value.type == float:
                description += f"    \"{key}\": 0.0,  // {value.description} Default: {default_val}\n"
            elif hasattr(value, 'choices') and value.choices:
                choices_str = "|".join([f'"{c}"' for c in value.choices])
                description += f"    \"{key}\": {choices_str},  // {value.description} Default: {default_val}\n"
            else:
                description += f"    \"{key}\": \"value\",  // {value.description} Default: {default_val}\n"
        
        description += """
          }}
        }}
        ```
        
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
        
        The API returns an object containing the processed files as Base64-encoded strings.
        """
        
        @api.post(f"/api/v1/process/{endpoint_name}", tags=["Audio Processing"])
        async def process_wrapper_json(
            request: JsonRequest = Body(...)
        ):
            # Use docstring from the description
            process_wrapper_json.__doc__ = description
            
            # Call the shared handler for JSON requests
            return self.handle_json_request(request, self.process_audio)
            
        return process_wrapper_json

    def create_json_models(self):
        """
        Create Pydantic models for JSON API requests and responses.
        
        Returns:
            Tuple containing (FileData, JsonRequest) models
        """
        from pydantic import BaseModel
        from typing import List, Optional, Annotated
        
        class FileData(BaseModel):
            filename: str = pydantic.Field(..., description="Name of the file with extension")
            content: str = pydantic.Field(..., description="Base64 encoded file content")
            
        # Create settings model from allowed_kwargs
        settings_model = self.create_settings_model()
        
        # Create a specific request model for this wrapper
        wrapper_name = self.__class__.__name__
        request_model_name = f"{wrapper_name}Request"
        
        # Create proper type annotations
        files_type = Annotated[List[FileData], pydantic.Field(description="Array of files to process")]
        settings_type = Annotated[Optional[settings_model], pydantic.Field(default=None, description=f"Settings for {self.title} processing")]
        
        request_model = type(request_model_name, (BaseModel,), {
            "__doc__": f"Request model for {self.title} processing",
            "__annotations__": {
                "files": files_type,
                "settings": settings_type
            }
        })
        
        return FileData, request_model
        
    def create_settings_model(self):
        """
        Create a Pydantic model from allowed_kwargs for validation and documentation.
        
        Returns:
            A Pydantic model class with fields based on allowed_kwargs
        """
        from pydantic import create_model
        from typing import Optional
        
        # Create fields dictionary
        fields = {}
        for key, value in self.allowed_kwargs.items():
            field_type = value.type
            field_kwargs = {
                "description": value.description
            }
            
            # Add validation parameters from the TypedInput field
            if value.field.default is not ...:
                field_kwargs["default"] = value.field.default
            if hasattr(value.field, "ge"):
                field_kwargs["ge"] = value.field.ge
            if hasattr(value.field, "le"):
                field_kwargs["le"] = value.field.le
            if hasattr(value.field, "min_length"):
                field_kwargs["min_length"] = value.field.min_length
            if hasattr(value.field, "max_length"):
                field_kwargs["max_length"] = value.field.max_length
            
            # Handle choices/enum if present
            if hasattr(value, "choices") and value.choices:
                if PYDANTIC_V2:
                    field_kwargs["json_schema_extra"] = {"enum": value.choices}
                else:
                    field_kwargs["enum"] = value.choices
            
            # Make field optional if it has a default value
            if value.field.default != ...:
                field_type = Optional[field_type]
            
            fields[key] = (field_type, pydantic.Field(**field_kwargs))
        
        # Create model with dynamic name
        model_name = f"{self.__class__.__name__}Settings"
        return create_model(
            model_name,
            __doc__=f"Settings model for {self.title} processing",
            **fields
        )
        
    def handle_json_request(self, request_data, processor_func):
        """
        Process a JSON API request with base64-encoded files.
        
        Args:
            request_data: The parsed JSON request data
            processor_func: Function to call for processing (usually self.process_audio)
            
        Returns:
            Dictionary response with base64-encoded output files
        """
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save uploaded files from base64
                input_files = []
                for file_data in request_data.files:
                    file_path = Path(temp_dir) / file_data.filename
                    
                    # Decode base64
                    try:
                        file_content = base64.b64decode(file_data.content)
                    except Exception as e:
                        raise HTTPException(
                            status_code=400, 
                            detail=f"Invalid base64 content for file {file_data.filename}: {str(e)}"
                        )
                    
                    with file_path.open("wb") as f:
                        f.write(file_content)
                    input_files.append(ProjectFiles(str(file_path)))
                
                # Process files
                settings_dict = {}
                if request_data.settings:
                    # Convert Pydantic model to dict if needed
                    if hasattr(request_data.settings, 'dict'):
                        settings_dict = request_data.settings.dict()
                    else:
                        settings_dict = request_data.settings
                
                # Track original video files for potential reconstruction
                original_videos = {}
                for project in input_files:
                    if project.src_file.lower().endswith(('.mp4', '.mov', '.avi', '.webm', '.mkv', '.flv')):
                        original_videos[project.src_file] = True
                
                # Add original videos to settings if this is the Export processor
                if self.__class__.__name__ == "Export" and original_videos:
                    if "original_videos" not in settings_dict:
                        settings_dict["original_videos"] = original_videos
                
                processed_files = processor_func(input_files, **settings_dict)
                
                # Return processed files as base64
                response_files = []
                for project in processed_files:
                    for output in project.last_outputs:
                        output_path = Path(output)
                        if output_path.exists():
                            # Read file and encode as base64
                            with open(output_path, "rb") as f:
                                file_content = base64.b64encode(f.read()).decode("utf-8")
                                
                            # Determine file type for better client-side handling
                            file_ext = output_path.suffix.lower()
                            file_type = "audio"
                            if file_ext in ['.mp4', '.mov', '.avi', '.webm', '.mkv', '.flv']:
                                file_type = "video"
                            elif file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']:
                                file_type = "image"
                                
                            response_files.append({
                                "filename": output_path.name,
                                "content": file_content,
                                "type": file_type
                            })
                
                return {"files": response_files}
                
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def create_json_endpoint_docs(self, endpoint_name, description=""):
        """
        Generate documentation for a JSON API endpoint.
        
        Args:
            endpoint_name: Name of the endpoint (e.g., "convert", "separate")
            description: Brief description of what the endpoint does
            
        Returns:
            Documentation string for the endpoint
        """
        base_docs = f"""
        {description} using JSON request.
        
        This endpoint is a JSON alternative to the multipart/form-data endpoint, allowing you to
        send files as Base64-encoded strings within a JSON payload. This can be more convenient 
        for some clients and allows for a consistent API style.
        
        ## Request Body
        
        ```json
        {{
          "files": [
            {{
              "filename": "audio.wav",
              "content": "base64_encoded_file_content..."
            }}
          ],
          "settings": {{
            "param1": "value1",
            "param2": "value2"
          }}
        }}
        ```
        
        ## Parameters
        
        - **files**: Array of file objects, each containing:
          - **filename**: Name of the file (with extension)
          - **content**: Base64-encoded file content
        - **settings**: Processing settings with options specific to this endpoint
          
        ## Example Request
        
        ```python
        import requests
        import base64
        
        url = "http://localhost:7860/api/v1/json/{endpoint_name}"
        
        # Read and encode file content
        with open('audio.wav', 'rb') as f:
            file_content = base64.b64encode(f.read()).decode('utf-8')
        
        # Prepare JSON payload
        payload = {{
            "files": [
                {{
                    "filename": "audio.wav",
                    "content": file_content
                }}
            ],
            "settings": {{
                "param1": "value1"
            }}
        }}
        
        # Send request
        response = requests.post(url, json=payload)
        
        # Process response - files will be returned as Base64 in the response
        result = response.json()
        for i, file_data in enumerate(result["files"]):
            with open(f"output_{{i}}.wav", "wb") as f:
                f.write(base64.b64decode(file_data["content"]))
        ```
        
        ## Response
        
        ```json
        {{
          "files": [
            {{
              "filename": "output.wav",
              "content": "base64_encoded_file_content..."
            }}
          ]
        }}
        ```
        
        The API returns an object containing an array of files, each with filename and Base64-encoded content.
        """
        
        return base_docs

    def render_options(self, container: gr.Column):
        """
        Render the options for this wrapper into the provided container.
        Groups elements into accordions if group_name is specified.
        """
        if not len(self.allowed_kwargs):
            return
            
        # First, organize elements by group
        groups = {}
        accordions = {}
        ungrouped = []
        
        for key, value in self.allowed_kwargs.items():
            if not value.render:
                continue

            if value.group_name:
                if value.group_name not in groups:
                    groups[value.group_name] = []
                groups[value.group_name].append((key, value))
            else:
                ungrouped.append((key, value))

        elements = {}

        with container:
            # First render ungrouped elements
            for key, value in ungrouped:
                elem = self.create_gradio_element(self.__class__.__name__, key, value)
                # If create_gradio_element returns a tuple (elem, refresh_button), handle it
                if isinstance(elem, tuple):
                    elements[key] = elem[0]
                    # The refresh button is already connected to the element
                else:
                    elements[key] = elem

            # Then render grouped elements in accordions
            for group_name, group_items in groups.items():
                visible = group_name not in self.hidden_groups
                with gr.Accordion(label=group_name, open=False, visible=visible) as accordion:
                    # Register the accordion element with a special ID for toggling
                    accordions[group_name] = accordion
                    for key, value in group_items:
                        if value.render:
                            all_hidden = False
                        elem = self.create_gradio_element(self.__class__.__name__, key, value)
                        # If create_gradio_element returns a tuple (elem, refresh_button), handle it
                        if isinstance(elem, tuple):
                            elements[key] = elem[0]
                            # The refresh button is already connected to the element
                        else:
                            elements[key] = elem
                    
            handler_keys = ["on_change", "on_click", "on_select"]
            for key, value in self.allowed_kwargs.items():
                for handler_key in handler_keys:
                    handler_func = handler_key.replace("on_", "")
                    if hasattr(value, handler_key) and callable(getattr(value, handler_key)):
                        logger.info(f"Setting up {handler_func} for {key}")
                        controls = value.controls
                        control_accordions = [accordions[control] for control in controls if control in accordions]
                        target_element = elements[key]
                        if hasattr(target_element, handler_func):
                            logger.info(f"Setting up {handler_func} for {key} (Found element)")
                            getattr(target_element, handler_func)(fn=getattr(value, handler_key), inputs=target_element, outputs=control_accordions)
                        else:
                            logger.info(f"Setting up {handler_func} for {key} (No element found)")

    def register_descriptions(self, arg_handler: ArgHandler):
        """
        Register the descriptions for the elements.
        """
        arg_handler.register_description(self.title, "description", self.description)
        for key, value in self.allowed_kwargs.items():
            arg_handler.register_description(self.__class__.__name__, key, value.description)

    def create_gradio_element(self, class_name: str, key: str, value: TypedInput):
        """
        Create and register a Gradio element for the specified input field.
        """
        arg_key = key
        key = " ".join([word.capitalize() for word in key.split("_")])

        # Extract `ge` and `le` from metadata if they exist
        ge_value = None
        le_value = None
        choices = None
        step = None
        for meta in value.field.metadata:
            if isinstance(meta, Ge):
                ge_value = meta.ge
            elif isinstance(meta, Le):
                le_value = meta.le

        if value.field.json_schema_extra:
            for extra in value.field.json_schema_extra:
                if extra == "enum":
                    choices = value.field.json_schema_extra[extra]
                if extra == "step":
                    step = value.field.json_schema_extra[extra]

        match value.gradio_type:
            case "Checkbox":
                elem = gr.Checkbox(label=key, value=value.field.default)
            case "Text":
                elem = gr.Textbox(label=key, value=value.field.default)
            case "Slider":
                elem = gr.Slider(
                    label=key,
                    value=value.field.default,
                    minimum=ge_value if ge_value is not None else 0,
                    maximum=le_value if le_value is not None else 100,
                    step=step if step is not None else 1 if isinstance(value.field.default, int) else 0.1
                )
            case "Number":
                elem = gr.Number(label=key, value=value.field.default)
            case "Textfield":
                elem = gr.Textbox(label=key, value=value.field.default, lines=3)
            case "Dropdown":
                elem = gr.Dropdown(label=key, choices=choices, value=value.field.default)
            case "File":
                elem = gr.File(label=key)
            case _:
                elem = gr.Textbox(label=key, value=value.field.default)

        elem.__setattr__("elem_id", f"{class_name}_{arg_key}")
        elem.__setattr__("elem_classes", ["hintitem"])
        elem.__setattr__("key", f"{class_name}_{arg_key}")

        self.arg_handler.register_element(class_name, arg_key, elem, value.description)
        setattr(elem, "visible", value.render)
        if isinstance(value.refresh, Callable):
            # Create a gradio button as well, to trigger the refresh
            refresh_button = gr.Button(value=f"Refresh {key}")
            getattr(refresh_button, "click")(fn=value.refresh, outputs=[elem])
            return elem, refresh_button
        return elem

    @staticmethod
    def filter_inputs(project: ProjectFiles, input_type: str = "audio") -> Tuple[List[str], List[str]]:
        """
        Filter the inputs to only include files that exist.
        """
        filtered_inputs, outputs = [], []
        extensions = []
        inputs = project.last_outputs
        
        # If no previous outputs exist, add the source file 
        if not inputs:
            # Check if a stems folder exists with files that match the naming convention
            stem_dir = os.path.join(project.project_dir, "stems")
            if os.path.exists(stem_dir):
                # If we have a vocals file in the stems directory, use that
                for f in os.listdir(stem_dir):
                    if "(Vocals)" in f:
                        vocals_file = os.path.join(stem_dir, f)
                        inputs = [vocals_file]
                        break
                # If we didn't find a vocals file, just use all files in the stems directory
                if not inputs:
                    stem_files = [os.path.join(stem_dir, f) for f in os.listdir(stem_dir) 
                                 if os.path.isfile(os.path.join(stem_dir, f)) and not f.endswith(".json")]
                    if stem_files:
                        inputs = stem_files
            
            # If we still have no inputs, use the source file
            if not inputs:
                inputs = [project.src_file]
                
        match input_type:
            case "audio":
                extensions = ["mp3", "wav", "flac", "m4a", "aac", "ogg", "opus"]
            case "image":
                extensions = ["jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp"]
            case "video":
                extensions = ["mp4", "mov", "avi", "webm", "mkv", "flv"]
            case "text":
                extensions = ["txt", "csv", "json"]
            case "any":
                # Accept any file type
                extensions = []
            case _:
                logger.warning(f"Unknown input type: {input_type}")
                pass
        
        # Special handling for video files - extract audio if needed
        if input_type == "audio":
            for input_file in inputs:
                file, ext = os.path.splitext(input_file)
                ext = ext[1:].lower()
                if ext in extensions:
                    filtered_inputs.append(input_file)
                elif ext in ["mp4", "mov", "avi", "webm", "mkv", "flv"]:
                    # This is a video file - extract the audio
                    audio_file = BaseWrapper.extract_audio_from_video(input_file)
                    if audio_file:
                        filtered_inputs.append(audio_file)
                        # Store the original video path in the project metadata
                        if not hasattr(project, "video_sources"):
                            project.video_sources = {}
                        project.video_sources[audio_file] = input_file
                    else:
                        outputs.append(input_file)
                else:
                    outputs.append(input_file)
        else:
            # Standard handling for non-audio file types
            for input_file in inputs:
                file, ext = os.path.splitext(input_file)
                if not extensions or ext[1:].lower() in extensions:
                    filtered_inputs.append(input_file)
                else:
                    outputs.append(input_file)
                
        return filtered_inputs, outputs
