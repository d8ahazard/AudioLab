import os
import io
import zipfile

from fastapi import HTTPException
import gradio as gr
from typing import List, Tuple
from datetime import datetime
from pathlib import Path
import logging
import traceback
import importlib
import sys
from fastapi import HTTPException, Body
from pydantic import BaseModel
from typing import List, Dict, Any
import base64
import tempfile
from pathlib import Path
import os


from handlers.args import ArgHandler
from handlers.config import output_path
from handlers.download import download_files
from util.data_classes import ProjectFiles
from wrappers.base_wrapper import BaseWrapper

logger = logging.getLogger(__name__)

# Global dictionary to store path mappings
path_to_filename = {}
filename_to_path = {}


def list_wrappers():
    project_root = Path(__file__).parent.parent.resolve()
    script_dir = project_root / 'wrappers'
    if str(script_dir) not in sys.path:
        sys.path.append(str(script_dir))

    all_wrappers = []
    for file in script_dir.iterdir():
        if file.suffix == '.py' and file.name != 'base_wrapper.py':
            module_name = file.stem
            try:
                module = importlib.import_module(f'wrappers.{module_name}')
                for name, obj in module.__dict__.items():
                    if isinstance(obj, type) and issubclass(obj, BaseWrapper) and obj is not BaseWrapper:
                        wrapper_instance = obj()
                        all_wrappers.append(wrapper_instance.title)
                        break
            except Exception as e:
                logger.warning(f"Error importing module {module_name}: {e}")
                traceback.print_exc()

    all_wrappers = sorted(all_wrappers, key=lambda x: get_processor(x).priority)
    selected_wrappers = [wrapper for wrapper in all_wrappers if
                         get_processor(wrapper).default or get_processor(wrapper).required]
    return all_wrappers, selected_wrappers


def get_processor(processor_title: str) -> BaseWrapper:
    project_root = Path(__file__).parent.parent.resolve()
    script_dir = project_root / 'wrappers'
    for file in script_dir.iterdir():
        if file.suffix == '.py' and file.name != 'base_wrapper.py':
            module_name = file.stem
            module = importlib.import_module(f'wrappers.{module_name}')
            for name, obj in module.__dict__.items():
                if isinstance(obj, type) and issubclass(obj, BaseWrapper) and obj is not BaseWrapper:
                    instance = obj()
                    if instance.title == processor_title:
                        return instance
    return None


def get_filename(file_path: str) -> str:
    """Extract filename from a path."""
    return os.path.basename(file_path)


def update_path_mappings(file_paths: List[str]):
    """Update the global path mappings dictionaries."""
    global path_to_filename, filename_to_path
    for path in file_paths:
        if path:  # Only process non-empty paths
            filename = get_filename(path)
            # Handle duplicate filenames by appending a number
            base_filename = filename
            counter = 1
            while filename in filename_to_path and filename_to_path[filename] != path:
                name, ext = os.path.splitext(base_filename)
                filename = f"{name}_{counter}{ext}"
                counter += 1
            path_to_filename[path] = filename
            filename_to_path[filename] = path


def get_audio_files(file_paths: List[str]) -> List[str]:
    audio_files = []
    for file_path in file_paths:
        if Path(file_path).suffix.lower() in ['.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg', '.opus']:
            audio_files.append(file_path)
    return audio_files


def get_image_files(file_paths: List[str]) -> List[str]:
    image_files = []
    for file_path in file_paths:
        if Path(file_path).suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']:
            image_files.append(file_path)
    return image_files


def get_video_files(file_paths: List[str]) -> List[str]:
    video_files = []
    for file_path in file_paths:
        if Path(file_path).suffix.lower() in ['.mp4', '.mov', '.avi', '.webm', '.mkv', '.flv']:
            video_files.append(file_path)
    return video_files


def is_audio(file_path: str) -> bool:
    return Path(file_path).suffix.lower() in ['.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg', '.opus']


def is_video(file_path: str) -> bool:
    return Path(file_path).suffix.lower() in ['.mp4', '.mov', '.avi', '.webm', '.mkv', '.flv']


def handle_video_input(video_path: str, progress=gr.Progress()) -> ProjectFiles:
    """
    Extract audio from video file and create project structure.

    Args:
        video_path: Path to the video file
        progress: Progress callback function

    Returns:
        ProjectFiles object with extracted audio and video metadata
    """
    try:
        # Create project directory
        video_name = Path(video_path).stem
        project_dir = os.path.join(output_path, "process", f"video_{video_name}_{int(datetime.now().timestamp())}")
        os.makedirs(project_dir, exist_ok=True)

        # Create source subdirectory
        source_dir = os.path.join(project_dir, "source")
        os.makedirs(source_dir, exist_ok=True)

        # Copy video file to source directory
        video_filename = Path(video_path).name
        video_dest = os.path.join(source_dir, video_filename)
        import shutil
        shutil.copy2(video_path, video_dest)

        # Extract audio from video
        progress(0.1, f"Extracting audio from {video_filename}...")
        extracted_audio = extract_audio_from_video(video_path, project_dir)

        if not extracted_audio or not os.path.exists(extracted_audio):
            logger.error(f"Failed to extract audio from video: {video_path}")
            return None

        # Create ProjectFiles object
        project_file = ProjectFiles(extracted_audio)
        project_file.project_dir = project_dir
        project_file.video_source = video_dest
        project_file.extracted_audio_path = extracted_audio

        logger.info(f"Successfully processed video: {video_filename}")
        return project_file

    except Exception as e:
        logger.error(f"Error handling video input {video_path}: {e}")
        return None


def extract_audio_from_video(video_path: str, output_dir: str) -> str:
    """
    Extract audio track from video file.

    Args:
        video_path: Path to video file
        output_dir: Directory to save extracted audio

    Returns:
        Path to extracted audio file
    """
    try:
        # Use ffmpeg to extract audio
        import subprocess

        video_name = Path(video_path).stem
        output_audio = os.path.join(output_dir, f"{video_name}_extracted.wav")

        # ffmpeg command to extract audio
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit
            '-ar', '44100',  # Sample rate
            '-ac', '2',  # Stereo
            '-y',  # Overwrite output file
            output_audio
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0 and os.path.exists(output_audio):
            logger.info(f"Successfully extracted audio: {output_audio}")
            return output_audio
        else:
            logger.error(f"ffmpeg failed: {result.stderr}")
            return None

    except Exception as e:
        logger.error(f"Error extracting audio from video: {e}")
        return None


def update_preview(output: str) -> Tuple[gr.update, gr.update, gr.update]:
    if not output:
        show_audio = False
        show_image = False
        show_video = False
    else:
        # Convert filename back to full path if needed
        full_path = filename_to_path.get(output, output)
        show_audio = is_audio(full_path)
        show_video = is_video(full_path)
        show_image = not show_audio and not show_video
    return (gr.update(visible=show_audio, value=full_path if show_audio else None),
            gr.update(visible=show_image, value=full_path if show_image else None),
            gr.update(visible=show_video, value=full_path if show_video else None))


def update_preview_select(input_files: List[str]) -> Tuple[gr.update, gr.update, gr.update, gr.update]:
    if not input_files:
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    
    # Update path mappings
    update_path_mappings(input_files)
    
    audio_files = get_audio_files(input_files)
    image_files = get_image_files(input_files)
    video_files = get_video_files(input_files)
    
    first_audio = audio_files[0] if audio_files else None
    first_image = image_files[0] if image_files else None
    first_video = video_files[0] if video_files else None
    
    # Convert paths to filenames for display
    display_choices = [path_to_filename[path] for path in input_files]
    first_preview_name = path_to_filename.get(first_audio or first_video or first_image) if (first_audio or first_video or first_image) else None
    
    return (gr.update(choices=display_choices, value=first_preview_name, visible=bool(audio_files or image_files or video_files)),
            gr.update(value=first_audio, visible=bool(first_audio)),
            gr.update(value=first_image, visible=bool(first_image)),
            gr.update(value=first_video, visible=bool(video_files)))


def toggle_visibility(processors: List[str], all_wrappers: List[str], all_accordions: List[gr.Accordion]):
    logger.info(f"Processors: {processors}")
    foo = processors
    return [
        gr.update(visible=(wrapper in processors and bool(get_processor(wrapper).allowed_kwargs)))
        for wrapper, acc in zip(all_wrappers, all_accordions)
    ]


def check_processor_conflicts(processors: List[str]) -> gr.update:
    """
    Check for processor combinations that might cause issues and update warning visibility
    """
    is_clone_enabled = "Clone" in processors
    is_separate_enabled = "Separate" in processors
    
    # Show warning if Clone is enabled but Separate is not
    show_warning = is_clone_enabled and not is_separate_enabled
    
    return gr.update(visible=show_warning)


def enforce_defaults(processors: List[str]):
    wrappers, _ = list_wrappers()
    required_wrappers = [wrapper for wrapper in wrappers if get_processor(wrapper).required]
    # Add any required_wrapper that is not already in processors
    for wrapper in required_wrappers:
        if wrapper not in processors:
            processors.append(wrapper)
    # Sort the processors by priority
    processors = sorted(processors, key=lambda x: get_processor(x).priority)
    return gr.update(value=processors)


def process(processors: List[str], inputs: List[str], progress=gr.Progress()) -> List[str]:
    start_time = datetime.now()
    settings = ArgHandler().get_args()

    progress(0, f"Processing with {len(processors)} processors...")
    outputs = []
    all_outputs = []

    # Handle video files first: extract audio and create project structure
    original_videos = {}
    processed_inputs = []

    for input_file in inputs:
        if is_video(input_file):
            # Extract audio from video and create project structure
            project_file = handle_video_input(input_file, progress)
            if project_file:
                original_videos[input_file] = project_file.video_source  # Pass the video path, not audio
                processed_inputs.append(project_file)
            else:
                logger.warning(f"Failed to process video file: {input_file}")
        else:
            # Regular audio file
            processed_inputs.append(ProjectFiles(input_file))

    inputs = processed_inputs

    # Now check for files that should skip separation (after video processing)
    special_dirs = ["tts", "zonos", "stable_audio"]
    has_special_files = any(any(special_dir in input_file for special_dir in special_dirs) for input_file in inputs)
    has_tts_files = any(os.path.basename(input_file).startswith(("TTS_", "ZONOS_")) for input_file in inputs)
    has_extracted_audio = any("_extracted.wav" in input_file for input_file in inputs)
    has_processed_outputs = any(
        any(output_dir in input_file.lower() for output_dir in ["outputs", "process"])
        for input_file in inputs
    )

    # Check if we should skip separation for certain file types
    skip_separation_files = has_special_files or has_tts_files or has_extracted_audio or has_processed_outputs

    # If we have files that should skip separation and Separate is in processors, evaluate if we need it
    if skip_separation_files and "Separate" in processors:
        # Check if any files actually need separation
        # We need separation if we have regular audio files that aren't already processed
        has_regular_audio = any(
            not (has_special_files or has_tts_files) and
            not ("_extracted.wav" in input_file) and
            not any(output_dir in input_file.lower() for output_dir in ["outputs", "process"])
            for input_file in inputs
        )

        if not has_regular_audio:
            logger.warning("Removing 'Separate' processor - all input files are already processed, extracted audio, or special files that don't need separation")
            processors = [p for p in processors if p != "Separate"]
        else:
            logger.info(f"Keeping 'Separate' processor - found {sum(1 for input_file in inputs if not (has_special_files or has_tts_files) and not ('_extracted.wav' in input_file) and not any(output_dir in input_file.lower() for output_dir in ['outputs', 'process']))} regular audio files that may need separation")
    # Store the clone pitch shift value for the next processor
    clone_pitch_shift = settings.get("Clone", {}).get("pitch_shift", 0)
    pitch_shift_vocals_only = settings.get("Clone", {}).get("pitch_shift_vocals_only", False)
    clone_voice = settings.get("Clone", {}).get("selected_voice", None)
    f0_method = settings.get("Clone", {}).get("pitch_extraction_method", "rmvpe+")
    for idx, processor_title in enumerate(processors):
        tgt_processor = get_processor(processor_title)
        processor_key = tgt_processor.title.replace(' ', '')
        processor_settings = settings.get(processor_key, {}) if settings else {}
        # If the processor is 'merge', set the pitch_shift to the last clone pitch shift
        if processor_title == 'Merge' or processor_title == "Export":
            processor_settings['pitch_shift'] = clone_pitch_shift if pitch_shift_vocals_only else 0
            processor_settings['selected_voice'] = clone_voice
            processor_settings['pitch_extraction_method'] = f0_method
            # If we have original videos, pass them to the Export processor
            if original_videos and processor_title == "Export":
                processor_settings['original_videos'] = original_videos
        
        if len(processor_settings):
            logger.info(f"Processor settings for {processor_title}:")
            logger.info("---------------------------------------------------------------------")
            for key in processor_settings:
                logger.info(f"{key}: {processor_settings[key]}")
            logger.info("---------------------------------------------------------------------")
        else:
            logger.info(f"No settings found for {processor_title}.")
        
        try:
            progress(0.1 + (0.8 * idx / len(processors)), f"Processing with {processor_title}...")
            processor_outputs = tgt_processor.process_audio(inputs, progress, **processor_settings)
            if processor_outputs:
                outputs = processor_outputs
                for output in outputs:
                    if output.last_outputs:
                        all_outputs.extend(output.last_outputs)
            else:
                logger.warning(f"No outputs from processor {processor_title}")
        except Exception as e:
            logger.error(f"Error processing with {processor_title}: {e}")
            traceback.print_exc()
            # Don't break completely, return what we have so far
            progress(1.0, f"Error in {processor_title}: {str(e)}")
            break
            
        inputs = outputs

    # Last output should be first in the list
    all_outputs.reverse()
    outputs = all_outputs
    output_audio_files = get_audio_files(outputs)
    output_image_files = get_image_files(outputs)
    output_video_files = get_video_files(outputs)
    output_files_all = output_audio_files + output_image_files + output_video_files
    
    # Update path mappings for outputs
    update_path_mappings(output_files_all)
    
    first_image = output_image_files[0] if output_image_files else None
    first_audio = output_audio_files[0] if output_audio_files else None
    first_video = output_video_files[0] if output_video_files else None
    first_output = first_video or first_audio or first_image
    first_output_name = path_to_filename.get(first_output) if first_output else None
    
    # Convert paths to filenames for display
    display_choices = [path_to_filename[path] for path in output_files_all]
    
    end_time = datetime.now()
    total_time_in_seconds = (end_time - start_time).total_seconds()
    logger.info(f"Processing complete with {len(processors)} processors in {total_time_in_seconds:.2f} seconds")
    return (gr.update(value=outputs, visible=bool(outputs)),
            gr.update(value=first_output_name, visible=bool(first_output), choices=display_choices, interactive=True),
            gr.update(value=first_audio, visible=bool(first_audio)),
            gr.update(value=first_image, visible=bool(first_image)),
            gr.update(value=first_video, visible=bool(first_video)),
            gr.update(
                value=f"Processing complete with {len(processors)} processors in {total_time_in_seconds:.2f} seconds"))


def register_descriptions(arg_handler: ArgHandler):
    """Register descriptions for all wrappers with the arg handler."""
    wrappers, _ = list_wrappers()
    for wrapper_name in wrappers:
        processor = get_processor(wrapper_name)
        processor.register_descriptions(arg_handler)


def list_projects() -> List[str]:
    projects_folder = os.path.join(output_path, "process")
    projects = []
    if os.path.exists(projects_folder):
        projects = [f for f in os.listdir(projects_folder) if os.path.isdir(os.path.join(projects_folder, f))]
    return projects


def refresh_projects() -> gr.update:
    return gr.update(value=list_projects())


def load_project(project_name: str, input_files) -> gr.update:
    project_folder = os.path.join(output_path, "process", project_name, "source")
    # The only mp3 in the project folder is the source file
    source_file = None
    for file in os.listdir(project_folder):
        if file.endswith(".mp3"):
            source_file = os.path.join(project_folder, file)
            break
    if source_file:
        # If input files is is a list, add the source file to the list if it is not already there
        if isinstance(input_files, list):
            if source_file not in input_files:
                input_files.append(source_file)
        else:
            input_files = [source_file]
    return gr.update(value=input_files)


def render(arg_handler: ArgHandler):
    wrappers, enabled_wrappers = list_wrappers()
    gr.Markdown("# 🔄 Audio Processing Pipeline")
    gr.Markdown("Modular audio processing pipeline for separating vocals, cloning voices, enhancing quality, and converting formats. Chain multiple processors to create custom workflows for any audio transformation.")
    processor_list = gr.CheckboxGroup(label='Processors', choices=wrappers, value=enabled_wrappers,
                                      elem_id='processor_list', key="main_processor_list")
    
    # Warning message for when Clone is enabled without Separate
    separation_warning = gr.HTML(
        value='<div style="background-color: #FFF3CD; color: #856404; padding: 10px; border-radius: 5px; margin: 10px 0;">'
              '<strong>⚠️ Warning:</strong> Voice cloning requires separated vocals. '
              'Disabling "Separate" may cause issues with the "Clone" processor. '
              'Only disable separation for TTS or pre-separated audio files.</div>',
        visible=False
    )
    
    progress_display = gr.HTML(label='Progress', value='')

    accordions = []
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🔧 Settings")
            for wrapper_name in wrappers:
                processor = get_processor(wrapper_name)
                all_kwargs = processor.allowed_kwargs
                # Filter by kwargs with render=True
                render_kwargs = {key: value for key, value in all_kwargs.items() if value.render}
                show_accordion = len(render_kwargs) > 0 and processor.default
                accordion = gr.Accordion(label=processor.title, visible=show_accordion, open=False)
                with accordion:
                    processor.render_options(gr.Column())
                accordions.append(accordion)

        with gr.Column():
            gr.Markdown("### 🎤 Inputs")
            input_select = gr.Dropdown(label='Select Input Preview', choices=[], value=None, visible=False,
                                       interactive=True, key="process_input_preview")
            input_audio = gr.Audio(label='Input Audio', value=None, visible=False,
                                   key="process_input_audio")
            input_image = gr.Image(label='Input Image', value=None, visible=False,
                                   key="process_input_image")
            input_video = gr.Video(label='Input Video', value=None, visible=False,
                                  key="process_input_video")
            input_files = gr.File(label='Input Files', file_count='multiple', file_types=['audio', 'video'],
                                  key="process_inputs")
            arg_handler.register_element("main", "process_inputs", input_files)
            with gr.Row():
                with gr.Column(scale=2):
                    input_project = gr.Dropdown(label='Existing Project', visible=True,
                                                interactive=True, key="process_input_project", choices=list_projects())
                with gr.Column():
                    input_project_button = gr.Button(value='Load Project', visible=True, interactive=True)
                with gr.Column():
                    refresh_projects_button = gr.Button(value='Refresh Projects', visible=True, interactive=True)
            with gr.Row():
                with gr.Column(scale=2):
                    input_url = gr.Textbox(label='Input URL(s)', placeholder='Enter URL', visible=True,
                                           interactive=True, key="process_input_url")
                with gr.Column():
                    input_url_button = gr.Button(value='Load URL(s)', visible=True, interactive=True)
        with gr.Column():
            gr.Markdown("### 🎮 Actions")
            with gr.Row():
                start_processing = gr.Button(value='Start Processing', variant='primary')
                cancel_processing = gr.Button(value='Cancel Processing', variant='secondary', visible=False)
                
            gr.Markdown("### 🎶 Outputs")
            output_select = gr.Dropdown(label='Select Output Preview', choices=[], value=None,
                                        visible=False, interactive=True, key="process_output_preview")
            output_audio = gr.Audio(label='Output Audio', value=None, visible=False,
                                    key="process_output_audio")
            output_image = gr.Image(label='Output Image', value=None, visible=False,
                                    key="process_output_image")
            output_video = gr.Video(label='Output Video', value=None, visible=False,
                                   key="process_output_video")
            output_files = gr.File(label='Output Files', file_count='multiple',
                                   file_types=['audio', 'video'],
                                   interactive=False, key="process_output_files")
            progress_display = gr.HTML(label='Progress', value='')

    processor_list.input(
        fn=enforce_defaults,
        inputs=[processor_list],
        outputs=[processor_list]
    )

    processor_list.change(
        fn=lambda processors: (
            *toggle_visibility(processors, wrappers, accordions),
            check_processor_conflicts(processors)
        ),
        inputs=[processor_list],
        outputs=[
            *accordions,
            separation_warning
        ]
    )

    input_files.change(
        fn=update_preview_select,
        inputs=[input_files],
        outputs=[input_select, input_audio, input_image, input_video]
    )

    input_select.change(
        fn=update_preview,
        inputs=[input_select],
        outputs=[input_audio, input_image, input_video]
    )

    output_select.change(
        fn=update_preview,
        inputs=[output_select],
        outputs=[output_audio, output_image, output_video]
    )

    input_project_button.click(
        fn=load_project,
        inputs=[input_project, input_files],
        outputs=[input_files]
    )

    refresh_projects_button.click(
        fn=refresh_projects,
        outputs=[input_project]
    )

    input_url_button.click(
        fn=download_files,
        inputs=[input_url, input_files],
        outputs=[input_files]
    )

    start_processing.click(
        fn=process,
        inputs=[processor_list, input_files],
        outputs=[output_files, output_select, output_audio, output_image, output_video, progress_display]
    )


def listen():
    # Add any event listeners that need to be registered outside the render function
    pass


def register_api_endpoints(api):
    """
    Register API endpoints for the Process layout.
    
    Args:
        api: FastAPI application instance
    """
    # Define Pydantic models for JSON requests
    class FileData(BaseModel):
        filename: str
        content: str  # base64 encoded file content
    
    class ProcessRequest(BaseModel):
        files: List[FileData]
        processors: List[str]  # List of processor names in order of execution
        settings: Dict[str, Dict[str, Any]] = {}  # Optional settings for processors
    
    class ProcessorSetting(BaseModel):
        name: str
        type: str
        default: Any
        description: str
        options: List[str] = None
        min_value: float = None
        max_value: float = None
    
    class ProcessorInfo(BaseModel):
        title: str
        description: str
        priority: int
        default: bool
        required: bool
        settings: List[ProcessorSetting] = []
        
    @api.get("/api/v1/process/processors", tags=["Audio Processing"])
    async def list_processors():
        """
        List all available audio processors.
        
        Returns:
            List of audio processors with their information and available settings
        """
        try:
            all_wrappers, default_wrappers = list_wrappers()
            result = []
            
            for wrapper_name in all_wrappers:
                processor = get_processor(wrapper_name)
                if not processor:
                    continue
                
                # Extract settings information
                settings = []
                for key, kwarg in processor.allowed_kwargs.items():
                    setting = {
                        "name": key,
                        "type": kwarg.type.__name__ if hasattr(kwarg.type, "__name__") else str(kwarg.type),
                        "default": kwarg.default,
                        "description": kwarg.description
                    }
                    
                    # Add options for choices
                    if hasattr(kwarg, "choices") and kwarg.choices:
                        setting["options"] = kwarg.choices
                    
                    # Add min/max for numeric types
                    if hasattr(kwarg, "min_value"):
                        setting["min_value"] = kwarg.min_value
                    if hasattr(kwarg, "max_value"):
                        setting["max_value"] = kwarg.max_value
                        
                    settings.append(setting)
                
                processor_info = {
                    "title": processor.title,
                    "description": processor.description,
                    "priority": processor.priority,
                    "default": processor.default,
                    "required": processor.required,
                    "settings": settings
                }
                result.append(processor_info)
            
            # Sort by priority
            result = sorted(result, key=lambda x: x["priority"])
            return result
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @api.get("/api/v1/process/projects", tags=["Audio Processing"])
    async def get_projects():
        """
        List all available projects.
        
        Returns:
            List of project names
        """
        try:
            projects = list_projects()
            return {
                "projects": projects
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @api.post("/api/v1/process/chain", tags=["Audio Processing"])
    async def process_chain(
        request: ProcessRequest = Body(...)
    ):
        """
        Process multiple audio files through a chain of processors.
        
        This endpoint allows for executing multiple processing steps on audio files
        in sequence, with the output of each step becoming input to the next step.
        
        Request body:
        - files: List of files (base64 encoded) to process
        - processors: List of processor names to apply in sequence
        - settings: Dictionary of processor settings keyed by processor name
        
        Returns:
        - JSON response with base64-encoded output files
        """
        try:
            # Validate request
            if not request.files:
                raise HTTPException(status_code=400, detail="No files provided")
            if not request.processors:
                raise HTTPException(status_code=400, detail="No processors specified")
            
            # Validate processors
            all_wrappers, _ = list_wrappers()
            for processor_name in request.processors:
                if processor_name not in all_wrappers:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unknown processor: {processor_name}"
                    )
            
            # Create temporary directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save input files
                input_paths = []
                for file_data in request.files:
                    try:
                        file_content = base64.b64decode(file_data.content)
                        file_path = os.path.join(temp_dir, file_data.filename)
                        
                        with open(file_path, "wb") as f:
                            f.write(file_content)
                        
                        input_paths.append(file_path)
                    except Exception as e:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Failed to process file {file_data.filename}: {str(e)}"
                        )
                
                # Initialize progress logger for API
                class APIProgress:
                    def __call__(self, progress=0, desc=""):
                        logger.info(f"API Process Progress: {progress:.2f} - {desc}")
                
                # Process using the same logic as the UI
                try:
                    # Add required processors if not already included
                    wrappers, _ = list_wrappers()
                    required_wrappers = [wrapper for wrapper in wrappers if get_processor(wrapper).required]
                    for wrapper in required_wrappers:
                        if wrapper not in request.processors:
                            request.processors.append(wrapper)
                    
                    # Sort processors by priority
                    request.processors = sorted(request.processors, key=lambda x: get_processor(x).priority)
                    
                    # Initialize ProjectFiles objects
                    inputs = [ProjectFiles(file_path) for file_path in input_paths]
                    outputs = []
                    all_outputs = []
                    
                    # Store custom settings in ArgHandler format
                    settings = {}
                    for processor_name, processor_settings in request.settings.items():
                        # Convert processor name to format used in settings
                        processor_key = processor_name.replace(' ', '')
                        settings[processor_key] = processor_settings
                    
                    # Store pitch shift value for chaining
                    clone_pitch_shift = settings.get("Clone", {}).get("pitch_shift", 0)
                    pitch_shift_vocals_only = settings.get("Clone", {}).get("pitch_shift_vocals_only", False)
                    clone_voice = settings.get("Clone", {}).get("selected_voice", None)
                    f0_method = settings.get("Clone", {}).get("pitch_extraction_method", "rmvpe+")
                    
                    # Process each processor in sequence
                    for idx, processor_title in enumerate(request.processors):
                        tgt_processor = get_processor(processor_title)
                        processor_key = tgt_processor.title.replace(' ', '')
                        processor_settings = settings.get(processor_key, {})
                        
                        # Handle special case for Merge and Export
                        if processor_title == 'Merge' or processor_title == "Export":
                            processor_settings['pitch_shift'] = clone_pitch_shift if pitch_shift_vocals_only else 0
                            processor_settings['selected_voice'] = clone_voice
                            processor_settings['pitch_extraction_method'] = f0_method
                        
                        # Process with current processor
                        outputs = tgt_processor.process_audio(inputs, APIProgress(), **processor_settings)
                        for output in outputs:
                            all_outputs.extend(output.last_outputs)
                        
                        inputs = outputs
                    
                    # Reverse to get most recent outputs first
                    all_outputs.reverse()
                    final_outputs = all_outputs
                    
                    # Prepare response with file contents
                    output_files = []
                    for file_path in final_outputs:
                        if os.path.exists(file_path):
                            with open(file_path, "rb") as f:
                                file_content = base64.b64encode(f.read()).decode("utf-8")
                            
                            output_files.append({
                                "filename": os.path.basename(file_path),
                                "path": file_path,
                                "content": file_content,
                                "type": "audio" if file_path.endswith((".wav", ".mp3", ".flac")) else "image"
                            })
                    
                    # Create zip file with all outputs
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                        for file_path in final_outputs:
                            if os.path.exists(file_path):
                                zip_file.write(
                                    file_path,
                                    arcname=os.path.basename(file_path)
                                )
                    
                    zip_content = base64.b64encode(zip_buffer.getvalue()).decode("utf-8")
                    
                    return {
                        "status": "success",
                        "message": "Processing complete",
                        "files": output_files,
                        "processors_applied": request.processors,
                        "zip": {
                            "filename": "processed_outputs.zip",
                            "content": zip_content
                        }
                    }
                    
                except Exception as e:
                    logger.error(f"Error in process chain: {e}")
                    import traceback
                    traceback.print_exc()
                    raise HTTPException(
                        status_code=500,
                        detail=f"Processing failed: {str(e)}"
                    )
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in process chain: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))
            
    @api.post("/api/v1/process/load_project", tags=["Audio Processing"])
    async def load_project_endpoint(
        project_name: str = Body(..., embed=True)
    ):
        """
        Load a project and return its source files.
        
        Returns:
            List of files in the project
        """
        try:
            if not project_name or project_name not in list_projects():
                raise HTTPException(status_code=404, detail="Project not found")
            
            project_folder = os.path.join(output_path, "process", project_name, "source")
            if not os.path.exists(project_folder):
                raise HTTPException(status_code=404, detail="Project source folder not found")
            
            # Find all audio files in the project
            project_files = []
            for file in os.listdir(project_folder):
                if file.endswith((".mp3", ".wav", ".flac")):
                    file_path = os.path.join(project_folder, file)
                    with open(file_path, "rb") as f:
                        file_content = base64.b64encode(f.read()).decode("utf-8")
                    
                    project_files.append({
                        "filename": file,
                        "path": file_path,
                        "content": file_content
                    })
            
            return {
                "status": "success",
                "project": project_name,
                "files": project_files
            }
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
