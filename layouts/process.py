import os

from fastapi import HTTPException
import gradio as gr
from typing import List, Tuple
from datetime import datetime
from pathlib import Path
import logging
import traceback
import importlib
import sys

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
        if Path(file_path).suffix in ['.wav', '.mp3', '.flac']:
            audio_files.append(file_path)
    return audio_files


def get_image_files(file_paths: List[str]) -> List[str]:
    image_files = []
    for file_path in file_paths:
        if Path(file_path).suffix in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']:
            image_files.append(file_path)
    return image_files


def is_audio(file_path: str) -> bool:
    return Path(file_path).suffix in ['.wav', '.mp3', '.flac']


def update_preview(output: str) -> Tuple[gr.update, gr.update]:
    if not output:
        show_audio = False
        show_image = False
    else:
        # Convert filename back to full path if needed
        full_path = filename_to_path.get(output, output)
        show_audio = is_audio(full_path)
        show_image = not show_audio
    return (gr.update(visible=show_audio, value=full_path if show_audio else None),
            gr.update(visible=show_image, value=full_path if show_image else None))


def update_preview_select(input_files: List[str]) -> Tuple[gr.update, gr.update, gr.update]:
    if not input_files:
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    
    # Update path mappings
    update_path_mappings(input_files)
    
    audio_files = get_audio_files(input_files)
    image_files = get_image_files(input_files)
    first_audio = audio_files[0] if audio_files else None
    first_image = image_files[0] if image_files else None
    
    # Convert paths to filenames for display
    display_choices = [path_to_filename[path] for path in input_files]
    first_audio_name = path_to_filename.get(first_audio) if first_audio else None
    
    return (gr.update(choices=display_choices, value=first_audio_name, visible=bool(audio_files) or bool(image_files)),
            gr.update(value=first_audio, visible=bool(audio_files)),
            gr.update(value=first_image, visible=bool(image_files)))


def toggle_visibility(processors: List[str], all_wrappers: List[str], all_accordions: List[gr.Accordion]):
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
    
    # Check for special directories in input files that should automatically skip separation
    special_dirs = ["tts", "zonos", "stable_audio"]
    has_special_files = any(any(special_dir in input_file for special_dir in special_dirs) for input_file in inputs)
    has_tts_files = any(os.path.basename(input_file).startswith(("TTS_", "ZONOS_")) for input_file in inputs)
    
    # If we have special files but Separate is in processors, note it but continue
    if (has_special_files or has_tts_files) and "Separate" in processors:
        logger.info("Special files detected that would typically skip separation - continuing with user-selected processors")
    
    inputs = [ProjectFiles(file_path) for file_path in inputs]
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
        if len(processor_settings):
            logger.info(f"Processor settings for {processor_title}:")
            logger.info("---------------------------------------------------------------------")
            for key in processor_settings:
                logger.info(f"{key}: {processor_settings[key]}")
            logger.info("---------------------------------------------------------------------")
        else:
            logger.info(f"No settings found for {processor_title}.")
        try:
            outputs = tgt_processor.process_audio(inputs, progress, **processor_settings)
            for output in outputs:
                all_outputs.extend(output.last_outputs)
        except Exception as e:
            logger.error(f"Error processing with {processor_title}: {e}")
            traceback.print_exc()
            break
        inputs = outputs

    # Last output should be first in the list
    all_outputs.reverse()
    outputs = all_outputs
    output_audio_files = get_audio_files(outputs)
    output_image_files = get_image_files(outputs)
    output_images_and_audio = output_audio_files + output_image_files
    
    # Update path mappings for outputs
    update_path_mappings(output_images_and_audio)
    
    first_image = output_image_files[0] if output_image_files else None
    first_audio = output_audio_files[0] if output_audio_files else None
    first_output_name = path_to_filename.get(output_images_and_audio[0]) if output_images_and_audio else None
    
    # Convert paths to filenames for display
    display_choices = [path_to_filename[path] for path in output_images_and_audio]
    
    end_time = datetime.now()
    total_time_in_seconds = (end_time - start_time).total_seconds()
    logger.info(f"Processing complete with {len(processors)} processors in {total_time_in_seconds:.2f} seconds")
    return (gr.update(value=outputs, visible=bool(outputs)),
            gr.update(value=first_output_name, visible=bool(first_audio), choices=display_choices, interactive=True),
            gr.update(value=first_audio, visible=bool(first_audio)),
            gr.update(value=first_image, visible=bool(first_image)),
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
        outputs=[input_select, input_audio, input_image]
    )

    input_select.change(
        fn=update_preview,
        inputs=[input_select],
        outputs=[input_audio, input_image]
    )

    output_select.change(
        fn=update_preview,
        inputs=[output_select],
        outputs=[output_audio, output_image]
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
        outputs=[output_files, output_select, output_audio, output_image, progress_display]
    )


def listen():
    # Add any event listeners that need to be registered outside the render function
    pass


def register_api_endpoints(api):
    """
    Register API endpoints for the Process module
    
    Args:
        api: FastAPI application instance
    """
    # Note: The main processing endpoints are already registered in api.py
    # This function adds additional endpoints for managing projects
    
    @api.get("/api/v1/process/projects", tags=["Multi-Processing"])
    async def api_list_projects():
        """
        List all available processing projects
        
        Returns:
            List of project names
        """
        try:
            projects = list_projects()
            return {
                "projects": projects
            }
            
        except Exception as e:
            logger.exception("Error listing projects:")
            raise HTTPException(status_code=500, detail=f"Error listing projects: {str(e)}")
            
    @api.get("/api/v1/process/processors", tags=["Multi-Processing"])
    async def api_list_processors():
        """
        List all available processors and their settings
        
        Returns:
            Dictionary of processors with their settings and descriptions
        """
        try:
            all_wrappers, _ = list_wrappers()
            result = {}
            
            for wrapper_name in all_wrappers:
                processor = get_processor(wrapper_name)
                
                # Skip if processor is None
                if not processor:
                    continue
                    
                processor_info = {
                    "title": processor.title,
                    "description": processor.description,
                    "priority": processor.priority,
                    "default": processor.default,
                    "required": processor.required,
                    "parameters": {}
                }
                
                # Add parameter information
                for param_name, param_info in processor.allowed_kwargs.items():
                    processor_info["parameters"][param_name] = {
                        "type": str(param_info.type.__name__),
                        "description": param_info.description,
                        "default": param_info.field.default if param_info.field.default != ... else None,
                        "required": param_info.required,
                        "render": param_info.render
                    }
                    
                    # Add additional parameter metadata if available
                    for meta_key in ["min", "max", "step", "choices"]:
                        if hasattr(param_info, meta_key):
                            processor_info["parameters"][param_name][meta_key] = getattr(param_info, meta_key)
                
                result[processor.title.lower().replace(" ", "_")] = processor_info
                
            return result
            
        except Exception as e:
            logger.exception("Error listing processors:")
            raise HTTPException(status_code=500, detail=f"Error listing processors: {str(e)}")
