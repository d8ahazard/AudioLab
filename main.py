import argparse
import importlib
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
import multiprocessing as mp

import gradio as gr
from torchaudio._extension import _init_dll_path

from handlers.args import ArgHandler
from handlers.config import model_path
from layouts.rvc_train import render as rvc_render
from util.data_classes import ProjectFiles
from wrappers.base_wrapper import BaseWrapper

if os.name == "nt" and (3, 8) <= sys.version_info < (3, 99):
    _init_dll_path()
mp.set_start_method("spawn")

# Stop caching models in limbo!!
hf_dir = os.path.join(model_path, "hf")
transformers_dir = os.path.join(model_path, "transformers")
os.makedirs(hf_dir, exist_ok=True)
os.makedirs(transformers_dir, exist_ok=True)
# Set HF_HUB_CACHE_DIR to the model_path
os.environ["HF_HOME"] = hf_dir
arg_handler = ArgHandler()

project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def list_wrappers():
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
                print(f"Error importing module {module_name}: {e}")
                traceback.print_exc()

    all_wrappers = sorted(all_wrappers, key=lambda x: get_processor(x).priority)
    selected_wrappers = [wrapper for wrapper in all_wrappers if get_processor(wrapper).default]
    print(f"Found {len(all_wrappers)} wrappers: {all_wrappers}")
    return all_wrappers, selected_wrappers


def get_processor(processor_title: str) -> BaseWrapper:
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


def toggle_visibility(processors: List[str], all_wrappers: List[str], all_accordions: List[gr.Accordion]):
    return [
        gr.update(visible=(wrapper in processors and bool(get_processor(wrapper).allowed_kwargs)))
        for wrapper, acc in zip(all_wrappers, all_accordions)
    ]


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


def update_output_preview(output: str) -> Tuple[gr.update, gr.update]:
    show_audio = is_audio(output)
    show_image = not show_audio
    return (gr.update(visible=show_audio, value=output if show_audio else None),
            gr.update(visible=show_image, value=output if show_image else None))


def process(processors: List[str], inputs: List[str], progress=gr.Progress()) -> List[str]:
    start_time = datetime.now()
    settings = arg_handler.get_args()

    def progress_callback(step, desc=None, total=None):
        progress(step, desc, total)

    progress(0, f"Processing with {len(processors)} processors...")
    outputs = []
    inputs = [ProjectFiles(file_path) for file_path in inputs]
    for processor_title in processors:
        tgt_processor = get_processor(processor_title)
        processor_key = tgt_processor.title.replace(' ', '')
        processor_settings = settings.get(processor_key, {}) if settings else {}
        if len(processor_settings):
            print(f"Processor settings for {processor_title}:")
            print("---------------------------------------------------------------------")
            for key in processor_settings:
                print(f"{key}: {processor_settings[key]}")
            print("---------------------------------------------------------------------")
        else:
            print(f"No settings found for {processor_title}.")
        outputs = tgt_processor.process_audio(inputs, progress_callback, **processor_settings)
        inputs = outputs

    last_output_files = []
    for output in outputs:
        last_output_files.extend(output.last_outputs)
    outputs = last_output_files
    output_audio_files = get_audio_files(outputs)
    output_image_files = get_image_files(outputs)
    output_images_and_audio = output_audio_files + output_image_files
    first_image = output_image_files[0] if output_image_files else None
    first_audio = output_audio_files[0] if output_audio_files else None
    end_time = datetime.now()
    total_time_in_seconds = (end_time - start_time).total_seconds()
    print(f"Processing complete with {len(processors)} processors in {total_time_in_seconds:.2f} seconds")
    return (gr.update(value=outputs, visible=bool(outputs)),
            gr.update(value=first_audio, visible=bool(first_audio), choices=output_images_and_audio, interactive=True),
            gr.update(value=first_audio, visible=bool(first_audio)),
            gr.update(value=first_image, visible=bool(first_image)),
            gr.update(
                value=f"Processing complete with {len(processors)} processors in {total_time_in_seconds:.2f} seconds"))


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="AudioLab Web Server")
    parser.add_argument('--listen', action='store_true', help="Enable server to listen on 0.0.0.0")
    parser.add_argument('--port', type=int, default=7860, help="Specify the port number (default: 7860)")
    args = parser.parse_args()

    # Determine the launch configuration
    server_name = "0.0.0.0" if args.listen else "127.0.0.1"
    server_port = args.port

    # Set up the UI
    wrappers, enabled_wrappers = list_wrappers()
    arg_handler = BaseWrapper().arg_handler

    with gr.Blocks(title='AudioLab') as ui:
        with gr.Tabs():
            with gr.Tab(label='Process'):
                processor_list = gr.CheckboxGroup(label='Processors', choices=wrappers, value=enabled_wrappers)
                progress_display = gr.HTML(label='Progress', value='')

                accordions = []
                with gr.Row():
                    start_processing = gr.Button(value='Start Processing')

                with gr.Row():
                    with gr.Column() as settings_ui:
                        for wrapper_name in wrappers:
                            processor = get_processor(wrapper_name)
                            show_accordion = len(processor.allowed_kwargs) > 0 and processor.default
                            accordion = gr.Accordion(label=processor.title, visible=show_accordion, open=False)
                            with accordion:
                                processor.render_options(gr.Column())
                            accordions.append(accordion)

                    with gr.Column():
                        input_files = gr.File(label='Input Files', file_count='multiple', file_types=['audio', 'video'])
                    with gr.Column():
                        output_files = gr.File(label='Output Files', file_count='multiple',
                                               file_types=['audio', 'video'],
                                               interactive=False)
                        output_select = gr.Dropdown(label='Select Output Preview', choices=[], value=None,
                                                    visible=False, interactive=True)
                        output_audio = gr.Audio(label='Output Audio', value=None, visible=False)
                        output_image = gr.Image(label='Output Image', value=None, visible=False)

                processor_list.change(
                    fn=lambda processors: toggle_visibility(processors, wrappers, accordions),
                    inputs=[processor_list],
                    outputs=[accordion for accordion in accordions]
                )

                output_select.change(
                    fn=update_output_preview,
                    inputs=[output_select],
                    outputs=[output_audio, output_image]
                )

                start_processing.click(
                    fn=process,
                    inputs=[processor_list, input_files],
                    outputs=[output_files, output_select, output_audio, output_image, progress_display]
                )
            with gr.Tab(label="Train"):
                rvc_render()

    # Launch the UI with specified host and port
    ui.launch(server_name=server_name, server_port=server_port)
