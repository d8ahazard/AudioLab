import re

import yt_dlp

import handlers.processing  # noqa (Keep this here, and first, as it is required for multiprocessing to work)
import argparse
import importlib
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import gradio as gr
from torchaudio._extension import _init_dll_path

from handlers.args import ArgHandler
from handlers.config import model_path, output_path
from handlers.download import download_files
from layouts.rvc_train import render as rvc_render, register_descriptions as rvc_register_descriptions
from layouts.music import render as render_music, register_descriptions as music_register_descriptions, \
    listen as music_listen
from layouts.tts import render_tts, register_descriptions as tts_register_descriptions, listen as tts_listen
from layouts.zonos import render_zonos, register_descriptions as zonos_register_descriptions, listen as zonos_listen
from util.data_classes import ProjectFiles
from wrappers.base_wrapper import BaseWrapper
import logging

logger = logging.getLogger(__name__)
# Set TF_ENABLE_ONEDNN_OPTS=0
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

if os.name == "nt" and (3, 8) <= sys.version_info < (3, 99):
    _init_dll_path()

os.environ["COQUI_TOS_AGREED"] = "1"
os.environ["TTS_HOME"] = model_path
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
                logger.warning(f"Error importing module {module_name}: {e}")
                traceback.print_exc()

    all_wrappers = sorted(all_wrappers, key=lambda x: get_processor(x).priority)
    selected_wrappers = [wrapper for wrapper in all_wrappers if
                         get_processor(wrapper).default or get_processor(wrapper).required]
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
        show_audio = is_audio(output)
        show_image = not show_audio
    return (gr.update(visible=show_audio, value=output if show_audio else None),
            gr.update(visible=show_image, value=output if show_image else None))


def update_preview_select(input_files: List[str]) -> Tuple[gr.update, gr.update, gr.update]:
    if not input_files:
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    audio_files = get_audio_files(input_files)
    image_files = get_image_files(input_files)
    first_audio = audio_files[0] if audio_files else None
    first_image = image_files[0] if image_files else None
    return (gr.update(choices=input_files, value=first_audio, visible=bool(audio_files) or bool(image_files)),
            gr.update(value=first_audio, visible=bool(audio_files)),
            gr.update(value=first_image, visible=bool(image_files)),
            )


def process(processors: List[str], inputs: List[str], progress=gr.Progress()) -> List[str]:
    start_time = datetime.now()
    settings = arg_handler.get_args()

    progress(0, f"Processing with {len(processors)} processors...")
    outputs = []
    all_outputs = []
    inputs = [ProjectFiles(file_path) for file_path in inputs]
    # Store the clone pitch shift value for the next processor
    clone_pitch_shift = settings.get("Clone", {}).get("pitch_shift", 0)
    for idx, processor_title in enumerate(processors):
        tgt_processor = get_processor(processor_title)
        processor_key = tgt_processor.title.replace(' ', '')
        processor_settings = settings.get(processor_key, {}) if settings else {}
        # If the processor is 'merge', set the pitch_shift to the last clone pitch shift
        if processor_title == 'Merge' or processor_title == "Export":
            processor_settings['pitch_shift'] = clone_pitch_shift
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
    first_image = output_image_files[0] if output_image_files else None
    first_audio = output_audio_files[0] if output_audio_files else None
    end_time = datetime.now()
    total_time_in_seconds = (end_time - start_time).total_seconds()
    logger.info(f"Processing complete with {len(processors)} processors in {total_time_in_seconds:.2f} seconds")
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
    for wrapper_name in wrappers:
        processor = get_processor(wrapper_name)
        processor.register_descriptions(arg_handler)
    music_register_descriptions(arg_handler)
    tts_register_descriptions(arg_handler)
    rvc_register_descriptions(arg_handler)
    zonos_register_descriptions(arg_handler)

    with open(project_root / 'css' / 'ui.css', 'r') as css_file:
        css = css_file.read()
        css = f'<style type="text/css">{css}</style>'
    # Load the contents of ./js/ui.js
    with open(project_root / 'js' / 'ui.js', 'r') as js_file:
        js = js_file.read()
        js += f"\n{arg_handler.get_descriptions_js()}"
        js = f'<script type="text/javascript">{js}</script>'
        js += f"\n{css}"

    with gr.Blocks(title='AudioLab', head=js, theme="d8ahazard/rd_blue") as ui:
        with gr.Tabs(selected="process"):
            with gr.Tab(label='Process', id="process"):
                gr.Markdown("## Music Processing")
                processor_list = gr.CheckboxGroup(label='Processors', choices=wrappers, value=enabled_wrappers,
                                                  elem_id='processor_list', key="main_processor_list")
                progress_display = gr.HTML(label='Progress', value='')

                accordions = []
                with gr.Row():
                    with gr.Column() as settings_ui:
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
                                input_url = gr.Textbox(label='Input URL', placeholder='Enter URL', visible=True,
                                                       interactive=True, key="process_input_url")
                            with gr.Column():
                                input_url_button = gr.Button(value='Load', visible=True, interactive=True)
                    with gr.Column():
                        gr.Markdown("### 🎶 Outputs")
                        with gr.Row():
                            start_processing = gr.Button(value='Start Processing', variant='primary')
                            cancel_processing = gr.Button(value='Cancel Processing', variant='secondary', visible=False)

                        output_select = gr.Dropdown(label='Select Output Preview', choices=[], value=None,
                                                    visible=False, interactive=True, key="process_output_preview")
                        output_audio = gr.Audio(label='Output Audio', value=None, visible=False,
                                                key="process_output_audio")
                        output_image = gr.Image(label='Output Image', value=None, visible=False,
                                                key="process_output_image")
                        output_files = gr.File(label='Output Files', file_count='multiple',
                                               file_types=['audio', 'video'],
                                               interactive=False, key="process_output_files")

                processor_list.input(
                    fn=enforce_defaults,
                    inputs=[processor_list],
                    outputs=[processor_list]
                )

                processor_list.change(
                    fn=lambda processors: toggle_visibility(processors, wrappers, accordions),
                    inputs=[processor_list],
                    outputs=[accordion for accordion in accordions]
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
            with gr.Tab(label="Train", id="train"):
                rvc_render()
            with gr.Tab(label="Music", id="music"):
                render_music(arg_handler)
            with gr.Tab(label='TTS', id="tts"):
                render_tts()
            with gr.Tab(label='Zonos', id="zonos"):
                render_zonos()

        tts_listen()
        music_listen()
        zonos_listen()
    # Launch the UI with specified host and port
    favicon_path = os.path.join(project_root, 'res', 'favicon.ico')
    ui.launch(server_name=server_name, server_port=server_port, favicon_path=favicon_path)
