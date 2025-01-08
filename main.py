import importlib
import os
import sys
import traceback
from pathlib import Path

import gradio as gr
from wrappers.base_wrapper import BaseWrapper

# Add the project root (AudioLab) to the Python path
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
    print(f"Added {project_root} to sys.path")

audio_sr_path = project_root / 'modules' / 'versatile_audio_super_resolution' / 'audiosr'
if str(audio_sr_path) not in sys.path:
    sys.path.append(str(audio_sr_path))
    print(f"Added {audio_sr_path} to sys.path")


def list_wrappers():
    # Get the directory containing this script
    script_dir = project_root / 'wrappers'

    # Add the script directory to sys.path
    if str(script_dir) not in sys.path:
        sys.path.append(str(script_dir))

    # List all files that inherit from BaseWrapper
    all_wrappers = []
    for file in script_dir.iterdir():
        if file.suffix == '.py' and file.name != 'base_wrapper.py':
            module_name = file.stem
            try:
                module = importlib.import_module(f'wrappers.{module_name}')
                for name, obj in module.__dict__.items():
                    if isinstance(obj, type) and issubclass(obj, BaseWrapper) and obj is not BaseWrapper:
                        all_wrappers.append(module_name)
                        break
            except Exception as e:
                print(f"Error importing module {module_name}: {e}")
                traceback.print_exc()

    print(f"Found {len(all_wrappers)} wrappers: {all_wrappers}")
    return all_wrappers


if __name__ == '__main__':
    # List all files that inherit from BaseWrapper
    all_wrappers = list_wrappers()

    # Build the Gradio UI
    with gr.Blocks(title='FaceFusion') as ui:
        processor_list = gr.Dropdown(label='Processor', choices=all_wrappers)
        input_files = gr.File(label='Input Files', file_count='multiple', file_types=['audio', 'video'])
        output_files = gr.File(label='Output Files', file_count='multiple', file_types=['audio', 'video'],
                               interactive=False)

    # Render and serve the UI
    ui.launch()
