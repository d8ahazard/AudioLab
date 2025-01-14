import importlib
import os
import sys
import traceback
from pathlib import Path
from typing import List

import gradio as gr
from torchaudio._extension import _init_dll_path

#from handlers.rvc_trainer import train_rvc_model
from layouts.rvc_infer import render
from layouts.rvc_train import render as rvc_render
from wrappers.audio_separator import AudioSeparator
from wrappers.base_wrapper import BaseWrapper
from handlers.config import model_path, output_path

if os.name == "nt" and (3, 8) <= sys.version_info < (3, 99):
    _init_dll_path()

# Stop caching models in limbo!!
hf_dir = os.path.join(model_path, "hf")
transformers_dir = os.path.join(model_path, "transformers")
os.makedirs(hf_dir, exist_ok=True)
os.makedirs(transformers_dir, exist_ok=True)
# Set HF_HUB_CACHE_DIR to the model_path
os.environ["HF_HOME"] = hf_dir

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
                        all_wrappers.append(module_name)
                        break
            except Exception as e:
                print(f"Error importing module {module_name}: {e}")
                traceback.print_exc()

    all_wrappers = sorted(all_wrappers, key=lambda x: get_processor(x).priority)
    print(f"Found {len(all_wrappers)} wrappers: {all_wrappers}")
    return all_wrappers


def get_processor(processor_name: str) -> BaseWrapper:
    module = importlib.import_module(f'wrappers.{processor_name}')
    for name, obj in module.__dict__.items():
        if isinstance(obj, type) and issubclass(obj, BaseWrapper) and obj is not BaseWrapper:
            return obj()
    return None


def toggle_visibility(processors: List[str], all_wrappers: List[str], all_accordions: List[gr.Accordion]):
    """
    Generate visibility updates for accordions based on the selected processors
    and whether the processor has allowed_kwargs.
    """
    return [
        gr.update(visible=(wrapper in processors and bool(get_processor(wrapper).allowed_kwargs)))
        for wrapper, accordion in zip(all_wrappers, all_accordions)
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
        if Path(file_path).suffix in ['.jpg', '.jpeg', '.png']:
            image_files.append(file_path)
    return image_files


def process(processors: List[str], inputs: List[str], progress=gr.Progress(), settings=None) -> List[str]:
    outputs = []

    def progress_callback(
            step: float | tuple[int, int | None] | None,
            desc: str | None = None,
            total: int | None = None
    ):
        progress(step, desc, total)

    progress(0, f"Processing with {len(processors)} processors...")
    all_outputs = []
    for processor_name in processors:
        tgt_processor = get_processor(processor_name)
        processor_key = tgt_processor.title.replace(' ', '')
        processor_settings = settings.get(processor_key, {}) if settings else {}
        # Log the keys in processor_settings
        print(f"Processor settings for {processor_name}:")
        print("---------------------------------------------------------------------")
        for key in processor_settings:
            print(f"{key}: {processor_settings[key]}")
        print("---------------------------------------------------------------------")
        outputs = tgt_processor.process_audio(inputs, progress_callback, **processor_settings)
        all_outputs.extend(outputs)
        inputs = outputs
    outputs = [output for output in all_outputs if os.path.exists(output)]
    # output_files, output_audio_select, output_audio_preview, output_image_preview, progress_display
    output_audio_files = get_audio_files(outputs)
    output_image_files = get_image_files(outputs)
    output_images_and_audio = output_audio_files + output_image_files
    first_image = output_image_files[0] if output_image_files else None
    first_audio = output_audio_files[0] if output_audio_files else None
    return (gr.update(value=outputs, visible=bool(outputs)),
            gr.update(value=first_audio, visible=bool(first_audio), choices=output_images_and_audio),
            gr.update(value=first_audio, visible=bool(first_audio)),
            gr.update(value=first_image, visible=bool(first_image)),
            gr.update(value=f"Processing complete with {len(processors)} processors."))


def separate_vocal(audio_files: List[str], progress=gr.Progress()) -> List[str]:
    progress(0, f"Separating vocals from {len(audio_files)} audio files...")
    separator = AudioSeparator()
    args = {
        "separate_stems": False,
        "bg_vocals_removal": "Nothing",
        "reverb_removal": "Nothing",
        "echo_removal": "Nothing",
        "delay_removal": "Nothing",
        "crowd_removal": "Nothing",
        "noise_removal": "Nothing",
        "delay_removal_model": "UVR-De-Echo-Normal.pth",
        "background_vocal_model": "UVR_MDXNET_KARA_2.onnx",
        "noise_removal_model": "UVR-DeNoise.pth",
        "crowd_removal_model": "UVR-MDX-NET_Crowd_HQ_1.onnx",
    }
    outputs = separator.process_audio(audio_files, progress, **args)
    existing_outputs = [output for output in outputs if os.path.exists(output)]
    vocal_outputs = [output for output in existing_outputs if
                     '(Vocals)' in output and "(BG Vocals)" not in output and "(Instrumental)" not in output]
    bg_vocal_outputs = [output for output in existing_outputs if '(BG Vocals)' in output]
    return vocal_outputs, bg_vocal_outputs


def start_training(
        inputs: List[str],
        separate_audio_first: bool,
        voice_name: str,
        total_epochs: int,
        batch_size: int,
        learning_rate: float,
        lr_decay: float,
        sampling_rate: str,
        progress=gr.Progress()
) -> str:
    log = f"Starting training with {len(inputs)} voice files for voice: {voice_name}.\n"

    def progress_callback(
            step: float | tuple[int, int | None] | None,
            desc: str | None = None,
            total: int | None = None
    ):
        progress(step, desc, total)

    #progress(0, f"Processing with {len(processors)} processors...")

    # Ensure target directory exists
    voice_dir = os.path.join(output_path, "voices", voice_name)
    target_dir = os.path.join(voice_dir, "raw")
    os.makedirs(target_dir, exist_ok=True)
    if separate_audio_first:
        log += "Separating vocals from audio files...\n"
        vocal_files, bg_vocal_files = separate_vocal(inputs, progress_callback)
        inputs = vocal_files
        log += f"Separated {len(inputs)} vocal files.\n"
    # Copy input files to the target directory
    for input_file in inputs:
        file_name = os.path.basename(input_file)
        target_file_path = os.path.join(target_dir, file_name.replace(' ', '_'))
        if not os.path.exists(target_file_path):
            with open(input_file, "rb") as src, open(target_file_path, "wb") as dst:
                dst.write(src.read())
        log += f"Copied {file_name} to {target_file_path}.\n"

    # Call train_rvc_model
    log += "Starting training process...\n"
    try:
        # train_rvc_model(
        #     trainset_dir=target_dir,
        #     exp_dir=voice_dir,
        #     voice_name=voice_name,
        #     sr=sampling_rate,
        #     total_epoch=total_epochs,
        #     batch_size=batch_size,
        #     lr=learning_rate,
        #     lr_decay=lr_decay,
        #
        # )
        log += "Training completed successfully.\n"
    except Exception as e:
        log += f"Error during training: {str(e)}\n"
        traceback.print_exc()

    return log


if __name__ == '__main__':
    wrappers = list_wrappers()
    arg_handler = BaseWrapper().arg_handler

    # Render the UI
    with gr.Blocks(title='AudioLab') as ui:
        with gr.Tabs():
            with gr.Tab(label='Process'):
                processor_list = gr.CheckboxGroup(label='Processors', choices=wrappers, value=wrappers)
                progress_display = gr.HTML(label='Progress', value='')

                # Settings UI
                accordions = []
                with gr.Row():
                    start_processing = gr.Button(value='Start Processing')
                    cancel_processing = gr.Button(value='Cancel Processing')

                with gr.Row():
                    with gr.Column() as settings_ui:
                        for wrapper_name in wrappers:
                            processor = get_processor(wrapper_name)
                            show_accordion = len(processor.allowed_kwargs) > 0
                            accordion = gr.Accordion(label=processor.title, visible=show_accordion, open=False)
                            with accordion:
                                processor.render_options(gr.Column())
                            accordions.append(accordion)

                    # Input and output sections
                    with gr.Column():
                        input_files = gr.File(label='Input Files', file_count='multiple', file_types=['audio', 'video'])
                    with gr.Column():
                        output_files = gr.File(label='Output Files', file_count='multiple',
                                               file_types=['audio', 'video'],
                                               interactive=False)
                        output_audio_select = gr.Dropdown(label='Output Preview', visible=False, interactive=True)
                        output_audio_preview = gr.Audio(label='Output Audio', visible=False)
                        output_image_preview = gr.Image(label='Output Image', visible=False)

                # Toggle visibility for accordions dynamically
                processor_list.change(
                    fn=lambda processors: toggle_visibility(processors, wrappers, accordions),
                    inputs=[processor_list],
                    outputs=[accordion for accordion in accordions]
                )

                start_processing.click(
                    fn=process,
                    inputs=[processor_list, input_files, gr.State(arg_handler.get_args())],
                    outputs=[output_files, output_audio_select, output_audio_preview, output_image_preview,
                             progress_display]
                )
            with gr.Tab(label='Clone'):
                render()
            with gr.Tab(label='Train'):
                rvc_render()
                # with gr.Row():
                #     start_training_button = gr.Button(value='Start Training')
                #     cancel_training_button = gr.Button(value='Cancel Training')
                # with gr.Row():
                #     with gr.Column():
                #         voice_name = gr.Textbox(label='Voice Name', placeholder='Enter the name for the voice.')
                #
                #         # Add training parameters as sliders
                #         total_epochs = gr.Slider(
                #             label='Total Epochs', minimum=1, maximum=100, step=1, value=20
                #         )
                #         batch_size = gr.Slider(
                #             label='Batch Size', minimum=1, maximum=64, step=1, value=8
                #         )
                #         learning_rate = gr.Slider(
                #             label='Learning Rate', minimum=1e-6, maximum=1e-2, step=1e-6, value=1.8e-4
                #         )
                #         lr_decay = gr.Slider(
                #             label='Learning Rate Decay', minimum=0.8, maximum=1.0, step=0.01, value=0.99
                #         )
                #         sampling_rate = gr.Dropdown(
                #             label='Sampling Rate',
                #             choices=['32k', '40k', '48k'],
                #             value='48k'
                #         )
                #         do_separate_audio_first = gr.Checkbox(label='Separate Audio First', value=True)
                #     with gr.Column():
                #         input_voice_files = gr.File(label='Input Voice Files', file_count='multiple',
                #                                     file_types=['audio'])
                #
                #     with gr.Column():
                #         output_log = gr.Textbox(label='Output Log', value='')
                #
                # # Connect the start_training function
                # start_training_button.click(
                #     fn=start_training,
                #     inputs=[
                #         input_voice_files,
                #         do_separate_audio_first,
                #         voice_name,
                #         total_epochs,
                #         batch_size,
                #         learning_rate,
                #         lr_decay,
                #         sampling_rate,
                #     ],
                #     outputs=[output_log]
                # )
    ui.launch()
