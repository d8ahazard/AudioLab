import json
import os
import random
import zipfile

import gradio as gr
import requests

from handlers.args import ArgHandler
from handlers.config import model_path
from modules.yue.inference.infer import generate_music
from modules.yue.inference.xcodec_mini_infer.utils.utils import seed_everything
import logging

logger = logging.getLogger(__name__)
SEND_TO_PROCESS_BUTTON: gr.Button = None
OUTPUT_MIX: gr.Audio = None
OUTPUT_FILES = []
arg_handler = ArgHandler()
# Language mapping for selecting the correct Stage 1 model
STAGE1_MODELS = {
    "English": {
        "cot": "m-a-p/YuE-s1-7B-anneal-en-cot",
        "icl": "m-a-p/YuE-s1-7B-anneal-en-icl"
    },
    "Mandarin/Cantonese": {
        "cot": "m-a-p/YuE-s1-7B-anneal-zh-cot",
        "icl": "m-a-p/YuE-s1-7B-anneal-zh-icl"
    },
    "Japanese/Korean": {
        "cot": "m-a-p/YuE-s1-7B-anneal-jp-kr-cot",
        "icl": "m-a-p/YuE-s1-7B-anneal-jp-kr-icl"
    }
}

base_model_url = "https://github.com/d8ahazard/AudioLab/releases/download/1.0.0/YuE_models.zip"


def fetch_and_extxract_models():
    model_dir = os.path.join(model_path, "YuE")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    files_to_check = ["hf_1_325000", "ckpt_00360000.pth", "config.yaml", "config_decoder.yaml", "decoder_131000.pth",
                      "decoder_151000.pth", "tokenizer.model"]
    if not all([os.path.exists(os.path.join(model_dir, f)) for f in files_to_check]):
        model_dl = os.path.join(model_dir, "YuE_models.zip")
        if os.path.exists(model_dl):
            os.remove(model_dl)
        with open(model_dl, "wb") as f:
            f.write(requests.get(base_model_url).content)
        with zipfile.ZipFile(os.path.join(model_dir, "YuE_models.zip"), 'r') as zip_ref:
            zip_ref.extractall(model_dir)
        # Delete the zip file
        os.remove(model_dl)


def download_output_files(output_files):
    """Create a zip file of all output files and return the path to download."""
    if not output_files or len(output_files) == 0:
        return None
    
    # Create a zip file with all the output files
    output_dir = os.path.dirname(output_files[0])
    zip_filename = os.path.join(output_dir, "music_outputs.zip")
    
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for file in output_files:
            if os.path.exists(file):
                # Add file to zip with just the filename, not the full path
                zipf.write(file, os.path.basename(file))
    
    return zip_filename


def update_output_preview(selected_file):
    """Update the output preview when a file is selected."""
    if not selected_file:
        return gr.update(value=None)
    return gr.update(value=selected_file)


def render(arg_handler: ArgHandler):
    global SEND_TO_PROCESS_BUTTON, OUTPUT_MIX, OUTPUT_FILES
    with gr.Blocks() as music:
        gr.Markdown("# 🎵 YuE Music Generation")
        gr.Markdown("Create complete music tracks with vocals and instrumentals from text descriptions. Generate original songs using lyrics, genre tags, and optional audio references.")

        with gr.Row():
            # Left Column - Settings
            with gr.Column():
                gr.Markdown("### 🔧 Settings")
                model_language = gr.Dropdown(
                    ["English", "Mandarin/Cantonese", "Japanese/Korean"],
                    value="English",
                    label="Model Language",
                    elem_classes="hintitem", elem_id="yue_model_language", key="yue_model_language"
                )
                max_new_tokens = gr.Slider(
                    500, 5000, value=3000, step=100,
                    label="Max New Tokens",
                    elem_classes="hintitem", elem_id="yue_max_new_tokens", key="yue_max_new_tokens"
                )
                run_n_segments = gr.Slider(
                    1, 10, value=2, step=1,
                    label="Run N Segments",
                    elem_classes="hintitem", elem_id="yue_run_n_segments", key="yue_run_n_segments"
                )
                stage2_batch_size = gr.Slider(
                    1, 8, value=4, step=1,
                    label="Stage 2 Batch Size",
                    elem_classes="hintitem", elem_id="yue_stage2_batch_size", key="yue_stage2_batch_size"
                )
                keep_intermediate = gr.Checkbox(
                    label="Keep Intermediate Files",
                    value=False,
                    elem_classes="hintitem", elem_id="yue_keep_intermediate", key="yue_keep_intermediate"
                )
                disable_offload_model = gr.Checkbox(
                    label="Disable Model Offloading",
                    value=False,
                    elem_classes="hintitem", elem_id="yue_disable_offload_model", key="yue_disable_offload_model"
                )
                rescale = gr.Checkbox(
                    label="Rescale Output",
                    elem_classes="hintitem", elem_id="yue_rescale", key="yue_rescale"
                )
                cuda_idx = gr.Number(
                    value=0, label="CUDA Index",
                    elem_classes="hintitem", elem_id="yue_cuda_idx", key="yue_cuda_idx"
                )
                seed = gr.Slider(
                    value=-1, label="Seed",
                    minimum=-1, maximum=4294967295, step=1,
                    elem_classes="hintitem", elem_id="yue_seed", key="yue_seed"
                )

            # Middle Column - Input Data
            with gr.Column():
                gr.Markdown("### 🎤 Inputs")
                genre_txt = gr.Textbox(
                    label="Genre Tags",
                    placeholder="e.g., uplifting pop airy vocal electronic bright",
                    lines=2,
                    elem_classes="hintitem", elem_id="yue_genre_txt", key="yue_genre_txt"
                )
                lyrics_txt = gr.Textbox(
                    label="Lyrics",
                    placeholder="Enter structured lyrics here... (Use [verse], [chorus] labels)",
                    lines=10,
                    elem_classes="hintitem", elem_id="yue_lyrics_txt", key="yue_lyrics_txt"
                )
                use_audio_prompt = gr.Checkbox(
                    label="Use Audio Reference (ICL Mode)",
                    elem_classes="hintitem", elem_id="yue_use_audio_prompt", key="yue_use_audio_prompt"
                )
                with gr.Row():
                    audio_prompt_path = gr.File(
                        label="Reference Audio File (Optional)",
                        elem_classes="hintitem", elem_id="yue_audio_prompt_path", key="yue_audio_prompt_path"
                    )
                    prompt_start_time = gr.Number(
                        value=0.0, label="Prompt Start Time (sec)",
                        elem_classes="hintitem", elem_id="yue_prompt_start_time", key="yue_prompt_start_time"
                    )
                    prompt_end_time = gr.Number(
                        value=30.0, label="Prompt End Time (sec)",
                        elem_classes="hintitem", elem_id="yue_prompt_end_time", key="yue_prompt_end_time"
                    )

            # Right Column - Start & Outputs
            with gr.Column():
                gr.Markdown("### 🎮 Actions")
                with gr.Row():
                    start_button = gr.Button(
                        "Generate Music",
                        elem_classes="hintitem",
                        elem_id="yue_start_button",
                        key="yue_start_button",
                        variant="primary"
                    )
                    download_button = gr.Button(
                        value="💾 Download All",
                        variant="secondary",
                        visible=False,
                        elem_classes="hintitem", 
                        elem_id="yue_download_button", 
                        key="yue_download_button"
                    )
                    SEND_TO_PROCESS_BUTTON = gr.Button(
                        value="Send to Process",
                        variant="secondary",
                        elem_classes="hintitem", elem_id="yue_send_to_process", key="yue_send_to_process"
                    )
                output_info = gr.Textbox(
                    label="Output Info",
                    value="",
                    max_lines=10,
                    elem_classes="hintitem", elem_id="yue_output_info", key="yue_output_info"
                )
                
                gr.Markdown("### 🎶 Outputs")
                output_selector = gr.Dropdown(
                    label="Select Output File",
                    choices=["Final Mix", "Vocals", "Instrumental"],
                    value="Final Mix",
                    visible=False,
                    elem_classes="hintitem", 
                    elem_id="yue_output_selector", 
                    key="yue_output_selector"
                )
                OUTPUT_MIX = gr.Audio(
                    label="Audio Preview",
                    elem_classes="hintitem", elem_id="yue_output_mix", key="yue_output_mix",
                    type="filepath",
                    sources=None,
                    interactive=False
                )
                output_list = gr.File(
                    label="All Generated Files",
                    file_count="multiple",
                    visible=False,
                    interactive=False,
                    elem_classes="hintitem", 
                    elem_id="yue_output_list", 
                    key="yue_output_list"
                )

                # Function to dynamically select the Stage 1 model
                def update_model_selection(model_language, use_audio_prompt):
                    model_type = "icl" if use_audio_prompt else "cot"
                    return STAGE1_MODELS[model_language][model_type]

                # Function to generate music
                def generate_callback(
                        model_language, use_audio_prompt, genre_txt, lyrics_txt,
                        audio_prompt_path, prompt_start_time, prompt_end_time,
                        max_new_tokens, run_n_segments, stage2_batch_size,
                        keep_intermediate, disable_offload_model, cuda_idx, rescale, seed,
                        progress=gr.Progress()
                ):
                    try:
                        global OUTPUT_FILES
                        # Calculate total steps: model setup + stage1 segments + stage2 processing + final processing
                        total_steps = 3 + run_n_segments + 2 + 2
                        current_step = 0

                        # Initial validation
                        if not any(char in lyrics_txt for char in ['[', ']']):
                            return [gr.update()] * 4 + [gr.update(value="Error: No [verse] or [chorus] labels found in lyrics")]

                        # Model setup
                        progress(current_step/total_steps, "Setting up models...")
                        fetch_and_extxract_models()
                        current_step += 1

                        if seed != -1:
                            seed_everything(seed)
                        else:
                            seed_everything(random.randint(0, 4294967295))
                        current_step += 1
                        
                        stage1_model = update_model_selection(model_language, use_audio_prompt)
                        current_step += 1
                        progress(current_step/total_steps, "Models initialized")
                        
                        def progress_callback(current, desc, total):
                            nonlocal current_step
                            # Map the inner progress to our overall progress
                            inner_progress = current/total if total > 0 else 0
                            effective_progress = (current_step + inner_progress)/total_steps
                            progress(effective_progress, desc)
                        
                        progress(current_step/total_steps, "Starting music generation...")
                        output_paths = generate_music(
                            stage1_model, "m-a-p/YuE-s2-1B-general", genre_txt, lyrics_txt, use_audio_prompt,
                            audio_prompt_path.name if audio_prompt_path else "",
                            prompt_start_time, prompt_end_time, max_new_tokens,
                            run_n_segments, stage2_batch_size, keep_intermediate,
                            disable_offload_model, cuda_idx, rescale,
                            top_p=0.93, temperature=1.0, repetition_penalty=1.2,
                            callback=progress_callback
                        )
                        
                        if not output_paths:
                            logger.error("No output paths returned from generate_music")
                            return [gr.update()] * 5 + [gr.update(value="Error: No output paths returned from generate_music")]
                        
                        # Validate outputs
                        current_step += 1
                        progress(current_step/total_steps, "Validating generated files...")
                        
                        # Log the returned paths to aid in debugging
                        logger.info(f"Output paths returned: {output_paths}")
                        
                        # Check if at least final output is available
                        if "final" not in output_paths and len(output_paths) == 0:
                            logger.error("No output paths returned")
                            return [gr.update()] * 5 + [gr.update(value="Error: No output files were generated")]
                        
                        # Make sure each path in output_paths exists
                        valid_outputs = {}
                        for key, path in output_paths.items():
                            if os.path.exists(path):
                                valid_outputs[key] = path
                            else:
                                logger.warning(f"Output file for '{key}' not found at path: {path}")
                        
                        # If we have no valid outputs, return an error
                        if not valid_outputs:
                            logger.error("No valid output files found")
                            return [gr.update()] * 5 + [gr.update(value="Error: No valid output files were generated")]
                        
                        # Prepare outputs, ensuring final exists
                        if "final" not in valid_outputs and len(valid_outputs) > 0:
                            # Use the first available output as final
                            first_key = next(iter(valid_outputs))
                            valid_outputs["final"] = valid_outputs[first_key]
                        
                        # Create defaults for missing outputs
                        final_output = valid_outputs.get("final", "")
                        vocal_output = valid_outputs.get("vocal", "")
                        instrumental_output = valid_outputs.get("instrumental", "")
                        
                        # Store paths in proper display format
                        display_outputs = []
                        output_map = {}
                        
                        if final_output:
                            display_outputs.append("Final Mix")
                            output_map["Final Mix"] = final_output
                        
                        if vocal_output:
                            display_outputs.append("Vocals")
                            output_map["Vocals"] = vocal_output
                        
                        if instrumental_output:
                            display_outputs.append("Instrumental")
                            output_map["Instrumental"] = instrumental_output
                        
                        # Store all valid output paths for download
                        OUTPUT_FILES = list(valid_outputs.values())
                        
                        # Success case
                        progress(1.0, "Generation complete!")
                        
                        # Return updated components
                        return [
                            gr.update(value=final_output),  # OUTPUT_MIX
                            gr.update(choices=display_outputs, value=display_outputs[0] if display_outputs else None, visible=True),  # output_selector
                            gr.update(value=OUTPUT_FILES, visible=len(OUTPUT_FILES) > 0),  # output_list
                            gr.update(visible=len(OUTPUT_FILES) > 0),  # download_button
                            gr.update(value="Generation complete!"),  # output_info
                            output_map  # hidden output map
                        ]
                    except Exception as e:
                        error_msg = str(e)
                        logger.exception("Error in music generation")
                        return [
                            gr.update(),  # OUTPUT_MIX
                            gr.update(visible=False),  # output_selector
                            gr.update(visible=False),  # output_list
                            gr.update(visible=False),  # download_button
                            gr.update(value=f"Error during music generation: {error_msg}"),  # output_info
                            {}  # empty output map
                        ]
                
                # Hidden component to store output_map
                output_map = gr.State({})

                # Start button click event
                start_button.click(
                    generate_callback,
                    inputs=[
                        model_language, use_audio_prompt, genre_txt, lyrics_txt,
                        audio_prompt_path, prompt_start_time, prompt_end_time,
                        max_new_tokens, run_n_segments, stage2_batch_size,
                        keep_intermediate, disable_offload_model, cuda_idx, rescale, seed
                    ],
                    outputs=[OUTPUT_MIX, output_selector, output_list, download_button, output_info, output_map]
                )
                
                # Output selector change event
                output_selector.change(
                    fn=lambda selection, output_map: gr.update(value=output_map.get(selection, None)),
                    inputs=[output_selector, output_map],
                    outputs=[OUTPUT_MIX]
                )
                
                # Download button click event
                download_button.click(
                    fn=download_output_files,
                    inputs=[output_list],
                    outputs=[gr.File(label="Download")]
                )

    return music


def listen():
    process_inputs = arg_handler.get_element("main", "process_inputs")
    if process_inputs:
        SEND_TO_PROCESS_BUTTON.click(fn=send_to_process, inputs=[OUTPUT_MIX, process_inputs], outputs=process_inputs)


def send_to_process(output_mix, process_inputs):
    if not output_mix or not os.path.exists(output_mix):
        return gr.update()
    if output_mix in process_inputs:
        return gr.update()
    process_inputs.append(output_mix)
    return gr.update(value=process_inputs)


def register_descriptions(arg_handler: ArgHandler):
    descriptions = {
        "model_language": "Select the language of the model to use for generation.",
        "use_audio_prompt": "Check this box if you want to use an audio reference for generation.",
        "genre_txt": "Enter genre tags to guide the music generation. Use spaces to separate multiple tags.",
        "lyrics_txt": "Enter structured lyrics with [verse], [chorus], [bridge] labels. Separate lines with newlines.",
        "audio_prompt_path": "Upload an audio file to use as a reference for generation.",
        "prompt_start_time": "Specify the start time in seconds for the audio prompt.",
        "prompt_end_time": "Specify the end time in seconds for the audio prompt.",
        "max_new_tokens": "Set the maximum number of tokens to generate.",
        "run_n_segments": "Specify how many segments to run during generation.",
        "stage2_batch_size": "Set the batch size for Stage 2 of generation.",
        "keep_intermediate": "Check this box to keep intermediate files generated during processing.",
        "disable_offload_model": "Check this box to disable model offloading and run everything on CPU.",
        "cuda_idx": "Specify the CUDA index to use for GPU processing.",
        "rescale": "Check this box to rescale the output audio files.",
        "seed": "Use -1 for random, or specify a seed for reproducibility."
    }
    for elem_id, description in descriptions.items():
        arg_handler.register_description("yue", elem_id, description)
