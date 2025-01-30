import os
import zipfile

import gradio as gr
import requests

from handlers.config import model_path
from modules.yue.inference.infer import generate_music

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
    files_to_check = ["hf_1_325000", "ckpt_00360000.pth", "config.yaml", "config_decoder.yaml", "decoder_131000.pth", "decoder_151000.pth", "tokenizer.model"]
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


def render():
    with gr.Blocks() as app:
        gr.Markdown("## 🎵 YuE Music Generation 🎵")

        with gr.Row():
            # Left Column - Settings
            with gr.Column():
                gr.Markdown("### 🔧 Settings")
                model_language = gr.Dropdown(
                    ["English", "Mandarin/Cantonese", "Japanese/Korean"],
                    value="English",
                    label="Model Language"
                )
                max_new_tokens = gr.Slider(500, 5000, value=3000, step=100, label="Max New Tokens")
                run_n_segments = gr.Slider(1, 10, value=2, step=1, label="Run N Segments")
                stage2_batch_size = gr.Slider(1, 8, value=4, step=1, label="Stage 2 Batch Size")
                keep_intermediate = gr.Checkbox(label="Keep Intermediate Files", value=False)
                disable_offload_model = gr.Checkbox(label="Disable Model Offloading", value=False)
                rescale = gr.Checkbox(label="Rescale Output")
                cuda_idx = gr.Number(value=0, label="CUDA Index")

            # Middle Column - Input Data
            with gr.Column():
                gr.Markdown("### 🎤 Input")
                genre_txt = gr.Textbox(
                    label="Genre Tags",
                    placeholder="e.g., uplifting pop airy vocal electronic bright",
                    lines=2
                )
                lyrics_txt = gr.Textbox(
                    label="Lyrics",
                    placeholder="Enter structured lyrics here... (Use [verse], [chorus] labels)",
                    lines=10
                )
                use_audio_prompt = gr.Checkbox(label="Use Audio Reference (ICL Mode)")
                with gr.Row():
                    audio_prompt_path = gr.File(label="Reference Audio File (Optional)")
                    prompt_start_time = gr.Number(value=0.0, label="Prompt Start Time (sec)")
                    prompt_end_time = gr.Number(value=30.0, label="Prompt End Time (sec)")

            # Right Column - Start & Outputs
            with gr.Column():
                gr.Markdown("### 🎶 Generate & Outputs")
                start_button = gr.Button("🚀 Generate Music")
                output_mix = gr.Audio(label="Final Mix")
                output_vocal = gr.Audio(label="Vocal Output")
                output_inst = gr.Audio(label="Instrumental Output")

                # Function to dynamically select the Stage 1 model
                def update_model_selection(model_language, use_audio_prompt):
                    model_type = "icl" if use_audio_prompt else "cot"
                    return STAGE1_MODELS[model_language][model_type]

                # Function to generate music
                def generate_callback(
                        model_language, use_audio_prompt, genre_txt, lyrics_txt,
                        audio_prompt_path, prompt_start_time, prompt_end_time,
                        max_new_tokens, run_n_segments, stage2_batch_size,
                        keep_intermediate, disable_offload_model, cuda_idx, rescale,
                        progress=gr.Progress(track_tqdm=True)
                ):
                    fetch_and_extxract_models()
                    stage1_model = update_model_selection(model_language, use_audio_prompt)
                    output_paths = generate_music(
                        stage1_model, "m-a-p/YuE-s2-1B-general", genre_txt, lyrics_txt, use_audio_prompt,
                        audio_prompt_path.name if audio_prompt_path else "",
                        prompt_start_time, prompt_end_time, max_new_tokens,
                        run_n_segments, stage2_batch_size, keep_intermediate,
                        disable_offload_model, cuda_idx, rescale,
                        top_p=0.93, temperature=1.0, repetition_penalty=1.2,
                        callback=progress
                    )
                    return output_paths

                # Start button click event
                start_button.click(
                    generate_callback,
                    inputs=[
                        model_language, use_audio_prompt, genre_txt, lyrics_txt,
                        audio_prompt_path, prompt_start_time, prompt_end_time,
                        max_new_tokens, run_n_segments, stage2_batch_size,
                        keep_intermediate, disable_offload_model, cuda_idx, rescale
                    ],
                    outputs=[output_mix, output_vocal, output_inst]
                )

    return app
