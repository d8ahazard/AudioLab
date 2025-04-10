"""
DiffRhythm UI Layout for AudioLab.
"""

import os
import logging
import torch
import gradio as gr
from pathlib import Path
import tempfile
import time
import zipfile
import shutil
import random
from typing import Optional, Dict, List
import json
from pydub import AudioSegment
import numpy as np
from muq import MuQMuLan
from handlers.args import ArgHandler
from handlers.config import output_path, model_path
from handlers.download import download_files
from layouts.rvc_train import separate_vocal
from layouts.transcribe import process_transcription
from modules.diffrythm.infer import (
    prepare_model, 
    get_lrc_token, 
    get_style_prompt, 
    get_negative_style_prompt, 
    get_reference_latent, 
    inference,
    check_download_model
)

from fastapi import UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

# Global variables for inter-tab communication
SEND_TO_PROCESS_BUTTON = None
OUTPUT_AUDIO = None
logger = logging.getLogger("ADLB.DiffRhythm")

# Available models and settings
AVAILABLE_MODELS = ["ASLP-lab/DiffRhythm-base", "ASLP-lab/DiffRhythm-full"]
MAX_LENGTHS = {"ASLP-lab/DiffRhythm-base": 95, "ASLP-lab/DiffRhythm-full": 285}

def download_output_files(output_files):
    """Create a zip file of all output files and return the path to download."""
    if not output_files or len(output_files) == 0:
        return None
    
    # Create a zip file with all the output files
    output_dir = os.path.dirname(output_files[0])
    zip_filename = os.path.join(output_dir, "diffrythm_outputs.zip")
    
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for file in output_files:
            if os.path.exists(file):
                # Add file to zip with just the filename, not the full path
                zipf.write(file, os.path.basename(file))
    
    return zip_filename

def list_diffrythm_projects():
    """List all available DiffRhythm projects in the output directory."""
    projects = []
    output_dir = os.path.join(output_path, "diffrythm_train")
    if os.path.exists(output_dir):
        projects = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    return projects

def convert_to_latents(audio_path, vae_model, muq_model):
    """Convert audio to latents and style embeddings"""
    try:
        # Load audio
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_channels(2)  # Convert to stereo
        audio = audio.set_frame_rate(44100)  # Convert to 44.1kHz
        
        # Convert to tensor
        audio_tensor = torch.tensor(audio.get_array_of_samples(), dtype=torch.float32)
        audio_tensor = audio_tensor.reshape(-1, 2)  # Reshape to [samples, channels]
        audio_tensor = audio_tensor / 32768.0  # Normalize to [-1, 1]
        
        # Convert to batch format [batch, channels, samples]
        audio_tensor = audio_tensor.transpose(0, 1).unsqueeze(0)
        
        # Get device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        # Move models to device
        vae_model = vae_model.to(device)
        muq_model = muq_model.to(device)
        
        # Convert audio to latents using VAE
        with torch.inference_mode():
            audio_tensor = audio_tensor.to(device, dtype)
            latents = vae_model.encode_export(audio_tensor)
            
            # Get style embeddings using MuQ
            style_emb = muq_model.get_music_embedding(audio_tensor, use_tensor=True)
            
        return latents.cpu().numpy(), style_emb.cpu().numpy()
        
    except Exception as e:
        logger.error(f"Error converting {audio_path} to latents: {e}")
        return None, None

def generate_song(
    lrc_path, 
    style_prompt, 
    ref_audio_path, 
    model_name, 
    chunked, 
    progress=gr.Progress(track_tqdm=True)
):
    """
    Generate a song using DiffRhythm.
    
    Args:
        lrc_path: Path to lyrics file
        style_prompt: Text prompt describing the music style
        ref_audio_path: Path to reference audio file
        model_name: Model to use for generation
        chunked: Whether to use chunked decoding
        progress: Progress tracker
        
    Returns:
        Tuple containing output audio path and message
    """
    try:
        # Set device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        progress(0.1, f"Using device: {device}")
        
        # Determine audio length based on model
        audio_length = MAX_LENGTHS.get(model_name, 95)
        max_frames = 2048 if audio_length == 95 else 6144
        
        # Check and download model if needed
        progress(0.15, f"Checking model availability: {model_name}...")
        model_path = check_download_model(model_name)
        if model_path:
            progress(0.2, f"Model found at: {model_path}")
        
        # Load models and run inference with appropriate precision
        using_half = device == "cuda"
        progress(0.25, f"Loading models to {device} with {'half' if using_half else 'full'} precision...")
        
        # Use mixed precision on CUDA to match the original implementation
        with torch.amp.autocast('cuda' if using_half else 'cpu'):
            cfm, tokenizer, muq, vae = prepare_model(max_frames, device, repo_id=model_name)
            
            # Process lyrics if provided
            if lrc_path and os.path.exists(lrc_path):
                progress(0.3, "Processing lyrics...")
                with open(lrc_path, "r", encoding="utf-8") as f:
                    lrc = f.read()
            else:
                progress(0.3, "No lyrics file provided. Generating instrumental music.")
                lrc = ""
            
            progress(0.4, "Preparing model inputs...")
            # Get tokens and other inputs for inference
            lrc_prompt, start_time = get_lrc_token(max_frames, lrc, tokenizer, device)
            
            # Style prompt handling
            if ref_audio_path and os.path.exists(ref_audio_path):
                progress(0.5, "Processing reference audio...")
                style_prompt_emb = get_style_prompt(muq, wav_path=ref_audio_path)
            else:
                progress(0.5, "Using text style prompt...")
                style_prompt_emb = get_style_prompt(muq, prompt=style_prompt)
            
            negative_style_prompt = get_negative_style_prompt(device)
            latent_prompt = get_reference_latent(device, max_frames)
            
            # Generate song
            progress(0.6, f"Generating a {audio_length}s song...")
            generated_song = inference(
                cfm_model=cfm,
                vae_model=vae,
                cond=latent_prompt,
                text=lrc_prompt,
                duration=max_frames,
                style_prompt=style_prompt_emb,
                negative_style_prompt=negative_style_prompt,
                start_time=start_time,
                chunked=chunked,
            )
        
        # Save output
        os.makedirs(os.path.join(output_path, "diffrythm"), exist_ok=True)
        timestamp = int(time.time())
        output_path_file = os.path.join(output_path, "diffrythm", f"diffrythm_output_{timestamp}.wav")
        
        progress(0.9, f"Saving output to {output_path_file}...")
        import torchaudio
        torchaudio.save(output_path_file, generated_song, sample_rate=44100)
        
        progress(1.0, "Generation complete!")
        return output_path_file, f"Generated a {audio_length}s song successfully!"
        
    except Exception as e:
        logger.error(f"Error generating song: {e}")
        import traceback
        traceback.print_exc()
        return None, f"Error: {str(e)}"

def send_to_process(file_to_send, existing_inputs):
    """
    Send the generated audio to the Process tab.
    
    Args:
        file_to_send: File path to send
        existing_inputs: Current process inputs
        
    Returns:
        Updated process inputs
    """
    if not file_to_send:
        return existing_inputs
    
    # If the existing_inputs is None, initialize it as an empty list
    if existing_inputs is None:
        existing_inputs = []
        
    # Add the file to the existing inputs
    existing_inputs.append(file_to_send)
    
    return existing_inputs

def render(arg_handler: ArgHandler):
    global SEND_TO_PROCESS_BUTTON, OUTPUT_AUDIO
    
    with gr.Tabs():
        # Inference Tab
        with gr.TabItem("Generate", id="diffrythm_generate"):
            gr.Markdown("# 🎵 DiffRhythm Song Generation")
            gr.Markdown("Generate full-length songs with lyrics and style prompts. Create high-quality stereo audio at 44.1kHz using the latent diffusion model.")
            
            with gr.Row():
                # Left Column - Settings
                with gr.Column():
                    gr.Markdown("### 🔧 Settings")
                    model_dropdown = gr.Dropdown(
                        choices=AVAILABLE_MODELS,
                        value="ASLP-lab/DiffRhythm-base",
                        label="Model",
                        elem_id="diffrythm_model",
                        elem_classes="hintitem"
                    )
                    
                    model_info = gr.Markdown(
                        """
                        **Model Capabilities:**
                        - DiffRhythm-base: 95s songs
                        - DiffRhythm-full: 285s songs
                        """
                    )
                    
                    chunked = gr.Checkbox(
                        label="Chunked Decoding (lower VRAM usage)",
                        value=True,
                        elem_classes="hintitem",
                        elem_id="diffrythm_chunked"
                    )
                    
                    style_prompt = gr.Textbox(
                        label="Style Prompt",
                        placeholder="e.g., Pop Emotional Piano, Jazzy Nightclub Vibe, Indie folk ballad",
                        lines=3,
                        elem_classes="hintitem",
                        elem_id="diffrythm_style_prompt"
                    )
                    
                    use_ref_audio = gr.Checkbox(
                        label="Use Reference Audio",
                        value=False,
                        elem_classes="hintitem",
                        elem_id="diffrythm_use_ref_audio"
                    )
                    
                    ref_audio_path = gr.Audio(
                        label="Reference Audio",
                        type="filepath",
                        visible=False,
                        elem_classes="hintitem",
                        elem_id="diffrythm_ref_audio"
                    )
                
                # Middle Column - Lyrics
                with gr.Column():
                    gr.Markdown("### 🎤 Lyrics Input")
                    lyrics_input = gr.Textbox(
                        label="Lyrics (LRC format)",
                        placeholder="Enter lyrics in LRC format with timestamps:\n[00:00.00]Verse one lyrics\n[00:15.00]More lyrics",
                        lines=15,
                        elem_classes="hintitem",
                        elem_id="diffrythm_lyrics"
                    )
                    
                    upload_lrc = gr.File(
                        label="Upload LRC File",
                        file_count="single",
                        file_types=[".lrc", ".txt"],
                        elem_classes="hintitem",
                        elem_id="diffrythm_upload_lrc"
                    )
                    
                    example_lrc = gr.Button(
                        "Load Example LRC",
                        elem_classes="hintitem",
                        elem_id="diffrythm_example_lrc"
                    )
                    
                    gr.Markdown(
                        """
                        **Note:**
                        - Lyrics must be in LRC format with timestamps
                        - LRC files contain timestamps in [MM:SS.xx] format
                        - For instrumental music, leave lyrics empty
                        """
                    )
                
                # Right Column - Actions & Output
                with gr.Column():
                    gr.Markdown("### 🎮 Actions")
                    with gr.Row():
                        generate_btn = gr.Button(
                            "Generate Song",
                            variant="primary",
                            elem_classes="hintitem",
                            elem_id="diffrythm_generate_btn"
                        )
                        
                        SEND_TO_PROCESS_BUTTON = gr.Button(
                            "Send to Process",
                            elem_classes="hintitem",
                            elem_id="diffrythm_send_to_process"
                        )
                        
                        download_btn = gr.Button(
                            "Download",
                            elem_classes="hintitem",
                            elem_id="diffrythm_download"
                        )
                    
                    gr.Markdown("### 🎶 Output")
                    OUTPUT_AUDIO = gr.Audio(
                        label="Generated Song",
                        type="filepath",
                        elem_classes="hintitem",
                        elem_id="diffrythm_output_audio"
                    )
                    
                    output_message = gr.Textbox(
                        label="Output Message",
                        elem_classes="hintitem",
                        elem_id="diffrythm_output_message"
                    )
            
            # Set up the event listeners
            def load_lrc_from_file(file):
                if not file:
                    return ""
                with open(file.name, "r", encoding="utf-8") as f:
                    return f.read()
            
            def load_example_lrc():
                example = """[00:00.00]This is an example LRC file
[00:05.00]Each line has a timestamp in brackets
[00:10.00]The format is [MM:SS.xx] with minutes, seconds, and centiseconds
[00:15.00]DiffRhythm uses these timestamps to align music with vocals
[00:20.00]You can create your own LRC files with any text editor
[00:25.00]Just add timestamps at the beginning of each line
[00:30.00]This helps create properly timed songs with lyrics
"""
                return example
            
            def toggle_ref_audio(use_ref):
                return gr.update(visible=use_ref)
            
            # Connect UI elements
            upload_lrc.change(load_lrc_from_file, inputs=[upload_lrc], outputs=[lyrics_input])
            example_lrc.click(load_example_lrc, inputs=[], outputs=[lyrics_input])
            use_ref_audio.change(toggle_ref_audio, inputs=[use_ref_audio], outputs=[ref_audio_path])
            
            # Handle temporary LRC file creation
            def save_lrc_to_temp(lyrics_text):
                if not lyrics_text or lyrics_text.strip() == "":
                    return None
                
                temp_dir = os.path.join(output_path, "diffrythm", "temp")
                os.makedirs(temp_dir, exist_ok=True)
                
                lrc_path = os.path.join(temp_dir, f"lyrics_{int(time.time())}.lrc")
                with open(lrc_path, "w", encoding="utf-8") as f:
                    f.write(lyrics_text)
                
                return lrc_path
            
            # Generation function that handles LRC creation
            def generate_with_lyrics(lyrics_text, style_prompt, ref_audio_path, model_name, chunked):
                # Save lyrics to a temporary file if provided
                lrc_path = save_lrc_to_temp(lyrics_text) if lyrics_text else None
                
                # Skip ref_audio if use_ref_audio is False
                actual_ref_path = ref_audio_path if use_ref_audio.value else None
                
                # Call the actual generation function
                return generate_song(lrc_path, style_prompt, actual_ref_path, model_name, chunked)
            
            # Connect the generate button
            generate_btn.click(
                fn=generate_with_lyrics,
                inputs=[lyrics_input, style_prompt, ref_audio_path, model_dropdown, chunked],
                outputs=[OUTPUT_AUDIO, output_message]
            )
            
            # Download button
            def prepare_download(audio_path):
                if not audio_path or not os.path.exists(audio_path):
                    return None
                return audio_path
            
            download_btn.click(
                fn=prepare_download,
                inputs=[OUTPUT_AUDIO],
                outputs=[gr.File(label="Download Generated Song")]
            )
        
        # Training Tab
        with gr.TabItem("Train", id="diffrythm_train"):
            gr.Markdown("# 🎵 DiffRhythm Model Training")
            gr.Markdown("Train custom DiffRhythm models with your own audio data. Features automatic vocal separation, transcription, and style extraction.")
            
            with gr.Row():
                # Left Column - Settings
                with gr.Column():
                    gr.Markdown("### 🔧 Settings")
                    project_name = gr.Textbox(
                        label="Project Name",
                        value="",
                        elem_classes="hintitem",
                        elem_id="diffrythm_project_name"
                    )
                    
                    with gr.Row():
                        existing_project = gr.Dropdown(
                            label="Existing Project",
                            choices=list_diffrythm_projects(),
                            value="",
                            elem_classes="hintitem",
                            elem_id="diffrythm_existing_project"
                        )
                        refresh_button = gr.Button(
                            "Refresh",
                            variant="secondary",
                            elem_classes="hintitem",
                            elem_id="diffrythm_refresh_button"
                        )
                    
                    base_model = gr.Dropdown(
                        label="Base Model",
                        choices=["ASLP-lab/DiffRhythm-base", "ASLP-lab/DiffRhythm-full"],
                        value="ASLP-lab/DiffRhythm-base",
                        elem_classes="hintitem",
                        elem_id="diffrythm_base_model"
                    )
                    
                    with gr.Accordion("Training Parameters", open=False):
                        batch_size = gr.Slider(
                            minimum=1,
                            maximum=32,
                            step=1,
                            label="Batch Size",
                            value=8,
                            elem_classes="hintitem",
                            elem_id="diffrythm_batch_size"
                        )
                        
                        epochs = gr.Slider(
                            minimum=1,
                            maximum=500,
                            step=1,
                            label="Training Epochs",
                            value=110,
                            elem_classes="hintitem",
                            elem_id="diffrythm_epochs"
                        )
                        
                        learning_rate = gr.Number(
                            label="Learning Rate",
                            value=7.5e-5,
                            elem_classes="hintitem",
                            elem_id="diffrythm_learning_rate"
                        )
                        
                        num_workers = gr.Slider(
                            minimum=1,
                            maximum=16,
                            step=1,
                            label="Number of Workers",
                            value=4,
                            elem_classes="hintitem",
                            elem_id="diffrythm_num_workers"
                        )
                        
                        save_steps = gr.Number(
                            label="Save Every N Steps",
                            value=5000,
                            elem_classes="hintitem",
                            elem_id="diffrythm_save_steps"
                        )
                        
                        warmup_steps = gr.Number(
                            label="Warmup Steps",
                            value=20,
                            elem_classes="hintitem",
                            elem_id="diffrythm_warmup_steps"
                        )
                        
                        max_grad_norm = gr.Number(
                            label="Max Gradient Norm",
                            value=1.0,
                            elem_classes="hintitem",
                            elem_id="diffrythm_max_grad_norm"
                        )
                        
                        grad_accumulation = gr.Number(
                            label="Gradient Accumulation Steps",
                            value=1,
                            elem_classes="hintitem",
                            elem_id="diffrythm_grad_accum"
                        )
                        
                        with gr.Row():
                            audio_drop_prob = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                step=0.1,
                                label="Audio Drop Probability",
                                value=0.3,
                                elem_classes="hintitem",
                                elem_id="diffrythm_audio_drop"
                            )
                            
                            style_drop_prob = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                step=0.1,
                                label="Style Drop Probability",
                                value=0.1,
                                elem_classes="hintitem",
                                elem_id="diffrythm_style_drop"
                            )
                            
                            lrc_drop_prob = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                step=0.1,
                                label="Lyrics Drop Probability",
                                value=0.1,
                                elem_classes="hintitem",
                                elem_id="diffrythm_lrc_drop"
                            )
                
                # Middle Column - Input
                with gr.Column():
                    gr.Markdown("### 🎵 Input")
                    input_files = gr.File(
                        label="Audio & Lyrics Files",
                        file_count="multiple",
                        file_types=["audio/*", "video/*", ".txt", ".lrc", ".json"],
                        elem_classes="hintitem",
                        elem_id="diffrythm_input_files"
                    )
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            input_url = gr.Textbox(
                                label="Audio URLs",
                                placeholder="Enter URLs separated by a new line",
                                elem_classes="hintitem",
                                elem_id="diffrythm_input_url"
                            )
                        with gr.Column():
                            input_url_button = gr.Button(
                                value='Load URLs',
                                variant='secondary',
                                elem_classes="hintitem",
                                elem_id="diffrythm_input_url_button"
                            )
                    
                    with gr.Row():
                        separate_vocals = gr.Checkbox(
                            label="Separate Vocals",
                            value=True,
                            elem_classes="hintitem",
                            elem_id="diffrythm_separate_vocals"
                        )
                        
                        transcribe_audio = gr.Checkbox(
                            label="Transcribe Audio",
                            value=True,
                            elem_classes="hintitem",
                            elem_id="diffrythm_transcribe_audio"
                        )
                
                # Right Column - Actions & Output
                with gr.Column():
                    gr.Markdown("### 🎮 Actions")
                    with gr.Row():
                        preprocess_btn = gr.Button(
                            "Preprocess Data",
                            variant="primary",
                        elem_classes="hintitem",
                            elem_id="diffrythm_preprocess"
                        )
                        
                        train_btn = gr.Button(
                            "Start Training",
                            variant="primary",
                            elem_classes="hintitem",
                            elem_id="diffrythm_train_btn"
                        )
                        
                        cancel_btn = gr.Button(
                            "Cancel",
                            variant="secondary",
                            visible=False,
                            elem_classes="hintitem",
                            elem_id="diffrythm_cancel"
                        )
                    
                    gr.Markdown("### 📊 Output")
                    info_box = gr.Textbox(
                        label="Status",
                        value="Ready to process",
                        elem_classes="hintitem",
                        elem_id="diffrythm_status"
                    )
                    
                    model_output = gr.File(
                        label="Trained Model",
                        file_count="single",
                        interactive=False,
                        visible=False,
                        elem_classes="hintitem",
                        elem_id="diffrythm_model_output"
                    )
            
            # Event handlers
            def list_projects():
                projects_dir = os.path.join(output_path, "diffrythm_train")
                if not os.path.exists(projects_dir):
                    return []
                return [""] + [d for d in os.listdir(projects_dir) 
                             if os.path.isdir(os.path.join(projects_dir, d))]
            
            refresh_button.click(
                fn=list_projects,
                outputs=[existing_project]
            )
            
            def update_time_info(input_files):
                if not input_files:
                    return gr.update(value="")
                total_length = 0
                for f in input_files:
                    try:
                        audio = AudioSegment.from_file(f)
                        total_length += len(audio) / 1000
                    except Exception as e:
                        logger.error(f"Error processing file {f}: {e}")
                total_length /= 60
                return gr.update(
                    value=f"Total length of input files: {total_length:.2f} minutes.\nRecommended is 30-60 minutes."
                )
            
            input_files.change(
                fn=update_time_info,
                inputs=[input_files],
                outputs=[info_box]
            )
            
            input_url_button.click(
                fn=download_files,
                inputs=[input_url, input_files],
                outputs=[input_files]
            )
            
            def preprocess_data(
                project_name,
                existing_project,
                input_files,
                separate_vocals,
                transcribe_audio,
                progress=gr.Progress()
            ):
                try:
                    if not project_name and not existing_project:
                        return "Please provide a project name"
                    
                    if project_name and existing_project:
                        return "Please provide only one project name"
                    
                    if not input_files:
                        return "Please provide input files"
                    
                    # Use existing project name if provided
                    if existing_project:
                        project_name = existing_project
                    
                    # Create project directory
                    project_dir = os.path.join(output_path, "diffrythm_train", project_name)
                    os.makedirs(project_dir, exist_ok=True)
                    
                    # Create dataset directories
                    latent_dir = os.path.join(project_dir, "latent")
                    style_dir = os.path.join(project_dir, "style")
                    lrc_dir = os.path.join(project_dir, "lrc")
                    raw_dir = os.path.join(project_dir, "raw")
                    
                    for d in [latent_dir, style_dir, lrc_dir, raw_dir]:
                        os.makedirs(d, exist_ok=True)
                    
                    progress(0.1, "Processing input files...")
                    
                    # Load models for latent conversion
                    progress(0.15, "Loading models...")
                    vae_ckpt_path = check_download_model(repo_id="ASLP-lab/DiffRhythm-vae")
                    vae = torch.jit.load(vae_ckpt_path)
                    
                    muq_model_dir = os.path.join(model_path, "diffrythm", "muq")
                    os.makedirs(muq_model_dir, exist_ok=True)
                    muq = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large", cache_dir=muq_model_dir)
                    
                    # Process each input file
                    processed_files = []
                    for idx, f in enumerate(input_files):
                        try:
                            progress((idx + 1) / len(input_files), 
                                   f"Processing file {idx + 1}/{len(input_files)}")
                            
                            # Get base name without extension
                            base_name = os.path.splitext(os.path.basename(f))[0]
                            
                            # Copy original file to raw directory
                            raw_path = os.path.join(raw_dir, f"{base_name}.wav")
                            if not os.path.exists(raw_path):
                                shutil.copy2(f, raw_path)
                            
                            # Separate vocals if requested
                            if separate_vocals:
                                progress((idx + 1) / len(input_files), 
                                       f"Separating vocals for {base_name}")
                                vocal_outputs, _ = separate_vocal([raw_path])
                                if vocal_outputs:
                                    raw_path = vocal_outputs[0]
                            
                            # Transcribe if requested and no matching lyrics file
                            if transcribe_audio:
                                matching_lyrics = None
                                if input_files:
                                    for lf in input_files:
                                        if os.path.splitext(os.path.basename(lf))[0] == base_name:
                                            matching_lyrics = lf
                                            break
                                
                                if not matching_lyrics:
                                    progress((idx + 1) / len(input_files), 
                                           f"Transcribing {base_name}")
                                    transcription = process_transcription(
                                        [raw_path],
                                        language="auto",
                                        align_output=True,
                                        assign_speakers=False
                                    )
                                    if transcription[1]:  # If transcription files were created
                                        # Find the .txt file
                                        txt_file = [f for f in transcription[1] 
                                                  if f.endswith('.txt')][0]
                                        matching_lyrics = txt_file
                            
                            # Convert audio to latents and style embeddings
                            progress((idx + 1) / len(input_files), 
                                   f"Converting {base_name} to latents")
                            
                            latents, style_emb = convert_to_latents(raw_path, vae, muq)
                            if latents is not None and style_emb is not None:
                                # Save latents and style embeddings
                                latent_path = os.path.join(latent_dir, f"{base_name}.npy")
                                style_path = os.path.join(style_dir, f"{base_name}.npy")
                                lrc_path = os.path.join(lrc_dir, f"{base_name}.lrc")
                                
                                np.save(latent_path, latents)
                                np.save(style_path, style_emb)
                                
                                # Copy or create LRC file
                                if matching_lyrics:
                                    shutil.copy2(matching_lyrics, lrc_path)
                                
                                processed_files.append(
                                    f"{raw_path}|{lrc_path}|{latent_path}|{style_path}"
                                )
                            
                        except Exception as e:
                            logger.error(f"Error processing file {f}: {e}")
                            continue
                    
                    # Create train.scp file
                    scp_path = os.path.join(project_dir, "train.scp")
                    with open(scp_path, "w") as f:
                        f.write("\n".join(processed_files))
                    
                    return "Preprocessing complete! Ready to start training."
                    
                except Exception as e:
                    logger.error(f"Error during preprocessing: {e}")
                    return f"Error during preprocessing: {str(e)}"
            
            preprocess_btn.click(
                fn=preprocess_data,
                inputs=[
                    project_name,
                    existing_project,
                    input_files,
                    separate_vocals,
                    transcribe_audio
                ],
                outputs=[info_box]
            )
            
            def start_training(
                project_name,
                existing_project,
                base_model,
                batch_size,
                epochs,
                learning_rate,
                num_workers,
                save_steps,
                warmup_steps,
                max_grad_norm,
                grad_accumulation,
                audio_drop_prob,
                style_drop_prob,
                lrc_drop_prob,
                progress=gr.Progress()
            ):
                try:
                    if not project_name and not existing_project:
                        return "Please provide a project name"
                    
                    if project_name and existing_project:
                        return "Please provide only one project name"
                    
                    # Use existing project name if provided
                    if existing_project:
                        project_name = existing_project
                    
                    # Check if project exists and has been preprocessed
                    project_dir = os.path.join(output_path, "diffrythm_train", project_name)
                    train_scp = os.path.join(project_dir, "train.scp")
                    
                    if not os.path.exists(project_dir) or not os.path.exists(train_scp):
                        return "Project not found or data not preprocessed. Please preprocess data first."
                    
                    # Import training function
                    from modules.diffrythm.train.train import TrainingArgs, train
                    
                    # Create training arguments
                    args = TrainingArgs(
                        project_dir=project_dir,
                        base_model=base_model,
                        batch_size=batch_size,
                        epochs=epochs,
                        learning_rate=learning_rate,
                        num_workers=num_workers,
                        save_steps=save_steps,
                        warmup_steps=warmup_steps,
                        max_grad_norm=max_grad_norm,
                        grad_accumulation=grad_accumulation,
                        audio_drop_prob=audio_drop_prob,
                        style_drop_prob=style_drop_prob,
                        lrc_drop_prob=lrc_drop_prob,
                        max_frames=2048,  # Fixed for now, could be made configurable
                        grad_ckpt=True,   # Enable gradient checkpointing by default
                        reset_lr=False,   # Don't reset learning rate by default
                        resumable_with_seed=42  # Fixed seed for reproducibility
                    )
                    
                    # Start training with progress updates
                    final_model_path = train(args, progress=progress)
                    
                    if final_model_path and os.path.exists(final_model_path):
                        return f"Training complete! Model saved to {final_model_path}"
                    else:
                        return "Training completed but model save failed."
                    
                except Exception as e:
                    logger.error(f"Error during training: {e}")
                    import traceback
                    traceback.print_exc()
                    return f"Error during training: {str(e)}"
            
            train_btn.click(
                fn=start_training,
                inputs=[
                    project_name,
                    existing_project,
                    base_model,
                    batch_size,
                    epochs,
                    learning_rate,
                    num_workers,
                    save_steps,
                    warmup_steps,
                    max_grad_norm,
                    grad_accumulation,
                    audio_drop_prob,
                    style_drop_prob,
                    lrc_drop_prob
                ],
                outputs=[info_box]
            )

def listen():
    """Set up event listeners for inter-tab communication."""
    global SEND_TO_PROCESS_BUTTON, OUTPUT_AUDIO
    
    arg_handler = ArgHandler()
    process_inputs = arg_handler.get_element("main", "process_inputs")
    
    if process_inputs and SEND_TO_PROCESS_BUTTON:
        SEND_TO_PROCESS_BUTTON.click(
            fn=send_to_process,
            inputs=[OUTPUT_AUDIO, process_inputs],
            outputs=[process_inputs]
        )

def register_descriptions(arg_handler: ArgHandler):
    """Register descriptions for UI elements"""
    descriptions = {
        # Generate tab
        "diffrythm_model": "Select the DiffRhythm model to use. The 'full' model can generate longer songs (up to 285s) but requires more VRAM. The 'base' model generates up to 95s.",
        "diffrythm_chunked": "Enable chunked decoding to reduce VRAM usage. Recommended for GPUs with less than 16GB VRAM. May slightly increase generation time.",
        "diffrythm_style_prompt": """Describe the musical style or mood for the generated song. Be specific and detailed.
Examples:
- "Upbeat pop song with electronic beats and emotional piano melody"
- "Slow jazz ballad with smooth saxophone and light drum brushes"
- "Epic orchestral piece with dramatic strings and powerful percussion"
""",
        "diffrythm_use_ref_audio": "Use a reference audio file to guide the style instead of text. The model will try to match the musical style of the reference.",
        "diffrythm_ref_audio": "Upload a reference audio file (WAV/MP3). The model will analyze this to influence the style of the generated song.",
        "diffrythm_lyrics": """Enter lyrics with timestamps in LRC format. Each line should start with a timestamp in [MM:SS.xx] format.

Example format:
[00:00.00] First line of lyrics
[00:05.50] Second line here
[00:10.75] Another line

Tips:
- Timestamps must be in [MM:SS.xx] format
- Each line must start with a timestamp
- Keep lines reasonably short
- Leave empty for instrumental music""",
        "diffrythm_upload_lrc": "Upload an existing LRC file. Must contain properly formatted timestamps.",
        "diffrythm_example_lrc": "Load an example LRC file to see the correct formatting.",
        "diffrythm_generate_btn": "Start generating a song with the current settings. This may take several minutes.",
        "diffrythm_send_to_process": "Send the generated song to the Process tab for further editing or effects.",
        "diffrythm_download": "Download the generated song as a WAV file.",
        "diffrythm_output_audio": "Preview the generated song. Click to play/pause.",
        "diffrythm_output_message": "Status messages and error reports from the generation process.",
        
        # Train tab
        "diffrythm_project_name": "Enter a unique name for your custom DiffRhythm model project. This will be used to identify your model and its training data.",
        "diffrythm_existing_project": "Select an existing project to continue training or view results.",
        "diffrythm_refresh_button": "Refresh the list of available projects.",
        "diffrythm_base_model": "Select the base DiffRhythm model to start from. The 'full' model supports longer songs but requires more training time and VRAM.",
        "diffrythm_input_files": """Upload your training data files. Supported formats:

Audio Files:
- WAV, MP3, FLAC, etc. (will be converted to WAV)
- Recommended length: 30-60 minutes total
- Should be high quality, clean recordings

Lyrics Files (optional):
- LRC format (.lrc): Contains timestamped lyrics
- Text format (.txt): Will be auto-transcribed if enabled
- JSON format: Custom format with timing information

File Naming:
- For manual lyrics, name the lyric file same as audio file
- Example: song.mp3 and song.lrc""",
        "diffrythm_input_url": "Enter URLs to download audio files. One URL per line. Supports YouTube, SoundCloud, and direct audio links.",
        "diffrythm_input_url_button": "Download the audio files from the provided URLs.",
        "diffrythm_separate_vocals": "Automatically separate vocals from instrumental tracks. Recommended for cleaner training data.",
        "diffrythm_transcribe_audio": "Automatically transcribe lyrics and create LRC files if no lyrics file is provided.",
        "diffrythm_batch_size": "Number of samples processed in each training step. Larger values use more VRAM but train faster. Reduce if running out of memory.",
        "diffrythm_epochs": "Number of complete passes through the training data. More epochs = better results but longer training time.",
        "diffrythm_learning_rate": "Controls how quickly the model learns. Default (7.5e-5) works well for most cases.",
        "diffrythm_num_workers": "Number of CPU threads for data loading. Set to number of CPU cores - 2 for best performance.",
        "diffrythm_save_steps": "Save a checkpoint every N steps. Use more frequent saves (lower number) for longer training runs.",
        "diffrythm_warmup_steps": "Number of steps to gradually increase learning rate. Helps stabilize early training.",
        "diffrythm_max_grad_norm": "Maximum gradient value for training stability. Reduce if training becomes unstable.",
        "diffrythm_grad_accum": "Number of batches to accumulate before updating. Increase to simulate larger batch size with less VRAM.",
        "diffrythm_audio_drop": "Probability of training without audio conditioning. Higher values = more robust generation.",
        "diffrythm_style_drop": "Probability of training without style conditioning. Higher values = more style flexibility.",
        "diffrythm_lrc_drop": "Probability of training without lyrics. Higher values = better instrumental generation.",
        "diffrythm_preprocess": "Prepare your data for training. This will separate vocals, transcribe lyrics, and convert audio to the required format.",
        "diffrythm_train_btn": "Start training your custom model. Training time depends on data size and epochs.",
        "diffrythm_cancel": "Stop the current training process. Note: This may leave the model in an inconsistent state.",
        "diffrythm_status": "Current status of preprocessing/training and any error messages.",
        "diffrythm_model_output": "Download your trained model for use in generation."
    }
    
    # Register all descriptions
    for elem_id, description in descriptions.items():
        arg_handler.register_description("diffrythm", elem_id, description)

def register_api_endpoints(api):
    """Register API endpoints for DiffRhythm."""
    @api.post("/api/v1/diffrythm/generate")
    async def api_generate_song(
        background_tasks: BackgroundTasks,
        style_prompt: str = Form(""),
        model_name: str = Form("ASLP-lab/DiffRhythm-base"),
        lyrics: Optional[str] = Form(None),
        reference_audio: Optional[UploadFile] = File(None),
        chunked: bool = Form(True)
    ):
        """
        Generate a song using DiffRhythm
        
        Args:
            background_tasks: FastAPI background tasks
            style_prompt: Text description of the music style
            model_name: DiffRhythm model to use
            lyrics: Lyrics in LRC format (optional)
            reference_audio: Audio file to use as style reference (optional)
            chunked: Whether to use chunked decoding
            
        Returns:
            Path to the generated audio file
        """
        try:
            # Create temporary directory for processing
            temp_dir = tempfile.mkdtemp(prefix="diffrythm_")
            
            # Save reference audio if provided
            ref_audio_path = None
            if reference_audio:
                ref_audio_path = os.path.join(temp_dir, "reference.wav")
                with open(ref_audio_path, "wb") as f:
                    f.write(await reference_audio.read())
            
            # Save lyrics if provided
            lrc_path = None
            if lyrics:
                lrc_path = os.path.join(temp_dir, "lyrics.lrc")
                with open(lrc_path, "w", encoding="utf-8") as f:
                    f.write(lyrics)
            
            # Generate song
            output_path, message = generate_song(
                lrc_path=lrc_path,
                style_prompt=style_prompt,
                ref_audio_path=ref_audio_path,
                model_name=model_name,
                chunked=chunked
            )
            
            if not output_path:
                raise HTTPException(status_code=500, detail=f"Failed to generate song: {message}")
            
            # Prepare response
            filename = os.path.basename(output_path)
            copied_path = os.path.join(temp_dir, filename)
            shutil.copy(output_path, copied_path)
            
            # Clean up temporary files after sending the response
            background_tasks.add_task(cleanup_temp_files, temp_dir)
            
            return FileResponse(
                output_path,
                media_type="audio/wav",
                filename=filename
            )
            
        except Exception as e:
            logger.exception("Error in DiffRhythm song generation:")
            raise HTTPException(status_code=500, detail=f"Song generation error: {str(e)}")

def cleanup_temp_files(temp_dir):
    """Clean up temporary files after a delay."""
    try:
        time.sleep(300)  # Wait 5 minutes before cleaning up
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    except Exception as e:
        logger.error(f"Error cleaning up temporary files: {e}") 