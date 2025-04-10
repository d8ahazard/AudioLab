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

from handlers.args import ArgHandler
from handlers.config import output_path, model_path
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
        
        # Training Tab (for future implementation)
        with gr.TabItem("Train", id="diffrythm_train"):
            gr.Markdown("# 🔬 DiffRhythm Model Training")
            gr.Markdown("This feature is planned for future releases. Currently, training requires a custom setup following the DiffRhythm repository instructions.")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🛠️ Training Resources")
                    gr.Markdown(
                        """
                        ### Training Requirements
                        
                        To train a custom DiffRhythm model, you'll need:
                        
                        1. **High-quality audio dataset** with properly formatted labels
                        2. **GPU with 24GB+ VRAM** (recommended)
                        3. **PyTorch environment** with accelerate and other dependencies
                        
                        ### Setup Instructions
                        
                        Please refer to the [DiffRhythm repository](https://github.com/ASLP-lab/DiffRhythm) for detailed training instructions:
                        
                        ```bash
                        # Example training command
                        bash scripts/train.sh
                        ```
                        
                        The training process requires careful dataset preparation and significant computing resources.
                        """
                    )
                
                with gr.Column():
                    gr.Markdown("### 📊 Dataset Format")
                    gr.Markdown(
                        """
                        ### Required Dataset Structure
                        
                        DiffRhythm training requires three types of data:
                        
                        1. **Audio latents**: VAE-encoded representation of audio
                        2. **Style embeddings**: MuQ-encoded style descriptors
                        3. **Lyrics tokens**: Tokenized lyrics with timestamps
                        
                        The dataset should be prepared with a structure like:
                        
                        ```
                        dataset/
                        ├── train.scp        # File index with paths
                        ├── latent/          # Audio latent files
                        ├── style/           # Style embedding files
                        └── lrc/             # Lyrics files
                        ```
                        
                        Each line in train.scp should contain:
                        `id|lrc_path|latent_path|style_path`
                        """
                    )
                
                with gr.Column():
                    gr.Markdown("### 🔮 Future Features")
                    gr.Markdown(
                        """
                        ### Coming in Future Updates
                        
                        We're working on implementing:
                        
                        1. **Simplified training interface** within AudioLab
                        2. **Dataset preparation tools** for custom songs
                        3. **Fine-tuning support** for customizing existing models
                        4. **Style transfer capabilities** for adapting new styles
                        5. **Model merging** to combine different model strengths
                        
                        Stay tuned for updates to the DiffRhythm module!
                        """
                    )
                    
                    notify_btn = gr.Button(
                        "Notify Me When Available",
                        elem_classes="hintitem",
                        elem_id="diffrythm_notify_btn"
                    )
                    
                    notify_email = gr.Textbox(
                        label="Email Address",
                        placeholder="your@email.com",
                        elem_classes="hintitem",
                        elem_id="diffrythm_notify_email",
                        visible=False
                    )
                    
                    notify_msg = gr.Markdown(visible=False)
                    
                    def toggle_notify():
                        return gr.update(visible=True), gr.update(visible=True)
                    
                    def submit_notify(email):
                        # This would actually save to a notification list in a real implementation
                        return gr.update(value="Thanks! We'll notify you when training features are available.", visible=True)
                    
                    notify_btn.click(toggle_notify, inputs=[], outputs=[notify_email, notify_msg])
                    notify_email.submit(submit_notify, inputs=[notify_email], outputs=[notify_msg])
    
    return gr.Markdown("DiffRhythm tabs initialized")

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
    """Register tooltips and descriptions for UI elements."""
    descriptions = {
        # Generate tab
        "diffrythm_model": "Select the DiffRhythm model to use. The 'full' model can generate longer songs but requires more VRAM.",
        "diffrythm_chunked": "Enable chunked decoding to reduce VRAM usage. Recommended for GPUs with less than 16GB VRAM.",
        "diffrythm_style_prompt": "Describe the musical style or mood for the generated song (e.g., 'Jazzy piano with emotional strings').",
        "diffrythm_use_ref_audio": "Use reference audio to guide the style of generation instead of a text prompt.",
        "diffrythm_ref_audio": "Upload a reference audio file to influence the style of the generated song.",
        "diffrythm_lyrics": "Lyrics in LRC format with timestamps in [MM:SS.xx] format. Leave empty for instrumental music.",
        "diffrythm_upload_lrc": "Upload an existing LRC file with formatted timestamps.",
        "diffrythm_example_lrc": "Load an example LRC file to see the correct formatting.",
        "diffrythm_generate_btn": "Start generating a song with the current settings.",
        "diffrythm_send_to_process": "Send the generated song to the Process tab for further processing.",
        "diffrythm_download": "Download the generated song.",
        "diffrythm_output_audio": "Preview the generated song.",
        "diffrythm_output_message": "Status messages about the generation process.",
        
        # Train tab (future)
        "diffrythm_notify_btn": "Get notified when training features become available.",
        "diffrythm_notify_email": "Your email address for notifications about new features."
    }
    
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