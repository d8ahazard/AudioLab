"""
ACE-Step UI Layout for AudioLab.
"""

import logging
import os
import traceback
                    
import torch
import time
import zipfile
import gradio as gr
import time
                

from handlers.args import ArgHandler
from handlers.config import output_path, model_path
from modules.acestep.process import (
    process,
    process_lora,
    process_retake,
    process_repaint,
    process_edit,
    DEFAULT_MODEL
)

# Global variables for inter-tab communication
SEND_TO_PROCESS_BUTTON = None
OUTPUT_AUDIO = None
logger = logging.getLogger("ADLB.ACEStep")

# Available models and settings
AVAILABLE_MODELS = ["ACE-Step/ACE-Step-v1-3.5B"]
DEFAULT_SCHEDULERS = ["euler", "heun", "pingpong"]
DEFAULT_CFG_TYPES = ["cfg", "apg", "cfg_star"]
AVAILABLE_LORAS = ["ACE-Step/ACE-Step-v1-chinese-rap-LoRA"]

# Determine if CUDA is available
CUDA_AVAILABLE = torch.cuda.is_available()
GPU_COUNT = torch.cuda.device_count() if CUDA_AVAILABLE else 0
GPU_NAMES = [torch.cuda.get_device_name(i) for i in range(GPU_COUNT)] if CUDA_AVAILABLE else []

def download_output_files(output_files):
    """Create a zip file of all output files and return the path to download."""
    if not output_files or len(output_files) == 0:
        return None

    # Create a zip file with all the output files
    output_dir = os.path.dirname(output_files[0])
    zip_filename = os.path.join(output_dir, "acestep_outputs.zip")

    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for file in output_files:
            if os.path.exists(file):
                # Add file to zip with just the filename, not the full path
                zipf.write(file, os.path.basename(file))

    return zip_filename

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
        # Main Generation Tab
        with gr.TabItem("Generate", id="acestep_generate"):
            gr.Markdown("# 🎵 ACE-Step: Music Generation")
            gr.Markdown(
                "Generate music and songs with vocals using ACE-Step, a foundation model for high-quality music generation. "
                "Create up to 4 minutes of music in seconds with text prompts and optional lyrics."
            )

            with gr.Row():
                # Left Column - Settings
                with gr.Column():
                    gr.Markdown("### 🔧 Settings")
                    
                    # LoRA checkbox to show/hide LoRA options
                    use_lora = gr.Checkbox(
                        label="Use LoRA Model (specialized models like RapMachine)",
                        value=False,
                        elem_id="acestep_use_lora",
                        elem_classes="hintitem"
                    )
                    
                    # Base model dropdown (always visible)
                    model_dropdown = gr.Dropdown(
                        choices=AVAILABLE_MODELS,
                        value=DEFAULT_MODEL,
                        label="Base Model",
                        elem_id="acestep_model",
                        elem_classes="hintitem"
                    )
                    
                    # LoRA model dropdown (only visible when use_lora is checked)
                    lora_model_dropdown = gr.Dropdown(
                        choices=AVAILABLE_LORAS,
                        value=AVAILABLE_LORAS[0] if AVAILABLE_LORAS else None,
                        label="LoRA Model",
                        elem_id="acestep_lora_model",
                        elem_classes="hintitem",
                        visible=False
                    )

                    with gr.Row():
                        audio_duration = gr.Slider(
                            minimum=15,
                            maximum=240,
                            value=60,
                            step=15,
                            label="Duration (seconds)",
                            elem_id="acestep_duration",
                            elem_classes="hintitem"
                        )
                        
                        seed_input = gr.Number(
                            label="Seed (leave empty for random)",
                            value=None,
                            precision=0,
                            elem_id="acestep_seed",
                            elem_classes="hintitem"
                        )

                    with gr.Accordion("Advanced Settings", open=False):
                        with gr.Row():
                            device_id = gr.Dropdown(
                                choices=[f"{i}: {name}" for i, name in enumerate(GPU_NAMES)] if GPU_NAMES else ["0: CPU"],
                                value="0: " + (GPU_NAMES[0] if GPU_NAMES else "CPU"),
                                label="Device",
                                elem_id="acestep_device",
                                elem_classes="hintitem"
                            )
                            
                            bf16 = gr.Checkbox(
                                label="Use BF16 Precision",
                                value=True,
                                elem_id="acestep_bf16",
                                elem_classes="hintitem"
                            )

                        with gr.Row():
                            torch_compile = gr.Checkbox(
                                label="Torch Compile (faster but uses more VRAM)",
                                value=False,
                                elem_id="acestep_torch_compile",
                                elem_classes="hintitem"
                            )
                            
                            cpu_offload = gr.Checkbox(
                                label="CPU Offload (less VRAM)",
                                value=False,
                                elem_id="acestep_cpu_offload",
                                elem_classes="hintitem"
                            )
                            
                        with gr.Row():
                            overlapped_decode = gr.Checkbox(
                                label="Overlapped Decoding",
                                value=False,
                                elem_id="acestep_overlapped_decode",
                                elem_classes="hintitem"
                            )
                            
                        # Add Low VRAM Mode checkbox
                        with gr.Row():
                            low_vram_mode = gr.Checkbox(
                                label="Low VRAM Mode (8GB or less)",
                                value=False,
                                elem_id="acestep_low_vram_mode",
                                elem_classes="hintitem"
                            )

                    with gr.Accordion("Generation Parameters", open=False):
                        with gr.Row():
                            infer_step = gr.Slider(
                                minimum=20,
                                maximum=100,
                                value=27,
                                step=1,
                                label="Inference Steps",
                                elem_id="acestep_infer_step",
                                elem_classes="hintitem"
                            )
                            
                            guidance_scale = gr.Slider(
                                minimum=1.0,
                                maximum=15.0,
                                value=7.5,
                                step=0.5,
                                label="Guidance Scale",
                                elem_id="acestep_guidance_scale",
                                elem_classes="hintitem"
                            )
                            
                        with gr.Row():
                            scheduler_type = gr.Radio(
                                choices=DEFAULT_SCHEDULERS,
                                value="euler",
                                label="Scheduler Type",
                                elem_id="acestep_scheduler",
                                elem_classes="hintitem",
                                info="Scheduler type for the generation. euler is recommended. heun will take more time. pingpong use SDE"
                            )
                            
                            cfg_type = gr.Radio(
                                choices=DEFAULT_CFG_TYPES,
                                value="apg",
                                label="CFG Type",
                                elem_id="acestep_cfg_type",
                                elem_classes="hintitem",
                                info="CFG type for the generation. apg is recommended. cfg and cfg_star are almost the same."
                            )
                            
                        with gr.Row():
                            omega_scale = gr.Slider(
                                minimum=1.0,
                                maximum=20.0,
                                value=10.0,
                                step=0.5,
                                label="Omega Scale",
                                elem_id="acestep_omega_scale",
                                elem_classes="hintitem"
                            )

                        with gr.Row():
                            use_erg_tag = gr.Checkbox(
                                label="Use ERG for tag",
                                value=True,
                                elem_id="acestep_use_erg_tag",
                                elem_classes="hintitem",
                                info="Use Entropy Rectifying Guidance for tag. It will multiple a temperature to the attention to make a weaker tag condition and make better diversity."
                            )
                            
                            use_erg_lyric = gr.Checkbox(
                                label="Use ERG for lyric",
                                value=False,
                                elem_id="acestep_use_erg_lyric",
                                elem_classes="hintitem",
                                info="The same but apply to lyric encoder's attention."
                            )
                            
                            use_erg_diffusion = gr.Checkbox(
                                label="Use ERG for diffusion",
                                value=True,
                                elem_id="acestep_use_erg_diffusion",
                                elem_classes="hintitem",
                                info="The same but apply to diffusion model's attention."
                            )

                        with gr.Row():
                            guidance_interval = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                step=0.01,
                                value=0.5,
                                label="Guidance Interval",
                                elem_id="acestep_guidance_interval",
                                elem_classes="hintitem",
                                info="Guidance interval for the generation. 0.5 means only apply guidance in the middle steps (0.25 * infer_steps to 0.75 * infer_steps)"
                            )
                            
                            guidance_interval_decay = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                step=0.01,
                                value=0.0,
                                label="Guidance Interval Decay",
                                elem_id="acestep_guidance_interval_decay",
                                elem_classes="hintitem",
                                info="Guidance interval decay for the generation. Guidance scale will decay from guidance_scale to min_guidance_scale in the interval. 0.0 means no decay."
                            )
                            
                            min_guidance_scale = gr.Slider(
                                minimum=0.0,
                                maximum=200.0,
                                step=0.1,
                                value=3.0,
                                label="Min Guidance Scale",
                                elem_id="acestep_min_guidance_scale",
                                elem_classes="hintitem",
                                info="Min guidance scale for guidance interval decay's end scale"
                            )

                        with gr.Row():
                            guidance_scale_text = gr.Slider(
                                minimum=0.0,
                                maximum=10.0,
                                step=0.1,
                                value=0.0,
                                label="Guidance Scale Text",
                                elem_id="acestep_guidance_scale_text",
                                elem_classes="hintitem",
                                info="Guidance scale for text condition. It can only apply to cfg. set guidance_scale_text=5.0, guidance_scale_lyric=1.5 for start"
                            )
                            
                            guidance_scale_lyric = gr.Slider(
                                minimum=0.0,
                                maximum=10.0,
                                step=0.1,
                                value=0.0,
                                label="Guidance Scale Lyric",
                                elem_id="acestep_guidance_scale_lyric",
                                elem_classes="hintitem"
                            )

                        with gr.Row():
                            oss_steps = gr.Textbox(
                                label="OSS Steps",
                                placeholder="16, 29, 52, 96, 129, 158, 172, 183, 189, 200",
                                value=None,
                                elem_id="acestep_oss_steps",
                                elem_classes="hintitem",
                                info="Optimal Steps for the generation. But not test well"
                            )

                    # Add Audio2Audio section
                    with gr.Accordion("Audio2Audio", open=False):
                        gr.Markdown("Use a reference audio file to guide the generation. The model will try to create music with similar characteristics.")
                        
                        with gr.Row():
                            audio2audio_enable = gr.Checkbox(
                                label="Enable Audio2Audio",
                                value=False,
                                elem_id="acestep_audio2audio_enable",
                                elem_classes="hintitem"
                            )
                            
                            ref_audio_strength = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=0.5,
                                step=0.1,
                                label="Reference Audio Strength",
                                elem_id="acestep_ref_audio_strength",
                                elem_classes="hintitem",
                                visible=False
                            )
                        
                        ref_audio_input = gr.Audio(
                            label="Reference Audio",
                            type="filepath",
                            elem_id="acestep_ref_audio_input",
                            elem_classes="hintitem",
                            show_download_button=True,
                            visible=False
                        )
                        
                        gr.Markdown("*Higher strength values will make the generated music more similar to the reference audio.*")

                    # Function to toggle reference audio visibility
                    def toggle_ref_audio_visibility(is_checked):
                        return [
                            gr.update(visible=is_checked),  # ref_audio_input
                            gr.update(visible=is_checked)   # ref_audio_strength
                        ]
                    
                    # Connect the toggle function
                    audio2audio_enable.change(
                        fn=toggle_ref_audio_visibility,
                        inputs=[audio2audio_enable],
                        outputs=[ref_audio_input, ref_audio_strength]
                    )

                # Middle Column - Prompt & Lyrics
                with gr.Column():
                    gr.Markdown("### 🎨 Music Description")
                    
                    prompt_input = gr.Textbox(
                        label="Music Prompt",
                        placeholder="Describe the music you want to generate (style, mood, instruments, etc.)",
                        lines=5,
                        elem_id="acestep_prompt",
                        elem_classes="hintitem"
                    )
                    
                    # Standard examples (visible when LoRA is not used)
                    standard_examples_accordion = gr.Accordion(
                        label="Example Prompts",
                        open=False,
                        elem_id="acestep_standard_examples_accordion",
                        elem_classes="hintitem"
                    )
                    
                    with standard_examples_accordion:
                        gr.Examples(
                        examples=[
                            "An upbeat pop song with electronic beats and catchy hooks",
                            "Emotional piano ballad with strings and soft percussion",
                            "Epic orchestral music with powerful brass and dramatic percussion",
                            "Relaxing lo-fi hip hop with jazzy piano and mellow beats",
                            "Acoustic folk song with guitar and warm vocals"
                        ],
                        inputs=prompt_input,
                            label=""
                        )
                    
                    # LoRA examples (only visible when use_lora is checked)
                    lora_examples_accordion = gr.Accordion(
                        label="Example Prompts for RapMachine",
                        open=False,
                        elem_id="acestep_lora_examples_accordion",
                        elem_classes="hintitem",
                        visible=False
                    )
                    
                    with lora_examples_accordion:
                        gr.Examples(
                            examples=[
                                "Chinese rap music with modern trap beats",
                                "Aggressive rap flow with booming 808s",
                                "Melodic rap with soft piano and emotional vocals",
                                "Old school rap with classic boom bap beats",
                                "Triplet flow rap with heavy bass and hi-hats"
                            ],
                            inputs=prompt_input,
                            label=""
                        )
                    
                    gr.Markdown("### 🎤 Lyrics")
                    
                    # Standard lyrics label
                    lyrics_label = gr.Markdown("Lyrics (Optional)", elem_id="acestep_lyrics_label")
                    
                    # LoRA lyrics label (only visible when use_lora is checked)
                    lora_lyrics_label = gr.Markdown("Lyrics (Required for Rap)", elem_id="acestep_lora_lyrics_label", visible=False)
                    
                    lyrics_input = gr.Textbox(
                        label="Lyrics",
                        placeholder="Enter lyrics for the vocals",
                        lines=10,
                        elem_id="acestep_lyrics",
                        elem_classes="hintitem"
                    )
                    
                    upload_lyrics = gr.File(
                        label="Upload Lyrics File",
                        file_count="single",
                        file_types=[".txt", ".lrc"],
                        elem_id="acestep_upload_lyrics",
                        elem_classes="hintitem"
                    )

                # Right Column - Actions & Output
                with gr.Column():
                    gr.Markdown("### 🎮 Actions")
                    
                    with gr.Row():
                        generate_btn = gr.Button(
                            "Generate Music",
                            variant="primary",
                            elem_id="acestep_generate_btn",
                            elem_classes="hintitem"
                        )
                        
                        SEND_TO_PROCESS_BUTTON = gr.Button(
                            "Send to Process",
                            elem_id="acestep_send_to_process",
                            elem_classes="hintitem"
                        )
                        
                        download_btn = gr.Button(
                            "Download",
                            elem_id="acestep_download",
                            elem_classes="hintitem"
                        )
                    
                    gr.Markdown("### 🎵 Output")
                    
                    OUTPUT_AUDIO = gr.Audio(
                        label="Generated Music",
                        type="filepath",
                        elem_id="acestep_output_audio",
                        elem_classes="hintitem"
                    )
                    
                    output_message = gr.Textbox(
                        label="Output Message",
                        elem_id="acestep_output_message",
                        elem_classes="hintitem"
                    )

            # Function to load lyrics from file
            def load_lyrics_from_file(file):
                if not file:
                    return ""
                with open(file.name, "r", encoding="utf-8") as f:
                    return f.read()

            # Function to toggle LoRA-specific UI elements
            def toggle_lora_ui(use_lora):
                return [
                    gr.update(visible=use_lora),  # lora_model_dropdown
                    gr.update(visible=not use_lora),  # standard_examples_accordion
                    gr.update(visible=use_lora),  # lora_examples_accordion
                    gr.update(visible=not use_lora),  # lyrics_label
                    gr.update(visible=use_lora)   # lora_lyrics_label
                ]

            # Connect event handlers
            upload_lyrics.change(load_lyrics_from_file, inputs=[upload_lyrics], outputs=[lyrics_input])

            # Connect LoRA toggle
            use_lora.change(
                fn=toggle_lora_ui,
                inputs=[use_lora],
                outputs=[lora_model_dropdown, standard_examples_accordion, lora_examples_accordion, lyrics_label, lora_lyrics_label]
            )

            # Function to handle Low VRAM Mode toggle
            def update_low_vram_settings(enable_low_vram):
                if enable_low_vram:
                    # Enable optimizations for low VRAM
                    return True, True, True
                else:
                    # Return to default values
                    return False, False, False
                
            # Connect Low VRAM Mode toggle
            low_vram_mode.change(
                fn=update_low_vram_settings,
                inputs=[low_vram_mode],
                outputs=[torch_compile, cpu_offload, overlapped_decode]
            )

            # Function to generate music
            def generate_music(
                use_lora,
                prompt, 
                lyrics, 
                duration, 
                base_model, 
                lora_model,
                seed,
                device,
                bf16,
                torch_compile,
                cpu_offload,
                overlapped_decode,
                infer_step,
                guidance_scale,
                scheduler_type,
                cfg_type,
                omega_scale,
                audio2audio_enable,
                ref_audio_strength,
                ref_audio_input,
                use_erg_tag,
                use_erg_lyric,
                use_erg_diffusion,
                guidance_interval,
                guidance_interval_decay,
                min_guidance_scale,
                guidance_scale_text,
                guidance_scale_lyric,
                oss_steps,
                progress=gr.Progress(track_tqdm=True)
            ):
                if not prompt or prompt.strip() == "":
                    return None, "Please provide a prompt describing the music you want to generate."
                
                # For LoRA models, check if lyrics are provided
                if use_lora and (not lyrics or lyrics.strip() == ""):
                    return None, "Lyrics are required for rap generation with LoRA models."
                
                # Extract device ID from dropdown selection
                device_id = int(device.split(":")[0])
                
                # Check if audio2audio is enabled but no reference audio is provided
                if audio2audio_enable and (not ref_audio_input or not os.path.exists(ref_audio_input)):
                    return None, "Audio2Audio is enabled but no reference audio file was provided. Please upload a reference audio file."
                
                # Choose whether to use LoRA or regular generation
                if use_lora:
                    # Run LoRA generation process
                    output_path, message = process_lora(
                        prompt=prompt,
                        lyrics=lyrics,
                        audio_duration=duration,
                        base_model_name=base_model,
                        lora_model_path=lora_model,
                        seed=seed if seed is not None else None,
                        device_id=device_id,
                        bf16=bf16,
                        torch_compile=torch_compile,
                        cpu_offload=cpu_offload,
                        infer_step=infer_step,
                        guidance_scale=guidance_scale,
                        scheduler_type=scheduler_type,
                        cfg_type=cfg_type,
                        omega_scale=omega_scale,
                        audio2audio_enable=audio2audio_enable,
                        ref_audio_strength=ref_audio_strength,
                        ref_audio_input=ref_audio_input,
                        use_erg_tag=use_erg_tag,
                        use_erg_lyric=use_erg_lyric,
                        use_erg_diffusion=use_erg_diffusion,
                        guidance_interval=guidance_interval,
                        guidance_interval_decay=guidance_interval_decay,
                        min_guidance_scale=min_guidance_scale,
                        guidance_scale_text=guidance_scale_text,
                        guidance_scale_lyric=guidance_scale_lyric,
                        oss_steps=oss_steps,
                        progress_callback=progress
                    )
                else:
                    # Run standard generation process
                    output_path, message = process(
                        prompt=prompt,
                        lyrics=lyrics if lyrics else "",
                        audio_duration=duration,
                        model_name=base_model,
                        seed=seed if seed is not None else None,
                        device_id=device_id,
                        bf16=bf16,
                        torch_compile=torch_compile,
                        cpu_offload=cpu_offload,
                        overlapped_decode=overlapped_decode,
                        infer_step=infer_step,
                        guidance_scale=guidance_scale,
                        scheduler_type=scheduler_type,
                        cfg_type=cfg_type,
                        omega_scale=omega_scale,
                        audio2audio_enable=audio2audio_enable,
                        ref_audio_strength=ref_audio_strength,
                        ref_audio_input=ref_audio_input,
                        use_erg_tag=use_erg_tag,
                        use_erg_lyric=use_erg_lyric,
                        use_erg_diffusion=use_erg_diffusion,
                        guidance_interval=guidance_interval,
                        guidance_interval_decay=guidance_interval_decay,
                        min_guidance_scale=min_guidance_scale,
                        guidance_scale_text=guidance_scale_text,
                        guidance_scale_lyric=guidance_scale_lyric,
                        oss_steps=oss_steps,
                        progress_callback=progress
                    )
                
                return output_path, message

            # Connect generation button
            generate_btn.click(
                fn=generate_music,
                inputs=[
                    use_lora,
                    prompt_input, 
                    lyrics_input, 
                    audio_duration, 
                    model_dropdown,
                    lora_model_dropdown,
                    seed_input,
                    device_id,
                    bf16,
                    torch_compile,
                    cpu_offload,
                    overlapped_decode,
                    infer_step,
                    guidance_scale,
                    scheduler_type,
                    cfg_type,
                    omega_scale,
                    audio2audio_enable,
                    ref_audio_strength,
                    ref_audio_input,
                    use_erg_tag,
                    use_erg_lyric,
                    use_erg_diffusion,
                    guidance_interval,
                    guidance_interval_decay,
                    min_guidance_scale,
                    guidance_scale_text,
                    guidance_scale_lyric,
                    oss_steps
                ],
                outputs=[OUTPUT_AUDIO, output_message]
            )

            # Function for downloading generated audio
            def prepare_download(audio_path):
                if not audio_path or not os.path.exists(audio_path):
                    return None
                return audio_path

            # Connect download button
            download_btn.click(
                fn=prepare_download,
                inputs=[OUTPUT_AUDIO],
                outputs=[gr.File(label="Download Generated Music")]
            )

        # Retake Tab - Create variations of existing music
        with gr.TabItem("Retake", id="acestep_retake"):
            gr.Markdown("# 🎲 ACE-Step: Retake")
            gr.Markdown(
                "Generate variations of existing music with different seeds but same parameters. "
                "This is useful for exploring multiple options with the same prompt and lyrics."
            )

            with gr.Row():
                # Left Column - Source Settings
                with gr.Column():
                    gr.Markdown("### 🔍 Source Audio")
                    
                    retake_source_type = gr.Radio(
                        choices=["Last Generated", "Upload File"],
                        value="Last Generated",
                        label="Source Type",
                        elem_id="acestep_retake_source_type",
                        elem_classes="hintitem"
                    )
                    
                    retake_upload = gr.Audio(
                        label="Upload Audio",
                        type="filepath",
                        visible=False,
                        elem_id="acestep_retake_upload",
                        elem_classes="hintitem"
                    )
                    
                    # Show/hide upload field based on source type
                    def toggle_retake_upload(source_type):
                        return gr.update(visible=source_type == "Upload File")
                    
                    retake_source_type.change(
                        fn=toggle_retake_upload,
                        inputs=[retake_source_type],
                        outputs=[retake_upload]
                    )
                    
                    gr.Markdown("### 🎛️ Variation Settings")
                    
                    # Add prompt field for retake
                    retake_prompt = gr.Textbox(
                        label="Prompt (Optional)",
                        placeholder="Enter the prompt that was used for the original generation",
                        value="background music",
                        elem_id="acestep_retake_prompt",
                        elem_classes="hintitem"
                    )

                    with gr.Row():
                        retake_variation_count = gr.Slider(
                            minimum=1,
                            maximum=5,
                            value=3,
                            step=1,
                            label="Number of Variations",
                            elem_id="acestep_retake_count",
                            elem_classes="hintitem"
                        )
                        
                        retake_variation_strength = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.5,
                            step=0.1,
                            label="Variation Strength",
                            elem_id="acestep_retake_strength",
                            elem_classes="hintitem"
                        )

                # Right Column - Controls and Output
                with gr.Column():
                    gr.Markdown("### 🎮 Actions")
                    
                    retake_btn = gr.Button(
                        "Generate Variations",
                        variant="primary",
                        elem_id="acestep_retake_btn",
                        elem_classes="hintitem"
                    )
                    
                    gr.Markdown("### 🎵 Generated Variations")
                    
                    retake_output = gr.Audio(
                        label="Variation 1",
                        type="filepath",
                        elem_id="acestep_retake_output",
                        elem_classes="hintitem"
                    )
                    
                    retake_output2 = gr.Audio(
                        label="Variation 2",
                        type="filepath",
                        elem_id="acestep_retake_output2",
                        elem_classes="hintitem",
                        visible=False
                    )
                    
                    retake_output3 = gr.Audio(
                        label="Variation 3",
                        type="filepath",
                        elem_id="acestep_retake_output3",
                        elem_classes="hintitem",
                        visible=False
                    )
                    
                    retake_message = gr.Textbox(
                        label="Output Message",
                        elem_id="acestep_retake_message",
                        elem_classes="hintitem"
                    )
                    
                    with gr.Row():
                        retake_send_to_process_btn = gr.Button(
                            "Send to Process",
                            elem_id="acestep_retake_send_to_process",
                                elem_classes="hintitem"
                            )
                            
                        retake_download_btn = gr.Button(
                            "Download All",
                            elem_id="acestep_retake_download",
                                elem_classes="hintitem"
                        )
            
            # Function to generate variations
            def generate_variations(source_type, upload_path, prompt, count, strength, last_generated_audio):
                # Determine source audio
                source_audio = None
                if source_type == "Last Generated":
                    source_audio = last_generated_audio
                    if not source_audio or not os.path.exists(source_audio):
                        return None, None, None, "No audio was previously generated."
                else:
                    source_audio = upload_path
                    if not source_audio or not os.path.exists(source_audio):
                        return None, None, None, "Please upload an audio file."
                
                # Extract device ID from dropdown
                try:
                    current_device_id = int(device_id.value.split(":")[0])
                except:
                    current_device_id = 0
                
                # Generate variations using the process_retake function
                try:
                    # When generating variations, we need to preserve the original prompt and lyrics
                    # These are necessary for the retake function to work properly
                    variations, message = process_retake(
                        source_audio_path=source_audio,
                        variation_count=count,
                        variation_strength=strength,
                        prompt=prompt,
                        device_id=current_device_id,
                        progress_callback=gr.Progress(track_tqdm=True)
                    )
                    
                    if not variations or len(variations) == 0:
                        return None, None, None, message
                    
                    # Prepare return values (one for each output component)
                    var1 = variations[0] if len(variations) > 0 else None
                    var2 = variations[1] if len(variations) > 1 else None
                    var3 = variations[2] if len(variations) > 2 else None
                    
                    return var1, var2, var3, message
                    
                except Exception as e:
                    logger.error(f"Error generating variations: {e}")
                    traceback.print_exc()
                    return None, None, None, f"Error: {str(e)}"
            
            # Function to show/hide variation outputs based on count
            def update_variation_visibility(count):
                return [
                    gr.update(visible=True),  # First output always visible
                    gr.update(visible=count >= 2),
                    gr.update(visible=count >= 3)
                ]
            
            # Connect retake UI elements
            retake_variation_count.change(
                fn=update_variation_visibility,
                inputs=[retake_variation_count],
                outputs=[retake_output, retake_output2, retake_output3]
            )
            
            retake_btn.click(
                fn=generate_variations,
                inputs=[retake_source_type, retake_upload, retake_prompt, retake_variation_count, retake_variation_strength, OUTPUT_AUDIO],
                outputs=[retake_output, retake_output2, retake_output3, retake_message]
            )
            
            # Function to download all variations
            def download_variations(var1, var2, var3):
                variations = [v for v in [var1, var2, var3] if v and os.path.exists(v)]
                if not variations:
                    return None
                
                return download_output_files(variations)
            
            # Connect download button
            retake_download_btn.click(
                fn=download_variations,
                inputs=[retake_output, retake_output2, retake_output3],
                outputs=[gr.File(label="Download Variations")]
            )
            
            # Connect send to process button for first variation
            retake_send_to_process_btn.click(
                fn=send_to_process,
                inputs=[retake_output, arg_handler.get_element("main", "process_inputs")],
                outputs=[arg_handler.get_element("main", "process_inputs")]
            )

        # Repaint Tab - Regenerate specific sections
        with gr.TabItem("Repaint", id="acestep_repaint"):
            gr.Markdown("# 🎨 ACE-Step: Repaint")
            gr.Markdown(
                "Selectively regenerate specific sections of your music while keeping the rest unchanged. "
                "This is useful for fixing or improving particular parts of a song."
                            )

            with gr.Row():
                # Left Column - Source & Settings
                with gr.Column():
                    gr.Markdown("### 🔍 Source Audio")
                    
                    repaint_source_type = gr.Radio(
                        choices=["Last Generated", "Last Repainted", "Upload File"],
                        value="Last Generated",
                        label="Source Type",
                        elem_id="acestep_repaint_source_type",
                                elem_classes="hintitem"
                            )
                            
                    repaint_upload = gr.Audio(
                        label="Upload Audio",
                        type="filepath",
                        visible=False,
                        elem_id="acestep_repaint_upload",
                                elem_classes="hintitem"
                            )

                    # Show/hide upload field based on source type
                    def toggle_repaint_upload(source_type):
                        return gr.update(visible=source_type == "Upload File")
                    
                    repaint_source_type.change(
                        fn=toggle_repaint_upload,
                        inputs=[repaint_source_type],
                        outputs=[repaint_upload]
                    )
                    
                    gr.Markdown("### ⏱️ Selection Range")
                    
                    with gr.Row():
                        repaint_start_time = gr.Slider(
                            minimum=0,
                            maximum=60,
                            value=10,
                                step=1,
                            label="Start Time (seconds)",
                            elem_id="acestep_repaint_start",
                                elem_classes="hintitem"
                            )
                            
                        repaint_end_time = gr.Slider(
                            minimum=0,
                            maximum=60,
                            value=20,
                            step=1,
                            label="End Time (seconds)",
                            elem_id="acestep_repaint_end",
                                elem_classes="hintitem"
                            )
                            
                    gr.Markdown("### 🎛️ Repaint Settings")
                    
                    with gr.Row():
                        repaint_strength = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.8,
                            step=0.1,
                            label="Repaint Strength",
                            elem_id="acestep_repaint_strength",
                                elem_classes="hintitem"
                            )
                            
                        repaint_seed = gr.Number(
                            label="Seed (leave empty for random)",
                            value=None,
                            precision=0,
                            elem_id="acestep_repaint_seed",
                            elem_classes="hintitem"
                        )
                
                # Right Column - Controls and Output
                with gr.Column():
                    gr.Markdown("### 🎮 Actions")
                    
                    repaint_btn = gr.Button(
                        "Repaint Selection",
                        variant="primary",
                        elem_id="acestep_repaint_btn",
                        elem_classes="hintitem"
                    )
                    
                    gr.Markdown("### 🎵 Output")
                    
                    repaint_output = gr.Audio(
                        label="Repainted Audio",
                        type="filepath",
                        elem_id="acestep_repaint_output",
                        elem_classes="hintitem"
                    )
                    
                    repaint_message = gr.Textbox(
                        label="Output Message",
                        elem_id="acestep_repaint_message",
                                elem_classes="hintitem"
                            )
                            
                    with gr.Row():
                        repaint_send_to_process_btn = gr.Button(
                            "Send to Process",
                            elem_id="acestep_repaint_send_to_process",
                                elem_classes="hintitem"
                            )

                        repaint_download_btn = gr.Button(
                            "Download",
                            elem_id="acestep_repaint_download",
                            elem_classes="hintitem"
                        )
            
            # Function to repaint a section of audio
            def repaint_audio_section(source_type, upload_path, start_time, end_time, strength, seed, last_generated_audio, last_repainted_audio):
                # Determine source audio
                source_audio = None
                if source_type == "Last Generated":
                    source_audio = last_generated_audio
                    if not source_audio or not os.path.exists(source_audio):
                        return None, "No audio was previously generated."
                elif source_type == "Last Repainted":
                    source_audio = last_repainted_audio
                    if not source_audio or not os.path.exists(source_audio):
                        return None, "No audio was previously repainted."
                else:
                    source_audio = upload_path
                    if not source_audio or not os.path.exists(source_audio):
                        return None, "Please upload an audio file."
                
                # Validate time range
                if end_time <= start_time:
                    return None, "End time must be greater than start time."
                
                # Extract device ID from dropdown
                try:
                    current_device_id = int(device_id.value.split(":")[0])
                except:
                    current_device_id = 0
                
                # Generate repainted audio using the process_repaint function
                try:
                    output_path_file, message = process_repaint(
                        source_audio_path=source_audio,
                        start_time=start_time,
                        end_time=end_time,
                        repaint_strength=strength,
                        device_id=current_device_id,
                        seed=seed if seed is not None else None,
                        progress_callback=gr.Progress(track_tqdm=True)
                    )
                    
                    return output_path_file, message
                    
                except Exception as e:
                    logger.error(f"Error repainting audio: {e}")
                    traceback.print_exc()
                    return None, f"Error: {str(e)}"
            
            # Connect repaint UI elements
            repaint_btn.click(
                fn=repaint_audio_section,
                inputs=[
                    repaint_source_type, repaint_upload, 
                    repaint_start_time, repaint_end_time,
                    repaint_strength, repaint_seed,
                    OUTPUT_AUDIO, repaint_output
                ],
                outputs=[repaint_output, repaint_message]
            )
            
            # Connect download button
            repaint_download_btn.click(
                fn=prepare_download,
                inputs=[repaint_output],
                outputs=[gr.File(label="Download Repainted Audio")]
            )
            
            # Connect send to process button
            repaint_send_to_process_btn.click(
                fn=send_to_process,
                inputs=[repaint_output, arg_handler.get_element("main", "process_inputs")],
                outputs=[arg_handler.get_element("main", "process_inputs")]
            )

        # Edit Tab - Modify lyrics in existing audio
        with gr.TabItem("Edit", id="acestep_edit"):
            gr.Markdown("# ✏️ ACE-Step: Lyric Edit")
            gr.Markdown(
                "Modify the lyrics of existing audio while maintaining the musical style and structure. "
                "This is useful for tweaking or improving the vocal content of a song."
            )
            
            with gr.Row():
                # Left Column - Source & Settings
                with gr.Column():
                    gr.Markdown("### 🔍 Source Audio")
                    
                    edit_source_type = gr.Radio(
                        choices=["Last Generated", "Last Edited", "Upload File"],
                        value="Last Generated",
                        label="Source Type",
                        elem_id="acestep_edit_source_type",
                        elem_classes="hintitem"
                    )
                    
                    edit_upload = gr.Audio(
                        label="Upload Audio",
                        type="filepath",
                        visible=False,
                        elem_id="acestep_edit_upload",
                        elem_classes="hintitem"
                    )
                    
                    # Show/hide upload field based on source type
                    def toggle_edit_upload(source_type):
                        return gr.update(visible=source_type == "Upload File")
                    
                    edit_source_type.change(
                        fn=toggle_edit_upload,
                        inputs=[edit_source_type],
                        outputs=[edit_upload]
                    )
                    
                    gr.Markdown("### 🎤 Current Lyrics")
                    
                    current_lyrics = gr.Textbox(
                        label="Current Lyrics",
                        placeholder="The system will try to detect the current lyrics (if available)",
                        lines=5,
                        elem_id="acestep_current_lyrics",
                        elem_classes="hintitem"
                    )
                    
                    transcribe_btn = gr.Button(
                        "Transcribe Audio",
                        elem_id="acestep_transcribe_btn",
                        elem_classes="hintitem"
                    )
                    
                    gr.Markdown("### ✏️ Edit Lyrics")
                    
                    edit_mode = gr.Radio(
                        choices=["only_lyrics", "remix"],
                        value="only_lyrics",
                        label="Edit Mode",
                        elem_id="acestep_edit_mode",
                        elem_classes="hintitem"
                    )
                    
                    with gr.Row():
                        edit_range_start = gr.Slider(
                            minimum=0,
                            maximum=60,
                            value=10,
                            step=1,
                            label="Start Time (seconds)",
                            elem_id="acestep_edit_start",
                        elem_classes="hintitem"
                    )

                        edit_range_end = gr.Slider(
                            minimum=0,
                            maximum=60,
                            value=20,
                            step=1,
                            label="End Time (seconds)",
                            elem_id="acestep_edit_end",
                            elem_classes="hintitem"
                        )
                    
                    new_lyrics = gr.Textbox(
                        label="New Lyrics",
                        placeholder="Enter the new lyrics for the selected time range",
                        lines=5,
                        elem_id="acestep_new_lyrics",
                        elem_classes="hintitem"
                    )
                
                # Right Column - Controls and Output
                with gr.Column():
                    gr.Markdown("### 🎮 Actions")
                    
                    edit_btn = gr.Button(
                        "Apply Lyric Edit",
                            variant="primary",
                        elem_id="acestep_edit_btn",
                            elem_classes="hintitem"
                        )
                        
                    gr.Markdown("### 🎵 Output")
                    
                    edit_output = gr.Audio(
                        label="Edited Audio",
                        type="filepath",
                        elem_id="acestep_edit_output",
                        elem_classes="hintitem"
                    )
                    
                    edit_message = gr.Textbox(
                        label="Output Message",
                        elem_id="acestep_edit_message",
                        elem_classes="hintitem"
                    )
                    
                    with gr.Row():
                        edit_send_to_process_btn = gr.Button(
                            "Send to Process",
                            elem_id="acestep_edit_send_to_process",
                            elem_classes="hintitem"
                        )
                        
                        edit_download_btn = gr.Button(
                            "Download",
                            elem_id="acestep_edit_download",
                            elem_classes="hintitem"
                        )
                    
            # Function to transcribe lyrics from audio
            def transcribe_audio(source_type, upload_path, last_generated_audio, last_edited_audio):
                # Determine source audio
                source_audio = None
                if source_type == "Last Generated":
                    source_audio = last_generated_audio
                    if not source_audio or not os.path.exists(source_audio):
                        return "No audio was previously generated."
                elif source_type == "Last Edited":
                    source_audio = last_edited_audio
                    if not source_audio or not os.path.exists(source_audio):
                        return "No audio was previously edited."
                else:
                    source_audio = upload_path
                    if not source_audio or not os.path.exists(source_audio):
                        return "Please upload an audio file."
                
                try:
                    # We would use an actual transcription function here
                    # For this implementation, we'll use a placeholder or try to use process_transcription if available
                    try:
                        from layouts.transcribe import process_transcription
                        result = process_transcription(source_audio, "auto")
                        if result and isinstance(result, dict) and "text" in result:
                            return result["text"]
                        else:
                            return "Placeholder: Detected lyrics would appear here."
                    except ImportError:
                        return "Placeholder: Detected lyrics would appear here."
                except Exception as e:
                    logger.error(f"Error transcribing audio: {e}")
                    return f"Error: {str(e)}"
            
            # Function to edit lyrics in audio
            def edit_audio_lyrics(source_type, upload_path, start_time, end_time, current_lyrics, new_lyrics, edit_mode, last_generated_audio, last_edited_audio):
                # Determine source audio
                source_audio = None
                if source_type == "Last Generated":
                    source_audio = last_generated_audio
                    if not source_audio or not os.path.exists(source_audio):
                        return None, "No audio was previously generated."
                elif source_type == "Last Edited":
                    source_audio = last_edited_audio
                    if not source_audio or not os.path.exists(source_audio):
                        return None, "No audio was previously edited."
                else:
                    source_audio = upload_path
                    if not source_audio or not os.path.exists(source_audio):
                        return None, "Please upload an audio file."
                
                # Validate edit parameters
                if end_time <= start_time:
                    return None, "End time must be greater than start time."
                
                if not new_lyrics or new_lyrics.strip() == "":
                    return None, "New lyrics cannot be empty."
                
                # Extract device ID from dropdown
                try:
                    current_device_id = int(device_id.value.split(":")[0])
                except:
                    current_device_id = 0
                
                # Edit audio using the process_edit function
                try:
                    output_path_file, message = process_edit(
                        source_audio_path=source_audio,
                        start_time=start_time,
                        end_time=end_time,
                        current_lyrics=current_lyrics,
                        new_lyrics=new_lyrics,
                        edit_mode=edit_mode,
                        device_id=current_device_id,
                        progress_callback=gr.Progress(track_tqdm=True)
                    )
                    
                    return output_path_file, message
                    
                except Exception as e:
                    logger.error(f"Error editing lyrics: {e}")
                    traceback.print_exc()
                    return None, f"Error: {str(e)}"
            
            # Connect transcribe button
            transcribe_btn.click(
                fn=transcribe_audio,
                inputs=[edit_source_type, edit_upload, OUTPUT_AUDIO, edit_output],
                outputs=[current_lyrics]
            )
            
            # Connect edit button
            edit_btn.click(
                fn=edit_audio_lyrics,
                inputs=[
                    edit_source_type, edit_upload,
                    edit_range_start, edit_range_end,
                    current_lyrics, new_lyrics, edit_mode,
                    OUTPUT_AUDIO, edit_output
                ],
                outputs=[edit_output, edit_message]
            )
            
            # Connect download button
            edit_download_btn.click(
                fn=prepare_download,
                inputs=[edit_output],
                outputs=[gr.File(label="Download Edited Audio")]
            )
            
            # Connect send to process button
            edit_send_to_process_btn.click(
                fn=send_to_process,
                inputs=[edit_output, arg_handler.get_element("main", "process_inputs")],
                outputs=[arg_handler.get_element("main", "process_inputs")]
            )
            
        # Training Tab
        with gr.TabItem("Train", id="acestep_train"):
            gr.Markdown("# 🎓 ACE-Step: LoRA Training")
            gr.Markdown(
                "Train your own LoRA models for specialized music generation. "
                "Create custom models like RapMachine for specific music styles or genres."
            )
            
            with gr.Row():
                # Left Column - Training Settings
                with gr.Column():
                    gr.Markdown("### 🔧 Training Settings")
                    
                    base_model_train = gr.Dropdown(
                        choices=AVAILABLE_MODELS,
                        value=DEFAULT_MODEL,
                        label="Base Model",
                        elem_id="acestep_train_base_model",
                        elem_classes="hintitem"
                    )
                    
                    output_model_name = gr.Textbox(
                        label="Output Model Name",
                        placeholder="Name for your trained LoRA model (e.g., my_rap_lora)",
                        value="my_acestep_lora",
                        elem_id="acestep_output_model",
                        elem_classes="hintitem"
                    )

                    with gr.Row():
                        dataset_path = gr.Textbox(
                            label="Dataset Path",
                            placeholder="Path to your training dataset directory",
                            elem_id="acestep_dataset_path",
                            elem_classes="hintitem"
                        )
                        
                        dataset_browser = gr.File(
                            label="Upload Dataset",
                            file_count="directory",
                            elem_id="acestep_dataset_upload",
                            elem_classes="hintitem"
                        )
                    
                    with gr.Accordion("Training Parameters", open=True):
                        with gr.Row():
                            learning_rate = gr.Slider(
                                minimum=1e-5,
                                maximum=1e-3,
                                value=1e-4,
                                step=1e-5,
                                label="Learning Rate",
                                elem_id="acestep_learning_rate",
                                elem_classes="hintitem"
                            )
                            
                            batch_size = gr.Slider(
                                minimum=1,
                                maximum=16,
                                value=4,
                                step=1,
                                label="Batch Size",
                                elem_id="acestep_batch_size",
                                elem_classes="hintitem"
                            )
                        
                        with gr.Row():
                            epochs = gr.Slider(
                                minimum=1,
                                maximum=100,
                                value=10,
                                step=1,
                                label="Epochs",
                                elem_id="acestep_epochs",
                                elem_classes="hintitem"
                            )
                            
                            gradient_accumulation_steps = gr.Slider(
                                minimum=1,
                                maximum=16,
                                value=4,
                                step=1,
                                label="Gradient Accumulation Steps",
                                elem_id="acestep_grad_accum",
                                elem_classes="hintitem"
                            )
                        
                        with gr.Row():
                            lora_rank = gr.Slider(
                                minimum=4,
                                maximum=128,
                                value=32,
                                step=4,
                                label="LoRA Rank",
                                elem_id="acestep_lora_rank",
                                elem_classes="hintitem"
                            )
                            
                            lora_alpha = gr.Slider(
                                minimum=1,
                                maximum=128,
                                value=32,
                                step=1,
                                label="LoRA Alpha",
                                elem_id="acestep_lora_alpha",
                                elem_classes="hintitem"
                            )
                    
                    with gr.Accordion("Advanced Options", open=False):
                        with gr.Row():
                            save_steps = gr.Slider(
                                minimum=100,
                                maximum=2000,
                                value=500,
                                step=100,
                                label="Save Steps",
                                elem_id="acestep_save_steps",
                                elem_classes="hintitem"
                            )
                            
                            eval_steps = gr.Slider(
                                minimum=100,
                                maximum=2000,
                                value=500,
                                step=100,
                                label="Evaluation Steps",
                                elem_id="acestep_eval_steps",
                                elem_classes="hintitem"
                            )
                        
                        with gr.Row():
                            train_device_id = gr.Dropdown(
                                choices=[f"{i}: {name}" for i, name in enumerate(GPU_NAMES)] if GPU_NAMES else ["0: CPU"],
                                value="0: " + (GPU_NAMES[0] if GPU_NAMES else "CPU"),
                                label="Training Device",
                                elem_id="acestep_train_device",
                                elem_classes="hintitem"
                            )
                            
                            use_8bit_adam = gr.Checkbox(
                                label="Use 8-bit Adam Optimizer",
                                value=True,
                                elem_id="acestep_8bit_adam",
                                elem_classes="hintitem"
                            )

                # Right Column - Dataset and Output
                with gr.Column():
                    gr.Markdown("### 📊 Dataset Information")
                    
                    dataset_type = gr.Radio(
                        choices=["Rap", "Vocal", "Instrumental", "Custom"],
                        value="Custom",
                        label="Dataset Type",
                        elem_id="acestep_dataset_type",
                        elem_classes="hintitem"
                    )
                    
                    dataset_format = gr.Markdown(
                        """
                        ### Dataset Format Requirements:
                        
                        Your dataset should be organized as follows:
                        
                        ```
                        dataset_directory/
                        ├── metadata.json
                        ├── audio_1.wav
                        ├── audio_2.wav
                        └── ...
                        ```
                        
                        The `metadata.json` file should contain:
                        
                        ```json
                        {
                          "audio_1.wav": {
                            "prompt": "Description of the audio",
                            "lyrics": "Lyrics for the vocals (if any)"
                          },
                          "audio_2.wav": {
                            "prompt": "...",
                            "lyrics": "..."
                          },
                          ...
                        }
                        ```
                        """
                    )
                    
                    gr.Markdown("### 🎮 Actions")
                    
                    with gr.Row():
                        train_btn = gr.Button(
                            "Start Training",
                            variant="primary",
                            elem_id="acestep_train_btn",
                            elem_classes="hintitem"
                        )
                        
                        stop_btn = gr.Button(
                            "Stop Training",
                            elem_id="acestep_stop_btn",
                            elem_classes="hintitem"
                        )
                    
                    training_output = gr.Textbox(
                        label="Training Log",
                        placeholder="Training logs will appear here...",
                        lines=15,
                        elem_id="acestep_training_log",
                        elem_classes="hintitem"
                    )
                    
                    gr.Markdown("### 💾 Trained Model")
                    
                    trained_model_path = gr.Textbox(
                        label="Trained Model Path",
                        elem_id="acestep_trained_model_path",
                        elem_classes="hintitem",
                        interactive=False
                    )
                    
                    with gr.Row():
                        test_trained_model_btn = gr.Button(
                            "Test Trained Model",
                            elem_id="acestep_test_model_btn",
                            elem_classes="hintitem"
                        )
                        
                        export_btn = gr.Button(
                            "Export Model",
                            elem_id="acestep_export_btn",
                            elem_classes="hintitem"
                        )
            
            # Function to train LoRA model
            def train_lora_model(
                base_model,
                output_name,
                dataset_path,
                learning_rate,
                batch_size,
                epochs,
                grad_accum,
                lora_rank,
                lora_alpha,
                save_steps,
                eval_steps,
                device,
                use_8bit_adam,
                progress=gr.Progress(track_tqdm=True)
            ):
                # This function would implement actual training using ACE-Step's training code
                # For now, we'll just return a placeholder message
                progress(0, "Starting training...")
                
                # Extract device ID from dropdown selection
                device_id = int(device.split(":")[0])
                
                # Placeholder for actual training implementation
                log = "Starting training with the following parameters:\n"
                log += f"Base Model: {base_model}\n"
                log += f"Output Model Name: {output_name}\n"
                log += f"Dataset Path: {dataset_path}\n"
                log += f"Learning Rate: {learning_rate}\n"
                log += f"Batch Size: {batch_size}\n"
                log += f"Epochs: {epochs}\n"
                log += f"Gradient Accumulation Steps: {grad_accum}\n"
                log += f"LoRA Rank: {lora_rank}\n"
                log += f"LoRA Alpha: {lora_alpha}\n"
                log += f"Device ID: {device_id}\n"
                log += f"Use 8-bit Adam: {use_8bit_adam}\n\n"
                
                # Simulate training process
                for epoch in range(epochs):
                    progress((epoch + 1) / epochs, f"Training epoch {epoch + 1}/{epochs}")
                    log += f"Epoch {epoch + 1}/{epochs}\n"
                    
                    # Simulate batches
                    for batch in range(10):
                        time.sleep(0.1)  # Simulate work
                        loss = 2.0 - (epoch * 0.1 + batch * 0.01)
                        if batch % 2 == 0:
                            log += f"  Batch {batch+1}/10: loss={loss:.4f}\n"
                
                # Create output directory for the model
                output_dir = os.path.join(model_path, "acestep", "lora", output_name)
                os.makedirs(output_dir, exist_ok=True)
                
                # Create a placeholder model file
                with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
                    f.write(f'{{"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": {lora_rank}, "lora_alpha": {lora_alpha}}}')
                
                log += f"\nTraining complete! Model saved to: {output_dir}\n"
                progress(1.0, "Training complete!")
                
                return log, output_dir
            
            # Connect training button
            train_btn.click(
                fn=train_lora_model,
                inputs=[
                    base_model_train,
                    output_model_name,
                    dataset_path,
                    learning_rate,
                    batch_size,
                    epochs,
                    gradient_accumulation_steps,
                    lora_rank,
                    lora_alpha,
                    save_steps,
                    eval_steps,
                    train_device_id,
                    use_8bit_adam
                ],
                outputs=[training_output, trained_model_path]
            )
            
            # Function to test trained model
            def test_trained_model(model_path):
                if not model_path or not os.path.exists(model_path):
                    return "Model path does not exist. Please train a model first."
                
                # This would implement actual testing using the trained LoRA model
                return f"Testing model from {model_path}. This feature will be implemented soon."
            
            # Connect test button
            test_trained_model_btn.click(
                fn=test_trained_model,
                inputs=[trained_model_path],
                outputs=[training_output]
            )
            
            # Function to export trained model
            def export_trained_model(model_path):
                if not model_path or not os.path.exists(model_path):
                    return None, "Model path does not exist. Please train a model first."
                
                try:
                    # Create a zip file of the model directory
                    model_name = os.path.basename(model_path)
                    zip_path = os.path.join(output_path, f"{model_name}.zip")
                    
                    with zipfile.ZipFile(zip_path, 'w') as zipf:
                        for root, dirs, files in os.walk(model_path):
                            for file in files:
                                file_path = os.path.join(root, file)
                                arcname = os.path.relpath(file_path, os.path.dirname(model_path))
                                zipf.write(file_path, arcname)
                    
                    return zip_path, f"Model exported to {zip_path}"
                except Exception as e:
                    return None, f"Error exporting model: {str(e)}"
            
            # Connect export button
            export_btn.click(
                fn=export_trained_model,
                inputs=[trained_model_path],
                outputs=[gr.File(label="Download Trained Model"), training_output]
            )


def listen():
    """Set up event listeners for inter-tab communication."""
    global SEND_TO_PROCESS_BUTTON, OUTPUT_AUDIO

    arg_handler = ArgHandler()
    process_inputs = arg_handler.get_element("main", "process_inputs")

    if process_inputs and isinstance(SEND_TO_PROCESS_BUTTON, gr.Button):
        SEND_TO_PROCESS_BUTTON.click(
            fn=send_to_process,
            inputs=[OUTPUT_AUDIO, process_inputs],
            outputs=[process_inputs]
        )


def register_descriptions(arg_handler: ArgHandler):
    """Register descriptions for UI elements"""
    descriptions = {
        # Generate tab
        "acestep_model": "Select the ACE-Step model to use for music generation.",
        "acestep_duration": "Set the duration of the generated audio in seconds (15-240s).",
        "acestep_seed": "Set a specific seed for reproducible results. Leave empty for random generation.",
        "acestep_device": "Select which GPU to use for generation if multiple are available.",
        "acestep_bf16": "Use bfloat16 precision to reduce memory usage. Recommended for most systems.",
        "acestep_torch_compile": "Use PyTorch compilation for faster inference. Requires more VRAM but speeds up generation.",
        "acestep_cpu_offload": "Offload models to CPU when not in use to reduce VRAM usage. Slightly slower but allows generation on lower-end GPUs.",
        "acestep_overlapped_decode": "Use overlapped decoding for more efficient processing. Helps with longer audio generation.",
        "acestep_low_vram_mode": "Enable optimizations for systems with limited VRAM (8GB or less). Automatically enables torch_compile, cpu_offload, and overlapped_decode.",
        "acestep_infer_step": "Number of inference steps. More steps = higher quality but slower generation.",
        "acestep_guidance_scale": "Controls how closely the generation follows the prompt. Higher values = stronger adherence to the prompt.",
        "acestep_scheduler": "The diffusion scheduler algorithm. 'flow_match_euler' is recommended for most cases. Other options include 'euler' (fast), 'heun' (higher quality but slower), and 'pingpong' (uses SDE).",
        "acestep_cfg_type": "Type of classifier-free guidance. 'apg' (Attention Predictive Guidance) is recommended.",
        "acestep_omega_scale": "Controls the noise schedule during diffusion. Higher values can produce more varied results.",
        "acestep_audio2audio_enable": "Enable Audio2Audio mode to use a reference audio file to guide generation.",
        "acestep_ref_audio_strength": "Controls how much the generated audio will resemble the reference audio. Higher values make the result more similar to the reference.",
        "acestep_ref_audio_input": "Upload a reference audio file to guide the generation process.",
        "acestep_prompt": "Describe the music you want to generate. Include style, instrumentation, mood, tempo, etc.",
        "acestep_lyrics": "Optional lyrics for the vocals. If provided, the model will try to match the singing to these lyrics.",
        "acestep_upload_lyrics": "Upload a text file (.txt) or LRC file (.lrc) containing lyrics.",
        "acestep_generate_btn": "Start generating music with the current settings.",
        "acestep_send_to_process": "Send the generated audio to the Process tab for further editing.",
        "acestep_download": "Download the generated audio file.",
        "acestep_output_audio": "Preview the generated music. Click to play/pause.",
        "acestep_output_message": "Status and information about the generation process.",
        "acestep_use_lora": "Enable the use of LoRA models for specialized music generation (e.g., RapMachine for Chinese rap).",
        "acestep_lora_model": "Select a specialized LoRA model to use for generation.",
        "acestep_standard_examples_accordion": "Example prompts for general music generation.",
        "acestep_lora_examples_accordion": "Example prompts specifically for LoRA models like RapMachine.",
        
        # LoRA tab
        "acestep_lora_duration": "Set the duration of the generated audio in seconds (15-240s).",
        "acestep_lora_seed": "Set a specific seed for reproducible results. Leave empty for random generation.",
        "acestep_lora_device": "Select which GPU to use for generation if multiple are available.",
        "acestep_lora_bf16": "Use bfloat16 precision to reduce memory usage. Recommended for most systems.",
        "acestep_lora_torch_compile": "Use PyTorch compilation for faster inference.",
        "acestep_lora_cpu_offload": "Offload models to CPU when not in use to reduce VRAM usage.",
        "acestep_lora_low_vram_mode": "Enable optimizations for systems with limited VRAM (8GB or less). Automatically enables torch_compile and cpu_offload.",
        "acestep_lora_infer_step": "Number of inference steps. More steps = higher quality but slower generation.",
        "acestep_lora_guidance_scale": "Controls how closely the generation follows the prompt.",
        "acestep_lora_scheduler": "The diffusion scheduler algorithm. 'flow_match_euler' is recommended for most cases. Other options include 'euler' (fast), 'heun' (higher quality but slower), and 'pingpong' (uses SDE).",
        "acestep_lora_cfg_type": "Type of classifier-free guidance.",
        "acestep_lora_omega_scale": "Controls the noise schedule during diffusion.",
        "acestep_lora_prompt": "Describe the music you want to generate. For RapMachine, focus on rap style and characteristics.",
        "acestep_lora_lyrics": "Lyrics for the rap vocals. For RapMachine, Chinese lyrics are recommended.",
        "acestep_lora_upload_lyrics": "Upload a text file containing lyrics.",
        "acestep_lora_generate_btn": "Start generating music with the LoRA model.",
        "acestep_lora_send_to_process": "Send the generated audio to the Process tab for further editing.",
        "acestep_lora_download": "Download the generated audio file.",
        "acestep_lora_output_audio": "Preview the generated music. Click to play/pause.",
        "acestep_lora_output_message": "Status and information about the generation process.",
        
        # Training tab
        "acestep_train_base_model": "Select the base model for training.",
        "acestep_output_model": "Enter the name for your trained LoRA model.",
        "acestep_dataset_path": "Enter the path to your training dataset directory.",
        "acestep_learning_rate": "Enter the learning rate for training.",
        "acestep_batch_size": "Enter the batch size for training.",
        "acestep_epochs": "Enter the number of training epochs.",
        "acestep_grad_accum": "Enter the gradient accumulation steps for training.",
        "acestep_lora_rank": "Enter the LoRA rank for training.",
        "acestep_lora_alpha": "Enter the LoRA alpha for training.",
        "acestep_save_steps": "Enter the save steps for training.",
        "acestep_eval_steps": "Enter the evaluation steps for training.",
        "acestep_train_device": "Select the device for training.",
        "acestep_8bit_adam": "Enable the use of 8-bit Adam optimizer for training.",
        "acestep_dataset_type": "Select the type of dataset for training.",
        "acestep_training_log": "Training logs will appear here.",
        "acestep_trained_model_path": "Trained model path.",
        "acestep_test_model_btn": "Test the trained model.",
        "acestep_export_btn": "Export the trained model.",
        
        # Retake tab
        "acestep_retake_source_type": "Select the source audio for variations generation.",
        "acestep_retake_upload": "Upload an audio file to generate variations from.",
        "acestep_retake_prompt": "Enter the prompt that was used for the original generation to ensure variations are consistent.",
        "acestep_retake_count": "Number of variations to generate.",
        "acestep_retake_strength": "Controls how different the variations will be from the original.",
        "acestep_retake_btn": "Generate variations of the source audio.",
        "acestep_retake_output": "Preview of the first generated variation.",
        "acestep_retake_output2": "Preview of the second generated variation.",
        "acestep_retake_output3": "Preview of the third generated variation.",
        "acestep_retake_message": "Status and information about the variation generation process.",
        "acestep_retake_send_to_process": "Send the first variation to the Process tab for further editing.",
        "acestep_retake_download": "Download all generated variations as a zip file.",
        
        # Repaint tab
        "acestep_repaint_source_type": "Select the source audio for repainting.",
        "acestep_repaint_upload": "Upload an audio file to repaint.",
        "acestep_repaint_start": "Start time of the section to repaint (in seconds).",
        "acestep_repaint_end": "End time of the section to repaint (in seconds).",
        "acestep_repaint_strength": "Controls how different the repainted section will be from the original.",
        "acestep_repaint_seed": "Seed for reproducible repainting. Leave empty for random seed.",
        "acestep_repaint_btn": "Repaint the selected section of the audio.",
        "acestep_repaint_output": "Preview of the repainted audio.",
        "acestep_repaint_message": "Status and information about the repainting process.",
        "acestep_repaint_send_to_process": "Send the repainted audio to the Process tab for further editing.",
        "acestep_repaint_download": "Download the repainted audio.",
        
        # Edit tab
        "acestep_edit_source_type": "Select the source audio for lyric editing.",
        "acestep_edit_upload": "Upload an audio file to edit lyrics.",
        "acestep_current_lyrics": "Current lyrics detected in the audio.",
        "acestep_transcribe_btn": "Transcribe the audio to detect current lyrics.",
        "acestep_edit_mode": "Choose between only modifying lyrics (preserves melody) or remixing the section.",
        "acestep_edit_start": "Start time of the section to edit lyrics (in seconds).",
        "acestep_edit_end": "End time of the section to edit lyrics (in seconds).",
        "acestep_new_lyrics": "New lyrics to replace the current lyrics in the selected time range.",
        "acestep_edit_btn": "Apply the lyric edit to the audio.",
        "acestep_edit_output": "Preview of the edited audio.",
        "acestep_edit_message": "Status and information about the lyric editing process.",
        "acestep_edit_send_to_process": "Send the edited audio to the Process tab for further editing.",
        "acestep_edit_download": "Download the edited audio."
    }

    # Register all descriptions
    for elem_id, description in descriptions.items():
        arg_handler.register_description("acestep", elem_id, description) 