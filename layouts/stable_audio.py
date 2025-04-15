import os
import logging
import gradio as gr
import time
import zipfile
from typing import List
import random
import shutil
import uuid
import io
import base64
import json

from handlers.args import ArgHandler
from handlers.config import output_path
from modules.stable_audio import StableAudioModel

logger = logging.getLogger("ADLB.Layout.StableAudio")

# Example prompts to help users get started
EXAMPLE_PROMPTS = [
    "The sound of rain hitting a window, soft and gentle",
    "A powerful thunderstorm with heavy rain and distant thunder",
    "An electronic beat with pulsing bass at 120 BPM",
    "A peaceful forest ambience with birds chirping and leaves rustling",
    "An industrial machinery operating with mechanical sounds",
    "Ocean waves crashing on a rocky shore",
    "A piano melody in a minor key, melancholic and slow",
    "Footsteps walking on a wooden floor in an empty room",
    "A sci-fi spaceship engine humming",
    "A 90s style hip hop drum loop with vinyl crackle"
]

EXAMPLE_NEGATIVE_PROMPTS = [
    "Low quality, distortion, noise, clipping, static",
    "Muddy sound, flat dynamics, background noise",
    "Harsh frequencies, digital artifacts, glitches",
    "Unbalanced mix, mono sound, poor stereo imaging",
    "Robotic voice, computerized sounds, synthetic artifacts"
]

# Global variable for the Send to Process button
SEND_TO_PROCESS_BUTTON = None
OUTPUT_AUDIO = None
OUTPUT_FILES = []
arg_handler = ArgHandler()

# Global dictionaries to store path mappings
path_to_filename = {}
filename_to_path = {}

def get_random_prompt():
    """Return a random example prompt to help users get started."""
    return random.choice(EXAMPLE_PROMPTS)

def get_random_negative_prompt():
    """Return a random example negative prompt."""
    return random.choice(EXAMPLE_NEGATIVE_PROMPTS)

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

def update_output_info(results=None, error=None):
    """Update the output info display with multiple output support."""
    if error:
        return f"Error: {error}", None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(choices=[], visible=False), gr.update(visible=False)
    
    if results and len(results) > 0:
        global OUTPUT_FILES
        OUTPUT_FILES = [result["file_path"] for result in results]
        
        # Update path mappings
        update_path_mappings(OUTPUT_FILES)
        
        # Create display choices using the mapped filenames
        display_choices = [f"Variation {i+1}: {path_to_filename[path]}" for i, path in enumerate(OUTPUT_FILES)]
        
        # Show output selector only if we have multiple variations
        selector_visible = len(results) > 1
        download_all_visible = len(results) > 1
        
        # Return the first audio file as default
        first_file_path = OUTPUT_FILES[0]
        first_display_name = display_choices[0]
        
        return "Generation complete!", first_file_path, gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(choices=display_choices, value=first_display_name, visible=selector_visible), gr.update(visible=download_all_visible), gr.update(value=OUTPUT_FILES, visible=True)
    
    return "No output generated", None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(choices=[], visible=False), gr.update(visible=False), gr.update(visible=False)

def update_audio_preview(selection):
    """Update the audio preview based on the selected variation."""
    if not selection:
        return gr.update(value=None)
    
    # Extract variation number from the selection (format: "Variation X: filename")
    try:
        variation_idx = int(selection.split(':')[0].replace('Variation ', '').strip()) - 1
        if 0 <= variation_idx < len(OUTPUT_FILES):
            return gr.update(value=OUTPUT_FILES[variation_idx])
    except (ValueError, IndexError):
        pass
        
    # Fallback: try to extract the filename and look it up
    try:
        filename = selection.split(':', 1)[1].strip()
        if filename in filename_to_path:
            return gr.update(value=filename_to_path[filename])
    except (IndexError, KeyError):
        pass
    
    # If all else fails, return the first output file if available
    if OUTPUT_FILES:
        return gr.update(value=OUTPUT_FILES[0])
    
    return gr.update(value=None)

def download_all_output_files():
    """Create a zip file of all output files and return the path to download."""
    global OUTPUT_FILES
    
    if not OUTPUT_FILES or len(OUTPUT_FILES) == 0:
        return None
    
    # Create a zip file with all the output files
    if len(OUTPUT_FILES) > 0:
        output_dir = os.path.dirname(OUTPUT_FILES[0])
        zip_filename = os.path.join(output_dir, "stable_audio_outputs.zip")
        
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            for file in OUTPUT_FILES:
                if os.path.exists(file):
                    # Add file to zip with just the filename, not the full path
                    zipf.write(file, os.path.basename(file))
        
        return zip_filename
    
    return None

def render(arg_handler: ArgHandler):
    """Render the Stable Audio tab UI."""
    global SEND_TO_PROCESS_BUTTON, OUTPUT_AUDIO, OUTPUT_FILES
    model = StableAudioModel()
    
    with gr.Column():
        gr.Markdown("# 🎵 Sound Forge - AI Audio Generation")
        gr.Markdown("Generate high-quality sound effects and ambient audio from text descriptions using Stability AI's Stable Audio model.")
        
        with gr.Row():
            # Left Column - Settings
            with gr.Column():
                gr.Markdown("### 🔧 Settings")
                    
                duration = gr.Slider(
                    label="Duration (seconds)",
                    minimum=1.0,
                    maximum=30.0,
                    step=1.0,
                    value=5.0,
                    elem_id="stable_audio_duration",
                    key="stable_audio_duration",
                    elem_classes="hintitem"
                )
                num_waveforms = gr.Slider(
                    label="Number of variations",
                    minimum=1,
                    maximum=3,
                    step=1,
                    value=1,
                    elem_id="stable_audio_num_waveforms",
                    key="stable_audio_num_waveforms",
                    elem_classes="hintitem"
                )
                
                inference_steps = gr.Slider(
                    label="Inference steps",
                    minimum=50,
                    maximum=500,
                    step=10,
                    value=100,
                    elem_id="stable_audio_inference_steps",
                    key="stable_audio_inference_steps",
                    elem_classes="hintitem"
                )
                guidance_scale = gr.Slider(
                    label="Guidance scale",
                    minimum=1.0,
                    maximum=15.0,
                    step=0.5,
                    value=7.0,
                    elem_id="stable_audio_guidance_scale",
                    key="stable_audio_guidance_scale",
                    elem_classes="hintitem"
                )
            
                seed = gr.Number(
                    label="Seed (-1 for random)",
                    value=-1,
                    precision=0,
                    elem_id="stable_audio_seed",
                    key="stable_audio_seed",
                    elem_classes="hintitem"
                )
                randomize_btn = gr.Button(
                    "🎲 Randomize",
                    elem_id="stable_audio_randomize",
                    key="stable_audio_randomize",
                    elem_classes="hintitem"
                )
            
            # Middle Column - Input Data
            with gr.Column():
                gr.Markdown("### 🎤 Inputs")
                
                # Initial audio reference options - moved to top of the column
                use_init_audio = gr.Checkbox(
                    label="Use Audio Reference",
                    value=False,
                    elem_id="stable_audio_use_init_audio",
                    key="stable_audio_use_init_audio",
                    elem_classes="hintitem"
                )
                
                with gr.Column(visible=False) as init_audio_options:
                    init_audio_path = gr.File(
                        label="Reference Audio File",
                        file_count="single",
                        file_types=["audio"],
                        elem_id="stable_audio_init_audio_path",
                        key="stable_audio_init_audio_path",
                        elem_classes="hintitem"
                    )
                    
                    # Add preview player for the reference audio
                    init_audio_preview = gr.Audio(
                        label="Reference Audio Preview",
                        type="filepath",
                        interactive=False,
                        visible=False,
                        elem_id="stable_audio_init_audio_preview",
                        key="stable_audio_init_audio_preview",
                        elem_classes="hintitem"
                    )
                    
                    init_noise_level = gr.Slider(
                        label="Reference Noise Level",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=0.7,
                        elem_id="stable_audio_init_noise_level",
                        key="stable_audio_init_noise_level",
                        elem_classes="hintitem"
                    )
                
                prompt = gr.Textbox(
                    label="Describe the sound you want to generate",
                    placeholder="E.g., A peaceful forest ambience with birds chirping and leaves rustling",
                    lines=3,
                    value="",
                    elem_id="stable_audio_prompt",
                    key="stable_audio_prompt",
                    elem_classes="hintitem"
                )
                
                # Convert examples to accordion
                with gr.Accordion("📋 Example prompts (click to expand)", open=False):
                    prompt_examples = gr.Dataset(
                        components=[prompt],
                        samples=[[p] for p in EXAMPLE_PROMPTS],
                        label="Click an example to use it",
                        elem_id="stable_audio_prompt_examples",
                        key="stable_audio_prompt_examples",
                        elem_classes="hintitem"
                    )
                
                negative_prompt = gr.Textbox(
                    label="What to avoid in the generation",
                    placeholder="E.g., Low quality, distortion, noise",
                    lines=2,
                    value="Low quality, distortion, noise",
                    elem_id="stable_audio_negative_prompt",
                    key="stable_audio_negative_prompt",
                    elem_classes="hintitem"
                )
                
                # Convert negative examples to accordion
                with gr.Accordion("📋 Example negative prompts (click to expand)", open=False):
                    negative_examples = gr.Dataset(
                        components=[negative_prompt],
                        samples=[[p] for p in EXAMPLE_NEGATIVE_PROMPTS],
                        label="Click an example to use it",
                        elem_id="stable_audio_negative_examples",
                        key="stable_audio_negative_examples",
                        elem_classes="hintitem"
                    )
        
            # Right Column - Actions & Output
            with gr.Column():
                gr.Markdown("### 🎮 Actions")
                with gr.Row():
                    generate_btn = gr.Button(
                        "🔊 Generate Audio", 
                        variant="primary",
                        elem_id="stable_audio_generate",
                        key="stable_audio_generate",
                        elem_classes="hintitem"
                    )
                    clear_btn = gr.Button(
                        "🧹 Clear",
                        elem_id="stable_audio_clear",
                        key="stable_audio_clear",
                        elem_classes="hintitem"
                    )
                    SEND_TO_PROCESS_BUTTON = gr.Button(
                        "Send to Process",
                        variant="secondary",
                        visible=False,
                        elem_id="stable_audio_send_to_process",
                        key="stable_audio_send_to_process",
                        elem_classes="hintitem"
                    )
                
                # Output section
                gr.Markdown("### 🎶 Outputs")
                status = gr.Markdown(
                    "Ready to generate",
                    elem_id="stable_audio_status",
                    key="stable_audio_status",
                    elem_classes="hintitem"
                )
                
                # Add output selector dropdown for multiple variations
                output_selector = gr.Dropdown(
                    label="Select Variation",
                    choices=[],
                    value=None,
                    visible=False,
                    elem_id="stable_audio_output_selector",
                    key="stable_audio_output_selector",
                    elem_classes="hintitem",
                    interactive=True
                )
                
                OUTPUT_AUDIO = gr.Audio(
                    label="Generated Audio",
                    type="filepath",
                    interactive=False,
                    elem_id="stable_audio_output",
                    key="stable_audio_output",
                    elem_classes="hintitem"
                )
                
                # Add output files component to show all generated files
                output_files = gr.File(
                    label="Generated Files",
                    file_count="multiple",
                    visible=False,
                    interactive=False,
                    elem_id="stable_audio_output_files",
                    key="stable_audio_output_files",
                    elem_classes="hintitem"
                )
                
                with gr.Row():
                    download_btn = gr.Button(
                        "💾 Download", 
                        visible=False,
                        elem_id="stable_audio_download",
                        key="stable_audio_download",
                        elem_classes="hintitem"
                    )
                    download_all_btn = gr.Button(
                        "💾 Download All",
                        visible=False,
                        variant="secondary",
                        elem_id="stable_audio_download_all",
                        key="stable_audio_download_all",
                        elem_classes="hintitem"
                    )
                    use_as_reference_btn = gr.Button(
                        "🔄 Use As Reference",
                        visible=False,
                        elem_id="stable_audio_use_as_reference",
                        key="stable_audio_use_as_reference",
                        elem_classes="hintitem"
                    )
                
                
    # Define functions for button actions
    def randomize_seed():
        return random.randint(0, 2147483647)
    
    def clear_outputs():
        """Clear all inputs and outputs, resetting the interface."""
        return "", get_random_negative_prompt(), 5.0, 1, 100, 7.0, -1, False, None, 0.7, "Ready to generate", None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(choices=[], visible=False), gr.update(visible=False), gr.update(visible=False)
    
    def toggle_init_audio(use_init):
        """Toggle visibility of the init audio options and reset preview if disabled."""
        if not use_init:
            return gr.update(visible=False), gr.update(value=None, visible=False)
        return gr.update(visible=True), gr.update(visible=False)  # Preview will be updated when file is selected
    
    def update_init_audio_preview(init_audio_path):
        """Update the preview of the reference audio file."""
        if not init_audio_path:
            return gr.update(value=None, visible=False)
        
        file_path = init_audio_path.name if hasattr(init_audio_path, 'name') else init_audio_path
        
        if file_path and os.path.exists(file_path):
            return gr.update(value=file_path, visible=True)
        return gr.update(value=None, visible=False)
    
    def use_output_as_reference(selected_variation):
        """Set the currently selected output as the reference input."""
        # Get the actual file path for the selected variation
        file_path = None
        if selected_variation:
            print(f"Selected variation: {selected_variation}")
            try:
                variation_idx = int(selected_variation.split(':')[0].replace('Variation ', '').strip()) - 1
                if 0 <= variation_idx < len(OUTPUT_FILES):
                    file_path = OUTPUT_FILES[variation_idx]
                    print(f"File path: {file_path}")
            except (ValueError, IndexError):
                pass
            
            # Fallback: try to extract the filename and look it up
            if not file_path:
                try:
                    filename = selected_variation.split(':', 1)[1].strip()
                    if filename in filename_to_path:
                        file_path = filename_to_path[filename]
                        print(f"File path: {file_path}")
                    else:
                        print(f"File path not found in filename_to_path: {filename}")
                except (IndexError, KeyError):
                    pass
        
        # If no selection or couldn't find file, use the first one if available
        if not file_path and OUTPUT_FILES:
            file_path = OUTPUT_FILES[0]
            print(f"File path: {file_path}")
        if file_path and os.path.exists(file_path):
            print(f"File path exists: {file_path}")
            # Enable the checkbox and set the file input with the correct format
            return (
                gr.update(value=True),
                gr.update(value=file_path),
                gr.update(value=file_path, visible=True)
            )
        return gr.update(), gr.update(), gr.update(visible=False)
    
    def generate_audio(progress=gr.Progress(), prompt="", negative_prompt="", duration=5.0, 
                      num_waveforms=1, inference_steps=100, guidance_scale=7.0, seed=-1,
                      use_init_audio=False, init_audio_path=None, init_noise_level=0.7):
        """Generate audio based on the text prompt and parameters."""
        try:
            progress(0, desc="Initializing...")
            
            # Validate inputs
            if not prompt.strip():
                return "Error: Please provide a text prompt", None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(choices=[], visible=False), gr.update(visible=False)
            
            # Process the audio generation
            progress(0.1, desc="Loading model...")
            
            # Determine if we should use init audio
            init_audio_file = None
            if use_init_audio and init_audio_path:
                # Check if init_audio_path is a string or has a 'name' attribute (File component)
                if isinstance(init_audio_path, str):
                    init_audio_file = init_audio_path
                elif hasattr(init_audio_path, 'name'):
                    init_audio_file = init_audio_path.name
                
                if init_audio_file:
                    progress(0.2, desc="Processing reference audio...")
            
            results = model.generate_audio(
                prompt=prompt,
                negative_prompt=negative_prompt,
                duration=duration,
                num_waveforms=num_waveforms,
                num_inference_steps=inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                init_audio_path=init_audio_file,
                init_noise_level=init_noise_level,
                callback=progress
            )
            
            return update_output_info(results)
        
        except Exception as e:
            logger.error(f"Error in generate_audio: {e}")
            return update_output_info(error=str(e))
    
    # Set up button click events
    generate_btn.click(
        fn=generate_audio,
        inputs=[prompt, negative_prompt, duration, num_waveforms, 
                inference_steps, guidance_scale, seed,
                use_init_audio, init_audio_path, init_noise_level],
        outputs=[status, OUTPUT_AUDIO, download_btn, SEND_TO_PROCESS_BUTTON, use_as_reference_btn, output_selector, download_all_btn, output_files]
    )
    
    clear_btn.click(
        fn=clear_outputs,
        inputs=[],
        outputs=[prompt, negative_prompt, duration, num_waveforms, 
                inference_steps, guidance_scale, seed,
                use_init_audio, init_audio_path, init_noise_level,
                status, OUTPUT_AUDIO, download_btn, SEND_TO_PROCESS_BUTTON, use_as_reference_btn, output_selector, download_all_btn, output_files, init_audio_preview]
    )
    
    randomize_btn.click(
        fn=randomize_seed,
        inputs=[],
        outputs=[seed]
    )
    
    # Toggle initial audio options visibility
    use_init_audio.change(
        fn=toggle_init_audio,
        inputs=[use_init_audio],
        outputs=[init_audio_options, init_audio_preview]
    )
    
    # Update preview when reference file is selected
    init_audio_path.change(
        fn=update_init_audio_preview,
        inputs=[init_audio_path],
        outputs=[init_audio_preview]
    )
    
    # Setup dataset example selection
    prompt_examples.click(
        fn=lambda x: x[0],
        inputs=[prompt_examples],
        outputs=[prompt]
    )
    
    negative_examples.click(
        fn=lambda x: x[0],
        inputs=[negative_examples],
        outputs=[negative_prompt]
    )

    # Add click handler for the "Use As Reference" button
    use_as_reference_btn.click(
        fn=use_output_as_reference,
        inputs=[output_selector],
        outputs=[use_init_audio, init_audio_path]
    )
    
    # Add change event for the output selector
    output_selector.change(
        fn=update_audio_preview,
        inputs=[output_selector],
        outputs=[OUTPUT_AUDIO]
    )

    # Add click handler for the "Use As Reference" button - now properly updates all three components
    use_as_reference_btn.click(
        fn=use_output_as_reference,
        inputs=[output_selector],
        outputs=[use_init_audio, init_audio_path, init_audio_preview]
    )
    
    # Add change event for the output selector
    output_selector.change(
        fn=update_audio_preview,
        inputs=[output_selector],
        outputs=[OUTPUT_AUDIO]
    )
    
    # Add download all button click event
    download_all_btn.click(
        fn=download_all_output_files,
        inputs=[],
        outputs=[output_files]
    )
    
    # Add Send to Process button click event directly here
    process_inputs = arg_handler.get_element("main", "process_inputs")
    if process_inputs:
        SEND_TO_PROCESS_BUTTON.click(
            fn=send_to_process, 
            inputs=[output_selector, process_inputs], 
            outputs=[process_inputs]
        )

def register_descriptions(arg_handler: ArgHandler):
    """Register descriptions for the Stable Audio module."""
    descriptions = {
        "prompt": "Describe the sound you want to generate in detail. The more specific, the better the results.",
        "negative_prompt": "Describe characteristics you want to avoid in the generated audio.",
        "duration": "Set the length of the generated audio in seconds (max 30 seconds).",
        "num_waveforms": "Number of variations to generate from the same prompt.",
        "inference_steps": "More steps can improve quality but take longer to generate.",
        "guidance_scale": "How closely to follow the prompt. Higher values are more accurate but may be less creative.",
        "seed": "Use -1 for random, or set a specific value for reproducible results.",
        "use_init_audio": "Enable to use an existing audio file as reference for generation.",
        "init_audio_path": "Upload an audio file to use as guidance for the generation.",
        "init_noise_level": "Controls how much of the reference audio to preserve (0=exact copy, 1=complete recreation)."
    }
    
    for elem_id, description in descriptions.items():
        arg_handler.register_description("stable_audio", elem_id, description)

def listen():
    """Set up event listeners for the Stable Audio tab."""
    # The Send to Process button click is now handled in the render function
    pass

def send_to_process(selected_variation, process_inputs):
    """Send the selected audio variation to the Process tab."""
    # Get the actual file path for the selected variation
    file_path = None
    if selected_variation:
        try:
            variation_idx = int(selected_variation.split(':')[0].replace('Variation ', '').strip()) - 1
            if 0 <= variation_idx < len(OUTPUT_FILES):
                file_path = OUTPUT_FILES[variation_idx]
        except (ValueError, IndexError):
            pass
        
        # Fallback: try to extract the filename and look it up
        if not file_path:
            try:
                filename = selected_variation.split(':', 1)[1].strip()
                if filename in filename_to_path:
                    file_path = filename_to_path[filename]
            except (IndexError, KeyError):
                pass
    
    # If no selection or couldn't find file, use the first one if available
    if not file_path and OUTPUT_FILES:
        file_path = OUTPUT_FILES[0]
        
    if not file_path or not os.path.exists(file_path):
        return gr.update()
    
    if file_path in process_inputs:
        return gr.update()
        
    process_inputs.append(file_path)
    return gr.update(value=process_inputs)

def register_api_endpoints(api):
    """
    Register API endpoints for stable audio generation functionality
    
    Args:
        api: FastAPI application instance
    """
    from fastapi import HTTPException, BackgroundTasks, Form, Query, UploadFile, File
    from fastapi.responses import JSONResponse, FileResponse
    from fastapi.encoders import jsonable_encoder
    from pydantic import BaseModel, Field, validator, root_validator
    from typing import Optional, List, Dict, Any, Union
    import os
    import uuid
    import time
    import json
    import re
    from io import BytesIO

    # Define Pydantic models for request validation
    class StableAudioGenerationOptions(BaseModel):
        """Request model for audio generation"""
        prompt: str = Field(
            ..., description="The text prompt describing the desired audio", min_length=1, max_length=1000
        )
        negative_prompt: Optional[str] = Field(
            None, description="Text describing what should be avoided in the audio"
        )
        duration_seconds: float = Field(
            10.0, description="Duration of the generated audio in seconds", ge=1.0, le=120.0
        )
        output_format: str = Field(
            "mp3", description="The format of the output audio file"
        )
        seed: Optional[int] = Field(
            None, description="Random seed for generation reproducibility"
        )
        num_outputs: int = Field(
            1, description="Number of audio outputs to generate", ge=1, le=4
        )
        model: str = Field(
            "stable-audio-1", description="The model to use for audio generation"
        )
        
        @validator('prompt')
        def validate_prompt(cls, v):
            if not v or not v.strip():
                raise ValueError("Prompt cannot be empty")
            if len(v) > 1000:
                raise ValueError("Prompt cannot exceed 1000 characters")
            return v.strip()
        
        @validator('negative_prompt')
        def validate_negative_prompt(cls, v):
            if v is not None and len(v) > 1000:
                raise ValueError("Negative prompt cannot exceed 1000 characters")
            return v.strip() if v else None
        
        @validator('duration_seconds')
        def validate_duration(cls, v):
            if v < 1.0:
                raise ValueError("Duration must be at least 1 second")
            if v > 120.0:
                raise ValueError("Duration cannot exceed 120 seconds (2 minutes)")
            return v
        
        @validator('output_format')
        def validate_output_format(cls, v):
            valid_formats = ["mp3", "wav", "flac", "ogg"]
            if v not in valid_formats:
                raise ValueError(f"Output format must be one of: {', '.join(valid_formats)}")
            return v
        
        @validator('seed')
        def validate_seed(cls, v):
            if v is not None and (v < 0 or v > 2**32-1):
                raise ValueError("Seed must be between 0 and 4294967295")
            return v
        
        @validator('num_outputs')
        def validate_num_outputs(cls, v):
            if v < 1:
                raise ValueError("Number of outputs must be at least 1")
            if v > 4:
                raise ValueError("Number of outputs cannot exceed 4")
            return v
        
        @validator('model')
        def validate_model(cls, v):
            valid_models = ["stable-audio-1", "stable-audio-1-high"]
            if v not in valid_models:
                raise ValueError(f"Model must be one of: {', '.join(valid_models)}")
            return v

    class ContinuationOptions(BaseModel):
        """Request model for audio continuation"""
        audio_file: str = Field(..., description="Base64 encoded audio file to continue from")
        prompt: Optional[str] = Field(
            None, description="Optional text prompt to guide the continuation"
        )
        continuation_seconds: float = Field(
            10.0, description="Duration of the continuation in seconds", ge=1.0, le=120.0
        )
        output_format: str = Field(
            "mp3", description="The format of the output audio file"
        )
        seed: Optional[int] = Field(
            None, description="Random seed for generation reproducibility"
        )
        model: str = Field(
            "stable-audio-1", description="The model to use for audio continuation"
        )
        
        @validator('audio_file')
        def validate_audio_file(cls, v):
            if not v:
                raise ValueError("Audio file reference cannot be empty")
            try:
                # Try to decode base64 to validate format
                base64.b64decode(v)
            except Exception as e:
                raise ValueError(f"Invalid base64 audio data: {str(e)}")
            return v
        
        @validator('prompt')
        def validate_prompt(cls, v):
            if v is not None and len(v) > 1000:
                raise ValueError("Prompt cannot exceed 1000 characters")
            return v.strip() if v else None
        
        @validator('continuation_seconds')
        def validate_continuation_duration(cls, v):
            if v < 1.0:
                raise ValueError("Continuation duration must be at least 1 second")
            if v > 120.0:
                raise ValueError("Continuation duration cannot exceed 120 seconds (2 minutes)")
            return v
        
        @validator('output_format')
        def validate_output_format(cls, v):
            valid_formats = ["mp3", "wav", "flac", "ogg"]
            if v not in valid_formats:
                raise ValueError(f"Output format must be one of: {', '.join(valid_formats)}")
            return v
        
        @validator('seed')
        def validate_seed(cls, v):
            if v is not None and (v < 0 or v > 2**32-1):
                raise ValueError("Seed must be between 0 and 4294967295")
            return v
        
        @validator('model')
        def validate_model(cls, v):
            valid_models = ["stable-audio-1", "stable-audio-1-high"]
            if v not in valid_models:
                raise ValueError(f"Model must be one of: {', '.join(valid_models)}")
            return v

    @api.post("/api/v1/audio/generate", tags=["Stable Audio"])
    async def api_generate_audio(
        request: StableAudioGenerationOptions,
        background_tasks: BackgroundTasks = None
    ):
        """
        Generate audio from a text prompt
        
        This endpoint generates audio based on the provided text prompt using Stable Audio models.
        """
        try:
            # Generate a unique ID for this audio generation
            generation_id = str(uuid.uuid4())
            timestamp = int(time.time())
            
            # Create output directory if it doesn't exist
            audio_dir = os.path.join(output_path, "stable_audio")
            os.makedirs(audio_dir, exist_ok=True)
            
            # Initialize Stable Audio engine
            stable_audio = StableAudioEngine()
            
            # Generate audio
            result = stable_audio.generate_audio(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                duration_seconds=request.duration_seconds,
                seed=request.seed,
                num_outputs=request.num_outputs,
                model=request.model
            )
            
            if not result.get("success", False):
                raise HTTPException(status_code=500, detail=result.get("error", "Failed to generate audio"))
            
            # Get generated audio data
            audio_data_list = result.get("audio_data", [])
            if not audio_data_list:
                raise HTTPException(status_code=500, detail="No audio data generated")
            
            # Save audio files and create download URLs
            output_files = []
            for i, audio_data in enumerate(audio_data_list):
                output_filename = f"stable_audio_{generation_id}_{i+1}.{request.output_format}"
                output_filepath = os.path.join(audio_dir, output_filename)
                
                # Save audio to file
                with open(output_filepath, "wb") as f:
                    f.write(audio_data)
                
                # Create download URL
                download_url = f"/api/v1/audio/download/{output_filename}"
                
                # Get file size
                file_size = os.path.getsize(output_filepath)
                
                output_files.append({
                    "id": f"{generation_id}_{i+1}",
                    "filename": output_filename,
                    "download_url": download_url,
                    "file_size_bytes": file_size
                })
                
                # Schedule file deletion after 24 hours
                if background_tasks:
                    background_tasks.add_task(
                        lambda p: os.remove(p) if os.path.exists(p) else None,
                        output_filepath,
                        delay=86400  # 24 hours
                    )
            
            # Save metadata
            metadata_filename = f"stable_audio_{generation_id}_metadata.json"
            metadata_filepath = os.path.join(audio_dir, metadata_filename)
            
            # Truncate prompt for metadata if too long
            prompt_preview = request.prompt
            if len(prompt_preview) > 100:
                prompt_preview = prompt_preview[:100] + "..."
            
            negative_prompt_preview = None
            if request.negative_prompt:
                negative_prompt_preview = request.negative_prompt
                if len(negative_prompt_preview) > 100:
                    negative_prompt_preview = negative_prompt_preview[:100] + "..."
            
            metadata = {
                "id": generation_id,
                "timestamp": timestamp,
                "prompt": prompt_preview,
                "negative_prompt": negative_prompt_preview,
                "duration_seconds": request.duration_seconds,
                "output_format": request.output_format,
                "seed": request.seed if request.seed is not None else "random",
                "num_outputs": request.num_outputs,
                "model": request.model,
                "output_files": output_files
            }
            
            with open(metadata_filepath, "w") as f:
                json.dump(metadata, f, indent=2)
            
            # Schedule metadata file deletion after 24 hours
            if background_tasks:
                background_tasks.add_task(
                    lambda p: os.remove(p) if os.path.exists(p) else None,
                    metadata_filepath,
                    delay=86400  # 24 hours
                )
            
            # Prepare response
            response_data = {
                "success": True,
                "generation_id": generation_id,
                "outputs": output_files,
                "metadata": {
                    "prompt": prompt_preview,
                    "negative_prompt": negative_prompt_preview,
                    "duration_seconds": request.duration_seconds,
                    "seed": request.seed,
                    "num_outputs": request.num_outputs,
                    "model": request.model,
                    "timestamp": timestamp
                }
            }
            
            return JSONResponse(content=jsonable_encoder(response_data))
            
        except ValueError as e:
            # Handle validation errors
            raise HTTPException(status_code=400, detail=str(e))
        except HTTPException:
            raise
        except Exception as e:
            import traceback
            logger.exception(f"API audio generation error: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=str(e))

    @api.post("/api/v1/audio/continue", tags=["Stable Audio"])
    async def api_continue_audio(
        request: ContinuationOptions,
        background_tasks: BackgroundTasks = None
    ):
        """
        Generate a continuation from an existing audio clip
        
        This endpoint creates a continuation of an uploaded audio file using Stable Audio models.
        """
        try:
            # Decode base64 audio data
            audio_data = base64.b64decode(request.audio_file)
            file_size = len(audio_data)
            
            # Check file size limit (25MB)
            file_size_limit = 25 * 1024 * 1024  # 25MB in bytes
            if file_size > file_size_limit:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Audio file exceeds size limit of 25MB. Uploaded file size: {file_size / (1024 * 1024):.2f}MB"
                )
            
            # Create a temporary file to store the uploaded audio
            temp_dir = os.path.join(output_path, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            
            temp_audio_id = str(uuid.uuid4())
            temp_audio_filename = f"temp_audio_{temp_audio_id}.wav"
            temp_audio_path = os.path.join(temp_dir, temp_audio_filename)
            
            with open(temp_audio_path, "wb") as f:
                f.write(audio_data)
            
            # Generate a unique ID for this continuation
            continuation_id = str(uuid.uuid4())
            timestamp = int(time.time())
            
            # Create output directory if it doesn't exist
            audio_dir = os.path.join(output_path, "stable_audio")
            os.makedirs(audio_dir, exist_ok=True)
            
            # Initialize Stable Audio engine
            stable_audio = StableAudioEngine()
            
            # Generate continuation
            result = stable_audio.continue_audio(
                audio_file=temp_audio_path,
                prompt=request.prompt,
                continuation_seconds=request.continuation_seconds,
                seed=request.seed,
                model=request.model
            )
            
            # Clean up temporary file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            
            if not result.get("success", False):
                raise HTTPException(status_code=500, detail=result.get("error", "Failed to generate audio continuation"))
            
            # Get generated audio data
            audio_data = result.get("audio_data")
            if not audio_data:
                raise HTTPException(status_code=500, detail="No audio data generated")
            
            # Set output filename and path
            output_filename = f"stable_audio_cont_{continuation_id}.{request.output_format}"
            output_filepath = os.path.join(audio_dir, output_filename)
            
            # Save audio to file
            with open(output_filepath, "wb") as f:
                f.write(audio_data)
            
            # Create download URL
            download_url = f"/api/v1/audio/download/{output_filename}"
            
            # Get file size
            file_size = os.path.getsize(output_filepath)
            
            # Save combined audio if available
            combined_output_file = None
            if result.get("combined_audio_data"):
                combined_filename = f"stable_audio_cont_{continuation_id}_combined.{request.output_format}"
                combined_filepath = os.path.join(audio_dir, combined_filename)
                
                with open(combined_filepath, "wb") as f:
                    f.write(result["combined_audio_data"])
                
                combined_download_url = f"/api/v1/audio/download/{combined_filename}"
                combined_file_size = os.path.getsize(combined_filepath)
                
                combined_output_file = {
                    "id": f"{continuation_id}_combined",
                    "filename": combined_filename,
                    "download_url": combined_download_url,
                    "file_size_bytes": combined_file_size
                }
                
                # Schedule file deletion after 24 hours
                if background_tasks:
                    background_tasks.add_task(
                        lambda p: os.remove(p) if os.path.exists(p) else None,
                        combined_filepath,
                        delay=86400  # 24 hours
                    )
            
            # Save metadata
            metadata_filename = f"stable_audio_cont_{continuation_id}_metadata.json"
            metadata_filepath = os.path.join(audio_dir, metadata_filename)
            
            # Truncate prompt for metadata if too long
            prompt_preview = None
            if request.prompt:
                prompt_preview = request.prompt
                if len(prompt_preview) > 100:
                    prompt_preview = prompt_preview[:100] + "..."
            
            metadata = {
                "id": continuation_id,
                "timestamp": timestamp,
                "original_filename": temp_audio_filename,
                "prompt": prompt_preview,
                "continuation_seconds": request.continuation_seconds,
                "output_format": request.output_format,
                "seed": request.seed if request.seed is not None else "random",
                "model": request.model,
                "output_file": {
                    "id": continuation_id,
                    "filename": output_filename,
                    "download_url": download_url,
                    "file_size_bytes": file_size
                },
                "combined_output_file": combined_output_file
            }
            
            with open(metadata_filepath, "w") as f:
                json.dump(metadata, f, indent=2)
            
            # Schedule file deletions after 24 hours
            if background_tasks:
                background_tasks.add_task(
                    lambda p: os.remove(p) if os.path.exists(p) else None,
                    output_filepath,
                    delay=86400  # 24 hours
                )
                
                background_tasks.add_task(
                    lambda p: os.remove(p) if os.path.exists(p) else None,
                    metadata_filepath,
                    delay=86400  # 24 hours
                )
            
            # Prepare response
            response_data = {
                "success": True,
                "continuation_id": continuation_id,
                "continuation_file": {
                    "filename": output_filename,
                    "download_url": download_url,
                    "file_size_bytes": file_size
                },
                "combined_file": combined_output_file,
                "metadata": {
                    "original_filename": temp_audio_filename,
                    "prompt": prompt_preview,
                    "continuation_seconds": request.continuation_seconds,
                    "seed": request.seed,
                    "model": request.model,
                    "timestamp": timestamp
                }
            }
            
            return JSONResponse(content=jsonable_encoder(response_data))
            
        except ValueError as e:
            # Handle validation errors
            raise HTTPException(status_code=400, detail=str(e))
        except HTTPException:
            raise
        except Exception as e:
            import traceback
            logger.exception(f"API audio continuation error: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=str(e))

    @api.get("/api/v1/audio/download/{filename}", tags=["Stable Audio"])
    async def api_download_audio(filename: str):
        """
        Download a generated audio file
        
        Parameters:
        - filename: The name of the audio file to download
        
        Returns:
        - The audio file
        """
        try:
            # Validate filename format (prevent path traversal)
            valid_pattern = r'^(stable_audio|stable_audio_cont)_[a-f0-9-]+(_\d+|_combined)?.(mp3|wav|flac|ogg)$'
            if not re.match(valid_pattern, filename):
                raise HTTPException(status_code=400, detail="Invalid filename format")
            
            # Build the file path
            file_path = os.path.join(output_path, "stable_audio", filename)
            
            # Check if file exists
            if not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail="Audio file not found")
            
            # Determine content type based on file extension
            file_ext = filename.split(".")[-1].lower()
            content_type_map = {
                "mp3": "audio/mpeg",
                "wav": "audio/wav",
                "flac": "audio/flac",
                "ogg": "audio/ogg"
            }
            
            content_type = content_type_map.get(file_ext, "application/octet-stream")
            
            # Return the file
            return FileResponse(
                path=file_path,
                filename=filename,
                media_type=content_type
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"API audio download error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @api.get("/api/v1/audio/models", tags=["Stable Audio"])
    async def api_get_audio_models():
        """
        Get available audio generation models
        
        Returns information about available models for audio generation.
        """
        models = [
            {
                "id": "stable-audio-1",
                "name": "Stable Audio 1",
                "description": "Standard audio generation model optimized for quality and speed",
                "max_duration_seconds": 120,
                "supported_formats": ["mp3", "wav", "flac", "ogg"]
            },
            {
                "id": "stable-audio-1-high",
                "name": "Stable Audio 1 High",
                "description": "High-quality audio generation model with improved audio fidelity",
                "max_duration_seconds": 120,
                "supported_formats": ["mp3", "wav", "flac", "ogg"]
            }
        ]
        return {"models": models}

    @api.get("/api/v1/audio/formats", tags=["Stable Audio"])
    async def api_get_audio_formats():
        """
        Get available audio formats
        
        Returns information about supported output formats for audio generation.
        """
        formats = [
            {
                "id": "mp3",
                "name": "MP3",
                "description": "Compressed audio format with good quality and small file size (default)",
                "mime_type": "audio/mpeg",
                "extension": ".mp3"
            },
            {
                "id": "wav",
                "name": "WAV",
                "description": "Uncompressed audio format with high quality and larger file size",
                "mime_type": "audio/wav",
                "extension": ".wav"
            },
            {
                "id": "flac",
                "name": "FLAC",
                "description": "Lossless audio compression format with excellent quality",
                "mime_type": "audio/flac",
                "extension": ".flac"
            },
            {
                "id": "ogg",
                "name": "OGG",
                "description": "Open container format supporting various codecs like Vorbis and Opus",
                "mime_type": "audio/ogg",
                "extension": ".ogg"
            }
        ]
        return {"formats": formats}

def delete_temp_file(file_path: str, delay: int = 3600):
    """Delete a temporary file after a delay"""
    try:
        time.sleep(delay)
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        logger.error(f"Error deleting temporary file {file_path}: {e}") 
