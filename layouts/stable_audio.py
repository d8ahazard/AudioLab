import os
import logging
import gradio as gr
from pathlib import Path
import tempfile
import time
import zipfile
from typing import List
import random
import shutil
import base64

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
    Register FastAPI endpoints for Stable Audio generation.
    
    Args:
        api: FastAPI application instance
    """
    from fastapi import UploadFile, File, Form, BackgroundTasks, HTTPException, Body
    from fastapi.responses import FileResponse, JSONResponse
    from typing import Optional
    from pydantic import BaseModel, Field
    
    # Define Pydantic models for JSON requests
    class FileData(BaseModel):
        filename: str
        content: str  # base64 encoded content
        
    class GenerateAudioRequest(BaseModel):
        prompt: str
        negative_prompt: str = ""
        duration: float = 5.0
        num_waveforms: int = 1
        inference_steps: int = 100
        guidance_scale: float = 7.0
        seed: int = -1
        use_reference_audio: bool = False
        reference_audio: Optional[FileData] = None
        noise_level: float = 0.7
    
    @api.post("/api/v1/stable-audio/generate", tags=["Stable Audio"])
    async def api_generate_audio(
        background_tasks: BackgroundTasks,
        prompt: str = Form(...),
        negative_prompt: str = Form(""),
        duration: float = Form(5.0),
        num_waveforms: int = Form(1),
        inference_steps: int = Form(100),
        guidance_scale: float = Form(7.0),
        seed: int = Form(-1),
        use_reference_audio: bool = Form(False),
        reference_audio: Optional[UploadFile] = File(None),
        noise_level: float = Form(0.7)
    ):
        """
        Generate audio from text using Stable Audio
        
        Args:
            background_tasks: FastAPI background tasks
            prompt: Text description of the desired audio
            negative_prompt: Text description of what to avoid
            duration: Duration of the generated audio in seconds
            num_waveforms: Number of audio samples to generate
            inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for classifier-free guidance
            seed: Random seed for generation (-1 for random)
            use_reference_audio: Whether to use a reference audio file
            reference_audio: Optional reference audio file
            noise_level: Amount of noise to add to reference audio (0.0-1.0)
            
        Returns:
            Dictionary containing information about generated audio files
        """
        try:
            # Validate inputs
            if not prompt or not prompt.strip():
                raise HTTPException(status_code=400, detail="Prompt cannot be empty")
                
            if duration < 1.0 or duration > 47.0:
                raise HTTPException(
                    status_code=400, 
                    detail="Duration must be between 1.0 and 47.0 seconds"
                )
                
            # Create temporary directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir_path = Path(temp_dir)
                
                # Save reference audio if provided
                reference_audio_path = None
                if use_reference_audio and reference_audio:
                    ref_file_path = temp_dir_path / reference_audio.filename
                    with ref_file_path.open("wb") as f:
                        content = await reference_audio.read()
                        f.write(content)
                    reference_audio_path = str(ref_file_path)
                
                # Generate audio with the shared implementation
                return await _generate_audio_impl(
                    background_tasks=background_tasks,
                    temp_dir=temp_dir,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    duration=duration,
                    num_waveforms=num_waveforms,
                    inference_steps=inference_steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                    reference_audio_path=reference_audio_path,
                    noise_level=noise_level,
                    return_json=False
                )
                
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            logger.exception("Error in Stable Audio generation:")
            raise HTTPException(status_code=500, detail=f"Audio generation error: {str(e)}")
    
    @api.post("/api/v1/stable-audio/generate_json", tags=["Stable Audio"])
    async def api_generate_audio_json(
        background_tasks: BackgroundTasks,
        request: GenerateAudioRequest = Body(...)
    ):
        """
        Generate audio from text using Stable Audio (JSON API)
        
        Request body:
        - prompt: Text description of the desired audio
        - negative_prompt: Text description of what to avoid (default: "")
        - duration: Duration of the generated audio in seconds (default: 5.0)
        - num_waveforms: Number of audio samples to generate (default: 1)
        - inference_steps: Number of denoising steps (default: 100)
        - guidance_scale: Guidance scale for classifier-free guidance (default: 7.0)
        - seed: Random seed for generation, -1 for random (default: -1)
        - use_reference_audio: Whether to use a reference audio file (default: false)
        - reference_audio: Optional reference audio object with filename and base64-encoded content
        - noise_level: Amount of noise to add to reference audio, 0.0-1.0 (default: 0.7)
        
        Returns:
        - JSON response with base64-encoded audio files
        """
        try:
            # Validate inputs
            if not request.prompt or not request.prompt.strip():
                raise HTTPException(status_code=400, detail="Prompt cannot be empty")
                
            if request.duration < 1.0 or request.duration > 47.0:
                raise HTTPException(
                    status_code=400, 
                    detail="Duration must be between 1.0 and 47.0 seconds"
                )
                
            # Create temporary directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir_path = Path(temp_dir)
                
                # Save reference audio if provided
                reference_audio_path = None
                if request.use_reference_audio and request.reference_audio:
                    ref_file_path = temp_dir_path / request.reference_audio.filename
                    with ref_file_path.open("wb") as f:
                        content = base64.b64decode(request.reference_audio.content)
                        f.write(content)
                    reference_audio_path = str(ref_file_path)
                
                # Generate audio with the shared implementation
                return await _generate_audio_impl(
                    background_tasks=background_tasks,
                    temp_dir=temp_dir,
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt,
                    duration=request.duration,
                    num_waveforms=request.num_waveforms,
                    inference_steps=request.inference_steps,
                    guidance_scale=request.guidance_scale,
                    seed=request.seed,
                    reference_audio_path=reference_audio_path,
                    noise_level=request.noise_level,
                    return_json=True
                )
                
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            logger.exception("Error in Stable Audio generation:")
            raise HTTPException(status_code=500, detail=f"Audio generation error: {str(e)}")
    
    async def _generate_audio_impl(
        background_tasks,
        temp_dir,
        prompt,
        negative_prompt,
        duration,
        num_waveforms,
        inference_steps,
        guidance_scale,
        seed,
        reference_audio_path,
        noise_level,
        return_json=False
    ):
        """Shared implementation for audio generation"""
        # Initialize model
        model = StableAudioModel()
        
        # Generate audio
        results = model.generate_audio(
            prompt=prompt,
            negative_prompt=negative_prompt,
            duration=duration,
            num_waveforms=num_waveforms,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            init_audio_path=reference_audio_path,
            init_noise_level=noise_level
        )
        
        if not results:
            raise HTTPException(
                status_code=500,
                detail="No output generated. Audio generation failed."
            )
        
        # Prepare response with file information
        response = {
            "status": "success",
            "message": f"Generated {len(results)} audio file(s)",
            "outputs": []
        }
        
        if return_json:
            # Return the outputs as base64-encoded content
            for result in results:
                src_path = result["file_path"]
                file_name = os.path.basename(src_path)
                
                with open(src_path, "rb") as f:
                    file_content = base64.b64encode(f.read()).decode("utf-8")
                
                response["outputs"].append({
                    "filename": file_name,
                    "content": file_content,
                    "seed": result["seed"],
                    "duration": duration,
                    "sample_rate": result["sample_rate"]
                })
            
            # If multiple files, create a zip file
            if len(results) > 1:
                zip_filename = f"stable_audio_{int(time.time())}.zip"
                zip_path = os.path.join(temp_dir, zip_filename)
                
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for result in results:
                        src_path = result["file_path"]
                        zipf.write(src_path, os.path.basename(src_path))
                
                # Add zip file to response as base64
                with open(zip_path, "rb") as f:
                    zip_content = base64.b64encode(f.read()).decode("utf-8")
                
                response["zip_file"] = {
                    "filename": zip_filename,
                    "content": zip_content
                }
            
            return response
        else:
            # Store files for download
            temp_api_dir = os.path.join(output_path, "temp_api", "stable_audio")
            os.makedirs(temp_api_dir, exist_ok=True)
            
            # Copy files to download location and add to response
            for result in results:
                src_path = result["file_path"]
                file_name = os.path.basename(src_path)
                dst_path = os.path.join(temp_api_dir, file_name)
                
                # Copy file
                shutil.copy2(src_path, dst_path)
                
                # Schedule cleanup
                background_tasks.add_task(
                    delete_temp_file,
                    dst_path, 
                    delay=3600  # 1 hour
                )
                
                # Add to response
                response["outputs"].append({
                    "filename": file_name,
                    "download_url": f"/api/v1/stable-audio/download/{file_name}",
                    "seed": result["seed"],
                    "duration": duration,
                    "sample_rate": result["sample_rate"]
                })
            
            # If multiple files, create a zip file
            if len(results) > 1:
                zip_filename = f"stable_audio_{int(time.time())}.zip"
                zip_path = os.path.join(temp_api_dir, zip_filename)
                
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for result in results:
                        src_path = result["file_path"]
                        zipf.write(src_path, os.path.basename(src_path))
                
                # Schedule cleanup
                background_tasks.add_task(
                    delete_temp_file,
                    zip_path, 
                    delay=3600  # 1 hour
                )
                
                # Add zip file to response
                response["zip_file"] = {
                    "filename": zip_filename,
                    "download_url": f"/api/v1/stable-audio/download/{zip_filename}"
                }
            
            return response
    
    @api.get("/api/v1/stable-audio/download/{file_name}", tags=["Stable Audio"])
    async def download_stable_audio_file(file_name: str):
        """
        Download a generated Stable Audio file
        
        Args:
            file_name: Name of the file to download
            
        Returns:
            Audio file as a response
        """
        file_path = os.path.join(output_path, "temp_api", "stable_audio", file_name)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
            
        # Determine content type based on file extension
        if file_name.lower().endswith(".zip"):
            media_type = "application/zip"
        else:
            media_type = "audio/wav"
            
        return FileResponse(
            file_path,
            media_type=media_type,
            filename=file_name
        )

def delete_temp_file(file_path: str, delay: int = 3600):
    """Delete a temporary file after a delay"""
    try:
        time.sleep(delay)
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        logger.error(f"Error deleting temporary file {file_path}: {e}") 
