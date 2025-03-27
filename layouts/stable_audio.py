import os
import logging
import gradio as gr
import torch
from typing import Dict, Any, List
import random
import zipfile

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

def get_random_prompt():
    """Return a random example prompt to help users get started."""
    return random.choice(EXAMPLE_PROMPTS)

def get_random_negative_prompt():
    """Return a random example negative prompt."""
    return random.choice(EXAMPLE_NEGATIVE_PROMPTS)

def update_output_info(results=None, error=None):
    """Update the output info display with multiple output support."""
    if error:
        return f"Error: {error}", None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(choices=[], visible=False), {}, gr.update(visible=False)
    
    if results and len(results) > 0:
        global OUTPUT_FILES
        OUTPUT_FILES = [result["file_path"] for result in results]
        
        # Create display names for each variation
        display_names = [f"Variation {i+1}" for i in range(len(results))]
        
        # Create mapping between display names and file paths
        output_map = {display_names[i]: results[i]["file_path"] for i in range(len(results))}
        
        # Show output selector only if we have multiple variations
        selector_visible = len(results) > 1
        download_all_visible = len(results) > 1
        
        # Return the first audio file as default
        file_path = results[0]["file_path"]
        return "Generation complete!", file_path, gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(choices=display_names, value=display_names[0], visible=selector_visible), output_map, gr.update(visible=download_all_visible)
    
    return "No output generated", None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(choices=[], visible=False), {}, gr.update(visible=False)

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
                    elem_classes="hintitem"
                )
                
                with gr.Row():
                    OUTPUT_AUDIO = gr.Audio(
                        label="Generated Audio",
                        type="filepath",
                        interactive=False,
                        elem_id="stable_audio_output",
                        key="stable_audio_output",
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
                
                # Progress indicator
                progress = gr.Progress()
                
                # Hidden state to store output mappings
                output_map = gr.State({})
    
    # Define functions for button actions
    def randomize_seed():
        return random.randint(0, 2147483647)
    
    def clear_outputs():
        return "", get_random_negative_prompt(), 5.0, 1, 100, 7.0, -1, False, None, 0.7, "Ready to generate", None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(choices=[], visible=False), {}, gr.update(visible=False)
    
    def toggle_init_audio(use_init):
        return gr.update(visible=use_init)
    
    def use_output_as_reference(output_audio):
        """Set the currently generated output as the reference input."""
        if output_audio and os.path.exists(output_audio):
            return (
                gr.update(value=True), 
                gr.update(value={"name": output_audio})
            )
        return gr.update(), gr.update()
    
    def generate_audio(progress=gr.Progress(), prompt="", negative_prompt="", duration=5.0, 
                      num_waveforms=1, inference_steps=100, guidance_scale=7.0, seed=-1,
                      use_init_audio=False, init_audio_path=None, init_noise_level=0.7):
        """Generate audio based on the text prompt and parameters."""
        try:
            progress(0, desc="Initializing...")
            
            # Validate inputs
            if not prompt.strip():
                return "Error: Please provide a text prompt", None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(choices=[], visible=False), {}, gr.update(visible=False)
            
            # Process the audio generation
            progress(0.1, desc="Loading model...")
            
            # Determine if we should use init audio
            init_audio_file = None
            if use_init_audio and init_audio_path:
                # Get actual file path from the File component
                init_audio_file = init_audio_path.name if init_audio_path else None
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
        outputs=[status, OUTPUT_AUDIO, download_btn, SEND_TO_PROCESS_BUTTON, use_as_reference_btn, output_selector, output_map, download_all_btn]
    )
    
    clear_btn.click(
        fn=clear_outputs,
        inputs=[],
        outputs=[prompt, negative_prompt, duration, num_waveforms, 
                inference_steps, guidance_scale, seed,
                use_init_audio, init_audio_path, init_noise_level,
                status, OUTPUT_AUDIO, download_btn, SEND_TO_PROCESS_BUTTON, use_as_reference_btn, output_selector, output_map, download_all_btn]
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
        outputs=[init_audio_options]
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
        inputs=[OUTPUT_AUDIO],
        outputs=[use_init_audio, init_audio_path]
    )
    
    # Add change event for the output selector
    output_selector.change(
        fn=lambda selection, output_map: gr.update(value=output_map.get(selection, None)),
        inputs=[output_selector, output_map],
        outputs=[OUTPUT_AUDIO]
    )
    
    # Add download all button click event
    download_all_btn.click(
        fn=download_all_output_files,
        inputs=[],
        outputs=[gr.File(label="Download")]
    )
    
    # Add Send to Process button click event directly here
    process_inputs = arg_handler.get_element("main", "process_inputs")
    if process_inputs:
        SEND_TO_PROCESS_BUTTON.click(
            fn=send_to_process, 
            inputs=[OUTPUT_AUDIO, process_inputs], 
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

def send_to_process(output_audio, process_inputs):
    """Send the generated audio to the Process tab."""
    if not output_audio or not os.path.exists(output_audio):
        return gr.update()
    
    if output_audio in process_inputs:
        return gr.update()
        
    process_inputs.append(output_audio)
    return gr.update(value=process_inputs) 