"""
WaveTransfer UI Layout for AudioLab.
A flexible end-to-end multi-instrument timbre transfer with diffusion.
"""

import logging
import os
import shutil
import time
import tempfile
from fastapi.responses import JSONResponse
import yaml
import json
import traceback
from typing import Optional, List
import threading
import torchaudio

import gradio as gr
from handlers.args import ArgHandler
from fastapi import Body, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from handlers.args import ArgHandler
from handlers.config import output_path
from modules.wavetransfer.main import train_model
from modules.wavetransfer.main_schedule_network import train_schedule_network, schedule_noise, infer_schedule_network
from modules.wavetransfer.params import get_default_params

# Global variables for inter-tab communication
SEND_TO_PROCESS_BUTTON = None
OUTPUT_AUDIO = None
logger = logging.getLogger("ADLB.WaveTransfer")

# Track active training processes for cancellation
class CancellationToken:
    def __init__(self):
        self.cancelled = False
        
    def cancel(self):
        self.cancelled = True

# Active training token
active_training_token = None

# Threaded training wrapper
class ThreadedTrainer:
    def __init__(self):
        self.result = None
        self.success = False
        self.error = None
        self.thread = None
        self.is_running = False
        self.progress = None
        
    def train_schedule_network_thread(self, config_path, project_dir):
        try:
            # Ensure we're using a single process
            # Modify environment to prevent subprocess spawning
            os.environ["USE_SINGLE_PROCESS"] = "1"
            os.environ["OMP_NUM_THREADS"] = "1"
            
            # Run the training function directly, no subprocess
            from modules.wavetransfer.main_schedule_network import train_schedule_network as train_func
            
            # Run training function with single process, passing the progress object
            self.success, self.result = train_func(config_path, project_dir, progress=self.progress)
        except Exception as e:
            self.success = False
            self.error = str(e)
            self.result = traceback.format_exc()
        finally:
            self.is_running = False
    
    def start_training(self, config_path, project_dir, progress=None):
        """Start the training in a thread without spawning processes"""
        if self.is_running:
            return False, "Training already in progress"
        
        self.is_running = True
        self.progress = progress
        self.thread = threading.Thread(
            target=self.train_schedule_network_thread,
            args=(config_path, project_dir)
        )
        self.thread.daemon = True
        self.thread.start()
        return True, "Training started in background thread"
    
    def get_result(self):
        """Get the current training result"""
        if self.is_running:
            return None, "Training still in progress"
        
        if self.success:
            return self.success, self.result
        else:
            return self.success, self.error or self.result

# Create a single instance of ThreadedTrainer to be used by both GUI and API
schedule_trainer = ThreadedTrainer()

# Available base models or configurations
DEFAULT_CONF_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "modules", "wavetransfer", "bddm", "conf.yml")
arg_handler = ArgHandler()


def resample_audio_file(source_path, target_path, target_sr=16000):
    """Resample an audio file to the target sample rate"""
    try:
        waveform, sr = torchaudio.load(source_path)
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
        torchaudio.save(target_path, waveform, target_sr)
        return True
    except Exception as e:
        logger.error(f"Error resampling audio file {source_path}: {str(e)}")
        return False


def preprocess_audio_files(files, output_dir, target_sr=16000, progress=None):
    """Preprocess audio files by resampling them to the target sample rate"""
    os.makedirs(output_dir, exist_ok=True)
    processed_files = []
    file_counts = {"wav": 0, "mp3": 0, "other": 0}
    
    # Count total valid files for progress tracking
    total_files = sum(1 for f in files if os.path.isfile(f) and (f.lower().endswith('.wav') or f.lower().endswith('.mp3')))
    
    for i, file_path in enumerate(files):
        if os.path.isfile(file_path):
            filename = os.path.basename(file_path)
            if filename.lower().endswith('.wav'):
                file_counts["wav"] += 1
                output_filename = filename
                output_path = os.path.join(output_dir, output_filename)
                success = resample_audio_file(file_path, output_path, target_sr)
                if success:
                    processed_files.append(output_path)
            elif filename.lower().endswith('.mp3'):
                file_counts["mp3"] += 1
                # Convert MP3 to WAV during preprocessing
                output_filename = filename.rsplit('.', 1)[0] + '.wav'
                output_path = os.path.join(output_dir, output_filename)
                success = resample_audio_file(file_path, output_path, target_sr)
                if success:
                    processed_files.append(output_path)
            else:
                file_counts["other"] += 1
                
            # Update progress if provided
            if progress and total_files > 0:
                progress_value = min(0.1 + (i / total_files) * 0.4, 0.5)  # Scale from 0.1 to 0.5
                progress(progress_value, f"Preprocessing file {i+1}/{total_files}: {filename}")
    
    return processed_files, file_counts


def list_wavetransfer_projects():
    """List all available WaveTransfer projects in the output directory."""
    projects = []
    output_dir = os.path.join(output_path, "wavetransfer")
    if os.path.exists(output_dir):
        projects = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    return projects


def send_to_process(file_to_send, existing_inputs):
    """Send generated audio to process tab."""
    if not file_to_send:
        return existing_inputs
    
    # Handle case where inputs is a list
    if isinstance(existing_inputs, list):
        updated_inputs = existing_inputs.copy()
        if file_to_send not in updated_inputs:
            updated_inputs.append(file_to_send)
    else:
        # Handle case where inputs is a FileList
        updated_inputs = [file_to_send]
    
    return updated_inputs


def create_project_config(project_dir, config_template=DEFAULT_CONF_PATH, **kwargs):
    """Create a configuration file for a project based on template or defaults."""
    # Create project directory
    os.makedirs(project_dir, exist_ok=True)
    
    # Start with default parameters
    config = get_default_params()
    
    # If template exists, load and override with it
    if config_template and os.path.exists(config_template):
        try:
            if config_template.endswith('.yml') or config_template.endswith('.yaml'):
                with open(config_template, 'r') as f:
                    template_config = yaml.safe_load(f)
            elif config_template.endswith('.json'):
                with open(config_template, 'r') as f:
                    template_config = json.load(f)
            else:
                # Assume YAML for other formats
                with open(config_template, 'r') as f:
                    template_config = yaml.safe_load(f)
                    
            # Override defaults with template
            config.override(template_config)
        except Exception as e:
            logger.error(f"Error loading config template {config_template}: {str(e)}")
            # Continue with defaults
    
    # Override with provided kwargs
    for key, value in kwargs.items():
        if key in config:
            config[key] = value
    
    # Write configuration file (both YAML and JSON for compatibility)
    config_path = os.path.join(project_dir, 'conf.yml')
    try:
        # Convert numpy arrays to lists for serialization
        config_dict = {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in config.items()}
        
        # Save as YAML
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f)
            
        # Also save as JSON for easier parsing      
        json_config_path = os.path.join(project_dir, 'conf.json')
        with open(json_config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
            
    except Exception as e:
        logger.error(f"Error saving config: {str(e)}")
        raise
    
    return config_path


def render(arg_handler: ArgHandler):
    global SEND_TO_PROCESS_BUTTON, OUTPUT_AUDIO
    
    with gr.Blocks() as app:
        gr.Markdown("# 🎵 WaveTransfer - Timbre Transfer with Diffusion")
        gr.Markdown(
            "WaveTransfer enables flexible end-to-end multi-instrument timbre transfer using diffusion models. "
            "You can train models that learn to transfer the timbre characteristics of one instrument to another."
        )
        
        # Create tabs for inference and training
        with gr.Tabs():
            # Inference Tab
            with gr.TabItem("Inference", id="wavetransfer_inference"):
                gr.Markdown("## 🎹 Instrument Timbre Transfer")
                gr.Markdown(
                    "Transfer the timbre of one instrument to another using trained diffusion models. "
                    "This lets you transform sounds while preserving the original musical content."
                )
                
                with gr.Row():
                    with gr.Column():
                        # Left column - Input files and options
                        gr.Markdown("### 📥 Input")
                        
                        project_selector = gr.Dropdown(
                            label="Project",
                            choices=list_wavetransfer_projects(),
                            interactive=True,
                            elem_id="wavetransfer_project_selector",
                            elem_classes="hintitem"
                        )
                        refresh_projects_btn = gr.Button("🔄 Refresh Projects")
                        
                        source_audio = gr.Audio(
                            label="Source Audio",
                            type="filepath",
                            elem_id="wavetransfer_source_audio",
                            elem_classes="hintitem"
                        )
                        
                    with gr.Column():
                        # Middle column - Processing options
                        gr.Markdown("### ⚙️ Options")
                        
                        chunked = gr.Checkbox(
                            label="Use Chunked Decoding",
                            value=True,
                            elem_id="wavetransfer_chunked",
                            elem_classes="hintitem"
                        )
                        
                        noise_schedule = gr.Radio(
                            label="Noise Schedule",
                            choices=["linear", "cosine"],
                            value="cosine",
                            elem_id="wavetransfer_noise_schedule",
                            elem_classes="hintitem"
                        )
                        
                        noise_steps = gr.Slider(
                            minimum=10,
                            maximum=1000,
                            step=10,
                            value=50,
                            label="Number of Noise Steps",
                            elem_id="wavetransfer_noise_steps",
                            elem_classes="hintitem"
                        )
                        
                    with gr.Column():
                        # Right column - Actions and output
                        gr.Markdown("### 🎨 Generate")
                        
                        with gr.Group():
                            with gr.Row():
                                generate_btn = gr.Button("🔄 Generate", variant="primary")
                                SEND_TO_PROCESS_BUTTON = gr.Button("📤 Send to Process")
                        
                        processing_status = gr.Markdown(visible=False)
                        
                        OUTPUT_AUDIO = gr.Audio(
                            label="Generated Audio",
                            elem_id="wavetransfer_output_audio",
                            elem_classes="hintitem"
                        )

                # Define inference functions
                def refresh_projects():
                    return gr.Dropdown.update(choices=list_wavetransfer_projects())
                
                def generate_audio(project, source_audio_path, chunked, noise_schedule, noise_steps, progress=gr.Progress()):
                    """Generate audio by inferring with the selected model"""
                    progress(0.1, "Starting generation process...")
                    
                    if not project:
                        return None, "⚠️ Please select a project first"
                    
                    if not source_audio_path or not os.path.exists(source_audio_path):
                        return None, "⚠️ Please provide a valid source audio file"
                    
                    # Create project directory path
                    project_dir = os.path.join(output_path, "wavetransfer", project)
                    if not os.path.exists(project_dir):
                        return None, f"⚠️ Project directory not found: {project_dir}"
                    
                    # Find config file
                    config_path = os.path.join(project_dir, "conf.yml")
                    if not os.path.exists(config_path):
                        return None, f"⚠️ Configuration file not found in project: {config_path}"
                    
                    # Update config with source audio path
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    # Update config parameters
                    config['inference_wav'] = source_audio_path
                    config['chunked'] = chunked
                    config['noise_schedule'] = noise_schedule
                    config['N'] = noise_steps
                    
                    # Write updated config
                    temp_config = os.path.join(tempfile.gettempdir(), f"wavetransfer_infer_{int(time.time())}.yml")
                    with open(temp_config, 'w') as f:
                        yaml.dump(config, f)
                    
                    progress(0.3, "Running inference with WaveTransfer model...")
                    
                    # Run inference
                    success, result = infer_schedule_network(temp_config, project_dir)
                    
                    # Clean up
                    if os.path.exists(temp_config):
                        os.remove(temp_config)
                    
                    if not success:
                        return None, f"⚠️ Generation failed: {result}"
                    
                    progress(0.9, "Generation complete!")
                    
                    # Find the first output file (should be a wav file)
                    if isinstance(result, list) and len(result) > 0:
                        return result[0], "✅ Generation complete! You can now play the audio or send it to processing."
                    else:
                        return None, "⚠️ No output files were generated"
                
                # Connect inference functions
                refresh_projects_btn.click(
                    fn=refresh_projects,
                    outputs=[project_selector]
                )
                
                generate_btn.click(
                    fn=generate_audio,
                    inputs=[
                        project_selector,
                        source_audio,
                        chunked,
                        noise_schedule,
                        noise_steps
                    ],
                    outputs=[
                        OUTPUT_AUDIO,
                        processing_status
                    ]
                )
            
            # Training Tab
            with gr.TabItem("Train", id="wavetransfer_train"):
                gr.Markdown("## 🔄 Train Timbre Transfer Model")
                gr.Markdown(
                    "Train a new WaveTransfer model to learn the timbre characteristics of your target instrument. "
                    "This involves two steps: (1) Training the main model and (2) Training the schedule network."
                )
                
                with gr.Row():
                    with gr.Column():
                        # Left column - Project settings
                        gr.Markdown("### 📂 Project")
                        
                        project_name = gr.Textbox(
                            label="Project Name",
                            placeholder="Enter a name for your project",
                            elem_id="wavetransfer_project_name",
                            elem_classes="hintitem"
                        )
                        
                        project_list = gr.Dropdown(
                            label="Existing Projects",
                            choices=list_wavetransfer_projects(),
                            interactive=True,
                            elem_id="wavetransfer_project_list",
                            elem_classes="hintitem"
                        )
                        refresh_train_projects_btn = gr.Button("🔄 Refresh Projects")
                        
                        training_mode = gr.Radio(
                            label="Training Mode",
                            choices=["New Project", "Continue Training", "Schedule Network"],
                            value="New Project",
                            elem_id="wavetransfer_training_mode",
                            elem_classes="hintitem"
                        )
                        
                    with gr.Column():
                        # Middle column - Training data and parameters
                        gr.Markdown("### 🎧 Training Data")
                        
                        with gr.Group(visible=True) as new_project_group:
                            data_dirs = gr.File(
                                label="Training Audio Files/Directory",
                                file_count="multiple",
                                elem_id="wavetransfer_data_dirs",
                                elem_classes="hintitem"
                            )
                            
                            gr.Markdown(
                                """
                                > **Note**: Audio files will be automatically resampled to 16kHz for optimal training. 
                                > This preprocessing step ensures best performance and quality.
                                """
                            )
                            
                            max_epochs = gr.Number(
                                label="Maximum Training Epochs",
                                value=10000,
                                elem_id="wavetransfer_max_epochs",
                                elem_classes="hintitem"
                            )
                            
                            batch_size = gr.Slider(
                                minimum=1,
                                maximum=64,
                                step=1,
                                value=32,
                                label="Batch Size",
                                elem_id="wavetransfer_batch_size",
                                elem_classes="hintitem"
                            )
                            
                            checkpoint_interval = gr.Number(
                                label="Checkpoint Interval",
                                value=5000,
                                elem_id="wavetransfer_checkpoint_interval",
                                elem_classes="hintitem"
                            )
                            
                            fp16 = gr.Checkbox(
                                label="Use FP16 (Half Precision)",
                                value=True,
                                elem_id="wavetransfer_fp16",
                                elem_classes="hintitem"
                            )
                            
                            # Note: Preprocessing (resampling to 16kHz) is always performed since WaveTransfer
                            # is optimized for 16kHz audio. This improves training speed and quality.
                        
                        with gr.Group(visible=False) as schedule_group:
                            noise_sched_steps = gr.Slider(
                                minimum=10,
                                maximum=1000,
                                step=10,
                                value=50,
                                label="Noise Schedule Steps",
                                elem_id="wavetransfer_noise_sched_steps",
                                elem_classes="hintitem"
                            )
                            
                            noise_sched_type = gr.Radio(
                                label="Noise Schedule Type",
                                choices=["linear", "cosine"],
                                value="cosine",
                                elem_id="wavetransfer_noise_sched_type",
                                elem_classes="hintitem"
                            )
                        
                        with gr.Group(visible=False) as schedule_status_group:
                            check_status_btn = gr.Button("🔄 Check Schedule Training Status")
                            schedule_status = gr.Markdown("No schedule network training in progress.")
                    
                    with gr.Column():
                        # Right column - Actions and logs
                        gr.Markdown("### 🚀 Training")
                        
                        with gr.Row():
                            train_btn = gr.Button("🚀 Start Training", variant="primary")
                            cancel_train_btn = gr.Button("🛑 Cancel Training", variant="stop")
                        
                        training_status = gr.Markdown(
                            value="Ready to train. Set up your project and click 'Start Training'.",
                            elem_id="wavetransfer_training_status",
                            elem_classes="hintitem"
                        )
                        
                        training_log = gr.Textbox(
                            label="Training Log",
                            interactive=False,
                            lines=10,
                            elem_id="wavetransfer_training_log",
                            elem_classes="hintitem"
                        )
                
                # Define training functions
                def refresh_train_projects():
                    return gr.Dropdown.update(choices=list_wavetransfer_projects())
                
                def toggle_training_mode(mode):
                    """Show/hide appropriate UI elements based on training mode"""
                    if mode == "New Project" or mode == "Continue Training":
                        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
                    else:  # Schedule Network
                        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)
                
                def update_project_name(selected_project):
                    """Update project name when an existing project is selected"""
                    if selected_project:
                        return selected_project
                    return ""
                
                def start_training(
                    project_name,
                    existing_project,
                    training_mode,
                    uploaded_files,
                    max_epochs,
                    batch_size,
                    checkpoint_interval,
                    fp16,
                    noise_sched_steps,
                    noise_sched_type,
                    progress=gr.Progress()
                ):
                    """Start training based on selected mode"""
                    # Create a cancellation token
                    global active_training_token
                    active_training_token = CancellationToken()
                    
                    # Use existing project if in continue mode, otherwise create new
                    if training_mode in ["Continue Training", "Schedule Network"]:
                        if not existing_project:
                            return "⚠️ Please select an existing project for continuing training.", ""
                        project = existing_project
                    else:  # New Project
                        if not project_name or project_name.strip() == "":
                            return "⚠️ Please provide a project name.", ""
                        project = project_name.strip()
                    
                    # Create project directory
                    project_dir = os.path.join(output_path, "wavetransfer", project)
                    os.makedirs(project_dir, exist_ok=True)
                    
                    log_content = f"Starting project: {project}\n"
                    log_content += f"Training mode: {training_mode}\n"
                    
                    # Handle different training modes
                    if training_mode == "Schedule Network":
                        progress(0.1, "Preparing to train schedule network...")
                        log_content += f"Noise schedule steps: {noise_sched_steps}\n"
                        log_content += f"Noise schedule type: {noise_sched_type}\n"
                        
                        # Find data directory to use
                        data_directory = os.path.join(project_dir, "data")
                        preproc_directory = os.path.join(project_dir, "data_preproc")
                        
                        # Check if we have preprocessed directory first
                        if os.path.exists(preproc_directory) and any(os.listdir(preproc_directory)):
                            data_dirs_list = [preproc_directory]
                            log_content += f"Using existing preprocessed data directory: {preproc_directory}\n"
                        elif os.path.exists(data_directory):
                            data_dirs_list = [data_directory]
                            log_content += f"Using existing data directory: {data_directory}\n"
                        else:
                            return "⚠️ No data directory found for existing project.", ""
                        
                        # Create/update config
                        config_path = create_project_config(
                            project_dir,
                            N=int(noise_sched_steps),
                            noise_schedule=noise_sched_type,
                            # Point to model checkpoint from previous training
                            load=os.path.join(project_dir, "model", "weights.pt"),
                            # Set data directories directly for schedule network training
                            data_dir=data_dirs_list,
                            # Use empty training/validation files to force direct file processing mode
                            training_file=[],
                            validation_file=[]
                        )
                        
                        progress(0.2, "Training schedule network...")
                        log_content += f"Using config: {config_path}\n"
                        
                        # Start the schedule network training in a background thread
                        start_success, start_msg = schedule_trainer.start_training(config_path, project_dir, progress=progress)
                        
                        if not start_success:
                            log_content += f"❌ Failed to start schedule network training: {start_msg}\n"
                            return f"❌ Failed to start schedule network training: {start_msg}", log_content
                        
                        # Poll for training progress
                        log_content += f"⏳ Schedule network training started in background.\n"
                        log_content += f"You can continue using other features while training runs.\n"
                        
                        # Check if training is done (for immediate completion)
                        training_result = schedule_trainer.get_result()
                        if training_result[0] is not None:  # Training completed
                            success, result = training_result
                            if success:
                                log_content += f"✅ Schedule network training completed.\n"
                                log_content += f"Results saved to: {result}\n"
                                return "✅ Schedule network training completed successfully!", log_content
                            else:
                                log_content += f"❌ Schedule network training failed: {result}\n"
                                return f"❌ Schedule network training failed: {result}", log_content
                        
                        # If not done, return the in-progress status
                        return "⏳ Schedule network training is running in the background. Check back later for results.", log_content
                    
                    else:  # Main model training (new or continue)
                        # Validate files for new project
                        if training_mode == "New Project" and (not uploaded_files or len(uploaded_files) == 0):
                            return "⚠️ Please upload training audio files.", ""
                        
                        # Process uploaded files
                        if training_mode == "New Project":
                            progress(0.1, "Processing uploaded files...")
                            data_directory = os.path.join(project_dir, "data")
                            os.makedirs(data_directory, exist_ok=True)
                            
                            # Optionally preprocess audio files
                            if True:  # Preprocessing is always performed
                                progress(0.1, "Preprocessing audio files (resampling to 16kHz)...")
                                log_content += f"Preprocessing audio files to 16kHz sample rate...\n"
                                
                                # Create preprocessing directory
                                preproc_directory = os.path.join(project_dir, "data_preproc")
                                processed_files, file_counts = preprocess_audio_files(
                                    uploaded_files, 
                                    preproc_directory, 
                                    target_sr=16000, 
                                    progress=progress
                                )
                                
                                # Check if preprocessing was successful
                                if not processed_files:
                                    return "⚠️ Audio preprocessing failed. Check logs for details.", ""
                                
                                log_content += f"Preprocessed {len(processed_files)} files to 16kHz sample rate.\n"
                                # Use preprocessed files for training
                                data_dirs_list = [preproc_directory]
                                
                                # Log file counts
                                log_content += f"\nPreprocessed {len(processed_files)} files:\n"
                                log_content += f"- WAV files: {file_counts['wav']}\n"
                                log_content += f"- MP3 files: {file_counts['mp3']}\n"
                                if file_counts["other"] > 0:
                                    log_content += f"- Other files: {file_counts['other']} (these were ignored)\n"
                                
                            else:
                                # Copy uploaded files to data directory without preprocessing
                                # Count file types for logging
                                file_counts = {"wav": 0, "mp3": 0, "other": 0}
                                
                                # Copy uploaded files to data directory
                                for file_path in uploaded_files:
                                    if os.path.isfile(file_path):
                                        filename = os.path.basename(file_path)
                                        dest_path = os.path.join(data_directory, filename)
                                        shutil.copy(file_path, dest_path)
                                        
                                        # Count file type
                                        if filename.lower().endswith('.wav'):
                                            file_counts["wav"] += 1
                                        elif filename.lower().endswith('.mp3'):
                                            file_counts["mp3"] += 1
                                        else:
                                            file_counts["other"] += 1
                                            
                                        log_content += f"Added training file: {filename}\n"
                                
                                # Log file counts
                                log_content += f"\nUploaded {len(uploaded_files)} files:\n"
                                log_content += f"- WAV files: {file_counts['wav']}\n"
                                log_content += f"- MP3 files: {file_counts['mp3']}\n"
                                if file_counts["other"] > 0:
                                    log_content += f"- Other files: {file_counts['other']} (these will be ignored)\n"
                                
                                # Use data directory as source
                                data_dirs_list = [data_directory]
                        else:
                            # For continue training, use existing data directory
                            data_directory = os.path.join(project_dir, "data")
                            preproc_directory = os.path.join(project_dir, "data_preproc")
                            
                            # Check if we have preprocessed directory first
                            if os.path.exists(preproc_directory) and any(os.listdir(preproc_directory)):
                                data_dirs_list = [preproc_directory]
                                log_content += f"Using existing preprocessed data directory: {preproc_directory}\n"
                            elif os.path.exists(data_directory):
                                data_dirs_list = [data_directory]
                                log_content += f"Using existing data directory: {data_directory}\n"
                            else:
                                return "⚠️ No data directory found for existing project.", ""
                        
                        # Set up model directory
                        model_directory = os.path.join(project_dir, "model")
                        os.makedirs(model_directory, exist_ok=True)
                        
                        progress(0.5, "Starting model training...")
                        log_content += f"Model directory: {model_directory}\n"
                        log_content += f"Max epochs: {max_epochs}\n"
                        log_content += f"Batch size: {batch_size}\n"
                        log_content += f"Checkpoint interval: {checkpoint_interval}\n"
                        log_content += f"Using FP16: {fp16}\n"
                        
                        # Train the model
                        success, result = train_model(
                            model_dir=model_directory,
                            data_dirs=data_dirs_list,
                            max_epochs=int(max_epochs) if max_epochs is not None else None,
                            checkpoint_interval=int(checkpoint_interval) if checkpoint_interval is not None else None,
                            fp16=fp16,
                            batch_size=int(batch_size) if batch_size is not None else None,
                            force_single_process=True,
                            progress=progress,
                            cancel_token=active_training_token,
                            config_data={"batch_size": int(batch_size)}
                        )
                        
                        if success:
                            log_content += f"✅ Model training completed.\n"
                            log_content += f"Results saved to: {result}\n"
                            
                            # Create default config for later use
                            config_path = create_project_config(
                                project_dir,
                                load=os.path.join(model_directory, "weights.pt"),
                                N=50,
                                noise_schedule="cosine"
                            )
                            log_content += f"Created default config: {config_path}\n"
                            
                            return "✅ Model training completed successfully! You may now train the schedule network.", log_content
                        else:
                            # Check if cancelled
                            if active_training_token and active_training_token.cancelled:
                                log_content += f"⏹️ Training cancelled by user.\n"
                                return f"⏹️ Training cancelled by user", log_content
                            
                            # Display detailed error information
                            log_content += f"❌ Model training failed with error:\n{result}\n"
                            
                            # Log the full error details
                            logger.error(f"Training failed for project {project}: {result}")
                            
                            # Save the error log to the project directory
                            try:
                                error_log_path = os.path.join(project_dir, "training_error.log")
                                with open(error_log_path, "w") as f:
                                    f.write(f"Training failed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                                    f.write(f"Error details:\n{result}\n")
                                log_content += f"Error details saved to: {error_log_path}\n"
                            except Exception as e:
                                log_content += f"Failed to save error log: {str(e)}\n"
                            
                            return f"❌ Model training failed. Check the training log for details.", log_content
                
                def cancel_training():
                    """Cancel the current training process."""
                    global active_training_token
                    if active_training_token:
                        active_training_token.cancel()
                        return "Cancellation requested. Training will stop after the current batch completes..."
                    return "No active training to cancel."
                
                def check_schedule_training_status():
                    """Check the status of currently running schedule network training"""
                    if not schedule_trainer.is_running:
                        # Check if we have a completed result
                        training_result = schedule_trainer.get_result()
                        if training_result[0] is not None:  # Training completed
                            success, result = training_result
                            if success:
                                return "✅ Schedule network training completed successfully.\nResults saved to: " + str(result)
                            else:
                                return "❌ Schedule network training failed:\n" + str(result)
                        return "No schedule network training in progress."
                    else:
                        return "⏳ Schedule network training is currently running..."
                
                # Connect training functions
                refresh_train_projects_btn.click(
                    fn=refresh_train_projects,
                    outputs=[project_list]
                )
                
                training_mode.change(
                    fn=toggle_training_mode,
                    inputs=[training_mode],
                    outputs=[new_project_group, schedule_group, schedule_status_group]
                )
                
                project_list.change(
                    fn=update_project_name,
                    inputs=[project_list],
                    outputs=[project_name]
                )
                
                train_btn.click(
                    fn=start_training,
                    inputs=[
                        project_name,
                        project_list,
                        training_mode,
                        data_dirs,
                        max_epochs,
                        batch_size,
                        checkpoint_interval,
                        fp16,
                        noise_sched_steps,
                        noise_sched_type
                    ],
                    outputs=[
                        training_status,
                        training_log
                    ]
                )
                
                cancel_train_btn.click(
                    fn=cancel_training,
                    outputs=[training_status]
                )
                
                check_status_btn.click(
                    fn=check_schedule_training_status,
                    outputs=[schedule_status]
                )
                
                # Add a third tab for information
                with gr.TabItem("Info", id="wavetransfer_info"):
                    gr.Markdown("## 📚 About WaveTransfer")
                    gr.Markdown(
                        """
                        # WaveTransfer: A Flexible End-to-end Multi-instrument Timbre Transfer with Diffusion
                        
                        WaveTransfer is a tool for transferring the timbre (sound characteristics) of one instrument to another
                        using diffusion models. It enables high-quality audio transformation while preserving the musical content.
                        
                        ## How It Works
                        
                        1. **Training the Main Model**: The first step is to train a diffusion model on your target instrument audio.
                           This model learns to generate audio with the specific timbre characteristics of your target instrument.
                           
                        2. **Training the Schedule Network**: After the main model is trained, you need to train a schedule network
                           which helps optimize the noise scheduling for the diffusion process.
                           
                        3. **Inference**: Finally, you can use the trained model to transform your source audio by applying the 
                           learned timbre characteristics while preserving the original musical content.
                        
                        ## Tips for Best Results
                        
                        - Use high-quality audio recordings of your target instrument for training
                        - Provide a diverse set of audio samples covering the full range of the instrument
                        - For best results, train with at least 30 minutes of audio data
                        - Use chunked decoding for faster generation with minimal quality loss
                        - Experiment with different noise schedules (linear vs. cosine) for different results
                        
                        ## Training Steps
                        
                        1. Create a new project and upload your training audio files
                        2. Train the main model (this may take several hours depending on your GPU)
                        3. Train the schedule network using the trained model
                        4. Generate new audio using your trained model with a source audio file
                        
                        ## Citation
                        
                        ```
                        @inproceedings{baoueb2024wavetransfer,
                          title={WaveTransfer: A Flexible End-to-end Multi-instrument Timbre Transfer with Diffusion},
                          author={Baoueb, Teysir and Bie, Xiaoyu and Janati, Hicham and Richard, Gaël},
                          booktitle={34th {IEEE} International Workshop on Machine Learning for Signal Processing, {MLSP} 2024, London, United Kingdom, September 22-25, 2024},
                          year={2024}
                        }
                        ```
                        """
                    )
    
    return app


def listen():
    """Set up event listeners for inter-tab communication."""
    # This function is called after all tabs are rendered
    if SEND_TO_PROCESS_BUTTON and OUTPUT_AUDIO:
        process_inputs = arg_handler.get_element("main", "process_inputs")
        if process_inputs:
            SEND_TO_PROCESS_BUTTON.click(
                fn=send_to_process,
                inputs=[OUTPUT_AUDIO, process_inputs],
                outputs=[process_inputs]
            )


def register_descriptions(arg_handler: ArgHandler):
    """Register tooltips and descriptions for UI elements."""
    descriptions = {
        # Inference tab
        "wavetransfer_project_selector": "Select a trained WaveTransfer project to use for generation.",
        "wavetransfer_source_audio": "Upload the source audio you want to transform with the selected model.",
        "wavetransfer_chunked": "Process audio in chunks to reduce memory usage (recommended for longer files).",
        "wavetransfer_noise_schedule": "The noise schedule to use for generation (cosine usually gives better results).",
        "wavetransfer_noise_steps": "Number of noise steps for generation. More steps = higher quality but slower.",
        "wavetransfer_output_audio": "The generated audio with transformed timbre characteristics.",
        
        # Training tab
        "wavetransfer_project_name": "Name for your WaveTransfer project. This will create a folder in the outputs directory.",
        "wavetransfer_project_list": "Select an existing project to continue training or to train the schedule network.",
        "wavetransfer_training_mode": "Choose whether to create a new project, continue training an existing model, or train the schedule network.",
        "wavetransfer_data_dirs": "Upload audio files of your target instrument. These will be used to train the model.",
        "wavetransfer_max_epochs": "Maximum number of training epochs. More epochs = better quality but longer training time.",
        "wavetransfer_batch_size": "Number of audio examples processed in each training iteration. Higher values use more memory but can speed up training.",
        "wavetransfer_checkpoint_interval": "How often to save model checkpoints during training.",
        "wavetransfer_fp16": "Use half-precision (16-bit) floating point for training. Speeds up training with minimal quality loss.",
        "wavetransfer_noise_sched_steps": "Number of noise steps for the schedule network. This affects generation quality and speed.",
        "wavetransfer_noise_sched_type": "Type of noise schedule to train. Cosine usually gives better results than linear.",
        "wavetransfer_training_status": "Current status of the training process.",
        "wavetransfer_training_log": "Detailed log information about the training process."
    }
    
    for elem_id, description in descriptions.items():
        arg_handler.register_description("wavetransfer", elem_id, description)


def register_api_endpoints(api):
    """Register API endpoints for the WaveTransfer layout."""
    
    # Define request/response models for the API
    class TrainWaveTransferRequest(BaseModel):
        project_name: str = Field(..., description="Name of the project to create or continue")
        continue_training: bool = Field(False, description="Whether to continue training an existing project")
        max_epochs: Optional[int] = Field(10000, description="Maximum number of training epochs")
        checkpoint_interval: Optional[int] = Field(5000, description="Interval between model checkpoints")
        fp16: bool = Field(True, description="Whether to use 16-bit floating point operations for training")
        batch_size: Optional[int] = Field(32, description="Number of audio examples to process in each training batch")
    
    class TrainWaveTransferScheduleRequest(BaseModel):
        project_name: str = Field(..., description="Name of the existing project")
        max_epochs: Optional[int] = Field(5000, description="Maximum number of training epochs")
        checkpoint_interval: Optional[int] = Field(1000, description="Interval between model checkpoints")
        fp16: bool = Field(True, description="Whether to use 16-bit floating point operations for training")
        noise_steps: int = Field(50, description="Number of noise steps for the schedule")
        noise_schedule_type: str = Field("cosine", description="Type of noise schedule (linear or cosine)")
    
    class ScheduleNetworkRequest(BaseModel):
        project_name: str = Field(..., description="Name of the existing project")
        noise_steps: int = Field(50, description="Number of noise steps for the schedule")
        noise_schedule_type: str = Field("cosine", description="Type of noise schedule (linear or cosine)")
    
    class InferenceRequest(BaseModel):
        project_name: str = Field(..., description="Name of the trained project to use")
        chunked: bool = Field(True, description="Whether to use chunked decoding")
        noise_steps: int = Field(50, description="Number of noise steps for generation")
        noise_schedule_type: str = Field("cosine", description="Type of noise schedule (linear or cosine)")
    
    # API endpoints
    @api.post("/api/v1/wavetransfer/train", tags=["Audio Generation"])
    async def train_wavetransfer_model(
        request: TrainWaveTransferRequest = Body(...),
        files: List[UploadFile] = File(None)
    ):
        """Train a WaveTransfer model with the specified parameters."""
        try:
            # Create or validate project
            project_name = request.project_name
            if not project_name:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Project name is required"}
                )
                
            # Set up project directory
            project_dir = os.path.join("outputs", "wavetransfer", project_name)
            if not os.path.exists(project_dir):
                os.makedirs(project_dir, exist_ok=True)
            
            # Check if main files exist
            data_dir = os.path.join(project_dir, "data")
            model_dir = os.path.join(project_dir, "model")
            os.makedirs(data_dir, exist_ok=True)
            os.makedirs(model_dir, exist_ok=True)
            
            # Handle uploaded files if present
            saved_files = []
            if files:
                for file in files:
                    content = await file.read()
                    filename = file.filename
                    file_path = os.path.join(data_dir, filename)
                    with open(file_path, "wb") as f:
                        f.write(content)
                    saved_files.append(file_path)
            
            # Verify data directory has files if not continuing training
            if not request.continue_training and not files:
                wav_files = [f for f in os.listdir(data_dir) if f.endswith('.wav') or f.endswith('.mp3')]
                if not wav_files:
                    return JSONResponse(
                        status_code=400,
                        content={"error": "No audio files provided for training. Please upload audio files."}
                    )
            
            # Handle preprocessing
            if saved_files:
                # Create preprocessing directory
                preproc_dir = os.path.join(project_dir, "data_preproc")
                processed_files, _ = preprocess_audio_files(saved_files, preproc_dir, target_sr=16000)
                
                if processed_files:
                    # Use preprocessed files for training
                    train_data_dir = preproc_dir
                else:
                    # Fallback to original files if preprocessing failed
                    train_data_dir = data_dir
            else:
                # Use original data directory
                train_data_dir = data_dir
            
            # Train the model
            success, result = train_model(
                model_dir=model_dir,
                data_dirs=[train_data_dir],
                max_epochs=request.max_epochs,
                checkpoint_interval=request.checkpoint_interval,
                fp16=request.fp16,
                batch_size=request.batch_size,
                force_single_process=True,
                progress=None  # No progress bar for API calls
            )
            
            if success:
                # Create default config
                config_path = create_project_config(
                    project_dir,
                    load=os.path.join(model_dir, "weights.pt"),
                    N=50,
                    noise_schedule="cosine"
                )
                
                return {
                    "success": True,
                    "message": "Model training completed successfully",
                    "model_dir": result,
                    "config_path": config_path
                }
            else:
                return JSONResponse(
                    status_code=500,
                    content={
                        "success": False,
                        "error": "Training failed",
                        "details": result
                    }
                )
                
        except Exception as e:
            traceback_str = traceback.format_exc()
            logger.error(f"API error in train_wavetransfer_model: {str(e)}\n{traceback_str}")
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": str(e),
                    "traceback": traceback_str
                }
            )
    
    @api.post("/api/v1/wavetransfer/schedule", tags=["Audio Generation"])
    async def train_wavetransfer_schedule(
        request: TrainWaveTransferScheduleRequest = Body(...),
    ):
        """Train a WaveTransfer noise schedule network."""
        try:
            # Create or validate project
            project_name = request.project_name
            if not project_name:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Project name is required"}
                )
            
            # Set up project directory  
            project_dir = os.path.join("outputs", "wavetransfer", project_name)
            os.makedirs(project_dir, exist_ok=True)
            
            # Set up directories
            model_dir = os.path.join(project_dir, "model")
            os.makedirs(model_dir, exist_ok=True)
            
            # Find data directory to use
            data_directory = os.path.join(project_dir, "data")
            preproc_directory = os.path.join(project_dir, "data_preproc")
            
            # Check if we have preprocessed directory first
            if os.path.exists(preproc_directory) and any(os.listdir(preproc_directory)):
                data_dirs_list = [preproc_directory]
            elif os.path.exists(data_directory) and any(os.listdir(data_directory)):
                data_dirs_list = [data_directory]
            else:
                return JSONResponse(
                    status_code=400,
                    content={"error": "No data directory found for existing project"}
                )
            
            # Create configuration for the schedule network
            config_path = create_project_config(
                project_dir,
                N=request.noise_steps,
                noise_schedule=request.noise_schedule_type,
                # Point to model checkpoint from previous training
                load=os.path.join(model_dir, "weights.pt"),
                # Set data directories directly for schedule network training
                data_dir=data_dirs_list,
                # Use empty training/validation files to force direct file processing mode
                training_file=[],
                validation_file=[]
            )
            
            # Train the schedule network
            start_success, start_msg = schedule_trainer.start_training(config_path, project_dir, progress=None)
            
            if not start_success:
                return JSONResponse(
                    status_code=500,
                    content={
                        "success": False, 
                        "error": "Failed to start training",
                        "details": start_msg
                    }
                )
            
            # For API we'll wait for the training to complete
            # We could make this async in a future update
            while True:
                training_result = schedule_trainer.get_result()
                if training_result[0] is not None:  # Training completed
                    success, result = training_result
                    break
                time.sleep(2)  # Poll every 2 seconds
            
            if success:
                return {
                    "success": True,
                    "message": "Schedule network training completed successfully",
                    "model_dir": result
                }
            else:
                return JSONResponse(
                    status_code=500,
                    content={
                        "success": False,
                        "error": "Training failed",
                        "details": result
                    }
                )
                
        except Exception as e:
            traceback_str = traceback.format_exc()
            logger.error(f"API error in train_wavetransfer_schedule: {str(e)}\n{traceback_str}")
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": str(e),
                    "traceback": traceback_str
                }
            )
    
    @api.post("/api/v1/wavetransfer/generate", tags=["Audio Generation"])
    async def generate_audio(
        request: InferenceRequest = Body(...),
        source_audio: UploadFile = File(...)
    ):
        """Generate audio using a trained WaveTransfer model."""
        try:
            # Check if project exists
            project_dir = os.path.join(output_path, "wavetransfer", request.project_name)
            if not os.path.exists(project_dir):
                raise HTTPException(status_code=404, detail=f"Project not found: {request.project_name}")
            
            # Save uploaded source audio
            temp_dir = tempfile.mkdtemp()
            source_path = os.path.join(temp_dir, source_audio.filename)
            with open(source_path, "wb") as f:
                f.write(await source_audio.read())
            
            # Create/update config
            config_path = create_project_config(
                project_dir,
                inference_wav=source_path,
                N=request.noise_steps,
                noise_schedule=request.noise_schedule_type,
                chunked=request.chunked
            )
            
            # Run inference
            success, result = infer_schedule_network(config_path, project_dir)
            
            # Clean up
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            if not success:
                raise HTTPException(status_code=500, detail=f"Generation failed: {result}")
            
            # Return the first output file
            if isinstance(result, list) and len(result) > 0:
                output_file = result[0]
                
                # Return file path for now (in a real API, would return file content)
                return {
                    "success": True,
                    "message": "Generation completed successfully",
                    "output_file": output_file
                }
            else:
                raise HTTPException(status_code=500, detail="No output files were generated")
            
        except Exception as e:
            logger.error(f"Error in generate_audio: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @api.get("/api/v1/wavetransfer/projects", tags=["Audio Generation"])
    async def list_projects():
        """List all available WaveTransfer projects."""
        try:
            projects = list_wavetransfer_projects()
            return {"projects": projects}
        except Exception as e:
            logger.error(f"Error in list_projects: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e)) 