"""
Orpheus TTS UI Layout for AudioLab.
"""

import os
import logging
import torch
import gradio as gr
from pathlib import Path
import tempfile
import time
from typing import Optional, List
import traceback
import shutil

from handlers.args import ArgHandler
from handlers.config import output_path, model_path
from modules.orpheus.model import OrpheusModel
from modules.orpheus.finetune import OrpheusFinetune

from fastapi import UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

# Global variables for inter-tab communication
SEND_TO_PROCESS_BUTTON = None
OUTPUT_AUDIO = None
FINETUNE_OUTPUT = None
orpheus_model = None
finetune_handler = None
logger = logging.getLogger("ADLB.Orpheus")

# Available voices and emotions
AVAILABLE_VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
AVAILABLE_EMOTIONS = ["None", "happy", "sad", "angry", "scared", "disgusted", "surprised"]

def load_model(model_name="canopylabs/orpheus-tts-0.1-finetune-prod"):
    """
    Load the Orpheus TTS model.
    
    Args:
        model_name: The name or path of the model to use
    
    Returns:
        The loaded model
    """
    global orpheus_model
    
    if orpheus_model is None:
        orpheus_model = OrpheusModel(model_name)
    
    return orpheus_model
    
def generate_speech(text, voice, emotion, temperature, top_p, repetition_penalty, progress=gr.Progress(track_tqdm=True)):
    """
    Generate speech using the Orpheus TTS model.
    
    Args:
        text: The text to synthesize
        voice: The voice to use
        emotion: The emotion to apply
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        repetition_penalty: Repetition penalty parameter
        progress: Progress tracker
    
    Returns:
        Path to the generated audio file
    """
    global orpheus_model
    
    try:
        model = load_model()
        
        # Process the emotion (convert "None" to an empty string)
        emotion_value = "" if emotion == "None" else emotion
        
        # Generate speech
        progress(0.1, "Loading model...")
        output_file = model.generate_speech_to_file(
            prompt=text,
            voice=voice,
            emotion=emotion_value,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )
        
        progress(1.0, "Generation complete")
        return output_file, f"Generated speech with voice '{voice}'" + (f" and emotion '{emotion}'" if emotion != "None" else "")
    
    except Exception as e:
        logger.error(f"Error generating speech: {e}")
        return None, f"Error: {str(e)}"

def prepare_dataset(audio_dir, speaker_name, progress=gr.Progress(track_tqdm=True)):
    """
    Prepare a dataset for fine-tuning.
    
    Args:
        audio_dir: Directory containing audio files
        speaker_name: Name to give the speaker/voice
        progress: Progress tracker
    
    Returns:
        Message about the dataset preparation
    """
    global finetune_handler
    
    try:
        if finetune_handler is None:
            finetune_handler = OrpheusFinetune()
        
        progress(0.1, "Scanning audio directory...")
        
        if not os.path.exists(audio_dir):
            return None, f"Error: Directory '{audio_dir}' does not exist"
        
        # Count audio files for progress updates
        audio_files = []
        for ext in [".wav", ".mp3", ".flac", ".m4a", ".ogg"]:
            audio_files.extend(list(Path(audio_dir).glob(f"*{ext}")))
        
        if not audio_files:
            return None, f"Error: No audio files found in '{audio_dir}'"
        
        file_count = len(audio_files)
        progress(0.2, f"Found {file_count} audio files. Preparing dataset...")
        
        # Prepare dataset
        dataset_dir = finetune_handler.prepare_dataset(audio_dir, speaker_name)
        
        # Update progress
        progress(0.8, "Dataset prepared. Transcribing audio...")
        
        # Start transcription in the background (this is time-consuming)
        # For simplicity, we won't wait for it to complete here
        # In a real implementation, you might want to use a separate button for this step
        
        return dataset_dir, f"Dataset prepared successfully with {file_count} audio files. Dataset saved to {dataset_dir}"
    
    except Exception as e:
        logger.error(f"Error preparing dataset: {e}")
        return None, f"Error: {str(e)}"

def transcribe_dataset(dataset_dir, progress=gr.Progress(track_tqdm=True)):
    """
    Transcribe a dataset for fine-tuning.
    
    Args:
        dataset_dir: Path to the prepared dataset directory
        progress: Progress tracker
    
    Returns:
        Message about the transcription process
    """
    global finetune_handler
    
    try:
        if finetune_handler is None:
            finetune_handler = OrpheusFinetune()
        
        progress(0.1, "Loading dataset...")
        
        if not os.path.exists(dataset_dir):
            return None, f"Error: Dataset directory '{dataset_dir}' does not exist"
        
        # Update progress
        def update_progress(value):
            progress(0.1 + value * 0.8, f"Transcribing audio ({int(value * 100)}%)...")
        
        # Transcribe dataset
        transcribed_dir = finetune_handler.transcribe_dataset(dataset_dir, update_progress)
        
        progress(1.0, "Transcription complete")
        return transcribed_dir, f"Dataset transcribed successfully. Transcribed dataset saved to {transcribed_dir}"
    
    except Exception as e:
        logger.error(f"Error transcribing dataset: {e}")
        return None, f"Error: {str(e)}"

def start_finetune(dataset_dir, speaker_name, base_model, learning_rate, epochs, 
                 batch_size, progress=gr.Progress(track_tqdm=True)):
    """
    Start the fine-tuning process.
    
    Args:
        dataset_dir: Path to the transcribed dataset directory
        speaker_name: Name of the speaker/voice to create
        base_model: Base model to fine-tune
        learning_rate: Learning rate for training
        epochs: Number of training epochs
        batch_size: Batch size for training
        progress: Progress tracker
    
    Returns:
        Message about the fine-tuning process
    """
    global finetune_handler
    
    try:
        if finetune_handler is None:
            finetune_handler = OrpheusFinetune()
        
        progress(0.1, "Preparing training configuration...")
        
        if not os.path.exists(dataset_dir):
            return None, f"Error: Dataset directory '{dataset_dir}' does not exist"
        
        # Prepare training arguments
        training_args = {
            "learning_rate": float(learning_rate),
            "num_train_epochs": int(epochs),
            "batch_size": int(batch_size),
        }
        
        # Prepare training configuration
        config_path = finetune_handler.prepare_training_config(
            dataset_dir=dataset_dir,
            speaker_name=speaker_name,
            base_model=base_model,
            training_args=training_args
        )
        
        # Start fine-tuning with progress updates
        def update_progress(value):
            progress(0.2 + value * 0.7, f"Fine-tuning in progress ({int(value * 100)}%)...")
        
        # Run fine-tuning
        output_dir = finetune_handler.run_finetune(config_path, update_progress)
        
        progress(1.0, "Fine-tuning complete")
        return output_dir, f"Fine-tuning completed successfully. Model saved to {output_dir}"
    
    except Exception as e:
        logger.error(f"Error during fine-tuning: {e}")
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

def render_orpheus(arg_handler: ArgHandler):
    """
    Render the Orpheus TTS UI tab.
    
    Args:
        arg_handler: The ArgHandler instance
        
    Returns:
        None
    """
    global SEND_TO_PROCESS_BUTTON, OUTPUT_AUDIO, FINETUNE_OUTPUT
    
    with gr.Tabs():
        # Inference Tab
        with gr.TabItem("Text to Speech", id="orpheus_tts"):
            with gr.Row():
                with gr.Column():  # Left column - Text Input and Parameters
                    # Text input area
                    text_input = gr.TextArea(
                        label="Text to Synthesize",
                        placeholder="Enter your text here...",
                        lines=10,
                        elem_id="orpheus_text_input",
                        elem_classes="hintitem"
                    )
                    
                    # Voice and emotion selection
                    with gr.Row():
                        voice_dropdown = gr.Dropdown(
                            choices=AVAILABLE_VOICES,
                            value="tara",
                            label="Voice",
                            elem_id="orpheus_voice",
                            elem_classes="hintitem"
                        )
                        
                        emotion_dropdown = gr.Dropdown(
                            choices=AVAILABLE_EMOTIONS,
                            value="None",
                            label="Emotion",
                            elem_id="orpheus_emotion",
                            elem_classes="hintitem"
                        )
                
                with gr.Column():  # Middle column - Advanced Parameters
                    with gr.Group():
                        gr.Markdown("### Generation Parameters")
                        temperature_slider = gr.Slider(
                            minimum=0.1,
                            maximum=1.5,
                            value=0.7,
                            step=0.1,
                            label="Temperature",
                            elem_id="orpheus_temperature",
                            elem_classes="hintitem"
                        )
                        
                        top_p_slider = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.9,
                            step=0.1,
                            label="Top-p",
                            elem_id="orpheus_top_p",
                            elem_classes="hintitem"
                        )
                        
                        repetition_penalty_slider = gr.Slider(
                            minimum=1.0,
                            maximum=2.0,
                            value=1.1,
                            step=0.1,
                            label="Repetition Penalty",
                            elem_id="orpheus_repetition_penalty",
                            elem_classes="hintitem"
                        )
                
                with gr.Column():  # Right column - Action Buttons and Output
                    with gr.Group():
                        with gr.Row():
                            generate_btn = gr.Button(
                                "Generate",
                                variant="primary",
                                elem_id="orpheus_generate_btn",
                                elem_classes="hintitem"
                            )
                            
                            SEND_TO_PROCESS_BUTTON = gr.Button(
                                "Send to Process",
                                elem_id="orpheus_send_to_process",
                                elem_classes="hintitem"
                            )
                    
                    # Output displays
                    OUTPUT_AUDIO = gr.Audio(
                        label="Generated Speech",
                        type="filepath",
                        elem_id="orpheus_output_audio",
                        elem_classes="hintitem"
                    )
                    
                    output_message = gr.Textbox(
                        label="Output Message",
                        elem_id="orpheus_output_message",
                        elem_classes="hintitem"
                    )
                    
            # Connect the generate button to the generate_speech function
            generate_btn.click(
                fn=generate_speech,
                inputs=[
                    text_input,
                    voice_dropdown,
                    emotion_dropdown,
                    temperature_slider,
                    top_p_slider,
                    repetition_penalty_slider
                ],
                outputs=[OUTPUT_AUDIO, output_message]
            )
        
        # Fine-tuning Tab
        with gr.TabItem("Fine-tune", id="orpheus_finetune"):
            with gr.Row():
                with gr.Column():  # Left column - Dataset Preparation
                    with gr.Group():
                        gr.Markdown("### Prepare Dataset")
                        audio_dir = gr.Textbox(
                            label="Audio Directory",
                            placeholder="Path to directory containing audio files",
                            elem_id="orpheus_audio_dir",
                            elem_classes="hintitem"
                        )
                        
                        speaker_name = gr.Textbox(
                            label="Speaker Name",
                            placeholder="Name for the new voice",
                            elem_id="orpheus_speaker_name",
                            elem_classes="hintitem"
                        )
                        
                        prepare_dataset_btn = gr.Button(
                            "Prepare Dataset",
                            variant="primary",
                            elem_id="orpheus_prepare_dataset_btn",
                            elem_classes="hintitem"
                        )
                        
                        dataset_output = gr.Textbox(
                            label="Dataset Directory",
                            placeholder="Will contain the path to the prepared dataset",
                            elem_id="orpheus_dataset_output",
                            elem_classes="hintitem"
                        )
                        
                        dataset_message = gr.Textbox(
                            label="Dataset Message",
                            elem_id="orpheus_dataset_message",
                            elem_classes="hintitem"
                        )
                
                with gr.Column():  # Middle column - Transcription
                    with gr.Group():
                        gr.Markdown("### Transcribe Dataset")
                        transcribe_dataset_btn = gr.Button(
                            "Transcribe Dataset",
                            variant="primary",
                            elem_id="orpheus_transcribe_dataset_btn",
                            elem_classes="hintitem"
                        )
                        
                        transcribed_output = gr.Textbox(
                            label="Transcribed Dataset",
                            placeholder="Will contain the path to the transcribed dataset",
                            elem_id="orpheus_transcribed_output",
                            elem_classes="hintitem"
                        )
                        
                        transcription_message = gr.Textbox(
                            label="Transcription Message",
                            elem_id="orpheus_transcription_message",
                            elem_classes="hintitem"
                        )
                
                with gr.Column():  # Right column - Fine-tuning
                    with gr.Group():
                        gr.Markdown("### Fine-tuning Parameters")
                        base_model = gr.Textbox(
                            label="Base Model",
                            value="canopylabs/orpheus-tts-0.1-pretrained",
                            elem_id="orpheus_base_model",
                            elem_classes="hintitem"
                        )
                        
                        learning_rate = gr.Number(
                            label="Learning Rate",
                            value=5e-5,
                            precision=5,
                            elem_id="orpheus_learning_rate",
                            elem_classes="hintitem"
                        )
                        
                        epochs = gr.Number(
                            label="Epochs",
                            value=3,
                            precision=0,
                            elem_id="orpheus_epochs",
                            elem_classes="hintitem"
                        )
                        
                        batch_size = gr.Number(
                            label="Batch Size",
                            value=1,
                            precision=0,
                            elem_id="orpheus_batch_size",
                            elem_classes="hintitem"
                        )
                        
                        finetune_btn = gr.Button(
                            "Start Fine-tuning",
                            variant="primary",
                            elem_id="orpheus_finetune_btn",
                            elem_classes="hintitem"
                        )
                        
                        FINETUNE_OUTPUT = gr.Textbox(
                            label="Fine-tuned Model",
                            placeholder="Will contain the path to the fine-tuned model",
                            elem_id="orpheus_finetune_output",
                            elem_classes="hintitem"
                        )
                        
                        finetune_message = gr.Textbox(
                            label="Fine-tuning Message",
                            elem_id="orpheus_finetune_message",
                            elem_classes="hintitem"
                        )
            
            # Connect the dataset preparation button
            prepare_dataset_btn.click(
                fn=prepare_dataset,
                inputs=[audio_dir, speaker_name],
                outputs=[dataset_output, dataset_message]
            )
            
            # Connect the transcription button
            transcribe_dataset_btn.click(
                fn=transcribe_dataset,
                inputs=[dataset_output],
                outputs=[transcribed_output, transcription_message]
            )
            
            # Connect the fine-tuning button
            finetune_btn.click(
                fn=start_finetune,
                inputs=[
                    transcribed_output,
                    speaker_name,
                    base_model,
                    learning_rate,
                    epochs,
                    batch_size
                ],
                outputs=[FINETUNE_OUTPUT, finetune_message]
            )

def listen():
    """
    Set up event listeners for inter-tab communication.
    """
    global SEND_TO_PROCESS_BUTTON, OUTPUT_AUDIO
    
    arg_handler = ArgHandler()
    process_inputs = arg_handler.get_element("main", "process_inputs")
    
    if process_inputs and SEND_TO_PROCESS_BUTTON is not None:
        SEND_TO_PROCESS_BUTTON.click(
            fn=send_to_process,
            inputs=[OUTPUT_AUDIO, process_inputs],
            outputs=[process_inputs]
        )

def register_descriptions(arg_handler: ArgHandler):
    """
    Register tooltips and descriptions for UI elements.
    
    Args:
        arg_handler: The ArgHandler instance
    """
    descriptions = {
        # TTS tab
        "orpheus_text_input": "Enter the text you want to convert to speech. You can use newlines to structure your text.",
        "orpheus_voice": "Select one of the available Orpheus voices.",
        "orpheus_emotion": "Apply an emotion to the generated speech. Leave as 'None' for neutral speech.",
        "orpheus_temperature": "Controls randomness in generation. Higher values produce more varied output but may be less coherent.",
        "orpheus_top_p": "Controls diversity of generation. Higher values allow more diverse word choices.",
        "orpheus_repetition_penalty": "Penalizes repetition in the generated speech. Higher values reduce repetition.",
        "orpheus_generate_btn": "Generate speech from the entered text.",
        "orpheus_send_to_process": "Send the generated audio to the Process tab for further processing.",
        "orpheus_output_audio": "The generated speech output.",
        "orpheus_output_message": "Status messages about the generation process.",
        
        # Fine-tune tab
        "orpheus_audio_dir": "Path to a directory containing audio files for voice cloning.",
        "orpheus_speaker_name": "Name to give the new voice you're creating.",
        "orpheus_prepare_dataset_btn": "Prepare a dataset from the audio files for fine-tuning.",
        "orpheus_dataset_output": "Path to the prepared dataset.",
        "orpheus_dataset_message": "Status messages about dataset preparation.",
        "orpheus_transcribe_dataset_btn": "Transcribe the audio files in the dataset (optional but recommended).",
        "orpheus_transcribed_output": "Path to the transcribed dataset.",
        "orpheus_transcription_message": "Status messages about transcription.",
        "orpheus_base_model": "The base Orpheus model to fine-tune. Default is the pretrained model.",
        "orpheus_learning_rate": "Learning rate for fine-tuning. Lower values are more stable but train slower.",
        "orpheus_epochs": "Number of training epochs. More epochs can improve quality but risk overfitting.",
        "orpheus_batch_size": "Batch size for training. Reduce if you encounter memory issues.",
        "orpheus_finetune_btn": "Start the fine-tuning process.",
        "orpheus_finetune_output": "Path to the fine-tuned model.",
        "orpheus_finetune_message": "Status messages about fine-tuning."
    }
    
    for elem_id, description in descriptions.items():
        arg_handler.register_description("orpheus", elem_id, description)

def register_api_endpoints(api):
    """
    Register API endpoints for Orpheus TTS
    
    Args:
        api: FastAPI application instance
    """
    @api.post("/api/v1/orpheus/generate")
    async def api_generate_speech(
        text: str = Form(...),
        voice: str = Form("tara"),
        emotion: str = Form("None"),
        temperature: float = Form(0.7),
        top_p: float = Form(0.9),
        repetition_penalty: float = Form(1.0)
    ):
        """
        Generate speech using the Orpheus TTS model
        
        Args:
            text: Text to convert to speech
            voice: Voice to use
            emotion: Emotion to apply
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            repetition_penalty: Repetition penalty parameter
            
        Returns:
            Generated audio file
        """
        try:
            # Validate input
            if not text or text.strip() == "":
                raise HTTPException(status_code=400, detail="Text cannot be empty")
                
            # Check if voice is valid
            if voice not in AVAILABLE_VOICES:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid voice: {voice}. Available voices: {', '.join(AVAILABLE_VOICES)}"
                )
                
            # Check if emotion is valid
            if emotion != "None" and emotion not in AVAILABLE_EMOTIONS:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid emotion: {emotion}. Available emotions: {', '.join(['None'] + AVAILABLE_EMOTIONS)}"
                )
                
            # Generate speech
            output_file, message = generate_speech(
                text=text,
                voice=voice,
                emotion=emotion,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )
            
            if not output_file or not os.path.exists(output_file):
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to generate speech: {message}"
                )
                
            return FileResponse(
                output_file,
                media_type="audio/wav",
                filename=os.path.basename(output_file)
            )
            
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            logger.exception("Error in Orpheus speech generation:")
            raise HTTPException(status_code=500, detail=f"Speech generation error: {str(e)}")
            
    @api.post("/api/v1/orpheus/finetune")
    async def api_finetune_model(
        background_tasks: BackgroundTasks,
        speaker_name: str = Form(...),
        base_model: str = Form("canopylabs/orpheus-tts-0.1-finetune-prod"),
        learning_rate: float = Form(1e-5),
        epochs: int = Form(10),
        batch_size: int = Form(16),
        audio_files: List[UploadFile] = File(...)
    ):
        """
        Finetune the Orpheus TTS model on a custom voice
        
        Args:
            background_tasks: FastAPI background tasks
            speaker_name: Name for the new voice
            base_model: Base model to finetune from
            learning_rate: Learning rate for training
            epochs: Number of training epochs
            batch_size: Training batch size
            audio_files: Audio files for the custom voice
            
        Returns:
            Status information and job ID
        """
        try:
            # Validate input
            if not speaker_name or speaker_name.strip() == "":
                raise HTTPException(status_code=400, detail="Speaker name cannot be empty")
                
            if not audio_files or len(audio_files) == 0:
                raise HTTPException(status_code=400, detail="No audio files provided")
                
            # Create temporary directory for processing
            temp_dir = tempfile.mkdtemp(prefix="orpheus_finetune_")
            dataset_dir = os.path.join(temp_dir, "dataset")
            os.makedirs(dataset_dir, exist_ok=True)
            
            # Save uploaded audio files
            for audio_file in audio_files:
                file_path = os.path.join(dataset_dir, audio_file.filename)
                with open(file_path, "wb") as f:
                    content = await audio_file.read()
                    f.write(content)
            
            # Start finetuning in the background
            job_id = f"finetune_{int(time.time())}"
            background_tasks.add_task(
                run_finetune_job,
                job_id=job_id,
                dataset_dir=dataset_dir,
                speaker_name=speaker_name,
                base_model=base_model,
                learning_rate=learning_rate,
                epochs=epochs,
                batch_size=batch_size
            )
            
            return {
                "status": "started",
                "job_id": job_id,
                "message": f"Finetuning job started for speaker '{speaker_name}' with {len(audio_files)} audio files"
            }
            
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            logger.exception("Error starting Orpheus finetuning:")
            raise HTTPException(status_code=500, detail=f"Finetuning error: {str(e)}")
            
    @api.get("/api/v1/orpheus/voices")
    async def api_list_voices():
        """
        List available Orpheus voices
        
        Returns:
            List of available voices and emotions
        """
        try:
            return {
                "voices": AVAILABLE_VOICES,
                "emotions": AVAILABLE_EMOTIONS
            }
            
        except Exception as e:
            logger.exception("Error listing Orpheus voices:")
            raise HTTPException(status_code=500, detail=f"Error listing voices: {str(e)}")
            
def run_finetune_job(job_id, dataset_dir, speaker_name, base_model, learning_rate, epochs, batch_size):
    """Run a finetuning job in the background"""
    try:
        # Prepare the dataset
        prepare_dataset(dataset_dir, speaker_name)
        
        # Transcribe the dataset
        transcribe_dataset(dataset_dir)
        
        # Start the finetuning
        start_finetune(
            dataset_dir=dataset_dir,
            speaker_name=speaker_name,
            base_model=base_model,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Record the job completion
        job_status = {
            "status": "completed",
            "job_id": job_id,
            "speaker_name": speaker_name,
            "completion_time": time.time()
        }
        
        # Save job status to a file
        status_dir = os.path.join(output_path, "orpheus", "jobs")
        os.makedirs(status_dir, exist_ok=True)
        with open(os.path.join(status_dir, f"{job_id}.json"), "w") as f:
            import json
            json.dump(job_status, f)
            
    except Exception as e:
        logger.error(f"Error in finetuning job {job_id}: {e}")
        traceback.print_exc()
        
        # Record job failure
        job_status = {
            "status": "failed",
            "job_id": job_id,
            "speaker_name": speaker_name,
            "error": str(e),
            "completion_time": time.time()
        }
        
        # Save job status to a file
        status_dir = os.path.join(output_path, "orpheus", "jobs")
        os.makedirs(status_dir, exist_ok=True)
        with open(os.path.join(status_dir, f"{job_id}.json"), "w") as f:
            import json
            json.dump(job_status, f)
    
    finally:
        # Clean up the temporary directory
        if os.path.exists(dataset_dir):
            try:
                shutil.rmtree(dataset_dir)
            except Exception as e:
                logger.error(f"Error cleaning up dataset directory: {e}")