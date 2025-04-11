import gc
import json
import logging
import os
import time
from typing import List, Optional

import gradio as gr
import whisperx
from fastapi import UploadFile, File, Form, HTTPException, Body
from fastapi.responses import FileResponse, JSONResponse
import tempfile
from pathlib import Path
import base64
from pydantic import BaseModel, Field

from handlers.args import ArgHandler
from handlers.config import output_path
from util.data_classes import ProjectFiles

logger = logging.getLogger(__name__)
arg_handler = ArgHandler()

# Global variables for inter-tab communication
SEND_TO_PROCESS_BUTTON = None
OUTPUT_TRANSCRIPTION = None
OUTPUT_AUDIO = None

def process_transcription(
    audio_files,
    language="auto",
    align_output=True,
    assign_speakers=True,
    min_speakers=None,
    max_speakers=None,
    batch_size=16,
    compute_type="float16",
    progress=gr.Progress(track_tqdm=True)
):
    """Transcribe audio files using WhisperX with alignment and speaker diarization."""
    try:
        # Initialize counters for progress tracking
        total_steps = len(audio_files) * (1 + (1 if align_output else 0) + (1 if assign_speakers else 0))
        current_step = 0
        
        # Convert min/max speakers to integers if provided
        if min_speakers and min_speakers.strip():
            min_speakers = int(min_speakers)
        else:
            min_speakers = None
            
        if max_speakers and max_speakers.strip():
            max_speakers = int(max_speakers)
        else:
            max_speakers = None
            
        device = "cuda"
        results = []
        output_files = []
        
        # Create output directory
        timestamp = int(time.time())
        output_folder = os.path.join(output_path, "transcriptions", f"transcribe_{timestamp}")
        os.makedirs(output_folder, exist_ok=True)
        
        progress(0, "Loading transcription model...")
        model = whisperx.load_model("large-v3", device, compute_type=compute_type)
        
        # Process each audio file
        for audio_idx, audio_file in enumerate(audio_files):
            try:
                file_name = os.path.basename(audio_file)
                base_name, _ = os.path.splitext(file_name)
                
                # 1. Transcribe with original whisper
                progress(current_step/total_steps, f"Transcribing {file_name}")
                audio = whisperx.load_audio(audio_file)
                
                # Determine language if set to auto
                detect_language = language == "auto"
                result = model.transcribe(
                    audio, 
                    batch_size=batch_size,
                    language=None if detect_language else language
                )
                current_step += 1
                
                # 2. Align whisper output if requested
                if align_output:
                    progress(current_step/total_steps, f"Aligning {file_name}")
                    model_a, metadata = whisperx.load_align_model(
                        language_code=result["language"], 
                        device=device
                    )
                    result = whisperx.align(
                        result["segments"], 
                        model_a, 
                        metadata, 
                        audio, 
                        device,
                        return_char_alignments=False
                    )
                    current_step += 1
                    # Clean up alignment model to save memory
                    del model_a
                    gc.collect()
                
                # 3. Assign speaker labels if requested
                if assign_speakers:
                    progress(current_step/total_steps, f"Assigning speakers for {file_name}")
                    diarize_model = whisperx.DiarizationPipeline(
                        model_name="tensorlake/speaker-diarization-3.1",
                        device=device
                    )
                    
                    # Add min/max number of speakers if specified
                    diarize_kwargs = {}
                    if min_speakers is not None:
                        diarize_kwargs["min_speakers"] = min_speakers
                    if max_speakers is not None:
                        diarize_kwargs["max_speakers"] = max_speakers
                        
                    diarize_segments = diarize_model(audio, **diarize_kwargs)
                    result = whisperx.assign_word_speakers(diarize_segments, result)
                    current_step += 1
                    
                    # Clean up diarization model
                    del diarize_model
                    gc.collect()
                
                # Save results to files
                output_json = os.path.join(output_folder, f"{base_name}.json")
                with open(output_json, 'w') as f:
                    json.dump(result, f, indent=4)
                    
                # Generate text file with transcript
                output_txt = os.path.join(output_folder, f"{base_name}.txt")
                with open(output_txt, 'w') as f:
                    # If speaker diarization was performed, include speaker information
                    if assign_speakers and "speaker" in result["segments"][0]:
                        for segment in result["segments"]:
                            speaker = segment.get("speaker", "UNKNOWN")
                            text = segment["text"]
                            start = segment["start"]
                            end = segment["end"]
                            f.write(f"[{speaker}] ({start:.2f}s - {end:.2f}s): {text}\n")
                    else:
                        for segment in result["segments"]:
                            text = segment["text"]
                            start = segment["start"]
                            end = segment["end"]
                            f.write(f"({start:.2f}s - {end:.2f}s): {text}\n")

                # Generate LRC format file
                output_lrc = os.path.join(output_folder, f"{base_name}.lrc")
                with open(output_lrc, 'w', encoding='utf-8') as f:
                    for segment in result["segments"]:
                        # Convert seconds to MM:SS.xx format
                        start_time = segment["start"]
                        minutes = int(start_time // 60)
                        seconds = start_time % 60
                        timestamp = f"[{minutes:02d}:{seconds:05.2f}]"
                        
                        # Add speaker label if available
                        text = segment["text"].strip()
                        if assign_speakers and "speaker" in segment:
                            text = f"[{segment['speaker']}] {text}"
                        
                        f.write(f"{timestamp}{text}\n")
                
                # Store results and output files
                results.append(result)
                output_files.extend([output_json, output_txt, output_lrc])
                
            except Exception as e:
                logger.exception(f"Error processing file {audio_file}: {e}")
                
        # Clean up main model
        del model
        gc.collect()
            
        # Return summary and output files
        summary = f"Transcribed {len(results)} files to {output_folder}"
        return summary, output_files
        
    except Exception as e:
        logger.exception(f"Transcription error: {e}")
        return f"Error: {str(e)}", []


def render(arg_handler: ArgHandler):
    global SEND_TO_PROCESS_BUTTON, OUTPUT_TRANSCRIPTION, OUTPUT_AUDIO
    
    with gr.Blocks() as transcribe:
        gr.Markdown("# 🎙️ Audio Transcription")
        gr.Markdown("Transcribe audio files with high accuracy using WhisperX. Supports multi-speaker diarization and precision timestamps.")
        
        with gr.Row():
            # Left Column - Settings
            with gr.Column():
                gr.Markdown("### 🔧 Settings")
                
                language = gr.Dropdown(
                    label="Language",
                    choices=["auto", "en", "fr", "de", "es", "it", "ja", "zh", "pt", "ru", "nl", "ko", "ar", "tr", "pl", "hu"],
                    value="auto",
                    elem_classes="hintitem",
                    elem_id="transcribe_language",
                    key="transcribe_language"
                )
                
                with gr.Row():
                    align_output = gr.Checkbox(
                        label="Align Output",
                        value=True,
                        elem_classes="hintitem",
                        elem_id="transcribe_align_output",
                        key="transcribe_align_output"
                    )
                    
                    assign_speakers = gr.Checkbox(
                        label="Assign Speakers",
                        value=True,
                        elem_classes="hintitem",
                        elem_id="transcribe_assign_speakers",
                        key="transcribe_assign_speakers"
                    )
                
                with gr.Row():
                    min_speakers = gr.Textbox(
                        label="Min Speakers",
                        placeholder="Leave empty for auto",
                        elem_classes="hintitem",
                        elem_id="transcribe_min_speakers",
                        key="transcribe_min_speakers"
                    )
                    
                    max_speakers = gr.Textbox(
                        label="Max Speakers",
                        placeholder="Leave empty for auto",
                        elem_classes="hintitem",
                        elem_id="transcribe_max_speakers",
                        key="transcribe_max_speakers"
                    )
                
                batch_size = gr.Slider(
                    label="Batch Size",
                    minimum=1,
                    maximum=32,
                    value=16,
                    step=1,
                    elem_classes="hintitem",
                    elem_id="transcribe_batch_size",
                    key="transcribe_batch_size"
                )
                
                compute_type = gr.Dropdown(
                    label="Compute Type",
                    choices=["float16", "float32", "int8"],
                    value="float16",
                    elem_classes="hintitem",
                    elem_id="transcribe_compute_type",
                    key="transcribe_compute_type"
                )
            
            # Middle Column - Input
            with gr.Column():
                gr.Markdown("### 🎤 Input")
                
                input_audio = gr.File(
                    label="Input Audio Files",
                    file_count="multiple",
                    file_types=["audio"],
                    elem_classes="hintitem",
                    elem_id="transcribe_input_audio",
                    key="transcribe_input_audio"
                )
                
                OUTPUT_AUDIO = gr.Audio(
                    label="Audio Preview",
                    type="filepath",
                    visible=False,
                    elem_classes="hintitem",
                    elem_id="transcribe_output_audio",
                    key="transcribe_output_audio"
                )
            
            # Right Column - Actions & Output
            with gr.Column():
                gr.Markdown("### 🎮 Actions")
                
                with gr.Row():
                    transcribe_button = gr.Button(
                        value="Transcribe Audio",
                        variant="primary",
                        elem_classes="hintitem",
                        elem_id="transcribe_button",
                        key="transcribe_button"
                    )
                    
                    SEND_TO_PROCESS_BUTTON = gr.Button(
                        value="Send to Process",
                        variant="secondary",
                        elem_classes="hintitem",
                        elem_id="transcribe_send_to_process",
                        key="transcribe_send_to_process"
                    )
                
                status_display = gr.Textbox(
                    label="Status",
                    placeholder="Ready to transcribe",
                    elem_classes="hintitem",
                    elem_id="transcribe_status",
                    key="transcribe_status"
                )
                
                gr.Markdown("### 📝 Output")
                
                OUTPUT_TRANSCRIPTION = gr.File(
                    label="Transcription Files",
                    file_count="multiple",
                    interactive=False,
                    elem_classes="hintitem",
                    elem_id="transcribe_output_files",
                    key="transcribe_output_files"
                )
                
                output_text = gr.Textbox(
                    label="Preview",
                    placeholder="Transcription will appear here",
                    lines=15,
                    elem_classes="hintitem",
                    elem_id="transcribe_output_text",
                    key="transcribe_output_text"
                )
        
        # Set up event handlers
        def update_preview(selected_file):
            """Update the preview with the content of the selected file"""
            if not selected_file:
                return "", gr.update(visible=False)
                
            # If the selected file is a text file, show its content
            if selected_file.endswith(".txt"):
                with open(selected_file, "r") as f:
                    content = f.read()
                return content, gr.update(visible=False)
                
            # If it's a JSON file, format its content
            elif selected_file.endswith(".json"):
                with open(selected_file, "r") as f:
                    data = json.load(f)
                
                # Format the JSON content for display
                content = "# Transcription Results\n\n"
                for segment in data.get("segments", []):
                    speaker = segment.get("speaker", "")
                    text = segment.get("text", "")
                    start = segment.get("start", 0)
                    end = segment.get("end", 0)
                    
                    if speaker:
                        content += f"[{speaker}] ({start:.2f}s - {end:.2f}s): {text}\n"
                    else:
                        content += f"({start:.2f}s - {end:.2f}s): {text}\n"
                        
                return content, gr.update(visible=False)
                
            # If it's an audio file, show the audio player
            elif selected_file.endswith((".wav", ".mp3", ".flac")):
                return "", gr.update(visible=True, value=selected_file)
                
            return "", gr.update(visible=False)
            
        # Event handler for the transcribe button
        transcribe_button.click(
            fn=process_transcription,
            inputs=[
                input_audio,
                language,
                align_output,
                assign_speakers,
                min_speakers,
                max_speakers,
                batch_size,
                compute_type
            ],
            outputs=[status_display, OUTPUT_TRANSCRIPTION]
        )
        
        # Event handler for file selection
        OUTPUT_TRANSCRIPTION.change(
            fn=update_preview,
            inputs=[OUTPUT_TRANSCRIPTION],
            outputs=[output_text, OUTPUT_AUDIO]
        )
    
    return transcribe


def send_to_process(file_to_send, existing_inputs):
    """Send a file to the process tab"""
    if not file_to_send or not os.path.exists(file_to_send):
        return gr.update()
        
    if file_to_send in existing_inputs:
        return gr.update()
        
    existing_inputs.append(file_to_send)
    return gr.update(value=existing_inputs)


def listen():
    """Set up inter-tab communication"""
    process_inputs = arg_handler.get_element("main", "process_inputs")
    if process_inputs and SEND_TO_PROCESS_BUTTON and OUTPUT_AUDIO:
        SEND_TO_PROCESS_BUTTON.click(
            fn=send_to_process,
            inputs=[OUTPUT_AUDIO, process_inputs],
            outputs=[process_inputs]
        )


def register_descriptions(arg_handler: ArgHandler):
    """Register descriptions for UI elements"""
    descriptions = {
        "language": "Select the language of the audio for better transcription accuracy, or choose 'auto' for automatic detection.",
        "align_output": "Enable to align the transcription with the audio for precise timestamps.",
        "assign_speakers": "Enable to detect and assign different speakers in the audio.",
        "min_speakers": "Minimum number of speakers to detect (leave empty for automatic detection).",
        "max_speakers": "Maximum number of speakers to detect (leave empty for automatic detection).",
        "batch_size": "Batch size for transcription processing. Higher values use more memory but can be faster.",
        "compute_type": "Precision level for computation. Lower precision uses less memory but may be less accurate.",
        "input_audio": "Upload one or more audio files to transcribe.",
        "output_audio": "Preview of the selected audio file.",
        "button": "Start the transcription process.",
        "send_to_process": "Send the selected audio file to the Process tab for further processing.",
        "status": "Current status of the transcription process.",
        "output_files": "Transcription files generated by the process.",
        "output_text": "Preview of the selected transcription file content."
    }
    
    for elem_id, description in descriptions.items():
        arg_handler.register_description("transcribe", elem_id, description)


def register_api_endpoints(api):
    """
    Register API endpoints for transcription functionality
    
    Args:
        api: FastAPI application instance
    """
    # Define Pydantic models for JSON requests
    class FileData(BaseModel):
        filename: str
        content: str  # base64 encoded content
        
    class TranscribeRequest(BaseModel):
        files: List[FileData]
        language: str = "auto"
        align_output: bool = True
        assign_speakers: bool = True
        min_speakers: Optional[str] = None
        max_speakers: Optional[str] = None
        batch_size: int = 16
        compute_type: str = "float16"
    
    @api.post("/api/v1/transcribe", tags=["Transcription"])
    async def api_transcribe(
        files: List[UploadFile] = File(...),
        language: str = Form("auto"),
        align_output: bool = Form(True),
        assign_speakers: bool = Form(True),
        min_speakers: Optional[str] = Form(None),
        max_speakers: Optional[str] = Form(None),
        batch_size: int = Form(16),
        compute_type: str = Form("float16")
    ):
        """
        Transcribe audio files using WhisperX with alignment and speaker diarization.
        
        This endpoint provides high-quality speech-to-text transcription with additional
        features like precise word-level timestamps and speaker identification.
        
        ## Parameters
        
        - **files**: Audio files to transcribe (WAV, MP3, FLAC, etc.)
        - **language**: Language code (ISO 639-1) or "auto" for automatic detection
          - Examples: "en", "fr", "de", "ja", "zh", "es", etc.
        - **align_output**: Create word-level timestamps using phoneme alignment (default: true)
        - **assign_speakers**: Detect and label different speakers in the audio (default: true)
        - **min_speakers**: Minimum number of speakers to detect (optional)
          - Leave empty for automatic detection
        - **max_speakers**: Maximum number of speakers to detect (optional)
          - Leave empty for automatic detection
        - **batch_size**: Batch size for processing (default: 16)
          - Higher values use more memory but may be faster
        - **compute_type**: Precision level for computation (default: "float16")
          - Options: "float16", "float32", "int8"
        
        ## Response Format
        
        The API returns a JSON object containing a summary and links to the generated transcription files.
        """
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save uploaded files
                temp_files = []
                for file in files:
                    file_path = Path(temp_dir) / file.filename
                    with file_path.open("wb") as f:
                        content = await file.read()
                        f.write(content)
                    temp_files.append(str(file_path))
                
                # Process with shared implementation
                return await _transcribe_impl(
                    temp_dir=temp_dir,
                    temp_files=temp_files,
                    language=language,
                    align_output=align_output,
                    assign_speakers=assign_speakers,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                    batch_size=batch_size,
                    compute_type=compute_type,
                    return_json=False
                )
                
        except Exception as e:
            logger.exception(f"API transcription error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @api.post("/api/v1/transcribe_json", tags=["Transcription"])
    async def api_transcribe_json(request: TranscribeRequest = Body(...)):
        """
        Transcribe audio files using WhisperX with alignment and speaker diarization (JSON API)
        
        Request body:
        - files: Array of file objects, each containing filename and base64-encoded content
        - language: Language code or "auto" for automatic detection (default: "auto")
        - align_output: Create word-level timestamps (default: true)
        - assign_speakers: Detect and label different speakers (default: true)
        - min_speakers: Minimum number of speakers to detect (optional)
        - max_speakers: Maximum number of speakers to detect (optional)
        - batch_size: Batch size for processing (default: 16)
        - compute_type: Precision level for computation (default: "float16")
        
        Response:
        - JSON response with base64-encoded transcription files
        """
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save uploaded files from base64
                temp_files = []
                for file_data in request.files:
                    file_path = Path(temp_dir) / file_data.filename
                    with file_path.open("wb") as f:
                        content = base64.b64decode(file_data.content)
                        f.write(content)
                    temp_files.append(str(file_path))
                
                # Process with shared implementation
                return await _transcribe_impl(
                    temp_dir=temp_dir,
                    temp_files=temp_files,
                    language=request.language,
                    align_output=request.align_output,
                    assign_speakers=request.assign_speakers,
                    min_speakers=request.min_speakers,
                    max_speakers=request.max_speakers,
                    batch_size=request.batch_size,
                    compute_type=request.compute_type,
                    return_json=True
                )
                
        except Exception as e:
            logger.exception(f"API transcription error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _transcribe_impl(
        temp_dir,
        temp_files,
        language="auto",
        align_output=True,
        assign_speakers=True,
        min_speakers=None,
        max_speakers=None,
        batch_size=16,
        compute_type="float16",
        return_json=False
    ):
        """Shared implementation for transcription"""
        # Process transcription
        summary, output_files = process_transcription(
            temp_files,
            language=language,
            align_output=align_output,
            assign_speakers=assign_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            batch_size=batch_size,
            compute_type=compute_type
        )
        
        # Filter for JSON files
        json_files = [output for output in output_files if output.endswith(".json")]
        
        if return_json:
            # Return the transcriptions as base64-encoded content
            response_data = {
                "summary": summary,
                "transcriptions": []
            }
            
            for json_file in json_files:
                with open(json_file, "r", encoding="utf-8") as f:
                    transcription_data = f.read()
                
                response_data["transcriptions"].append({
                    "filename": os.path.basename(json_file),
                    "content": transcription_data
                })
            
            return response_data
        else:
            # Return results with file URLs
            return {
                "summary": summary,
                "output_files": [FileResponse(output) for output in json_files]
            } 