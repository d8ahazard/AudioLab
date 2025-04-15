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
from fastapi import UploadFile, File, Form, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Optional, List
from pydantic import BaseModel, Field
import base64
import shutil

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
    Register API endpoints for audio transcription functionality
    
    Args:
        api: FastAPI application instance
    """
    from fastapi import HTTPException, BackgroundTasks, Query, Form, UploadFile, File
    from fastapi.responses import JSONResponse, FileResponse
    from fastapi.encoders import jsonable_encoder
    from pydantic import BaseModel, Field, validator
    from typing import Optional, List, Dict, Any, Union
    import os
    import uuid
    import time
    import json
    import re
    from io import BytesIO

    # Define Pydantic models for request validation
    class TranscriptionOptions(BaseModel):
        model: str = Field(
            "whisper-1", description="The model to use for audio transcription"
        )
        language: Optional[str] = Field(
            None, description="The language of the audio (ISO-639-1 code)"
        )
        prompt: Optional[str] = Field(
            None, description="Text to guide the transcription model"
        )
        response_format: str = Field(
            "json", description="The format of the transcription response"
        )
        temperature: float = Field(
            0.0, description="Sampling temperature", ge=0.0, le=1.0
        )
        timestamp_granularities: Optional[List[str]] = Field(
            None, description="Timestamp granularity levels"
        )
        
        @validator('model')
        def validate_model(cls, v):
            valid_models = ["whisper-1"]
            if v not in valid_models:
                raise ValueError(f"Model must be one of: {', '.join(valid_models)}")
            return v
        
        @validator('language')
        def validate_language(cls, v):
            if v is not None:
                # Validate ISO-639-1 code (2-letter language code)
                if not re.match(r'^[a-z]{2}$', v):
                    raise ValueError("Language must be a valid ISO-639-1 two-letter language code")
            return v
        
        @validator('prompt')
        def validate_prompt(cls, v):
            if v is not None and len(v) > 1000:
                raise ValueError("Prompt cannot exceed 1000 characters")
            return v
        
        @validator('response_format')
        def validate_response_format(cls, v):
            valid_formats = ["json", "text", "srt", "vtt", "verbose_json"]
            if v not in valid_formats:
                raise ValueError(f"Response format must be one of: {', '.join(valid_formats)}")
            return v
        
        @validator('temperature')
        def validate_temperature(cls, v):
            if v < 0.0 or v > 1.0:
                raise ValueError("Temperature must be between 0.0 and 1.0")
            return v
        
        @validator('timestamp_granularities')
        def validate_timestamp_granularities(cls, v):
            if v is not None:
                valid_granularities = ["word", "segment"]
                for item in v:
                    if item not in valid_granularities:
                        raise ValueError(f"Timestamp granularity must be one of: {', '.join(valid_granularities)}")
            return v

    class TranslationOptions(BaseModel):
        model: str = Field(
            "whisper-1", description="The model to use for audio translation"
        )
        prompt: Optional[str] = Field(
            None, description="Text to guide the translation model"
        )
        response_format: str = Field(
            "json", description="The format of the translation response"
        )
        temperature: float = Field(
            0.0, description="Sampling temperature", ge=0.0, le=1.0
        )
        timestamp_granularities: Optional[List[str]] = Field(
            None, description="Timestamp granularity levels"
        )
        
        @validator('model')
        def validate_model(cls, v):
            valid_models = ["whisper-1"]
            if v not in valid_models:
                raise ValueError(f"Model must be one of: {', '.join(valid_models)}")
            return v
        
        @validator('prompt')
        def validate_prompt(cls, v):
            if v is not None and len(v) > 1000:
                raise ValueError("Prompt cannot exceed 1000 characters")
            return v
        
        @validator('response_format')
        def validate_response_format(cls, v):
            valid_formats = ["json", "text", "srt", "vtt", "verbose_json"]
            if v not in valid_formats:
                raise ValueError(f"Response format must be one of: {', '.join(valid_formats)}")
            return v
        
        @validator('temperature')
        def validate_temperature(cls, v):
            if v < 0.0 or v > 1.0:
                raise ValueError("Temperature must be between 0.0 and 1.0")
            return v
        
        @validator('timestamp_granularities')
        def validate_timestamp_granularities(cls, v):
            if v is not None:
                valid_granularities = ["word", "segment"]
                for item in v:
                    if item not in valid_granularities:
                        raise ValueError(f"Timestamp granularity must be one of: {', '.join(valid_granularities)}")
            return v

    @api.post("/api/v1/audio/transcriptions", tags=["Audio Transcription"])
    async def api_transcribe_audio(
        file: UploadFile = File(...),
        model: str = Form("whisper-1"),
        language: Optional[str] = Form(None),
        prompt: Optional[str] = Form(None),
        response_format: str = Form("json"),
        temperature: float = Form(0.0),
        timestamp_granularities: Optional[List[str]] = Form(None),
        background_tasks: BackgroundTasks = None
    ):
        """
        Transcribe audio to text
        
        This endpoint transcribes the uploaded audio file to text using the Whisper model.
        
        Parameters:
        - file: The audio file to transcribe (required, max 25MB)
        - model: The transcription model to use (default: "whisper-1")
        - language: The language of the audio in ISO-639-1 format (optional, e.g., "en", "fr")
        - prompt: Text to guide the transcription model (optional, max 1000 chars)
        - response_format: Output format (default: "json", options: "json", "text", "srt", "vtt", "verbose_json")
        - temperature: Sampling temperature (default: 0.0, range: 0.0-1.0)
        - timestamp_granularities: Timestamp detail level (optional, options: "word", "segment")
        
        Returns:
        - JSON with transcription details and results
        """
        try:
            # Check file size limit (25MB)
            file_size_limit = 25 * 1024 * 1024  # 25MB in bytes
            if file.size > file_size_limit:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Audio file exceeds size limit of 25MB. Uploaded file size: {file.size / (1024 * 1024):.2f}MB"
                )
            
            # Read file content
            audio_content = await file.read()
            
            # Validate input parameters
            options = TranscriptionOptions(
                model=model,
                language=language,
                prompt=prompt,
                response_format=response_format,
                temperature=temperature,
                timestamp_granularities=timestamp_granularities
            )
            
            # Create a temporary file to store the uploaded audio
            temp_dir = os.path.join(output_path, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            
            temp_audio_id = str(uuid.uuid4())
            temp_audio_filename = f"temp_audio_{temp_audio_id}_{file.filename}"
            temp_audio_path = os.path.join(temp_dir, temp_audio_filename)
            
            with open(temp_audio_path, "wb") as f:
                f.write(audio_content)
            
            # Generate a unique ID for this transcription
            transcription_id = str(uuid.uuid4())
            timestamp = int(time.time())
            
            # Create output directory if it doesn't exist
            transcription_dir = os.path.join(output_path, "transcriptions")
            os.makedirs(transcription_dir, exist_ok=True)
            
            # Initialize transcription engine
            transcriber = WhisperTranscriber()
            
            # Perform transcription
            result = transcriber.transcribe(
                audio_file=temp_audio_path,
                model=options.model,
                language=options.language,
                prompt=options.prompt,
                response_format=options.response_format,
                temperature=options.temperature,
                timestamp_granularities=options.timestamp_granularities
            )
            
            # Clean up temporary file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            
            if not result.get("success", False):
                raise HTTPException(status_code=500, detail=result.get("error", "Failed to transcribe audio"))
            
            # Get transcription data
            transcription_data = result.get("transcription")
            if not transcription_data:
                raise HTTPException(status_code=500, detail="No transcription data generated")
            
            # Save transcription results based on format
            if options.response_format == "json":
                output_filename = f"transcription_{transcription_id}.json"
                output_filepath = os.path.join(transcription_dir, output_filename)
                with open(output_filepath, "w", encoding="utf-8") as f:
                    json.dump(transcription_data, f, indent=2, ensure_ascii=False)
            
            elif options.response_format == "verbose_json":
                output_filename = f"transcription_{transcription_id}.json"
                output_filepath = os.path.join(transcription_dir, output_filename)
                with open(output_filepath, "w", encoding="utf-8") as f:
                    json.dump(transcription_data, f, indent=2, ensure_ascii=False)
            
            elif options.response_format in ["text", "srt", "vtt"]:
                output_filename = f"transcription_{transcription_id}.{options.response_format}"
                output_filepath = os.path.join(transcription_dir, output_filename)
                with open(output_filepath, "w", encoding="utf-8") as f:
                    f.write(transcription_data)
            
            # Create download URL
            download_url = f"/api/v1/audio/transcription/download/{output_filename}"
            
            # Get file size
            file_size = os.path.getsize(output_filepath)
            
            # Save metadata
            metadata_filename = f"transcription_{transcription_id}_metadata.json"
            metadata_filepath = os.path.join(transcription_dir, metadata_filename)
            
            metadata = {
                "id": transcription_id,
                "timestamp": timestamp,
                "original_filename": file.filename,
                "model": options.model,
                "language": options.language,
                "prompt": options.prompt,
                "response_format": options.response_format,
                "temperature": options.temperature,
                "timestamp_granularities": options.timestamp_granularities,
                "output_file": {
                    "filename": output_filename,
                    "format": options.response_format,
                    "download_url": download_url,
                    "file_size_bytes": file_size
                }
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
            # For text format, include the text content directly
            transcription_content = None
            if options.response_format == "text":
                transcription_content = transcription_data
            elif options.response_format in ["json", "verbose_json"]:
                transcription_content = transcription_data
            
            response_data = {
                "success": True,
                "transcription_id": transcription_id,
                "output_file": {
                    "filename": output_filename,
                    "format": options.response_format,
                    "download_url": download_url,
                    "file_size_bytes": file_size
                },
                "metadata": {
                    "original_filename": file.filename,
                    "model": options.model,
                    "language": options.language,
                    "prompt": options.prompt,
                    "response_format": options.response_format,
                    "temperature": options.temperature,
                    "timestamp_granularities": options.timestamp_granularities,
                    "timestamp": timestamp
                }
            }
            
            # Include transcription content in the response
            if transcription_content:
                response_data["transcription"] = transcription_content
            
            return JSONResponse(content=jsonable_encoder(response_data))
            
        except ValueError as e:
            # Handle validation errors
            raise HTTPException(status_code=400, detail=str(e))
        except HTTPException:
            raise
        except Exception as e:
            import traceback
            logger.exception(f"API transcription error: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=str(e))

    @api.post("/api/v1/audio/translations", tags=["Audio Transcription"])
    async def api_translate_audio(
        file: UploadFile = File(...),
        model: str = Form("whisper-1"),
        prompt: Optional[str] = Form(None),
        response_format: str = Form("json"),
        temperature: float = Form(0.0),
        timestamp_granularities: Optional[List[str]] = Form(None),
        background_tasks: BackgroundTasks = None
    ):
        """
        Translate audio to English text
        
        This endpoint translates the uploaded audio file to English text using the Whisper model.
        
        Parameters:
        - file: The audio file to translate (required, max 25MB)
        - model: The translation model to use (default: "whisper-1")
        - prompt: Text to guide the translation model (optional, max 1000 chars)
        - response_format: Output format (default: "json", options: "json", "text", "srt", "vtt", "verbose_json")
        - temperature: Sampling temperature (default: 0.0, range: 0.0-1.0)
        - timestamp_granularities: Timestamp detail level (optional, options: "word", "segment")
        
        Returns:
        - JSON with translation details and results
        """
        try:
            # Check file size limit (25MB)
            file_size_limit = 25 * 1024 * 1024  # 25MB in bytes
            if file.size > file_size_limit:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Audio file exceeds size limit of 25MB. Uploaded file size: {file.size / (1024 * 1024):.2f}MB"
                )
            
            # Read file content
            audio_content = await file.read()
            
            # Validate input parameters
            options = TranslationOptions(
                model=model,
                prompt=prompt,
                response_format=response_format,
                temperature=temperature,
                timestamp_granularities=timestamp_granularities
            )
            
            # Create a temporary file to store the uploaded audio
            temp_dir = os.path.join(output_path, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            
            temp_audio_id = str(uuid.uuid4())
            temp_audio_filename = f"temp_audio_{temp_audio_id}_{file.filename}"
            temp_audio_path = os.path.join(temp_dir, temp_audio_filename)
            
            with open(temp_audio_path, "wb") as f:
                f.write(audio_content)
            
            # Generate a unique ID for this translation
            translation_id = str(uuid.uuid4())
            timestamp = int(time.time())
            
            # Create output directory if it doesn't exist
            translation_dir = os.path.join(output_path, "translations")
            os.makedirs(translation_dir, exist_ok=True)
            
            # Initialize translation engine
            translator = WhisperTranslator()
            
            # Perform translation
            result = translator.translate(
                audio_file=temp_audio_path,
                model=options.model,
                prompt=options.prompt,
                response_format=options.response_format,
                temperature=options.temperature,
                timestamp_granularities=options.timestamp_granularities
            )
            
            # Clean up temporary file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            
            if not result.get("success", False):
                raise HTTPException(status_code=500, detail=result.get("error", "Failed to translate audio"))
            
            # Get translation data
            translation_data = result.get("translation")
            if not translation_data:
                raise HTTPException(status_code=500, detail="No translation data generated")
            
            # Save translation results based on format
            if options.response_format == "json":
                output_filename = f"translation_{translation_id}.json"
                output_filepath = os.path.join(translation_dir, output_filename)
                with open(output_filepath, "w", encoding="utf-8") as f:
                    json.dump(translation_data, f, indent=2, ensure_ascii=False)
            
            elif options.response_format == "verbose_json":
                output_filename = f"translation_{translation_id}.json"
                output_filepath = os.path.join(translation_dir, output_filename)
                with open(output_filepath, "w", encoding="utf-8") as f:
                    json.dump(translation_data, f, indent=2, ensure_ascii=False)
            
            elif options.response_format in ["text", "srt", "vtt"]:
                output_filename = f"translation_{translation_id}.{options.response_format}"
                output_filepath = os.path.join(translation_dir, output_filename)
                with open(output_filepath, "w", encoding="utf-8") as f:
                    f.write(translation_data)
            
            # Create download URL
            download_url = f"/api/v1/audio/translation/download/{output_filename}"
            
            # Get file size
            file_size = os.path.getsize(output_filepath)
            
            # Save metadata
            metadata_filename = f"translation_{translation_id}_metadata.json"
            metadata_filepath = os.path.join(translation_dir, metadata_filename)
            
            metadata = {
                "id": translation_id,
                "timestamp": timestamp,
                "original_filename": file.filename,
                "model": options.model,
                "prompt": options.prompt,
                "response_format": options.response_format,
                "temperature": options.temperature,
                "timestamp_granularities": options.timestamp_granularities,
                "output_file": {
                    "filename": output_filename,
                    "format": options.response_format,
                    "download_url": download_url,
                    "file_size_bytes": file_size
                }
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
            # For text format, include the text content directly
            translation_content = None
            if options.response_format == "text":
                translation_content = translation_data
            elif options.response_format in ["json", "verbose_json"]:
                translation_content = translation_data
            
            response_data = {
                "success": True,
                "translation_id": translation_id,
                "output_file": {
                    "filename": output_filename,
                    "format": options.response_format,
                    "download_url": download_url,
                    "file_size_bytes": file_size
                },
                "metadata": {
                    "original_filename": file.filename,
                    "model": options.model,
                    "prompt": options.prompt,
                    "response_format": options.response_format,
                    "temperature": options.temperature,
                    "timestamp_granularities": options.timestamp_granularities,
                    "timestamp": timestamp
                }
            }
            
            # Include translation content in the response
            if translation_content:
                response_data["translation"] = translation_content
            
            return JSONResponse(content=jsonable_encoder(response_data))
            
        except ValueError as e:
            # Handle validation errors
            raise HTTPException(status_code=400, detail=str(e))
        except HTTPException:
            raise
        except Exception as e:
            import traceback
            logger.exception(f"API translation error: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=str(e))

    @api.get("/api/v1/audio/transcription/download/{filename}", tags=["Audio Transcription"])
    async def api_download_transcription(filename: str):
        """
        Download a transcription file
        
        Parameters:
        - filename: The name of the transcription file to download
        
        Returns:
        - The transcription file
        """
        try:
            # Validate filename format (prevent path traversal)
            valid_pattern = r'^transcription_[a-f0-9-]+\.(json|text|srt|vtt)$'
            if not re.match(valid_pattern, filename):
                raise HTTPException(status_code=400, detail="Invalid filename format")
            
            # Build the file path
            file_path = os.path.join(output_path, "transcriptions", filename)
            
            # Check if file exists
            if not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail="Transcription file not found")
            
            # Determine content type based on file extension
            file_ext = filename.split(".")[-1].lower()
            content_type_map = {
                "json": "application/json",
                "text": "text/plain",
                "srt": "application/x-subrip",
                "vtt": "text/vtt"
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
            logger.exception(f"API transcription download error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @api.get("/api/v1/audio/translation/download/{filename}", tags=["Audio Transcription"])
    async def api_download_translation(filename: str):
        """
        Download a translation file
        
        Parameters:
        - filename: The name of the translation file to download
        
        Returns:
        - The translation file
        """
        try:
            # Validate filename format (prevent path traversal)
            valid_pattern = r'^translation_[a-f0-9-]+\.(json|text|srt|vtt)$'
            if not re.match(valid_pattern, filename):
                raise HTTPException(status_code=400, detail="Invalid filename format")
            
            # Build the file path
            file_path = os.path.join(output_path, "translations", filename)
            
            # Check if file exists
            if not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail="Translation file not found")
            
            # Determine content type based on file extension
            file_ext = filename.split(".")[-1].lower()
            content_type_map = {
                "json": "application/json",
                "text": "text/plain",
                "srt": "application/x-subrip",
                "vtt": "text/vtt"
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
            logger.exception(f"API translation download error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @api.get("/api/v1/audio/transcription/models", tags=["Audio Transcription"])
    async def api_get_transcription_models():
        """
        Get available transcription models
        
        Returns information about available models for audio transcription.
        """
        models = [
            {
                "id": "whisper-1",
                "name": "Whisper v1",
                "description": "General-purpose speech recognition model supporting transcription in multiple languages",
                "supported_languages": [
                    {"code": "en", "name": "English"},
                    {"code": "zh", "name": "Chinese"},
                    {"code": "de", "name": "German"},
                    {"code": "es", "name": "Spanish"},
                    {"code": "ru", "name": "Russian"},
                    {"code": "ko", "name": "Korean"},
                    {"code": "fr", "name": "French"},
                    {"code": "ja", "name": "Japanese"},
                    {"code": "pt", "name": "Portuguese"},
                    {"code": "tr", "name": "Turkish"},
                    {"code": "pl", "name": "Polish"},
                    {"code": "ca", "name": "Catalan"},
                    {"code": "nl", "name": "Dutch"},
                    {"code": "ar", "name": "Arabic"},
                    {"code": "sv", "name": "Swedish"},
                    {"code": "it", "name": "Italian"},
                    {"code": "id", "name": "Indonesian"},
                    {"code": "hi", "name": "Hindi"},
                    {"code": "fi", "name": "Finnish"},
                    {"code": "vi", "name": "Vietnamese"}
                ],
                "supported_formats": ["json", "text", "srt", "vtt", "verbose_json"]
            }
        ]
        return {"models": models}

    @api.get("/api/v1/audio/transcription/formats", tags=["Audio Transcription"])
    async def api_get_transcription_formats():
        """
        Get available transcription formats
        
        Returns information about supported output formats for audio transcription.
        """
        formats = [
            {
                "id": "json",
                "name": "JSON",
                "description": "Simple JSON containing the transcription text (default)",
                "mime_type": "application/json",
                "extension": ".json"
            },
            {
                "id": "text",
                "name": "Text",
                "description": "Plain text transcription",
                "mime_type": "text/plain",
                "extension": ".text"
            },
            {
                "id": "srt",
                "name": "SRT",
                "description": "SubRip subtitle format with timestamps",
                "mime_type": "application/x-subrip",
                "extension": ".srt"
            },
            {
                "id": "vtt",
                "name": "VTT",
                "description": "WebVTT subtitle format with timestamps",
                "mime_type": "text/vtt",
                "extension": ".vtt"
            },
            {
                "id": "verbose_json",
                "name": "Verbose JSON",
                "description": "Detailed JSON with additional information including word-level timestamps",
                "mime_type": "application/json",
                "extension": ".json"
            }
        ]
        return {"formats": formats}

# Example test function
def test():
    """
    Test function for the transcribe module.
    
    This function tests the core functionality of the transcribe module
    without requiring external dependencies or actual transcription.
    """
    print("Running transcribe layout test...")
    
    # Test the audio file duration detection function
    import numpy as np
    from scipy.io import wavfile
    import os
    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test audio file
        sample_rate = 16000
        duration = 2  # seconds
        audio_data = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration)))
        
        # Save as WAV
        test_file = os.path.join(temp_dir, "test_audio.wav")
        wavfile.write(test_file, sample_rate, audio_data.astype(np.float32))
        
        # Test duration detection
        detected_duration = get_audio_duration(test_file)
        expected_duration = duration
        
        # Allow for small floating point differences
        duration_diff = abs(detected_duration - expected_duration)
        if duration_diff > 0.1:  # Allow 100ms difference for processing overhead
            raise ValueError(f"Duration detection failed. Expected ~{expected_duration}s, got {detected_duration}s")
        
        print(f"Duration detection passed: {detected_duration}s")
        
        # Test audio loading
        audio, sr = load_audio_file(test_file)
        if sr != sample_rate:
            raise ValueError(f"Sample rate mismatch. Expected {sample_rate}, got {sr}")
        
        if len(audio) != int(sample_rate * duration):
            raise ValueError(f"Audio length mismatch. Expected {int(sample_rate * duration)}, got {len(audio)}")
        
        print(f"Audio loading passed: {len(audio)} samples at {sr}Hz")
        
        # Test audio preview generation
        preview_image = get_audio_preview(test_file)
        if preview_image is None or not isinstance(preview_image, np.ndarray):
            raise ValueError("Audio preview generation failed")
        
        print(f"Audio preview generation passed: {preview_image.shape}")
    
    print("Transcribe layout test completed successfully!")
    return True 