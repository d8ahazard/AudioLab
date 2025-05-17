import gc
import json
import logging
import os
import time
import shutil

import gradio as gr
from huggingface_hub import snapshot_download
import whisperx
import tempfile
import base64

from handlers.args import ArgHandler
from handlers.config import output_path, model_path
import base64

logger = logging.getLogger(__name__)
arg_handler = ArgHandler()

# Global variables for inter-tab communication
SEND_TO_PROCESS_BUTTON = None
OUTPUT_TRANSCRIPTION = None
OUTPUT_AUDIO = None

def fetch_model(tgt_model):
    """
    Download the model if needed and return the path to the local model directory.
    
    For HuggingFace models, this downloads the model to the local cache.
    For OpenAI Whisper models, it returns the model name as is since they're handled differently.
    """
    # OpenAI Whisper models are handled differently - just return the model name
    if tgt_model in ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "turbo"]:
        return tgt_model
    
    # For HuggingFace models, download and return the path
    model_name = tgt_model.split("/")[-1]
    # Create the directory where models will be stored
    model_dir = os.path.join(model_path, "whisperx", model_name)
    is_model_dir_empty = os.path.exists(model_dir) and not os.listdir(model_dir)
    
    # Download the model if it doesn't exist or the directory is empty
    if not os.path.exists(model_dir) or is_model_dir_empty:
        logger.info(f"Downloading model {model_name} from {tgt_model} to {model_dir}")
        snapshot_download(tgt_model, local_dir=model_dir)
    
    # Return the local model directory path
    return model_dir

def process_transcription(
    audio_files,
    engine="whisperx",
    language="auto",
    align_output=True,
    assign_speakers=True,
    min_speakers=None,
    max_speakers=None,
    batch_size=16,
    compute_type="float16",
    return_char_alignments=False,
    progress=gr.Progress(track_tqdm=True)
):
    """Main transcription function that selects the appropriate engine."""
    try:
        # Select the appropriate transcription engine
        if engine == "whisperx":
            return process_transcription_whisperx(
                audio_files,
                language=language,
                align_output=align_output,
                assign_speakers=assign_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                batch_size=batch_size,
                compute_type=compute_type,
                return_char_alignments=return_char_alignments,
                progress=progress
            )
        elif engine == "whisper":
            return process_transcription_whisper(
                audio_files,
                language=language,
                assign_speakers=assign_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                compute_type=compute_type,
                word_timestamps=True, # Original whisper supports word timestamps
                progress=progress
            )
        else:
            raise ValueError(f"Unknown transcription engine: {engine}")
    except Exception as e:
        logger.exception(f"Transcription error: {e}")
        return f"Error: {str(e)}", []


def process_transcription_whisperx(
    audio_files,
    language="auto",
    align_output=True,
    assign_speakers=True,
    min_speakers=None,
    max_speakers=None,
    batch_size=16,
    compute_type="float16",
    return_char_alignments=False,
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
        # Get the local model directory path
        model_dir = fetch_model("Systran/faster-whisper-large-v3")
        logger.info(f"Model directory: {model_dir}")
        
        # Load the model directly from the local path
        model = whisperx.load_model(
            model_dir,
            device, 
            compute_type=compute_type
        )
        
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
                    # Note: WhisperX automatically generates word-level timestamps during alignment
                    # The return_char_alignments parameter can be used for more detailed character-level timing
                    result = whisperx.align(
                        result["segments"], 
                        model_a, 
                        metadata, 
                        audio, 
                        device,
                        return_char_alignments=return_char_alignments
                    )
                    current_step += 1
                    # Clean up alignment model to save memory
                    del model_a
                    gc.collect()
                
                # 3. Assign speaker labels if requested
                if assign_speakers:
                    progress(current_step/total_steps, f"Assigning speakers for {file_name}")
                    dia_model_path = fetch_model("fatymatariq/speaker-diarization-3.1")
                    dia_model_path = os.path.join(dia_model_path, "config.yaml")
                    # Fix: Use the correct import path for DiarizationPipeline
                    diarize_model = whisperx.diarize.DiarizationPipeline(model_name=dia_model_path, device=device)
                    
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
                
                # Generate output files
                output_json, output_txt, output_lrc, output_srt, output_vtt = generate_output_files(
                    result, output_folder, base_name, assign_speakers
                )
                
                # Store results and output files
                results.append(result)
                output_files.extend([output_json, output_txt, output_lrc, output_srt, output_vtt])
                
            except Exception as e:
                logger.exception(f"Error processing file {audio_file}: {e}")
                
        # Clean up main model
        del model
        gc.collect()
            
        # Return summary and output files
        summary = f"Transcribed {len(results)} files to {output_folder}"
        return summary, output_files
        
    except Exception as e:
        logger.exception(f"WhisperX transcription error: {e}")
        return f"Error: {str(e)}", []


def process_transcription_whisper(
    audio_files,
    language="auto",
    assign_speakers=True,
    min_speakers=None,
    max_speakers=None,
    compute_type="float16",
    word_timestamps=True,
    progress=gr.Progress(track_tqdm=True)
):
    """Transcribe audio files using OpenAI's Whisper with word-level timestamps and optional speaker diarization."""
    try:
        # Import whisper
        import whisper
        import torch
        
        # Initialize counters for progress tracking
        total_steps = len(audio_files) * (1 + (1 if assign_speakers else 0))
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
            
        device = "cuda" if torch.cuda.is_available() else "cpu"
        fp16 = True if compute_type == "float16" and device == "cuda" else False
        
        results = []
        output_files = []
        
        # Create output directory
        timestamp = int(time.time())
        output_folder = os.path.join(output_path, "transcriptions", f"transcribe_{timestamp}")
        os.makedirs(output_folder, exist_ok=True)
        
        progress(0, "Loading transcription model...")
        # Use the model size based on compute_type
        model_size = "large-v3"  # Default to large-v3 for best results
        
        # Load the model
        model = whisper.load_model(model_size, device=device)
        logger.info(f"Loaded whisper {model_size} model on {device}")
        
        # Process each audio file
        for audio_idx, audio_file in enumerate(audio_files):
            try:
                file_name = os.path.basename(audio_file)
                base_name, _ = os.path.splitext(file_name)
                
                # 1. Transcribe with whisper
                progress(current_step/total_steps, f"Transcribing {file_name}")
                
                # Set up transcription options
                whisper_options = {
                    "verbose": False,
                    "word_timestamps": word_timestamps,
                    "condition_on_previous_text": False,  # More accurate for shorter clips
                }
                
                # Set language if specified
                if language != "auto":
                    whisper_options["language"] = language
                
                # Transcribe
                result = model.transcribe(audio_file, **whisper_options)
                current_step += 1
                
                # 2. Assign speaker labels if requested
                if assign_speakers:
                    progress(current_step/total_steps, f"Assigning speakers for {file_name}")
                    
                    # Load audio
                    audio = whisper.load_audio(audio_file)
                    
                    # Import pyannote only when needed
                    try:
                        from pyannote.audio import Pipeline
                    except ImportError:
                        raise ImportError("Speaker diarization requires pyannote.audio. Install it with: pip install pyannote.audio")
                    
                    dia_model_path = fetch_model("fatymatariq/speaker-diarization-3.1")
                    dia_model_path = os.path.join(dia_model_path, "config.yaml")
                    
                    # Initialize diarization pipeline
                    pipeline = Pipeline.from_pretrained(dia_model_path)
                    
                    # Process diarization
                    diarization_options = {}
                    if min_speakers is not None:
                        diarization_options["min_speakers"] = min_speakers
                    if max_speakers is not None:
                        diarization_options["max_speakers"] = max_speakers
                    
                    diarization = pipeline(audio_file, **diarization_options)
                    
                    # Map diarization results to segments
                    # This is a simplification - detailed speaker assignment would require more complex logic
                    for segment in result["segments"]:
                        segment_start = segment["start"]
                        segment_end = segment["end"]
                        segment_mid = (segment_start + segment_end) / 2
                        
                        # Find the speaker at the middle of the segment
                        speaker = None
                        for turn, _, speaker_id in diarization.itertracks(yield_label=True):
                            if turn.start <= segment_mid <= turn.end:
                                speaker = f"SPEAKER_{speaker_id}"
                                break
                        
                        if speaker:
                            segment["speaker"] = speaker
                        
                        # If word timestamps are available, assign speakers to words
                        if "words" in segment:
                            for word in segment["words"]:
                                word_mid = (word["start"] + word["end"]) / 2
                                word_speaker = None
                                
                                for turn, _, speaker_id in diarization.itertracks(yield_label=True):
                                    if turn.start <= word_mid <= turn.end:
                                        word_speaker = f"SPEAKER_{speaker_id}"
                                        break
                                
                                if word_speaker:
                                    word["speaker"] = word_speaker
                    
                    current_step += 1
                
                # Generate output files
                output_json, output_txt, output_lrc, output_srt, output_vtt = generate_output_files(
                    result, output_folder, base_name, assign_speakers
                )
                
                # Store results and output files
                results.append(result)
                output_files.extend([output_json, output_txt, output_lrc, output_srt, output_vtt])
                
            except Exception as e:
                logger.exception(f"Error processing file {audio_file}: {e}")
                
        # Clean up main model
        del model
        gc.collect()
            
        # Return summary and output files
        summary = f"Transcribed {len(results)} files to {output_folder}"
        return summary, output_files
        
    except Exception as e:
        logger.exception(f"Whisper transcription error: {e}")
        return f"Error: {str(e)}", []


def generate_output_files(result, output_folder, base_name, assign_speakers=False):
    """Generate output files in various formats from the transcription result."""
    # Save results to files
    output_json = os.path.join(output_folder, f"{base_name}.json")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
        
    # Generate text file with transcript
    output_txt = os.path.join(output_folder, f"{base_name}.txt")
    with open(output_txt, 'w', encoding='utf-8') as f:
        # If speaker diarization was performed, include speaker information
        if assign_speakers and "segments" in result and len(result["segments"]) > 0 and "speaker" in result["segments"][0]:
            for segment in result["segments"]:
                speaker = segment.get("speaker", "UNKNOWN")
                text = segment["text"]
                start = segment["start"]
                end = segment["end"]
                f.write(f"[{speaker}] ({start:.2f}s - {end:.2f}s): {text}\n")
        else:
            for segment in result.get("segments", []):
                text = segment["text"]
                start = segment["start"]
                end = segment["end"]
                f.write(f"({start:.2f}s - {end:.2f}s): {text}\n")

    # Generate LRC format file
    output_lrc = os.path.join(output_folder, f"{base_name}.lrc")
    with open(output_lrc, 'w', encoding='utf-8') as f:
        for segment in result.get("segments", []):
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
    
    # Generate SRT (SubRip) format file
    output_srt = os.path.join(output_folder, f"{base_name}.srt")
    with open(output_srt, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(result.get("segments", [])):
            # SubRip index (starts at 1)
            f.write(f"{i+1}\n")
            
            # Format timestamps as HH:MM:SS,mmm --> HH:MM:SS,mmm
            start_time = segment["start"]
            end_time = segment["end"]
            
            start_hrs = int(start_time // 3600)
            start_mins = int((start_time % 3600) // 60)
            start_secs = int(start_time % 60)
            start_ms = int((start_time % 1) * 1000)
            
            end_hrs = int(end_time // 3600)
            end_mins = int((end_time % 3600) // 60)
            end_secs = int(end_time % 60)
            end_ms = int((end_time % 1) * 1000)
            
            timestamp_line = f"{start_hrs:02d}:{start_mins:02d}:{start_secs:02d},{start_ms:03d} --> "
            timestamp_line += f"{end_hrs:02d}:{end_mins:02d}:{end_secs:02d},{end_ms:03d}"
            f.write(f"{timestamp_line}\n")
            
            # Text content, with speaker if available
            text = segment["text"].strip()
            if assign_speakers and "speaker" in segment:
                text = f"[{segment['speaker']}] {text}"
            f.write(f"{text}\n\n")
    
    # Generate VTT (WebVTT) format file
    output_vtt = os.path.join(output_folder, f"{base_name}.vtt")
    with open(output_vtt, 'w', encoding='utf-8') as f:
        # Write VTT header
        f.write("WEBVTT\n\n")
        
        for i, segment in enumerate(result.get("segments", [])):
            # Format timestamps as HH:MM:SS.mmm --> HH:MM:SS.mmm (note: VTT uses . instead of , for milliseconds)
            start_time = segment["start"]
            end_time = segment["end"]
            
            start_hrs = int(start_time // 3600)
            start_mins = int((start_time % 3600) // 60)
            start_secs = int(start_time % 60)
            start_ms = int((start_time % 1) * 1000)
            
            end_hrs = int(end_time // 3600)
            end_mins = int((end_time % 3600) // 60)
            end_secs = int(end_time % 60)
            end_ms = int((end_time % 1) * 1000)
            
            timestamp_line = f"{start_hrs:02d}:{start_mins:02d}:{start_secs:02d}.{start_ms:03d} --> "
            timestamp_line += f"{end_hrs:02d}:{end_mins:02d}:{end_secs:02d}.{end_ms:03d}"
            
            # Optionally add cue identifier
            cue_id = f"cue-{i+1}"
            f.write(f"{cue_id}\n")
            f.write(f"{timestamp_line}\n")
            
            # Text content, with speaker if available
            text = segment["text"].strip()
            if assign_speakers and "speaker" in segment:
                text = f"[{segment['speaker']}] {text}"
            f.write(f"{text}\n\n")
            
    return output_json, output_txt, output_lrc, output_srt, output_vtt


def render(arg_handler: ArgHandler):
    global SEND_TO_PROCESS_BUTTON, OUTPUT_TRANSCRIPTION, OUTPUT_AUDIO
    
    with gr.Blocks() as transcribe:
        gr.Markdown("# 🎙️ Audio Transcription")
        gr.Markdown("Transcribe audio files with high accuracy using WhisperX or OpenAI Whisper. Supports multi-speaker diarization and precision timestamps.")
        
        with gr.Row():
            # Left Column - Settings
            with gr.Column():
                gr.Markdown("### 🔧 Settings")
                
                engine = gr.Dropdown(
                    label="Transcription Engine",
                    choices=[
                        ("WhisperX (Recommended, faster)", "whisperx"),
                        ("OpenAI Whisper (Original)", "whisper")
                    ],
                    value="whisperx",
                    elem_classes="hintitem",
                    elem_id="transcribe_engine",
                    key="transcribe_engine"
                )
                
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
                    return_char_alignments = gr.Checkbox(
                        label="Character Alignments",
                        value=False,
                        elem_classes="hintitem",
                        elem_id="transcribe_return_char_alignments",
                        key="transcribe_return_char_alignments"
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
                
                # Show/hide options based on selected engine
                def update_engine_options(engine_choice):
                    # WhisperX specific options
                    align_visibility = engine_choice == "whisperx"
                    char_align_visibility = engine_choice == "whisperx"
                    batch_visibility = engine_choice == "whisperx"
                    
                    return [
                        gr.update(visible=align_visibility),
                        gr.update(visible=char_align_visibility),
                        gr.update(visible=batch_visibility)
                    ]
                
                engine.change(
                    fn=update_engine_options,
                    inputs=[engine],
                    outputs=[align_output, return_char_alignments, batch_size]
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
                
                # Add file selector dropdown
                file_selector = gr.Dropdown(
                    label="Select Output File",
                    choices=[],
                    visible=False,
                    elem_classes="hintitem",
                    elem_id="transcribe_file_selector",
                    key="transcribe_file_selector"
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
                try:
                    with open(selected_file, "r", encoding="utf-8") as f:
                        content = f.read()
                    return content, gr.update(visible=False)
                except Exception as e:
                    return f"Error loading text file: {str(e)}", gr.update(visible=False)
                
            # If it's a JSON file, format its content
            elif selected_file.endswith(".json"):
                try:
                    with open(selected_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    # Format the JSON content for display
                    content = "# Transcription Results\n\n"
                    
                    # Handle missing or empty segments
                    segments = data.get("segments", [])
                    if not segments and isinstance(data, list):
                        # Some JSON files might have segments as the root array
                        segments = data
                    
                    if not segments:
                        content += "No segments found in the transcription.\n"
                    else:
                        for segment in segments:
                            speaker = segment.get("speaker", "")
                            text = segment.get("text", "")
                            start = segment.get("start", 0)
                            end = segment.get("end", 0)
                            
                            if speaker:
                                content += f"[{speaker}] ({start:.2f}s - {end:.2f}s): {text}\n"
                            else:
                                content += f"({start:.2f}s - {end:.2f}s): {text}\n"
                    
                    return content, gr.update(visible=False)
                except json.JSONDecodeError:
                    return "Error: Invalid JSON format in the file.", gr.update(visible=False)
                except Exception as e:
                    return f"Error loading JSON file: {str(e)}", gr.update(visible=False)
                
            # If it's an SRT or VTT file, show its content
            elif selected_file.endswith((".srt", ".vtt")):
                try:
                    with open(selected_file, "r", encoding="utf-8") as f:
                        content = f.read()
                    return content, gr.update(visible=False)
                except Exception as e:
                    return f"Error loading subtitle file: {str(e)}", gr.update(visible=False)
                
            # If it's an LRC file, format it nicely
            elif selected_file.endswith(".lrc"):
                try:
                    with open(selected_file, "r", encoding="utf-8") as f:
                        content = f.read()
                    return content, gr.update(visible=False)
                except Exception as e:
                    return f"Error loading LRC file: {str(e)}", gr.update(visible=False)
                
            # If it's an audio file, show the audio player
            elif selected_file.endswith((".wav", ".mp3", ".flac")):
                if os.path.exists(selected_file):
                    return "", gr.update(visible=True, value=selected_file)
                else:
                    return "Error: Audio file not found.", gr.update(visible=False)
                
            return f"Unsupported file type: {os.path.basename(selected_file)}", gr.update(visible=False)

        # Event handler for the transcribe button
        transcribe_button.click(
            fn=process_transcription,
            inputs=[
                input_audio,
                engine,
                language,
                align_output,
                assign_speakers,
                min_speakers,
                max_speakers,
                batch_size,
                compute_type,
                return_char_alignments
            ],
            outputs=[status_display, OUTPUT_TRANSCRIPTION]
        )


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
        "engine": "Select the transcription engine to use. WhisperX is faster and supports more features, while original Whisper might be better for some languages.",
        "language": "Select the language of the audio for better transcription accuracy, or choose 'auto' for automatic detection.",
        "align_output": "Enable to align the transcription with the audio for precise timestamps (WhisperX only).",
        "assign_speakers": "Enable to detect and assign different speakers in the audio.",
        "return_char_alignments": "Enable to generate character-level timestamps (more detailed but increases processing time, WhisperX only).",
        "min_speakers": "Minimum number of speakers to detect (leave empty for automatic detection).",
        "max_speakers": "Maximum number of speakers to detect (leave empty for automatic detection).",
        "batch_size": "Batch size for transcription processing. Higher values use more memory but can be faster (WhisperX only).",
        "compute_type": "Precision level for computation. Lower precision uses less memory but may be less accurate.",
        "input_audio": "Upload one or more audio files to transcribe.",
        "output_audio": "Preview of the selected audio file.",
        "button": "Start the transcription process.",
        "send_to_process": "Send the selected audio file to the Process tab for further processing.",
        "status": "Current status of the transcription process.",
        "output_files": "Transcription files generated by the process.",
        "file_selector": "Select a transcription file to preview.",
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
        """Request model for audio transcription"""
        audio_file: str = Field(..., description="Base64 encoded audio file to transcribe")
        engine: str = Field(
            "whisperx", description="The transcription engine to use: 'whisperx' or 'whisper'"
        )
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
        align_output: bool = Field(
            True, description="Whether to align the transcription with the audio for precise timestamps"
        )
        assign_speakers: bool = Field(
            False, description="Whether to detect and assign different speakers in the audio"
        )
        min_speakers: Optional[int] = Field(
            None, description="Minimum number of speakers to detect (optional)"
        )
        max_speakers: Optional[int] = Field(
            None, description="Maximum number of speakers to detect (optional)"
        )
        return_char_alignments: bool = Field(
            False, description="Whether to generate character-level timestamps"
        )
        timestamp_granularities: Optional[List[str]] = Field(
            None, description="Timestamp granularity levels"
        )
        
        @validator('engine')
        def validate_engine(cls, v):
            valid_engines = ["whisperx", "whisper"]
            if v not in valid_engines:
                raise ValueError(f"Engine must be one of: {', '.join(valid_engines)}")
            return v
        
        @validator('model')
        def validate_model(cls, v):
            valid_models = ["whisper-1", "whisper-large-v3"]
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
        """Request model for audio translation"""
        audio_file: str = Field(..., description="Base64 encoded audio file to translate")
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
        request: TranscriptionOptions,
        background_tasks: BackgroundTasks = None
    ):
        """
        Transcribe audio to text
        
        This endpoint transcribes the uploaded audio file to text using the selected transcription engine.
        """
        try:
            # Check file size limit (25MB)
            audio_data = base64.b64decode(request.audio_file)
            file_size = len(audio_data)
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
            
            # Generate a unique ID for this transcription
            transcription_id = str(uuid.uuid4())
            
            # Process using the selected transcription engine
            try:
                # Use the common process_transcription function
                summary, output_files = process_transcription(
                    audio_files=[temp_audio_path],
                    engine=request.engine,
                    language=request.language if request.language else "auto",
                    align_output=request.align_output,
                    assign_speakers=request.assign_speakers,
                    min_speakers=str(request.min_speakers) if request.min_speakers is not None else None,
                    max_speakers=str(request.max_speakers) if request.max_speakers is not None else None,
                    batch_size=16,
                    compute_type="float16",
                    return_char_alignments=request.return_char_alignments,
                    progress=None
                )
                
                if not output_files:
                    raise HTTPException(status_code=500, detail="Failed to transcribe audio: No output files generated")
                
                # Get the output format
                output_format = request.response_format.lower()
                
                # Get corresponding output file
                output_file_path = None
                for file_path in output_files:
                    if file_path.endswith(f".{output_format}"):
                        output_file_path = file_path
                        break
                    elif output_format == "text" and file_path.endswith(".txt"):
                        output_file_path = file_path
                        break
                    elif output_format == "verbose_json" and file_path.endswith(".json"):
                        output_file_path = file_path
                        break
                
                if not output_file_path:
                    raise HTTPException(status_code=500, detail=f"Failed to find output file with format {output_format}")
                
                # Read the output file content
                with open(output_file_path, "r", encoding="utf-8") as f:
                    if output_format in ["json", "verbose_json"]:
                        response_content = json.load(f)
                    else:
                        response_content = f.read()
                
                # Get output directory path
                output_dir = os.path.dirname(output_file_path)
                
                # Create download URLs for all formats
                download_urls = {}
                for file_path in output_files:
                    file_ext = os.path.splitext(file_path)[1][1:]  # Get extension without the dot
                    download_urls[file_ext] = f"/api/v1/audio/transcription/download/{transcription_id}/{file_ext}"
                
                # Create response data
                response_data = {
                    "success": True,
                    "transcription_id": transcription_id,
                    "engine": request.engine,
                    "language": request.language,
                    "output_formats": download_urls,
                    "metadata": {
                        "original_filename": temp_audio_filename,
                        "engine": request.engine,
                        "model": request.model,
                        "align_output": request.align_output,
                        "assign_speakers": request.assign_speakers,
                        "return_char_alignments": request.return_char_alignments,
                        "timestamp": int(time.time())
                    }
                }
                
                # Add the transcription content to the response if applicable
                if response_content:
                    response_data["transcription"] = response_content
                
                # Clean up temporary audio file
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
                
                # Store the output directory for download
                # Create a symlink in a standard location
                api_output_dir = os.path.join(output_path, "transcriptions", f"api_{transcription_id}")
                if os.path.exists(api_output_dir):
                    shutil.rmtree(api_output_dir)
                os.symlink(output_dir, api_output_dir)
                
                # Schedule cleanup after 24 hours
                if background_tasks:
                    background_tasks.add_task(
                        lambda p: shutil.rmtree(p) if os.path.exists(p) else None,
                        api_output_dir,
                        delay=86400  # 24 hours
                    )
                
                return JSONResponse(content=jsonable_encoder(response_data))
                
            except Exception as e:
                logger.exception(f"Transcription error: {e}")
                import traceback
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")
            
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
        request: TranslationOptions,
        background_tasks: BackgroundTasks = None
    ):
        """
        Translate audio to English text
        
        This endpoint translates the uploaded audio file to English text using the Whisper model.
        """
        try:
            # Check file size limit (25MB)
            audio_data = base64.b64decode(request.audio_file)
            file_size = len(audio_data)
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
                model=request.model,
                prompt=request.prompt,
                response_format=request.response_format,
                temperature=request.temperature,
                timestamp_granularities=request.timestamp_granularities
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
            if request.response_format == "json":
                output_filename = f"translation_{translation_id}.json"
                output_filepath = os.path.join(translation_dir, output_filename)
                with open(output_filepath, "w", encoding="utf-8") as f:
                    json.dump(translation_data, f, indent=2, ensure_ascii=False)
            
            elif request.response_format == "verbose_json":
                output_filename = f"translation_{translation_id}.json"
                output_filepath = os.path.join(translation_dir, output_filename)
                with open(output_filepath, "w", encoding="utf-8") as f:
                    json.dump(translation_data, f, indent=2, ensure_ascii=False)
            
            elif request.response_format in ["text", "srt", "vtt"]:
                output_filename = f"translation_{translation_id}.{request.response_format}"
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
                "original_filename": temp_audio_filename,
                "model": request.model,
                "prompt": request.prompt,
                "response_format": request.response_format,
                "temperature": request.temperature,
                "timestamp_granularities": request.timestamp_granularities,
                "output_file": {
                    "filename": output_filename,
                    "format": request.response_format,
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
            if request.response_format == "text":
                translation_content = translation_data
            elif request.response_format in ["json", "verbose_json"]:
                translation_content = translation_data
            
            response_data = {
                "success": True,
                "translation_id": translation_id,
                "output_file": {
                    "filename": output_filename,
                    "format": request.response_format,
                    "download_url": download_url,
                    "file_size_bytes": file_size
                },
                "metadata": {
                    "original_filename": temp_audio_filename,
                    "model": request.model,
                    "prompt": request.prompt,
                    "response_format": request.response_format,
                    "temperature": request.temperature,
                    "timestamp_granularities": request.timestamp_granularities,
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

    @api.get("/api/v1/audio/transcription/download/{transcription_id}/{format_name}", tags=["Audio Transcription"])
    async def api_download_transcription(transcription_id: str, format_name: str):
        """
        Download a transcription file
        
        Parameters:
        - transcription_id: The ID of the transcription
        - format_name: The format of the transcription file to download (json, text, srt, vtt, lrc)
        
        Returns:
        - The transcription file
        """
        try:
            # Validate transcription ID format
            valid_id_pattern = r'^[a-f0-9-]+$'
            if not re.match(valid_id_pattern, transcription_id):
                raise HTTPException(status_code=400, detail="Invalid transcription ID format")
            
            # Validate format name
            valid_formats = ["json", "text", "srt", "vtt", "lrc"]
            if format_name not in valid_formats:
                raise HTTPException(status_code=400, detail=f"Invalid format name. Must be one of: {', '.join(valid_formats)}")
            
            # Build the file path
            file_path = os.path.join(output_path, "transcriptions", f"api_{transcription_id}", f"transcript.{format_name}")
            
            # Check if file exists
            if not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail=f"Transcription file in {format_name} format not found")
            
            # Determine content type based on format
            content_type_map = {
                "json": "application/json",
                "text": "text/plain",
                "srt": "application/x-subrip",
                "vtt": "text/vtt",
                "lrc": "text/plain"
            }
            
            content_type = content_type_map.get(format_name, "application/octet-stream")
            
            # Return the file
            return FileResponse(
                path=file_path,
                filename=f"transcript.{format_name}",
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
            # WhisperX models
            {
                "id": "whisperx-large-v3",
                "name": "WhisperX (large-v3)",
                "engine": "whisperx",
                "description": "Fast automatic speech recognition with word-level timestamps and speaker diarization",
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
                "features": ["word-level-timestamps", "speaker-diarization", "character-level-timestamps"],
                "supported_formats": ["json", "text", "srt", "vtt", "lrc"]
            },
            
            # OpenAI Whisper models
            {
                "id": "whisper-large-v3",
                "name": "OpenAI Whisper (large-v3)",
                "engine": "whisper",
                "description": "Original OpenAI Whisper model with word-level timestamps and speaker diarization",
                "supported_languages": ["en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi"],
                "features": ["word-level-timestamps", "speaker-diarization"],
                "supported_formats": ["json", "text", "srt", "vtt", "lrc", "verbose_json"]
            },
            {
                "id": "whisper-large-v2",
                "name": "OpenAI Whisper (large-v2)",
                "engine": "whisper",
                "description": "Original OpenAI Whisper model with word-level timestamps and speaker diarization",
                "supported_languages": ["en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi"],
                "features": ["word-level-timestamps", "speaker-diarization"],
                "supported_formats": ["json", "text", "srt", "vtt", "lrc", "verbose_json"]
            },
            {
                "id": "whisper-large",
                "name": "OpenAI Whisper (large)",
                "engine": "whisper",
                "description": "Original OpenAI Whisper model with word-level timestamps and speaker diarization",
                "supported_languages": ["en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi"],
                "features": ["word-level-timestamps", "speaker-diarization"],
                "supported_formats": ["json", "text", "srt", "vtt", "lrc", "verbose_json"]
            },
            {
                "id": "whisper-medium",
                "name": "OpenAI Whisper (medium)",
                "engine": "whisper",
                "description": "Medium-sized OpenAI Whisper model with word-level timestamps",
                "supported_languages": ["en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi"],
                "features": ["word-level-timestamps", "speaker-diarization"],
                "supported_formats": ["json", "text", "srt", "vtt", "lrc", "verbose_json"]
            },
            {
                "id": "whisper-small",
                "name": "OpenAI Whisper (small)",
                "engine": "whisper",
                "description": "Small-sized OpenAI Whisper model",
                "supported_languages": ["en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi"],
                "features": ["word-level-timestamps", "speaker-diarization"],
                "supported_formats": ["json", "text", "srt", "vtt", "lrc", "verbose_json"]
            },
            {
                "id": "whisper-base",
                "name": "OpenAI Whisper (base)",
                "engine": "whisper",
                "description": "Base-sized OpenAI Whisper model",
                "supported_languages": ["en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi"],
                "features": ["word-level-timestamps", "speaker-diarization"],
                "supported_formats": ["json", "text", "srt", "vtt", "lrc", "verbose_json"]
            },
            {
                "id": "whisper-tiny",
                "name": "OpenAI Whisper (tiny)",
                "engine": "whisper",
                "description": "Tiny-sized OpenAI Whisper model, fastest but less accurate",
                "supported_languages": ["en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi"],
                "features": ["word-level-timestamps", "speaker-diarization"],
                "supported_formats": ["json", "text", "srt", "vtt", "lrc", "verbose_json"]
            },
            {
                "id": "whisper-turbo",
                "name": "OpenAI Whisper (turbo)",
                "engine": "whisper",
                "description": "Optimized OpenAI Whisper model for faster transcription",
                "supported_languages": ["en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi"],
                "features": ["word-level-timestamps", "speaker-diarization"],
                "supported_formats": ["json", "text", "srt", "vtt", "lrc", "verbose_json"]
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
                "description": "Simple JSON containing the transcription text with segments and timestamps (default)",
                "mime_type": "application/json",
                "extension": ".json"
            },
            {
                "id": "text",
                "name": "Text",
                "description": "Plain text transcription with timestamps",
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
                "id": "lrc",
                "name": "LRC",
                "description": "Lyrics format with timestamps, compatible with media players",
                "mime_type": "text/plain",
                "extension": ".lrc"
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