import os
import logging

import gradio as gr
from huggingface_hub import hf_hub_download

from handlers.args import ArgHandler
from handlers.tts import TTSHandler
from handlers.config import output_path, model_path, app_path

arg_handler = ArgHandler()
SEND_TO_PROCESS_BUTTON: gr.Button = None
OUTPUT_AUDIO: gr.Audio = None
logger = logging.getLogger(__name__)
zonos_model = None
dia_model = None  # Add DIA model global variable
speaker_sample_file = None

# Maps bracket tags to an 8-D emotion vector (sum=1).
# If the user picks "Normal" in the dropdown or we have None,
# we let the model's internal default remain in place (emotion=None).
EMOTION_MAP = {
    "Happiness": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "Happy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "Sadness": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "Sad": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "Disgust": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "Disgusted": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "Fear": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    "Scared": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    "Surprise": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "Anger": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    "Mad": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    "Other": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    "Neutral": None,
}

# Import Zonos functionality
from modules.zonos.conditioning import supported_language_codes

def download_model():
    """Download Zonos model files if they don't exist."""
    repo_id = "Zyphra/Zonos-v0.1-transformer"
    model_dir = os.path.join(model_path, "zonos")
    if not os.path.exists(model_dir):
        logger.info("Downloading Zonos model...")
        os.makedirs(model_dir, exist_ok=True)
        # Download config.json and model.pth
        try:
            _ = hf_hub_download(repo_id=repo_id, filename="config.json", local_dir=model_dir)
            _ = hf_hub_download(repo_id=repo_id, filename="model.safetensors", local_dir=model_dir)
        except Exception as e:
            logger.error(f"Error downloading Zonos model: {e}")
            raise e
    return model_dir

def download_speaker_model():
    """Download Zonos speaker embedding model files."""
    model_dir = os.path.join(model_path, "zonos")
    _ = hf_hub_download(repo_id="Zyphra/Zonos-v0.1-speaker-embedding",
                        filename="ResNet293_SimAM_ASP_base.pt", local_dir=model_dir)
    _ = hf_hub_download(repo_id="Zyphra/Zonos-v0.1-speaker-embedding",
                        filename="ResNet293_SimAM_ASP_base_LDA-128.pt", local_dir=model_dir)
    return model_dir

def download_dia_model():
    """Download DIA model files if they don't exist."""
    repo_id = "nari-labs/Dia-1.6B"
    model_dir = os.path.join(model_path, "diatts")
    os.makedirs(model_dir, exist_ok=True)
    
    try:
        logger.info("Downloading DIA model files...")
        # Download model files
        files_to_download = [
            "config.json",
            "dia-v0_1.pth"
        ]
        
        for filename in files_to_download:
            filepath = os.path.join(model_dir, filename)
            if not os.path.exists(filepath):
                _ = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=model_dir)
                logger.info(f"Downloaded {filename}")
        
        logger.info("DIA model files downloaded successfully")
    except Exception as e:
        logger.error(f"Error downloading DIA model: {e}")
        raise e
    
    return model_dir

def set_espeak_lib_path_win():
    """Set the path to espeak library on Windows."""
    if os.name == "nt":
        # Look in Program Files\espeak-ng for espeak-ng.dll
        installed_path = os.path.join(
            os.environ.get("ProgramFiles", "C:\\Program Files"),
            "eSpeak NG",
            "lib\\espeak-ng.dll"
        )
        # Check and set PHONEMIZER_ESPEAK_LIBRARY if it exists
        if os.path.exists(installed_path):
            logger.info(f"Found espeak-ng.dll at {installed_path}")
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = installed_path
        else:
            espeak_lib_path = os.path.join(app_path, "libs", "libespeak-ng.dll")
            logger.info(f"Using bundled espeak-ng.dll at {espeak_lib_path}")
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = espeak_lib_path

def _parse_text_and_emotions(full_text: str, default_emotion: str):
    """Parse text and extract emotion tags for Zonos with improved chunking."""
    import re

    current_emotion_str = None if default_emotion == "Normal" else default_emotion

    def get_emotion_vector(emo_str):
        if not emo_str or emo_str not in EMOTION_MAP:
            return None
        emo_val = EMOTION_MAP[emo_str]
        default = [0.3077, 0.0256, 0.0256, 0.0256, 0.0256, 0.0256, 0.2564, 0.3077]
        if emo_val is None:  # "Neutral"
            return None
        return [min(1.0, emo_val[i] + default[i]) for i in range(8)]

    # Store chunks with emotion and special markers
    chunks_with_metadata = []
    lines = full_text.splitlines()

    def flush_chunk(buffer, emotion):
        text = buffer.strip()
        if text:
            chunks_with_metadata.append({"text": text, "emotion": emotion, "newline_after": False})

    chunk_buffer = ""
    prev_line_empty = False
    
    for i, line in enumerate(lines):
        # Handle empty lines as pause markers
        if not line.strip():
            if chunk_buffer:
                flush_chunk(chunk_buffer, current_emotion_str)
                chunk_buffer = ""
            # Mark the last chunk as having a newline after it
            if chunks_with_metadata and not prev_line_empty:
                chunks_with_metadata[-1]["newline_after"] = True
            prev_line_empty = True
            continue
        
        prev_line_empty = False
        
        # Check for emotion tag at the beginning of the line
        leading_emotion = re.match(r'^\[(\w+)\]\s*(.*)', line.strip())
        if leading_emotion:
            if chunk_buffer:
                flush_chunk(chunk_buffer, current_emotion_str)
                chunk_buffer = ""
            bracket_emo = leading_emotion.group(1).strip()
            remainder_text = leading_emotion.group(2).strip()
            if bracket_emo in EMOTION_MAP:
                current_emotion_str = bracket_emo
            if remainder_text:
                chunk_buffer = remainder_text
        else:
            # If this is not the first line and we have content in buffer, add a space
            if chunk_buffer and i > 0:
                chunk_buffer += " "
            chunk_buffer += line.strip()

        # Process inline emotion tags
        while True:
            match = re.search(r'\[(\w+)\]', chunk_buffer)
            if not match:
                break
            pre_text = chunk_buffer[:match.start()].strip()
            bracket_emo = match.group(1)
            post_text = chunk_buffer[match.end():].strip()

            if pre_text:
                chunks_with_metadata.append({"text": pre_text, "emotion": current_emotion_str, "newline_after": False})
            if bracket_emo in EMOTION_MAP:
                current_emotion_str = bracket_emo
            chunk_buffer = post_text

    # Don't forget the last chunk
    if chunk_buffer:
        flush_chunk(chunk_buffer, current_emotion_str)

    # Now split by sentences or punctuation if chunks are too large
    final_chunks = []
    max_chars_per_chunk = 200  # Maximum characters per chunk
    
    for chunk in chunks_with_metadata:
        text = chunk["text"]
        
        # Split by sentence-ending punctuation if the text is too long
        if len(text) > max_chars_per_chunk:
            # Split by sentence endings
            sentence_pattern = r'(?<=[.!?])\s+'
            sentences = re.split(sentence_pattern, text)
            
            # Further split long sentences by commas, semicolons, etc.
            for sentence in sentences:
                if len(sentence) > max_chars_per_chunk:
                    clause_pattern = r'(?<=[,;:])\s+'
                    clauses = re.split(clause_pattern, sentence)
                    
                    # Add each clause as a separate chunk
                    for i, clause in enumerate(clauses):
                        is_last = (i == len(clauses) - 1)
                        final_chunks.append({
                            "text": clause,
                            "emotion": chunk["emotion"],
                            "newline_after": chunk["newline_after"] if is_last else False
                        })
                else:
                    final_chunks.append({
                        "text": sentence,
                        "emotion": chunk["emotion"],
                        "newline_after": chunk["newline_after"]
                    })
        else:
            final_chunks.append(chunk)
    
    # Convert to format expected by the rest of the code
    result = []
    for chunk in final_chunks:
        vec = get_emotion_vector(chunk["emotion"])
        result.append({
            "text": chunk["text"],
            "emotion": vec,
            "newline_after": chunk["newline_after"]
        })
    
    return result

def run_zonos_tts(language, emotion_choice, text, speaker_sample, speed, progress=gr.Progress(track_tqdm=True)):
    """Generate audio using the Zonos TTS model."""
    import time
    import torch
    import torchaudio
    from pydub import AudioSegment, effects  # for compression & normalization
    import numpy as np

    from modules.zonos.conditioning import make_cond_dict
    from modules.zonos.model import Zonos

    global zonos_model, speaker_sample_file

    try:
        # 1) Load Zonos model
        if not zonos_model:
            z_path = download_model()
            zonos_model = Zonos.from_pretrained(z_path, device="cuda" if torch.cuda.is_available() else "cpu")

        # 2) Load speaker sample -> speaker embedding
        logger.info("Loading speaker sample...")
        wav, sampling_rate = torchaudio.load(speaker_sample)
        s_path = download_speaker_model()
        speaker = zonos_model.make_speaker_embedding(wav, sampling_rate, s_path)

        # 3) Parse text into chunks with metadata
        chunks = _parse_text_and_emotions(text, emotion_choice)
        logger.info(f"Prepared {len(chunks)} chunk(s).")

        sr = zonos_model.autoencoder.sampling_rate
        max_new_tokens = 86 * 30
        audio_segments = []

        # 4) Generate each chunk -> store as [samples]
        for idx, chunk_data in enumerate(chunks):
            chunk_text = chunk_data["text"]
            chunk_emotion = chunk_data["emotion"]
            newline_after = chunk_data["newline_after"]
            
            if not chunk_text:
                continue
                
            logger.info(f"Generating chunk {idx+1}/{len(chunks)}: {chunk_text[:60]}")
            cond_dict = make_cond_dict(
                text=chunk_text,
                speaker=speaker,
                language=language,
                emotion=chunk_emotion,
            )
            conditioning = zonos_model.prepare_conditioning(cond_dict)
            codes = zonos_model.generate(conditioning, max_new_tokens=max_new_tokens)
            wavs = zonos_model.autoencoder.decode(codes).cpu()
            
            if wavs.shape[-1] == 0:
                logger.warning(f"Chunk {idx+1} produced empty audio, skipping")
                continue
                
            audio_segments.append(wavs[0])  # shape [samples]
            
            # Add a pause (0.5s) if this chunk should be followed by a newline
            if newline_after:
                pause_samples = int(0.5 * sr)  # 0.5 seconds of silence
                # Create silence with same dimensions as the generated audio
                if wavs[0].dim() == 1:
                    # If 1D tensor, create 1D silence
                    silence = torch.zeros(pause_samples, dtype=wavs[0].dtype)
                else:
                    # If multi-dimensional, match the shape pattern
                    # For example, if shape is [channels, time], create [channels, pause_samples]
                    shape = list(wavs[0].shape)
                    shape[-1] = pause_samples  # Replace last dimension with pause length
                    silence = torch.zeros(shape, dtype=wavs[0].dtype)
                
                audio_segments.append(silence)
                logger.info(f"Added 0.5s pause after chunk {idx+1}")

        if not audio_segments:
            return "Error: No chunks generated."

        # 5) Concatenate => final_audio [samples]
        # All segments should now have the same dimension structure
        final_audio = torch.cat(audio_segments, dim=-1)

        # 6) (Optional) Apply VAD to remove large leading/trailing silence
        # -> shape [1, time]
        final_audio = final_audio.unsqueeze(0)
        final_audio = torchaudio.functional.vad(final_audio, sr)
        # Some versions produce [batch, channels, time], forcibly squeeze
        while final_audio.dim() > 2:
            final_audio = final_audio.squeeze(0)
        # => [1, time]

        # Convert to pydub AudioSegment => we can do cross‐platform compression & normalization
        # final_audio is shape [1, samples], so .squeeze(0) => [samples]
        samples = final_audio.squeeze(0).numpy()  # float32 or float64
        # Convert from -1..1 float to 16-bit PCM
        samples_int16 = (samples * 32767.0).astype(np.int16)

        seg = AudioSegment(
            samples_int16.tobytes(),
            frame_rate=sr,
            sample_width=2,   # 16-bit => 2 bytes
            channels=1
        )

        # 7) Use pydub to compress + normalize
        # compress_dynamic_range defaults to threshold=-20 dB, ratio=4:1, etc.
        seg = effects.compress_dynamic_range(seg, threshold=-25.0, ratio=3.0, attack=5.0, release=50.0)
        # Then normalize to 0 dBFS peak
        seg = effects.normalize(seg)

        # 8) Apply speed if requested (note: speedup also shifts pitch)
        if speed != 1.0:
            seg = effects.speedup(seg, playback_speed=float(speed))

        # 9) Export final cross‐platform
        out_file = os.path.join(output_path, "zonos", f"ZONOS_{int(time.time())}.wav")
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        seg.export(out_file, format="wav")

        return gr.update(value=out_file)

    except Exception as e:
        logger.exception("Error in Zonos TTS:")
        return f"Error: {str(e)}"

def run_dia_tts(text, prompt_text, speaker_sample, speed, progress=gr.Progress(track_tqdm=True)):
    """Generate audio using the DIA TTS model."""
    import time
    import torch
    import soundfile as sf
    import numpy as np
    from modules.diatts.dia.model import Dia
    
    global dia_model
    
    try:
        # 1) Load DIA model if not already loaded
        if not dia_model:
            logger.info("Loading DIA model...")
            dia_path = download_dia_model()
            dia_model = Dia.from_pretrained(dia_path, device="cuda" if torch.cuda.is_available() else "cpu")
            logger.info("DIA model loaded successfully")
        
        # 2) Generate audio
        logger.info("Generating audio with DIA...")
        
        # Apply voice cloning if speaker_sample is provided
        if speaker_sample and prompt_text:
            logger.info("Using voice cloning with provided audio sample")
            output_audio_np = dia_model.generate(prompt_text + text, audio_prompt_path=speaker_sample)
            # os.remove(temp_file)
        else:
            # Generate without voice cloning
            logger.info("Generating without voice cloning")
            output_audio_np = dia_model.generate(text)
        
        # 3) Apply speed adjustment if needed (simple resampling)
        output_sr = 44100

            # --- Slow down audio ---
        original_len = len(output_audio_np)
        # Ensure speed_factor is positive and not excessively small/large to avoid issues
        speed_factor = max(0.1, min(speed, 5.0))
        target_len = int(
            original_len / speed_factor
        )  # Target length based on speed_factor
        if (
            target_len != original_len and target_len > 0
        ):  # Only interpolate if length changes and is valid
            x_original = np.arange(original_len)
            x_resampled = np.linspace(0, original_len - 1, target_len)
            resampled_audio_np = np.interp(x_resampled, x_original, output_audio_np)
            output_audio_np = resampled_audio_np
            print(
                f"Resampled audio from {original_len} to {target_len} samples for {speed_factor:.2f}x speed."
            )
        else:
            print(f"Skipping audio speed adjustment (factor: {speed_factor:.2f}).")
    
        # 4) Save to file
        timestamp = int(time.time())
        out_dir = os.path.join(output_path, "dia")
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"DIA_{timestamp}.wav")
        
        # Save the file (using 44.1kHz sampling rate as per the example)
        sf.write(out_file, output_audio_np, output_sr)
        logger.info(f"Audio saved to {out_file}")
        
        return gr.update(value=out_file)
    
    except Exception as e:
        logger.exception("Error in DIA TTS:")
        return f"Error: {str(e)}"

def render_tts():
    global SEND_TO_PROCESS_BUTTON, OUTPUT_AUDIO
    tts_handler = TTSHandler()
    set_espeak_lib_path_win()  # Set up espeak for Zonos
    
    def toggle_ui_elements(model_name):
        """Toggle UI elements based on selected model."""
        zonos_selected = model_name == "Zonos"
        dia_selected = model_name == "DIA"
        regular_tts = not (zonos_selected or dia_selected)
        
        if zonos_selected:
            # When Zonos is selected
            languages = supported_language_codes
            language_value = "en-us"
            return (
                gr.update(choices=languages, value=language_value),  # language
                gr.update(visible=True),  # emotion_dropdown
                gr.update(visible=False),  # speaker_list
                gr.update(visible=False),  # dia_prompt_text
                gr.update(placeholder="Enter text to synthesize. For Zonos, you can include [Emotion] tags.\nUse empty lines for pauses (0.5s each).")  # input_text placeholder
            )
        elif dia_selected:
            # When DIA is selected
            return (
                gr.update(choices=["en"], value="en"),  # language (DIA only supports English)
                gr.update(visible=False),  # emotion_dropdown
                gr.update(visible=False),  # speaker_list
                gr.update(visible=True),  # dia_prompt_text
                gr.update(placeholder="Enter text to synthesize with DIA. Use [S1] and [S2] tags for different speakers.\nYou can include non-verbal sounds in parentheses like (laughs), (coughs), etc.")  # input_text placeholder
            )
        else:
            # For regular TTS models
            tts_handler.language = "en"  # Default to English
            languages = tts_handler.available_languages()
            models = tts_handler.available_models()
            tts_handler.load_model(model_name)  # Use the currently selected model instead of the first one
            speakers = tts_handler.available_speakers()
            speaker = speakers[0] if speakers else None
            
            return (
                gr.update(choices=languages, value="en"),  # language
                gr.update(visible=False),  # emotion_dropdown
                gr.update(choices=speakers, value=speaker, visible=True),  # speaker_list
                gr.update(visible=False),  # dia_prompt_text
                gr.update(placeholder="Enter text to synthesize.")  # input_text placeholder
            )

    def update_tts_model(language):
        """Update available models based on selected language (regular TTS)."""
        tts_handler.language = language
        models = tts_handler.available_models()
        # Always include DIA and Zonos as the first options
        return gr.update(choices=["DIA", "Zonos"] + models, value="DIA")

    def select_tts_model(model):
        """Handle model selection for regular TTS."""
        if model == "Zonos" or model == "DIA":
            # For Zonos or DIA, don't try to load it as a regular TTS model
            return gr.update(choices=[], value=None, visible=False)
        else:
            # For regular TTS models
            try:
                tts_handler.load_model(model)
                speakers = tts_handler.available_speakers()
                speaker = speakers[0] if speakers else None
                return gr.update(choices=speakers, value=speaker, visible=True)
            except Exception as e:
                logger.warning(f"Error loading model {model}: {e}")
                # If there's an error loading the model, return empty speaker list
                return gr.update(choices=[], value=None, visible=True)

    def generate_tts(model, text, language, emotion, speaker_sample, dia_prompt, speaker, speed, progress=gr.Progress(track_tqdm=True)):
        """Dispatch to appropriate TTS generation function based on model."""
        try:
            if not text or text.strip() == "":
                return "Error: Please enter some text to speak."
            
            if model == "Zonos":
                # For Zonos, we need a speaker reference sample
                if not speaker_sample:
                    return "Error: Zonos requires a speaker audio reference file."
                
                # Generate using Zonos
                return run_zonos_tts(language, emotion, text, speaker_sample, speed, progress)
            elif model == "DIA":
                # Generate using DIA
                return run_dia_tts(text, dia_prompt, speaker_sample, speed, progress)
            else:
                # Generate using regular TTS models
                try:
                    spoken = tts_handler.handle(
                        text=text, model_name=model, speaker_wav=speaker_sample, selected_speaker=speaker, speed=speed
                    )
                    return gr.update(value=spoken)
                except Exception as e:
                            logger.exception(f"Error in regular TTS generation with model {model}:")
                            return f"Error: Could not generate speech with {model}: {str(e)}"
        except Exception as e:
            logger.exception("Error in TTS generation:")
            return f"Error: {str(e)}"

    with gr.Blocks() as tts:
        gr.Markdown("# 🗣️ Text to Speech")
        gr.Markdown("Convert text to natural-sounding speech using DIA for realistic dialogues, Zonos for emotional synthesis, or various other TTS models. Supports voice cloning from audio references, multiple languages, and adjustable speech parameters.")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🔧 Settings")
                # Adding DIA as the first model, followed by Zonos and others
                tts_model = gr.Dropdown(
                    label="Model",
                    choices=["DIA", "Zonos"] + tts_handler.available_models(),
                    value="DIA",  # Set DIA as default
                    elem_classes="hintitem", elem_id="tts_infer_model", key="tts_infer_model"
                )
                
                tts_language = gr.Dropdown(
                    label="Language",
                    choices=["en"],  # Default to English for DIA
                    value="en",  # Default for DIA
                    elem_classes="hintitem", elem_id="tts_infer_language", key="tts_infer_language"
                )
                
                # Add emotion dropdown for Zonos (initially hidden since DIA is default)
                emotion_dropdown = gr.Dropdown(
                    label="Emotion",
                    choices=["Normal"] + list(EMOTION_MAP.keys()),
                    value="Normal",
                    visible=False,  # Initially hidden since DIA is default
                    elem_classes="hintitem", elem_id="tts_infer_emotion", key="tts_infer_emotion"
                )
                
                # Regular TTS speaker list (initially hidden)
                available_speakers = tts_handler.available_speakers()
                selected_speaker = available_speakers[0] if available_speakers else None
                speaker_list = gr.Dropdown(
                    label="Speaker",
                    choices=available_speakers,
                    value=selected_speaker,
                    visible=False,  # Initially hidden since DIA is default
                    elem_classes="hintitem", elem_id="tts_infer_speaker", key="tts_infer_speaker"
                )
                
                # Add DIA prompt textbox
                dia_prompt_text = gr.Textbox(
                    label="Voice Clone Prompt (Optional)",
                    placeholder="For voice cloning, enter the transcript that matches your speaker audio. Use the same [S1]/[S2] format.",
                    lines=3,
                    visible=True,  # Initially visible since DIA is default
                    elem_classes="hintitem", elem_id="tts_infer_dia_prompt", key="tts_infer_dia_prompt"
                )
                
                speed_slider = gr.Slider(
                    label="Speech Speed",
                    minimum=0.0,
                    maximum=2.0,
                    step=0.1,
                    value=1.0,
                    elem_classes="hintitem", elem_id="tts_infer_speed", key="tts_infer_speed"
                )
                
                input_text = gr.Textbox(
                    label="Input Text",
                    placeholder="Enter text to synthesize with DIA. Use [S1] and [S2] tags for different speakers.\nYou can include non-verbal sounds in parentheses like (laughs), (coughs), etc.",
                    lines=5,
                    elem_classes="hintitem", elem_id="tts_infer_input_text", key="tts_infer_input_text"
                )
            with gr.Column():
                gr.Markdown("### 🎤 Input")
                speaker_wav = gr.File(
                    label="Speaker Audio",
                    file_count="single",
                    file_types=["audio"],
                    elem_classes="hintitem", elem_id="tts_infer_speaker_wav", key="tts_infer_speaker_wav"
                )
            with gr.Column():
                gr.Markdown("### 🎮 Actions")
                with gr.Row():
                    start_tts = gr.Button(
                        value="Start TTS",
                        variant="primary",
                        elem_classes="hintitem", elem_id="tts_infer_start_button", key="tts_infer_start_button"
                    )
                    SEND_TO_PROCESS_BUTTON = gr.Button(
                        value="Send to Process",
                        variant="secondary",
                        elem_classes="hintitem", elem_id="tts_infer_send_to_process", key="tts_infer_send_to_process"
                    )
                
                gr.Markdown("### 🎶 Outputs")
                OUTPUT_AUDIO = gr.Audio(
                    label="Output Audio",
                    elem_classes="hintitem", elem_id="tts_infer_output_audio", key="tts_infer_output_audio",
                    type="filepath",
                    sources=None,
                    interactive=False
                )

        # Set up event handlers
        tts_model.change(
            fn=toggle_ui_elements,
            inputs=[tts_model],
            outputs=[tts_language, emotion_dropdown, speaker_list, dia_prompt_text, input_text]
        )
        
        # Regular TTS model event handlers - only handle language change if model isn't a special model
        tts_language.change(
            fn=lambda lang, model: update_tts_model(lang) if model not in ["Zonos", "DIA"] else gr.update(),
            inputs=[tts_language, tts_model],
            outputs=[tts_model]
        )
        
        # Ensure proper model selection handling
        tts_model.change(
            fn=select_tts_model,
            inputs=[tts_model],
            outputs=[speaker_list]
        )
        
        # Generation button
        start_tts.click(
            fn=generate_tts,
            inputs=[tts_model, input_text, tts_language, emotion_dropdown, speaker_wav, dia_prompt_text, speaker_list, speed_slider],
            outputs=[OUTPUT_AUDIO]
        )

    return tts

def send_to_process(file_to_send, existing_inputs):
    if not file_to_send or not os.path.exists(file_to_send):
        return gr.update()
    if not existing_inputs:
        existing_inputs = []
    if file_to_send in existing_inputs:
        return gr.update()
    existing_inputs.append(file_to_send)
    return gr.update(value=existing_inputs)

def listen():
    process_inputs = arg_handler.get_element("main", "process_inputs")
    if process_inputs:
        SEND_TO_PROCESS_BUTTON.click(fn=send_to_process, inputs=[OUTPUT_AUDIO, process_inputs], outputs=process_inputs)

def register_api_endpoints(api):
    """
    Register API endpoints for text-to-speech functionality
    
    Args:
        api: FastAPI application instance
    """
    from fastapi import HTTPException, BackgroundTasks, Form, Query
    from fastapi.responses import JSONResponse, FileResponse
    from fastapi.encoders import jsonable_encoder
    from pydantic import BaseModel, Field, validator
    from typing import Optional, List, Dict, Any, Union
    import os
    import uuid
    import time
    import json
    import re

    # Define Pydantic models for request validation
    class TTSOptions(BaseModel):
        """Request model for speech generation"""
        model: str = Field(
            "tts-1", description="The TTS model to use"
        )
        voice: str = Field(
            "alloy", description="The voice to use for speech generation"
        )
        input: str = Field(
            ..., description="The text to convert to speech", max_length=4096
        )
        response_format: str = Field(
            "mp3", description="The format of the generated audio"
        )
        speed: float = Field(
            1.0, description="The speed of the generated speech", ge=0.25, le=4.0
        )
        
        @validator('model')
        def validate_model(cls, v):
            valid_models = ["tts-1", "tts-1-hd"]
            if v not in valid_models:
                raise ValueError(f"Model must be one of: {', '.join(valid_models)}")
            return v
            
        @validator('voice')
        def validate_voice(cls, v):
            valid_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
            if v not in valid_voices:
                raise ValueError(f"Voice must be one of: {', '.join(valid_voices)}")
            return v
            
        @validator('input')
        def validate_input(cls, v):
            if not v:
                raise ValueError("Input text cannot be empty")
            if len(v) > 4096:
                raise ValueError("Input text cannot exceed 4096 characters")
            return v
            
        @validator('response_format')
        def validate_response_format(cls, v):
            valid_formats = ["mp3", "opus", "aac", "flac", "wav"]
            if v not in valid_formats:
                raise ValueError(f"Response format must be one of: {', '.join(valid_formats)}")
            return v
            
        @validator('speed')
        def validate_speed(cls, v):
            if v < 0.25 or v > 4.0:
                raise ValueError("Speed must be between 0.25 and 4.0")
            return v

    @api.post("/api/v1/audio/speech", tags=["Speech"])
    async def api_generate_speech(
        request: TTSOptions,
        background_tasks: BackgroundTasks = None
    ):
        """
        Generate speech from text
        
        This endpoint generates audio from the input text using the specified voice and model.
        """
        try:
            # Generate a unique ID for this speech generation
            speech_id = str(uuid.uuid4())
            timestamp = int(time.time())
            
            # Create output directory if it doesn't exist
            speech_dir = os.path.join(output_path, "speech")
            os.makedirs(speech_dir, exist_ok=True)
            
            # Initialize TTS engine
            tts_engine = TTSEngine()
            
            # Generate speech
            result = tts_engine.generate_speech(
                text=request.input,
                model=request.model,
                voice=request.voice,
                response_format=request.response_format,
                speed=request.speed
            )
            
            if not result.get("success", False):
                raise HTTPException(status_code=500, detail=result.get("error", "Failed to generate speech"))
            
            # Get audio data and save to file
            audio_data = result.get("audio_data")
            if not audio_data:
                raise HTTPException(status_code=500, detail="No audio data generated")
            
            # Determine file extension based on response format
            format_extensions = {
                "mp3": ".mp3",
                "opus": ".opus",
                "aac": ".aac",
                "flac": ".flac",
                "wav": ".wav"
            }
            output_ext = format_extensions.get(request.response_format, ".mp3")
            output_filename = f"speech_{speech_id}{output_ext}"
            output_filepath = os.path.join(speech_dir, output_filename)
            
            # Save audio to file
            with open(output_filepath, "wb") as f:
                f.write(audio_data)
            
            # Create download URL
            download_url = f"/api/v1/audio/speech/download/{output_filename}"
            
            # Get file size
            file_size = os.path.getsize(output_filepath)
            
            # Save metadata
            metadata_filename = f"speech_{speech_id}_metadata.json"
            metadata_filepath = os.path.join(speech_dir, metadata_filename)
            
            # Truncate input text for metadata if too long
            input_preview = request.input
            if len(input_preview) > 100:
                input_preview = input_preview[:100] + "..."
            
            metadata = {
                "id": speech_id,
                "timestamp": timestamp,
                "model": request.model,
                "voice": request.voice,
                "input_text": input_preview,
                "response_format": request.response_format,
                "speed": request.speed,
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
            response_data = {
                "success": True,
                "speech_id": speech_id,
                "output_file": {
                    "filename": output_filename,
                    "format": request.response_format,
                    "download_url": download_url,
                    "file_size_bytes": file_size
                },
                "metadata": {
                    "input_text_preview": input_preview,
                    "model": request.model,
                    "voice": request.voice,
                    "response_format": request.response_format,
                    "speed": request.speed,
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
            logger.exception(f"API speech generation error: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=str(e))

    @api.get("/api/v1/audio/speech/download/{filename}", tags=["Speech"])
    async def api_speech_download(filename: str):
        """
        Download a generated speech file
        
        Parameters:
        - filename: The name of the speech file to download
        
        Returns:
        - The audio file
        """
        try:
            # Validate filename format (prevent path traversal)
            if not re.match(r'^speech_[a-f0-9-]+\.(mp3|opus|aac|flac|wav)$', filename):
                raise HTTPException(status_code=400, detail="Invalid filename format")
            
            # Build the file path
            file_path = os.path.join(output_path, "speech", filename)
            
            # Check if file exists
            if not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail="Speech file not found")
            
            # Determine content type based on file extension
            file_ext = filename.split(".")[-1].lower()
            content_type_map = {
                "mp3": "audio/mpeg",
                "opus": "audio/opus",
                "aac": "audio/aac",
                "flac": "audio/flac",
                "wav": "audio/wav"
            }
            
            content_type = content_type_map.get(file_ext, "audio/mpeg")
            
            # Return the file
            return FileResponse(
                path=file_path,
                filename=filename,
                media_type=content_type
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"API speech download error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @api.get("/api/v1/audio/speech/models", tags=["Speech"])
    async def api_get_speech_models():
        """
        Get available speech models
        
        Returns information about available models for text-to-speech generation.
        """
        models = [
            {
                "id": "tts-1",
                "name": "TTS-1",
                "description": "Standard text-to-speech model optimized for quality and speed",
                "max_input_length": 4096,
                "supported_formats": ["mp3", "opus", "aac", "flac", "wav"]
            },
            {
                "id": "tts-1-hd",
                "name": "TTS-1-HD",
                "description": "High-definition text-to-speech model for premium audio quality",
                "max_input_length": 4096,
                "supported_formats": ["mp3", "opus", "aac", "flac", "wav"]
            }
        ]
        return {"models": models}

    @api.get("/api/v1/audio/speech/voices", tags=["Speech"])
    async def api_get_speech_voices():
        """
        Get available speech voices
        
        Returns information about available voices for text-to-speech generation.
        """
        voices = [
            {
                "id": "alloy",
                "name": "Alloy",
                "description": "Versatile neutral voice with balanced tone",
                "gender": "neutral"
            },
            {
                "id": "echo",
                "name": "Echo",
                "description": "Soft-spoken and professional voice with clarity",
                "gender": "male"
            },
            {
                "id": "fable",
                "name": "Fable",
                "description": "Narrative voice with warm and animated quality",
                "gender": "female"
            },
            {
                "id": "onyx",
                "name": "Onyx",
                "description": "Deep and authoritative voice with gravitas",
                "gender": "male"
            },
            {
                "id": "nova",
                "name": "Nova",
                "description": "Bright and energetic female voice",
                "gender": "female"
            },
            {
                "id": "shimmer",
                "name": "Shimmer",
                "description": "Melodic and dynamic voice with clear articulation",
                "gender": "female"
            }
        ]
        return {"voices": voices}
    
    @api.get("/api/v1/audio/speech/formats", tags=["Speech"])
    async def api_get_speech_formats():
        """
        Get available speech formats
        
        Returns information about supported output formats for speech generation.
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
                "id": "opus",
                "name": "Opus",
                "description": "Modern compressed audio format with excellent quality at low bitrates",
                "mime_type": "audio/opus",
                "extension": ".opus"
            },
            {
                "id": "aac",
                "name": "AAC",
                "description": "Advanced audio coding format with good compression and quality",
                "mime_type": "audio/aac",
                "extension": ".aac"
            },
            {
                "id": "flac",
                "name": "FLAC",
                "description": "Lossless audio compression format with larger file size",
                "mime_type": "audio/flac",
                "extension": ".flac"
            },
            {
                "id": "wav",
                "name": "WAV",
                "description": "Uncompressed audio format with highest quality and largest file size",
                "mime_type": "audio/wav",
                "extension": ".wav"
            }
        ]
        return {"formats": formats}

def register_descriptions(arg_handler: ArgHandler):
    descriptions = {
        "infer_language": "Select the language for text-to-speech synthesis.",
        "infer_model": "Choose the TTS model to use for generating speech. DIA is for realistic dialogues, Zonos for emotional synthesis.",
        "infer_speaker": "Select a speaker from the available voices for the chosen model. Not all models have multiple speakers.",
        "infer_speed": "Adjust the speed of speech output. 1.0 is normal speed.",
        "infer_input_text": "Enter the text to be converted to speech. For DIA, use [S1]/[S2] tags to indicate different speakers and parentheses for non-verbal sounds like (laughs).",
        "infer_speaker_wav": "Upload an audio file to provide a reference speaker voice. Should be 5-15s. Enables voice cloning for DIA and Zonos.",
        "infer_start_button": "Click to generate speech from the input text using the selected model and speaker.",
        "infer_send_to_process": "Send the generated speech output for further processing.",
        "infer_output_audio": "The synthesized speech output will be displayed here as an audio file.",
        "infer_emotion": "Select an overall emotion for Zonos TTS or choose Normal (default) which lets text brackets control emotion.",
        "infer_dia_prompt": "For DIA voice cloning, enter the transcript that matches your reference audio. Use [S1]/[S2] tags as in the main input."
    }
    for elem_id, description in descriptions.items():
        arg_handler.register_description("tts", elem_id, description)

def estimate_audio_duration(text_length, speed=1.0):
    """
    Estimate the duration of generated audio based on text length and speed
    
    Args:
        text_length: Length of the text in characters
        speed: Speaking speed factor
        
    Returns:
        Estimated duration in seconds
    """
    # Approximate speaking rate is about 150 words per minute or 1000 characters per minute at normal speed
    # This is a rough estimate and may vary by voice and content
    chars_per_second = (1000 / 60) * speed
    estimated_seconds = text_length / chars_per_second
    
    # Round to one decimal place
    return round(estimated_seconds, 1)
