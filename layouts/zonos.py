import logging
import os

import gradio as gr
from huggingface_hub import hf_hub_download
from torch import Tensor

from handlers.args import ArgHandler
from handlers.config import output_path, model_path, app_path
from modules.zonos.conditioning import supported_language_codes

logger = logging.getLogger(__name__)
arg_handler = ArgHandler()
SEND_TO_PROCESS_BUTTON: gr.Button = None
OUTPUT_AUDIO: gr.Audio = None
zonos_model = None
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


def download_model():
    repo_id = "Zyphra/Zonos-v0.1-transformer"
    model_dir = os.path.join(model_path, "zonos")
    if not os.path.exists(model_dir):
        logger.info("Downloading Zonos model...")
    os.makedirs(model_dir, exist_ok=True)
    # Download config.json and model.pth
    _ = hf_hub_download(repo_id=repo_id, filename="config.json", local_dir=model_dir)
    _ = hf_hub_download(repo_id=repo_id, filename="model.safetensors", local_dir=model_dir)
    return model_dir


def download_speaker_model():
    model_dir = os.path.join(model_path, "zonos")
    _ = hf_hub_download(repo_id="Zyphra/Zonos-v0.1-speaker-embedding",
                        filename="ResNet293_SimAM_ASP_base.pt", local_dir=model_dir)
    _ = hf_hub_download(repo_id="Zyphra/Zonos-v0.1-speaker-embedding",
                        filename="ResNet293_SimAM_ASP_base_LDA-128.pt", local_dir=model_dir)
    return model_dir


def set_espeak_lib_path_win():
    if os.name == "nt":
        # Look in Program Files\espeak-ng for espeak-ng.dll
        installed_path = os.path.join(
            os.environ.get("ProgramFiles", "C:\\Program Files"),
            "eSpeak NG",
            "lib    espeak-ng.dll"
        )
        # Check and set PHONEMIZER_ESPEAK_LIBRARY if it exists
        if os.path.exists(installed_path):
            print(f"Found espeak-ng.dll at {installed_path}")
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = installed_path
        else:
            espeak_lib_path = os.path.join(app_path, "libs", "libespeak-ng.dll")
            print(f"Using bundled espeak-ng.dll at {espeak_lib_path}")
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = espeak_lib_path


def render_zonos():
    global SEND_TO_PROCESS_BUTTON, OUTPUT_AUDIO, zonos_model, speaker_sample_file
    set_espeak_lib_path_win()

    def _parse_text_and_emotions(full_text: str, default_emotion: str):
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

        chunks_with_emotion = []
        lines = full_text.splitlines()

        def flush_chunk(buffer, emotion):
            text = buffer.strip()
            if text:
                chunks_with_emotion.append((text, emotion))

        chunk_buffer = ""
        for line in lines:
            if not line.strip():
                flush_chunk(chunk_buffer, current_emotion_str)
                chunk_buffer = ""
                continue

            leading_emotion = re.match(r'^\[(\w+)\]\s*(.*)', line.strip())
            if leading_emotion:
                flush_chunk(chunk_buffer, current_emotion_str)
                chunk_buffer = ""
                bracket_emo = leading_emotion.group(1).strip()
                remainder_text = leading_emotion.group(2).strip()
                if bracket_emo in EMOTION_MAP:
                    current_emotion_str = bracket_emo
                if remainder_text:
                    chunk_buffer = remainder_text
            else:
                if chunk_buffer:
                    chunk_buffer += " "
                chunk_buffer += line.strip()

            while True:
                match = re.search(r'\[(\w+)\]', chunk_buffer)
                if not match:
                    break
                pre_text = chunk_buffer[:match.start()].strip()
                bracket_emo = match.group(1)
                post_text = chunk_buffer[match.end():].strip()

                if pre_text:
                    chunks_with_emotion.append((pre_text, current_emotion_str))

                if bracket_emo in EMOTION_MAP:
                    current_emotion_str = bracket_emo

                chunk_buffer = post_text

        flush_chunk(chunk_buffer, current_emotion_str)

        final_chunks = []
        chunk_word_limit = 75
        for txt, emo in chunks_with_emotion:
            words = txt.split()
            temp_buf = []
            for w in words:
                temp_buf.append(w)
                if len(temp_buf) >= chunk_word_limit:
                    final_chunks.append((" ".join(temp_buf), emo))
                    temp_buf = []
            if temp_buf:
                final_chunks.append((" ".join(temp_buf), emo))

        result = []
        for txt, e in final_chunks:
            vec = get_emotion_vector(e)
            result.append((txt, vec))
        return result

    def run_tts(language, emotion_choice, text, speaker_sample, speed, progress=gr.Progress(track_tqdm=True)):
        import time
        import torchaudio
        import torch
        import logging
        from modules.zonos.conditioning import make_cond_dict
        from modules.zonos.model import Zonos

        logger = logging.getLogger(__name__)
        global zonos_model, speaker_sample_file

        try:
            if not zonos_model:
                z_path = download_model()
                zonos_model = Zonos.from_pretrained(z_path, device="cuda")

            logger.info("Loading speaker sample...")
            wav, sampling_rate = torchaudio.load(speaker_sample)
            s_path = download_speaker_model()
            logger.info("Computing speaker embedding...")
            speaker = zonos_model.make_speaker_embedding(wav, sampling_rate, s_path)

            chunks = _parse_text_and_emotions(text, emotion_choice)
            logger.info(f"Prepared {len(chunks)} chunk(s) after bracket parsing")

            max_new_tokens = 86 * 30
            sr = zonos_model.autoencoder.sampling_rate

            audio_segments = []
            ref_peak_amplitude = None

            # Generate audio for each parsed chunk
            for idx, (chunk_text, chunk_emotion) in enumerate(chunks):
                if not chunk_text:
                    logger.warning("Skipping empty chunk!")
                    continue

                logger.info(f"Generating chunk {idx+1}/{len(chunks)}: {chunk_text[:60]}...")
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
                    logger.warning("Zero-length chunk, skipping.")
                    continue

                current_chunk = wavs[0]  # shape [time]
                current_peak = current_chunk.abs().max().item()

                if current_peak > 0:
                    if ref_peak_amplitude is None:
                        ref_peak_amplitude = current_peak
                    else:
                        scale_factor = ref_peak_amplitude / current_peak
                        current_chunk *= scale_factor

                audio_segments.append(current_chunk)

            if not audio_segments:
                return "Error: Could not generate audio from any chunks."

            # Concatenate all segments => shape [time]
            final_audio = torch.cat(audio_segments, dim=-1)

            # (A) Give VAD a 2D input => [channels=1, time]
            final_audio = final_audio.unsqueeze(0)  # => [1, time]
            final_audio = torchaudio.functional.vad(final_audio, sr)
            # Some torchaudio versions return shape [batch, channels, time] (i.e. 3D).
            # We'll force-squeeze any leading dimension if it happens:
            while final_audio.dim() > 2:
                final_audio = final_audio.squeeze(0)
            # Now final_audio should be [1, time].

            # (B) Optional final normalization to -1 dB
            peak_val = final_audio.abs().max().item()
            if peak_val > 0:
                desired_linear = 10 ** (-1.0 / 20)  # -1 dB
                scale = desired_linear / peak_val
                if scale < 1.0:
                    final_audio *= scale

            # (C) Speed transformations can also add dims => re-squeeze after
            if speed != 1.0:
                effects = [
                    ["speed", str(speed)],
                    ["rate", str(sr)]
                ]
                final_audio, _ = torchaudio.sox_effects.apply_effects_tensor(final_audio, sr, effects)
                # Potentially [1, time] or [batch, channel, time].
                while final_audio.dim() > 2:
                    final_audio = final_audio.squeeze(0)
                # Ensure [1, time] again.

            # final_audio is [1, time], 2D => valid for torchaudio.save
            out_file = os.path.join(output_path, "zonos", f"ZONOS_{int(time.time())}.wav")
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            torchaudio.save(out_file, final_audio, sr)

            return gr.update(value=out_file)

        except Exception as e:
            logger.exception("Error in run_tts:")
            return f"Error: {str(e)}"

    with gr.Blocks() as tts:
        gr.Markdown("## Zonos Text to Speech")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🔧 Settings")
                tts_language = gr.Dropdown(
                    label="Language",
                    choices=supported_language_codes,
                    value="en-us",
                    elem_classes="hintitem", elem_id="zonos_infer_language", key="zonos_infer_language"
                )
                emotion_dropdown = gr.Dropdown(
                    label="Emotion",
                    choices=["Normal"] + list(EMOTION_MAP.keys()),
                    value="Normal",
                    elem_classes="hintitem", elem_id="zonos_infer_emotion", key="zonos_infer_emotion"
                )
                speed_slider = gr.Slider(
                    label="Speech Speed",
                    minimum=0.0,
                    maximum=2.0,
                    step=0.1,
                    value=1.0,
                    elem_classes="hintitem", elem_id="zonos_infer_speed", key="zonos_infer_speed"
                )
                input_text = gr.Textbox(
                    label="Input Text",
                    placeholder="Enter text to synthesize",
                    lines=3,
                    elem_classes="hintitem", elem_id="zonos_infer_input_text", key="zonos_infer_input_text"
                )
            with gr.Column():
                gr.Markdown("### 🎤 Input")
                speaker_wav = gr.File(
                    label="Speaker Audio",
                    file_count="single",
                    file_types=["audio"],
                    elem_classes="hintitem", elem_id="zonos_infer_speaker_wav", key="zonos_infer_speaker_wav"
                )
            with gr.Column():
                gr.Markdown("### 🎶 Outputs")
                with gr.Row():
                    start_tts = gr.Button(
                        value="Start Zonos",
                        variant="primary",
                        elem_classes="hintitem", elem_id="zonos_infer_start_button", key="zonos_infer_start_button"
                    )
                    SEND_TO_PROCESS_BUTTON = gr.Button(
                        value="Send to Process",
                        variant="secondary",
                        elem_classes="hintitem", elem_id="zonos_infer_send_to_process",
                        key="zonos_infer_send_to_process"
                    )
                OUTPUT_AUDIO = gr.Audio(
                    label="Output Audio",
                    elem_classes="hintitem", elem_id="zonos_infer_output_audio", key="zonos_infer_output_audio",
                    type="filepath"
                )

        start_tts.click(
            fn=run_tts,
            inputs=[tts_language, emotion_dropdown, input_text, speaker_wav, speed_slider],
            outputs=OUTPUT_AUDIO
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
    if process_inputs and SEND_TO_PROCESS_BUTTON:
        SEND_TO_PROCESS_BUTTON.click(fn=send_to_process, inputs=[OUTPUT_AUDIO, process_inputs], outputs=process_inputs)


def register_descriptions(arg_handler: ArgHandler):
    descriptions = {
        "infer_language": "Select the language for text-to-speech synthesis.",
        "infer_model": "Choose the TTS model to use for generating speech.",
        "infer_speaker": "Select a speaker from the available voices for the chosen model. Not all models have multiple speakers.",
        "infer_speed": "Adjust the speed of speech output. 1.0 is normal speed.",
        "infer_input_text": "Enter the text to be converted to speech. Supports multiple lines. You can also include [Emotion] brackets. Possible emotions: Happiness, Happy, Sadness, Sad, Disgust, Disgusted, Fear, Scared, Surprise, Anger, Mad, Other, Neutral.",
        "infer_speaker_wav": "Upload an audio file to provide a reference speaker voice. Should be 5-15s, doesn't work with all tts models.",
        "infer_start_button": "Click to generate speech from the input text using the selected model and speaker.",
        "infer_send_to_process": "Send the generated speech output for further processing.",
        "infer_output_audio": "The synthesized speech output will be displayed here as an audio file.",
        "infer_emotion": "Select an overall emotion or choose Normal (default) which lets text brackets control emotion."
    }
    for elem_id, description in descriptions.items():
        arg_handler.register_description("zonos", elem_id, description)
