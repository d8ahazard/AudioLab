import logging
import os

import gradio as gr
from huggingface_hub import hf_hub_download

from handlers.args import ArgHandler
from handlers.config import output_path, model_path, app_path

logger = logging.getLogger(__name__)
arg_handler = ArgHandler()
SEND_TO_PROCESS_BUTTON: gr.Button = None
OUTPUT_AUDIO: gr.Audio = None
zonos_model = None
speaker_sample_file = None


def download_model():
    repo_id = "Zyphra/Zonos-v0.1-transformer"
    model_dir = os.path.join(model_path, "zonos")
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

        installed_path = os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "eSpeak NG", "lib    espeak-ng.dll")
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

    def run_tts(text, speaker_sample, speed, progress=gr.Progress(track_tqdm=True)):
        import os
        import time
        import re
        import torch
        import torchaudio
        import logging
        from modules.zonos.conditioning import make_cond_dict
        from modules.zonos.model import Zonos

        logger = logging.getLogger(__name__)
        global zonos_model, speaker_sample_file
        try:
            # Download model if needed
            if not zonos_model:
                logger.info("Downloading Zonos model...")
                z_path = download_model()
                zonos_model = Zonos.from_pretrained(z_path, device="cuda")

            # Load speaker sample and compute speaker embedding
            logger.info("Loading speaker sample...")
            wav, sampling_rate = torchaudio.load(speaker_sample)
            s_path = download_speaker_model()
            logger.info("Computing speaker embedding...")
            speaker = zonos_model.make_speaker_embedding(wav, sampling_rate, s_path)

            # -------------------------------------------------------
            # 1) Split input text into sentences using punctuation
            # -------------------------------------------------------
            sentences = re.split(r'(?<=[.!?])\s+', text.strip())
            sentences = [s.strip() for s in sentences if s.strip()]
            logger.info(f"Split into {len(sentences)} raw sentences.")

            # -------------------------------------------------------
            # 2) Group sentences into chunks so each chunk is under
            #    ~86×30 tokens. We'll just group by approximate word
            #    count and trust the model's internal limit.
            #    (We aim for ~75 words per chunk, ~30s)
            # -------------------------------------------------------
            chunk_word_limit = 75
            chunks = []
            current_chunk_words = []
            for sentence in sentences:
                words = sentence.split()
                if len(current_chunk_words) + len(words) > chunk_word_limit:
                    # start a new chunk
                    if current_chunk_words:
                        chunks.append(" ".join(current_chunk_words))
                    current_chunk_words = words
                else:
                    current_chunk_words.extend(words)
            if current_chunk_words:
                chunks.append(" ".join(current_chunk_words))

            logger.info(f"Final chunk count: {len(chunks)}")
            logger.debug(f"Chunks: {chunks}")

            # -------------------------------------------------------
            # 3) Generate audio for each chunk, inserting a small pause
            # -------------------------------------------------------
            audio_segments = []
            max_new_tokens = 86 * 30  # safe upper bound for each chunk
            for idx, chunk_text in enumerate(chunks):
                if not chunk_text:
                    logger.warning("Skipping empty chunk!")
                    continue

                logger.info(f"Generating chunk {idx + 1}/{len(chunks)}: {chunk_text[:60]}...")
                cond_dict = make_cond_dict(text=chunk_text, speaker=speaker, language="en-us")
                conditioning = zonos_model.prepare_conditioning(cond_dict)

                # Run model inference
                codes = zonos_model.generate(conditioning, max_new_tokens=max_new_tokens)
                logger.debug(f"Generated codes shape: {codes.shape}")

                # Decode to waveform
                wavs = zonos_model.autoencoder.decode(codes).cpu()
                logger.debug(f"Decoded waveform shape: {wavs.shape if isinstance(wavs, torch.Tensor) else 'N/A'}")

                if wavs.shape[-1] == 0:
                    logger.warning("Zero-length waveform, skipping.")
                    continue

                audio_segments.append(wavs[0])

                # Insert a short silence after this chunk if not last
                if idx < len(chunks) - 1:
                    # We'll insert ~0.3s real-time pause
                    # (the final audio is scaled by speed, so we do 0.3 * speed now)
                    pause_duration = 0.3 * speed
                    sr = zonos_model.autoencoder.sampling_rate
                    num_pause_samples = int(sr * pause_duration)
                    silence = torch.zeros(wavs[0].shape[0], num_pause_samples)
                    audio_segments.append(silence)

            if not audio_segments:
                return "Error: Could not generate audio from any chunks."

            # -------------------------------------------------------
            # 4) Concatenate all audio segments
            # -------------------------------------------------------
            combined_audio = torch.cat(audio_segments, dim=1)
            logger.info(f"Combined audio shape: {combined_audio.shape}")

            # -------------------------------------------------------
            # 5) Apply speed if needed
            # -------------------------------------------------------
            if speed != 1.0:
                effects = [
                    ["speed", str(speed)],
                    ["rate", str(zonos_model.autoencoder.sampling_rate)]
                ]
                combined_audio, _ = torchaudio.sox_effects.apply_effects_tensor(
                    combined_audio, zonos_model.autoencoder.sampling_rate, effects
                )
                logger.info(f"Applied speed: {speed}")

            # -------------------------------------------------------
            # 6) Save final audio
            # -------------------------------------------------------
            out_file = os.path.join(output_path, "zonos", f"ZONOS_{int(time.time())}.wav")
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            logger.info(f"Saving output to {out_file}")
            torchaudio.save(out_file, combined_audio, zonos_model.autoencoder.sampling_rate)

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
                    choices=["en"],
                    value="en",
                    elem_classes="hintitem", elem_id="zonos_infer_language", key="zonos_infer_language"
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
            inputs=[input_text, speaker_wav, speed_slider],
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
        "infer_input_text": "Enter the text to be converted to speech. Supports multiple lines.",
        "infer_speaker_wav": "Upload an audio file to provide a reference speaker voice. Should be 5-15s, doesn't work with all tts models.",
        "infer_start_button": "Click to generate speech from the input text using the selected model and speaker.",
        "infer_send_to_process": "Send the generated speech output for further processing.",
        "infer_output_audio": "The synthesized speech output will be displayed here as an audio file."
    }
    for elem_id, description in descriptions.items():
        arg_handler.register_description("zonos", elem_id, description)
