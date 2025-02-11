import os
import time

import gradio as gr
import torchaudio
from huggingface_hub import snapshot_download

from handlers.args import ArgHandler
from handlers.config import output_path, model_path

arg_handler = ArgHandler()
SEND_TO_PROCESS_BUTTON: gr.Button = None
OUTPUT_AUDIO: gr.Audio = None
zonos_model = None
speaker_sample_file = None


def download_model():
    model_repo = "Zyphra/Zonos-v0.1-transformer"
    model_dir = os.path.join(model_path, "zonos")
    os.makedirs(model_dir, exist_ok=True)
    zonos_path = snapshot_download(model_repo, local_dir=model_dir)
    return zonos_path


def render_zonos():
    global SEND_TO_PROCESS_BUTTON, OUTPUT_AUDIO, zonos_model, speaker_sample_file

    def run_tts(text, speaker_sample, speed, progress=gr.Progress(track_tqdm=True)):
        from modules.zonos.conditioning import make_cond_dict
        from modules.zonos.model import Zonos

        global zonos_model, speaker_sample_file
        try:
            if not zonos_model:
                z_path = download_model()
                zonos_model = Zonos.from_pretrained(z_path, device="cuda")
            wav, sampling_rate = torchaudio.load(speaker_sample)
            speaker = zonos_model.make_speaker_embedding(wav, sampling_rate)
            cond_dict = make_cond_dict(text=text, speaker=speaker, language="en-us")
            conditioning = zonos_model.prepare_conditioning(cond_dict)
            codes = zonos_model.generate(conditioning)
            wavs = zonos_model.autoencoder.decode(codes).cpu()
            out_file = os.path.join(output_path, "zonos", f"ZONOS_{int(time.time())}.wav")
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            torchaudio.save(out_file, wavs[0], zonos_model.autoencoder.sampling_rate)
            return gr.update(value=out_file)
        except Exception as e:
            return f"Error: {str(e)}"

    # If we're on windows, return a message that we don't support TTS YET
    if os.name == "nt":
        tts = gr.HTML("Zonos Text-to-speech is not supported on Windows yet, we're working on it!")
        return tts
    with gr.Blocks() as tts:
        gr.Markdown("## Text to Speech")

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
