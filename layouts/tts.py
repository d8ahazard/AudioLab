import os

import gradio as gr

from handlers.args import ArgHandler
from handlers.tts import TTSHandler

arg_handler = ArgHandler()
SEND_TO_PROCESS_BUTTON: gr.Button = None
OUTPUT_AUDIO: gr.Audio = None


def render_tts():
    global SEND_TO_PROCESS_BUTTON, OUTPUT_AUDIO
    tts_handler = TTSHandler()
    tts_handler.load_model("multilingual/multi-dataset/xtts_v2")

    def update_tts_model(language):
        tts_handler.language = language
        models = tts_handler.available_models()
        return gr.update(choices=models, value=models[0])

    def select_tts_model(model):
        tts_handler.load_model(model)
        speakers = tts_handler.available_speakers()
        speaker = speakers[0] if speakers else None
        return gr.update(choices=speakers, value=speaker)

    def run_tts(text, model, speaker_sample, speaker, speed, progress=gr.Progress(track_tqdm=True)):
        try:
            spoken = tts_handler.handle(
                text=text, model_name=model, speaker_wav=speaker_sample, selected_speaker=speaker, speed=speed
            )
            return gr.update(value=spoken)
        except Exception as e:
            return f"Error: {str(e)}"

    with gr.Blocks() as tts:
        gr.Markdown("## Text to Speech")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🔧 Settings")
                tts_language = gr.Dropdown(
                    label="Language",
                    choices=tts_handler.available_languages(),
                    value="en",
                    elem_classes="hintitem", elem_id="tts_infer_language", key="tts_infer_language"
                )
                tts_model = gr.Dropdown(
                    label="Model",
                    choices=tts_handler.available_models(),
                    value="multilingual/multi-dataset/xtts_v2",
                    elem_classes="hintitem", elem_id="tts_infer_model", key="tts_infer_model"
                )
                available_speakers = tts_handler.available_speakers()
                selected_speaker = available_speakers[0] if available_speakers else None
                speaker_list = gr.Dropdown(
                    label="Speaker",
                    choices=available_speakers,
                    value=selected_speaker,
                    elem_classes="hintitem", elem_id="tts_infer_speaker", key="tts_infer_speaker"
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
                    placeholder="Enter text to synthesize",
                    lines=3,
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
                gr.Markdown("### 🎶 Outputs")
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
                OUTPUT_AUDIO = gr.Audio(
                    label="Output Audio",
                    elem_classes="hintitem", elem_id="tts_infer_output_audio", key="tts_infer_output_audio",
                    type="filepath",
                    sources=None,
                    interactive=False
                )

        tts_language.change(update_tts_model, inputs=tts_language, outputs=tts_model)
        tts_model.change(select_tts_model, inputs=tts_model, outputs=[speaker_list])
        start_tts.click(
            fn=run_tts,
            inputs=[input_text, tts_model, speaker_wav, speaker_list, speed_slider],
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
    if process_inputs:
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
        arg_handler.register_description("tts", elem_id, description)
