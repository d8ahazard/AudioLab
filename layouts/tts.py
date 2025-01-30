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

    def update_tts_model(language):
        tts_handler.language = language
        models = tts_handler.available_models()
        return gr.update(choices=models, value=models[0])

    def select_tts_model(model):
        tts_handler.load_model(model)
        speakers = tts_handler.available_speakers()
        speaker = speakers[0] if speakers else None
        return (
            gr.update(choices=speakers, value=speaker)
        )

    def run_tts(text, model, speaker_sample, speaker, speed, progress=gr.Progress(track_tqdm=True)):
        try:
            spoken = tts_handler.handle(
                text=text, model_name=model, speaker_wav=speaker_sample, selected_speaker=speaker, speed=speed
            )
            return gr.update(value=spoken)
        except Exception as e:
            return f"Error: {str(e)}"

    with gr.Row():
        with gr.Column():
            tts_language = gr.Dropdown(label="Language", choices=tts_handler.available_languages(), value="en")
            tts_model = gr.Dropdown(label="Model", choices=tts_handler.available_models(),
                                    value="multilingual/multi-dataset/xtts_v2")
            available_speakers = tts_handler.available_speakers()
            selected_speaker = available_speakers[0] if available_speakers else None
            speaker_list = gr.Dropdown(label="Speaker", choices=available_speakers, value=selected_speaker)
            speed_slider = gr.Slider(label="Speech Speed", minimum=0.0, maximum=2.0, step=0.1, value=1.0)
            input_text = gr.Textbox(label="Input Text", placeholder="Enter text to synthesize")
        with gr.Column():
            speaker_wav = gr.File(label="Speaker Audio", file_count="single", file_types=["audio"])
        with gr.Column():
            with gr.Row():
                start_tts = gr.Button(value="Start TTS")
                SEND_TO_PROCESS_BUTTON = gr.Button(value="Send to Process", variant="secondary")
            OUTPUT_AUDIO = gr.Audio(label="Output Audio")

    tts_language.change(update_tts_model, inputs=tts_language, outputs=tts_model)
    tts_model.change(select_tts_model, inputs=tts_model, outputs=[speaker_list])
    start_tts.click(
        fn=run_tts,
        inputs=[input_text, tts_model, speaker_wav, speaker_list, speed_slider],
        outputs=OUTPUT_AUDIO
    )


def send_to_process(file_to_send, existing_inputs):
    if not file_to_send or not os.path.exists(file_to_send):
        return gr.update()
    if file_to_send in existing_inputs:
        return gr.update()
    existing_inputs.append(file_to_send)
    return gr.update(value=existing_inputs)


def listen():
    process_inputs = arg_handler.get_element("main", "process_inputs")
    if process_inputs:
        SEND_TO_PROCESS_BUTTON.click(fn=send_to_process, inputs=[OUTPUT_AUDIO, process_inputs], outputs=process_inputs)
