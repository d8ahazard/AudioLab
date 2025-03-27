import argparse
import logging
import os
import sys
from pathlib import Path

# Configure logging and fix formatting so time, name, level, are each in []
logging.basicConfig(format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',level=logging.DEBUG)

# Silence other loggers except our app logger
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("fsspec").setLevel(logging.WARNING)
logging.getLogger("onnx2torch").setLevel(logging.WARNING)
logging.getLogger("tensorflow").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("root").setLevel(logging.WARNING)
logging.getLogger("python_multipart").setLevel(logging.WARNING)

# Keep our app logger at DEBUG level
logger = logging.getLogger("ADLB")
logger.setLevel(logging.DEBUG)

import gradio as gr
import uvicorn
from torchaudio._extension import _init_dll_path

import handlers.processing  # noqa (Keep this here, and first, as it is required for multiprocessing to work)
from api import app
from handlers.args import ArgHandler
from handlers.config import model_path
from layouts.music import render as render_music, register_descriptions as music_register_descriptions, \
    listen as music_listen
from layouts.process import render as render_process, register_descriptions as process_register_descriptions, \
    listen as process_listen
from layouts.rvc_train import render as rvc_render, register_descriptions as rvc_register_descriptions
from layouts.tts import render_tts, register_descriptions as tts_register_descriptions, listen as tts_listen
from layouts.stable_audio import render as render_stable_audio, register_descriptions as stable_audio_register_descriptions, \
    listen as stable_audio_listen

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

if os.name == "nt" and (3, 8) <= sys.version_info < (3, 99):
    _init_dll_path()

os.environ["COQUI_TOS_AGREED"] = "1"
os.environ["TTS_HOME"] = model_path
# Stop caching models in limbo!!
hf_dir = os.path.join(model_path, "hf")
transformers_dir = os.path.join(model_path, "transformers")
os.makedirs(hf_dir, exist_ok=True)
os.makedirs(transformers_dir, exist_ok=True)
# Set HF_HUB_CACHE_DIR to the model_path
os.environ["HF_HOME"] = hf_dir

project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="AudioLab Web Server")
    parser.add_argument('--listen', action='store_true', help="Enable server to listen on 0.0.0.0")
    parser.add_argument('--port', type=int, default=7860, help="Specify the port number (default: 7860)")
    parser.add_argument('--api-only', action='store_true', help="Run only the API server without Gradio UI")
    args = parser.parse_args()

    # Determine the launch configuration
    server_name = "0.0.0.0" if args.listen else "127.0.0.1"
    server_port = args.port

    if args.api_only:
        # Run only the FastAPI server
        uvicorn.run(app, host=server_name, port=server_port)
    else:
        # Set up the UI
        arg_handler = ArgHandler()
        process_register_descriptions(arg_handler)
        music_register_descriptions(arg_handler)
        tts_register_descriptions(arg_handler)
        rvc_register_descriptions(arg_handler)
        stable_audio_register_descriptions(arg_handler)

        with open(project_root / 'css' / 'ui.css', 'r') as css_file:
            css = css_file.read()
            css = f'<style type="text/css">{css}</style>'
        # Load the contents of ./js/ui.js
        with open(project_root / 'js' / 'ui.js', 'r') as js_file:
            js = js_file.read()
            js += f"\n{arg_handler.get_descriptions_js()}"
            js = f'<script type="text/javascript">{js}</script>'
            js += f"\n{css}"

        with gr.Blocks(title='AudioLab', head=js, theme="d8ahazard/rd_blue") as ui:
            with gr.Tabs(selected="process"):
                with gr.Tab(label='Process', id="process"):
                    render_process(arg_handler)
                with gr.Tab(label="Train", id="train"):
                    rvc_render()
                with gr.Tab(label="Music", id="music"):
                    render_music(arg_handler)
                with gr.Tab(label='TTS', id="tts"):
                    render_tts()
                with gr.Tab(label='Sound Forge', id="soundforge"):
                    render_stable_audio(arg_handler)

            tts_listen()
            music_listen()
            process_listen()
            stable_audio_listen()

        # Launch both FastAPI and Gradio
        ui.queue()  # Enable queuing for better handling of concurrent requests

        # Mount Gradio app into FastAPI at the root path
        favicon_path = os.path.join(project_root, 'res', 'favicon.ico')
        app = gr.mount_gradio_app(app, ui, path="/", favicon_path=favicon_path)

        # Start the combined server
        logger.info(f"Server running on http://{server_name}:{server_port}")
        logger.info(f"API documentation available at http://{server_name}:{server_port}/docs")
        uvicorn.run(app, host=server_name, port=server_port)
