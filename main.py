import argparse
import logging
import os
import sys
import signal
import threading
import time
import traceback
import warnings
from pathlib import Path
import gradio as gr
import uvicorn
from torchaudio._extension import _init_dll_path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# Suppress FutureWarnings (particularly from xformers)
warnings.filterwarnings("ignore", category=FutureWarning)

from api import app as api_router
from handlers.args import ArgHandler
from handlers.config import model_path
from layouts.music import render as render_music, register_descriptions as music_register_descriptions, \
    listen as music_listen
from layouts.process import render as render_process, register_descriptions as process_register_descriptions, \
    listen as process_listen
from layouts.rvc_train import render as rvc_render, register_descriptions as rvc_register_descriptions
from layouts.tts import render_tts, register_descriptions as tts_register_descriptions, listen as tts_listen
from layouts.stable_audio import render as render_stable_audio, \
    register_descriptions as stable_audio_register_descriptions, \
    listen as stable_audio_listen
from layouts.orpheus import listen as orpheus_listen, render_orpheus, \
    register_descriptions as orpheus_register_descriptions
from layouts.diffrythm import listen as diffrythm_listen, register_descriptions as diffrythm_register_descriptions, \
    render as render_diffrythm
from layouts.transcribe import listen as transcribe_listen, register_descriptions as transcribe_register_descriptions, \
    render as render_transcribe
from layouts.wavetransfer import listen as wavetransfer_listen, register_descriptions as wavetransfer_register_descriptions, \
    render as render_wavetransfer
from layouts.acestep import listen as acestep_listen, register_descriptions as acestep_register_descriptions, \
    render as render_acestep

# Configure logging and fix formatting so time, name, level, are each in []
logging.basicConfig(format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s', level=logging.DEBUG)

keys_to_silence = [
    "httpx",
    "urllib3",
    "httpcore",
    "asyncio",
    "faiss",
    "fsspec",
    "onnx2torch",
    "tensorflow",
    "matplotlib",
    "PIL",
    "torio",
    "h5py",
    "python_multipart",
    "wandb",
    "sentry_sdk",
    "git",
    "speechbrain",
    "xformers"
]

for key in keys_to_silence:
    logging.getLogger(key).setLevel(logging.INFO)

logging.getLogger("httpx").setLevel(logging.WARNING)

# Keep our app logger at DEBUG level
logger = logging.getLogger("ADLB")
logger.setLevel(logging.DEBUG)

hf_dir = os.path.join(model_path, "hf")
transformers_dir = os.path.join(model_path, "transformers")
os.makedirs(hf_dir, exist_ok=True)
os.makedirs(transformers_dir, exist_ok=True)

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

if os.name == "nt" and (3, 8) <= sys.version_info < (3, 99):
    _init_dll_path()

os.environ["COQUI_TOS_AGREED"] = "1"
os.environ["TTS_HOME"] = model_path
# Stop caching models in limbo!!
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

    # Create a server instance that we can control
    server = None
    shutdown_event = threading.Event()
    
    def signal_handler(sig, frame):
        logger.info("SIGINT or SIGTERM received, shutting down gracefully...")
        shutdown_event.set()
        
        # Give the server some time to shut down gracefully
        def force_shutdown():
            time.sleep(5)
            if not shutdown_event.is_set():
                logger.info("Forcing shutdown after timeout...")
                os._exit(0)
        
        # Start force shutdown timer
        threading.Thread(target=force_shutdown, daemon=True).start()
        
        # If we have a server, try to shut it down gracefully
        if server:
            logger.info("Stopping server...")
            server.should_exit = True
        else:
            # If no server instance available, exit directly
            os._exit(0)

    # Register the signal handler
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)

    try:
        if args.api_only:
            # Run only the FastAPI server
            config = uvicorn.Config(api_router, host=server_name, port=server_port)
            server = uvicorn.Server(config)
            server.run()
        else:
            # Set up the UI
            arg_handler = ArgHandler()
            process_register_descriptions(arg_handler)
            music_register_descriptions(arg_handler)
            tts_register_descriptions(arg_handler)
            rvc_register_descriptions(arg_handler)
            stable_audio_register_descriptions(arg_handler)
            orpheus_register_descriptions(arg_handler)
            diffrythm_register_descriptions(arg_handler)
            transcribe_register_descriptions(arg_handler)
            wavetransfer_register_descriptions(arg_handler)
            acestep_register_descriptions(arg_handler)

            with open(project_root / 'css' / 'ui.css', 'r') as css_file:
                css = css_file.read()
                css = f'<style type="text/css">{css}</style>'
            # Load the contents of ./js/ui.js
            with open(project_root / 'js' / 'ui.js', 'r') as js_file:
                js = js_file.read()
                js += f"\n{arg_handler.get_descriptions_js()}"
                js = f'<script type="text/javascript">{js}</script>'
                js += f"\n{css}"

            with gr.Blocks(title='AudioLab', head=js, theme="d8ahazard/rd_blue") as demo:
                with gr.Tabs(selected="process"):
                    with gr.Tab(label='Process', id="process"):
                        render_process(arg_handler)
                    with gr.Tab(label="Train RVC", id="train"):
                        rvc_render()
                    with gr.Tab(label="Music", id="music"):
                        with gr.Tab(label='ACE-Step', id="acestep"):
                            render_acestep(arg_handler)                    
                        with gr.Tab(label='DiffRhythm', id="diffrythm"):
                            render_diffrythm(arg_handler)                    
                        with gr.Tab(label='Stable-Audio', id="soundforge"):
                            render_stable_audio(arg_handler)                    
                        with gr.Tab(label="YuE", id="yue"):                            
                            render_music(arg_handler)                        
                    with gr.Tab(label='TTS', id="tts"):
                        render_tts()
                    with gr.Tab(label='Orpheus', id="orpheus"):
                        render_orpheus(arg_handler)
                    with gr.Tab(label='Transcribe', id="transcribe"):
                        render_transcribe(arg_handler)
                    with gr.Tab(label='WaveTransfer', id="wavetransfer"):
                        render_wavetransfer(arg_handler)

                tts_listen()
                music_listen()
                process_listen()
                stable_audio_listen()
                orpheus_listen()
                diffrythm_listen()
                acestep_listen()
                transcribe_listen()
                wavetransfer_listen()
                # demo.queue()

            # Create a unified FastAPI app that serves both the API and the Gradio UI
            favicon_path = os.path.join(project_root, "res", "favicon.ico")
            main_app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
            # Serve favicon by mounting static files
            favicon_dir = os.path.dirname(favicon_path)
            main_app.mount("/favicon.ico", StaticFiles(directory=favicon_dir, html=True), name="favicon")
            # Mount the API under /api so that endpoints are available at /api/v1/process/...
            main_app.mount("/api", api_router)
            # Mount the Gradio UI on the root path without including its routes in the schema
            # main_app.mount("/", demo.app, name="gradio")
            main_app = gr.mount_gradio_app(main_app, demo, path="")

            logger.info(f"Server running on http://{server_name}:{server_port}")
            logger.info(f"API documentation available at http://{server_name}:{server_port}/api/docs")

            # Run the unified server with explicit server object
            config = uvicorn.Config(main_app, host=server_name, port=server_port)
            server = uvicorn.Server(config)
            server.run()
    except Exception as e:
        logger.error(f"Error running server: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        logger.info("Server shutdown complete")
