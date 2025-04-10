import argparse
import logging
import os
import sys
from pathlib import Path

from layouts.orpheus import render_orpheus, register_descriptions as orpheus_register_descriptions, listen as orpheus_listen
from layouts.transcribe import render as render_transcribe, register_descriptions as transcribe_register_descriptions, listen as transcribe_listen

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
from fastapi.middleware.cors import CORSMiddleware

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

def setup_middleware(app):
    """Setup CORS and other middleware for the app"""
    # Remove any existing CORS middleware to avoid conflicts
    app.user_middleware = [x for x in app.user_middleware if x.cls.__name__ != 'CORSMiddleware']
    
    # Configure CORS with a permissive policy
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Disable any redirect handling to prevent issues with iframe embedding
    @app.middleware("http")
    async def prevent_redirects(request, call_next):
        # Store original method and URL
        original_method = request.method
        original_url = str(request.url)
        
        # Process the request
        response = await call_next(request)
        
        # If response is a redirect, return a 200 OK instead with the original content
        if response.status_code in (301, 302, 307, 308):
            logger.debug(f"Preventing redirect from {original_url} to {response.headers.get('location', 'unknown')}")
            # Simply drop the status code to 200 and remove location header
            response.status_code = 200
            if 'location' in response.headers:
                del response.headers['location']
            if 'Location' in response.headers:
                del response.headers['Location']
            
        return response
    
    return app

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
        # Run only the FastAPI server without Gradio
        setup_middleware(app)
        uvicorn.run(app, host=server_name, port=server_port)
    else:
        # Set up the UI components and descriptions
        arg_handler = ArgHandler()
        process_register_descriptions(arg_handler)
        music_register_descriptions(arg_handler)
        tts_register_descriptions(arg_handler)
        rvc_register_descriptions(arg_handler)
        stable_audio_register_descriptions(arg_handler)
        orpheus_register_descriptions(arg_handler)
        transcribe_register_descriptions(arg_handler)

        # Load CSS and JS
        with open(project_root / 'css' / 'ui.css', 'r') as css_file:
            css = css_file.read()
            css = f'<style type="text/css">{css}</style>'
        with open(project_root / 'js' / 'ui.js', 'r') as js_file:
            js = js_file.read()
            js += f"\n{arg_handler.get_descriptions_js()}"
            js = f'<script type="text/javascript">{js}</script>'
            js += f"\n{css}"

        # Define the Gradio UI
        demo = gr.Blocks(title='AudioLab', head=js, theme="d8ahazard/rd_blue", analytics_enabled=False)
        
        # Build the interface with tabs
        with demo:
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
                with gr.Tab(label='Orpheus', id="orpheus"):
                    render_orpheus(arg_handler)
                with gr.Tab(label='Transcribe', id="transcribe"):
                    render_transcribe(arg_handler)

            # Set up inter-tab listeners
            tts_listen()
            music_listen()
            process_listen()
            stable_audio_listen()
            orpheus_listen()
            transcribe_listen()

        # Enable queue for handling concurrent requests
        demo.queue()
        
        # ---------------------------------------------------------------------------------
        # Directly adapt Automatic1111's approach to prevent redirect issues
        # ---------------------------------------------------------------------------------
        
        # Set up CORS and other middleware
        setup_middleware(app)
        
        # Gradio uses a very open CORS policy via app.user_middleware, which makes it possible for
        # an attacker to trick the user into opening a malicious HTML page, which makes a request to the
        # running web ui and do whatever the attacker wants. We disable this here. Suggested by RyotaK.
        app.user_middleware = [x for x in app.user_middleware if x.cls.__name__ != 'CORSMiddleware']
        
        # Launch Gradio with specific settings to prevent redirects
        # This is similar to Automatic1111's approach
        auth_creds = None  # No authentication
        
        # Launch with prevent_thread_lock to allow server to run in the main thread
        app, local_url, share_url = demo.launch(
            share=False,
            server_name=server_name,
            server_port=server_port,
            debug=False,
            auth=auth_creds,
            inbrowser=False,
            prevent_thread_lock=True,
            favicon_path=os.path.join(project_root, 'res', 'favicon.ico'),
            show_api=False,
            root_path="",  # Important: empty root_path helps prevent redirects
            app_kwargs={
                "docs_url": "/docs",
                "redoc_url": "/redoc",
            }
        )
        
        # Print startup information
        logger.info(f"Server running on http://{server_name}:{server_port}")
        logger.info(f"API documentation available at http://{server_name}:{server_port}/docs")
        
        # Keep the server running
        import time
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Keyboard interrupt received, shutting down...")
            sys.exit(0)
