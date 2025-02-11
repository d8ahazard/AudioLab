import os
import re

import yt_dlp

from handlers.config import output_path
import gradio as gr
import logging
logger = logging.getLogger(__name__)


def download_files(url, input_files):
    if not input_files:
        input_files = []
    # 1. Validate the URL
    if not url or not url.startswith('http'):
        return gr.update()

    # Split by commas and newlines if present
    urls = re.split(r',|\n', url)
    valid_urls = []
    for url in urls:
        url = url.strip()
        if not url or not url.startswith('http'):
            continue
        valid_urls.append(url)

    for url in valid_urls:
        try:
            # 2. Pre-fetch media information
            ydl_opts = {
                'format': 'bestaudio/best',  # Ensure we get the highest quality audio
                'noplaylist': True,  # Avoid downloading entire playlists
                'quiet': True,  # Minimize yt-dlp output
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if 'entries' in info:  # Handle playlist case (unlikely with noplaylist=True)
                    info = info['entries'][0]

                # Extract and sanitize the title to use as a filename
                title = info.get('title', 'unknown_title')
                sanitized_title = re.sub(r'[\\/*?:"<>|]', "_", title)

                # Construct the file path
                download_dir = os.path.join(output_path, "downloaded")
                os.makedirs(download_dir, exist_ok=True)
                file_path = os.path.join(download_dir, f"{sanitized_title}.mp3")

                # 6. Check if the file already exists
                if os.path.exists(file_path):
                    if not file_path in input_files:
                        input_files.append(file_path)

                # Update ydl_opts for downloading
                ydl_opts.update({
                    'outtmpl': os.path.join(download_dir, sanitized_title),  # Exclude extension
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'mp3',
                        'preferredquality': '192',
                    }]
                })

                # 4. Handle downloading the file
                with yt_dlp.YoutubeDL(ydl_opts) as ydl_download:
                    ydl_download.download([url])

                # Ensure the file was downloaded
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"File not found after download: {file_path}")
                if file_path not in input_files:
                    input_files.append(file_path)

        except Exception as e:
            logger.warning(f"Error: {e}")
            # Return the file path for gr.File
    return gr.update(value=input_files)
