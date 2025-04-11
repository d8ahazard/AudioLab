import os
import re
import yt_dlp
from typing import Optional, List, Tuple
from handlers.config import output_path
import gradio as gr
import logging
logger = logging.getLogger(__name__)

def convert_vtt_to_lrc(vtt_path: str, lrc_path: str) -> bool:
    """Convert VTT subtitles to LRC format
    
    Args:
        vtt_path: Path to VTT file
        lrc_path: Path to output LRC file
    
    Returns:
        bool: True if conversion successful
    """
    try:
        import webvtt
        
        with open(lrc_path, 'w', encoding='utf-8') as f:
            for caption in webvtt.read(vtt_path):
                # Convert timestamp to LRC format
                start_parts = caption.start.split(':')
                if len(start_parts) == 3:  # HH:MM:SS.mmm
                    mins = int(start_parts[0]) * 60 + int(start_parts[1])
                    secs = float(start_parts[2])
                else:  # MM:SS.mmm
                    mins = int(start_parts[0])
                    secs = float(start_parts[1])
                
                # Format as [MM:SS.xx]
                timestamp = f"[{mins:02d}:{secs:05.2f}]"
                
                # Write each line
                for line in caption.text.strip().split('\n'):
                    if line.strip():
                        f.write(f"{timestamp}{line.strip()}\n")
        return True
    
    except Exception as e:
        logger.error(f"Error converting subtitles: {e}")
        return False

def download_files(url, input_files, include_captions: bool = False) -> gr.update:
    """Download files from URLs with optional caption extraction
    
    Args:
        url: URL or newline-separated URLs to download from
        input_files: Existing list of files
        include_captions: Whether to extract and convert captions
    
    Returns:
        gr.update with updated file list
    """
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
                'quiet': True,  # Minimize yt-dlp output
            }
            
            # Add caption options if requested
            if include_captions:
                ydl_opts.update({
                    'writeautomaticsub': True,  # Auto-generated subs if available
                    'writesubtitles': True,     # Uploaded subs if available
                    'subtitlesformat': 'vtt',   # VTT format includes timing info
                })
            
            # Create download directory
            download_dir = os.path.join(output_path, "downloaded")
            os.makedirs(download_dir, exist_ok=True)
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                if 'entries' in info:  # This is a playlist
                    logger.info(f"Processing playlist with {len(info['entries'])} items")
                    
                    # Update download options
                    ydl_opts.update({
                        'outtmpl': os.path.join(download_dir, '%(title)s'),
                        'postprocessors': [{
                            'key': 'FFmpegExtractAudio',
                            'preferredcodec': 'mp3',
                            'preferredquality': '192',
                        }]
                    })
                    
                    # Download all items in the playlist
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl_download:
                        ydl_download.download([url])
                    
                    # Add all downloaded files to input_files
                    for entry in info['entries']:
                        if not entry:
                            continue
                        title = entry.get('title', '')
                        if not title:
                            continue
                        
                        sanitized_title = re.sub(r'[\\/*?:"<>|]', "_", title)
                        file_path = os.path.join(download_dir, f"{sanitized_title}.mp3")
                        
                        if os.path.exists(file_path) and file_path not in input_files:
                            logger.info(f"Added playlist item: {sanitized_title}")
                            input_files.append(file_path)
                            
                            # Handle captions if requested
                            if include_captions:
                                base_path = os.path.join(download_dir, sanitized_title)
                                lang = entry.get('language', 'en')
                                
                                # Check for manual subs first, then auto subs
                                sub_files = [
                                    f"{base_path}.{lang}.vtt",         # Manual subs
                                    f"{base_path}.{lang}-orig.vtt",    # Manual subs (alternate)
                                    f"{base_path}.{lang}.automated.vtt" # Auto subs
                                ]
                                
                                for sub_path in sub_files:
                                    if os.path.exists(sub_path):
                                        lrc_path = os.path.join(download_dir, f"{sanitized_title}.lrc")
                                        if convert_vtt_to_lrc(sub_path, lrc_path):
                                            input_files.append(lrc_path)
                                        break
                
                else:  # Single video
                    # Extract and sanitize the title to use as a filename
                    title = info.get('title', 'unknown_title')
                    sanitized_title = re.sub(r'[\\/*?:"<>|]', "_", title)

                    # Construct the file paths
                    file_path = os.path.join(download_dir, f"{sanitized_title}.mp3")
                    
                    # Check if the file already exists
                    if os.path.exists(file_path):
                        if file_path not in input_files:
                            input_files.append(file_path)
                    else:
                        # Update ydl_opts for downloading
                        ydl_opts.update({
                            'outtmpl': os.path.join(download_dir, sanitized_title),  # Exclude extension
                            'postprocessors': [{
                                'key': 'FFmpegExtractAudio',
                                'preferredcodec': 'mp3',
                                'preferredquality': '192',
                            }]
                        })

                        # Download the file
                        with yt_dlp.YoutubeDL(ydl_opts) as ydl_download:
                            ydl_download.download([url])

                        # Ensure the file was downloaded
                        if os.path.exists(file_path):
                            if file_path not in input_files:
                                input_files.append(file_path)
                    
                    # Handle captions if requested
                    if include_captions:
                        base_path = os.path.join(download_dir, sanitized_title)
                        lang = info.get('language', 'en')
                        
                        # Check for manual subs first, then auto subs
                        sub_files = [
                            f"{base_path}.{lang}.vtt",         # Manual subs
                            f"{base_path}.{lang}-orig.vtt",    # Manual subs (alternate)
                            f"{base_path}.{lang}.automated.vtt" # Auto subs
                        ]
                        
                        for sub_path in sub_files:
                            if os.path.exists(sub_path):
                                lrc_path = os.path.join(download_dir, f"{sanitized_title}.lrc")
                                if convert_vtt_to_lrc(sub_path, lrc_path):
                                    input_files.append(lrc_path)
                                break

        except Exception as e:
            logger.warning(f"Error downloading {url}: {e}")
            
    # Return the file path for gr.File
    return gr.update(value=input_files)
