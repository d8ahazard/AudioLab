import os
import re
import yt_dlp
import requests
from typing import Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from handlers.config import output_path, model_path
from tqdm import tqdm
import gradio as gr
import logging
logger = logging.getLogger(__name__)

# Timeout for yt-dlp operations (seconds)
YTDLP_TIMEOUT = 120  # 2 minutes max per video


def _register_ca_bundle() -> None:
    """
    Point libcurl / requests at certifi's CA bundle when available.
    This is especially important for yt-dlp's curl_cffi impersonation backend on Windows.
    """
    try:
        import certifi
    except Exception:
        return
    try:
        bundle = certifi.where()
        if not bundle or not os.path.isfile(bundle):
            return
        for env_var in ("SSL_CERT_FILE", "CURL_CA_BUNDLE", "REQUESTS_CA_BUNDLE"):
            os.environ.setdefault(env_var, bundle)
    except Exception:
        return


def _get_impersonate_target():
    """
    Return a yt-dlp impersonation target (Chrome) if curl_cffi backend is available.
    Mirrors FaceFusion's approach: be loud about version mismatches instead of silently failing.
    """
    try:
        from yt_dlp.networking.impersonate import ImpersonateTarget
    except ImportError as exc:
        logger.info(f"yt-dlp impersonation unavailable: {exc}")
        return None
    try:
        import yt_dlp.networking._curlcffi  # noqa: F401 (verifies backend available)
    except ImportError as exc:
        logger.warning(
            "yt-dlp impersonation disabled (curl_cffi missing or version mismatch). "
            f"Install `curl-cffi>=0.14,<0.15`. Details: {exc}"
        )
        return None
    except Exception as exc:
        logger.warning(f"curl_cffi handler failed to load; impersonate disabled: {exc}")
        return None
    try:
        target = ImpersonateTarget.from_str("chrome")
        logger.info(f"yt-dlp impersonation enabled (target={target})")
        return target
    except Exception as exc:
        logger.warning(f"Failed to create impersonation target; impersonate disabled: {exc}")
        return None


def _is_cookie_db_copy_error(err: Exception) -> bool:
    msg = str(err) if err is not None else ""
    msg_low = msg.lower()
    return (
        ("could not copy" in msg_low and "cookie database" in msg_low)
        or "yt-dlp/issues/7271" in msg_low
    )


def _is_youtube_bot_check_error(err: Exception) -> bool:
    msg = str(err) if err is not None else ""
    msg_low = msg.lower()
    return (
        "sign in to confirm you" in msg_low and "not a bot" in msg_low
    ) or (
        "use --cookies-from-browser" in msg_low
        and "confirm" in msg_low
        and "bot" in msg_low
    )


def _build_ydl_opts(
    *,
    include_captions: bool,
    impersonate_target,
    cookie_browser: Optional[str],
    cookiefile: Optional[str],
) -> dict:
    # Use format and client that avoids SABR streaming issues (see https://github.com/yt-dlp/yt-dlp/issues/12482)
    ydl_opts = {
        'format': 'bestaudio/best',  # Let yt-dlp pick the best available
        'quiet': True,  # Minimize yt-dlp output
        'no_warnings': False,  # Keep warnings visible for debugging
        'socket_timeout': 30,  # Timeout for network operations (seconds)
        'retries': 3,  # Retry failed downloads
        'fragment_retries': 3,  # Retry failed fragments
        'extractor_retries': 3,  # Retry failed extractors
        'file_access_retries': 3,  # Retry failed file access
        'noprogress': True,  # Disable progress bar to avoid blocking
        'ignoreerrors': False,  # Don't ignore errors
        'extract_flat': False,  # Extract full info
        'live_from_start': False,  # Don't try to get live from start
        # Prefer clients that don't require PO tokens / visitor_data
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'tv'],
            }
        },
    }

    # Cookies can help for bot detection. Prefer an explicit cookiefile when provided.
    if cookiefile:
        ydl_opts["cookiefile"] = cookiefile
    elif cookie_browser:
        ydl_opts["cookiesfrombrowser"] = (cookie_browser,)

    # Optional: TLS-fingerprint impersonation (curl_cffi) for sites that block stock clients.
    # When impersonation is on, disable certificate verification (matches yt-dlp CLI usage).
    if impersonate_target is not None:
        ydl_opts["impersonate"] = impersonate_target
        ydl_opts["nocheckcertificate"] = True

    # Add caption options if requested
    if include_captions:
        ydl_opts.update({
            'writeautomaticsub': True,  # Auto-generated subs if available
            'writesubtitles': True,     # Uploaded subs if available
            'subtitlesformat': 'vtt',   # VTT format includes timing info
        })

    # YouTube now relies on EJS challenge solvers (yt-dlp-ejs). If the local
    # package isn't installed, allow yt-dlp to fetch the scripts remotely.
    allow_remote = os.getenv("AUDIOLAB_YTDLP_REMOTE_COMPONENTS", "1") != "0"
    if allow_remote:
        try:
            import importlib.util

            has_ejs = importlib.util.find_spec("yt_dlp_ejs") is not None
        except Exception:
            has_ejs = False
        if not has_ejs:
            ydl_opts["remote_components"] = ["ejs:github"]

    return ydl_opts


def _extract_info_with_timeout(ydl, url, download=False, timeout=YTDLP_TIMEOUT):
    """Run yt-dlp extract_info with a hard timeout to prevent hangs."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(ydl.extract_info, url, download=download)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            logger.error(f"yt-dlp timed out after {timeout}s for URL: {url}")
            raise TimeoutError(f"Download timed out after {timeout} seconds")


def _download_with_timeout(ydl, urls, timeout=YTDLP_TIMEOUT):
    """Run yt-dlp download with a hard timeout to prevent hangs."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(ydl.download, urls)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            logger.error(f"yt-dlp download timed out after {timeout}s")
            raise TimeoutError(f"Download timed out after {timeout} seconds")

def download_hubert_model() -> str:
    """
    Download HuBERT base model from HuggingFace if not already present.
    
    Returns:
        Path to the HuBERT model file
        
    Raises:
        RuntimeError: If download fails
    """
    hubert_dir = os.path.join(model_path, "rvc")
    os.makedirs(hubert_dir, exist_ok=True)
    
    hubert_path = os.path.join(hubert_dir, "hubert_base.pt")
    
    # Check if already downloaded
    if os.path.exists(hubert_path) and os.path.getsize(hubert_path) > 100_000_000:  # > 100MB
        logger.info(f"HuBERT model already exists at {hubert_path}")
        return hubert_path
    
    # HuBERT model URL from official RVC repository
    hubert_url = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt"
    
    try:
        logger.info(f"Downloading HuBERT model from {hubert_url}...")
        logger.info(f"This may take a few minutes (file size: ~189MB)")
        
        response = requests.get(hubert_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc="Downloading HuBERT")
        
        with open(hubert_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    progress_bar.update(len(chunk))
                    f.write(chunk)
        
        progress_bar.close()
        
        # Verify download
        if total_size > 0 and os.path.getsize(hubert_path) < total_size * 0.95:
            raise RuntimeError("Downloaded file size doesn't match expected size")
        
        logger.info(f"HuBERT model downloaded successfully to {hubert_path}")
        return hubert_path
        
    except Exception as e:
        # Clean up partial download
        if os.path.exists(hubert_path):
            os.remove(hubert_path)
        
        error_msg = (
            f"Failed to download HuBERT model: {str(e)}\n\n"
            "Please manually download from:\n"
            f"{hubert_url}\n"
            f"And save to: {hubert_path}"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)


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

    _register_ca_bundle()
    impersonate_target = _get_impersonate_target()

    for url in valid_urls:
        try:
            cookiefile = os.getenv("AUDIOLAB_YTDLP_COOKIES", "").strip()
            if cookiefile and not os.path.exists(cookiefile):
                logger.warning(f"AUDIOLAB_YTDLP_COOKIES points to missing file: {cookiefile}")
                cookiefile = ""

            cookie_browsers = ["chrome", "edge", "firefox", "brave", "chromium", "opera", "vivaldi", "whale"]
            attempts = [("none", None)]
            if cookiefile:
                attempts.append(("cookiefile", cookiefile))
            attempts.extend([("browser", b) for b in cookie_browsers])
            
            # Create download directory
            download_dir = os.path.join(output_path, "downloaded")
            os.makedirs(download_dir, exist_ok=True)

            last_err: Optional[Exception] = None
            info = None
            ydl_opts = None

            for kind, val in attempts:
                try:
                    ydl_opts = _build_ydl_opts(
                        include_captions=include_captions,
                        impersonate_target=impersonate_target,
                        cookie_browser=val if kind == "browser" else None,
                        cookiefile=val if kind == "cookiefile" else None,
                    )
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = _extract_info_with_timeout(ydl, url, download=False)
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    if _is_cookie_db_copy_error(e):
                        logger.warning(f"Could not read {val or 'browser'} cookies (locked/unavailable). Trying next cookie source.")
                        continue
                    if _is_youtube_bot_check_error(e):
                        # If no-cookies attempt hits bot check, move on to cookie attempts.
                        continue
                    raise

            if last_err is not None and info is None:
                raise last_err
                
            if info is None:
                logger.warning(f"Failed to extract info for {url}")
                continue
                
            if 'entries' in info:  # This is a playlist
                logger.info(f"Processing playlist with {len(info['entries'])} items")
                    
                # Update download options (timeout settings already set above)
                ydl_opts.update({
                    'outtmpl': os.path.join(download_dir, '%(title)s'),
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'mp3',
                        'preferredquality': '192',
                    }],
                    'http_chunk_size': 10485760,  # 10MB chunks for more reliable downloads
                })
                    
                # Download all items in the playlist (longer timeout for playlists)
                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl_download:
                        _download_with_timeout(ydl_download, [url], timeout=YTDLP_TIMEOUT * len(info['entries']))
                except Exception as e:
                    if _is_cookie_db_copy_error(e) and "cookiesfrombrowser" in ydl_opts:
                        logger.warning("Chrome cookies failed during download. Retrying playlist download without cookies.")
                        ydl_opts.pop("cookiesfrombrowser", None)
                        with yt_dlp.YoutubeDL(ydl_opts) as ydl_download:
                            _download_with_timeout(ydl_download, [url], timeout=YTDLP_TIMEOUT * len(info['entries']))
                    else:
                        raise
                    
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
                    # Update ydl_opts for downloading (timeout settings already set above)
                    ydl_opts.update({
                        'outtmpl': os.path.join(download_dir, sanitized_title),  # Exclude extension
                        'postprocessors': [{
                            'key': 'FFmpegExtractAudio',
                            'preferredcodec': 'mp3',
                            'preferredquality': '192',
                        }],
                        'http_chunk_size': 10485760,  # 10MB chunks for more reliable downloads
                    })

                    # Download the file (retry without cookies if Chrome cookie DB copy fails)
                    try:
                        with yt_dlp.YoutubeDL(ydl_opts) as ydl_download:
                            _download_with_timeout(ydl_download, [url])
                    except Exception as e:
                        if _is_cookie_db_copy_error(e) and "cookiesfrombrowser" in ydl_opts:
                            logger.warning("Chrome cookies failed during download. Retrying without cookies.")
                            ydl_opts.pop("cookiesfrombrowser", None)
                            with yt_dlp.YoutubeDL(ydl_opts) as ydl_download:
                                _download_with_timeout(ydl_download, [url])
                        else:
                            raise

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

        except TimeoutError as e:
            logger.warning(f"Timeout downloading {url}: {e} - skipping")
        except Exception as e:
            logger.warning(f"Error downloading {url}: {e}")
            
    # Return the file path for gr.File
    return gr.update(value=input_files)
