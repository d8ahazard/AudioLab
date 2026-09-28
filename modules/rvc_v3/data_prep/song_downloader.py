"""
Song downloader for RVC V3 data preparation.

Downloads high-quality audio from YouTube and other streaming services.
"""

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class SongDownloader:
    """
    Download songs from various sources using yt-dlp.
    
    Supports YouTube, SoundCloud, and other platforms.
    Downloads in high quality (preferring 44.1 kHz or higher).
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize the downloader.
        
        Args:
            output_dir: Base directory for downloads (e.g., outputs/rvc_v3_data)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if yt-dlp is available
        self._check_ytdlp()

    @staticmethod
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

    @staticmethod
    def _impersonation_supported() -> bool:
        """
        Detect whether yt-dlp impersonation (curl_cffi backend) is available in this environment.
        """
        try:
            from yt_dlp.networking.impersonate import ImpersonateTarget  # noqa: F401
        except Exception:
            return False
        try:
            import yt_dlp.networking._curlcffi  # noqa: F401
        except Exception:
            return False
        return True

    @staticmethod
    def _is_chrome_cookie_copy_error(msg: str) -> bool:
        if not msg:
            return False
        msg_low = msg.lower()
        return (
            ("could not copy" in msg_low and "cookie database" in msg_low)
            or "yt-dlp/issues/7271" in msg_low
        )
    
    def _check_ytdlp(self) -> bool:
        """Check if yt-dlp is installed."""
        try:
            result = subprocess.run(
                ["python", "-m", "yt_dlp", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                logger.info(f"yt-dlp version: {result.stdout.strip()}")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        logger.warning(
            "yt-dlp not found. Please install it: pip install yt-dlp"
        )
        return False
    
    def download(
        self,
        url: str,
        project_name: str,
        format_preference: str = "bestaudio",
        extract_audio: bool = True
    ) -> Tuple[Optional[str], Dict]:
        """
        Download a song from a URL.
        
        Args:
            url: URL of the song to download
            project_name: Name of the project (creates subdirectory)
            format_preference: Format preference for yt-dlp
            extract_audio: Whether to extract audio only
        
        Returns:
            Tuple of (output_path, metadata) or (None, error_dict)
        """
        project_dir = self.output_dir / project_name / "raw"
        project_dir.mkdir(parents=True, exist_ok=True)
        
        output_template = str(project_dir / "%(title)s.%(ext)s")
        
        # yt-dlp command options
        # Use format and client that avoids SABR streaming issues (see https://github.com/yt-dlp/yt-dlp/issues/12482)
        self._register_ca_bundle()
        cmd = [
            "python", "-m", "yt_dlp",
            url,
            "-o", output_template,
            "--no-playlist",
            "--write-info-json",
            "--socket-timeout", "30",  # Timeout for network operations
            "--retries", "3",  # Retry failed downloads
            "--fragment-retries", "3",  # Retry failed fragments
            "--extractor-retries", "3",  # Retry failed extractors
            "--no-live-from-start",  # Don't try to get live from start
            # Prefer clients that don't require PO tokens / visitor_data
            "--extractor-args", "youtube:player_client=android,tv",
        ]

        # YouTube EJS: allow remote solver scripts when local yt-dlp-ejs isn't installed.
        allow_remote = os.getenv("AUDIOLAB_YTDLP_REMOTE_COMPONENTS", "1") != "0"
        if allow_remote:
            try:
                import importlib.util

                has_ejs = importlib.util.find_spec("yt_dlp_ejs") is not None
            except Exception:
                has_ejs = False
            if not has_ejs:
                cmd.extend(["--remote-components", "ejs:github"])

        # Cookies help with bot checks. Prefer a cookiefile if provided; else try browser cookies.
        cookiefile = os.getenv("AUDIOLAB_YTDLP_COOKIES", "").strip()
        cookiefile_args = []
        cookies_args = []
        cookie_browsers = ["chrome", "edge", "firefox", "brave", "chromium", "opera", "vivaldi", "whale"]

        if cookiefile and os.path.isfile(cookiefile):
            cookiefile_args = ["--cookies", cookiefile]
            cmd.extend(cookiefile_args)
        else:
            # Default: try Chrome browser cookies first
            cookies_args = ["--cookies-from-browser", "chrome"]
            cmd.extend(cookies_args)

        # Optional: TLS-fingerprint impersonation (curl_cffi) for sites that block stock clients.
        # When impersonation is on, disable certificate verification (matches yt-dlp CLI usage).
        if self._impersonation_supported():
            cmd.extend(["--impersonate", "chrome", "--no-check-certificate"])
        
        if extract_audio:
            cmd.extend([
                "-f", "bestaudio/best",  # Let yt-dlp pick best available
                "-x",  # Extract audio
                "--audio-format", "wav",
                "--audio-quality", "0",  # Best quality
                "--postprocessor-args", "ffmpeg:-ar 44100",  # Resample to 44.1kHz
            ])
        else:
            cmd.extend([
                "-f", f"{format_preference}/best",
            ])
        
        try:
            logger.info(f"Downloading from {url}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10 minute timeout

            # If browser cookie DB copy fails, try other browsers before giving up on cookies.
            if result.returncode != 0 and cookies_args and self._is_chrome_cookie_copy_error(result.stderr or result.stdout):
                for browser in cookie_browsers[1:]:
                    logger.warning(f"{cookies_args[-1]} cookies unavailable/locked; retrying with --cookies-from-browser {browser}.")
                    cmd_retry = [c for c in cmd if c not in cookies_args]
                    cmd_retry.extend(["--cookies-from-browser", browser])
                    result = subprocess.run(cmd_retry, capture_output=True, text=True, timeout=600)
                    if result.returncode == 0:
                        break

            # As a last resort, retry without browser cookies
            if result.returncode != 0 and cookies_args and self._is_chrome_cookie_copy_error(result.stderr or result.stdout):
                logger.warning("Browser cookies unavailable/locked; retrying yt-dlp without --cookies-from-browser.")
                cmd_retry = [c for c in cmd if c not in cookies_args]
                result = subprocess.run(cmd_retry, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                error_msg = result.stderr or result.stdout
                logger.error(f"Download failed: {error_msg}")
                return None, {"error": error_msg}
            
            # Find the downloaded file
            downloaded_files = list(project_dir.glob("*.wav"))
            if not downloaded_files:
                downloaded_files = list(project_dir.glob("*.*"))
                downloaded_files = [
                    f for f in downloaded_files 
                    if f.suffix not in ['.json', '.part']
                ]
            
            if not downloaded_files:
                return None, {"error": "No downloaded file found"}
            
            output_path = str(downloaded_files[0])
            
            # Load metadata
            metadata = self._load_metadata(project_dir, downloaded_files[0])
            
            logger.info(f"Downloaded successfully: {output_path}")
            return output_path, metadata
            
        except subprocess.TimeoutExpired:
            logger.error("Download timed out")
            return None, {"error": "Download timed out"}
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return None, {"error": str(e)}
    
    def _load_metadata(self, project_dir: Path, audio_file: Path) -> Dict:
        """Load metadata from yt-dlp info json."""
        json_file = audio_file.with_suffix('.info.json')
        
        if not json_file.exists():
            # Try to find any json file in the directory
            json_files = list(project_dir.glob("*.info.json"))
            if json_files:
                json_file = json_files[0]
            else:
                return {}
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Extract relevant fields
            return {
                "title": metadata.get("title", ""),
                "artist": metadata.get("artist", "") or metadata.get("uploader", ""),
                "url": metadata.get("webpage_url", ""),
                "duration": metadata.get("duration", 0),
                "upload_date": metadata.get("upload_date", ""),
                "description": metadata.get("description", ""),
            }
        except Exception as e:
            logger.warning(f"Could not load metadata: {e}")
            return {}
    
    def get_project_info(self, project_name: str) -> Optional[Dict]:
        """Get information about a downloaded project."""
        project_dir = self.output_dir / project_name / "raw"
        
        if not project_dir.exists():
            return None
        
        audio_files = list(project_dir.glob("*.wav"))
        if not audio_files:
            audio_files = list(project_dir.glob("*.mp3"))
        
        if not audio_files:
            return {"status": "no_audio", "path": str(project_dir)}
        
        metadata = self._load_metadata(project_dir, audio_files[0])
        metadata["audio_path"] = str(audio_files[0])
        metadata["status"] = "downloaded"
        
        return metadata

