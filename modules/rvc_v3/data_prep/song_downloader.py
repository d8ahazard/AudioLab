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
            # Use cookies from browser to bypass bot detection
            "--cookies-from-browser", "chrome",
            # Use TV client (no PO token needed, no SABR) with mweb fallback
            "--extractor-args", "youtube:player_client=tv,mweb;player_skip=webpage,configs",
        ]
        
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
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
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

