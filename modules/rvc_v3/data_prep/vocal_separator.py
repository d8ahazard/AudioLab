"""
Vocal separation for RVC V3 data preparation.

Integrates with existing separation tools to extract clean vocals.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class VocalSeparator:
    """
    Separate vocals from audio using existing separation models.
    
    Uses the separate_music function from modules.separator.stem_separator.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize the vocal separator.
        
        Args:
            output_dir: Base directory for output (e.g., outputs/rvc_v3_data)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Vocal separator initialized")
    
    def separate_vocals(
        self,
        audio_path: str,
        project_name: str,
        dereverb: bool = False
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Separate vocals from audio.
        
        Args:
            audio_path: Path to the audio file
            project_name: Name of the project
            dereverb: Whether to apply dereverb
        
        Returns:
            Tuple of (vocals_path, instrumental_path) or (None, None) on error
        """
        project_dir = self.output_dir / project_name
        vocals_dir = project_dir / "vocals"
        vocals_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Import separation function
            from modules.separator.stem_separator import separate_music
            
            logger.info(f"Separating vocals from {audio_path}")
            
            # Prepare input dict: {output_dir: [input_files]}
            input_dict = {
                str(vocals_dir): [audio_path]
            }
            
            # Perform separation - vocals_only=True for just vocals and instrumental
            output_paths = separate_music(
                input_dict,
                callback=None,
                vocals_only=True,
                reverb_removal="Main Vocals" if dereverb else "Nothing",
                echo_removal="Main Vocals" if dereverb else "Nothing",
                crowd_removal="Nothing",
                noise_removal="Nothing"
            )
            
            vocals_path = None
            instrumental_path = None
            
            # Find vocals and instrumental outputs
            for path in output_paths:
                path_lower = path.lower()
                if "vocal" in path_lower and "instrumental" not in path_lower:
                    vocals_path = path
                elif "instrumental" in path_lower:
                    instrumental_path = path
            
            if vocals_path is None:
                logger.error("Could not find vocals output")
                # Check what files were created
                logger.info(f"Output files: {output_paths}")
                return None, None
            
            logger.info(f"Separation complete: {vocals_path}")
            return vocals_path, instrumental_path
            
        except Exception as e:
            logger.error(f"Separation failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def get_vocals_path(self, project_name: str) -> Optional[str]:
        """Get the path to separated vocals for a project."""
        project_dir = self.output_dir / project_name / "vocals"
        
        if not project_dir.exists():
            return None
        
        # Look for vocals files
        vocals_files = list(project_dir.glob("*vocals*.wav"))
        if not vocals_files:
            vocals_files = list(project_dir.glob("*.wav"))
        
        if vocals_files:
            return str(vocals_files[0])
        
        return None

