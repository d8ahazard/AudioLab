"""
ACE-Step module for AudioLab.

A foundation model for high-quality music generation that can create full-length songs
with vocals and instrumentals in various styles.
"""

from modules.acestep.process import process, process_lora, process_retake, process_repaint, process_edit
from modules.acestep.api import register_api_endpoints

__all__ = ["process", "process_lora", "process_retake", "process_repaint", "process_edit", "register_api_endpoints"]
