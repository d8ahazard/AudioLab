"""
DiffRhythm module for AudioLab - End-to-End full-length song generation

Based on the DiffRhythm model: https://github.com/ASLP-lab/DiffRhythm
"""

from modules.diffrythm.model import CFM, DiT
from modules.diffrythm.infer import inference, prepare_model, get_lrc_token, get_style_prompt, get_negative_style_prompt, get_reference_latent, decode_audio
from modules.diffrythm.infer.infer_utils import check_download_model

__all__ = [
    "CFM",
    "DiT",
    "inference",
    "prepare_model",
    "get_lrc_token",
    "get_style_prompt",
    "get_negative_style_prompt", 
    "get_reference_latent",
    "decode_audio",
    "check_download_model"
] 