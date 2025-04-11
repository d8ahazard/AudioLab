"""
DiffRhythm model module

Contains the DiT (Diffusion Transformer) and CFM (Conditional Flow Matching) models
"""

from modules.diffrythm.model.dit import DiT
from modules.diffrythm.model.cfm import CFM

__all__ = ["DiT", "CFM"] 