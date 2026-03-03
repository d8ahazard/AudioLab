"""
RVC V3 Model Components.

Core neural network architectures for the V3 system.
"""

# IMPORTANT:
# Avoid eager imports here. Some optional components (content encoders) depend on
# heavyweight stacks (fairseq/tensorboard/tensorflow) that may not be available
# in all environments. Import from the specific module you need.
from .text_encoder import TextEncoder
from .retrieval import RetrievalIndex
from .generator import RVCV3Generator
from .vocoder import StereoVocoder

__all__ = [
    'TextEncoder',
    'RetrievalIndex',
    'RVCV3Generator',
    'StereoVocoder',
]

