"""
RVC V3 Training Pipeline.
"""

# IMPORTANT:
# Avoid eager imports here. Some optional components (e.g. feature extraction)
# depend on heavyweight stacks (fairseq/tensorboard/tensorflow) that may not be
# available in all environments. Import from the specific module you need.
from .dataset import RVCV3Dataset
from .train import RVCV3Trainer

__all__ = [
    'RVCV3Dataset',
    'RVCV3Trainer',
]

