 """
AudioLab runtime environment patches.

This file is automatically imported by Python at startup (if it's on sys.path),
which is true when running commands from the repo root.

Why this exists:
- `fairseq` imports `torch.utils.tensorboard`, which imports `tensorboard.compat.tf`.
  In this Windows environment, the installed `tensorboard` package is missing
  `tensorboard.compat.notf`, so it falls back to importing TensorFlow and crashes
  due to an `ml_dtypes` binary mismatch.

We don't need TensorFlow for core RVC inference/training, so we provide a tiny
stub to keep imports stable.
"""

from __future__ import annotations

import sys
import types

# Force TensorBoard to use its tensorflow stub (never import TensorFlow)
sys.modules.setdefault("tensorboard.compat.notf", types.ModuleType("tensorboard.compat.notf"))

