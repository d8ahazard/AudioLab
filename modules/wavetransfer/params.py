"""
Parameter handling for WaveTransfer models.
Provides AttrDict class for convenient parameter management.
"""

import numpy as np


class AttrDict(dict):
  """Dictionary subclass that exposes keys as attributes and allows for easy override."""
  
  def __init__(self, *args, **kwargs):
      super(AttrDict, self).__init__(*args, **kwargs)
      self.__dict__ = self

  def override(self, attrs):
    """Override keys with new values from different sources."""
    if isinstance(attrs, dict):
      self.__dict__.update(**attrs)
    elif isinstance(attrs, (list, tuple, set)):
      for attr in attrs:
        self.override(attr)
    elif attrs is None:
      pass
    else:
      raise NotImplementedError
    return self


def get_default_params():
  """Get default parameter settings for WaveTransfer models."""
  return AttrDict(
    # Training params
    batch_size=32,
    learning_rate=2e-4,
    max_grad_norm=1.0,

    # Data params
    sample_rate=16000,  # 44100 for 44.1 kHz
    hop_samples=300,    # Don't change this. Really.
    n_mels=128,
    fmin=0,
    fmax=8000,          # 22050 for 44.1 kHz
    crop_mel_frames=66,

    # Mode selection
    train_mixtures=0,   # 0: handle both mixtures and individual stems, 1: mixtures only

    # Noise scheduling
    noise_schedule=np.linspace(1e-6, 0.01, 1000).tolist(),
    inference_noise_schedule=[7e-6, 1.4e-4, 2.1e-3, 2.8e-2, 3.5e-1, 7e-1],  # WG-6
  )
