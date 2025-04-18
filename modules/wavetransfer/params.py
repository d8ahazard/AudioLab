# Adapted from https://github.com/lmnt-com/wavegrad under the Apache-2.0 license.

# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np


class AttrDict(dict):
  def __init__(self, *args, **kwargs):
      super(AttrDict, self).__init__(*args, **kwargs)
      self.__dict__ = self

  def override(self, attrs):
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


params = AttrDict(
    # Training params
    batch_size=32,
    learning_rate=2e-4,
    max_grad_norm=1.0,

    # Data params
    sample_rate=16000,# 44100 # for 44.1 kHz
    hop_samples=300,  # Don't change this. Really.
    n_mels=128,
    fmin=0,
    fmax=8000, # 22050 # for 44.1 kHz
    crop_mel_frames=66,

    train_mixtures=0, # Indicates the mode of training for the model. When set to 0, the model is trained to handle both mixtures and individual stems for timbre transfer. When set to 1, the model is trained to perform timbre transfer exclusively on mixtures of sounds.

    # Model params
    noise_schedule=np.linspace(1e-6, 0.01, 1000).tolist(),
    inference_noise_schedule=[7e-6, 1.4e-4, 2.1e-3, 2.8e-2, 3.5e-1, 7e-1], # WG-6
    # inference_noise_schedule=np.linspace(1e-6, 0.01, 1000).tolist(),
)
