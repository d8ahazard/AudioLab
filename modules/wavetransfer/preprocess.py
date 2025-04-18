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
import torch
import torchaudio as T

from librosa.filters import mel as librosa_mel_fn

mel_basis = {}
hann_window = {}

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
  return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
  output = dynamic_range_compression_torch(magnitudes)
  return output

def get_spec(audio, params, center = True):
  sr = params.sample_rate
  hop = params.hop_samples
  n_mels = params.n_mels
  win = hop * 4
  n_fft = 2**((win-1).bit_length())
  fmax = params.fmax
  fmin = params.fmin
  y = audio

  if torch.min(y) < -1.:
      print('min value is ', torch.min(y))
  if torch.max(y) > 1.:
      print('max value is ', torch.max(y))

  global mel_basis, hann_window
  if fmax not in mel_basis:
      mel = librosa_mel_fn(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
      mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
      hann_window[str(y.device)] = torch.hann_window(win).to(y.device)

  spec = torch.stft(y, n_fft, hop_length=hop, win_length=win, window=hann_window[str(y.device)],
                    center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
  spec = torch.sqrt(torch.abs(spec)**2+1e-9)

  spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
  spec = spectral_normalize_torch(spec)
  return spec
