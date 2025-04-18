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
import os
import random
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm

from glob import glob
from torch.utils.data.distributed import DistributedSampler
from modules.wavetransfer.preprocess import get_spec


class NumpyDataset(torch.utils.data.Dataset):
  def __init__(self, paths, files, params, is_training=True):
    super().__init__()
    self.params = params
    self.is_training = is_training
    indices = [(0, 3), (3, 0)] if params.train_mixtures else [(0, 3), (3, 0), (1, 4), (4, 1), (2, 5), (5, 2)]
    self.filenames = []
    if not files:
      filenames = []
      for path in paths:
        filenames += glob(f'{path}/**/*.wav', recursive=True)
      seen = set()
      for x in filenames:
          base_filename = x[:-6]
          if base_filename not in seen:
              seen.add(base_filename)
              self.filenames += [(f'{base_filename}.{i}.wav', f'{base_filename}.{j}.wav') for i, j in indices]
    else:
      assert len(files) == len(paths)
      for path, f in zip(paths, files):
        with open(f, 'r', encoding='utf-8') as fi:
          for x in fi.read().split('\n'):
            if len(x) > 0:
              # 0, 3 (mixtures)
              # 1, 4 (clarinet, strings)
              # 2, 5 (vibraphone, piano)
              self.filenames += [(os.path.join(path, x + f'.{i}.wav'), os.path.join(path, x + f'.{j}.wav')) for i, j in indices]

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, idx):
    audio_filenames = self.filenames[idx]
    # The goal is to preserve the content of instrument1 while using the timbre of instrument2.
    # For data loading purposes, we require the mel-spectrogram of instrument1 and the waveform of instrument2.
    instrument1_filename, instrument2_filename = audio_filenames
    signal1, sr1 = torchaudio.load(instrument1_filename)
    signal2, sr2 = torchaudio.load(instrument2_filename)

    if (self.params.sample_rate != sr1) or (self.params.sample_rate != sr2):
      raise ValueError(f'Invalid sample rate: sr1 = {sr1} and sr2 = {sr2} while sample rate in params is {self.params.sample_rate}.')
    assert signal1.shape == signal2.shape # signals have the same length (number of samples)
    assert signal1.shape[0] == 1 and signal2.shape[0] == 1 # signals are mono

    signal1 = torch.nn.functional.normalize(signal1, p=float('inf'), dim=-1, eps=1e-12)*0.95
    signal1 = signal1.squeeze(0)
    signal2 = torch.nn.functional.normalize(signal2, p=float('inf'), dim=-1, eps=1e-12)*0.95
    signal2 = signal2.squeeze(0)

    if self.is_training:
      if signal1.shape[0] >= (self.params.crop_mel_frames - 1) * self.params.hop_samples:
        start = random.randint(0, signal1.shape[0] - (self.params.crop_mel_frames - 1) * self.params.hop_samples)
        end = start + (self.params.crop_mel_frames - 1) * self.params.hop_samples
        # get segment of audio
        signal1 = signal1[start:end]
        signal2 = signal2[start:end]
      
    try:
      spectrogram = get_spec(signal1, self.params)
    except Exception as e:
      spectrogram = None
    if self.is_training:
      signal2 = torch.hstack([signal2, torch.zeros(self.params.hop_samples)])
      signal1 = torch.hstack([signal1, torch.zeros(self.params.hop_samples)])
    else:
      signal2 = torch.hstack([signal2, torch.zeros(spectrogram.squeeze(0).shape[-1]*self.params.hop_samples - len(signal2))])
      signal1 = torch.hstack([signal1, torch.zeros(spectrogram.squeeze(0).shape[-1]*self.params.hop_samples - len(signal1))])
      assert len(signal2) % self.params.hop_samples == 0
    spectrogram = spectrogram.squeeze(0).T if spectrogram is not None else None
    return {
        'audio': signal2, # Audio that we want the model to generate given the conditioning mel spectrogram and random noise
        'spectrogram': spectrogram, # Mel spectrogram of the conditioning instrument
        'audio_cond_inst': signal1, # Audio of the conditioning instrument
    }


class Collator:
  def __init__(self, params, is_training = True):
    self.params = params
    self.is_training = is_training

  def collate(self, minibatch):
    samples_per_frame = self.params.hop_samples
    for record in minibatch:
      # Filter out records that aren't long enough.
      if (record['spectrogram'] is None) or (len(record['spectrogram']) < self.params.crop_mel_frames):
        del record['spectrogram']
        del record['audio']
        del record['audio_cond_inst']
        continue

      record['spectrogram'] = record['spectrogram'].T
      record['audio'] = record['audio']
      record['audio_cond_inst'] = record['audio_cond_inst']

    audio = torch.stack([record['audio'] for record in minibatch if 'audio' in record])
    audio_cond_inst = torch.stack([record['audio_cond_inst'] for record in minibatch if 'audio_cond_inst' in record])
    spectrogram = torch.stack([record['spectrogram'] for record in minibatch if 'spectrogram' in record])
    return {
        'audio': audio,
        'spectrogram': spectrogram,
        'audio_cond_inst': audio_cond_inst
    }

def from_path(data_dirs, training_files, params, is_distributed=False):
  dataset = NumpyDataset(data_dirs, training_files, params, is_training=True)
  print(len(dataset), "files for training")
  return torch.utils.data.DataLoader(
      dataset,
      batch_size=params.batch_size,
      collate_fn=Collator(params).collate,
      shuffle=not is_distributed,
      sampler=DistributedSampler(dataset) if is_distributed else None,
      pin_memory=True,
      drop_last=True,
      num_workers=os.cpu_count())

def from_path_valid(data_dirs, validation_files, params, is_distributed=False):
  dataset = NumpyDataset(data_dirs, validation_files, params, is_training=False)
  return torch.utils.data.DataLoader(
      dataset,
      batch_size=1,
      collate_fn=Collator(params, is_training = False).collate,
      shuffle=False,
      num_workers=1,
      sampler=DistributedSampler(dataset) if is_distributed else None,
      pin_memory=False,
      drop_last=False)
