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
  def __init__(self, paths, files, params, crop_mel = None, is_training = True):
    super().__init__()
    self.params = params
    self.is_training = is_training
    self.filenames = []
    self.crop_mel = crop_mel
    
    if not files:
      # Modified to handle standard filenames without suffixes
      filenames = []
      for path in paths:
        wav_files = glob(f'{path}/**/*.wav', recursive=True)
        mp3_files = glob(f'{path}/**/*.mp3', recursive=True)
        filenames += wav_files + mp3_files
      
      print(f"Found {len(filenames)} audio files")
      
      # Create self-pairs for timbre transfer training
      # Each file is used as both source and target
      for filename in filenames:
        self.filenames.append((filename, filename))
    else:
      # Original code for handling training files list
      assert len(files) == len(paths)
      for path, f in zip(paths, files):
        with open(f, 'r', encoding='utf-8') as fi:
          for x in fi.read().split('\n'):
            if len(x) > 0:
              # 0, 3 (mixtures)
              # 1, 4 (clarinet, strings)
              # 2, 5 (vibraphone, piano)
              indices = [(0, 3), (3, 0)] if params.train_mixtures else [(0, 3), (3, 0), (1, 4), (4, 1), (2, 5), (5, 2)]
              self.filenames += [(os.path.join(path, x + f'.{i}.wav'), os.path.join(path, x + f'.{j}.wav')) for i, j in indices]
    
    print(f"Dataset initialized with {len(self.filenames)} file pairs")
    if len(self.filenames) > 0:
      print(f"First few files: {self.filenames[:2]}")

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, idx):
    audio_filenames = self.filenames[idx]
    # The goal is to preserve the content of instrument1 while using the timbre of instrument2.
    # For data loading purposes, we require the mel-spectrogram of instrument1 and the waveform of instrument2.
    instrument1_filename, instrument2_filename = audio_filenames
    
    try:
      signal1, sr1 = torchaudio.load(instrument1_filename)
      signal2, sr2 = torchaudio.load(instrument2_filename)
      
      # Resample if needed
      if sr1 != self.params.sample_rate:
        resampler = torchaudio.transforms.Resample(sr1, self.params.sample_rate)
        signal1 = resampler(signal1)
      
      if sr2 != self.params.sample_rate:
        resampler = torchaudio.transforms.Resample(sr2, self.params.sample_rate)
        signal2 = resampler(signal2)
      
      # Convert to mono if needed
      if signal1.shape[0] > 1:
        signal1 = torch.mean(signal1, dim=0, keepdim=True)
        
      if signal2.shape[0] > 1:
        signal2 = torch.mean(signal2, dim=0, keepdim=True)
      
      # Normalize
      signal1 = torch.nn.functional.normalize(signal1, p=float('inf'), dim=-1, eps=1e-12)*0.95
      signal1 = signal1.squeeze(0)
      signal2 = torch.nn.functional.normalize(signal2, p=float('inf'), dim=-1, eps=1e-12)*0.95
      signal2 = signal2.squeeze(0)
      
      crop_mel = self.crop_mel if self.crop_mel else self.params.crop_mel_frames
      
      # Make sure we have enough samples
      if signal1.shape[0] >= (crop_mel - 1) * self.params.hop_samples:
        start = random.randint(0, signal1.shape[0] - (crop_mel - 1) * self.params.hop_samples)
        end = start + (crop_mel - 1) * self.params.hop_samples
        # get segment of audio
        signal1 = signal1[start:end]
        signal2 = signal2[start:end]
      else:
        # Pad if too short
        needed_length = (crop_mel - 1) * self.params.hop_samples
        padding = needed_length - signal1.shape[0]
        if padding > 0:
          signal1 = torch.cat([signal1, torch.zeros(padding)])
          signal2 = torch.cat([signal2, torch.zeros(padding)])
        
      try:
        spectrogram = get_spec(signal1, self.params)
      except Exception as e:
        print(f"Error generating spectrogram: {str(e)}")
        spectrogram = None
        
      signal2 = torch.cat([signal2, torch.zeros(self.params.hop_samples)])
      signal1 = torch.cat([signal1, torch.zeros(self.params.hop_samples)])
      spectrogram = spectrogram.squeeze(0).T if spectrogram is not None else None
      
      return {
          'audio': signal2, # Audio that we want the model to generate given the conditioning mel spectrogram and random noise
          'spectrogram': spectrogram, # Mel spectrogram of the conditioning instrument
          'audio_cond_inst': signal1, # Audio of the conditioning instrument
      }
      
    except Exception as e:
      print(f"Error processing audio file {instrument1_filename}: {str(e)}")
      # Return empty tensors as fallback
      dummy_audio = torch.zeros(self.params.hop_samples * self.params.crop_mel_frames)
      dummy_spec = torch.zeros(self.params.n_mels, self.params.crop_mel_frames)
      return {
          'audio': dummy_audio,
          'spectrogram': dummy_spec,
          'audio_cond_inst': dummy_audio,
      }


class Collator:
  def __init__(self, params, crop_mel = None, is_training = True):
    self.params = params
    self.is_training = is_training
    self.crop_mel = crop_mel

  def collate(self, minibatch):
    samples_per_frame = self.params.hop_samples
    crop_mel = self.crop_mel if self.crop_mel else self.params.crop_mel_frames
    
    valid_entries = []
    rejection_reasons = {}
    
    for i, record in enumerate(minibatch):
      record_id = f"record_{i}"
      
      # Skip None entries
      if record is None:
        rejection_reasons[record_id] = "Record is None"
        continue
      
      # Filter out records that aren't long enough or invalid
      if 'spectrogram' not in record or record['spectrogram'] is None:
        rejection_reasons[record_id] = "Missing or None spectrogram"
        continue
        
      if record['audio'] is None or record['audio_cond_inst'] is None:
        rejection_reasons[record_id] = "Missing audio or conditioning audio"
        continue
      
      # Check for zero-sized tensors
      if record['audio'].numel() == 0 or record['audio_cond_inst'].numel() == 0:
        rejection_reasons[record_id] = f"Zero-sized tensors: audio={record['audio'].shape}, cond={record['audio_cond_inst'].shape}"
        continue
      
      # Make sure spectrogram has valid dimensions
      if len(record['spectrogram'].shape) < 2:
        rejection_reasons[record_id] = f"Invalid spectrogram dimensions: {record['spectrogram'].shape}"
        continue
      
      # Ensure we have enough frames in the spectrogram
      if len(record['spectrogram']) < crop_mel:
        # Instead of deleting, try to pad
        try:
          pad_length = crop_mel - len(record['spectrogram'])
          record['spectrogram'] = torch.cat([
              record['spectrogram'], 
              torch.zeros(pad_length, record['spectrogram'].shape[1], dtype=record['spectrogram'].dtype)
          ], dim=0)
        except Exception as e:
          rejection_reasons[record_id] = f"Failed to pad spectrogram: {str(e)}"
          continue
      
      # Prepare the record
      try:
        record['spectrogram'] = record['spectrogram'].T
        valid_entries.append(record)
      except Exception as e:
        rejection_reasons[record_id] = f"Error processing record: {str(e)}"
        continue

    # If no valid entries were found, provide detailed error information
    if len(valid_entries) == 0:
      if len(rejection_reasons) > 0:
        print(f"WARNING: All {len(minibatch)} entries in batch were rejected.")
        print("Rejection reasons:")
        for record_id, reason in rejection_reasons.items():
          print(f"  - {record_id}: {reason}")
        print(f"Required spectrogram frames: {crop_mel}")
      
      # Create a minimal dummy batch
      print("Creating minimal dummy batch...")
      dummy_audio = torch.zeros(self.params.hop_samples * crop_mel)
      dummy_spec = torch.zeros(self.params.n_mels, crop_mel)
      dummy_record = {
        'audio': dummy_audio,
        'spectrogram': dummy_spec,
        'audio_cond_inst': dummy_audio
      }
      valid_entries = [dummy_record]
      print(f"Created dummy batch with audio shape: {dummy_audio.shape}, spec shape: {dummy_spec.shape}")
    
    # Stack valid entries
    try:
      audio = torch.stack([record['audio'] for record in valid_entries])
      audio_cond_inst = torch.stack([record['audio_cond_inst'] for record in valid_entries])
      spectrogram = torch.stack([record['spectrogram'] for record in valid_entries])
      
      return {
          'audio': audio,
          'spectrogram': spectrogram,
          'audio_cond_inst': audio_cond_inst
      }
    except Exception as e:
      print(f"Error stacking batch: {str(e)}")
      # Return a minimal valid batch
      dummy_audio = torch.zeros(1, self.params.hop_samples * crop_mel)
      dummy_spec = torch.zeros(1, self.params.n_mels, crop_mel)
      return {
          'audio': dummy_audio,
          'spectrogram': dummy_spec,
          'audio_cond_inst': dummy_audio
      }

def from_path(data_dirs, training_files, params, batch_size, num_workers, is_distributed = False):
  dataset = NumpyDataset(data_dirs, training_files, params, is_training = True)
  print(len(dataset), "files for training")
  # Force using 0 workers to prevent process spawning
  return torch.utils.data.DataLoader(
      dataset,
      batch_size=batch_size,
      collate_fn=Collator(params).collate,
      shuffle=not is_distributed,
      sampler=DistributedSampler(dataset) if is_distributed else None,
      pin_memory=False,
      drop_last=True,
      num_workers=0)  # Always use 0 workers to prevent spawning subprocesses

def from_path_valid(data_dirs, validation_files, params, num_workers, crop_mel = None, is_distributed = False):
  dataset = NumpyDataset(data_dirs, validation_files, params, crop_mel, is_training = False)
  # Force using 0 workers to prevent process spawning
  return torch.utils.data.DataLoader(
      dataset,
      batch_size=1,
      collate_fn=Collator(params, crop_mel, is_training = False).collate,
      shuffle=False,
      num_workers=0,  # Always use 0 workers to prevent spawning subprocesses
      sampler=DistributedSampler(dataset) if is_distributed else None,
      pin_memory=False,
      drop_last=False)
