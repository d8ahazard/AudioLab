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
import tempfile
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
    self.filenames = []
    
    # Directly handle audio files - AudioLab mode
    # Each file is used for both source and target (self-training)
    print("Using direct file processing mode")
    for path in paths:
      print(f"Searching for audio files in: {path}")
      
      # Get all audio files recursively
      wav_files = glob(f'{path}/**/*.wav', recursive=True)
      mp3_files = glob(f'{path}/**/*.mp3', recursive=True)
      audio_files = wav_files + mp3_files
      
      print(f"Found {len(wav_files)} WAV files and {len(mp3_files)} MP3 files in {path}")
      
      # Use each file as both source and target (self-transfer learning)
      for audio_file in audio_files:
        self.filenames.append((audio_file, audio_file))
    
    # Print dataset stats
    print(f"Dataset initialized with {len(self.filenames)} audio files")
    print(f"First few files: {[os.path.basename(f[0]) for f in self.filenames[:5]]}")
    if len(self.filenames) == 0:
      print("WARNING: No audio files found. Check your data directory paths and file formats.")

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, idx):
    audio_filenames = self.filenames[idx]
    # The goal is to preserve the content of instrument1 while using the timbre of instrument2.
    # For data loading purposes, we require the mel-spectrogram of instrument1 and the waveform of instrument2.
    instrument1_filename, instrument2_filename = audio_filenames
    
    # # Only print info every 10th file to reduce log spam
    # if idx % 10 == 0:
    #   print(f"Loading file {idx}: {os.path.basename(instrument1_filename)}")
    
    # Convert MP3 to WAV if needed
    temp_files = []
    
    if instrument1_filename.endswith('.mp3'):
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_files.append(temp_wav.name)
        try:
            # Use torchaudio to load and resample MP3
            mp3_audio, mp3_sr = torchaudio.load(instrument1_filename)
            if mp3_sr != self.params.sample_rate:
                resampler = torchaudio.transforms.Resample(mp3_sr, self.params.sample_rate)
                mp3_audio = resampler(mp3_audio)
            torchaudio.save(temp_wav.name, mp3_audio, self.params.sample_rate)
            instrument1_filename = temp_wav.name
        except Exception as e:
            print(f"Error converting MP3 file {instrument1_filename}: {str(e)}")
            # Clean up temp files
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass
            # Return empty tensors with correct shapes as fallback
            dummy_audio = torch.zeros(self.params.hop_samples * self.params.crop_mel_frames)
            dummy_spec = torch.zeros(self.params.n_mels, self.params.crop_mel_frames)
            return {
                'audio': dummy_audio,
                'spectrogram': dummy_spec,
                'audio_cond_inst': dummy_audio,
            }
    
    if instrument2_filename.endswith('.mp3'):
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_files.append(temp_wav.name)
        try:
            # Use torchaudio to load and resample MP3
            mp3_audio, mp3_sr = torchaudio.load(instrument2_filename)
            if mp3_sr != self.params.sample_rate:
                resampler = torchaudio.transforms.Resample(mp3_sr, self.params.sample_rate)
                mp3_audio = resampler(mp3_audio)
            torchaudio.save(temp_wav.name, mp3_audio, self.params.sample_rate)
            instrument2_filename = temp_wav.name
        except Exception as e:
            print(f"Error converting MP3 file {instrument2_filename}: {str(e)}")
            # Clean up temp files
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass
            # Return empty tensors with correct shapes as fallback
            dummy_audio = torch.zeros(self.params.hop_samples * self.params.crop_mel_frames)
            dummy_spec = torch.zeros(self.params.n_mels, self.params.crop_mel_frames)
            return {
                'audio': dummy_audio,
                'spectrogram': dummy_spec,
                'audio_cond_inst': dummy_audio,
            }
    
    try:
        # Load audio files
        signal1, sr1 = torchaudio.load(instrument1_filename)
        signal2, sr2 = torchaudio.load(instrument2_filename)
        
        # Only report audio lengths occasionally to reduce logging
        if idx % 10 == 0:
            print(f"  - Signal length: {signal1.shape[1]} samples, Sample rate: {sr1} Hz")

        # Resample if necessary
        if (self.params.sample_rate != sr1) or (self.params.sample_rate != sr2):
            if idx % 10 == 0:
                print(f"  - Resampling from {sr1}/{sr2} to {self.params.sample_rate}")
            if self.params.sample_rate != sr1:
                resampler = torchaudio.transforms.Resample(sr1, self.params.sample_rate)
                signal1 = resampler(signal1)
            if self.params.sample_rate != sr2:
                resampler = torchaudio.transforms.Resample(sr2, self.params.sample_rate)
                signal2 = resampler(signal2)
        
        # Ensure both signals are mono
        if signal1.shape[0] > 1:
            signal1 = torch.mean(signal1, dim=0, keepdim=True)
        if signal2.shape[0] > 1:
            signal2 = torch.mean(signal2, dim=0, keepdim=True)
            
        # Ensure both signals have the same length (number of samples)
        min_length = min(signal1.shape[1], signal2.shape[1])
        signal1 = signal1[:, :min_length]
        signal2 = signal2[:, :min_length]
        
        # Normalize signals
        signal1 = torch.nn.functional.normalize(signal1, p=float('inf'), dim=-1, eps=1e-12)*0.95
        signal1 = signal1.squeeze(0)
        signal2 = torch.nn.functional.normalize(signal2, p=float('inf'), dim=-1, eps=1e-12)*0.95
        signal2 = signal2.squeeze(0)
        
        # Check required length for training
        required_length = (self.params.crop_mel_frames - 1) * self.params.hop_samples
        
        # For training mode, handle signal cropping/padding
        if self.is_training:
            # Make sure we can train on any length audio by taking segments
            if signal1.shape[0] >= required_length:
                # Random crop for training - these are full songs so plenty of material
                start = random.randint(0, signal1.shape[0] - required_length)
                end = start + required_length
                signal1 = signal1[start:end]
                signal2 = signal2[start:end]
                if idx % 10 == 0:
                    print(f"  - Cropped audio from position {start} to {end}")
            else:
                # If short audio, pad with zeros (happens when crop_mel_frames is large)
                if idx % 10 == 0:
                    print(f"  - Audio too short ({signal1.shape[0]} samples), padding to {required_length}")
                pad_length = required_length - signal1.shape[0]
                signal1 = torch.cat([signal1, torch.zeros(pad_length)])
                signal2 = torch.cat([signal2, torch.zeros(pad_length)])
          
        # Generate spectrogram
        try:
            spectrogram = get_spec(signal1, self.params)
            if idx % 10 == 0:
                print(f"  - Generated spectrogram with shape: {spectrogram.shape}")
        except Exception as e:
            print(f"Error generating spectrogram: {str(e)}")
            spectrogram = None
            
        if spectrogram is None:
            # Return empty tensors with correct shapes if spectrogram generation failed
            dummy_audio = torch.zeros(self.params.hop_samples * self.params.crop_mel_frames)
            dummy_spec = torch.zeros(self.params.n_mels, self.params.crop_mel_frames)
            return {
                'audio': dummy_audio,
                'spectrogram': dummy_spec,
                'audio_cond_inst': dummy_audio,
            }
            
        # Add extra padding according to training mode
        if self.is_training:
            signal2 = torch.cat([signal2, torch.zeros(self.params.hop_samples)])
            signal1 = torch.cat([signal1, torch.zeros(self.params.hop_samples)])
        else:
            # For inference, ensure we have enough audio for the spectrogram
            expected_audio_length = spectrogram.squeeze(0).shape[-1] * self.params.hop_samples
            if len(signal2) < expected_audio_length:
                pad_length = expected_audio_length - len(signal2)
                signal2 = torch.cat([signal2, torch.zeros(pad_length)])
                signal1 = torch.cat([signal1, torch.zeros(pad_length)])
            
            assert len(signal2) % self.params.hop_samples == 0, f"Audio length {len(signal2)} not divisible by hop size {self.params.hop_samples}"
            
        spectrogram = spectrogram.squeeze(0).T if spectrogram is not None else None
        
        # Clean up any temporary files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
                
        return {
            'audio': signal2,  # Audio that we want the model to generate given the conditioning mel spectrogram and random noise
            'spectrogram': spectrogram,  # Mel spectrogram of the conditioning instrument
            'audio_cond_inst': signal1,  # Audio of the conditioning instrument
        }
    
    except Exception as e:
        print(f"Error processing audio file {instrument1_filename}: {str(e)}")
        # Clean up any temporary files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
        
        # Return empty tensors with correct shapes as fallback
        dummy_audio = torch.zeros(self.params.hop_samples * self.params.crop_mel_frames)
        dummy_spec = torch.zeros(self.params.n_mels, self.params.crop_mel_frames)
        return {
            'audio': dummy_audio, 
            'spectrogram': dummy_spec,
            'audio_cond_inst': dummy_audio,
        }


class Collator:
  def __init__(self, params, is_training = True):
    self.params = params
    self.is_training = is_training

  def collate(self, minibatch):
    samples_per_frame = self.params.hop_samples
    valid_entries = []
    
    # Record reasons for rejection to help with debugging
    rejection_reasons = {}
    
    for i, record in enumerate(minibatch):
      # Keep track of why records might be rejected
      record_id = f"record_{i}"
      
      # Filter out records that aren't valid or long enough
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
        
      # Always accept and pad if needed (both for training and inference)
      # This is key for handling full songs where we're taking segments
      if len(record['spectrogram']) < self.params.crop_mel_frames:
        # Pad the spectrogram
        pad_length = self.params.crop_mel_frames - len(record['spectrogram'])
        record['spectrogram'] = torch.cat([
            record['spectrogram'], 
            torch.zeros(pad_length, record['spectrogram'].shape[1], dtype=record['spectrogram'].dtype)
        ], dim=0)
      
      record['spectrogram'] = record['spectrogram'].T
      valid_entries.append(record)

    # If no valid entries were found, provide detailed error information and try to create a minimal valid batch
    if len(valid_entries) == 0:
      if len(rejection_reasons) > 0:
        print(f"WARNING: All {len(minibatch)} entries in batch were rejected.")
        print("Rejection reasons:")
        for record_id, reason in rejection_reasons.items():
          print(f"  - {record_id}: {reason}")
        print(f"Required spectrogram frames: {self.params.crop_mel_frames}")
        
        # Try to create a minimal valid batch with the first entry
        if len(minibatch) > 0 and 'spectrogram' in minibatch[0] and minibatch[0]['spectrogram'] is not None:
          print("Attempting to create minimal valid batch...")
          
          # Make a copy to avoid modifying the original data
          record = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in minibatch[0].items()}
            
          # Ensure the spectrogram has enough frames even if we need to pad
          if len(record['spectrogram']) < self.params.crop_mel_frames:
            pad_length = self.params.crop_mel_frames - len(record['spectrogram'])
            record['spectrogram'] = torch.cat([
                record['spectrogram'], 
                torch.zeros(pad_length, record['spectrogram'].shape[1], dtype=record['spectrogram'].dtype)
            ], dim=0)
          
          # Ensure audio matches the spectrogram length
          expected_audio_length = self.params.hop_samples * len(record['spectrogram'])
          if len(record['audio']) < expected_audio_length:
            record['audio'] = torch.cat([
                record['audio'],
                torch.zeros(expected_audio_length - len(record['audio']), dtype=record['audio'].dtype)
            ])
            
          if len(record['audio_cond_inst']) < expected_audio_length:
            record['audio_cond_inst'] = torch.cat([
                record['audio_cond_inst'],
                torch.zeros(expected_audio_length - len(record['audio_cond_inst']), dtype=record['audio_cond_inst'].dtype)
            ])
          
          record['spectrogram'] = record['spectrogram'].T
          valid_entries = [record]
          print(f"Successfully created minimal valid batch with shape: {record['spectrogram'].shape}")
      
      if len(valid_entries) == 0:
        print("WARNING: Could not create a valid batch. Creating empty dummy batch.")
        # Create dummy batch with valid shapes
        dummy_audio = torch.zeros(self.params.hop_samples * self.params.crop_mel_frames)
        dummy_spec = torch.zeros(self.params.n_mels, self.params.crop_mel_frames)
        
        # Create a single valid entry for the batch
        dummy_record = {
            'audio': dummy_audio,
            'spectrogram': dummy_spec.T,  # Transpose to match expected format
            'audio_cond_inst': dummy_audio
        }
        valid_entries = [dummy_record]
        print(f"Created dummy batch with audio shape: {dummy_audio.shape}, spec shape: {dummy_spec.shape}")
    
    # Stack valid entries
    audio = torch.stack([record['audio'] for record in valid_entries])
    audio_cond_inst = torch.stack([record['audio_cond_inst'] for record in valid_entries])
    spectrogram = torch.stack([record['spectrogram'] for record in valid_entries])
    
    return {
        'audio': audio,
        'spectrogram': spectrogram,
        'audio_cond_inst': audio_cond_inst
    }

def from_path(data_dirs, training_files, params, is_distributed=False):
  try:
    print("Initializing dataset with paths:", data_dirs)
    dataset = NumpyDataset(data_dirs, training_files, params, is_training=True)
    if len(dataset) == 0:
      error_msg = "No valid audio files found for training. Please check your data paths:\n"
      error_msg += "- Data directories: " + ", ".join(data_dirs) + "\n"
      if training_files:
        error_msg += "- Training files: " + ", ".join(training_files) + "\n"
      error_msg += "Make sure your audio files are in WAV format or MP3 format and are accessible."
      print(error_msg)
      raise ValueError(error_msg)
      
    print(len(dataset), "files found for training")
    
    # Wrap dataloader in try/except to catch any collation errors
    try:
      # Set up a small batch size for testing - this helps with debugging
      effective_batch_size = min(params.batch_size, 2)  # Start with a small batch size
      print(f"Using batch size: {effective_batch_size} (original: {params.batch_size})")
      
      # Use 0 workers to prevent spawning subprocesses in application context
      num_workers = 0
      
      # Create a custom collator with better error handling
      collator = Collator(params)
      print("Created collator for dataset")
      
      data_loader = torch.utils.data.DataLoader(
          dataset,
          batch_size=effective_batch_size,
          collate_fn=collator.collate,
          shuffle=not is_distributed,
          sampler=DistributedSampler(dataset) if is_distributed else None,
          pin_memory=True,
          drop_last=True,
          num_workers=num_workers)  # Use 0 workers to prevent spawning processes
          
      # Validate that we can get at least one batch from the data loader
      try:
        print("Attempting to load first batch...")
        sample_batch = next(iter(data_loader), None)
        if sample_batch is None:
          raise ValueError("Failed to load any batch. Data loader returned None.")
        
        if len(sample_batch['audio']) == 0:
          raise ValueError("Loaded batch has zero length. Check your audio files and parameters.")
          
        print(f"Successfully loaded first batch with shapes:")
        print(f"- Audio: {sample_batch['audio'].shape}")
        print(f"- Spectrogram: {sample_batch['spectrogram'].shape}")
        print(f"- Conditioning audio: {sample_batch['audio_cond_inst'].shape}")
        
      except Exception as e:
        # Catch any errors that might occur during the first batch load
        error_msg = f"Error loading first batch: {str(e)}\n"
        error_msg += "This could indicate issues with audio file format, length, or corrupt files."
        print(error_msg)
        raise ValueError(error_msg)
      
      return data_loader
    except Exception as e:
      error_msg = f"Error creating data loader: {str(e)}\n"
      error_msg += "This could be due to invalid audio files or batch size issues."
      print(error_msg)
      raise ValueError(error_msg)
      
  except Exception as e:
    error_msg = f"Failed to initialize dataset: {str(e)}"
    print(error_msg)
    raise ValueError(error_msg)


def from_path_valid(data_dirs, validation_files, params, is_distributed=False):
  try:
    dataset = NumpyDataset(data_dirs, validation_files, params, is_training=False)
    if len(dataset) == 0:
      print("Warning: No validation files found. Validation will be skipped.")
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        collate_fn=Collator(params, is_training=False).collate,
        shuffle=False,
        num_workers=0,  # Use 0 workers to prevent spawning processes
        sampler=DistributedSampler(dataset) if is_distributed else None,
        pin_memory=False,
        drop_last=False)
        
  except Exception as e:
    print(f"Error setting up validation dataset: {str(e)}")
    # Return an empty dataloader for validation (won't crash training)
    empty_dataset = torch.utils.data.TensorDataset(torch.zeros(0))
    return torch.utils.data.DataLoader(empty_dataset, batch_size=1)
