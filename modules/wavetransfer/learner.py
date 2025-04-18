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
import torch
import torchaudio
import torch.nn as nn
import json

from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from modules.wavetransfer.dataset import from_path as dataset_from_path
from modules.wavetransfer.dataset import from_path_valid as dataset_from_path_valid
from modules.wavetransfer.model import WaveGrad
from modules.wavetransfer.params import get_default_params
from modules.wavetransfer.preprocess import get_spec
from modules.wavetransfer.utils import plot_spectrogram, plot_audio, len_audio


def _nested_map(struct, map_fn):
  if isinstance(struct, tuple):
    return tuple(_nested_map(x, map_fn) for x in struct)
  if isinstance(struct, list):
    return [_nested_map(x, map_fn) for x in struct]
  if isinstance(struct, dict):
    return { k: _nested_map(v, map_fn) for k, v in struct.items() }
  return map_fn(struct)


class WaveGradLearner:
  def __init__(self, model_dir, checkpoint_interval, summary_interval, validation_interval, model, dataset, dataset_val, optimizer, params, *args, **kwargs):
    os.makedirs(model_dir, exist_ok=True)
    self.model_dir = model_dir
    self.model = model
    self.dataset = dataset
    self.dataset_val = dataset_val
    self.checkpoint_interval = checkpoint_interval if checkpoint_interval else len(dataset)
    self.summary_interval = summary_interval
    self.validation_interval = validation_interval
    self.optimizer = optimizer
    self.params = params
    self.autocast = torch.cuda.amp.autocast(enabled=kwargs.get('fp16', False))
    self.scaler = torch.cuda.amp.GradScaler(enabled=kwargs.get('fp16', False))
    self.step = 0
    self.is_master = True

    beta = np.array(self.params.noise_schedule)
    noise_level = np.cumprod(1 - beta)**0.5
    noise_level = np.concatenate([[1.0], noise_level], axis=0)
    self.noise_level = torch.tensor(noise_level.astype(np.float32))
    self.loss_fn = nn.L1Loss()
    self.summary_writer = None

  def state_dict(self):
    if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
      model_state = self.model.module.state_dict()
    else:
      model_state = self.model.state_dict()
    return {
        'step': self.step,
        'model': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items() },
        'optimizer': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in self.optimizer.state_dict().items() },
        'params': dict(self.params),
        'scaler': self.scaler.state_dict(),
    }

  def load_state_dict(self, state_dict):
    if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
      self.model.module.load_state_dict(state_dict['model'])
    else:
      self.model.load_state_dict(state_dict['model'])
    self.optimizer.load_state_dict(state_dict['optimizer'])
    self.scaler.load_state_dict(state_dict['scaler'])
    self.step = state_dict['step']

  def save_to_checkpoint(self, filename='weights'):
    save_basename = f'{filename}-{self.step}.pt'
    save_name = f'{self.model_dir}/{save_basename}'
    link_name = f'{self.model_dir}/{filename}.pt'
    torch.save(self.state_dict(), save_name)
    if os.name == 'nt':
      torch.save(self.state_dict(), link_name)
    else:
      if os.path.islink(link_name):
        os.unlink(link_name)
      os.symlink(save_basename, link_name)

  def restore_from_checkpoint(self, filename='weights'):
    try:
      checkpoint = torch.load(f'{self.model_dir}/{filename}.pt')
      self.load_state_dict(checkpoint)
      print("Restored model from checkpoint: {}".format(f'{self.model_dir}/{filename}.pt'))
      return True
    except FileNotFoundError:
      print("No checkpoint found. Starting training from scratch.")
      return False

  def train(self, max_steps=None, max_epochs=None, tqdm_handler=None, cancel_token=None):
    device = next(self.model.parameters()).device
    
    # Check if dataset is empty
    if len(self.dataset) == 0:
      raise ValueError("Dataset is empty. No valid audio files were found. Check your data directory and file formats.")
    
    # Calculate current epoch
    dataset_size = len(self.dataset)
    current_epoch = self.step // dataset_size
    
    # Save current epoch to training state
    self._save_training_state(current_epoch, max_epochs or (max_steps // dataset_size if max_steps else 10))
    
    # Main training loop
    while True:
      # Check if we've reached max epochs
      if max_epochs is not None and current_epoch >= max_epochs:
        logger.info(f"Reached maximum epochs ({max_epochs}). Training complete.")
        return
        
      # Create appropriate progress bar based on tqdm_handler or default tqdm
      if tqdm_handler:
        # Use the provided tqdm_handler to create a progress tracking object
        progress_bar = tqdm_handler(len(self.dataset), f'Epoch {current_epoch}')
        use_progress = True
      else:
        # Use standard tqdm only if this is the master process
        if self.is_master:
          progress_bar = tqdm(self.dataset, desc=f'Epoch {current_epoch}')
          use_progress = True
        else:
          progress_bar = self.dataset
          use_progress = False
      
      for features in progress_bar:
        # Check for cancellation
        if cancel_token and hasattr(cancel_token, 'cancelled') and cancel_token.cancelled:
          logger.info("Training cancelled by user.")
          if use_progress and hasattr(progress_bar, 'close'):
            progress_bar.close()
          return
          
        # Check if we've reached max steps (legacy support)
        if max_steps is not None and self.step >= max_steps:
          if use_progress and hasattr(progress_bar, 'close'):
            progress_bar.close()
          return
          
        features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
        loss = self.train_step(features)
        if torch.isnan(loss).any():
          raise RuntimeError(f'Detected NaN loss at step {self.step}.')
        if self.is_master:
          if self.step % self.summary_interval == 0:
            self._write_summary(self.step, features, loss)
          if self.step % self.validation_interval == 0 and self.step != 0:
            self.run_valid_loop(tqdm_handler=tqdm_handler)
          if self.step % self.checkpoint_interval == 0 and self.step != 0:
            print("INFO: saving checkpoint at step {}".format(self.step))
            self.save_to_checkpoint()
        
        # Update progress bar
        if use_progress and hasattr(progress_bar, 'update'):
          progress_bar.update(1)
          
        self.step += 1
      
      # Close progress bar at the end of each epoch
      if use_progress and hasattr(progress_bar, 'close'):
        progress_bar.close()
      
      # Update and save current epoch
      current_epoch += 1
      self._save_training_state(current_epoch, max_epochs or (max_steps // dataset_size if max_steps else 10))
  
  def _save_training_state(self, current_epoch, total_epochs):
    """Save training state to file for resuming training later."""
    if not self.is_master:
      return
      
    training_state = {
      'current_epoch': current_epoch,
      'total_epochs': total_epochs,
      'step': self.step,
      'dataset_size': len(self.dataset)
    }
    
    try:
      with open(os.path.join(self.model_dir, 'training_state.json'), 'w') as f:
        json.dump(training_state, f, indent=2)
    except Exception as e:
      logger.warning(f"Failed to save training state: {str(e)}")

  def train_step(self, features):
    for param in self.model.parameters():
      param.grad = None

    audio = features['audio']
    spectrogram = features['spectrogram']
    audio_cond_inst = features['audio_cond_inst']

    N, T = audio.shape
    S = 1000
    device = audio.device
    self.noise_level = self.noise_level.to(device)

    with self.autocast:
      s = torch.randint(1, S + 1, [N], device=audio.device)
      l_a, l_b = self.noise_level[s-1], self.noise_level[s]
      noise_scale = l_a + torch.rand(N, device=audio.device) * (l_b - l_a)
      noise_scale = noise_scale.unsqueeze(1)
      noise = torch.randn_like(audio)
      noisy_audio = noise_scale * audio + (1.0 - noise_scale**2)**0.5 * noise

      predicted = self.model(noisy_audio, spectrogram, noise_scale.squeeze(1))
      loss = self.loss_fn(noise, predicted.squeeze(1))

    self.scaler.scale(loss).backward()
    self.scaler.unscale_(self.optimizer)
    self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.params.max_grad_norm)
    self.scaler.step(self.optimizer)
    self.scaler.update()
    return loss

  def _write_summary(self, step, features, loss):
    writer = self.summary_writer or SummaryWriter(self.model_dir, purge_step=step)
    writer.add_scalar('training/loss', loss, step)
    writer.add_scalar('training/grad_norm', self.grad_norm, step)
    writer.flush()
    self.summary_writer = writer

  def _write_summary_valid(self, step, loss, loss_l1, audio_preds, spec_preds, audio_gt, spec_gt, audio_cond_inst_gt, spec_cond_gt):
    writer = self.summary_writer or SummaryWriter(self.model_dir, purge_step=step)
    for i in range(len(audio_preds)):
      if step == self.validation_interval:
        writer.add_audio('ground_truth/audio_{}'.format(i), audio_gt[i], step, self.params.sample_rate)
        writer.add_audio('ground_truth/audio_cond_inst_{}'.format(i), audio_cond_inst_gt[i], step, self.params.sample_rate)
        writer.add_figure('ground_truth/spec_{}'.format(i), plot_spectrogram(spec_gt[i]), step)
        writer.add_figure('ground_truth/spec_cond_inst_{}'.format(i), plot_spectrogram(spec_cond_gt[i]), step)
      writer.add_audio('validation/audio_pred_{}'.format(i), audio_preds[i], step, sample_rate=self.params.sample_rate)
      writer.add_figure('validation/spec_pred{}'.format(i), plot_spectrogram(spec_preds[i]), step)
    writer.add_scalar('validation/loss', loss, step)
    writer.add_scalar('validation/mel_spectrogram_loss', loss_l1, step)
    writer.flush()
    self.summary_writer = writer

  def predict(self, spectrogram):
    with torch.no_grad():
      device = next(self.model.parameters()).device
      beta = np.array(self.params.inference_noise_schedule)
      alpha = 1 - beta
      alpha_cum = np.cumprod(alpha)

      # Expand rank 2 tensors by adding a batch dimension.
      if len(spectrogram.shape) == 2:
        spectrogram = spectrogram.unsqueeze(0)
      spectrogram = spectrogram.to(device)

      audio = torch.randn(spectrogram.shape[0], self.params.hop_samples * spectrogram.shape[-1], device=device)
      noise_scale = torch.from_numpy(alpha_cum**0.5).float().unsqueeze(1).to(device)

      for n in range(len(alpha) - 1, -1, -1):
        c1 = 1 / alpha[n]**0.5
        c2 = (1 - alpha[n]) / (1 - alpha_cum[n])**0.5
        audio = c1 * (audio - c2 * self.model(audio, spectrogram, noise_scale[n]).squeeze(1))
        if n > 0:
          noise = torch.randn_like(audio)
          sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
          audio += sigma * noise
        audio = torch.clamp(audio, -1.0, 1.0)
    return audio

  def run_valid_loop(self, tqdm_handler=None):
    with torch.no_grad():
      device = next(self.model.parameters()).device
      losses = []
      mel_spec_losses = []
      audio_preds = []
      spec_preds = []
      audio_gt = []
      spec_gts = []
      audio_cond_inst_gt = []
      spec_cond_gt = []
      
      # Create appropriate progress bar for validation
      if tqdm_handler:
        # Use the provided tqdm_handler to create a progress tracking object
        progress_bar = tqdm_handler(len(self.dataset_val), f'Validation')
        use_progress = True
      else:
        # Use standard tqdm only if this is the master process
        if self.is_master:
          progress_bar = tqdm(self.dataset_val, desc=f'Validation')
          use_progress = True
        else:
          progress_bar = self.dataset_val
          use_progress = False
      
      for i, features in enumerate(progress_bar):
        features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)

        audio = features['audio']
        spectrogram = features['spectrogram']
        audio_cond_inst = features['audio_cond_inst']

        N, T = audio.shape
        S = 1000
        device = audio.device
        self.noise_level = self.noise_level.to(device)

        s = torch.randint(1, S + 1, [N], device=audio.device)
        l_a, l_b = self.noise_level[s-1], self.noise_level[s]
        noise_scale = l_a + torch.rand(N, device=audio.device) * (l_b - l_a)
        noise_scale = noise_scale.unsqueeze(1)

        noise = torch.randn_like(audio)
        noisy_audio = noise_scale * audio + (1.0 - noise_scale**2)**0.5 * noise

        predicted = self.model(noisy_audio, spectrogram, noise_scale.squeeze(1))
        loss = self.loss_fn(noise, predicted.squeeze(1))
        losses.append(loss.cpu().numpy())

        audio_pred = self.predict(spectrogram)
        audio_len = len_audio(spectrogram)
        spec_pred = get_spec(audio_pred.squeeze(0)[:audio_len], self.params)
        spec_gt = get_spec(audio.squeeze(0)[:audio_len], self.params)

        if i<5:
          audio_preds.append(audio_pred.squeeze(0).cpu().numpy())
          spec_preds.append(spec_pred.squeeze(0).cpu().numpy())
          audio_gt.append(audio.squeeze(0).cpu().numpy())
          spec_cond_gt.append(spectrogram.squeeze(0).cpu().numpy())
          spec_gts.append(spec_gt.squeeze(0).cpu().numpy())
          audio_cond_inst_gt.append(audio_cond_inst.squeeze(0).cpu().numpy())

        # L1 Mel-Spectrogram Loss
        mel_spec_loss = torch.nn.L1Loss()(spec_pred.squeeze(0), spec_gt.squeeze(0)).item()
        mel_spec_losses.append(mel_spec_loss)
        
        # Update progress bar
        if use_progress and hasattr(progress_bar, 'update'):
          progress_bar.update(1)
      
      # Close progress bar
      if use_progress and hasattr(progress_bar, 'close'):
        progress_bar.close()

      loss_valid = np.mean(losses)
      mel_spec_losses_mean = np.mean(mel_spec_losses)
      self._write_summary_valid(self.step, loss_valid, mel_spec_losses_mean,
                                audio_preds, spec_preds, audio_gt, spec_gts, audio_cond_inst_gt, spec_cond_gt)


def _train_impl(replica_id, model, dataset, dataset_val, args, params, tqdm_handler=None):
  torch.backends.cudnn.benchmark = True
  opt = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

  learner = WaveGradLearner(args.model_dir, args.checkpoint_interval, args.summary_interval,
                            args.validation_interval, model, dataset, dataset_val, opt, params, fp16=args.fp16)
  learner.is_master = (replica_id == 0)
  learner.restore_from_checkpoint()
  learner.train(max_steps=args.max_steps, tqdm_handler=tqdm_handler)


def train(args, params, tqdm_handler=None):
  dataset = dataset_from_path(args.data_dirs, args.training_files, params)
  dataset_val = dataset_from_path_valid(args.data_dirs, args.validation_files, params)
  model = WaveGrad(params).cuda()
  print("Model params: {}".format(sum(p.numel() for p in model.parameters())))
  _train_impl(0, model, dataset, dataset_val, args, params, tqdm_handler=tqdm_handler)


def train_distributed(replica_id, replica_count, port, args, params):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = str(port)
  torch.distributed.init_process_group('nccl', rank=replica_id, world_size=replica_count)

  device = torch.device('cuda', replica_id)
  torch.cuda.set_device(device)
  model = WaveGrad(params).to(device)
  model = DistributedDataParallel(model, device_ids=[replica_id])
  dataset = dataset_from_path(args.data_dirs, args.training_files, params, is_distributed=True)
  if replica_id == 0 and args.validation_files:
    dataset_val = dataset_from_path_valid(args.data_dirs, args.validation_files, params, is_distributed=False)
  else:
    dataset_val = None
  _train_impl(replica_id, model, dataset, dataset_val, args, params)
