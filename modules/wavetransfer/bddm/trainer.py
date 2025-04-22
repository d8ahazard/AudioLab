# Adapted from https://github.com/tencent-ailab/bddm under the Apache-2.0 license.

#!/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
#  BDDM Trainer
#
#  Author: Max W. Y. Lam (maxwylam@tencent.com)
#  Copyright (c) 2021Tencent. All Rights Reserved
#
########################################################################


from __future__ import absolute_import

import os
import time
import copy
import torch
import torch.nn as nn
from tqdm import tqdm
import yaml

from modules.wavetransfer.bddm.ema import EMAHelper
from modules.wavetransfer.bddm.loss import StepLoss
from modules.wavetransfer.bddm.log_utils import log
from modules.wavetransfer.bddm.diffusion_utils import compute_diffusion_params
from modules.wavetransfer.bddm.models import get_schedule_network

from modules.wavetransfer.model import WaveGrad
from modules.wavetransfer.params import AttrDict, get_default_params
from modules.wavetransfer.bddm.data_loader import from_path, from_path_valid


def _nested_map(struct, map_fn):
  if isinstance(struct, tuple):
    return tuple(_nested_map(x, map_fn) for x in struct)
  if isinstance(struct, list):
    return [_nested_map(x, map_fn) for x in struct]
  if isinstance(struct, dict):
    return { k: _nested_map(v, map_fn) for k, v in struct.items() }
  return map_fn(struct)


# Class to adapt tqdm to Gradio progress
class TqdmToGradioProgress:
    """Adapts tqdm to report progress to a gr.Progress object."""
    def __init__(self, progress_obj, desc=None, total=None):
        self.progress_obj = progress_obj
        self.desc = desc
        self.total = total
        self.n = 0
        self.last_progress = 0
  
    def update(self, n=1):
        self.n += n
        if self.total and self.progress_obj:
            new_progress = min(self.n / self.total, 1.0)
            # Update progress
            try:
                self.progress_obj(new_progress, f"{self.desc}: {self.n}/{self.total}")
                self.last_progress = new_progress
            except Exception as e:
                # Silent exception handling for progress bar errors
                pass
  
    def close(self):
        if self.total and self.progress_obj:
            try:
                self.progress_obj(1.0, f"{self.desc}: Completed")
            except Exception as e:
                # Silent exception handling for progress bar errors
                pass
            
    def set_description(self, desc):
        self.desc = desc
        if self.progress_obj:
            try:
                self.progress_obj(self.last_progress, f"{self.desc}: {self.n}/{self.total}")
            except Exception as e:
                # Silent exception handling for progress bar errors
                pass
                
    # Required for tqdm to use this as a file-like object
    def write(self, s):
        # This is required for tqdm to work
        pass
        
    # Required for tqdm cleanup
    def flush(self):
        # This is required for tqdm to work
        pass



class Trainer(object):

    def __init__(self, config):
        """
        Trainer Constructor

        Parameters:
            config (namespace): BDDM Configuration
        """
        self.config = config
        self.local_rank = config.local_rank
        
        # Set default command if not present (for compatibility with SimpleNamespace objects from UI)
        if not hasattr(config, 'command'):
            config.command = 'train'
            print(f"No command specified, defaulting to 'train'")

        if config.command == 'train' and config.schedule_net == 'GALR':
            self.load = config.bddm_load
            self.model = WaveGrad(AttrDict(get_default_params())).cuda().train()
            
            # Always create a new schedule network for training
            self.model.schedule_net = get_schedule_network(config).cuda().train()
            
            # Initialize the training environment
            # Note: This is a custom change for the schedule network
            print(f"Initializing schedule network for training mode")
            self.training_target = 'schedule_nets'
            self.n_training_steps = config.schedule_net_training_steps
        else:
            # Default for other commands
            print(f"Initializing for command: {config.command}")
            self.load = config.load
            self.training_target = None
            self.n_training_steps = config.schedule_net_training_steps
            self.model = WaveGrad(AttrDict(get_default_params())).cuda().eval()
            
            # Load the schedule network for testing
            if hasattr(self.config, 'sampling_noise_schedule') and self.config.sampling_noise_schedule and config.command == 'generate':
                print(f"Using noise schedule file: {self.config.sampling_noise_schedule}")
            elif config.command != 'train':
                print(f"Initializing default schedule network for non-training mode")
                self.model.schedule_net = get_schedule_network(config).cuda().eval()

        # Prepare EMA (Exponential Moving Average) for training
        self.ema_helper = EMAHelper(mu=config.ema_rate)
        self.ema_helper.register(self.model.schedule_net)
        self.clip = config.grad_clip
        self.exp_dir = config.exp_dir
        torch.autograd.set_detect_anomaly(True)
        # Initialize diffusion parameters using a pre-specified linear schedule
        noise_schedule = torch.linspace(config.beta_0, config.beta_T, config.T).cuda()
        self.diff_params = compute_diffusion_params(noise_schedule)
        if self.training_target == 'schedule_nets':
            self.diff_params["tau"] = config.tau
            for p in self.model.parameters():
                p.requires_grad = False
            self.loss_func = StepLoss(config, self.diff_params)
            # In practice using batch size = 1 would lead to much lower step loss
            config.batch_size = 1
            model_to_train = self.model.schedule_net
        # Define optimizer
        self.optimizer = torch.optim.AdamW(model_to_train.parameters(),
            lr=config.lr, weight_decay=config.weight_decay, amsgrad=True)
        self.device = torch.device("cuda:{}".format(config.local_rank))
        
        # Get Gradio progress object if available
        self.gradio_progress = getattr(config, 'gradio_progress', None)
        
        # Get data loaders with error handling
        try:
            self.tr_loader = from_path(config.data_dir, config.training_file, get_default_params(), config.batch_size, config.n_worker)
            self.vl_loader = from_path_valid(config.data_dir, config.validation_file, get_default_params(), config.n_worker)
            log('Successfully created data loaders', config)
        except Exception as e:
            log(f'Error creating data loaders: {str(e)}', config)
            print(f"Error creating data loaders: {str(e)}")
            
            # Create empty loaders as fallback
            from torch.utils.data import DataLoader, TensorDataset
            dummy_dataset = TensorDataset(torch.zeros(1, 1), torch.zeros(1, 1))
            self.tr_loader = DataLoader(dummy_dataset, batch_size=1)
            self.vl_loader = DataLoader(dummy_dataset, batch_size=1)
            
            # Reraise exception if this is not a UI call (no gradio progress)
            if not self.gradio_progress:
                raise
        
        self.reset()

    def reset(self):
        """
        Reset training environment
        """
        self.tr_loss, self.vl_loss = [], []
        self.training_step = 0
        if self.load != '':
            # First try to load with safetensors
            safetensors_path = self.load
            # Convert .pkl or .pt extension to .safetensors if needed
            if safetensors_path.endswith('.pkl') or safetensors_path.endswith('.pt'):
                safetensors_path = safetensors_path.rsplit('.', 1)[0] + '.safetensors'
                
            try:
                # Try to load with safetensors first
                from safetensors.torch import load_file
                if os.path.exists(safetensors_path):
                    model_state = load_file(safetensors_path, device='cuda')
                    
                    # Fix key prefixes for schedule_net
                    if self.training_target == 'schedule_nets':
                        # Check if keys start with schedule_net or not
                        has_schedule_prefix = any(k.startswith('schedule_net.') for k in model_state.keys())
                        
                        if not has_schedule_prefix:
                            # This is a schedule network-only checkpoint
                            # Load directly into the schedule network
                            schedule_state = {}
                            for k, v in model_state.items():
                                if k.startswith('ratio_nn.'):
                                    schedule_state['schedule_net.' + k] = v
                                else:
                                    schedule_state[k] = v
                            
                            # Initialize the model with schedule_net parameters only
                            self.model.schedule_net.load_state_dict({k.replace('schedule_net.', ''): v 
                                                                    for k, v in schedule_state.items() 
                                                                    if k.startswith('schedule_net.')})
                            log('Loaded schedule network weights with safetensors: %s' % safetensors_path, self.config)
                        else:
                            # This is a combined checkpoint, load normally
                            self.model.load_state_dict(model_state)
                            log('Loaded checkpoint with safetensors: %s' % safetensors_path, self.config)
                    else:
                        # For non-schedule network training, load normally
                        self.model.load_state_dict(model_state)
                        log('Loaded checkpoint with safetensors: %s' % safetensors_path, self.config)
                else:
                    # Fall back to torch.load
                    self._load_torch_checkpoint(self.load)
            except (ImportError, FileNotFoundError, RuntimeError) as e:
                # Fall back to torch.load
                log(f'Error loading safetensors, falling back to torch: {str(e)}', self.config)
                self._load_torch_checkpoint(self.load)
        
        torch.cuda.empty_cache()
        if self.training_target is None:
            self.training_target, self.n_training_steps = (
                'schedule_nets', self.config.schedule_net_training_steps)
        self.prev_val_loss, self.min_val_loss = float("inf"), float("inf")
        self.val_no_impv, self.halving = 0, 0

    def _load_torch_checkpoint(self, checkpoint_path):
        """Helper method to load PyTorch checkpoints with proper key handling"""
        package = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda())
        
        if 'model_state_dict' in package:
            model_state = package['model_state_dict']
        elif 'model' in package:
            model_state = package['model']
        else:
            model_state = package  # Assume the entire package is the state dict
                
        # Fix key prefixes for schedule_net
        if self.training_target == 'schedule_nets':
            # Check if keys start with schedule_net or not
            has_schedule_prefix = any(k.startswith('schedule_net.') for k in model_state.keys())
            
            if not has_schedule_prefix:
                # This is a schedule network-only checkpoint
                # Load directly into the schedule network
                try:
                    self.model.schedule_net.load_state_dict(model_state)
                    log('Loaded schedule network weights directly: %s' % checkpoint_path, self.config)
                except RuntimeError:
                    # Maybe the keys have weird structure, try to fix them
                    schedule_state = {}
                    for k, v in model_state.items():
                        if k.startswith('ratio_nn.'):
                            schedule_state[k] = v
                        
                    self.model.schedule_net.load_state_dict(schedule_state)
                    log('Loaded schedule network weights with key fixing: %s' % checkpoint_path, self.config)
            else:
                # This is a combined checkpoint, extract schedule_net part
                schedule_state = {k.replace('schedule_net.', ''): v 
                                 for k, v in model_state.items() 
                                 if k.startswith('schedule_net.')}
                
                self.model.schedule_net.load_state_dict(schedule_state)
                log('Loaded schedule network from full checkpoint: %s' % checkpoint_path, self.config)
        else:
            # For non-schedule network training, load normally
            self.model.load_state_dict(model_state)
            log('Loaded checkpoint with torch: %s' % checkpoint_path, self.config)

    def train(self):
        """
        Train model
        """
        # Get data loaders and optimizer
        self.reset()
        
        # Set the total steps we want to train
        total_steps = self.n_training_steps
        
        # Set up gradio progress tracking if available
        try:
            self.gradio_progress = self.config.progress_callback
        except Exception as e:
            # Gracefully handle any Gradio progress errors
            print(f"Progress bar error: {str(e)}")
            self.gradio_progress = None
            
        # Main training loop
        while self.training_step < total_steps:
            # Calculate overall progress (as a percentage)
            overall_progress = min(1.0, self.training_step / total_steps)
            progress_percent = int(overall_progress * 100)
            
            try:
                if self.gradio_progress is not None:
                    self.gradio_progress(overall_progress, f"Overall progress: {progress_percent}% - Step {self.training_step}/{total_steps}")
            except Exception as e:
                # Gracefully handle progress bar errors
                print(f"Progress bar error: {str(e)}")
                self.gradio_progress = None
            
            # Train one epoch
            self.model.train()
            tr_avg_loss = self._run_one_epoch(validate=False)
            
            # Start validation
            try:
                if self.gradio_progress is not None:
                    self.gradio_progress(min(0.95, overall_progress + 0.05), 
                        f"Running validation (Overall: {progress_percent}%)")
            except Exception as e:
                # Gracefully handle progress bar errors
                self.gradio_progress = None
                
            self.model.eval()
            with torch.no_grad():
                val_loss = self._run_one_epoch(validate=True)
            
            # Check for improvement
            if val_loss >= self.min_val_loss:
                # LR decays
                self.val_no_impv += 1
                if self.val_no_impv == self.config.patience:
                    print(f"No improvement for {self.config.patience} epochs, early stopped!")
                    break
                if self.val_no_impv >= self.config.patience // 2:
                    self.model.load_state_dict(best_state)
            else:
                self.val_no_impv = 0
                self.min_val_loss = val_loss
                best_state = copy.deepcopy(self.ema_helper.state_dict())
                model_serialized = self.serialize()
                file_path = os.path.join(self.exp_dir, self.training_target,
                                            '%d.safetensors' % self.training_step)
                try:
                    from safetensors.torch import save_file
                    # Convert model state dict to a format suitable for safetensors
                    
                    # If this is a schedule_net-only training, save just those weights
                    if self.training_target == 'schedule_nets':
                        # Extract only schedule network state for saving
                        # When saving schedule-only, don't use the "schedule_net." prefix
                        save_state = model_serialized['model_state_dict']
                        save_file(save_state, file_path)
                        
                        # Also save a metadata file with training step
                        metadata_path = os.path.join(self.exp_dir, self.training_target,
                                                '%d_metadata.pt' % self.training_step)
                        torch.save({
                            'schedule_net_training_step': self.training_step,
                            'config': self.config
                        }, metadata_path)
                    else:
                        # Save full model with original method
                        save_file(model_serialized['model_state_dict'], file_path)
                    
                    print(f"Found better model, saved to {file_path}")
                except ImportError:
                    # Fallback to torch.save if safetensors is not available
                    file_path = os.path.join(self.exp_dir, self.training_target,
                                            '%d.pkl' % self.training_step)
                    torch.save(model_serialized, file_path)
                    print(f"Found better model, saved to {file_path} (using torch.save)")
        
        # Complete progress
        try:
            if self.gradio_progress is not None:
                self.gradio_progress(1.0, "Schedule network training completed!")
        except Exception as e:
            # Gracefully handle progress bar errors
            pass

    def _run_one_epoch(self, validate=False):
        """
        Run one epoch

        Parameters:
            validate (bool):      whether to run a valiation epoch or a training epoch
        Returns:
            average loss (float): the average training/validation loss
        """
        start = time.time()
        total_loss, total_cnt = 0, 0
        data_loader = self.vl_loader if validate else self.tr_loader
        start_step = self.training_step
        
        # Create progress bar
        desc = 'Validation' if validate else 'Training'
        total_batches = len(data_loader)
        epoch_info = f"Step {self.training_step}/{self.n_training_steps}" if not validate else ""
        
        # Create standard tqdm for console output
        progress_bar = tqdm(enumerate(data_loader), desc=f"{desc} {epoch_info}", total=total_batches)
        
        # Set models to appropriate mode
        if validate:
            self.model.eval()
        else:
            self.model.train()
            
        for i, batch in progress_bar:
            if batch == None:
                continue
                
            features = _nested_map(batch, lambda x: x.cuda() if isinstance(x, torch.Tensor) else x)
            
            with torch.set_grad_enabled(not validate):
                loss = self.loss_func(self.model, features)
                total_loss += loss.detach().sum()
                total_cnt += len(loss)
                avg_loss = loss.mean()
                
                # Update progress bar description
                try:
                    progress_bar.set_description(f"{desc} {epoch_info} | Loss: {avg_loss:.5f}")
                    
                    # Update Gradio progress separately (don't use tqdm with custom file)
                    if self.gradio_progress is not None:
                        # Calculate fraction complete for this batch
                        batch_progress = (i + 1) / total_batches
                        self.gradio_progress(batch_progress, f"{desc}: {i+1}/{total_batches} | Loss: {avg_loss:.5f}")
                except Exception:
                    # Silently handle any progress bar issues
                    pass
                
                if not validate:
                    self.optimizer.zero_grad()
                    avg_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                    self.optimizer.step()
                    self.training_step += len(loss)
                    
                    # Don't log training steps to console
                    if self.training_target == 'schedule_nets':
                        # Update EMA for schedule network
                        self.ema_helper.update(self.model.schedule_net)
                        
                    if self.training_step >= self.n_training_steps or\
                            max(i, self.training_step - start_step) >= self.config.steps_per_epoch:
                        # Release grad memory
                        self.optimizer.zero_grad()
                        return total_loss / total_cnt
            
        if not validate:
            # Release grad memory
            self.optimizer.zero_grad()
            
        torch.cuda.empty_cache()
        return total_loss / total_cnt

    def serialize(self):
        """
        Pack the model and configurations into a dictionary

        Returns:
            package (dict): the serialized package to be saved
        """
        # Check if we're training the schedule network only
        if self.training_target == 'schedule_nets':
            # Save just the schedule network parameters
            model_state = copy.deepcopy(self.model.schedule_net.state_dict())
            ema_state = copy.deepcopy(self.ema_helper.state_dict())
            
            # When saving schedule network, don't use the 'schedule_net.' prefix
            # as it'll be loaded directly into the schedule network
            if self.config.save_fp16:
                for p in model_state:
                    model_state[p] = model_state[p].half()
                for p in ema_state:
                    ema_state[p] = ema_state[p].half()
        else:
            # Full model serialization
            model_state = copy.deepcopy(self.model.state_dict())
            ema_state = copy.deepcopy(self.ema_helper.state_dict())
            for p in self.ema_helper.state_dict():
                model_state['schedule_net.'+p] = ema_state[p]
            if self.config.save_fp16:
                for p in model_state:
                    model_state[p] = model_state[p].half()
                    
        package = {
            # state
            'model_state_dict': model_state
        }
        # Since safetensors doesn't support arbitrary Python objects, 
        # we'll need to save config separately when using safetensors
        package['schedule_net_training_step'] = self.training_step
        return package
