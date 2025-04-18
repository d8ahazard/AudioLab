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

from torch.cuda import device_count
from torch.multiprocessing import spawn, set_start_method
import os
import shutil
import traceback
import logging
import yaml
import json
from tqdm.auto import tqdm

from modules.wavetransfer.learner import train, train_distributed
from modules.wavetransfer.params import AttrDict, get_default_params

# Set up logger
logger = logging.getLogger("ADLB.WaveTransfer.Training")

# Set the multiprocessing start method to 'spawn'
# This is safer than 'fork' when running in an application context
try:
    set_start_method('spawn', force=True)
except RuntimeError:
    # Method already set, which is fine
    pass


def _get_free_port():
  import socketserver
  with socketserver.TCPServer(('localhost', 0), None) as s:
    return s.server_address[1]


# Simple Namespace class to avoid argparse dependency
class SimpleNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class TqdmToGradioProgress:
    """
    Adapts tqdm to report progress to a gr.Progress object.
    """
    def __init__(self, progress_obj, desc=None, total=None):
        self.progress_obj = progress_obj
        self.desc = desc
        self.total = total
        self.n = 0
        self.last_progress = 0
  
    def update(self, n=1):
        self.n += n
        if self.total:
            new_progress = min(self.n / self.total, 1.0)
            # Always update progress to ensure UI stays responsive
            self.progress_obj(new_progress, f"{self.desc}: {self.n}/{self.total}")
            self.last_progress = new_progress
  
    def close(self):
        if self.total:
            self.progress_obj(1.0, f"{self.desc}: Completed")


def create_params_from_config(config_data=None):
    """
    Create params object from config data or use defaults
    
    Args:
        config_data: Dictionary with config parameters to override defaults
        
    Returns:
        AttrDict with parameters
    """
    # Start with default parameters
    params = get_default_params()
    
    # Override with provided config data if any
    if config_data:
        params.override(config_data)
    
    # Using 64 as crop_mel_frames value - this means we need ~1.2 seconds of audio
    # per training example, which should be easily available in full songs
    params.crop_mel_frames = 64  # Slightly smaller than default (66)
    
    # Enable better debugging
    print(f"PARAMETER SETTINGS:")
    print(f"- crop_mel_frames: {params.crop_mel_frames}")
    print(f"- hop_samples: {params.hop_samples}")
    print(f"- sample_rate: {params.sample_rate}")
    print(f"- Min audio length needed (seconds): {(params.crop_mel_frames - 1) * params.hop_samples / params.sample_rate}")
    
    return params


def train_model(
    model_dir,
    data_dirs,
    training_files=None,
    validation_files=None,
    checkpoint_interval=None,
    summary_interval=100,
    validation_interval=1000,
    max_steps=None,
    max_epochs=None,  # New parameter for epoch-based training
    fp16=False,
    config_data=None,
    force_single_process=True,  # Always use single process in AudioLab
    progress=None,  # gr.Progress object for updating UI
    cancel_token=None  # Token for cancellation
):
    """
    Train (or resume training) a WaveTransfer model
    
    Args:
        model_dir: directory in which to store model checkpoints and training logs
        data_dirs: list of directories from which to read .wav files for training
        training_files: list of files containing the list of wav samples used for training
        validation_files: list of files containing the list of wav samples used for validation
        checkpoint_interval: interval between model checkpoints
        summary_interval: interval between training summaries
        validation_interval: interval between validations
        max_steps: maximum number of training steps (deprecated, use max_epochs instead)
        max_epochs: maximum number of training epochs
        fp16: whether to use 16-bit floating point operations for training
        config_data: optional dictionary with custom parameter values
        force_single_process: whether to force using a single process for training (always True in AudioLab)
        progress: gr.Progress object for updating training progress in the UI
        cancel_token: object with a 'cancelled' attribute to check for cancellation
        
    Returns:
        Tuple of (success, model_dir or error_message)
    """
    try:
        # Initialize progress if provided
        if progress:
            progress(0.05, "Initializing training...")
            
        # Set environment variable to inform child processes they should use single process mode
        os.environ['USE_SINGLE_PROCESS'] = '1'
            
        # Validate input directories
        for dir_path in data_dirs:
            if not os.path.exists(dir_path):
                error_message = f"Data directory not found: {dir_path}"
                logger.error(error_message)
                return False, error_message
                
        # Verify there are audio files in the directories
        has_audio_files = False
        for dir_path in data_dirs:
            wav_files = [f for f in os.listdir(dir_path) if f.endswith('.wav') or f.endswith('.mp3')]
            if wav_files:
                has_audio_files = True
                break
                
        if not has_audio_files:
            error_message = f"No audio files (.wav or .mp3) found in any of the provided directories: {data_dirs}"
            logger.error(error_message)
            return False, error_message
            
        # Log training start with parameters
        logger.info(f"Starting WaveTransfer training with model_dir={model_dir}, data_dirs={data_dirs}")
        logger.info(f"Training parameters: max_epochs={max_epochs}, checkpoint_interval={checkpoint_interval}, fp16={fp16}")
        
        # Initialize parameters
        params = create_params_from_config(config_data)
        
        # Create a configuration object without using argparse
        args = SimpleNamespace(
            model_dir=model_dir,
            data_dirs=data_dirs,
            training_files=training_files,
            validation_files=validation_files,
            checkpoint_interval=checkpoint_interval,
            summary_interval=summary_interval,
            validation_interval=validation_interval,
            max_steps=None,  # We'll convert epochs to steps
            max_epochs=max_epochs,
            fp16=fp16
        )
        
        # Create model directory and save params
        os.makedirs(args.model_dir, exist_ok=True)
        
        # Try to create the dataset first to catch potential data issues
        try:
            # Try to create the dataset first to catch potential data issues
            from modules.wavetransfer.dataset import from_path as dataset_from_path
            dataset = dataset_from_path(args.data_dirs, args.training_files, params)
            dataset_size = len(dataset)
            
            # Calculate total epochs
            current_epoch = 0
            total_epochs = max_epochs if max_epochs else 10  # Default to 10 epochs if not specified
            
            # If max_steps was provided (legacy), convert to epochs
            if max_steps and not max_epochs:
                total_epochs = max(1, max_steps // dataset_size)
                args.max_epochs = total_epochs
            
            # Calculate steps per epoch
            steps_per_epoch = dataset_size
            
            # Set max_steps based on epochs for backward compatibility
            args.max_steps = total_epochs * steps_per_epoch if args.max_epochs else None
            
            # Load current epoch from config if available
            if os.path.exists(os.path.join(model_dir, 'training_state.json')):
                try:
                    with open(os.path.join(model_dir, 'training_state.json'), 'r') as f:
                        training_state = json.load(f)
                        current_epoch = training_state.get('current_epoch', 0)
                        logger.info(f"Resuming training from epoch {current_epoch}")
                except Exception as e:
                    logger.warning(f"Failed to load training state: {str(e)}")
            
            logger.info(f"Dataset contains {dataset_size} examples. Training for {total_epochs} epochs.")
            
            if progress:
                progress(0.1, f"Dataset initialized with {dataset_size} examples. Starting training...")
                
        except Exception as e:
            error_message = f"Failed to initialize dataset: {str(e)}"
            logger.error(error_message)
            return False, error_message
        
        # Save the initial training state
        training_state = {
            'current_epoch': current_epoch,
            'total_epochs': total_epochs,
            'steps_per_epoch': steps_per_epoch,
            'dataset_size': dataset_size
        }
        with open(os.path.join(model_dir, 'training_state.json'), 'w') as f:
            json.dump(training_state, f, indent=2)
        
        # Set up custom tqdm handler for progress reporting
        if progress:
            # Create a tqdm handler that will be passed to the training function
            def tqdm_handler(total, desc):
                # Directly update the progress bar for better UI responsiveness
                if "Epoch" in desc:
                    try:
                        epoch_num = int(desc.split()[-1])
                        # Calculate progress as percentage of total epochs
                        progress_value = min((epoch_num / total_epochs) * 0.9, 0.9)
                        # Update the UI with a descriptive message
                        progress(progress_value, f"Training epoch {epoch_num} of {total_epochs} ({epoch_num/total_epochs:.0%} complete)")
                    except ValueError:
                        # If we can't parse the epoch number, use a simple message
                        progress(0.5, f"Training in progress: {desc}")
                
                # For validation, show a different message
                elif "Validation" in desc:
                    progress(0.95, f"Running validation...")
                
                # Return an object compatible with tqdm's interface
                return TqdmToGradioProgress(progress, desc=desc, total=total)
                
            progress(0.15, f"Starting training for {total_epochs} epochs...")
        else:
            tqdm_handler = None
            
        # Always use single process mode in AudioLab
        replica_count = device_count()
        if replica_count > 1:
            logger.info(f"Multiple GPUs detected ({replica_count}), but using single process mode as required in AudioLab")
            
        # Call train with the progress handler and cancellation token
        train(args, params, tqdm_handler=tqdm_handler, cancel_token=cancel_token)
        
        # Final progress update
        if progress:
            progress(1.0, "Training completed successfully!")
            
        logger.info(f"WaveTransfer training completed successfully: {model_dir}")
        return True, model_dir
        
    except Exception as e:
        # Capture full traceback
        error_traceback = traceback.format_exc()
        error_message = f"{str(e)}\n\nTraceback:\n{error_traceback}"
        
        # Log the detailed error
        logger.error(f"WaveTransfer training failed: {error_message}")
        
        # Save the error to a log file in the model directory
        try:
            if os.path.exists(model_dir):
                with open(os.path.join(model_dir, 'error_log.txt'), 'w') as f:
                    f.write(error_message)
        except Exception as log_error:
            logger.error(f"Failed to write error log: {log_error}")
        
        return False, error_message


if __name__ == '__main__':
    # This block won't be used in AudioLab context
    pass

