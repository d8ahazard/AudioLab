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

from modules.wavetransfer.learner import train, train_distributed
from modules.wavetransfer.params import AttrDict, get_default_params

# Set up logger
logger = logging.getLogger("ADLB.WaveTransfer.Training")

# Try to set the multiprocessing start method to 'spawn'
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
    fp16=False,
    config_data=None,
    force_single_process=False
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
        max_steps: maximum number of training steps
        fp16: whether to use 16-bit floating point operations for training
        config_data: optional dictionary with custom parameter values
        force_single_process: whether to force using a single process for training (for use within GUI apps)
        
    Returns:
        Tuple of (success, model_dir or error_message)
    """
    try:
        # Detect if we're running in an application context
        in_application = (
            hasattr(logging.getLogger(), 'app_context') or 
            'ADLB.' in logging.Logger.manager.loggerDict or
            any(name for name in logging.Logger.manager.loggerDict.keys() if 'app' in name.lower()) or
            os.environ.get('USE_SINGLE_PROCESS') == '1'
        )
        
        # If we're in an application and force_single_process isn't set, warn and force it
        if in_application and not force_single_process:
            logger.warning("Detected application context but force_single_process=False. Setting to True to prevent spawning issues.")
            force_single_process = True
        
        # Set environment variable to inform child processes they should use single process mode
        if force_single_process:
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
        logger.info(f"Training parameters: max_steps={max_steps}, checkpoint_interval={checkpoint_interval}, fp16={fp16}, force_single_process={force_single_process}")
        
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
            max_steps=max_steps,
            fp16=fp16
        )
        
        # Create model directory and save params
        os.makedirs(args.model_dir, exist_ok=True)
        
        # Save params as a proper config file rather than copying the module
        params_file = os.path.join(args.model_dir, 'params_config.json')
        with open(params_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            params_dict = {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in params.items()}
            json.dump(params_dict, f, indent=2)
        
        try:
            # Try to create the dataset first to catch potential data issues
            from modules.wavetransfer.dataset import from_path as dataset_from_path
            _ = dataset_from_path(args.data_dirs, args.training_files, params)
            logger.info("Dataset successfully initialized. Starting training...")
        except Exception as e:
            error_message = f"Failed to initialize dataset: {str(e)}"
            logger.error(error_message)
            return False, error_message
        
        # Always use single process for any in-app training
        if in_application:
            force_single_process = True
            logger.info("Running in application context, forcing single process mode")
        
        # Check if multi-GPU training is possible and desired
        replica_count = device_count()
        if replica_count > 1 and not force_single_process:
            if params.batch_size % replica_count != 0:
                logger.error(f'Batch size {params.batch_size} is not evenly divisible by # GPUs {replica_count}.')
                logger.info(f'Adjusting batch size from {params.batch_size} to {params.batch_size - (params.batch_size % replica_count)}')
                # Adjust batch size to be divisible by replica count
                params.batch_size = params.batch_size - (params.batch_size % replica_count)
                if params.batch_size == 0:
                    params.batch_size = replica_count
                    logger.info(f'Batch size set to minimum value: {params.batch_size}')
            
            params.batch_size = params.batch_size // replica_count
            port = _get_free_port()
            
            # Use distributed training with spawning
            logger.info(f"Using distributed training across {replica_count} GPUs")
            spawn(train_distributed, args=(replica_count, port, args, params), nprocs=replica_count, join=True)
        else:
            # Use simple single-process training for GUI apps or when forced
            if force_single_process and replica_count > 1:
                logger.info(f"Multiple GPUs detected ({replica_count}), but using single process mode due to force_single_process=True")
                logger.info("Multi-GPU capabilities will not be utilized")
            train(args, params)
        
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
    # Only import argparse when running as script
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description='train (or resume training) a WaveGrad model')
    parser.add_argument('--model_dir',
        help='directory in which to store model checkpoints and training logs')
    parser.add_argument('--data_dirs', nargs='+',
        help='space separated list of directories from which to read .wav files for training')
    parser.add_argument('--training_files', nargs='+', default=None,
        help='space separated list of files containing the list of wav samples used for training')
    parser.add_argument('--validation_files', nargs='+', default=None,
        help='space separated list of files containing the list of wav samples used for validation')
    parser.add_argument('--checkpoint_interval', default=None, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)
    parser.add_argument('--max_steps', default=None, type=int,
        help='maximum number of training steps')
    parser.add_argument('--fp16', action='store_true', default=False,
        help='use 16-bit floating point operations for training')
    parser.add_argument('--config', type=str, default=None,
        help='path to config JSON file with custom parameters')
    parser.add_argument('--force_single_process', action='store_true', default=False,
        help='force using a single process for training (for use within GUI apps)')
    
    args = parser.parse_args()
    
    # Load custom config if provided
    config_data = None
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_data = json.load(f)
    
    train_model(
        args.model_dir,
        args.data_dirs,
        args.training_files,
        args.validation_files,
        args.checkpoint_interval,
        args.summary_interval,
        args.validation_interval,
        args.max_steps,
        args.fp16,
        config_data,
        args.force_single_process
    )

