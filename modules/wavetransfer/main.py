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
from torch.multiprocessing import spawn
import os
import shutil

from modules.wavetransfer.learner import train, train_distributed
from modules.wavetransfer.params import params


def _get_free_port():
  import socketserver
  with socketserver.TCPServer(('localhost', 0), None) as s:
    return s.server_address[1]


# Simple Namespace class to avoid argparse dependency
class SimpleNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def train_model(
    model_dir,
    data_dirs,
    training_files=None,
    validation_files=None,
    checkpoint_interval=None,
    summary_interval=100,
    validation_interval=1000,
    max_steps=None,
    fp16=False
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
        
    Returns:
        Tuple of (success, model_dir or error_message)
    """
    try:
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
        params_file = os.path.join(args.model_dir, 'params_saved.py')
        if not os.path.exists(params_file):
            shutil.copy('modules/wavetransfer/params.py', params_file)
        
        # Check if multi-GPU training is possible and desired
        replica_count = device_count()
        if replica_count > 1:
            if params.batch_size % replica_count != 0:
                raise ValueError(f'Batch size {params.batch_size} is not evenly divisible by # GPUs {replica_count}.')
            params.batch_size = params.batch_size // replica_count
            port = _get_free_port()
            spawn(train_distributed, args=(replica_count, port, args, params), nprocs=replica_count, join=True)
        else:
            train(args, params)
        
        return True, model_dir
        
    except Exception as e:
        return False, str(e)


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
    
    args = parser.parse_args()
    train_model(
        args.model_dir,
        args.data_dirs,
        args.training_files,
        args.validation_files,
        args.checkpoint_interval,
        args.summary_interval,
        args.validation_interval,
        args.max_steps,
        args.fp16
    )

