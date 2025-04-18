# Adapted from https://github.com/tencent-ailab/bddm under the Apache-2.0 license.

import os
import sys
import json
import shutil
import hashlib
import numpy as np
import yaml

import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

from modules.wavetransfer.bddm import trainer, sampler
from modules.wavetransfer.bddm.log_utils import log


def dict_hash_5char(dictionary):
    ''' Map a unique dictionary into a 5-character string '''
    dhash = hashlib.md5()
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()[:5]


def start_exp(config, config_hash):
    ''' Create experiment directory or set it to an existing directory '''
    if config.load != '' and '_nets' in config.load:
        config.exp_dir = '/'.join(config.load.split('/')[:-2])
    else:
        config.exp_dir += '/%s-%s_conf-hash-%s' % (
            config.score_net, config.schedule_net, config_hash)
    if config.local_rank != 0:
        return
    log('Experiment directory: %s' % (config.exp_dir), config)
    # Backup the config file
    shutil.copyfile(config.config, os.path.join(config.exp_dir, 'conf.yml'))
    # Create a backup scripts sub-folder
    os.makedirs(os.path.join(config.exp_dir, 'backup_scripts'), exist_ok=True)
    # Backup all .py files under bddm/
    backup_files = []
    for root, _, files in os.walk("bddm"):
        if 'egs' in root:
            continue
        for f in files:
            if f.endswith(".py"):
                backup_files.append(os.path.join(root, f))
    for src_file in backup_files:
        basename = src_file.split('/')[-1]
        dst_file = os.path.join(config.exp_dir, 'backup_scripts', basename)
        dst_dir = '/'.join(dst_file.split('/')[:-1])
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        shutil.copyfile(src_file, dst_file)
    # Prepare sub-folders for saving model checkpoints
    os.makedirs(os.path.join(config.exp_dir, 'schedule_nets'), exist_ok=True)


# Namespace-like class for config (to avoid argparse dependencies)
class SimpleNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def setup_config(config_path, local_rank=0, seed=42):
    """Set up configuration from YAML file"""
    with open(config_path) as f:
        yaml_config = yaml.safe_load(f)
    
    config_hash = dict_hash_5char(yaml_config)
    config = yaml_config.copy()
    config['config'] = config_path
    config['local_rank'] = local_rank
    config['seed'] = seed
    
    # Convert dict to namespace without using argparse
    config = SimpleNamespace(**config)
    
    # Set random seed for reproducible results
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.cuda.set_device(config.local_rank)
    
    return config, config_hash


def train_schedule_network(config_path, project_dir=None, local_rank=0):
    """Train the schedule network using configuration from YAML file"""
    config, config_hash = setup_config(config_path, local_rank)
    
    # Override exp_dir if project_dir is provided
    if project_dir:
        config.exp_dir = project_dir
    
    # Create/retrieve exp dir
    start_exp(config, config_hash)
    log(f'Starting training with config: {config_path}', config)
    
    try:
        # Create Trainer for training
        model_trainer = trainer.Trainer(config)
        model_trainer.train()
        log('-' * 80, config)
        return True, config.exp_dir
        
    except Exception as e:
        log('-' * 80, config)
        log(f'Error during training: {str(e)}', config)
        return False, str(e)


def schedule_noise(config_path, project_dir=None, local_rank=0):
    """Run noise scheduling using trained model"""
    config, config_hash = setup_config(config_path, local_rank)
    
    # Override exp_dir if project_dir is provided
    if project_dir:
        config.exp_dir = project_dir
    
    # Create/retrieve exp dir
    start_exp(config, config_hash)
    log(f'Starting noise scheduling with config: {config_path}', config)
    
    try:
        # Create Sampler for noise scheduling
        model_sampler = sampler.Sampler(config)
        model_sampler.noise_scheduling_without_params()
        log('-' * 80, config)
        return True, config.exp_dir
        
    except Exception as e:
        log('-' * 80, config)
        log(f'Error during noise scheduling: {str(e)}', config)
        return False, str(e)


def infer_schedule_network(config_path, project_dir=None, local_rank=0):
    """Generate audio using trained model"""
    config, config_hash = setup_config(config_path, local_rank)
    
    # Override exp_dir if project_dir is provided
    if project_dir:
        config.exp_dir = project_dir
    
    # Create/retrieve exp dir
    start_exp(config, config_hash)
    log(f'Starting generation with config: {config_path}', config)
    
    try:
        # Create Sampler for generation
        model_sampler = sampler.Sampler(config)
        output_files = model_sampler.generate()
        log('-' * 80, config)
        return True, output_files
        
    except Exception as e:
        log('-' * 80, config)
        log(f'Error during generation: {str(e)}', config)
        return False, str(e)


if __name__ == '__main__':
    # Only import argparse when running as a script
    import argparse
    
    parser = argparse.ArgumentParser(description='Bilateral Denoising Diffusion Models')
    parser.add_argument('--command',
                        type=str,
                        default='train',
                        help='available commands: train | schedule | generate')
    parser.add_argument('--config',
                        '-c',
                        type=str,
                        default='conf.yml',
                        help='config .yml path')
    parser.add_argument('--local_rank',
                        type=int,
                        default=0,
                        help='process device ID for multi-GPU training')

    arg_config = parser.parse_args()

    # Parse yaml and define configurations
    config = arg_config.__dict__
    with open(arg_config.config) as f:
        yaml_config = yaml.safe_load(f)
    HASH = dict_hash_5char(yaml_config)
    for key in yaml_config:
        config[key] = yaml_config[key]
    config = argparse.Namespace(**config)

    # Set random seed for reproducible results
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.cuda.set_device(config.local_rank)

    # Check if the command is valid or not
    commands = ['train', 'schedule', 'generate']
    assert config.command in commands, 'Error: %s command not found.'%(config.command)

    # Create/retrieve exp dir
    start_exp(config, HASH)
    log('Argv: %s' % (' '.join(sys.argv)), config)

    try:
        if config.command == 'train':
            # Create Trainer for training
            trainer_obj = trainer.Trainer(config)
            trainer_obj.train()
        elif config.command == 'schedule':
            # Create Sampler for noise scheduling
            sampler_obj = sampler.Sampler(config)
            sampler_obj.noise_scheduling_without_params()
        elif config.command == 'generate':
            # Create Sampler for generation
            sampler_obj = sampler.Sampler(config)
            sampler_obj.generate()
        log('-' * 80, config)

    except KeyboardInterrupt:
        log('-' * 80, config)
        log('Exiting early', config)


