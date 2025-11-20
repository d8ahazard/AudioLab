"""
Wrapper for RVC V3 training to integrate with the UI.

Adapts v2-style hparams to v3 config and wraps the RVCV3Trainer.
"""

import logging
import os
from pathlib import Path

import gradio as gr
import torch

from handlers.config import output_path, model_path
from modules.rvc_v3.configs.v3_config import RVCV3Config, get_default_config
from modules.rvc_v3.training.train import RVCV3Trainer
from modules.rvc_v3.data_prep.phonemizer import Phonemizer

logger = logging.getLogger(__name__)


def hparams_to_v3_config(hparams) -> RVCV3Config:
    """
    Convert v2-style hparams to v3 config.
    
    Args:
        hparams: V2 HParams object
        
    Returns:
        RVCV3Config instance
    """
    # Get default config for the sample rate
    if hasattr(hparams, 'sample_rate'):
        sr_str = hparams.sample_rate
        if sr_str == "48k":
            sample_rate = 48000
        elif sr_str == "40k":
            sample_rate = 40000
        elif sr_str == "32k":
            sample_rate = 32000
        else:
            sample_rate = 48000
    else:
        sample_rate = 48000
    
    config = get_default_config(sample_rate)
    
    # Map v2 hparams to v3 config
    if hasattr(hparams, 'train') and hasattr(hparams.train, 'learning_rate'):
        config.learning_rate = hparams.train.learning_rate
    elif hasattr(hparams, 'learning_rate'):
        config.learning_rate = hparams.learning_rate
    
    if hasattr(hparams, 'train') and hasattr(hparams.train, 'batch_size'):
        config.batch_size = hparams.train.batch_size
    elif hasattr(hparams, 'batch_size'):
        config.batch_size = hparams.batch_size
    
    if hasattr(hparams, 'train') and hasattr(hparams.train, 'epochs'):
        config.epochs = hparams.train.epochs
    elif hasattr(hparams, 'total_epoch'):
        config.epochs = hparams.total_epoch
    
    # Audio/model settings from hparams
    if hasattr(hparams, 'data'):
        if hasattr(hparams.data, 'sampling_rate'):
            config.sampling_rate = hparams.data.sampling_rate
        if hasattr(hparams.data, 'filter_length'):
            config.filter_length = hparams.data.filter_length
        if hasattr(hparams.data, 'hop_length'):
            config.hop_length = hparams.data.hop_length
        if hasattr(hparams.data, 'win_length'):
            config.win_length = hparams.data.win_length
        if hasattr(hparams.data, 'n_mel_channels'):
            config.n_mel_channels = hparams.data.n_mel_channels
            config.spec_channels = hparams.data.n_mel_channels
        if hasattr(hparams.data, 'mel_fmin'):
            config.mel_fmin = hparams.data.mel_fmin
        if hasattr(hparams.data, 'mel_fmax') and hparams.data.mel_fmax is not None:
            config.mel_fmax = hparams.data.mel_fmax
    
    # Model architecture settings
    if hasattr(hparams, 'model'):
        if hasattr(hparams.model, 'inter_channels'):
            config.inter_channels = hparams.model.inter_channels
        if hasattr(hparams.model, 'hidden_channels'):
            config.hidden_channels = hparams.model.hidden_channels
        if hasattr(hparams.model, 'filter_channels'):
            config.filter_channels = hparams.model.filter_channels
        if hasattr(hparams.model, 'n_heads'):
            config.n_heads = hparams.model.n_heads
        if hasattr(hparams.model, 'n_layers'):
            config.n_layers = hparams.model.n_layers
        if hasattr(hparams.model, 'kernel_size'):
            config.kernel_size = hparams.model.kernel_size
        if hasattr(hparams.model, 'p_dropout'):
            config.p_dropout = hparams.model.p_dropout
        if hasattr(hparams.model, 'resblock'):
            config.resblock = hparams.model.resblock
        if hasattr(hparams.model, 'resblock_kernel_sizes'):
            config.resblock_kernel_sizes = hparams.model.resblock_kernel_sizes
        if hasattr(hparams.model, 'resblock_dilation_sizes'):
            config.resblock_dilation_sizes = hparams.model.resblock_dilation_sizes
        if hasattr(hparams.model, 'upsample_rates'):
            config.upsample_rates = hparams.model.upsample_rates
        if hasattr(hparams.model, 'upsample_initial_channel'):
            config.upsample_initial_channel = hparams.model.upsample_initial_channel
        if hasattr(hparams.model, 'upsample_kernel_sizes'):
            config.upsample_kernel_sizes = hparams.model.upsample_kernel_sizes
        if hasattr(hparams.model, 'gin_channels'):
            config.gin_channels = hparams.model.gin_channels
        if hasattr(hparams.model, 'spk_embed_dim'):
            config.spk_embed_dim = hparams.model.spk_embed_dim
    
    # Training settings
    if hasattr(hparams, 'train'):
        if hasattr(hparams.train, 'segment_size'):
            config.segment_size = hparams.train.segment_size
        if hasattr(hparams.train, 'lr_decay'):
            config.lr_decay = hparams.train.lr_decay
        if hasattr(hparams.train, 'betas'):
            config.betas = hparams.train.betas
        if hasattr(hparams.train, 'eps'):
            config.eps = hparams.train.eps
        if hasattr(hparams.train, 'fp16_run'):
            config.fp16_run = hparams.train.fp16_run
        if hasattr(hparams.train, 'c_mel'):
            config.c_mel = hparams.train.c_mel
        if hasattr(hparams.train, 'c_kl'):
            config.c_kl = hparams.train.c_kl
    
    # V3-specific: Use single HuBERT encoder by default
    config.use_dual_encoder = False
    config.hubert_dim = 768
    
    # Vocoder type (default to bigvgan for v3)
    config.vocoder_type = 'bigvgan'
    
    return config


def train_rvc_v3(hparams, progress: gr.Progress = None):
    """
    Train RVC V3 model using hparams from UI.
    
    Args:
        hparams: V2-style HParams from UI
        progress: Gradio progress callback
    """
    logger.info("Starting RVC V3 training")
    
    # Convert hparams to v3 config
    config = hparams_to_v3_config(hparams)
    
    # Get project directory
    project_name = hparams.name
    project_dir = hparams.model_dir
    
    logger.info(f"Project: {project_name}")
    logger.info(f"Project directory: {project_dir}")
    logger.info(f"Sample rate: {config.sampling_rate}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Epochs: {config.epochs}")
    
    # Save v3 config to project directory
    config_save_path = os.path.join(project_dir, "config_v3.json")
    config.save(config_save_path)
    logger.info(f"V3 config saved to {config_save_path}")
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Initialize trainer
    trainer = RVCV3Trainer(
        config=config,
        project_dir=project_dir,
        device=device,
        checkpoint_dir=os.path.join(project_dir, "checkpoints")
    )
    
    # Load pretrained weights if available
    if hasattr(hparams, 'pretrainG') and hparams.pretrainG and hparams.pretrainG != "":
        pretrain_g = hparams.pretrainG
        pretrain_d = hparams.pretrainD if hasattr(hparams, 'pretrainD') and hparams.pretrainD else ""
        
        if pretrain_d and pretrain_d != "":
            logger.info(f"Loading pretrained weights: G={pretrain_g}, D={pretrain_d}")
            
            # Get sample rate for expansion
            if hasattr(hparams, 'sample_rate'):
                sr_str = hparams.sample_rate
                if sr_str == "48k":
                    sample_rate = 48000
                elif sr_str == "40k":
                    sample_rate = 40000
                elif sr_str == "32k":
                    sample_rate = 32000
                else:
                    sample_rate = 48000
            else:
                sample_rate = config.sampling_rate
            
            # Get if_f0
            if_f0 = hparams.if_f0 == 1 if hasattr(hparams, 'if_f0') else True
            
            try:
                trainer.load_pretrained(pretrain_g, pretrain_d, sample_rate=sample_rate, if_f0=if_f0)
            except Exception as e:
                logger.warning(f"Failed to load pretrained weights: {e}")
                logger.warning("Training from scratch")
        else:
            logger.warning("Discriminator path not provided, skipping pretrained weights")
    else:
        logger.info("No pretrained weights specified, training from scratch")
    
    # Initialize phonemizer for text encoding
    # For now, use simple character-level tokenization
    # TODO: Replace with proper phonemizer when available
    phonemizer = SimplePhonemizer()
    
    # Create progress callback wrapper
    def progress_callback(prog, message, total_steps=None):
        if progress is not None:
            progress(prog, message)
        logger.info(f"Progress: {prog*100:.1f}% - {message}")
    
    # Start training
    try:
        trainer.train(
            num_epochs=config.epochs,
            phonemizer=phonemizer,
            callback=progress_callback if progress else None
        )
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
    
    # After training, save the final model in v2-compatible format for inference
    final_model_path = os.path.join(model_path, "trained", f"{project_name}.pth")
    logger.info(f"Saving final model to {final_model_path}")
    
    # Save in format compatible with inference
    checkpoint = {
        'model': trainer.generator.state_dict(),
        'text_encoder': trainer.text_encoder.state_dict(),
        'config': config.to_dict(),
        'version': 'v3',
        'iteration': trainer.global_step,
        'learning_rate': config.learning_rate
    }
    torch.save(checkpoint, final_model_path)
    
    logger.info("RVC V3 training complete")


class SimplePhonemizer:
    """
    Simple character-level phonemizer for testing.
    
    TODO: Replace with proper phonemizer (espeak-ng or similar)
    """
    
    def __init__(self):
        self.vocab = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?'\"")
        self.char_to_id = {c: i for i, c in enumerate(self.vocab)}
        self.id_to_char = {i: c for i, c in enumerate(self.vocab)}
    
    def phonemize(self, text: str):
        """Convert text to phoneme IDs."""
        # Simple character-level tokenization
        ids = []
        for char in text:
            if char in self.char_to_id:
                ids.append(self.char_to_id[char])
            else:
                ids.append(0)  # Unknown character
        return ids
    
    @property
    def vocab_size(self):
        return len(self.vocab)

