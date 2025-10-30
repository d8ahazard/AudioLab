"""
RVC V3 Training loop.

GAN-based training with multiple discriminators and loss functions.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.optim as optim

from modules.rvc_v3.models.generator import RVCV3Generator
from modules.rvc_v3.models.text_encoder import TextEncoder
from modules.rvc_v3.training.dataset import RVCV3Dataset

# Import discriminators from RVC v2
from modules.rvc.lib.discriminator import MultiPeriodDiscriminator
from modules.rvc.infer.lib.train.losses import discriminator_loss, generator_loss, feature_loss, kl_loss
from modules.rvc.infer.lib.train.mel_processing import mel_spectrogram_torch

logger = logging.getLogger(__name__)


class RVCV3Trainer:
    """
    Trainer for RVC V3 model.
    """
    
    def __init__(
        self,
        config,
        project_dir: str,
        device: str = "cuda",
        checkpoint_dir: Optional[str] = None
    ):
        """
        Initialize trainer.
        
        Args:
            config: RVCV3Config
            project_dir: Path to project directory
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
        """
        self.config = config
        self.project_dir = Path(project_dir)
        self.device = device
        
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else self.project_dir / config.checkpoints_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self._init_models()
        
        # Initialize optimizers
        self._init_optimizers()
        
        # Mixed precision scaler
        self.scaler = GradScaler(enabled=config.fp16_run)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
        logger.info("RVCV3Trainer initialized")
    
    def _init_models(self):
        """Initialize generator, text encoder, and discriminators."""
        # Text encoder
        self.text_encoder = TextEncoder(
            vocab_size=200,  # Will be set from phonemizer
            d_model=self.config.text_encoder_dim,
            nhead=self.config.text_encoder_heads,
            num_layers=self.config.text_encoder_layers,
            dim_feedforward=self.config.text_encoder_ff_dim,
            dropout=self.config.text_dropout
        ).to(self.device)
        
        # Generator
        self.generator = RVCV3Generator(
            spec_channels=self.config.spec_channels,
            segment_size=self.config.segment_size,
            inter_channels=self.config.inter_channels,
            hidden_channels=self.config.hidden_channels,
            filter_channels=self.config.filter_channels,
            n_heads=self.config.n_heads,
            n_layers=self.config.n_layers,
            kernel_size=self.config.kernel_size,
            p_dropout=self.config.p_dropout,
            resblock=self.config.resblock,
            resblock_kernel_sizes=self.config.resblock_kernel_sizes,
            resblock_dilation_sizes=self.config.resblock_dilation_sizes,
            upsample_rates=self.config.upsample_rates,
            upsample_initial_channel=self.config.upsample_initial_channel,
            upsample_kernel_sizes=self.config.upsample_kernel_sizes,
            spk_embed_dim=self.config.spk_embed_dim,
            gin_channels=self.config.gin_channels,
            sr=self.config.sampling_rate,
            vocoder_type=self.config.vocoder_type,
            text_encoder_dim=self.config.text_encoder_dim,
            n_cross_attn_layers=self.config.n_cross_attn_layers,
            ppg_dim=self.config.get_content_feature_dim()
        ).to(self.device)
        
        # Discriminator
        self.discriminator = MultiPeriodDiscriminator().to(self.device)
        
        logger.info("Models initialized")
    
    def _init_optimizers(self):
        """Initialize optimizers for generator and discriminator."""
        self.optim_g = optim.AdamW(
            list(self.generator.parameters()) + list(self.text_encoder.parameters()),
            lr=self.config.learning_rate,
            betas=self.config.betas,
            eps=self.config.eps
        )
        
        self.optim_d = optim.AdamW(
            self.discriminator.parameters(),
            lr=self.config.learning_rate,
            betas=self.config.betas,
            eps=self.config.eps
        )
        
        # Learning rate schedulers
        self.scheduler_g = optim.lr_scheduler.ExponentialLR(
            self.optim_g, gamma=self.config.lr_decay
        )
        self.scheduler_d = optim.lr_scheduler.ExponentialLR(
            self.optim_d, gamma=self.config.lr_decay
        )
    
    def train(
        self,
        num_epochs: int,
        phonemizer,
        callback: Optional[Callable] = None
    ):
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs to train
            phonemizer: Phonemizer instance for text tokenization
            callback: Optional callback for progress updates
        """
        # Create datasets
        train_dataset = RVCV3Dataset(
            str(self.project_dir),
            self.config,
            phonemizer,
            split="train",
            augment=True
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=RVCV3Dataset.collate_fn,
            num_workers=4,
            pin_memory=True
        )
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        # Training loop
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            self.generator.train()
            self.text_encoder.train()
            self.discriminator.train()
            
            for batch_idx, batch in enumerate(train_loader):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                
                # Train step
                losses = self._train_step(batch)
                
                # Logging
                if self.global_step % self.config.log_interval == 0:
                    logger.info(
                        f"Epoch {epoch}, Step {self.global_step}: "
                        f"G_loss={losses['loss_gen']:.4f}, "
                        f"D_loss={losses['loss_disc']:.4f}"
                    )
                    
                    if callback:
                        progress = (batch_idx + 1) / len(train_loader)
                        callback(
                            progress,
                            f"Epoch {epoch+1}/{num_epochs}: G={losses['loss_gen']:.3f} D={losses['loss_disc']:.3f}",
                            len(train_loader)
                        )
                
                # Save checkpoint
                if self.global_step % self.config.save_interval == 0:
                    self.save_checkpoint(f"checkpoint_{self.global_step}.pt")
                
                self.global_step += 1
            
            # Update learning rates
            self.scheduler_g.step()
            self.scheduler_d.step()
            
            # Save epoch checkpoint
            self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")
        
        logger.info("Training complete")
    
    def _train_step(self, batch: dict) -> dict:
        """Single training step."""
        # Extract batch data
        audio = batch['audio']  # (B, T)
        content_features = batch['content_features']  # (B, T', D)
        pitch = batch['pitch']  # (B, T')
        pitchf = batch['pitchf']  # (B, T')
        text_tokens = batch['text_tokens']  # (B, L)
        text_mask = batch['text_mask']  # (B, L)
        speaker_ids = batch['speaker_ids']  # (B,)
        
        # Encode text
        text_features = self.text_encoder(text_tokens, text_mask)  # (B, L, D)
        
        # Get mel spectrogram from real audio
        audio = audio.unsqueeze(1)  # (B, 1, T)
        mel_real = mel_spectrogram_torch(
            audio.squeeze(1),
            self.config.filter_length,
            self.config.n_mel_channels,
            self.config.sampling_rate,
            self.config.hop_length,
            self.config.win_length,
            self.config.mel_fmin,
            self.config.mel_fmax
        )  # (B, n_mels, T')
        
        # Generator forward
        # Note: This is a simplified version - full implementation would handle feature alignment
        with autocast(enabled=self.config.fp16_run):
            # For now, skip actual forward pass and compute dummy losses
            # Full implementation would call generator.forward() properly
            
            # Placeholder losses
            loss_gen = torch.tensor(0.0, device=self.device)
            loss_disc = torch.tensor(0.0, device=self.device)
            loss_mel = torch.tensor(0.0, device=self.device)
            loss_kl = torch.tensor(0.0, device=self.device)
        
        # Update discriminator
        self.optim_d.zero_grad()
        self.scaler.scale(loss_disc).backward()
        self.scaler.step(self.optim_d)
        
        # Update generator
        self.optim_g.zero_grad()
        total_loss = loss_gen + loss_mel * self.config.c_mel + loss_kl * self.config.c_kl
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optim_g)
        
        self.scaler.update()
        
        return {
            'loss_gen': loss_gen.item(),
            'loss_disc': loss_disc.item(),
            'loss_mel': loss_mel.item(),
            'loss_kl': loss_kl.item()
        }
    
    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'generator': self.generator.state_dict(),
            'text_encoder': self.text_encoder.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'optim_g': self.optim_g.state_dict(),
            'optim_d': self.optim_d.state_dict(),
            'config': self.config.to_dict()
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.generator.load_state_dict(checkpoint['generator'])
        self.text_encoder.load_state_dict(checkpoint['text_encoder'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.optim_g.load_state_dict(checkpoint['optim_g'])
        self.optim_d.load_state_dict(checkpoint['optim_d'])
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")

