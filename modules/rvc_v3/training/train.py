"""
RVC V3 Training loop.

GAN-based training with multiple discriminators and loss functions.
"""

import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import torch.optim as optim

from modules.rvc_v3.io.checkpoint_io import (
    load_inference_model,
    load_pretrained_d,
    load_training_checkpoint,
    save_training_checkpoint,
)
from modules.rvc_v3.models.generator import RVCV3Generator
from modules.rvc_v3.models.text_encoder import TextEncoder
from modules.rvc_v3.training.dataset import RVCV3Dataset

# Import discriminators from RVC v2
from modules.rvc.lib.discriminator import MultiPeriodDiscriminator
from modules.rvc.infer.lib.train.losses import discriminator_loss, generator_loss, feature_loss, kl_loss
from modules.rvc.infer.lib.train.mel_processing import mel_spectrogram_torch, spec_to_mel_torch, spectrogram_torch
from modules.rvc.lib import commons as rvc_commons

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
        self.debug_stall = os.environ.get("SMOKE_DEBUG_STALL", "0").strip().lower() in ("1", "true", "yes")
        self.debug_sync = os.environ.get("SMOKE_DEBUG_CUDA_SYNC", "0").strip().lower() in ("1", "true", "yes")
        self.last_train_stage = "init"
        
        logger.info("RVCV3Trainer initialized")

    def _mark_train_stage(self, stage: str) -> None:
        self.last_train_stage = stage
        if self.debug_stall:
            logger.info("[stall-debug] _train_step stage=%s", stage)
        if self.debug_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
    
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
            # RVC models expect segment_size in *frames* (not samples).
            # The config stores segment_size in samples to match existing RVC configs.
            segment_size=self.config.segment_size // self.config.hop_length,
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
        """
        Single training step.
        
        Supports two batch formats:
        - **RVC v2-style**: phone/pitch/spec/wave/sid fields (preferred; matches existing preprocess pipeline)
        - **V3-native**: audio/content_features/pitch/pitchf/text_tokens/text_mask/speaker_ids
        """
        # -------------------------
        # Unify batch into v2-style
        # -------------------------
        self._mark_train_stage("unify_batch")
        if "spec" in batch and "wave" in batch and "phone" in batch:
            phone = batch["phone"]
            phone_lengths = batch["phone_lengths"]
            pitch = batch.get("pitch", None)
            pitchf = batch.get("pitchf", None)
            spec = batch["spec"]
            spec_lengths = batch["spec_lengths"]
            wave = batch["wave"]
            sid = batch["sid"]
            text_tokens = batch.get("text_tokens", None)
            text_mask = batch.get("text_mask", None)
        else:
            # V3-native fallback (used if we run the V3 dataset directly)
            audio = batch["audio"]  # (B, T)
            phone = batch["content_features"]  # (B, T', D)
            pitch = batch.get("pitch", None)  # (B, T')
            pitchf = batch.get("pitchf", None)  # (B, T')
            text_tokens = batch.get("text_tokens", None)  # (B, L)
            text_mask = batch.get("text_mask", None)  # (B, L)
            sid = batch.get("speaker_ids", None)  # (B,)

            if sid is None:
                sid = torch.zeros(audio.size(0), device=audio.device, dtype=torch.long)

            # Wave expected shape: (B, 1, T)
            wave = audio.unsqueeze(1)

            # Spec expected shape: (B, F, T_spec)
            with torch.no_grad():
                spec = spectrogram_torch(
                    wave.squeeze(1),
                    self.config.filter_length,
                    self.config.sampling_rate,
                    self.config.hop_length,
                    self.config.win_length,
                    center=False,
                )
            spec_lengths = torch.full(
                (spec.size(0),), spec.size(-1), device=spec.device, dtype=torch.long
            )

            # Lengths
            phone_lengths = torch.full(
                (phone.size(0),), phone.size(1), device=phone.device, dtype=torch.long
            )

            # Ensure pitch dtypes match RVC expectations
            if pitch is not None and pitch.dtype != torch.long:
                pitch = pitch.long()
            if pitchf is not None and pitchf.dtype != torch.float32:
                pitchf = pitchf.float()

            # Align phone/spec/wave lengths like v2 loader does
            len_phone = phone.size(1)
            len_spec = spec.size(-1)
            if len_phone != len_spec:
                len_min = min(len_phone, len_spec)
                len_wav = len_min * self.config.hop_length
                phone = phone[:, :len_min, :]
                spec = spec[:, :, :len_min]
                wave = wave[:, :, :len_wav]
                phone_lengths = torch.clamp(phone_lengths, max=len_min)
                spec_lengths = torch.clamp(spec_lengths, max=len_min)
                if pitch is not None:
                    pitch = pitch[:, :len_min]
                if pitchf is not None:
                    pitchf = pitchf[:, :len_min]

        # -------------------------
        # Optional text conditioning
        # -------------------------
        self._mark_train_stage("text_conditioning")
        text_features = None
        if text_tokens is not None and text_mask is not None and not bool(torch.all(text_mask)):
            text_features = self.text_encoder(text_tokens, text_mask)  # (B, L, d_model)
        else:
            text_mask = None

        # -------------------------
        # Generator forward
        # -------------------------
        self._mark_train_stage("generator_forward")
        with autocast("cuda", enabled=self.config.fp16_run):
            (y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q)) = self.generator(
                phone,
                phone_lengths,
                pitch,
                pitchf,
                spec,
                spec_lengths,
                sid,
                text_features=text_features,
                text_mask=text_mask,
            )

            # Convert real spec -> mel
            self._mark_train_stage("mel_from_spec")
            mel = spec_to_mel_torch(
                spec,
                self.config.filter_length,
                self.config.n_mel_channels,
                self.config.sampling_rate,
                self.config.mel_fmin,
                self.config.mel_fmax,
            )
            y_mel = rvc_commons.slice_segments(
                mel, ids_slice, self.config.segment_size // self.config.hop_length
            )

            # Pred mel from waveform (disable autocast for stability)
            self._mark_train_stage("mel_from_y_hat")
            with autocast("cuda", enabled=False):
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.float().squeeze(1),
                    self.config.filter_length,
                    self.config.n_mel_channels,
                    self.config.sampling_rate,
                    self.config.hop_length,
                    self.config.win_length,
                    self.config.mel_fmin,
                    self.config.mel_fmax,
                )

            # Slice real waveform to match ids_slice
            self._mark_train_stage("wave_slice")
            wave_slice = rvc_commons.slice_segments(
                wave, ids_slice * self.config.hop_length, self.config.segment_size
            )

            # -------------------------
            # Discriminator update
            # -------------------------
            self._mark_train_stage("disc_forward_detach")
            y_d_hat_r, y_d_hat_g, _, _ = self.discriminator(wave_slice, y_hat.detach())
            with autocast("cuda", enabled=False):
                self._mark_train_stage("disc_loss")
                loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)

        self._mark_train_stage("disc_backward")
        self.optim_d.zero_grad(set_to_none=True)
        self.scaler.scale(loss_disc).backward()
        self._mark_train_stage("disc_step")
        self.scaler.step(self.optim_d)

        # -------------------------
        # Generator update
        # -------------------------
        self._mark_train_stage("gen_disc_forward")
        with autocast("cuda", enabled=self.config.fp16_run):
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.discriminator(wave_slice, y_hat)
            with autocast("cuda", enabled=False):
                self._mark_train_stage("gen_losses")
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * float(self.config.c_mel)
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * float(self.config.c_kl)
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, _ = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl

        self._mark_train_stage("gen_backward")
        self.optim_g.zero_grad(set_to_none=True)
        self.scaler.scale(loss_gen_all).backward()
        self._mark_train_stage("gen_step")
        self.scaler.step(self.optim_g)
        self._mark_train_stage("scaler_update")
        self.scaler.update()
        self._mark_train_stage("loss_itemize")

        return {
            "loss_gen": float(loss_gen.detach().item()),
            "loss_disc": float(loss_disc.detach().item()),
            "loss_mel": float(loss_mel.detach().item()),
            "loss_fm": float(loss_fm.detach().item()),
            "loss_kl": float(loss_kl.detach().item()),
            "loss_total_g": float(loss_gen_all.detach().item()),
        }

    def _prune_epoch_checkpoints(self, keep_last: int = 2) -> list:
        """Remove old epoch checkpoints, keeping only the most recent keep_last."""
        pattern = re.compile(r"checkpoint_epoch_(\d+)\.(?:safetensors|pt)$")
        candidates = []
        for p in self.checkpoint_dir.iterdir():
            m = pattern.match(p.name)
            if m and p.is_file():
                candidates.append((int(m.group(1)), p))
        if len(candidates) <= keep_last:
            return []
        candidates.sort(key=lambda x: x[0], reverse=True)
        to_remove = [p for _, p in candidates[keep_last:]]
        kept = [p.name for _, p in candidates[:keep_last]]
        for p in to_remove:
            try:
                p.unlink()
                logger.info("Pruned old checkpoint: %s", p.name)
                meta_file = p.parent / (p.stem + "_meta.json")
                if meta_file.exists():
                    meta_file.unlink()
            except OSError as e:
                logger.warning("Failed to prune %s: %s", p.name, e)
        return kept

    def save_checkpoint(self, filename: str):
        """Save training checkpoint with atomic write and retry on I/O errors."""
        path_base = self.checkpoint_dir / filename
        if path_base.suffix:
            path_base = path_base.with_suffix("")

        required_bytes = 600 * 1024 * 1024
        try:
            usage = shutil.disk_usage(path_base.parent)
            if usage.free < required_bytes:
                raise RuntimeError(
                    f"Insufficient disk space: {usage.free / (1024**3):.1f} GB free, "
                    f"need ~600 MB for checkpoint. Free space before training to avoid failures."
                )
        except OSError:
            pass

        last_err = None
        for attempt in range(3):
            try:
                safe_path, meta_path = save_training_checkpoint(
                    path_base,
                    self.epoch,
                    self.global_step,
                    self.generator,
                    self.text_encoder,
                    self.discriminator,
                    self.optim_g,
                    self.optim_d,
                    self.config.to_dict(),
                )
                size_mb = Path(safe_path).stat().st_size / (1024 * 1024)
                logger.info(f"Checkpoint saved: {safe_path} ({size_mb:.2f} MB)")
                return
            except (OSError, RuntimeError) as e:
                last_err = e
                logger.warning("Checkpoint save attempt %d failed: %s", attempt + 1, e)
                if attempt < 2:
                    time.sleep(2)
            finally:
                for suffix in (".safetensors.tmp", "_meta.json.tmp"):
                    p = path_base.parent / (path_base.name + suffix)
                    if p.exists():
                        try:
                            p.unlink()
                        except OSError:
                            pass

        raise RuntimeError(f"Checkpoint save failed after 3 attempts: {last_err}") from last_err
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint from .safetensors or .pt."""
        checkpoint = load_training_checkpoint(
            checkpoint_path,
            device=self.device,
            optim_g=self.optim_g,
            optim_d=self.optim_d,
        )
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.generator.load_state_dict(checkpoint['generator'])
        self.text_encoder.load_state_dict(checkpoint['text_encoder'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.optim_g.load_state_dict(checkpoint['optim_g'])
        self.optim_d.load_state_dict(checkpoint['optim_d'])
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def load_pretrained(self, pretrain_g: str, pretrain_d: str, sample_rate: int = 48000, if_f0: bool = True):
        """
        Load pretrained v2 or v3 weights.
        
        If v3 weights exist, load them directly.
        If only v2 weights exist, expand them on-the-fly.
              
        Args:
            pretrain_g: Path to pretrained generator (v2 or v3)
            pretrain_d: Path to pretrained discriminator (v2 or v3)
            sample_rate: Sample rate for the model
            if_f0: Whether model uses F0
        """
        from handlers.config import model_path
        
        # Check if v3 pretrained weights exist
        v3_pretrain_dir = os.path.join(model_path, "rvc", "pretrained_v3")
        f0_str = 'f0' if if_f0 else ''
        sr_str = f"{sample_rate // 1000}k"
        
        v3_g_stem = os.path.join(v3_pretrain_dir, f"{f0_str}G{sr_str}")
        v3_d_stem = os.path.join(v3_pretrain_dir, f"{f0_str}D{sr_str}")
        v3_g_path = v3_g_stem + ".safetensors" if os.path.exists(v3_g_stem + ".safetensors") else v3_g_stem + ".pth"
        v3_d_path = v3_d_stem + ".safetensors" if os.path.exists(v3_d_stem + ".safetensors") else v3_d_stem + ".pth"

        # Try to load v3 weights first
        if os.path.exists(v3_g_path) and os.path.exists(v3_d_path):
            logger.info(f"Loading v3 pretrained weights from {v3_pretrain_dir}")
            self._load_v3_pretrained(v3_g_path, v3_d_path)
        elif os.path.exists(pretrain_g) and os.path.exists(pretrain_d):
            # Check if provided paths are v3 or v2
            logger.info(f"Checking pretrained weights: {pretrain_g}")
            is_v3 = False
            p = Path(pretrain_g)
            config_path = p.parent / (p.stem + "_config.json")
            if config_path.exists():
                import json
                with open(config_path, "r") as f:
                    meta = json.load(f)
                is_v3 = meta.get("version") == "v3"
            else:
                if p.suffix.lower() == ".safetensors":
                    logger.warning(
                        "No _config.json found for safetensors pretrained model %s; assuming v3 format.",
                        pretrain_g,
                    )
                    is_v3 = True
                else:
                    ckpt = torch.load(pretrain_g, map_location="cpu", weights_only=True)
                    is_v3 = ckpt.get("version") == "v3"
            if is_v3:
                logger.info("Loading v3 weights directly")
                self._load_v3_pretrained(pretrain_g, pretrain_d)
            else:
                logger.info("Detected v2 weights, expanding to v3 architecture...")
                self._expand_and_load_v2(pretrain_g, pretrain_d, sample_rate, if_f0)
        else:
            logger.warning("No pretrained weights found, training from scratch")
    
    def _load_v3_pretrained(self, g_path: str, d_path: str):
        """Load v3 pretrained weights from .safetensors or .pth."""
        gen_sd, text_sd, _ = load_inference_model(g_path, device=self.device)
        d_state = load_pretrained_d(d_path, device=self.device)
        g_state = gen_sd
        g_checkpoint = {"model": g_state, "text_encoder": text_sd}
        d_checkpoint = {"model": d_state}

        # Load generator
        gen_state_dict = self.generator.state_dict()
        
        loaded = 0
        not_loaded = 0
        
        for k in gen_state_dict.keys():
            if k in g_state:
                if gen_state_dict[k].shape == g_state[k].shape:
                    gen_state_dict[k] = g_state[k]
                    loaded += 1
                else:
                    logger.warning(f"Shape mismatch for generator {k}")
                    not_loaded += 1
            else:
                not_loaded += 1
        
        self.generator.load_state_dict(gen_state_dict, strict=False)
        logger.info(f"Generator: loaded {loaded} layers, {not_loaded} randomly initialized")
        
        # Load text encoder if present
        if 'text_encoder' in g_checkpoint:
            text_state = g_checkpoint['text_encoder']
            text_state_dict = self.text_encoder.state_dict()
            
            text_loaded = 0
            text_not_loaded = 0
            
            for k in text_state_dict.keys():
                if k in text_state:
                    if text_state_dict[k].shape == text_state[k].shape:
                        text_state_dict[k] = text_state[k]
                        text_loaded += 1
                    else:
                        text_not_loaded += 1
                else:
                    text_not_loaded += 1
            
            self.text_encoder.load_state_dict(text_state_dict, strict=False)
            logger.info(f"Text encoder: loaded {text_loaded} layers, {text_not_loaded} randomly initialized")
        
        # Load discriminator
        d_state = d_checkpoint['model']
        disc_state_dict = self.discriminator.state_dict()
        
        d_loaded = 0
        d_not_loaded = 0
        
        for k in disc_state_dict.keys():
            if k in d_state:
                if disc_state_dict[k].shape == d_state[k].shape:
                    disc_state_dict[k] = d_state[k]
                    d_loaded += 1
                else:
                    d_not_loaded += 1
            else:
                d_not_loaded += 1
        
        self.discriminator.load_state_dict(disc_state_dict, strict=False)
        logger.info(f"Discriminator: loaded {d_loaded} layers, {d_not_loaded} randomly initialized")
    
    def _expand_and_load_v2(self, v2_g_path: str, v2_d_path: str, sample_rate: int, if_f0: bool):
        """Expand v2 weights and load them."""
        from modules.rvc_v3.training.expand_weights import expand_v2_to_v3
        from handlers.config import model_path
        import tempfile
        
        # Create temporary directory for expanded weights
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info("Expanding v2 weights to v3 architecture...")
            v3_g_path, v3_d_path = expand_v2_to_v3(
                v2_g_path,
                v2_d_path,
                temp_dir,
                sample_rate=sample_rate,
                if_f0=if_f0
            )
            
            # Load the expanded weights
            self._load_v3_pretrained(v3_g_path, v3_d_path)
            
            # Optionally save to pretrained_v3 directory for future use
            v3_pretrain_dir = os.path.join(model_path, "rvc", "pretrained_v3")
            os.makedirs(v3_pretrain_dir, exist_ok=True)
            import shutil
            f0_str = "f0" if if_f0 else ""
            sr_str = f"{sample_rate // 1000}k"
            final_g_base = os.path.join(v3_pretrain_dir, f"{f0_str}G{sr_str}")
            final_d_base = os.path.join(v3_pretrain_dir, f"{f0_str}D{sr_str}")
            for src, dst_base in [(v3_g_path, final_g_base), (v3_d_path, final_d_base)]:
                if os.path.exists(src):
                    shutil.copy(src, dst_base + ".safetensors")
                cfg_src = src.replace(".safetensors", "_config.json")
                if os.path.exists(cfg_src):
                    shutil.copy(cfg_src, dst_base + "_config.json")
            
            logger.info(f"Expanded weights saved to {v3_pretrain_dir} for future use")

