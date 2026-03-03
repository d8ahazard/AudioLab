"""
Wrapper for RVC V3 training to integrate with the UI.

Adapts v2-style hparams to v3 config and wraps the RVCV3Trainer.
"""

import logging
import math
import os
import shutil
import time
from pathlib import Path

import gradio as gr
import torch
from tqdm import tqdm

from handlers.config import output_path, model_path
from modules.rvc.infer.lib.train.early_stopping import EarlyStoppingMonitor
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
        # spec_channels is *linear spectrogram* bins = n_fft//2 + 1
        if hasattr(hparams.data, 'filter_length'):
            config.spec_channels = hparams.data.filter_length // 2 + 1
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
    
    # Keep vocoder compatible with v2 warm starts unless explicitly overridden.
    # Most shipped v2 pretrains are HiFiGAN-oriented.
    if hasattr(hparams, "vocoder_type") and hparams.vocoder_type:
        config.vocoder_type = str(hparams.vocoder_type).lower()
    else:
        config.vocoder_type = "hifigan"
    
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
    
    logger.info("Project: %s", project_name)
    logger.info("Project directory: %s", project_dir)
    logger.info("Sample rate: %s", config.sampling_rate)
    logger.info("Batch size: %s", config.batch_size)
    logger.info("Epochs: %s", config.epochs)
    
    # Save v3 config to project directory
    config_save_path = os.path.join(project_dir, "config_v3.json")
    config.save(config_save_path)
    logger.info("V3 config saved to %s", config_save_path)
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)
    
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
            logger.info("Loading pretrained weights: G=%s, D=%s", pretrain_g, pretrain_d)
            
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
                logger.warning("Failed to load pretrained weights: %s", e)
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
        logger.info("Progress: %.1f%% - %s", prog * 100.0, message)
    
    # ---------------------------------------
    # Start training (use existing v2 artifacts)
    # ---------------------------------------
    # The RVC Train UI produces classic artifacts under:
    #   outputs/voices/<name>/{0_gt_wavs,2a_f0,2b-f0nsf,3_feature768,filelist.txt}
    # We intentionally reuse the proven v2 dataset/loader so V3 trains on the same inputs.
    try:
        from torch.utils.data import DataLoader
        from modules.rvc_v3.training.v3_dataset import (
            TextAudioLoaderMultiNSFsidV3,
            TextAudioCollateMultiNSFsidV3,
        )

        filelist = os.path.join(project_dir, "filelist.txt")
        if not os.path.exists(filelist):
            raise FileNotFoundError(f"Expected filelist not found: {filelist}")

        # V3 dataset with lyrics/text conditioning
        train_dataset = TextAudioLoaderMultiNSFsidV3(
            filelist,
            hparams.data,
            project_dir=project_dir,
            phonemizer=phonemizer,
        )
        collate_fn = TextAudioCollateMultiNSFsidV3()

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,  # Windows-safe
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_fn,
        )

        trainer.generator.train()
        trainer.text_encoder.train()
        trainer.discriminator.train()

        total_epochs = max(1, int(config.epochs))
        early_stop_monitor = EarlyStoppingMonitor(
            ema_alpha=0.05,
            plateau_patience=config.early_stop_plateau_patience,
            uptrend_patience=config.early_stop_uptrend_patience,
            min_improvement_ratio=0.01,
            min_epochs=config.early_stop_min_epochs,
            composite_weight_fm=0.3,
        )
        total_steps = max(1, (len(train_loader) * total_epochs))
        step_no = 0
        best_composite: float | None = None

        epoch_bar = tqdm(
            range(total_epochs),
            total=total_epochs,
            desc="V3 training",
            unit="epoch",
        )
        for epoch in epoch_bar:
            trainer.epoch = epoch
            epoch_start = time.perf_counter()
            epoch_loss_sums = {
                "loss_mel": 0.0,
                "loss_fm": 0.0,
                "loss_gen": 0.0,
                "loss_disc": 0.0,
                "loss_kl": 0.0,
                "loss_total_g": 0.0,
            }
            batch_count = 0

            batch_bar = tqdm(
                enumerate(train_loader, start=1),
                total=len(train_loader),
                desc=f"Epoch {epoch + 1}/{total_epochs}",
                unit="batch",
                leave=False,
            )
            for batch_idx, info in batch_bar:
                (
                    phone,
                    phone_lengths,
                    pitch,
                    pitchf,
                    spec,
                    spec_lengths,
                    wave,
                    wave_lengths,
                    sid,
                    text_tokens,
                    text_mask,
                ) = info

                # Move to device
                if torch.cuda.is_available():
                    phone = phone.to(device, non_blocking=True)
                    phone_lengths = phone_lengths.to(device, non_blocking=True)
                    pitch = pitch.to(device, non_blocking=True)
                    pitchf = pitchf.to(device, non_blocking=True)
                    spec = spec.to(device, non_blocking=True)
                    spec_lengths = spec_lengths.to(device, non_blocking=True)
                    wave = wave.to(device, non_blocking=True)
                    sid = sid.to(device, non_blocking=True)
                    if text_tokens is not None:
                        text_tokens = text_tokens.to(device, non_blocking=True)
                        text_mask = text_mask.to(device, non_blocking=True)

                batch = {
                    "phone": phone,
                    "phone_lengths": phone_lengths,
                    "pitch": pitch,
                    "pitchf": pitchf,
                    "spec": spec,
                    "spec_lengths": spec_lengths,
                    "wave": wave,
                    "sid": sid,
                    "text_tokens": text_tokens,
                    "text_mask": text_mask,
                }

                losses = trainer._train_step(batch)
                trainer.global_step += 1

                loss_values = {
                    "loss_mel": float(losses.get("loss_mel", 0.0)),
                    "loss_fm": float(losses.get("loss_fm", 0.0)),
                    "loss_gen": float(losses.get("loss_gen", 0.0)),
                    "loss_disc": float(losses.get("loss_disc", 0.0)),
                    "loss_kl": float(losses.get("loss_kl", 0.0)),
                    "loss_total_g": float(losses.get("loss_total_g", 0.0)),
                }
                if not all(math.isfinite(v) for v in loss_values.values()):
                    raise RuntimeError(
                        "Non-finite loss detected "
                        f"(epoch={epoch + 1}, batch={batch_idx}, losses={loss_values})"
                    )

                for key, value in loss_values.items():
                    epoch_loss_sums[key] += value
                batch_count += 1

                early_stop_monitor.update(
                    loss_values["loss_total_g"],
                    loss_values["loss_disc"],
                    loss_values["loss_mel"],
                    loss_values["loss_kl"],
                    loss_values["loss_fm"],
                )

                current_lr = float(trainer.optim_g.param_groups[0]["lr"])
                batch_bar.set_postfix(
                    {
                        "mel": f"{loss_values['loss_mel']:.3f}",
                        "fm": f"{loss_values['loss_fm']:.3f}",
                        "g": f"{loss_values['loss_gen']:.3f}",
                        "d": f"{loss_values['loss_disc']:.3f}",
                        "lr": f"{current_lr:.2e}",
                        "step": trainer.global_step,
                    }
                )

                step_no += 1
                progress_callback(
                    step_no / total_steps,
                    f"Epoch {epoch + 1}/{total_epochs} step {batch_idx}/{len(train_loader)} "
                    f"mel={loss_values['loss_mel']:.3f} fm={loss_values['loss_fm']:.3f} "
                    f"g={loss_values['loss_gen']:.3f} d={loss_values['loss_disc']:.3f} "
                    f"lr={current_lr:.2e}",
                    total_steps,
                )

            batch_bar.close()

            early_stop_monitor.on_epoch_end(epoch)
            if early_stop_monitor.should_stop():
                logger.info("[EarlyStop] %s", early_stop_monitor.reason())
                break

            avg_losses = {
                key: (value / max(1, batch_count))
                for key, value in epoch_loss_sums.items()
            }
            composite = avg_losses["loss_mel"] + 0.3 * avg_losses["loss_fm"]

            # Best checkpoint: save when we beat previous best (lower composite = better)
            checkpoint_interval = getattr(config, "checkpoint_interval", 25)
            checkpoint_keep_last = getattr(config, "checkpoint_keep_last", 2)
            if best_composite is None or composite < best_composite:
                trainer.save_checkpoint("checkpoint_best")
                best_composite = composite
                logger.info("New best checkpoint (composite=%.4f)", composite)

            # Periodic checkpoint: every N epochs
            ckpt_path = None
            if (epoch + 1) % checkpoint_interval == 0 or epoch == 0:
                ckpt_name = f"checkpoint_epoch_{epoch}"
                trainer.save_checkpoint(ckpt_name)
                trainer._prune_epoch_checkpoints(keep_last=checkpoint_keep_last)
                ckpt_path = os.path.join(project_dir, "checkpoints", ckpt_name + ".safetensors")
            ckpt_size_mb = (
                os.path.getsize(ckpt_path) / (1024 * 1024)
                if ckpt_path and os.path.exists(ckpt_path)
                else 0.0
            )
            trainer.scheduler_g.step()
            trainer.scheduler_d.step()

            epoch_dur = max(1e-6, time.perf_counter() - epoch_start)
            samples_seen = batch_count * int(config.batch_size)
            samples_per_sec = samples_seen / epoch_dur
            logger.info(
                "Epoch %s/%s complete | steps=%s | sec=%.2f | samples/sec=%.2f | "
                "mel=%.4f fm=%.4f g=%.4f d=%.4f kl=%.4f g_total=%.4f | checkpoint=%s (%.2f MB)",
                epoch + 1,
                total_epochs,
                batch_count,
                epoch_dur,
                samples_per_sec,
                avg_losses["loss_mel"],
                avg_losses["loss_fm"],
                avg_losses["loss_gen"],
                avg_losses["loss_disc"],
                avg_losses["loss_kl"],
                avg_losses["loss_total_g"],
                ckpt_path or "(best only)",
                ckpt_size_mb,
            )
            epoch_bar.set_postfix(
                {
                    "mel": f"{avg_losses['loss_mel']:.3f}",
                    "g": f"{avg_losses['loss_gen']:.3f}",
                    "d": f"{avg_losses['loss_disc']:.3f}",
                    "lr": f"{float(trainer.optim_g.param_groups[0]['lr']):.2e}",
                }
            )

        epoch_bar.close()
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error("Training failed: %s", e, exc_info=True)
        raise
    
    # After training, save the final model (safetensors + config)
    v3_trained_dir = os.path.join(model_path, "trained", "v3")
    os.makedirs(v3_trained_dir, exist_ok=True)
    final_model_base = os.path.join(v3_trained_dir, project_name)
    logger.info("Saving final model to %s.safetensors", final_model_base)

    # Check disk space before final save (~500 MB required)
    try:
        usage = shutil.disk_usage(v3_trained_dir)
        if usage.free < 550 * 1024 * 1024:
            raise RuntimeError(
                f"Insufficient disk space for final model: {usage.free / (1024**3):.1f} GB free, "
                "need ~550 MB. Free space before saving."
            )
    except OSError:
        pass

    from modules.rvc_v3.io.checkpoint_io import save_inference_model
    save_inference_model(
        final_model_base,
        trainer.generator,
        trainer.text_encoder,
        config.to_dict(),
        version="v3",
        iteration=trainer.global_step,
        lr=config.learning_rate,
    )
    final_safe_path = final_model_base + ".safetensors"
    final_size_mb = os.path.getsize(final_safe_path) / (1024 * 1024)
    logger.info("Final V3 model written: %s (%.2f MB)", final_safe_path, final_size_mb)
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

