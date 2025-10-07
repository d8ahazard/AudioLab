import datetime
from collections import deque
import logging
import os
from handlers.config import model_path
from random import randint, shuffle
import gradio as gr
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.distributed_c10d import is_initialized
from torch.utils.data import DataLoader
from tqdm import tqdm
import shutil

from modules.rvc.infer.lib.infer_pack import commons
from modules.rvc.infer.lib.train import utils
from modules.rvc.infer.lib.train.data_utils import (
    DistributedBucketSampler,
    TextAudioCollate,
    TextAudioCollateMultiNSFsid,
    TextAudioLoader,
    TextAudioLoaderMultiNSFsid,
)
from modules.rvc.infer.lib.train.losses import (
    discriminator_loss,
    feature_loss,
    generator_loss,
    kl_loss,
)
from modules.rvc.infer.lib.train.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from modules.rvc.infer.lib.train.process_ckpt import savee

try:
    import intel_extension_for_pytorch as ipex
    if torch.xpu.is_available():
        from modules.rvc.infer.modules.ipex import ipex_init
        from modules.rvc.infer.modules.ipex.gradscaler import gradscaler_init
        from torch.xpu.amp import autocast
        GradScaler = gradscaler_init()
        ipex_init()
    else:
        from torch.cuda.amp import GradScaler, autocast
except Exception:
    from torch.cuda.amp import GradScaler, autocast
from time import time as ttime

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
global_step = 0
last_saved_epoch = None  # Track the last saved epoch for cleanup
loss_tracker = None  # Global loss tracker for early stopping/auto-save


class LossTracker:
    """
    Tracks moving averages and trends of losses to detect overtraining/plateaus
    and decide on auto-saving and early stopping.
    """
    def __init__(self,
                 ema_alpha: float = 0.05,
                 min_delta: float = 1e-4,
                 zero_threshold: float = 0.02,
                 slope_window: int = 200,
                 slope_min: float = 1e-5,
                 upslope_patience_steps: int = 2000,
                 plateau_patience_steps: int = 4000):
        self.ema_alpha = float(ema_alpha)
        self.min_delta = float(min_delta)
        self.zero_threshold = float(zero_threshold)
        self.slope_window = int(max(10, slope_window))
        self.slope_min = float(slope_min)
        self.upslope_patience_steps = int(upslope_patience_steps)
        self.plateau_patience_steps = int(plateau_patience_steps)

        self.ema_gen = None
        self.ema_disc = None
        self.ema_mel = None
        self.ema_kl = None
        self.ema_fm = None

        self.best_gen = float('inf')
        self.steps_since_best = 0
        self.steps_since_last_save = 0
        self.gen_hist = deque(maxlen=self.slope_window)

        self.upslope_counter = 0
        self.plateau_counter = 0

    def _ema(self, prev, val):
        if prev is None:
            return float(val)
        a = self.ema_alpha
        return (1.0 - a) * float(prev) + a * float(val)

    def update(self, loss_gen_all, loss_disc, loss_mel, loss_kl, loss_fm):
        self.ema_gen = self._ema(self.ema_gen, loss_gen_all)
        self.ema_disc = self._ema(self.ema_disc, loss_disc)
        self.ema_mel = self._ema(self.ema_mel, loss_mel)
        self.ema_kl = self._ema(self.ema_kl, loss_kl)
        self.ema_fm = self._ema(self.ema_fm, loss_fm)

        self.gen_hist.append(self.ema_gen)
        self.steps_since_last_save += 1
        self.steps_since_best += 1

        # Track upslope
        if self.ema_gen > (self.best_gen + self.min_delta):
            self.upslope_counter += 1
        else:
            self.upslope_counter = 0

        # Track plateau via simple slope across window
        if len(self.gen_hist) >= self.gen_hist.maxlen:
            start = self.gen_hist[0]
            end = self.gen_hist[-1]
            slope = (end - start) / float(self.gen_hist.maxlen)
            if abs(slope) < self.slope_min:
                self.plateau_counter += self.gen_hist.maxlen
            else:
                # decay plateau counter if we see movement
                self.plateau_counter = max(0, self.plateau_counter - self.gen_hist.maxlen)

    def should_save_best(self) -> bool:
        if self.ema_gen is None:
            return False
        if self.ema_gen + self.min_delta < self.best_gen:
            self.best_gen = self.ema_gen
            self.steps_since_best = 0
            return True
        return False

    def near_zero(self) -> bool:
        return (self.ema_gen is not None) and (self.ema_gen <= self.zero_threshold)

    def should_early_stop(self) -> bool:
        # Early stop if consistent upslope or long plateau
        if self.upslope_counter >= self.upslope_patience_steps:
            return True
        if self.plateau_counter >= self.plateau_patience_steps:
            return True
        return False

    def reset_after_save(self):
        self.steps_since_last_save = 0

    def status_str(self) -> str:
        return (
            f"EMA(gen={self.ema_gen:.4f} disc={self.ema_disc:.4f} mel={self.ema_mel:.4f} "
            f"kl={self.ema_kl:.4f} fm={self.ema_fm:.4f}), "
            f"best_gen={self.best_gen:.4f}, steps_since_save={self.steps_since_last_save}, "
            f"upslope={self.upslope_counter}, plateau={self.plateau_counter}"
        )


class EpochRecorder:
    def __init__(self):
        self.last_time = ttime()

    def record(self):
        now_time = ttime()
        elapsed_time = now_time - self.last_time
        self.last_time = now_time
        elapsed_time_str = str(datetime.timedelta(seconds=elapsed_time))
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{current_time}] | ({elapsed_time_str})"


def train_main(hps, progress: gr.Progress):
    os.environ["CUDA_VISIBLE_DEVICES"] = hps.gpus.replace("-", ",")
    global global_step
    global_step = 0

    n_gpus = torch.cuda.device_count()
    if not torch.cuda.is_available() and torch.backends.mps.is_available():
        n_gpus = 1
    if n_gpus < 1:
        print("NO GPU DETECTED: falling back to CPU - this may take a while")
        n_gpus = 1
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))
    children = []
    logger = utils.get_logger(hps.model_dir)
    for i in range(n_gpus):
        subproc = mp.Process(
            target=run,
            args=(i, n_gpus, hps, logger, progress),
        )
        children.append(subproc)
        subproc.start()

    for i in range(n_gpus):
        children[i].join()


def run(rank, n_gpus, hps, logger: logging.Logger, progress: gr.Progress):
    if hps.version == "v1":
        from modules.rvc.infer.lib.infer_pack.models import MultiPeriodDiscriminator
        from modules.rvc.infer.lib.infer_pack.models import SynthesizerTrnMs256NSFsid as RVC_Model_f0
        from modules.rvc.infer.lib.infer_pack.models import SynthesizerTrnMs256NSFsid_nono as RVC_Model_nof0
    else:
        from modules.rvc.infer.lib.infer_pack.models import (
            SynthesizerTrnMs768NSFsid as RVC_Model_f0,
            SynthesizerTrnMs768NSFsid_nono as RVC_Model_nof0,
            MultiPeriodDiscriminatorV2 as MultiPeriodDiscriminator,
        )

    global global_step
    if rank == 0:
        logger.info(hps)

    # Initialize distributed only for multi-GPU training
    backend = "nccl" if torch.cuda.is_available() and os.name != "nt" else "gloo"
    if n_gpus > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://", world_size=n_gpus, rank=rank)
        # Ensure all ranks reach this point before proceeding
        dist.barrier()

    torch.manual_seed(hps.train.seed)

    if hps.if_f0 == 1:
        train_dataset = TextAudioLoaderMultiNSFsid(hps.data.training_files, hps.data)
    else:
        train_dataset = TextAudioLoader(hps.data.training_files, hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size * n_gpus,
        [100, 200, 300, 400, 500, 600, 700, 800, 900],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )

    if hps.if_f0 == 1:
        collate_fn = TextAudioCollateMultiNSFsid()
    else:
        collate_fn = TextAudioCollate()
    train_loader = DataLoader(
        train_dataset,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=8,
    )
    if hps.if_f0 == 1:
        net_g = RVC_Model_f0(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model,
            is_half=hps.train.fp16_run,
            sr=hps.sample_rate,
        )
    else:
        net_g = RVC_Model_nof0(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model,
            is_half=hps.train.fp16_run,
        )
    if torch.cuda.is_available():
        net_g = net_g.cuda(rank)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm)
    if torch.cuda.is_available():
        net_d = net_d.cuda(rank)
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    if n_gpus > 1 and dist.is_initialized():
        if torch.cuda.is_available():
            # Wrap with DDP on the specific device
            net_g = DDP(net_g, device_ids=[rank])
            net_d = DDP(net_d, device_ids=[rank])
        else:
            # CPU DDP fallback
            net_g = DDP(net_g)
            net_d = DDP(net_d)

    d_path = utils.latest_checkpoint_path(os.path.join(hps.model_dir, "saves"), "D_*.pth")
    g_path = utils.latest_checkpoint_path(os.path.join(hps.model_dir, "saves"), "G_*.pth")
    if d_path is None or g_path is None:
        logger.info("No checkpoint found, using default initialization.")
        epoch_str = 1
        global_step = 0
        if hps.pretrainG != "":
            if rank == 0:
                logger.info(f"Loading (initial) pretrainedG {hps.pretrainG}.")
            if hasattr(net_g, "module"):
                logger.info(net_g.module.load_state_dict(torch.load(hps.pretrainG, map_location="cpu")["model"]))
            else:
                logger.info(net_g.load_state_dict(torch.load(hps.pretrainG, map_location="cpu")["model"]))
        if hps.pretrainD != "":
            if rank == 0:
                logger.info(f"Loading (initial) pretrainedD {hps.pretrainD}.")
            if hasattr(net_d, "module"):
                logger.info(net_d.module.load_state_dict(torch.load(hps.pretrainD, map_location="cpu")["model"]))
            else:
                logger.info(net_d.load_state_dict(torch.load(hps.pretrainD, map_location="cpu")["model"]))
    else:
        logger.info(f"Loading checkpoint weights from {d_path} and {g_path}.")
        _, _, _, epoch_str = utils.load_checkpoint(d_path, net_d, optim_d)
        if rank == 0:
            logger.info("loaded D")
        _, _, _, epoch_str = utils.load_checkpoint(g_path, net_g, optim_g)
        global_step = (epoch_str - 1) * len(train_loader)

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )

    scaler = GradScaler(enabled=hps.train.fp16_run)
    cache = []
    for epoch in range(epoch_str, hps.train.epochs + 1):
        train_and_evaluate(
            rank,
            epoch,
            hps,
            [net_g, net_d],
            [optim_g, optim_d],
            [scheduler_g, scheduler_d],
            scaler,
            [train_loader, None],
            logger,
            None,
            cache,
            progress,
        )
        scheduler_g.step()
        scheduler_d.step()

        # if global_step >= hps.train.total_steps:
        #     if rank == 0:
        #         logger.info("Global step limit reached. Stopping training.")
        #     break

        if epoch >= hps.train.epochs:
            if rank == 0:
                logger.info(f"Training completed at epoch {epoch}.")
            break


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers, cache, progress):
    net_g, net_d = nets
    optim_g, optim_d = optims
    train_loader, eval_loader = loaders
    if writers is not None:
        _, _ = writers

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step
    global last_saved_epoch

    net_g.train()
    net_d.train()

    total_epochs = hps.train.epochs
    total_steps = len(train_loader)
    current_epoch = epoch - 1  # Convert to 0-based for progress calculation
    
    # Calculate overall progress percentage
    def update_progress(batch_idx):
        if rank == 0:
            epoch_progress = batch_idx / total_steps
            overall_progress = (current_epoch + epoch_progress) / total_epochs
            progress_msg = f"Training: Epoch {epoch}/{total_epochs} - {epoch_progress*100:.1f}%"
            return overall_progress, progress_msg
        return None, None

    # Prepare data iterator (with caching and TQDM)
    if hps.if_cache_data_in_gpu:
        if not cache:
            if rank == 0:
                for batch_idx, info in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch} (caching)"):
                    progress_val, msg = update_progress(batch_idx)
                    if progress_val is not None:
                        progress(progress_val, msg)
                    if hps.if_f0 == 1:
                        (phone, phone_lengths, pitch, pitch_f, spec, spec_lengths, wave, wave_lengths, sid) = info
                    else:
                        (phone, phone_lengths, spec, spec_lengths, wave, wave_lengths, sid) = info
                        pitch = None
                        pitch_f = None
                    if torch.cuda.is_available():
                        phone = phone.cuda(rank, non_blocking=True)
                        phone_lengths = phone_lengths.cuda(rank, non_blocking=True)
                        if hps.if_f0 == 1:
                            pitch = pitch.cuda(rank, non_blocking=True)
                            pitch_f = pitch_f.cuda(rank, non_blocking=True)
                        sid = sid.cuda(rank, non_blocking=True)
                        spec = spec.cuda(rank, non_blocking=True)
                        spec_lengths = spec_lengths.cuda(rank, non_blocking=True)
                        wave = wave.cuda(rank, non_blocking=True)
                        wave_lengths = wave_lengths.cuda(rank, non_blocking=True)
                    if hps.if_f0 == 1:
                        cache.append((batch_idx, (phone, phone_lengths, pitch, pitch_f, spec, spec_lengths, wave, wave_lengths, sid)))
                    else:
                        cache.append((batch_idx, (phone, phone_lengths, spec, spec_lengths, wave, wave_lengths, sid)))
            else:
                for batch_idx, info in enumerate(train_loader):
                    progress_val, msg = update_progress(batch_idx)
                    if progress_val is not None:
                        progress(progress_val, msg)
                    if hps.if_f0 == 1:
                        (phone, phone_lengths, pitch, pitch_f, spec, spec_lengths, wave, wave_lengths, sid) = info
                    else:
                        (phone, phone_lengths, spec, spec_lengths, wave, wave_lengths, sid) = info
                        pitch = None
                        pitch_f = None
                    if torch.cuda.is_available():
                        phone = phone.cuda(rank, non_blocking=True)
                        phone_lengths = phone_lengths.cuda(rank, non_blocking=True)
                        if hps.if_f0 == 1:
                            pitch = pitch.cuda(rank, non_blocking=True)
                            pitch_f = pitch_f.cuda(rank, non_blocking=True)
                        sid = sid.cuda(rank, non_blocking=True)
                        spec = spec.cuda(rank, non_blocking=True)
                        spec_lengths = spec_lengths.cuda(rank, non_blocking=True)
                        wave = wave.cuda(rank, non_blocking=True)
                        wave_lengths = wave_lengths.cuda(rank, non_blocking=True)
                    if hps.if_f0 == 1:
                        cache.append((batch_idx, (phone, phone_lengths, pitch, pitch_f, spec, spec_lengths, wave, wave_lengths, sid)))
                    else:
                        cache.append((batch_idx, (phone, phone_lengths, spec, spec_lengths, wave, wave_lengths, sid)))
        else:
            shuffle(cache)
        if rank == 0:
            data_iterator = tqdm(cache, total=len(cache), desc=f"Epoch {epoch}")
        else:
            data_iterator = cache
    else:
        if rank == 0:
            data_iterator = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
        else:
            data_iterator = enumerate(train_loader)

    epoch_recorder = EpochRecorder()
    for batch_idx, info in data_iterator:
        progress_val, msg = update_progress(batch_idx)
        if progress_val is not None:
            progress(progress_val, msg)

        if hps.if_f0 == 1:
            (phone, phone_lengths, pitch, pitch_f, spec, spec_lengths, wave, wave_lengths, sid) = info
        else:
            phone, phone_lengths, spec, spec_lengths, wave, wave_lengths, sid = info
            pitch = None
            pitch_f = None
        if (not hps.if_cache_data_in_gpu) and torch.cuda.is_available():
            phone = phone.cuda(rank, non_blocking=True)
            phone_lengths = phone_lengths.cuda(rank, non_blocking=True)
            if hps.if_f0 == 1 and pitch is not None:
                pitch = pitch.cuda(rank, non_blocking=True)
                pitch_f = pitch_f.cuda(rank, non_blocking=True)
            sid = sid.cuda(rank, non_blocking=True)
            spec = spec.cuda(rank, non_blocking=True)
            spec_lengths = spec_lengths.cuda(rank, non_blocking=True)
            wave = wave.cuda(rank, non_blocking=True)

        with autocast(enabled=hps.train.fp16_run):
            if hps.if_f0 == 1:
                (y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q)) = net_g(
                    phone, phone_lengths, pitch, pitch_f, spec, spec_lengths, sid
                )
            else:
                (y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q)) = net_g(
                    phone, phone_lengths, spec, spec_lengths, sid
                )
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )
            with autocast(enabled=False):
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.float().squeeze(1),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
            if hps.train.fp16_run:
                y_hat_mel = y_hat_mel.half()
            wave = commons.slice_segments(
                wave, ids_slice * hps.data.hop_length, hps.train.segment_size
            )

            y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
        optim_d.zero_grad()
        scaler.scale(loss_disc).backward()
        scaler.unscale_(optim_d)
        _ = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=hps.train.fp16_run):
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
            with autocast(enabled=False):
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        _ = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        # Update global loss tracker and possibly log/save
        global loss_tracker
        if loss_tracker is None and rank == 0:
            # Initialize with default heuristics; could be moved to hps later
            loss_tracker = LossTracker(
                ema_alpha=0.05,
                min_delta=1e-4,
                zero_threshold=0.02,
                slope_window=200,
                slope_min=1e-5,
                upslope_patience_steps=2000,
                plateau_patience_steps=4000,
            )
        if rank == 0 and loss_tracker is not None:
            loss_tracker.update(
                float(loss_gen_all.detach().cpu()),
                float(loss_disc.detach().cpu()),
                float(loss_mel.detach().cpu()),
                float(loss_kl.detach().cpu()),
                float(loss_fm.detach().cpu()),
            )
            if global_step % hps.train.log_interval == 0:
                logger.info(f"[Tracker] {loss_tracker.status_str()}")

            # Early stop if sustained upslope or long plateau after at least one save
            if loss_tracker.should_early_stop():
                if rank == 0:
                    logger.info("[Tracker] Early stopping: sustained upslope or plateau detected.")
                # Break out of batch loop; outer loop will handle termination
                break

        if rank == 0 and global_step % hps.train.log_interval == 0:
            lr = optim_g.param_groups[0]["lr"]
            logger.info("Train Epoch: {} [{:.0f}%]".format(epoch, 100.0 * batch_idx / len(train_loader)))
            logger.info([global_step, lr])
            logger.info(f"loss_disc={loss_disc:.3f}, loss_gen={loss_gen:.3f}, loss_fm={loss_fm:.3f}, loss_mel={loss_mel:.3f}, loss_kl={loss_kl:.3f}")

        global_step += 1

    if rank == 0:
        # Save model if it's time for a periodic save or if training is complete
        should_save = (epoch % hps.save_epoch_frequency == 0) or (epoch >= hps.train.epochs)

        # Also save if we have a very good loss (auto-save at epoch boundaries)
        if loss_tracker is not None and (loss_tracker.near_zero() or loss_tracker.should_save_best()):
            should_save = True
            logger.info("[Tracker] Auto-saving at epoch boundary due to best/near-zero gen loss.")
        
        if should_save:
            # If save_latest_only is True and we have a previous save, clean up old checkpoints
            if hps.save_latest_only and last_saved_epoch is not None:
                # Clean up files from previous save in both directories
                for save_dir in [os.path.join(model_path, "trained"), os.path.join(hps.model_dir, "saves")]:
                    if os.path.exists(save_dir):
                        for file in os.listdir(save_dir):
                            if f"_v{last_saved_epoch}" in file and (file.endswith(".pth") or file.endswith(".index")):
                                try:
                                    os.remove(os.path.join(save_dir, file))
                                except Exception as e:
                                    logger.warning(f"Failed to remove old checkpoint {file} from {save_dir}: {e}")

            # Get model state
            if hasattr(net_g, "module"):
                ckpt = net_g.module.state_dict()
            else:
                ckpt = net_g.state_dict()
            
            # Save checkpoints in saves directory (only when save frequency is met or final epoch)
            if should_save:
                utils.save_checkpoint(
                    net_g,
                    optim_g,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "saves", f"G_e{epoch}.pth"),
                )
                utils.save_checkpoint(
                    net_d,
                    optim_d,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "saves", f"D_e{epoch}.pth"),
                )
            
            # Determine the model name for this save (with epoch suffix for intermediate saves)
            model_name = f"{hps.name}_v{epoch}" if epoch < hps.train.epochs else hps.name

            # Save the model in trained directory
            save_result = savee(
                ckpt,
                hps.sample_rate,
                hps.if_f0,
                model_name,
                epoch,
                hps.version,
                hps,
            )

            if save_result != "Success.":
                logger.error(f"Failed to save model: {save_result}")
                # Skip the rest of the save logic for this epoch
                # Note: Continue not used here as we're not in a loop

            # Update last_saved_epoch for next cleanup
            last_saved_epoch = epoch

            # Reset loss tracker after saving
            if loss_tracker is not None:
                loss_tracker.reset_after_save()

            # Copy index file if it exists - look for added_*.index files
            index_path = None
            for file in os.listdir(hps.model_dir):
                if file.endswith(".index") and ("added_" in file or file.startswith("added")):
                    index_path = os.path.join(hps.model_dir, file)
                    break

            if index_path is not None and os.path.exists(index_path):
                target_index = os.path.join(model_path, "trained", f"{model_name}.index")
                shutil.copy2(index_path, target_index)
                logger.info(f"Copied index file to {target_index}")
            else:
                logger.warning(f"No index file found in {hps.model_dir} to copy alongside model")
            
            model_file_path = os.path.join(model_path, "trained", f"{model_name}.pth")
            logger.info(f"Saved checkpoint: {model_file_path}")
            
        if epoch >= hps.train.epochs:
            logger.info("Training is done. The program is closed.")

            # Ensure final model and index are properly saved to trained folder
            final_model_name = f"{hps.name}.pth"
            final_model_path = os.path.join(model_path, "trained", final_model_name)

            # Check if final model exists and has corresponding index
            if os.path.exists(final_model_path):
                final_index_path = final_model_path.replace(".pth", ".index")

                # If index doesn't exist alongside final model, try to copy it
                if not os.path.exists(final_index_path):
                    # Look for the most recent index file in the experiment directory
                    index_files = [f for f in os.listdir(hps.model_dir) if f.endswith(".index") and ("added_" in f or f.startswith("added"))]
                    if index_files:
                        latest_index = max(index_files, key=lambda f: os.path.getmtime(os.path.join(hps.model_dir, f)))
                        latest_index_path = os.path.join(hps.model_dir, latest_index)
                        shutil.copy2(latest_index_path, final_index_path)
                        logger.info(f"Copied final index to trained folder: {final_index_path}")

                logger.info(f"Final model and index saved: {final_model_path} and {final_index_path}")

            return
