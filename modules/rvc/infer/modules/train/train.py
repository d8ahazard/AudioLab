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
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

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
from modules.rvc.infer.lib.train.early_stopping import EarlyStoppingMonitor
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
loss_tracker = None  # Global loss tracker for auto-save/plotting
early_stop_monitor = None  # Shared early-stop monitor (actually stops training)


class LossTracker:
    """
    Tracks moving averages and trends of losses to detect overtraining/plateaus
    and decide on auto-saving and early stopping with intelligent save controls.
    
    Key metrics for decisions:
    - PRIMARY: ema_mel (log-mel reconstruction loss) - tracks intelligibility/timbre
    - SECONDARY: ema_fm (feature-matching loss) - tracks naturalness/artifacts
    - COMPOSITE: mel + 0.3 × fm - combined quality metric
    - IGNORE for plateau: loss_gen, loss_disc (adversarial signals that oscillate)
    - MONITOR for health: ema_kl (avoid posterior collapse, but don't use for stopping)
    """
    def __init__(self,
                 ema_alpha: float = 0.05,
                 min_delta: float = 1e-4,
                 min_save_interval: int = 5,  # Minimum epochs between best-model saves
                 significant_improvement_threshold: float = 0.01,  # 1% improvement required
                 max_best_saves: int = 3,  # Keep only the best N saves
                 total_epochs: int = 100,  # Total training epochs for warmup calculation
                 warmup_ratio: float = 0.25,  # Warmup period as ratio of total epochs
                 plateau_patience_epochs: int = 10,  # Epochs to check for plateau
                 composite_weight_fm: float = 0.3):  # Weight for FM in composite score
        self.ema_alpha = float(ema_alpha)
        self.min_delta = float(min_delta)
        self.min_save_interval = int(min_save_interval)
        self.significant_improvement_threshold = float(significant_improvement_threshold)
        self.max_best_saves = int(max_best_saves)
        self.total_epochs = int(total_epochs)
        self.warmup_epochs = int(total_epochs * warmup_ratio)  # 25% of total epochs by default
        self.plateau_patience_epochs = int(plateau_patience_epochs)
        self.composite_weight_fm = float(composite_weight_fm)

        # EMA tracking for all losses (for logging/monitoring)
        self.ema_gen = None
        self.ema_disc = None
        self.ema_mel = None
        self.ema_kl = None
        self.ema_fm = None

        # PRIMARY: Track best mel and composite scores (NOT gen loss)
        self.best_mel = float('inf')
        self.best_composite = float('inf')
        self.steps_since_best = 0
        self.steps_since_last_save = 0
        
        # Rolling window tracking for plateau detection (track mel over epochs)
        self.mel_history = deque(maxlen=self.plateau_patience_epochs)
        self.composite_history = deque(maxlen=self.plateau_patience_epochs)
        
        # Track validation trend for overfitting detection
        self.mel_uptrend_epochs = 0  # Count epochs where mel is increasing
        self.max_uptrend_patience = 5  # Stop if mel rises for 5+ epochs

        # Intelligent save tracking
        self.epochs_since_best_save = 0  # Track epochs since last best save
        self.best_saves_history = []  # Keep track of best saves for cleanup
        self.current_epoch = 0  # Track current epoch
        
        # Loss history for plotting (store epoch-level averages)
        self.epoch_history = []  # List of dicts with epoch number and losses

    def _ema(self, prev, val):
        if prev is None:
            return float(val)
        a = self.ema_alpha
        return (1.0 - a) * float(prev) + a * float(val)

    def compute_composite(self):
        """
        Compute composite quality score: mel + 0.3 × fm
        This combines intelligibility (mel) with naturalness (fm)
        """
        if self.ema_mel is None or self.ema_fm is None:
            return None
        return self.ema_mel + self.composite_weight_fm * self.ema_fm

    def update(self, loss_gen_all, loss_disc, loss_mel, loss_kl, loss_fm):
        """Update EMA for all losses (step-level tracking)"""
        self.ema_gen = self._ema(self.ema_gen, loss_gen_all)
        self.ema_disc = self._ema(self.ema_disc, loss_disc)
        self.ema_mel = self._ema(self.ema_mel, loss_mel)
        self.ema_kl = self._ema(self.ema_kl, loss_kl)
        self.ema_fm = self._ema(self.ema_fm, loss_fm)

        self.steps_since_last_save += 1
        self.steps_since_best += 1
        # NOTE: epochs_since_best_save is now incremented in on_epoch_end()
    
    def on_epoch_end(self, epoch: int):
        """
        Called at the end of each epoch to update epoch-based tracking.
        Tracks mel and composite score history for plateau detection.
        """
        self.current_epoch = epoch
        self.epochs_since_best_save += 1
        
        # Add current mel and composite to history for rolling window tracking
        if self.ema_mel is not None:
            self.mel_history.append(self.ema_mel)
        
        composite = self.compute_composite()
        if composite is not None:
            self.composite_history.append(composite)
        
        # Track if mel is trending upward (potential overfitting)
        if len(self.mel_history) >= 2:
            if self.mel_history[-1] > self.mel_history[-2]:
                self.mel_uptrend_epochs += 1
            else:
                self.mel_uptrend_epochs = 0
        
        # Store epoch-level loss snapshot for plotting
        if all(x is not None for x in [self.ema_mel, self.ema_fm, self.ema_gen, self.ema_disc, self.ema_kl]):
            self.epoch_history.append({
                'epoch': epoch,
                'mel': float(self.ema_mel),
                'fm': float(self.ema_fm),
                'composite': float(composite) if composite is not None else None,
                'gen': float(self.ema_gen),
                'disc': float(self.ema_disc),
                'kl': float(self.ema_kl)
            })

    def should_save_best(self) -> bool:
        """
        Check if current model should be saved based on mel loss (PRIMARY metric).
        Uses mel reconstruction loss as the key indicator of model quality.
        """
        if self.ema_mel is None:
            return False
        if self.ema_mel + self.min_delta < self.best_mel:
            self.best_mel = self.ema_mel
            composite = self.compute_composite()
            if composite is not None:
                self.best_composite = composite
            self.steps_since_best = 0
            return True
        return False

    def should_save_intelligent_best(self, current_epoch: int) -> tuple[bool, str]:
        """
        Intelligently decide if we should save a best model based on:
        1. Warmup period (no saves during initial 25% of training)
        2. Minimum interval between saves
        3. Significant improvement threshold (≥1% improvement in mel or composite)
        4. Maximum number of best saves to keep

        PRIMARY: Uses mel reconstruction loss
        SECONDARY: Uses composite score (mel + 0.3 × fm)

        Returns: (should_save, reason)
        """
        if self.ema_mel is None:
            return False, "No loss data"

        # Don't save during warmup period
        if current_epoch < self.warmup_epochs:
            return False, f"Still in warmup period (epoch {current_epoch}/{self.warmup_epochs})"

        # Check if enough epochs have passed since last best save
        if self.epochs_since_best_save < self.min_save_interval:
            return False, f"Only {self.epochs_since_best_save}/{self.min_save_interval} epochs since last best save"

        # Check if this is a significant improvement in mel loss
        if self.best_mel != float('inf'):
            improvement_ratio = (self.best_mel - self.ema_mel) / self.best_mel
            if improvement_ratio < self.significant_improvement_threshold:
                return False, f"Mel improvement {improvement_ratio:.3f} < threshold {self.significant_improvement_threshold}"

        # Check if we should update best and save (based on mel)
        if self.ema_mel + self.min_delta < self.best_mel:
            self.best_mel = self.ema_mel
            composite = self.compute_composite()
            if composite is not None:
                self.best_composite = composite
            self.steps_since_best = 0
            return True, f"New best mel loss achieved: {self.ema_mel:.4f}"

        return False, "No significant improvement"

    def set_epoch(self, epoch: int):
        """Update the current epoch tracking"""
        self.current_epoch = epoch

    def reset_best_save_counter(self):
        """Reset the counter when we save a best model"""
        self.epochs_since_best_save = 0

    def near_zero(self) -> bool:
        """Check if mel loss is near zero (exceptional case for early completion)"""
        return (self.ema_mel is not None) and (self.ema_mel <= 0.01)

    def should_early_stop(self) -> bool:
        """
        Decide if training should stop early based on mel loss plateau or overfitting.
        
        Stop conditions:
        1. Mel hasn't improved by ≥1% over last 8-10 epochs (plateau)
        2. Mel is trending upward for ≥5 consecutive epochs (overfitting)
        3. Composite score shows no improvement
        
        IGNORES: gen/disc losses (they oscillate and aren't reliable)
        """
        # Need enough history to make a decision
        if len(self.mel_history) < 8:
            return False
        
        # Check 1: Plateau detection - mel hasn't improved by ≥1% over rolling window
        oldest_mel = self.mel_history[0]
        current_mel = self.mel_history[-1]
        improvement_ratio = (oldest_mel - current_mel) / oldest_mel if oldest_mel > 0 else 0
        
        if improvement_ratio < self.significant_improvement_threshold:
            # Mel hasn't improved enough - potential plateau
            return True
        
        # Check 2: Overfitting detection - mel trending upward for too long
        if self.mel_uptrend_epochs >= self.max_uptrend_patience:
            return True
        
        return False

    def reset_after_save(self):
        self.steps_since_last_save = 0
        self.reset_best_save_counter()

    def add_best_save(self, save_path: str):
        """Add a best save to history and cleanup old ones if needed"""
        self.best_saves_history.append(save_path)

        # Keep only the best N saves
        if len(self.best_saves_history) > self.max_best_saves:
            # Remove oldest saves (keep the most recent N)
            saves_to_remove = self.best_saves_history[:-self.max_best_saves]
            for old_save in saves_to_remove:
                try:
                    if os.path.exists(old_save):
                        os.remove(old_save)
                        print(f"Cleaned up old best save: {old_save}")
                except Exception as e:
                    print(f"Failed to remove old best save {old_save}: {e}")
            self.best_saves_history = self.best_saves_history[-self.max_best_saves:]

    def status_str(self) -> str:
        """
        Status string highlighting PRIMARY metrics (mel/fm/composite) over gen/disc.
        """
        composite = self.compute_composite()
        composite_str = f"{composite:.4f}" if composite is not None else "N/A"
        
        # Calculate improvement over rolling window
        improvement_str = "N/A"
        if len(self.mel_history) >= 8:
            oldest_mel = self.mel_history[0]
            current_mel = self.mel_history[-1]
            improvement_pct = ((oldest_mel - current_mel) / oldest_mel * 100) if oldest_mel > 0 else 0
            improvement_str = f"{improvement_pct:.2f}%"
        
        return (
            f"EMA(mel={self.ema_mel:.4f} [best={self.best_mel:.4f}], "
            f"fm={self.ema_fm:.4f}, composite={composite_str} [best={self.best_composite:.4f}]) | "
            f"gen={self.ema_gen:.4f} disc={self.ema_disc:.4f} kl={self.ema_kl:.4f} | "
            f"improvement_over_{len(self.mel_history)}ep={improvement_str}, "
            f"uptrend_epochs={self.mel_uptrend_epochs}, "
            f"epochs_since_best_save={self.epochs_since_best_save}"
        )
    
    def plot_losses(self, save_path: str, project_name: str = "RVC"):
        """
        Plot and save loss curves with PRIMARY metrics (mel/fm/composite) prominent.
        
        Args:
            save_path: Path to save the plot image
            project_name: Name of the project for the plot title
        """
        if len(self.epoch_history) < 2:
            return  # Not enough data to plot
        
        try:
            # Extract data
            epochs = [h['epoch'] for h in self.epoch_history]
            mel_losses = [h['mel'] for h in self.epoch_history]
            fm_losses = [h['fm'] for h in self.epoch_history]
            composite_losses = [h['composite'] for h in self.epoch_history if h['composite'] is not None]
            gen_losses = [h['gen'] for h in self.epoch_history]
            disc_losses = [h['disc'] for h in self.epoch_history]
            kl_losses = [h['kl'] for h in self.epoch_history]
            
            # Create figure with 2 subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            fig.suptitle(f'Training Losses - {project_name}', fontsize=16, fontweight='bold')
            
            # Top plot: PRIMARY metrics (mel, fm, composite)
            ax1.set_title('PRIMARY Metrics (mel/fm/composite) - Use for Quality Assessment', fontweight='bold')
            ax1.plot(epochs, mel_losses, 'b-', linewidth=2, label='MEL (log-mel reconstruction)', marker='o', markersize=3)
            ax1.plot(epochs, fm_losses, 'g-', linewidth=2, label='FM (feature matching)', marker='s', markersize=3)
            if len(composite_losses) == len(epochs):
                ax1.plot(epochs, composite_losses, 'purple', linewidth=2.5, label='COMPOSITE (mel + 0.3×fm)', marker='D', markersize=3, linestyle='--')
            
            # Mark best mel
            if self.best_mel != float('inf'):
                best_mel_epoch = min(range(len(mel_losses)), key=lambda i: mel_losses[i])
                ax1.axhline(y=self.best_mel, color='b', linestyle=':', alpha=0.5, label=f'Best MEL: {self.best_mel:.4f}')
                ax1.plot(epochs[best_mel_epoch], mel_losses[best_mel_epoch], 'b*', markersize=15, label='Best MEL checkpoint')
            
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss Value')
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(left=1)
            
            # Bottom plot: SECONDARY metrics (gen, disc, kl) - for monitoring only
            ax2.set_title('SECONDARY Metrics (gen/disc/kl) - For Monitoring Only (NOT for decisions)', fontweight='bold', color='gray')
            ax2.plot(epochs, gen_losses, 'r-', linewidth=1.5, label='Generator Loss', alpha=0.7, marker='.')
            ax2.plot(epochs, disc_losses, 'orange', linewidth=1.5, label='Discriminator Loss', alpha=0.7, marker='.')
            ax2.plot(epochs, kl_losses, 'brown', linewidth=1.5, label='KL Divergence', alpha=0.7, marker='.')
            
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss Value')
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(left=1)
            
            # Add text annotation
            fig.text(0.5, 0.02, 
                    'Watch MEL (primary) and FM (secondary) for quality. Ignore GEN/DISC oscillations.',
                    ha='center', fontsize=10, style='italic', color='darkblue')
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            print(f"Error plotting losses: {e}")
            # Don't fail training if plotting fails
            pass


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
        early_stopped = train_and_evaluate(
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

        if early_stopped:
            if rank == 0:
                logger.info("Training stopped early due to loss plateau or uptrend.")
            break

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

        # Update global loss tracker and early-stop monitor
        global loss_tracker, early_stop_monitor
        if loss_tracker is None and rank == 0:
            # Initialize with mel-based tracking (PRIMARY metrics: mel/fm, IGNORE: gen/disc)
            loss_tracker = LossTracker(
                ema_alpha=0.05,
                min_delta=1e-4,
                min_save_interval=5,  # Save best models every 5 epochs minimum
                significant_improvement_threshold=0.01,  # 1% improvement required in mel
                max_best_saves=3,  # Keep only 3 best saves
                total_epochs=hps.train.epochs,  # Total training epochs for warmup
                warmup_ratio=0.25,  # 25% warmup period
                plateau_patience_epochs=10,  # 10 epoch rolling window for plateau detection
                composite_weight_fm=0.3,  # Weight for FM in composite score (mel + 0.3*fm)
            )
        if early_stop_monitor is None and rank == 0:
            early_stop_monitor = EarlyStoppingMonitor(
                ema_alpha=0.05,
                plateau_patience=getattr(hps.train, "early_stop_plateau_patience", 20),
                uptrend_patience=getattr(hps.train, "early_stop_uptrend_patience", 10),
                min_improvement_ratio=0.01,
                min_epochs=8,
                composite_weight_fm=0.3,
            )
        if rank == 0 and loss_tracker is not None:
            loss_tracker.update(
                float(loss_gen_all.detach().cpu()),
                float(loss_disc.detach().cpu()),
                float(loss_mel.detach().cpu()),
                float(loss_kl.detach().cpu()),
                float(loss_fm.detach().cpu()),
            )
            if early_stop_monitor is not None:
                early_stop_monitor.update(
                    float(loss_gen_all.detach().cpu()),
                    float(loss_disc.detach().cpu()),
                    float(loss_mel.detach().cpu()),
                    float(loss_kl.detach().cpu()),
                    float(loss_fm.detach().cpu()),
                )
            if global_step % hps.train.log_interval == 0:
                logger.info(f"[Tracker] {loss_tracker.status_str()}")

            # Early stop based on mel loss plateau or uptrend (actually stops training)
            if early_stop_monitor is not None and early_stop_monitor.should_stop():
                logger.info("[EarlyStop] %s", early_stop_monitor.reason())
                break

        global_step += 1

    early_stopped = False
    if rank == 0:
        # Update epoch tracking at the end of each epoch
        if loss_tracker is not None:
            loss_tracker.on_epoch_end(epoch)
        if early_stop_monitor is not None:
            early_stop_monitor.on_epoch_end(epoch)
            early_stopped = early_stop_monitor._stopped
        
        # Save model if it's time for a periodic save or if training is complete
        should_save = (epoch % hps.save_epoch_frequency == 0) or (epoch >= hps.train.epochs)

        # Also save if we have a very good loss (intelligent auto-save at epoch boundaries)
        if loss_tracker is not None:
            should_save_intelligent, reason = loss_tracker.should_save_intelligent_best(epoch)
            if should_save_intelligent:
                should_save = True
                logger.info(f"[Tracker] Auto-saving at epoch boundary: {reason}")

            # Also save if near zero mel loss (exceptional case - perfect reconstruction)
            if loss_tracker.near_zero():
                should_save = True
                logger.info("[Tracker] Auto-saving at epoch boundary due to near-zero mel loss (excellent reconstruction).")
        
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
            
            # Generate and save loss plot
            if loss_tracker is not None:
                plot_save_path = os.path.join(hps.model_dir, f"loss_plot_epoch{epoch}.png")
                loss_tracker.plot_losses(plot_save_path, project_name=hps.name)
                logger.info(f"Saved loss plot to {plot_save_path}")
                
                # Also save a copy to the trained directory with the model
                trained_plot_path = os.path.join(model_path, "trained", f"{model_name}_losses.png")
                try:
                    shutil.copy2(plot_save_path, trained_plot_path)
                    logger.info(f"Copied loss plot to {trained_plot_path}")
                except Exception as e:
                    logger.warning(f"Failed to copy loss plot to trained directory: {e}")

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

            # Track this save for intelligent cleanup if it was an intelligent best save
            if loss_tracker is not None and should_save_intelligent:
                loss_tracker.add_best_save(model_file_path)

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

            return early_stopped
    return early_stopped
