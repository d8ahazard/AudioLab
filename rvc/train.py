import datetime
import json
import logging
import os
import sys
import time
from collections import OrderedDict
from typing import Callable

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader
#from tqdm import tqdm

import rvc.lib.commons as commons
from rvc.data_preparation import (
    DistributedBucketSampler,
    TextAudioCollate,
    TextAudioLoader,
)
from rvc.lib.discriminator import MultiPeriodDiscriminatorV2 as MultiPeriodDiscriminator
from rvc.lib.models import RVCModel
from rvc.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from rvc.training_utils import (
    discriminator_loss,
    feature_loss,
    generator_loss,
    kl_loss,
)
from rvc.utils import load_checkpoint, latest_checkpoint_path, \
    save_checkpoint, plot_spectrogram_to_numpy, savee, HParams, get_hparams_from_file

has_wandb = False
try:
    # noinspection PyUnresolvedReferences
    import wandb

    has_wandb = True
except:
    pass

logger = logging.getLogger(__name__)

now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir))

global_step = 0
#train_progress = tqdm(desc="Training")


class EpochRecorder:
    def __init__(self):
        self.last_time = time.time()

    def record(self):
        now_time = time.time()
        elapsed_time = now_time - self.last_time
        self.last_time = now_time
        elapsed_time_str = str(datetime.timedelta(seconds=elapsed_time))
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{current_time}] | ({elapsed_time_str})"


def prepare_data_loaders(hps, n_gpus, rank):
    train_dataset = TextAudioLoader(hps.data.training_files, hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size * n_gpus,
        [100, 200, 300, 400, 500, 600, 700, 800, 900],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    collate_fn = TextAudioCollate()
    train_loader = DataLoader(
        train_dataset,
        num_workers=16,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=8,
    )
    return train_loader


def initialize_models_and_optimizers(hps):
    net_g = RVCModel(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    )
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm)

    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        weight_decay=0.02,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    return net_g, net_d, optim_g, optim_d


def load_model_checkpoint(hps, train_loader, net_g, net_d, optim_g, optim_d, rank, logger):
    try:
        _, _, _, epoch_str = load_checkpoint(
            latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d, load_opt=hps.enable_opt_load
        )
        if rank == 0:
            logger.info(f"loaded D")
        _, _, _, epoch_str = load_checkpoint(
            latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g, load_opt=hps.enable_opt_load
        )
        global_step = (epoch_str - 1) * len(train_loader)
    except:  # If cannot load, load pretrain
        epoch_str = 1
        global_step = 0
        if hps.pretrainG != "":
            if rank == 0:
                logger.info(f"loaded pretrained {hps.pretrainG}")
            state_dict_g = torch.load(hps.pretrainG, map_location="cpu")["model"]
            if hasattr(net_g, "module"):
                logger.info(
                    net_g.module.load_state_dict(state_dict_g)
                )
            else:
                if hps.enable_opt_load == 0:
                    excluded_keys = {"emb_g.weight"}
                    new_sd = OrderedDict()
                    for k, v in state_dict_g.items():
                        if k not in excluded_keys:
                            new_sd[k] = v
                    state_dict_g = new_sd

                logger.info(
                    net_g.load_state_dict(state_dict_g, strict=False)
                )
        if hps.pretrainD != "":
            if rank == 0:
                logger.info(f"loaded pretrained {hps.pretrainD}")
            state_dict_d = torch.load(hps.pretrainD, map_location="cpu")["model"]
            if hasattr(net_d, "module"):
                logger.info(
                    net_d.module.load_state_dict(state_dict_d)
                )
            else:
                logger.info(
                    net_d.load_state_dict(state_dict_d)
                )
    return epoch_str, global_step


def setup_schedulers(hps, net_g, net_d, optim_g, optim_d, epoch_str):
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )

    return scheduler_g, scheduler_d


def log_metrics(scalar_dict, image_dict, global_step):
    if not has_wandb:
        return
    wandb.log(scalar_dict, step=global_step)
    image_dict_wandb = {k: wandb.Image(v) for k, v in image_dict.items()}
    wandb.log(image_dict_wandb, step=global_step)


def run_training_epoch(epoch, hps, nets, optims, schedulers, train_loader, logger, accelerator, callback: Callable = None):
    net_g, net_d = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers
    global global_step
    if epoch % hps.save_interval == 0 or epoch <= 1:
        accelerator.save_state(hps.model_dir + f"/checkpoint_{epoch}")
        save_checkpoint(
            accelerator.unwrap_model(net_g),
            optim_g,
            hps.train.learning_rate,
            epoch,
            os.path.join(hps.model_dir, f"G_{global_step}.pth"),
            accelerator
        )
        save_checkpoint(
            accelerator.unwrap_model(net_d),
            optim_d,
            hps.train.learning_rate,
            epoch,
            os.path.join(hps.model_dir, f"D_{global_step}.pth"),
            accelerator
        )
        logger.info(f"saved ckpt {hps.model_dir}/checkpoint_{epoch}")
        if hps.save_every_weights == "1":
            logger.info(f"saved ckpt {hps.model_dir}/checkpoint_{epoch} (save_every)")
            ckpt = accelerator.get_state_dict(net_g)
            savee(
                ckpt,
                hps.sample_rate,
                hps.name + "_e%s_s%s" % (epoch, global_step),
                epoch,
                hps.version,
                hps,
            )
            logger.info(f"saving ckpt {hps.name}_e{epoch}:{global_step}")

    net_g.train()
    net_d.train()

    epoch_recorder = EpochRecorder()
    print("epoch recorder created, starting training")
    for batch_idx, data in enumerate(train_loader):
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
            ppg
        ) = data

        model_output = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid,
                             ppg=ppg, enable_perturbation=False)
        y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = model_output

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
        wave = commons.slice_segments(
            wave, ids_slice * hps.data.hop_length, hps.train.segment_size
        )

        y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
            y_d_hat_r, y_d_hat_g
        )

        optim_d.zero_grad()
        accelerator.backward(loss_disc)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), 1000.0)
        optim_d.step()

        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
        loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, losses_gen = generator_loss(y_d_hat_g)
        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl

        optim_g.zero_grad()
        accelerator.backward(loss_gen_all)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), 1000.0)
        optim_g.step()

        if batch_idx % hps.train.log_interval == 0:
            lr = optim_g.param_groups[0]["lr"]
            logger.info(
                f"Train Epoch: {epoch} [{100.0 * batch_idx / len(train_loader):.0f}%]"
            )
            logger.info([global_step, lr])
            logger.info(
                f"loss_disc={loss_disc:.3f}, loss_gen={loss_gen:.3f}, loss_fm={loss_fm:.3f},loss_mel={loss_mel:.3f}, loss_kl={loss_kl:.3f}"
            )
            if has_wandb:
                try:
                    scalar_dict = {
                        "loss/total/g": loss_gen_all,
                        "loss/total/d": loss_disc,
                        "learning_rate": lr,
                        "step": global_step,
                        "grad_norm_d": grad_norm_d,
                        "grad_norm_g": grad_norm_g,
                        "loss/g/fm": loss_fm,
                        "loss/g/mel": loss_mel,
                        "loss/g/kl": loss_kl,
                        "loss/g/gen": loss_gen,
                    }
                    image_dict = {
                        "slice/mel_org": plot_spectrogram_to_numpy(
                            y_mel[0].data.cpu().numpy()
                        ),
                        "slice/mel_gen": plot_spectrogram_to_numpy(
                            y_hat_mel[0].data.cpu().numpy()
                        ),
                        "all/mel": plot_spectrogram_to_numpy(
                            mel[0].data.cpu().numpy()
                        ),
                    }
                    log_metrics(scalar_dict, image_dict, global_step)
                except:
                    pass

        global_step += 1
        if callback is not None:
            desc_string = f"Epoch {epoch} | Step {global_step}"
            callback(global_step, f"Training ({desc_string})", len(train_loader))

    logger.info(f"====> Epoch: {epoch} {epoch_recorder.record()}")
    if epoch >= hps.total_epoch or global_step >= hps.train.total_steps:
        logger.info("Training is done. The program is closed.")

        if hasattr(net_g, "module"):
            ckpt = net_g.module.state_dict()
        else:
            ckpt = net_g.state_dict()
        savee(
            ckpt, hps.model.sr, hps.name, epoch, hps.version, hps
        )
        logger.info(f"saving final ckpt: {hps.name}")
        time.sleep(1)
        # os._exit(2333333)


def run_new(
        exp_dir: str,
        data_path: str,
        filter_length: int,
        hop_length: int,
        win_length: int,
        segment_size: int,
        sampling_rate: int,
        total_epochs: int,
        batch_size: int,
        learning_rate: float,
        lr_decay: float,
        pretrain_g: str = None,
        pretrain_d: str = None,
        save_every_epoch: int = 5,
        n_gpus: int = 1,
        rank: int = 0,
        callback: Callable = None,
        **kwargs,  # Allow passing extra parameters if needed
):
    """
    Simplified wrapper to call the original run() function by creating an hparams object.

    Args:
        All parameters required for training, passed directly.
    """
    # Create the hparams object
    hparams = HParams(
        exp_dir=exp_dir,
        data_path=data_path,
        filter_length=filter_length,
        hop_length=hop_length,
        win_length=win_length,
        segment_size=segment_size,
        sampling_rate=sampling_rate,
        total_epochs=total_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        lr_decay=lr_decay,
        pretrain_g=pretrain_g,
        pretrain_d=pretrain_d,
        save_every_epoch=save_every_epoch,
        n_gpus=n_gpus,
        rank=rank,
        **kwargs,  # Pass additional attributes if needed
    )

    hps = get_hparams_from_file(os.path.join(exp_dir, "config.json"))
    hps.data.training_files = os.path.join(exp_dir, "filelist.txt")
    hps.model.sr = sampling_rate
    hps.data.data_root = None
    hps.model_dir = hps.experiment_dir = exp_dir
    hps.save_dir = os.path.join(exp_dir, "samples")
    os.makedirs(hps.save_dir, exist_ok=True)
    hps.save_interval = 5
    hps.total_epoch = total_epochs
    hps.pretrainG = pretrain_g
    hps.pretrainD = pretrain_d
    hps.enable_opt_load = 1
    hps.train.batch_size = batch_size
    hps.train.total_steps = 10000000
    hps.save_every_weights = 0
    hps.if_cache_data_in_gpu = 1
    hps.train.learning_rate = learning_rate
    hps.train.lr_decay = lr_decay
    hps.train.log_interval = 100
    hps.resume = kwargs.get("resume", None)
    # Maybe not maybe?
    hps.train.seed = kwargs.get("seed", 42)  # Default seed
    hps.train.epochs = total_epochs
    hps.version = "v2"  # Optional versioning
    voice_name = os.path.basename(exp_dir)
    hps.name = kwargs.get("name", voice_name)  # Default experiment name

    for key, value in hparams.items():
        if key not in hps:
            hps[key] = value

    hps.if_ppg = 0
    hps.enable_perturbation = 0
    hps.if_latest = 0

    print(f"Running with hparams: {json.dumps(hps.__repr__(), indent=2)}")

    global global_step

    # if rank == 0:
    # logger = get_logger(hps.model_dir)
    # logger.info(hps)
    # for _ in range(3):
    #     try:
    #         wandb.init(project=wandb_project_name, name=hps.name, config=hps)
    #         break
    #     except:
    #         time.sleep(1)
    #         print("wandb init failed")
    #         pass

    torch.manual_seed(hps.train.seed)
    print("torch seed set")
    train_loader = prepare_data_loaders(hps, n_gpus, rank)

    total_steps = 0
    total_epochs_steps = total_epochs * len(train_loader)
    # if epoch >= hps.total_epoch or global_step >= hps.train.total_steps
    total_steps = total_epochs_steps if total_epochs_steps < hps.train.total_steps else hps.train.total_steps

    def progress_callback(
            step: float | tuple[int, int | None] | None,
            desc: str | None = None,
            total: int | None = None
    ):
        if callback:
            callback(step, desc, total_steps)

    print("data loaders prepared")
    net_g, net_d, optim_g, optim_d = initialize_models_and_optimizers(hps)
    print("models and optimizers initialized")
    epoch_str, global_step = load_model_checkpoint(hps, train_loader, net_g, net_d, optim_g, optim_d, rank, logger)
    print("model checkpoint loaded")
    scheduler_g, scheduler_d = setup_schedulers(hps, net_g, net_d, optim_g, optim_d, epoch_str)
    print("schedulers setup")
    # Explicit device settings for debugging
    if not torch.cuda.is_available():
        print("CUDA is not available. Ensure CUDA is properly installed and configured.")
    else:
        device = torch.device("cuda")
        print(f"Using device: {device}")

    # Check Accelerator's device allocation
    accelerator = Accelerator()
    print(f"Accelerator is using device: {accelerator.device}")
    print("accelerator initialized")
    train_loader, net_g, net_d, optim_g, optim_d, scheduler_g, scheduler_d = accelerator.prepare(
        train_loader, net_g, net_d, optim_g, optim_d, scheduler_g, scheduler_d
    )
    print("accelerator prepared")
    if hps.resume is not None:
        accelerator.load_state(hps.resume)
        print("loaded accelerator state provided by --resume")

    for epoch in range(epoch_str, hps.train.epochs + 1):
        if global_step > hps.train.total_steps:
            break
        run_training_epoch(
            epoch,
            hps,
            [net_g, net_d],
            [optim_g, optim_d],
            [scheduler_g, scheduler_d],
            train_loader,
            logger if rank == 0 else None,
            accelerator,
            progress_callback
        )
        scheduler_g.step()
        scheduler_d.step()
