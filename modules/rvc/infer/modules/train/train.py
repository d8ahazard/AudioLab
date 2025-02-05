import datetime
import logging
import os
from random import randint, shuffle
import gradio as gr
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

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

    dist.init_process_group("gloo", "env://", world_size=n_gpus, rank=rank)
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
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        pass
    elif torch.cuda.is_available():
        net_g = DDP(net_g, device_ids=[rank])
        net_d = DDP(net_d, device_ids=[rank])
    else:
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
        if rank == 0:
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
            )
        else:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, None],
                None,
                None,
                cache,
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


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers, cache):
    net_g, net_d = nets
    optim_g, optim_d = optims
    train_loader, eval_loader = loaders
    if writers is not None:
        _, _ = writers

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()

    # Prepare data iterator (with caching and TQDM)
    if hps.if_cache_data_in_gpu:
        if not cache:
            if rank == 0:
                for batch_idx, info in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch} (caching)"):
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
        # Check global step limit
        # if global_step >= hps.train.total_steps:
        #     if rank == 0:
        #         data_iterator.close()
        #     break

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

        if rank == 0 and global_step % hps.train.log_interval == 0:
            lr = optim_g.param_groups[0]["lr"]
            logger.info("Train Epoch: {} [{:.0f}%]".format(epoch, 100.0 * batch_idx / len(train_loader)))
            logger.info([global_step, lr])
            logger.info(f"loss_disc={loss_disc:.3f}, loss_gen={loss_gen:.3f}, loss_fm={loss_fm:.3f}, loss_mel={loss_mel:.3f}, loss_kl={loss_kl:.3f}")

        global_step += 1

    if rank == 0 and (epoch % hps.save_every_epoch == 0):
        if hps.if_latest == 0:
            utils.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "saves", "G_{}.pth".format(global_step)),
            )
            utils.save_checkpoint(
                net_d,
                optim_d,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "saves", "D_{}.pth".format(global_step)),
            )
        else:
            utils.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "saves", "G_{}.pth".format(2333333)),
            )
            utils.save_checkpoint(
                net_d,
                optim_d,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "saves", "D_{}.pth".format(2333333)),
            )
        if rank == 0 and hps.save_every_weights == "1":
            if hasattr(net_g, "module"):
                ckpt = net_g.module.state_dict()
            else:
                ckpt = net_g.state_dict()
            logger.info(
                "saving ckpt %s_e%s:%s"
                % (
                    hps.name,
                    epoch,
                    savee(
                        ckpt,
                        hps.sample_rate,
                        hps.if_f0,
                        hps.name + "_e%s_s%s" % (epoch, global_step),
                        epoch,
                        hps.version,
                        hps,
                    ),
                )
            )

    if epoch >= hps.train.epochs and rank == 0:
        logger.info("Training is done. The program is closed.")
        if hasattr(net_g, "module"):
            ckpt = net_g.module.state_dict()
        else:
            ckpt = net_g.state_dict()
        logger.info("saving final ckpt:%s" % (savee(ckpt, hps.sample_rate, hps.if_f0, hps.name, epoch, hps.version, hps)))
        return
