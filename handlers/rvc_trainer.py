import os
import json
import random
import shutil
from typing import Union

import torch
import logging
from pathlib import Path

from modules.rvc_train.configs.config import Config
from modules.rvc_train.infer.modules.train.preprocess import PreProcess
from modules.rvc_train.train import run_new
from handlers.config import model_path

logger = logging.getLogger(__name__)
logging.getLogger("numba").setLevel(logging.WARNING)

# Global setup
model_dir = os.path.join(model_path, "pretrained_v2")
os.makedirs(model_dir, exist_ok=True)
config = Config()
# Pretrained model URLs
model_urls = {
    "f0D32k": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/pretrained_v2/f0D32k.pth?download=true",
    "f0G32k": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/pretrained_v2/f0G32k.pth?download=true",
    "f0D48k": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/pretrained_v2/f0D48k.pth?download=true",
    "f0G48k": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/pretrained_v2/f0G48k.pth?download=true",
    "f0D40k": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/pretrained_v2/f0D40k.pth?download=true",
    "f0G40k": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/pretrained_v2/f0G40k.pth?download=true",
}


def mp3_to_wav(mp3_path: str):
    """
    Convert a mp3 file to wav format.

    Args:
        mp3_path (str): Path to the mp3 file.
    """
    wav_path = mp3_path.replace(".mp3", ".wav")
    os.system(f"ffmpeg -i {mp3_path} -acodec pcm_s16le -ac 1 -ar 44100 {wav_path}")
    return wav_path


def preprocess_data(
        inp_root: Union[str, Path],
        sr: int = 44100,
        n_p: int = 1,
        exp_dir: str = "logs/mute",
        per: float = 3.0,
        name2id_save_path: str = "logs/mute/name2id.json",
        start_idx: int = 0,
):
    """
    Preprocess the training data.

    Args:
        inp_root (str): Path to the input root directory.
        sr (int): Sampling rate for the training data.
        n_p (int): Number of parallel processes.
        exp_dir (str): Path to the experiment directory.
        per (float): Perceptual entropy threshold.
        name2id_save_path (str): Path to save the mapping name2idx.
        start_idx (int): Start index for new datasets.
    """
    if isinstance(inp_root, str):
        inp_root = Path(inp_root)
    if isinstance(name2id_save_path, str):
        name2id_save_path = Path(name2id_save_path)

    for audio_file in Path(inp_root).rglob("*.mp3"):
        _ = mp3_to_wav(str(audio_file))
        #os.remove(str(audio_file))

    pp = PreProcess(sr, exp_dir, per, start_idx)
    pp.pipeline_mp_inp_dir(inp_root, n_p=n_p, name2id_save_path=name2id_save_path)
    print("end preprocess")


def prepare_dataset(
        exp_dir,
        sr,
        spk_mapping,
        config_path=f"v2/40k.json",
):
    cur_dir = Path(__file__).parent
    logger.info(f"Current dir: {cur_dir}")
    # filelist
    gt_wavs_dir = f"{exp_dir}/0_gt_wavs"
    feature_dir = f"{exp_dir}/3_feature768"
    f0_dir = f"{exp_dir}/2a_f0"
    f0nsf_dir = f"{exp_dir}/2b-f0nsf"
    names = (
            set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
            & set([name.split(".")[0] for name in os.listdir(feature_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
    )
    opt = []
    for name in names:
        # with f0
        opt.append(
            "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
            % (
                gt_wavs_dir.replace("\\", "\\\\"),
                name,
                feature_dir.replace("\\", "\\\\"),
                name,
                f0_dir.replace("\\", "\\\\"),
                name,
                f0nsf_dir.replace("\\", "\\\\"),
                name,
                spk_mapping.get(name, spk_mapping[f"{name.split('_')[0]}"]),
            )
        )

    feature_dim = 768
    # for _ in range(2):
    #     opt.append(
    #         f"{cur_dir}/logs/mute/0_gt_wavs/mute{sr}.wav|{cur_dir}/logs/mute/3_feature{feature_dim}/mute.npy|{cur_dir}/logs/mute/2a_f0/mute.wav.npy|{cur_dir}/logs/mute/2b-f0nsf/mute.wav.npy|0"
    #     )
    random.shuffle(opt)
    Path(f"{exp_dir}/filelist.txt").write_text("\n".join(opt))

    logger.debug("Write filelist done")
    config_save_path = Path(exp_dir) / "config.json"
    if not config_save_path.exists():
        config_save_path.write_text(
            json.dumps(
                config.json_config[config_path],
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
            )
        )


def download_pretrained_models():
    """Download all necessary pretrained models if they do not exist."""
    for model_name, url in model_urls.items():
        model_file = Path(model_dir) / f"{model_name}.pth"
        if not model_file.exists():
            logger.info(f"Downloading {model_name} from {url}...")
            os.system(f"curl -L -o {model_file} {url}")
            logger.info(f"Downloaded {model_name} to {model_file}.")


def find_and_rename_final_model(exp_dir, voice_name):
    """
    Find the final trained model in the experiment directory and rename it to include the voice name.

    Args:
        exp_dir (str): The experiment directory where models are saved.
        voice_name (str): The desired name for the final model.
    """
    final_model = None
    for file in Path(exp_dir).glob("G_*.pth"):
        if not final_model or file.stat().st_mtime > final_model.stat().st_mtime:
            final_model = file

    if final_model:
        new_name = Path(exp_dir) / f"{voice_name}_final.pth"
        shutil.move(final_model, new_name)
        logger.info(f"Renamed final model to: {new_name}")
    else:
        logger.warning("No final model found to rename.")


def train_rvc_model(
        trainset_dir: str,
        exp_dir: str,
        voice_name: str,
        sr: str = "48k",
        total_epoch: int = 20,
        batch_size: int = 8,
        lr: float = 1.8e-4,
        lr_decay: float = 0.99,
        pretrained_g: str = os.path.join(model_dir, "f0G48k.pth"),
        pretrained_d: str = os.path.join(model_dir, "f0D48k.pth"),
        save_every_epoch: int = 5,
        n_gpus: int = 1,
        rank: int = 0,
):
    """
    Main entry point to prepare dataset, download models, train, and rename the final model.

    Args:
        trainset_dir (str): Path to the training dataset.
        exp_dir (str): Path to the experiment directory.
        voice_name (str): Name for the final trained model.
        sr (str): Sampling rate for training.
        total_epoch (int): Total number of training epochs.
        batch_size (int): Batch size for training.
        lr (float): Learning rate.
        lr_decay (float): Learning rate decay.
        pretrained_g (str): Path to pretrained generator model.
        pretrained_d (str): Path to pretrained discriminator model.
        save_every_epoch (int): Save model after this many epochs.
        n_gpus (int): Number of GPUs for training.
        rank (int): GPU rank for distributed training.
    """
    logger.info("Checking and downloading pretrained models...")
    download_pretrained_models()

    logger.info("Preparing dataset...")
    preprocess_data(
        inp_root=trainset_dir,
        sr=int(sr[:-1]) * 1000,
        exp_dir=exp_dir,
        name2id_save_path=f"{exp_dir}/name2id.json",
    )
    spk_mapping = {}  # Adjust this to load actual speaker mapping
    prepare_dataset(
        exp_dir=exp_dir,
        sr=sr,
        spk_mapping=spk_mapping,
        config_path=f"v2/{sr}.json",
    )

    logger.info("Starting training...")
    run_new(
        exp_dir=exp_dir,
        data_path=trainset_dir,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        segment_size=8192,
        sampling_rate=int(sr[:-1]) * 1000,
        total_epochs=total_epoch,
        batch_size=batch_size,
        learning_rate=lr,
        lr_decay=lr_decay,
        pretrain_g=pretrained_g,
        pretrain_d=pretrained_d,
        save_every_epoch=save_every_epoch,
        n_gpus=n_gpus,
        rank=rank,
    )

    logger.info("Training completed. Renaming final model...")
    find_and_rename_final_model(exp_dir, voice_name)
