import json
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Union, Callable

import faiss
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from handlers.config import model_path
from rvc.configs.config import Config
from rvc.infer.modules.train.extract_f0_rmvpe import FeatureInput
from rvc.infer.modules.train.extract_feature_print import extract_features
from rvc.infer.modules.train.extract_ppg import extract_ppg_features, diarize_and_create_speaker_mapping
from rvc.infer.modules.train.preprocess import PreProcess, preprocess_trainset
from rvc.train import run_new

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


def train_index(exp_dir, version="v2"):
    """
    Create and train a FAISS index for fast similarity search on audio features.

    Args:
        exp_dir (str): Path to the experiment directory containing feature files.
        version (str): Model version (e.g., "v1" or "v2").
    """
    exp_dir = Path(exp_dir)
    feature_dir = exp_dir / ("3_feature256" if version == "v1" else "3_feature768")

    if not feature_dir.exists() or not any(feature_dir.iterdir()):
        raise RuntimeError("Feature directory is empty or missing. Please extract features first!")

    logger.info("Starting FAISS index training...")
    features = []
    for feature_file in sorted(feature_dir.glob("*.npy")):
        features.append(np.load(feature_file))

    big_npy = np.concatenate(features, axis=0)
    logger.info(f"Aggregated feature shape: {big_npy.shape}")

    if big_npy.shape[0] > 200_000:
        logger.info(f"Reducing features using K-Means clustering to 10,000 centroids.")
        big_npy = MiniBatchKMeans(
            n_clusters=10_000,
            verbose=True,
            batch_size=256 * config.n_cpu,
            compute_labels=False,
            init="random"
        ).fit(big_npy).cluster_centers_

    np.save(exp_dir / "total_fea.npy", big_npy)

    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    logger.info(f"Creating FAISS index with {n_ivf} inverted lists.")

    index = faiss.index_factory(big_npy.shape[1], f"IVF{n_ivf},Flat")
    index.train(big_npy)
    logger.info("FAISS index training complete. Adding features to the index...")

    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i: i + batch_size_add])

    index_path = exp_dir / f"trained_IVF{n_ivf}_Flat_{version}.index"
    faiss.write_index(index, str(index_path))
    logger.info(f"FAISS index saved to {index_path}")


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
        # os.remove(str(audio_file))

    pp = PreProcess(sr, exp_dir, per, start_idx)
    pp.pipeline_mp_inp_dir(inp_root, n_p=n_p, name2id_save_path=name2id_save_path)
    print("end preprocess")


def preprocess_dataset(exp_dir, sr=44100, n_p=1, per=3.0, start_idx=0, callback: Callable = None):
    """
    Run the full dataset preprocessing pipeline.

    Args:
        exp_dir (str): Directory for experiment data.
        sr (int): Sampling rate for preprocessing.
        n_p (int): Number of parallel processes.
        per (float): Segment length for slicing.
        start_idx (int): Starting index for dataset naming.
        callback (Callable): Optional callback function to run after preprocessing.
    """
    inp_root = Path(exp_dir) / "raw"
    name2id_save_path = Path(exp_dir) / "name2id.json"

    # Step 1: Preprocess raw audio
    preprocess_trainset(inp_root, sr, n_p, exp_dir, per, start_idx, name2id_save_path, callback)
    print("Preprocessing raw audio complete.")

    # Step 2: Extract F0 features
    f0_input = FeatureInput(samplerate=16000, hop_size=160)
    wav_paths = list(Path(exp_dir).glob("1_16k_wavs/*.wav"))
    f0_paths = [
        (str(wav), str(Path(exp_dir) / "2a_f0" / f"{wav.stem}.npy"),
         str(Path(exp_dir) / "2b-f0nsf" / f"{wav.stem}.npy"))
        for wav in wav_paths
    ]
    f0_input.go(f0_paths, f0_method="rmvpe", callback=callback)
    print("F0 extraction complete.")

    # Step 3: Extract features
    extract_features(exp_dir=exp_dir, n_part=n_p, callback=callback)
    print("Feature extraction complete.")

    # Step 4: Extract PPG features
    extract_ppg_features(exp_dir=exp_dir, n_part=n_p, callback=callback)
    print("PPG feature extraction complete.")

    # Step 5: Create speaker mappings
    diarize_and_create_speaker_mapping(exp_dir, callback=callback)
    print("Speaker mapping complete.")


def prepare_dataset(
        exp_dir,
        sr,
        spk_mapping,
        config_path="v2/40k.json",
):
    """
    Prepares the dataset by gathering file names and writing a filelist.txt,
    then writes a config.json if it doesn't already exist.
    """

    try:
        cur_dir = Path(__file__).parent
        logger.info(f"Current dir: {cur_dir}")

        # Directories
        gt_wavs_dir = Path(exp_dir) / "0_gt_wavs"
        feature_dir = Path(exp_dir) / "3_feature768"
        f0_dir = Path(exp_dir) / "2a_f0"
        f0nsf_dir = Path(exp_dir) / "2b-f0nsf"

        # Log the directories we’re using
        logger.debug(f"gt_wavs_dir: {gt_wavs_dir}")
        logger.debug(f"feature_dir: {feature_dir}")
        logger.debug(f"f0_dir: {f0_dir}")
        logger.debug(f"f0nsf_dir: {f0nsf_dir}")

        # (Optional) Check directories exist
        for d in [gt_wavs_dir, feature_dir, f0_dir, f0nsf_dir]:
            if not d.exists():
                logger.warning(f"Directory does not exist: {d}")

        # Gather names
        # Get the base names without extensions
        def get_base_names(directory):
            return {os.path.splitext(name)[0] for name in os.listdir(directory)}

        gt_wavs_names = get_base_names(gt_wavs_dir)
        feature_names = get_base_names(feature_dir)
        f0_names = get_base_names(f0_dir)
        f0nsf_names = get_base_names(f0nsf_dir)

        # Find the intersection of all sets
        names = gt_wavs_names & feature_names & f0_names & f0nsf_names

        logger.debug(f"Number of matching names found: {len(names)}")

        opt = []
        for name in names:
            # Get the speaker mapping, defaulting to 0 if missing
            speaker = spk_mapping.get(name, spk_mapping.get(name.split("_")[0], 0))

            # Create the line using os.path.join for platform-independent paths
            line = "|".join([
                os.path.join(gt_wavs_dir, f"{name}.wav"),
                os.path.join(feature_dir, f"{name}.npy"),
                os.path.join(f0_dir, f"{name}.npy"),
                os.path.join(f0nsf_dir, f"{name}.npy"),
                str(speaker),
            ])
            opt.append(line)

        # Shuffle the list
        random.shuffle(opt)
        output_file = Path(exp_dir) / "filelist.txt"
        # Write to the output file
        with open(output_file, "w") as f:
            f.write("\n".join(opt))

        logger.info(f"Filelist written to {output_file}")
        # Save config if needed
        config_save_path = Path(exp_dir) / "config.json"
        if not config_save_path.exists():
            logger.debug(f"Config file not found. Creating {config_save_path}")

            config_save_path.write_text(
                json.dumps(
                    config.json_config[config_path],
                    ensure_ascii=False,
                    indent=4,
                    sort_keys=True,
                )
            )
        else:
            logger.debug(f"Config file already exists at {config_save_path}")

        logger.info("prepare_dataset completed successfully.")

    except Exception as e:
        logger.error("Error in prepare_dataset", exc_info=True)
        # Reraise if you want the calling code to handle the exception further
        raise


def download_pretrained_models():
    """Download all necessary pretrained models if they do not exist."""
    for model_name, url in model_urls.items():
        model_file = Path(model_dir) / f"{model_name}.pth"
        if not model_file.exists():
            logger.info(f"Downloading {model_name} from {url}...")
            os.system(f"curl -L -o {model_file} {url}")
            logger.info(f"Downloaded {model_name} to {model_file}.")
    rvc_model_url = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt"
    rvc_model_file = os.path.join(model_path, "rvc", "rmvpe.pt")
    os.makedirs(os.path.dirname(rvc_model_file), exist_ok=True)
    if not Path(rvc_model_file).exists():
        logger.info(f"Downloading RVC model from {rvc_model_url}...")
        os.system(f"curl -L -o {rvc_model_file} {rvc_model_url}")
        logger.info(f"Downloaded RVC model to {rvc_model_file}.")
    hubert_model_url = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt"
    hubert_model_file = os.path.join(model_path, "rvc", "hubert_base.pt")
    os.makedirs(os.path.dirname(hubert_model_file), exist_ok=True)
    if not Path(hubert_model_file).exists():
        logger.info(f"Downloading Hubert model from {hubert_model_url}...")
        os.system(f"curl -L -o {hubert_model_file} {hubert_model_url}")
        logger.info(f"Downloaded Hubert model to {hubert_model_file}.")
    that_other_fucking_stupid_model_url = "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt"
    that_other_fucking_stupid_model_file = os.path.join(model_path, "rvc", "large-v3.pt")
    os.makedirs(os.path.dirname(that_other_fucking_stupid_model_file), exist_ok=True)
    if not Path(that_other_fucking_stupid_model_file).exists():
        logger.info(f"Downloading that other fucking stupid model from {that_other_fucking_stupid_model_url}...")
        os.system(f"curl -L -o {that_other_fucking_stupid_model_file} {that_other_fucking_stupid_model_url}")
        logger.info(f"Downloaded that other fucking stupid model to {that_other_fucking_stupid_model_file}.")


def find_and_rename_final_model(exp_dir, voice_name):
    """
    Find the final trained model in the experiment directory and rename it to include the voice name.

    Args:
        exp_dir (str): The experiment directory where models are saved.
        voice_name (str): The desired name for the final model.
    """
    final_model = None
    final_index = None
    all_models = os.path.join(model_path, "trained")
    os.makedirs(all_models, exist_ok=True)

    for file in Path(exp_dir).glob("G_*.pth"):
        if not final_model or file.stat().st_mtime > final_model.stat().st_mtime:
            final_model = file
            break

    if final_model:
        new_name = Path(exp_dir) / f"{voice_name}_final.pth"
        shutil.move(final_model, new_name)
        # Copy the final model to the trained models directory
        shutil.copy(new_name, os.path.join(all_models, f"{voice_name}_final.pth"))
        logger.info(f"Renamed final model to: {new_name} and copied to {all_models}.")
    else:
        logger.warning("No final model found to rename.")

    for file in Path(exp_dir).glob("trained_IVF*.index"):
        if not final_index or file.stat().st_mtime > final_index.stat().st_mtime:
            final_index = file
            break

    if final_index:
        new_name = Path(exp_dir) / f"{voice_name}_index.index"
        shutil.move(final_index, new_name)
        # Copy the final index to the trained models directory
        shutil.copy(new_name, os.path.join(all_models, f"{voice_name}_index.index"))
        logger.info(f"Renamed final index to: {new_name}.")


def train_rvc_model(
        trainset_dir: str,
        exp_dir: str,
        voice_name: str,
        sr: str = "48k",
        total_epoch: int = 20,
        batch_size: int = 8,
        lr: float = 1.8e-4,
        lr_decay: float = 0.99,
        pretrained_g: str = None,
        pretrained_d: str = None,
        save_every_epoch: int = 5,
        n_gpus: int = 1,
        rank: int = 0,
        callback: Callable = None,
):
    """
    Main entry point to prepare dataset, download models, train, and finalize the model.

    Args:
        trainset_dir (str): Path to the training dataset.
        exp_dir (str): Path to the experiment directory.
        voice_name (str): Name for the final trained model.
        sr (str): Sampling rate for training (e.g., "48k").
        total_epoch (int): Total number of training epochs.
        batch_size (int): Batch size for training.
        lr (float): Learning rate.
        lr_decay (float): Learning rate decay factor.
        pretrained_g (str): Path to pretrained generator model.
        pretrained_d (str): Path to pretrained discriminator model.
        save_every_epoch (int): Save model checkpoint after this many epochs.
        n_gpus (int): Number of GPUs to use for training.
        rank (int): GPU rank for distributed training.
        callback (Callable): Optional callback function to run after training.
    """
    logger.info("Downloading pretrained models if necessary...")
    download_pretrained_models()

    logger.info("Starting preprocessing pipeline...")
    preprocess_dataset(
        exp_dir=exp_dir,
        sr=int(sr[:-1]) * 1000,
        n_p=4,  # Use parameterized n_p for flexibility
        per=3.0,
    )

    spk_mapping_file = Path(exp_dir) / "spk_map.json"
    with spk_mapping_file.open("r") as f:
        spk_mapping = json.load(f)

    logger.info("Preparing dataset for training...")
    prepare_dataset(
        exp_dir=exp_dir,
        sr=int(sr[:-1]) * 1000,
        spk_mapping=spk_mapping,
        config_path=f"v3/{sr}.json",
    )

    logger.info("Starting model training...")
    # run_new(
    #     exp_dir=exp_dir,
    #     data_path=trainset_dir,
    #     filter_length=1024,
    #     hop_length=256,
    #     win_length=1024,
    #     segment_size=8192,
    #     sampling_rate=int(sr[:-1]) * 1000,
    #     total_epochs=total_epoch,
    #     batch_size=batch_size,
    #     learning_rate=lr,
    #     lr_decay=lr_decay,
    #     pretrain_g=pretrained_g or os.path.join(model_dir, f"f0G{sr}.pth"),
    #     pretrain_d=pretrained_d or os.path.join(model_dir, f"f0D{sr}.pth"),
    #     save_every_epoch=save_every_epoch,
    #     n_gpus=n_gpus,
    #     rank=rank,
    # )

    logger.info("Training FAISS index...")
    train_index(exp_dir, version="v2")
    logger.info("FAISS index training complete.")

    logger.info("Training complete. Finalizing the model...")
    find_and_rename_final_model(exp_dir, voice_name)


