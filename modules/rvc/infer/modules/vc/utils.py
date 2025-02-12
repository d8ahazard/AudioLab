import os
from functools import lru_cache

import os
from functools import lru_cache

import librosa
import pyworld
import torch
from fairseq import checkpoint_utils
from torch.nn import functional as F

from handlers.config import model_path


def get_index_path_from_model(sid):
    sid_base = os.path.basename(sid)
    sid_base = sid_base.replace(".pth", "")
    sid_base = sid_base.replace("_final", "")
    # Find file that contains sid_base and .index
    for file in os.listdir(os.path.join(model_path, "trained")):
        if sid_base in file and ".index" in file:
            return file
    return None


def load_hubert(config):
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        [os.path.join(model_path, "rvc", "hubert_base.pt")],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    return hubert_model.eval()


def check_faiss_index_file(file_path):
    """
    Checks whether the given file appears to be a raw FAISS index file or a ZIP archive.

    Parameters:
        file_path (str): Path to the file to check.

    Returns:
        bool: True if the file appears to be a valid FAISS index (non-ZIP), False otherwise.
    """
    import os

    if not os.path.exists(file_path):
        print(f"File does not exist: {file_path}")
        return False

    with open(file_path, "rb") as f:
        header = f.read(4)
    if header == b"PK\x03\x04":
        print("The file appears to be a ZIP archive. Please unzip it to obtain the FAISS index file.")
        return False
    else:
        print("The file appears to be a valid FAISS index file (based on header check).")
        return True


def extract_index_from_zip(zip_path, extract_to):
    """
    Extracts files from a ZIP archive and returns the path to the first extracted file.

    Parameters:
        zip_path (str): Path to the ZIP archive.
        extract_to (str): Directory where the contents should be extracted.

    Returns:
        str: Path to the extracted index file.

    Raises:
        ValueError: If the file is not a valid ZIP archive or if extraction yields no files.
    """
    import os
    import zipfile

    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"{zip_path} is not a valid ZIP file.")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    extracted_files = [f for f in os.listdir(extract_to) if os.path.isfile(os.path.join(extract_to, f))]
    if not extracted_files:
        raise ValueError("No files were extracted from the ZIP archive.")

    # Assumes that the first extracted file is the FAISS index file
    extracted_file_path = os.path.join(extract_to, extracted_files[0])
    print(f"Extracted file: {extracted_file_path}")
    return extracted_file_path


@lru_cache
def cache_harvest_f0(input_audio_path, fs, f0max, f0min, frame_period):
    from modules.rvc.infer.modules.vc.pipeline import input_audio_path2wav

    audio = input_audio_path2wav[input_audio_path]
    f0, t = pyworld.harvest(
        audio,
        fs=fs,
        f0_ceil=f0max,
        f0_floor=f0min,
        frame_period=frame_period,
    )
    f0 = pyworld.stonemask(audio, f0, t, fs)
    return f0


def change_rms(data1, sr1, data2, sr2, rate):
    rms1 = librosa.feature.rms(
        y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2
    )  # 每半秒一个点
    rms2 = librosa.feature.rms(y=data2, frame_length=sr2 // 2 * 2, hop_length=sr2 // 2)
    rms1 = torch.from_numpy(rms1)
    rms1 = F.interpolate(rms1.unsqueeze(0), size=data2.shape[0], mode="linear").squeeze()
    rms2 = torch.from_numpy(rms2)
    rms2 = F.interpolate(rms2.unsqueeze(0), size=data2.shape[0], mode="linear").squeeze()
    rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-6)
    data2 *= (
            torch.pow(rms1, torch.tensor(1 - rate))
            * torch.pow(rms2, torch.tensor(rate - 1))
    ).numpy()
    return data2


