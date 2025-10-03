import json
import os
import numpy as np
import scipy
import librosa
import torch
from attributedict.collections import AttributeDict
from safetensors.torch import load_file as safe_load_file

from .config import (
    FRAME_RATE,
    TOLERANCES,
    SAMPLE_RATE,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    NORM,
    CRNN_MODEL_PATH,
    CRNN_MODEL_SAFE,
    CRNN_MODEL_PT,
    CRNN_CONFIG_JSON,
)
from .CRNN_model import CRNN


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load_crnn():
    if os.path.exists(CRNN_MODEL_SAFE) and os.path.exists(CRNN_CONFIG_JSON):
        with open(CRNN_CONFIG_JSON, "r", encoding="utf-8") as f:
            meta = json.load(f)
        config = AttributeDict(meta["config"])  # model hyperparams
        consts = AttributeDict(meta["consts"])  # numeric constants
        config.device = DEVICE
        model = CRNN(config, consts).to(DEVICE)
        state = safe_load_file(CRNN_MODEL_SAFE)
        model.load_state_dict(state, strict=False)
        model.eval()
        return model
    # Fallback: legacy .pt bundle
    save_data = torch.load(CRNN_MODEL_PT if os.path.exists(CRNN_MODEL_PT) else CRNN_MODEL_PATH, map_location=DEVICE)
    model_state_dict = save_data["model_state_dict"]
    config = AttributeDict(save_data["config"])  # contains model hyperparams
    consts = AttributeDict(save_data["consts"])  # numeric constants
    config.device = DEVICE
    model = CRNN(config, consts).to(DEVICE)
    model.load_state_dict(model_state_dict, strict=False)
    model.eval()
    return model


# Lazy singleton for the CRNN model
_CRNN_MODEL = None


def get_crnn_model():
    global _CRNN_MODEL
    if _CRNN_MODEL is None:
        _CRNN_MODEL = _load_crnn()
    return _CRNN_MODEL


def softmax_with_temperature(z, T=1):
    y = np.exp(z / T) / np.sum(np.exp(z / T), axis=0)
    return y


def process_chroma(y):
    chroma_stft = librosa.feature.chroma_stft(
        y=y,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT,
        norm=NORM,
        center=False,
    )
    return chroma_stft


def process_mel(y):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT,
        norm=NORM,
        n_mels=N_MELS,
        center=False,
    )
    return np.log1p(mel * 5) / 4


def process_phonemes(y):
    model = get_crnn_model()
    x_tensor = torch.from_numpy(y).float().to(DEVICE)

    with torch.no_grad():
        batch = {"audio": x_tensor}
        predictions = model.run_on_batch(batch, cal_loss=False)

    phonemes = predictions["frame"].squeeze().T.detach().cpu().numpy()  # [C, T]
    phonemes = phonemes[:, 1:-1]  # remove <sos>, <eos>
    phonemes = softmax_with_temperature(phonemes, T=1)
    phonemes = np.log1p(phonemes * 5) / 4
    return phonemes


def compute_strict_alignment_path_mask(P):
    P = np.array(P, copy=True)
    N, M = P[-1]
    keep_mask = (P[1:, 0] > P[:-1, 0]) & (P[1:, 1] > P[:-1, 1])
    keep_mask = np.concatenate(([True], keep_mask))
    keep_mask[(P[:, 0] == N) | (P[:, 1] == M)] = False
    keep_mask[-1] = True
    P_mod = P[keep_mask, :]
    return P_mod


def make_path_strictly_monotonic(P: np.ndarray) -> np.ndarray:
    P_mod = compute_strict_alignment_path_mask(P.T)
    return P_mod.T


def transfer_positions(wp, times_src, feature_rate=FRAME_RATE):
    x, y = wp[0] / feature_rate, wp[1] / feature_rate
    f = scipy.interpolate.interp1d(x, y, kind="linear", fill_value="extrapolate")
    mapped = f(times_src)
    return mapped


