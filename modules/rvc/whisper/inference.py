import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import argparse
import torch

from modules.rvc.whisper.model import Whisper, ModelDimensions
from modules.rvc.whisper.audio import load_audio, log_mel_spectrogram


def load_model(path, device) -> Whisper:
    checkpoint = torch.load(path, map_location="cpu")
    dims = ModelDimensions(**checkpoint["dims"])
    # print(dims)
    model = Whisper(dims)
    del model.decoder
    cut = len(model.encoder.blocks) // 4
    cut = -1 * cut
    del model.encoder.blocks[cut:]
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    if not (device == "cpu"):
        model.half()
    model.to(device)
    # torch.save({
    #     'dims': checkpoint["dims"],
    #     'model_state_dict': model.state_dict(),
    # }, "large-v2.pt")
    return model


def pred_ppg(whisper: Whisper, wav_path, ppg_path, device):
    audio = load_audio(wav_path)
    audln = audio.shape[0]
    ppg_a = []
    idx_s = 0
    while idx_s + 15 * 16000 < audln:
        short = audio[idx_s:idx_s + 15 * 16000]
        idx_s = idx_s + 15 * 16000
        ppg_length = 15 * 16000 // 320
        # short = pad_or_trim(short)
        mel = log_mel_spectrogram(short).to(device)
        if not (device == "cpu"):
            mel = mel.half()
        with torch.no_grad():
            mel = mel + torch.randn_like(mel) * 0.1
            # ppg = whisper.encoder(mel.unsqueeze(0))[:, :, :, None].squeeze().transpose(1, 2).permute(0, 2, 1).reshape(
            #     -1, whisper.output_dim).data.cpu().numpy()
            ppg = whisper.encoder(mel.unsqueeze(0)).squeeze().data.cpu().float().numpy()
            ppg = ppg[:ppg_length, ]  # [length, dim=1024]
            ppg_a.extend(ppg)
    if (idx_s < audln):
        short = audio[idx_s:audln]
        ppg_length = (audln - idx_s) // 320
        # short = pad_or_trim(short)
        mel = log_mel_spectrogram(short).to(device)
        if not (device == "cpu"):
            mel = mel.half()
        with torch.no_grad():
            mel = mel + torch.randn_like(mel) * 0.1
            ppg = whisper.encoder(mel.unsqueeze(0)).squeeze().data.cpu().float().numpy()
            ppg = ppg[:ppg_length, ]  # [length, dim=1024]
            ppg_a.extend(ppg)
    np.save(ppg_path, ppg_a, allow_pickle=False)
