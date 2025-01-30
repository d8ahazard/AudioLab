import argparse
import os
import sys
import typing as tp
from pathlib import Path
from time import time

import numpy as np
import torch
import torchaudio
from omegaconf import OmegaConf
from tqdm import tqdm

from modules.yue.inference.xcodec_mini_infer.vocos.pretrained import VocosDecoder


def build_soundstream_model(config):
    model = eval(config.generator.name)(**config.generator.config)
    return model


def build_codec_model(config_path, vocal_decoder_path, inst_decoder_path):
    vocal_decoder = VocosDecoder.from_hparams(config_path=config_path)
    vocal_decoder.load_state_dict(torch.load(vocal_decoder_path))
    inst_decoder = VocosDecoder.from_hparams(config_path=config_path)
    inst_decoder.load_state_dict(torch.load(inst_decoder_path))
    return vocal_decoder, inst_decoder


def save_audio(wav: torch.Tensor, path: tp.Union[Path, str], sample_rate: int, rescale: bool = False):
    limit = 0.99
    mx = wav.abs().max()
    if rescale:
        wav = wav * min(limit / mx, 1)
    else:
        wav = wav.clamp(-limit, limit)

    path = str(Path(path).with_suffix('.mp3'))
    torchaudio.save(path, wav, sample_rate=sample_rate)


def process_audio(input_file, output_file, rescale, decoder, soundstream, cuda_idx: int = 0):
    compressed = np.load(input_file, allow_pickle=True).astype(np.int16)
    print(f"Processing {input_file}")
    print(f"Compressed shape: {compressed.shape}")
    compressed = torch.as_tensor(compressed, dtype=torch.long).unsqueeze(1)
    compressed = soundstream.get_embed(compressed.to(f"cuda:{cuda_idx}"))
    compressed = torch.tensor(compressed).to(f"cuda:{cuda_idx}")

    start_time = time()
    with torch.no_grad():
        decoder.eval()
        decoder = decoder.to(f"cuda:{cuda_idx}")
        out = decoder(compressed)
        out = out.detach().cpu()
    duration = time() - start_time
    rtf = (out.shape[1] / 44100.0) / duration
    print(f"Decoded in {duration:.2f}s ({rtf:.2f}x RTF)")
    print(f"Saving to {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    save_audio(out, output_file, 44100, rescale=rescale)
    print(f"Saved: {output_file}")
    return out


def find_matching_pairs(input_folder):
    if str(input_folder).endswith('.lst'):  # Convert to string
        with open(input_folder, 'r') as file:
            files = [line.strip() for line in file if line.strip()]
    else:
        files = list(Path(input_folder).glob('*.npy'))
    print(f"found {len(files)} npy.")
    instrumental_files = {}
    vocal_files = {}

    for file in files:
        if not isinstance(file, Path):
            file = Path(file)
        name = file.stem
        if 'instrumental' in name.lower():
            base_name = name.lower().replace('instrumental', '')  # .strip('_')
            instrumental_files[base_name] = file
        elif 'vocal' in name.lower():
            # base_name = name.lower().replace('vocal', '').strip('_')
            last_index = name.lower().rfind('vocal')
            if last_index != -1:
                # Create a new string with the last 'vocal' removed
                base_name = name.lower()[:last_index] + name.lower()[last_index + len('vocal'):]
            else:
                base_name = name.lower()
            vocal_files[base_name] = file

    # Find matching pairs
    pairs = []
    for base_name in instrumental_files.keys():
        if base_name in vocal_files:
            pairs.append((
                instrumental_files[base_name],
                vocal_files[base_name],
                base_name
            ))

    return pairs
