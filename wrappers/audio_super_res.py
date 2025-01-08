import gc
import os
import warnings

import librosa
import numpy as np
import soundfile as sf
import torch
from scipy import signal

from modules.versatile_audio_super_resolution.audiosr import build_model, super_resolution
from wrappers.base_wrapper import BaseWrapper

# Suppress warnings
warnings.filterwarnings("ignore")

# Set environment variables and Torch settings
os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.set_float32_matmul_precision("high")


def match_array_shapes(array_1: np.ndarray, array_2: np.ndarray):
    if (len(array_1.shape) == 1) & (len(array_2.shape) == 1):
        if array_1.shape[0] > array_2.shape[0]:
            array_1 = array_1[:array_2.shape[0]]
        elif array_1.shape[0] < array_2.shape[0]:
            array_1 = np.pad(array_1, (array_2.shape[0] - array_1.shape[0], 0), 'constant', constant_values=0)
    else:
        if array_1.shape[1] > array_2.shape[1]:
            array_1 = array_1[:, :array_2.shape[1]]
        elif array_1.shape[1] < array_2.shape[1]:
            padding = array_2.shape[1] - array_1.shape[1]
            array_1 = np.pad(array_1, ((0, 0), (0, padding)), 'constant', constant_values=0)
    return array_1


def lr_filter(audio, cutoff, filter_type, order=12, sr=48000):
    audio = audio.T
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order // 2, normal_cutoff, btype=filter_type, analog=False)
    sos = signal.tf2sos(b, a)
    filtered_audio = signal.sosfiltfilt(sos, audio)
    return filtered_audio.T


class AudioSuperRes(BaseWrapper):
    def __init__(self):
        self.model_name = None
        self.device = None
        self.sr = None
        self.audiosr = None
        super().__init__()

    def register_api_endpoint(self, api):
        pass

    def setup(self, model_name="basic", device="auto"):
        self.model_name = model_name
        self.device = device
        self.sr = 48000
        print("Loading Model...")
        self.audiosr = build_model(model_name=self.model_name, device=self.device)
        print("Model loaded!")

    def process_audio(self, inputs, chunk_size=5.12, overlap=0.1, seed=None, guidance_scale=3.5, ddim_steps=50):
        print(f"Upscaling audio from {inputs}")
        tgt_file = inputs[0]  # Assuming a single file is passed as a list
        audio, sr = librosa.load(tgt_file, sr=self.sr * 2, mono=False)
        temp_dir = os.path.join(os.path.dirname(tgt_file), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        audio = audio.T
        print(f"audio.shape = {audio.shape}")

        # Determine if audio is stereo or mono
        is_stereo = len(audio.shape) == 2
        audio_channels = [audio] if not is_stereo else [audio[:, 0], audio[:, 1]]

        # Chunk processing
        chunk_samples = int(chunk_size * sr)
        overlap_samples = int(overlap * chunk_samples)
        output_chunk_samples = int(chunk_size * self.sr)
        output_overlap_samples = int(overlap * output_chunk_samples)
        enable_overlap = overlap > 0

        def process_chunks(audio):
            chunks = []
            start = 0
            while start < len(audio):
                end = min(start + chunk_samples, len(audio))
                chunk = audio[start:end]
                if len(chunk) < chunk_samples:
                    chunk = np.concatenate([chunk, np.zeros(chunk_samples - len(chunk))])
                chunks.append(chunk)
                start += chunk_samples - overlap_samples if enable_overlap else chunk_samples
            return chunks

        chunks_per_channel = [process_chunks(channel) for channel in audio_channels]

        # Reconstruct output
        reconstructed_channels = [np.zeros((1, len(audio) * self.sr // sr)) for _ in audio_channels]

        # Process chunks
        for ch_idx, chunks in enumerate(chunks_per_channel):
            for i, chunk in enumerate(chunks):
                temp_wav = os.path.join(temp_dir, f"chunk{ch_idx}_{i}.wav")
                sf.write(temp_wav, chunk, sr)

                out_chunk = super_resolution(
                    self.audiosr,
                    temp_wav,
                    seed=seed,
                    guidance_scale=guidance_scale,
                    ddim_steps=ddim_steps,
                    latent_t_per_second=12.8
                )[0]

                # Place chunk in reconstructed audio
                start = i * (output_chunk_samples - output_overlap_samples)
                end = start + len(out_chunk.T)
                reconstructed_channels[ch_idx][0, start:end] += out_chunk.flatten()

        # Merge channels
        reconstructed_audio = np.stack(reconstructed_channels, axis=-1) if is_stereo else reconstructed_channels[0]
        return reconstructed_audio

    def clean(self):
        del self.audiosr
        gc.collect()
        torch.cuda.empty_cache()
