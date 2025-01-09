import gc
import os
import warnings
from typing import List, Callable, Dict, Any

import librosa
import numpy as np
import soundfile as sf
import torch
from scipy import signal

from audiosr.pipeline import build_model, super_resolution
from tqdm import tqdm

from handlers.config import output_path
from wrappers.base_wrapper import BaseWrapper, TypedInput

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
    priority = 10
    allowed_kwargs = {
        "ddim_steps": TypedInput(
            description="Number of inference steps",
            default=50,
            ge=10,
            le=500,
            type=int,
            gradio_type="Slider"
        ),
        "guidance_scale": TypedInput(
            description="Scale for classifier free guidance",
            default=3.5,
            ge=1.0,
            le=20.0,
            type=float,
            gradio_type="Slider"
        ),
        "overlap": TypedInput(
            description="Overlap size",
            default=0.04,
            ge=0.0,
            le=0.5,
            type=float,
            gradio_type="Slider"
        ),
        "chunk_size": TypedInput(
            description="Chunk size",
            default=10.24,
            le=20.0,
            ge=5.0,
            type=float,
            gradio_type="Slider"
        ),
        "seed": TypedInput(
            description="Random seed. Leave blank to randomize the seed",
            default=None,
            type=int,
            ge=0,
            le=10000,
            gradio_type="Slider"
        ),
        "output_folder": TypedInput(
            description="Output folder",
            default=None,
            type=str,
            render=False
        ),
    }

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

    def process_audio(self, inputs: List[str], callback: Callable = None, **kwargs: Dict[str, Any]) -> List[str]:
        print(f"Upscaling audio from {inputs}")
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in self.allowed_kwargs.keys()}
        chunk_size = filtered_kwargs.get("chunk_size", 10.24)
        overlap = filtered_kwargs.get("overlap", 0.04)
        seed = filtered_kwargs.get("seed", None)
        if not seed:
            seed = np.random.randint(0, 10000)
        guidance_scale = filtered_kwargs.get("guidance_scale", 3.5)
        ddim_steps = filtered_kwargs.get("ddim_steps", 50)
        output_folder = os.path.join(output_path, "super_res")
        os.makedirs(output_folder, exist_ok=True)
        temp_folder = os.path.join(output_folder, "temp")
        os.makedirs(temp_folder, exist_ok=True)

        for temp_file in os.listdir(temp_folder):
            os.remove(os.path.join(temp_folder, temp_file))

        self.setup()
        outputs = []
        for tgt_file in inputs:
            print(f"Processing {tgt_file}")
            tgt_name, _ = os.path.splitext(os.path.basename(tgt_file))
            audio, sr = librosa.load(tgt_file, sr=self.sr * 2, mono=False)
            audio = audio.T
            is_stereo = len(audio.shape) == 2
            audio_channels = [audio] if not is_stereo else [audio[:, 0], audio[:, 1]]

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

            reconstructed_channels = [np.zeros((len(audio) * self.sr // sr), dtype=np.float32) for _ in audio_channels]

            total_chunks = sum(len(chunks) for chunks in chunks_per_channel)
            total_steps = total_chunks * ddim_steps
            progress_tqdm = tqdm(total=total_steps, desc="Processing", unit="steps", position=0, leave=True)

            def update_progress(progress):
                progress_tqdm.update(1)

            for ch_idx, chunks in enumerate(chunks_per_channel):
                for i, chunk in enumerate(chunks):
                    temp_wav = os.path.join(temp_folder, f"chunk{ch_idx}_{i}.wav")
                    sf.write(temp_wav, chunk, sr, format="WAV", subtype="PCM_16")

                    out_chunk = super_resolution(
                        self.audiosr,
                        temp_wav,
                        seed=seed,
                        guidance_scale=guidance_scale,
                        ddim_steps=ddim_steps,
                        latent_t_per_second=12.8,
                        callback=update_progress
                    )[0]

                    start = i * (output_chunk_samples - output_overlap_samples)
                    end = start + out_chunk.shape[1]
                    reconstructed_channels[ch_idx][start:end] += out_chunk.flatten()

            reconstructed_audio = (
                np.stack(reconstructed_channels, axis=-1) if is_stereo else reconstructed_channels[0]
            )
            reconstructed_audio = np.clip(reconstructed_audio, -1.0, 1.0).astype(np.float32)

            output_file = os.path.join(output_folder, f"super_res_{tgt_name}.wav")
            sf.write(output_file, reconstructed_audio, self.sr, format="WAV", subtype="PCM_16")

            for temp_file in os.listdir(temp_folder):
                os.remove(os.path.join(temp_folder, temp_file))
            outputs.append(output_file)

        self.clean()
        return outputs

    def clean(self):
        del self.audiosr
        gc.collect()
        torch.cuda.empty_cache()
