import gc
import os
import warnings
from typing import List, Callable, Dict, Any

import librosa
import numpy as np
import soundfile as sf
import torch
from scipy import signal
import pyloudnorm as pyln
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


class SuperResolution(BaseWrapper):
    priority = 2
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
            default=-1,
            type=int,
            ge=-1,
            le=10000,
            gradio_type="Slider"
        ),
        "output_folder": TypedInput(
            description="Output folder",
            default=None,
            type=str,
            render=False
        ),
        "tgt_ensemble": TypedInput(
            description="Enable target audio ensemble",
            default=False,
            type=bool
        ),
        "tgt_cutoff": TypedInput(
            description="Cutoff frequency for target audio ensemble",
            default=12000,
            ge=500,
            le=24000,
            type=int,
            gradio_type="Slider"
        )
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
        seed = filtered_kwargs.get("seed", -1)
        if seed == -1:
            seed = np.random.randint(0, 10000)
        guidance_scale = filtered_kwargs.get("guidance_scale", 3.5)
        ddim_steps = filtered_kwargs.get("ddim_steps", 50)
        output_folder = os.path.join(output_path, "super_res")
        os.makedirs(output_folder, exist_ok=True)
        temp_dir = os.path.join(output_folder, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        tgt_ensemble = filtered_kwargs.get("tgt_ensemble", False)
        tgt_cutoff = filtered_kwargs.get("tgt_cutoff", 12000)

        crossover_freq = tgt_cutoff - 1000

        for temp_file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, temp_file))

        self.setup()
        inputs, outputs = self.filter_inputs(inputs, "audio")
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
                original_lengths = []
                start = 0
                while start < len(audio):
                    end = min(start + chunk_samples, len(audio))
                    chunk = audio[start:end]
                    if len(chunk) < chunk_samples:
                        original_lengths.append(len(chunk))
                        chunk = np.concatenate([chunk, np.zeros(chunk_samples - len(chunk))])
                    else:
                        original_lengths.append(chunk_samples)
                    chunks.append(chunk)
                    start += chunk_samples - overlap_samples if enable_overlap else chunk_samples
                return chunks, original_lengths

            # Process both channels (mono or stereo)
            chunks_per_channel = [process_chunks(channel) for channel in audio_channels]
            sample_rate_ratio = self.sr / sr
            total_length = len(chunks_per_channel[0][0]) * output_chunk_samples - (
                        len(chunks_per_channel[0][0]) - 1) * (
                               output_overlap_samples if enable_overlap else 0)
            reconstructed_channels = [np.zeros((1, total_length)) for _ in audio_channels]

            meter_before = pyln.Meter(sr)
            meter_after = pyln.Meter(self.sr)

            total_chunks = sum([len(chunks) for chunks, _ in chunks_per_channel])
            total_steps = total_chunks * ddim_steps
            # Single global progress bar
            progress_tqdm = tqdm(total=total_steps, desc="Processing", unit="steps", position=0, leave=True)

            def update_progress(progress):
                    progress_tqdm.update(1)
                    if callable(callback):
                        callback(progress / total_steps, f"Processing {progress}/{total_steps} steps", total_steps)

            for ch_idx, (chunks, original_lengths) in enumerate(chunks_per_channel):
                for i, chunk in enumerate(chunks):
                    try:
                        temp_wav = os.path.join(temp_dir, f"chunk{ch_idx}_{i}.wav")
                        loudness_before = meter_before.integrated_loudness(chunk)
                        if not isinstance(chunk, np.ndarray):
                            raise ValueError("Audio chunk must be a NumPy array.")
                        if not isinstance(sr, int) or sr <= 0:
                            raise ValueError("Sample rate must be a positive integer.")
                        sf.write(temp_wav, chunk, sr)

                        out_chunk = super_resolution(
                            self.audiosr,
                            temp_wav,
                            seed=seed,
                            guidance_scale=guidance_scale,
                            ddim_steps=ddim_steps,
                            latent_t_per_second=12.8,
                            callback=update_progress
                        )

                        out_chunk = out_chunk[0]
                        num_samples_to_keep = int(original_lengths[i] * sample_rate_ratio)
                        out_chunk = out_chunk[:, :num_samples_to_keep].squeeze()
                        loudness_after = meter_after.integrated_loudness(out_chunk)
                        out_chunk = pyln.normalize.loudness(out_chunk, loudness_after, loudness_before)

                        if enable_overlap:
                            actual_overlap_samples = min(output_overlap_samples, num_samples_to_keep)
                            fade_out = np.linspace(1., 0., actual_overlap_samples)
                            fade_in = np.linspace(0., 1., actual_overlap_samples)

                            if i == 0:
                                out_chunk[-actual_overlap_samples:] *= fade_out
                            elif i < len(chunks) - 1:
                                out_chunk[:actual_overlap_samples] *= fade_in
                                out_chunk[-actual_overlap_samples:] *= fade_out
                            else:
                                out_chunk[:actual_overlap_samples] *= fade_in

                            start = i * (
                                output_chunk_samples - output_overlap_samples if enable_overlap else output_chunk_samples)
                            end = start + out_chunk.shape[0]
                            reconstructed_channels[ch_idx][0, start:end] += out_chunk.flatten()
                    except Exception as e:
                        print(f"Error processing chunk {i + 1} of {len(chunks)}: {e}")
                        continue

            reconstructed_audio = np.stack(reconstructed_channels, axis=-1) if is_stereo else reconstructed_channels[0]

            if tgt_ensemble:
                low, _ = librosa.load(tgt_file, sr=48000, mono=False)
                output = match_array_shapes(reconstructed_audio[0].T, low)
                low = lr_filter(low.T, crossover_freq, 'lowpass', order=10)
                high = lr_filter(output.T, crossover_freq, 'highpass', order=10)
                high = lr_filter(high, 23000, 'lowpass', order=2)
                output = low + high
            else:
                output = reconstructed_audio[0]

            output_file = os.path.join(output_folder, f"super_res_{tgt_name}.wav")
            sf.write(output_file, output, self.sr, format="WAV", subtype="PCM_16")

            for temp_file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, temp_file))
            outputs.append(output_file)

        self.clean()
        return outputs

    def clean(self):
        del self.audiosr
        gc.collect()
        torch.cuda.empty_cache()
