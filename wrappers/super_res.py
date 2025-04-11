import gc
import logging
import os
import warnings
from typing import List, Dict, Any, Optional

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import torch
from audiosr.pipeline import build_model, super_resolution
from scipy import signal

from util.data_classes import ProjectFiles
from wrappers.base_wrapper import BaseWrapper, TypedInput

logger = logging.getLogger(__name__)
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
    title = "Super Resolution"
    description = "Upscale audio files to a higher sample rate using a pre-trained Super Resolution model."
    priority = 8
    allowed_kwargs = {
        "ddim_steps": TypedInput(
            description="The number of diffusion steps used during inference. A higher number provides better quality results but increases processing time.",
            default=50,
            ge=10,
            le=500,
            type=int,
            gradio_type="Slider"
        ),
        "guidance_scale": TypedInput(
            description="The strength of classifier-free guidance applied during processing. Higher values produce sharper results but may reduce diversity in the output.",
            default=3.5,
            ge=1.0,
            le=20.0,
            type=float,
            gradio_type="Slider"
        ),
        "overlap": TypedInput(
            description="The proportion of overlap between audio chunks during processing. Higher overlap helps smooth transitions but increases computation.",
            default=0.04,
            ge=0.0,
            le=0.5,
            type=float,
            gradio_type="Slider"
        ),
        "chunk_size": TypedInput(
            description="The length of each audio chunk (in seconds) used for processing. Smaller chunks reduce memory usage but may increase transition artifacts.",
            default=10.24,
            le=20.0,
            ge=5.0,
            type=float,
            gradio_type="Slider"
        ),
        "seed": TypedInput(
            description="The random seed for reproducibility. Set to -1 for a randomized seed, which can produce varied outputs on each run.",
            default=-1,
            type=int,
            ge=-1,
            le=10000,
            gradio_type="Slider"
        ),
        "output_folder": TypedInput(
            description="The directory where the processed audio files will be saved. If not provided, files are saved in a default location.",
            default=None,
            type=str,
            render=False
        ),
        "tgt_ensemble": TypedInput(
            description="When enabled, combines the output with a low-pass filtered version of the original audio for enhanced quality and naturalness.",
            default=False,
            type=bool,
            gradio_type="Checkbox"
        ),
        "tgt_cutoff": TypedInput(
            description="Specifies the cutoff frequency (in Hz) for the target audio ensemble's low-pass filter. Adjust this to fine-tune the balance of high and low frequencies.",
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
        """
        Register FastAPI endpoint for super resolution processing.
        
        Args:
            api: FastAPI application instance
            
        Returns:
            The registered endpoint route
        """
        from fastapi import File, UploadFile, HTTPException, Body
        from fastapi.responses import FileResponse
        from pydantic import BaseModel, create_model
        from typing import List, Optional
        import tempfile
        from pathlib import Path
        
        # Create Pydantic model for settings
        fields = {}
        for key, value in self.allowed_kwargs.items():
            field_type = value.type
            fields[key] = (field_type, value.field)
        
        SettingsModel = create_model(f"{self.__class__.__name__}Settings", **fields)
        
        # Create models for JSON API
        FileData, JsonRequest = self.create_json_models()
        
        @api.post("/api/v1/process/super_resolution", tags=["Audio Processing"])
        async def process_super_resolution(
            files: List[UploadFile] = File(...),
            settings: Optional[SettingsModel] = None
        ):
            """
            Process audio files with super resolution.
            
            Args:
                files: List of audio files to process
                settings: Super resolution settings
                
            Returns:
                List of processed audio files
            """
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Save uploaded files
                    input_files = []
                    for file in files:
                        file_path = Path(temp_dir) / file.filename
                        with file_path.open("wb") as f:
                            content = await file.read()
                            f.write(content)
                        input_files.append(ProjectFiles(str(file_path)))
                    
                    # Process files
                    settings_dict = settings.dict() if settings else {}
                    processed_files = self.process_audio(input_files, **settings_dict)
                    
                    # Return processed files
                    output_files = []
                    for project in processed_files:
                        for output in project.last_outputs:
                            output_path = Path(output)
                            if output_path.exists():
                                output_files.append(FileResponse(output))
                    
                    return output_files
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @api.post("/api/v2/process/super_resolution", tags=["Audio Processing"])
        async def process_super_resolution_json(
            request: JsonRequest = Body(...)
        ):
            """
            Process audio files with super resolution.
            
            This endpoint enhances the quality of audio files by applying AI-powered super resolution 
            techniques. The process can dramatically improve detail, clarity, and perceived audio quality 
            by reconstructing high-frequency information, enhancing transients, and improving overall 
            resolution. Audio files are upscaled to 48kHz sample rate with enhanced detail.
            
            ## Request Body
            
            ```json
            {
              "files": [
                {
                  "filename": "low_quality.mp3",
                  "content": "base64_encoded_file_content..."
                }
              ],
              "settings": {
                "ddim_steps": 50,
                "guidance_scale": 3.5,
                "tgt_ensemble": true,
                "tgt_cutoff": 12000,
                "seed": 42
              }
            }
            ```
            
            ## Parameters
            
            - **files**: Array of file objects, each containing:
              - **filename**: Name of the file (with extension)
              - **content**: Base64-encoded file content
            - **settings**: Super resolution settings with the following options:
              - **ddim_steps**: Number of diffusion steps for inference (default: 50)
                - Range: 10-500 (higher values increase quality but take longer)
              - **guidance_scale**: Strength of classifier-free guidance (default: 3.5)
                - Range: 1.0-20.0 (higher values make the output more distinct but less natural)
              - **overlap**: Proportion of overlap between audio chunks (default: 0.04)
                - Range: 0.0-0.5 (higher values improve transitions but increase computation)
              - **chunk_size**: Length of each audio chunk in seconds (default: 10.24)
                - Range: 5.0-20.0 (smaller chunks use less memory but may introduce artifacts)
              - **seed**: Random seed for reproducibility (default: -1)
                - Set to -1 for random seed on each run
              - **tgt_ensemble**: Combine output with filtered original audio (default: false)
                - When true, creates a more balanced, natural sound
              - **tgt_cutoff**: Cutoff frequency for ensemble filtering (default: 12000)
                - Range: 500-24000 Hz (adjusts blend between processed and original audio)
            
            ## Response
            
            ```json
            {
              "files": [
                {
                  "filename": "enhanced_audio.wav",
                  "content": "base64_encoded_file_content..."
                }
              ]
            }
            ```
            
            The API returns an object containing the enhanced high-resolution audio file as a Base64-encoded string.
            """
            # Use the handle_json_request helper from BaseWrapper
            return self.handle_json_request(request, self.process_audio)
            
        return process_super_resolution_json

    def setup(self, model_name="basic", device="auto"):
        self.model_name = model_name
        self.device = device
        self.sr = 48000
        print("Loading Model...")
        self.audiosr = build_model(model_name=self.model_name, device=self.device)
        print("Model loaded!")

    def process_audio(self, inputs: List[ProjectFiles], callback=None, **kwargs: Dict[str, Any]) -> List[ProjectFiles]:
        print(f"Upscaling audio from {inputs}")
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in self.allowed_kwargs.keys()}
        chunk_size = filtered_kwargs.get("chunk_size", 10.24)
        overlap = filtered_kwargs.get("overlap", 0.04)
        seed = filtered_kwargs.get("seed", -1)
        if seed == -1:
            seed = np.random.randint(0, 10000)
        guidance_scale = filtered_kwargs.get("guidance_scale", 3.5)
        ddim_steps = filtered_kwargs.get("ddim_steps", 50)
        tgt_ensemble = filtered_kwargs.get("tgt_ensemble", False)
        tgt_cutoff = filtered_kwargs.get("tgt_cutoff", 12000)

        crossover_freq = tgt_cutoff - 1000

        self.setup()
        pj_outputs = []
        try:
            for project in inputs:
                output_folder = os.path.join(project.project_dir, "super_res")
                os.makedirs(output_folder, exist_ok=True)
                temp_dir = os.path.join(output_folder, "temp")
                os.makedirs(temp_dir, exist_ok=True)

                for temp_file in os.listdir(temp_dir):
                    os.remove(os.path.join(temp_dir, temp_file))
                pj_inputs, _ = self.filter_inputs(project, "audio")
                outputs = []
                for tgt_file in pj_inputs:
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
                    current_step = 0

                    # Single global progress bar
                    def update_progress(pct, desc, total):
                        if callback is not None:
                            callback(pct, desc, total)
                        logger.info(f"{desc}: {pct:.2f}%")

                    update_progress(0, "Processing", total_chunks)
                    for ch_idx, (chunks, original_lengths) in enumerate(chunks_per_channel):
                        for i, chunk in enumerate(chunks):
                            update_progress(current_step / total_chunks, "Processing", total_chunks)
                            current_step += 1
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
                                    latent_t_per_second=12.8
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

                    reconstructed_audio = np.stack(reconstructed_channels, axis=-1) if is_stereo else \
                        reconstructed_channels[0]

                    if tgt_ensemble:
                        low, _ = librosa.load(tgt_file, sr=48000, mono=False)
                        output = match_array_shapes(reconstructed_audio[0].T, low)
                        low = lr_filter(low.T, crossover_freq, 'lowpass', order=10)
                        high = lr_filter(output.T, crossover_freq, 'highpass', order=10)
                        high = lr_filter(high, 23000, 'lowpass', order=2)
                        output = low + high
                    else:
                        output = reconstructed_audio[0]
                    update_progress(100, "Processing", total_chunks)
                    output_file = os.path.join(output_folder, f"{tgt_name}(Super_Res).wav")
                    sf.write(output_file, output, self.sr, format="WAV", subtype="PCM_16")

                    for temp_file in os.listdir(temp_dir):
                        os.remove(os.path.join(temp_dir, temp_file))
                    outputs.append(output_file)
                project.add_output("super_res", outputs)
                pj_outputs.append(project)
        except Exception as e:
            logger.error(f"Error upscaling audio: {e}")
            if callback is not None:
                callback(1, "Error upscaling audio")
            raise e
        self.clean()
        return pj_outputs

    def clean(self):
        del self.audiosr
        gc.collect()
        torch.cuda.empty_cache()
