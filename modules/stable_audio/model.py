import os
import torch
import torchaudio
import logging
import numpy as np
import requests
import json
from typing import List, Dict, Any, Optional
from einops import rearrange
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict
from stable_audio_tools.inference.generation import generate_diffusion_cond
from handlers.config import model_path

logger = logging.getLogger("ADLB.StableAudio")

class StableAudioModel:
    """
    StabilityAI's Stable Audio model for text-to-audio generation.
    A standalone module (not a BaseWrapper) for the Sound Forge tab.
    Uses stable-audio-tools for generation but loads the model from local files.
    """
    
    def __init__(self):
        self.model = None
        self.model_config = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_folder = os.path.join(model_path, "stable_audio")
        self.output_dir = os.path.join(model_path, "stable_audio_outputs")
        os.makedirs(self.output_dir, exist_ok=True)
        
    def download_model_files(self):
        """Download the model files if they don't exist locally."""
        os.makedirs(self.model_folder, exist_ok=True)
        
        # Define the files we need to download
        files_to_download = {
            "model.safetensors": "https://huggingface.co/audo/stable-audio-open-1.0/resolve/main/model.safetensors?download=true",
            "model_config.json": "https://huggingface.co/audo/stable-audio-open-1.0/resolve/main/model_config.json?download=true"
        }
        
        # Download files if they don't exist
        for filename, url in files_to_download.items():
            file_path = os.path.join(self.model_folder, filename)
                
            if not os.path.exists(file_path):
                logger.info(f"Downloading {filename} from {url}...")
                try:
                    response = requests.get(url, stream=True)
                    response.raise_for_status()
                    
                    with open(file_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    logger.info(f"Downloaded {filename} successfully.")
                except Exception as e:
                    logger.error(f"Error downloading {filename}: {e}")
                    raise
        
        return self.model_folder
        
    def load_model(self):
        """
        Load the Stable Audio model directly from local files.
        Uses the same pattern as get_pretrained_model but with our local files.
        """
        if self.model is not None:
            return self.model
            
        try:
            # First, ensure we have all the model files locally
            self.download_model_files()
            
            # Get paths to our local files
            model_path = os.path.join(self.model_folder, "model.safetensors")
            config_path = os.path.join(self.model_folder, "model_config.json")
            
            # Make sure files exist
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found at {config_path}")
            
            # Load config from json file (same as in get_pretrained_model)
            logger.info(f"Loading model configuration from {config_path}")
            with open(config_path) as f:
                self.model_config = json.load(f)
            
            # Create model from config (same as in get_pretrained_model)
            logger.info("Creating model from config")
            self.model = create_model_from_config(self.model_config)
            
            # Load the model weights (same as in get_pretrained_model)
            logger.info(f"Loading model weights from {model_path}")
            self.model.load_state_dict(load_ckpt_state_dict(model_path))
            
            # Move model to appropriate device
            self.model = self.model.to(self.device)
            
            # Store important config values
            self.sample_rate = self.model_config["sample_rate"]
            self.sample_size = self.model_config["sample_size"]
            
            logger.info(f"Successfully loaded Stable Audio model to {self.device}")
            logger.info(f"Sample rate: {self.sample_rate}, Sample size: {self.sample_size}")
            
            return self.model
            
        except Exception as load_error:
            logger.error(f"Failed to load Stable Audio model: {load_error}", exc_info=True)
            self.model = None
            raise RuntimeError(f"Could not load Stable Audio model: {load_error}")
    
    def generate_audio(self, prompt: str, negative_prompt: str = None, 
                      duration: float = 5.0, num_inference_steps: int = 100,
                      num_waveforms: int = 1, seed: int = -1, 
                      guidance_scale: float = 7.0,
                      init_audio_path: str = None, init_noise_level: float = 1.0,
                      callback = None) -> List[Dict[str, Any]]:
        """
        Generate audio from text using Stable Audio.
        
        Args:
            prompt: Text description of the desired audio
            negative_prompt: Text description of what to avoid in the generation
            duration: Duration of the generated audio in seconds
            num_inference_steps: Number of denoising steps
            num_waveforms: Number of audio samples to generate
            seed: Random seed for generation (use -1 for random)
            guidance_scale: Guidance scale for classifier-free guidance
            init_audio_path: Path to an initial audio file to guide generation (optional)
            init_noise_level: Amount of noise to add to the initial audio (0.0-1.0)
            callback: Optional callback function for progress updates
            
        Returns:
            List of dictionaries containing generated audio data
        """
        # Load the model
        model = self.load_model()
        if model is None:
            error_msg = "Stable Audio model not available"
            logger.error(error_msg)
            if callback is not None:
                try:
                    callback(1.0, f"Error: {error_msg}")
                except Exception as e:
                    logger.warning(f"Callback error (non-fatal): {e}")
            raise RuntimeError(error_msg)
            
        try:
            num_inference_steps = int(num_inference_steps)
            if num_inference_steps < 10:
                num_inference_steps = 10
        except (ValueError, TypeError):
            num_inference_steps = 100
            
        try:
            guidance_scale = float(guidance_scale)
            if guidance_scale < 1.0:
                guidance_scale = 1.0
        except (ValueError, TypeError):
            guidance_scale = 7.0
            
        try:
            init_noise_level = float(init_noise_level)
            if init_noise_level < 0.0:
                init_noise_level = 0.0
            elif init_noise_level > 1.0:
                init_noise_level = 1.0
        except (ValueError, TypeError):
            init_noise_level = 1.0
        
        # Set up seed for reproducibility - fix bounds to avoid numpy int32 overflow
        MAX_SEED = 2**31 - 1  # Max value for int32 to avoid numpy overflow
        if seed < 0:
            # Use a smaller range for the seed to avoid integer overflow
            seed = torch.randint(0, MAX_SEED, (1,)).item()
        else:
            # Ensure provided seed is within valid range for numpy int32
            seed = int(seed) % MAX_SEED
            
        logger.info(f"Using seed: {seed}")
        
        # Set both torch and numpy random seeds for consistency
        torch.manual_seed(seed)
        np.random.seed(seed)
        if self.device == "cuda":
            torch.cuda.manual_seed(seed)
        
        try:
            # Start generation process
            if callback is not None:
                try:
                    callback(0.25, "Initializing generation...")
                except Exception as e:
                    logger.warning(f"Callback error (non-fatal): {e}")
            
            # Process initial audio if provided
            init_audio = None
            if init_audio_path and os.path.exists(init_audio_path):
                try:
                    if callback is not None:
                        try:
                            callback(0.30, "Loading initial audio file...")
                        except Exception as e:
                            logger.warning(f"Callback error (non-fatal): {e}")
                    
                    logger.info(f"Loading initial audio from {init_audio_path}")
                    # Load audio file
                    waveform, sample_rate = torchaudio.load(init_audio_path)
                    
                    # Resample if needed to match model sample rate
                    if sample_rate != self.sample_rate:
                        logger.info(f"Resampling audio from {sample_rate}Hz to {self.sample_rate}Hz")
                        resampler = torchaudio.transforms.Resample(
                            orig_freq=sample_rate, 
                            new_freq=self.sample_rate
                        )
                        waveform = resampler(waveform)
                    
                    # Convert to correct format (stereo if mono)
                    if waveform.shape[0] == 1:
                        # Duplicate mono channel to stereo
                        waveform = torch.cat([waveform, waveform], dim=0)
                    elif waveform.shape[0] > 2:
                        # Use only first two channels if more than stereo
                        waveform = waveform[:2]
                    
                    # Normalize audio to be between -1 and 1
                    waveform = waveform / max(torch.abs(waveform).max(), 1e-8)
                    
                    # Move to the appropriate device
                    waveform = waveform.to(self.device)
                    
                    # Set up init_audio tuple as (sample_rate, waveform)
                    init_audio = (self.sample_rate, waveform)
                    logger.info(f"Initial audio loaded successfully: shape={waveform.shape}")
                    
                except Exception as e:
                    logger.error(f"Error loading initial audio: {e}")
                    logger.warning("Proceeding without initial audio")
                    init_audio = None
            
            logger.info(f"Generating audio with prompt: '{prompt}' and duration: {duration}s")
            
            # Calculate the exact sample size based on the requested duration
            # This ensures the output audio will be exactly the requested length
            target_sample_count = int(duration * self.sample_rate)
            adjusted_sample_size = min(self.sample_size, target_sample_count)
            logger.info(f"Using sample size: {adjusted_sample_size} for {duration}s at {self.sample_rate}Hz")
            
            # Set up text and timing conditioning
            conditioning = [{
                "prompt": prompt,
                "seconds_start": 0,
                "seconds_total": duration
            }]
            
            # Include negative prompt if provided
            if negative_prompt and len(negative_prompt.strip()) > 0:
                conditioning[0]["negative_prompt"] = negative_prompt
                logger.info(f"Using negative prompt: '{negative_prompt}'")
            
            if callback is not None:
                try:
                    callback(0.50, "Running diffusion model...")
                except Exception as e:
                    logger.warning(f"Callback error (non-fatal): {e}")
            
            # Generate multiple audio samples if requested
            results = []
            for i in range(num_waveforms):
                logger.info(f"Generating audio sample {i+1}/{num_waveforms}...")
                
                # Generate audio - explicitly pass valid seed value
                output = generate_diffusion_cond(
                    model,
                    steps=num_inference_steps,
                    cfg_scale=guidance_scale,
                    conditioning=conditioning,
                    sample_size=adjusted_sample_size,  # Use the calculated sample size
                    sigma_min=0.3,
                    sigma_max=500,
                    sampler_type="dpmpp-3m-sde",
                    device=self.device,
                    seed=seed,  # Explicitly pass our valid seed
                    init_audio=init_audio,  # Pass initial audio if available
                    init_noise_level=init_noise_level  # Pass noise level for init_audio
                )
                
                # Modify seed for next variation if generating multiple
                if num_waveforms > 1:
                    # Increment seed for next generation but keep in valid range
                    seed = (seed + 1) % MAX_SEED
                
                # Rearrange audio batch to a single sequence
                output = rearrange(output, "b d n -> d (b n)")
                
                # Process audio (convert to float32 for further processing)
                audio_tensor = output.to(torch.float32)
                
                # Ensure audio is exactly the requested length
                expected_samples = int(duration * self.sample_rate)
                
                # Log the shape to understand the dimensions
                logger.info(f"Audio tensor shape before trimming: {audio_tensor.shape}")
                
                # Check if we're dealing with stereo or mono audio
                # For stereo: shape should be [2, samples]
                # For mono: shape should be [1, samples]
                if len(audio_tensor.shape) == 2:
                    channels = audio_tensor.shape[0]
                    current_samples = audio_tensor.shape[1]
                    
                    if current_samples > expected_samples:
                        # Trim the audio if it's longer than expected   
                        logger.info(f"Trimming audio from {current_samples} to {expected_samples} samples")
                        audio_tensor = audio_tensor[:, :expected_samples]
                    elif current_samples < expected_samples:
                        # Pad the audio if it's shorter with proper dimensions
                        logger.info(f"Padding audio from {current_samples} to {expected_samples} samples")
                        padding = torch.zeros((channels, expected_samples - current_samples), 
                                           device=audio_tensor.device, dtype=audio_tensor.dtype)
                        audio_tensor = torch.cat([audio_tensor, padding], dim=1)
                else:
                    # For handling potential 1D case (mono)
                    current_samples = audio_tensor.shape[0]
                    
                    if current_samples > expected_samples:
                        # Trim the audio if it's longer than expected   
                        logger.info(f"Trimming mono audio from {current_samples} to {expected_samples} samples")
                        audio_tensor = audio_tensor[:expected_samples]
                    elif current_samples < expected_samples:
                        # Pad the audio if it's shorter
                        logger.info(f"Padding mono audio from {current_samples} to {expected_samples} samples")
                        padding = torch.zeros((expected_samples - current_samples), 
                                           device=audio_tensor.device, dtype=audio_tensor.dtype)
                        audio_tensor = torch.cat([audio_tensor, padding], dim=0)
                
                # Create a clean filename based on the prompt
                clean_prompt = "".join(c if c.isalnum() else "_" for c in prompt[:30])
                output_file = os.path.join(self.output_dir, f"{clean_prompt}_{seed}_{i}.wav")
                
                # Save normalized audio
                norm_audio = audio_tensor.div(torch.max(torch.abs(audio_tensor))).clamp(-1, 1)
                int16_audio = norm_audio.mul(32767).to(torch.int16).cpu()
                
                logger.info(f"Final audio shape: {int16_audio.shape}")
                
                # Ensure the audio is in the correct format for torchaudio (channels, samples)
                if len(int16_audio.shape) == 1:
                    # Convert mono to [1, samples] format
                    int16_audio = int16_audio.unsqueeze(0)
                
                # Double-check the audio duration
                if len(int16_audio.shape) == 2:
                    actual_duration = int16_audio.shape[1] / self.sample_rate
                else:
                    actual_duration = int16_audio.shape[0] / self.sample_rate
                logger.info(f"Final audio duration: {actual_duration:.2f}s")
                
                torchaudio.save(output_file, int16_audio, self.sample_rate)
                
                # Also create a float32 numpy version for the return value
                audio_np = norm_audio.cpu().numpy()
                
                # Add result to the list
                results.append({
                    "file_path": output_file,
                    "audio": audio_np,
                    "sample_rate": self.sample_rate,
                    "seed": seed,
                    "index": i
                })
                
                # Log audio stats
                audio_std = np.std(audio_np)
                audio_mean = np.mean(audio_np)
                logger.info(f"Audio {i} stats: mean={audio_mean:.6f}, std={audio_std:.6f}")
                logger.info(f"Saved audio to {output_file}")
            
            if callback is not None:
                try:
                    callback(1.0, "Audio generation complete!")
                except Exception as e:
                    logger.warning(f"Callback error (non-fatal): {e}")
                
            return results
                
        except Exception as e:
            error_msg = f"Error generating audio with Stable Audio: {e}"
            logger.error(error_msg, exc_info=True)
            
            if callback is not None:
                try:
                    callback(1.0, f"Error: {str(e)}")
                except Exception as callback_e:
                    logger.warning(f"Callback error (non-fatal): {callback_e}")
            raise 