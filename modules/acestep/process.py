import os
import logging
import torch
import uuid
from typing import Optional, List, Tuple, Any
from modules.acestep.acestep.pipeline_ace_step import ACEStepPipeline
from handlers.config import model_path, output_path
    

# Configure logging
logger = logging.getLogger("ADLB.ACEStep")

ACESTEP_AVAILABLE = True
# Import AudioLab configuration

# Default ACE-Step model
DEFAULT_MODEL = "ACE-Step/ACE-Step-v1-3.5B"

def check_acestep_installed() -> bool:
    """
    Check if ACE-Step is properly installed and available.
    
    Returns:
        bool: True if ACE-Step is available, False otherwise
    """
    return ACESTEP_AVAILABLE

def download_acestep_model(model_name: str = DEFAULT_MODEL) -> str:
    """
    Download ACE-Step model if not already downloaded.
    
    Args:
        model_name: Name of the model to download
        
    Returns:
        str: Path to the downloaded model
    """
    # Create ACE-Step models directory if it doesn't exist
    acestep_models_dir = os.path.join(model_path, "acestep")
    os.makedirs(acestep_models_dir, exist_ok=True)
    
    # Convert model name to directory name
    model_dir_name = model_name.replace("/", "_")
    model_dir = os.path.join(acestep_models_dir, model_dir_name)
    
    # Check if model is already downloaded
    if os.path.exists(model_dir) and len(os.listdir(model_dir)) > 0:
        logger.info(f"ACE-Step model '{model_name}' already downloaded to {model_dir}")
        return model_dir
    
    # Download model using huggingface_hub
    try:
        from huggingface_hub import snapshot_download
        logger.info(f"Downloading ACE-Step model '{model_name}'...")
        
        # Download the model
        snapshot_download(
            repo_id=model_name,
            local_dir=model_dir,
            local_dir_use_symlinks=False
        )
        
        logger.info(f"Successfully downloaded ACE-Step model to {model_dir}")
        return model_dir
    
    except Exception as e:
        logger.error(f"Error downloading ACE-Step model: {e}")
        raise RuntimeError(f"Failed to download ACE-Step model: {e}")

def initialize_acestep_pipeline(
    checkpoint_path: str,
    bf16: bool = True,
    torch_compile: bool = False,
    cpu_offload: bool = False,
    overlapped_decode: bool = False,
    device_id: int = 0
) -> Any:
    """
    Initialize ACE-Step pipeline for audio generation.
    
    Args:
        checkpoint_path: Path to the checkpoint directory
        bf16: Whether to use bfloat16 precision
        torch_compile: Whether to use torch compile for faster inference
        cpu_offload: Whether to offload models to CPU to save VRAM
        overlapped_decode: Whether to use overlapped decoding
        device_id: GPU device ID to use
        
    Returns:
        ACEStepPipeline: Initialized pipeline
    """
    if not ACESTEP_AVAILABLE:
        raise RuntimeError("ACE-Step is not available. Please install required dependencies.")
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    
    try:
        pipeline = ACEStepPipeline(
            checkpoint_dir=checkpoint_path,
            dtype="bfloat16" if bf16 else "float32",
            torch_compile=torch_compile,
            cpu_offload=cpu_offload,
            overlapped_decode=overlapped_decode
        )
        logger.info("ACE-Step pipeline initialized successfully")
        return pipeline
    except Exception as e:
        logger.error(f"Failed to initialize ACE-Step pipeline: {e}")
        raise RuntimeError(f"Failed to initialize ACE-Step pipeline: {e}")

def process(
    # Core parameters
    prompt: str,
    lyrics: str = "",
    audio_duration: float = 60.0,
    
    # Model parameters
    model_name: str = DEFAULT_MODEL,
    bf16: bool = True,
    torch_compile: bool = False,
    cpu_offload: bool = False,
    overlapped_decode: bool = False,
    device_id: int = 0,
    
    # Generation parameters
    infer_step: int = 27,
    guidance_scale: float = 7.5,
    scheduler_type: str = "flow_match_euler",
    cfg_type: str = "apg",
    omega_scale: float = 10.0,
    guidance_interval: float = 1.0,
    guidance_interval_decay: float = 1.0,
    min_guidance_scale: float = 1.0,
    
    # Advanced parameters
    seed: Optional[int] = None,
    actual_seeds: Optional[List[int]] = None,
    guidance_scale_text: float = 0.0,
    guidance_scale_lyric: float = 0.0,
    
    # ERG parameters
    use_erg_tag: bool = False,
    use_erg_lyric: bool = False,
    use_erg_diffusion: bool = False,
    oss_steps: Optional[List[int]] = None,
    
    # Audio2Audio parameters
    audio2audio_enable: bool = False,
    ref_audio_strength: float = 0.5,
    ref_audio_input: Optional[str] = None,
    
    # Output parameters
    output_filename: Optional[str] = None,
    progress_callback: Optional[callable] = None
) -> Tuple[str, str]:
    """
    Generate music using ACE-Step model.
    
    Args:
        prompt: Text prompt describing the music to generate
        lyrics: Lyrics for the vocals (optional)
        audio_duration: Duration of generated audio in seconds
        model_name: Name of the ACE-Step model to use
        bf16: Whether to use bfloat16 precision
        torch_compile: Whether to use torch compile for faster inference
        cpu_offload: Whether to offload models to CPU to save VRAM
        overlapped_decode: Whether to use overlapped decoding
        device_id: GPU device ID to use
        infer_step: Number of inference steps
        guidance_scale: Guidance scale for generation
        scheduler_type: Scheduler type (flow_match_euler, euler, dpm)
        cfg_type: CFG type (apg, cfg)
        omega_scale: Omega scale parameter
        guidance_interval: Guidance interval
        guidance_interval_decay: Guidance interval decay
        min_guidance_scale: Minimum guidance scale
        seed: Random seed for reproducibility (overrides actual_seeds if provided)
        actual_seeds: List of random seeds for reproducibility
        guidance_scale_text: Guidance scale for text
        guidance_scale_lyric: Guidance scale for lyrics
        use_erg_tag: Whether to use ERG tag
        use_erg_lyric: Whether to use ERG lyric
        use_erg_diffusion: Whether to use ERG diffusion
        oss_steps: List of OSS steps
        audio2audio_enable: Whether to enable audio2audio mode
        ref_audio_strength: Reference audio strength (0.0-1.0) for audio2audio
        ref_audio_input: Path to reference audio file for audio2audio
        output_filename: Custom output filename (optional)
        progress_callback: Callback function for progress updates
        
    Returns:
        Tuple[str, str]: Path to the generated audio file, message
    """
    # Ensure ACE-Step is available
    if not check_acestep_installed():
        return "", "ACE-Step is not installed. Please install required dependencies."
    
    # Progress update
    if progress_callback is not None:
        progress_callback(0.1, "Checking model availability...")
    
    # Download/retrieve model
    try:
        checkpoint_path = download_acestep_model(model_name)
    except Exception as e:
        logger.error(f"Failed to download/retrieve model: {e}")
        return "", f"Failed to download/retrieve model: {e}"
    
    # Progress update
    if progress_callback is not None:
        progress_callback(0.2, "Initializing pipeline...")
    
    # Initialize pipeline
    try:
        pipeline = initialize_acestep_pipeline(
            checkpoint_path=checkpoint_path,
            bf16=bf16,
            torch_compile=torch_compile,
            cpu_offload=cpu_offload,
            overlapped_decode=overlapped_decode,
            device_id=device_id
        )
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        return "", f"Failed to initialize pipeline: {e}"
    
    # Progress update
    if progress_callback is not None:
        progress_callback(0.3, "Preparing generation parameters...")
    
    # Set up seeds
    if seed is not None:
        # If a single seed is provided, use it
        actual_seeds = [seed]
    elif actual_seeds is None:
        # If no seeds are provided, generate a random one
        actual_seeds = [torch.randint(0, 2**32 - 1, (1,)).item()]
    
    # Set default OSS steps if not provided
    if oss_steps is None:
        oss_steps = []
    
    # Generate output path
    os.makedirs(os.path.join(output_path, "acestep"), exist_ok=True)
    timestamp = uuid.uuid4().hex[:8]
    if output_filename:
        # Use provided filename but ensure it has .wav extension
        if not output_filename.endswith(".wav"):
            output_filename = f"{output_filename}.wav"
        output_path_file = os.path.join(output_path, "acestep", output_filename)
    else:
        # Generate filename based on timestamp and seed
        output_path_file = os.path.join(output_path, "acestep", f"acestep_{timestamp}_{actual_seeds[0]}.wav")
    
    # Progress update
    if progress_callback is not None:
        progress_callback(0.4, "Starting audio generation...")
    
    # Prepare parameters
    try:
        # Check for audio2audio mode
        mode_description = "audio2audio mode" if audio2audio_enable and ref_audio_input else "text2music mode"
        if progress_callback is not None:
            progress_callback(0.45, f"Processing in {mode_description}...")
            
        # Ensure scheduler_type is one of the valid values to prevent undefined scheduler
        if scheduler_type not in ["euler", "heun", "flow_match_euler"]:
            logger.warning(f"Invalid scheduler_type '{scheduler_type}'. Defaulting to 'euler'.")
            scheduler_type = "euler"
            
        # Run the ACE-Step pipeline
        pipeline(
            audio_duration=audio_duration,
            prompt=prompt,
            lyrics=lyrics,
            infer_step=infer_step,
            guidance_scale=guidance_scale,
            scheduler_type=scheduler_type,  # Explicitly use the validated scheduler_type
            cfg_type=cfg_type,
            omega_scale=omega_scale,
            manual_seeds=", ".join(map(str, actual_seeds)),
            guidance_interval=guidance_interval,
            guidance_interval_decay=guidance_interval_decay,
            min_guidance_scale=min_guidance_scale,
            use_erg_tag=use_erg_tag,
            use_erg_lyric=use_erg_lyric,
            use_erg_diffusion=use_erg_diffusion,
            oss_steps=", ".join(map(str, oss_steps)),
            guidance_scale_text=guidance_scale_text,
            guidance_scale_lyric=guidance_scale_lyric,
            save_path=output_path_file,
            audio2audio_enable=audio2audio_enable,
            ref_audio_strength=ref_audio_strength,
            ref_audio_input=ref_audio_input
        )
        
        # Progress update
        if progress_callback is not None:
            progress_callback(1.0, "Audio generation complete!")
        
        return output_path_file, f"Successfully generated audio with ACE-Step using seed {actual_seeds[0]}"
    
    except Exception as e:
        logger.error(f"Error generating audio: {e}")
        import traceback
        traceback.print_exc()
        return "", f"Error generating audio: {e}"

def process_lora(
    # Core parameters
    prompt: str,
    lyrics: str = "",
    audio_duration: float = 60.0,
    
    # Model parameters
    base_model_name: str = DEFAULT_MODEL,
    lora_model_path: str = "",
    bf16: bool = True,
    torch_compile: bool = False,
    cpu_offload: bool = False,
    device_id: int = 0,
    
    # Generation parameters
    infer_step: int = 27,
    guidance_scale: float = 7.5,
    scheduler_type: str = "flow_match_euler",
    cfg_type: str = "apg",
    omega_scale: float = 10.0,
    seed: Optional[int] = None,
    
    # Audio2Audio parameters
    audio2audio_enable: bool = False,
    ref_audio_strength: float = 0.5,
    ref_audio_input: Optional[str] = None,
    
    # Output parameters
    output_filename: Optional[str] = None,
    progress_callback: Optional[callable] = None
) -> Tuple[str, str]:
    """
    Generate music using ACE-Step with a LoRA model (like RapMachine).
    
    Args:
        prompt: Text prompt describing the music to generate
        lyrics: Lyrics for the vocals (optional)
        audio_duration: Duration of generated audio in seconds
        base_model_name: Name of the base ACE-Step model 
        lora_model_path: Path to the LoRA model weights
        bf16: Whether to use bfloat16 precision
        torch_compile: Whether to use torch compile for faster inference
        cpu_offload: Whether to offload models to CPU to save VRAM
        device_id: GPU device ID to use
        infer_step: Number of inference steps
        guidance_scale: Guidance scale for generation
        scheduler_type: Scheduler type
        cfg_type: CFG type
        omega_scale: Omega scale parameter
        seed: Random seed for reproducibility
        audio2audio_enable: Whether to enable audio2audio mode
        ref_audio_strength: Reference audio strength (0.0-1.0) for audio2audio
        ref_audio_input: Path to reference audio file for audio2audio
        output_filename: Custom output filename (optional)
        progress_callback: Callback function for progress updates
        
    Returns:
        Tuple[str, str]: Path to the generated audio file, message
    """
    # Ensure ACE-Step is available
    if not check_acestep_installed():
        return "", "ACE-Step is not installed. Please install required dependencies."
    
    # Progress update
    if progress_callback is not None:
        progress_callback(0.1, "Checking model availability...")
    
    # Download/retrieve base model
    try:
        base_checkpoint_path = download_acestep_model(base_model_name)
    except Exception as e:
        logger.error(f"Failed to download/retrieve base model: {e}")
        return "", f"Failed to download/retrieve base model: {e}"
    
    # Download/retrieve LoRA model
    try:
        # Create LoRA models directory if it doesn't exist
        acestep_lora_dir = os.path.join(model_path, "acestep", "lora")
        os.makedirs(acestep_lora_dir, exist_ok=True)
        
        # Handle either a HuggingFace model ID or a local path
        if "/" in lora_model_path and not os.path.exists(lora_model_path):
            # Looks like a HuggingFace model ID (e.g., "ACE-Step/ACE-Step-v1-chinese-rap-LoRA")
            from huggingface_hub import snapshot_download
            
            # Convert model name to directory name
            lora_dir_name = lora_model_path.replace("/", "_")
            lora_dir = os.path.join(acestep_lora_dir, lora_dir_name)
            
            # Check if model is already downloaded
            if not (os.path.exists(lora_dir) and len(os.listdir(lora_dir)) > 0):
                logger.info(f"Downloading LoRA model '{lora_model_path}'...")
                
                # Download the LoRA model
                snapshot_download(
                    repo_id=lora_model_path,
                    local_dir=lora_dir,
                    local_dir_use_symlinks=False
                )
            
            logger.info(f"Using LoRA model from {lora_dir}")
            lora_checkpoint_path = lora_dir
        else:
            # Use the provided local path
            if not os.path.exists(lora_model_path):
                return "", f"LoRA model path does not exist: {lora_model_path}"
            
            lora_checkpoint_path = lora_model_path
            logger.info(f"Using LoRA model from {lora_checkpoint_path}")
    
    except Exception as e:
        logger.error(f"Failed to download/retrieve LoRA model: {e}")
        return "", f"Failed to download/retrieve LoRA model: {e}"
    
    # Progress update
    if progress_callback is not None:
        progress_callback(0.2, "Initializing pipeline with LoRA...")
    
    # Initialize pipeline with LoRA
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        
        try:
            # Initialize the ACEStepPipeline with LoRA
            from modules.acestep.acestep.pipeline_ace_step import ACEStepPipeline
            
            pipeline = ACEStepPipeline(
                checkpoint_dir=base_checkpoint_path,
                dtype="bfloat16" if bf16 else "float32",
                torch_compile=torch_compile,
                cpu_offload=cpu_offload,
                overlapped_decode=False,  # Set overlapped_decode to False when using LoRA
                lora_checkpoint_dir=lora_checkpoint_path  # Add LoRA checkpoint path
            )
            
            logger.info("ACE-Step pipeline with LoRA initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ACE-Step pipeline with LoRA: {e}")
            return "", f"Failed to initialize ACE-Step pipeline with LoRA: {e}"
    except Exception as e:
        logger.error(f"Error setting up LoRA pipeline: {e}")
        return "", f"Error setting up LoRA pipeline: {e}"
    
    # Progress update
    if progress_callback is not None:
        progress_callback(0.3, "Preparing generation parameters...")
    
    # Set up seeds
    if seed is not None:
        # If a single seed is provided, use it
        actual_seeds = [seed]
    else:
        # If no seed is provided, generate a random one
        actual_seeds = [torch.randint(0, 2**32 - 1, (1,)).item()]
    
    # Generate output path
    os.makedirs(os.path.join(output_path, "acestep"), exist_ok=True)
    timestamp = uuid.uuid4().hex[:8]
    if output_filename:
        # Use provided filename but ensure it has .wav extension
        if not output_filename.endswith(".wav"):
            output_filename = f"{output_filename}.wav"
        output_path_file = os.path.join(output_path, "acestep", output_filename)
    else:
        # Generate filename based on timestamp, seed, and LoRA model
        lora_name = os.path.basename(lora_checkpoint_path)
        output_path_file = os.path.join(output_path, "acestep", f"acestep_lora_{lora_name}_{timestamp}_{actual_seeds[0]}.wav")
    
    # Progress update
    if progress_callback is not None:
        progress_callback(0.4, "Starting audio generation with LoRA...")
    
    # Check for audio2audio mode
    mode_description = "audio2audio mode with LoRA" if audio2audio_enable and ref_audio_input else "text2music mode with LoRA"
    if progress_callback is not None:
        progress_callback(0.45, f"Processing in {mode_description}...")
        
    # Ensure scheduler_type is one of the valid values to prevent undefined scheduler
    if scheduler_type not in ["euler", "heun", "flow_match_euler"]:
        logger.warning(f"Invalid scheduler_type '{scheduler_type}'. Defaulting to 'euler'.")
        scheduler_type = "euler"
        
    # Generate audio with LoRA
    try:
        # Run the ACE-Step pipeline with LoRA
        pipeline(
            audio_duration=audio_duration,
        prompt=prompt,
        lyrics=lyrics,
            infer_step=infer_step,
            guidance_scale=guidance_scale,
            scheduler_type=scheduler_type,  # Explicitly use the validated scheduler_type
            cfg_type=cfg_type,
            omega_scale=omega_scale,
            manual_seeds=", ".join(map(str, actual_seeds)),
            save_path=output_path_file,
            audio2audio_enable=audio2audio_enable,
            ref_audio_strength=ref_audio_strength,
            ref_audio_input=ref_audio_input,
            lora_name_or_path=lora_checkpoint_path
        )
        
        # Progress update
        if progress_callback is not None:
            progress_callback(1.0, "Audio generation with LoRA complete!")
        
        return output_path_file, f"Successfully generated audio with ACE-Step LoRA using seed {actual_seeds[0]}"
    
    except Exception as e:
        logger.error(f"Error generating audio with LoRA: {e}")
        import traceback
        traceback.print_exc()
        return "", f"Error generating audio with LoRA: {e}"

def process_retake(
    # Source audio
    source_audio_path: str,
    
    # Variation settings
    variation_count: int = 3,
    variation_strength: float = 0.5,
    
    # Content parameters
    prompt: str = "background music",
    lyrics: str = "",
    
    # Model parameters
    model_name: str = DEFAULT_MODEL,
    bf16: bool = True,
    torch_compile: bool = False,
    cpu_offload: bool = False,
    device_id: int = 0,
    
    # Generation parameters
    infer_step: int = 27,
    guidance_scale: float = 7.5,
    scheduler_type: str = "flow_match_euler",
    cfg_type: str = "apg",
    omega_scale: float = 10.0,
    seed: Optional[int] = None,
    
    # Output parameters
    output_filename: Optional[str] = None,
    progress_callback: Optional[callable] = None
) -> Tuple[List[str], str]:
    """
    Generate variations of existing music using ACE-Step model.
    
    Args:
        source_audio_path: Path to the source audio file
        variation_count: Number of variations to generate
        variation_strength: How much to vary from original (0.0-1.0)
        prompt: Prompt describing the audio content
        lyrics: Lyrics for the audio (if any)
        model_name: Name of the ACE-Step model to use
        bf16: Whether to use bfloat16 precision
        torch_compile: Whether to use torch compile for faster inference
        cpu_offload: Whether to offload models to CPU to save VRAM
        device_id: GPU device ID to use
        infer_step: Number of inference steps
        guidance_scale: Guidance scale for generation
        scheduler_type: Scheduler type (flow_match_euler, euler, dpm)
        cfg_type: CFG type (apg, cfg)
        omega_scale: Omega scale parameter
        seed: Random seed for reproducibility
        output_filename: Custom output filename (optional)
        progress_callback: Callback function for progress updates
        
    Returns:
        Tuple[List[str], str]: List of paths to the generated audio files, message
    """
    # Ensure source audio exists
    if not source_audio_path or not os.path.exists(source_audio_path):
        return [], "Source audio file not found. Please provide a valid audio file."
    
    # Validate parameters
    if variation_count <= 0:
        return [], "Variation count must be greater than 0."
    
    if variation_strength < 0.0 or variation_strength > 1.0:
        return [], "Variation strength must be between 0.0 and 1.0."
    
    # Ensure ACE-Step is available
    if not check_acestep_installed():
        return [], "ACE-Step is not installed. Please install required dependencies."
    
    # Progress update
    if progress_callback is not None:
        progress_callback(0.1, "Checking model availability...")
    
    # Download/retrieve model
    try:
        checkpoint_path = download_acestep_model(model_name)
    except Exception as e:
        logger.error(f"Failed to download/retrieve model: {e}")
        return [], f"Failed to download/retrieve model: {e}"
    
    # Progress update
    if progress_callback is not None:
        progress_callback(0.2, "Initializing pipeline...")
    
    # Initialize pipeline
    try:
        pipeline = initialize_acestep_pipeline(
            checkpoint_path=checkpoint_path,
            bf16=bf16,
            torch_compile=torch_compile,
            cpu_offload=cpu_offload,
            overlapped_decode=False,  # Disable overlapped decoding for retake
            device_id=device_id
        )
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        return [], f"Failed to initialize pipeline: {e}"
    
    # Progress update
    if progress_callback is not None:
        progress_callback(0.3, "Analyzing source audio...")
    
    # Get audio duration from source file
    try:
        import torchaudio
        audio, sr = torchaudio.load(source_audio_path)
        audio_duration = audio.shape[1] / sr
        logger.info(f"Source audio duration: {audio_duration:.2f} seconds")
    except Exception as e:
        logger.error(f"Failed to get audio duration: {e}")
        # Use default duration if analysis fails
        audio_duration = 60.0
    
    # Generate output path
    os.makedirs(os.path.join(output_path, "acestep"), exist_ok=True)
    
    # Generate variations
    variation_paths = []
    current_progress = 0.4  # Starting progress after initialization
    progress_per_variation = 0.6 / variation_count  # Remaining progress divided by variation count
    
    # Set up seeds for reproducibility and diversity
    if seed is None:
        # Use a single random seed if none provided
        current_seed = torch.randint(0, 2**32 - 1, (1,)).item()
    else:
        # Use provided seed for first variation
        current_seed = seed
    
    # Use the same or incremented seeds for all variations
    variation_seeds = [current_seed + i for i in range(variation_count)]
    
    for i in range(variation_count):
        # Set current seed
        current_seed = variation_seeds[i]
        
        # Progress update
        if progress_callback is not None:
            progress_callback(current_progress, f"Generating variation {i+1}/{variation_count}...")
        
        # Generate filename for this variation
        timestamp = uuid.uuid4().hex[:8]
        if output_filename:
            # Use provided filename with suffix for variations
            base_name, ext = os.path.splitext(output_filename)
            if not ext:
                ext = ".wav"
            variation_filename = f"{base_name}_var{i+1}{ext}"
        else:
            # Generate filename based on timestamp and seed
            variation_filename = f"acestep_variation_{timestamp}_{current_seed}_var{i+1}.wav"
        
        variation_path = os.path.join(output_path, "acestep", variation_filename)
        
        try:
            # Run the ACE-Step pipeline in retake mode
            # Note: The retake functionality in ACE-Step requires a prompt
            # We use a generic prompt and empty lyrics as fallback
            generic_prompt = "background music"
            
            # Ensure scheduler_type is one of the valid values to prevent undefined scheduler
            if scheduler_type not in ["euler", "heun", "flow_match_euler"]:
                logger.warning(f"Invalid scheduler_type '{scheduler_type}'. Defaulting to 'euler'.")
                scheduler_type = "euler"
            
            pipeline(
                audio_duration=audio_duration,
                task="repaint",  # Use repaint instead of retake task mode (works the same)
                src_audio_path=source_audio_path,  # Source audio
                prompt=prompt if prompt else generic_prompt,  # Using provided prompt or fallback
                lyrics=lyrics,  # Using provided lyrics or fallback
                retake_variance=variation_strength,  # Control variation strength
                manual_seeds=str(current_seed),  # Use specific seed
                retake_seeds=str(current_seed),  # Use specific seed for retake
                infer_step=infer_step,
                guidance_scale=guidance_scale,
                scheduler_type=scheduler_type,  # Explicitly use the validated scheduler_type
                cfg_type=cfg_type,
                omega_scale=omega_scale,
                repaint_start=0,  # Start from beginning
                repaint_end=int(audio_duration),  # Use entire duration, converted to int
                save_path=variation_path
            )
            
            # Add to paths if successful
            variation_paths.append(variation_path)
            
        except Exception as e:
            logger.error(f"Error generating variation {i+1}: {e}")
            import traceback
            traceback.print_exc()
            # Continue with next variation even if this one failed
        
        # Update progress
        current_progress += progress_per_variation
    
    # Final progress update
    if progress_callback is not None:
        progress_callback(1.0, "Variations generation complete!")
    
    # Return results
    if not variation_paths:
        return [], "Failed to generate any variations."
    
    return variation_paths, f"Successfully generated {len(variation_paths)} variations with strength {variation_strength}."

def process_repaint(
    # Source audio
    source_audio_path: str,
    
    # Repaint settings
    start_time: float = 0.0,
    end_time: float = 0.0,
    repaint_strength: float = 0.8,
    
    # Model parameters
    model_name: str = DEFAULT_MODEL,
    bf16: bool = True,
    torch_compile: bool = False,
    cpu_offload: bool = False,
    device_id: int = 0,
    
    # Generation parameters
    infer_step: int = 27,
    guidance_scale: float = 7.5,
    scheduler_type: str = "flow_match_euler",
    cfg_type: str = "apg",
    omega_scale: float = 10.0,
    seed: Optional[int] = None,
    
    # Output parameters
    output_filename: Optional[str] = None,
    progress_callback: Optional[callable] = None
) -> Tuple[str, str]:
    """
    Selectively regenerate specific sections of music using ACE-Step model.
    
    Args:
        source_audio_path: Path to the source audio file
        start_time: Start time of the section to repaint (in seconds)
        end_time: End time of the section to repaint (in seconds)
        repaint_strength: Strength of repainting (0.1-1.0, higher = more different)
        model_name: Name of the ACE-Step model to use
        bf16: Whether to use bfloat16 precision
        torch_compile: Whether to use torch compile for faster inference
        cpu_offload: Whether to offload models to CPU to save VRAM
        device_id: GPU device ID to use
        infer_step: Number of inference steps
        guidance_scale: Guidance scale for generation
        scheduler_type: Scheduler type (flow_match_euler, euler, dpm)
        cfg_type: CFG type (apg, cfg)
        omega_scale: Omega scale parameter
        seed: Random seed for reproducibility
        output_filename: Custom output filename (optional)
        progress_callback: Callback function for progress updates
        
    Returns:
        Tuple[str, str]: Path to the generated audio file, message
    """
    # Ensure source audio exists
    if not source_audio_path or not os.path.exists(source_audio_path):
        return "", "Source audio file not found. Please provide a valid audio file."
    
    # Validate time range
    if end_time <= start_time and end_time != 0:
        return "", "End time must be greater than start time."
    
    # Ensure ACE-Step is available
    if not check_acestep_installed():
        return "", "ACE-Step is not installed. Please install required dependencies."
    
    # Progress update
    if progress_callback is not None:
        progress_callback(0.1, "Checking model availability...")
    
    # Download/retrieve model
    try:
        checkpoint_path = download_acestep_model(model_name)
    except Exception as e:
        logger.error(f"Failed to download/retrieve model: {e}")
        return "", f"Failed to download/retrieve model: {e}"
    
    # Progress update
    if progress_callback is not None:
        progress_callback(0.2, "Initializing pipeline...")
    
    # Initialize pipeline
    try:
        pipeline = initialize_acestep_pipeline(
            checkpoint_path=checkpoint_path,
            bf16=bf16,
            torch_compile=torch_compile,
            cpu_offload=cpu_offload,
            overlapped_decode=False,  # Disable overlapped decoding for repaint
            device_id=device_id
        )
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        return "", f"Failed to initialize pipeline: {e}"
    
    # Progress update
    if progress_callback is not None:
        progress_callback(0.3, "Analyzing source audio...")
    
    # Get audio duration from source file
    try:
        import torchaudio
        audio, sr = torchaudio.load(source_audio_path)
        audio_duration = audio.shape[1] / sr
        logger.info(f"Source audio duration: {audio_duration:.2f} seconds")
        
        # Validate time range against audio length
        if end_time > audio_duration:
            logger.warning(f"End time {end_time} exceeds audio duration {audio_duration}. Clamping to {audio_duration}.")
            end_time = audio_duration
    except Exception as e:
        logger.error(f"Failed to get audio duration: {e}")
        # Use default duration if analysis fails
        audio_duration = 60.0
    
    # Generate output path
    os.makedirs(os.path.join(output_path, "acestep"), exist_ok=True)
    timestamp = uuid.uuid4().hex[:8]
    
    # Set up seed
    if seed is None:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    
    # Generate filename
    if output_filename:
        # Use provided filename but ensure it has .wav extension
        if not output_filename.endswith(".wav"):
            output_filename = f"{output_filename}.wav"
        output_path_file = os.path.join(output_path, "acestep", output_filename)
    else:
        # Generate filename based on timestamp and seed
        output_path_file = os.path.join(output_path, "acestep", f"acestep_repaint_{timestamp}_{seed}.wav")
    
    # Progress update
    if progress_callback is not None:
        progress_callback(0.4, f"Repainting section from {start_time:.1f}s to {end_time:.1f}s...")
    
    try:
        # Convert start and end times to integers for the pipeline
        repaint_start_int = int(start_time)
        repaint_end_int = int(end_time)
        
        # Ensure scheduler_type is one of the valid values to prevent undefined scheduler
        if scheduler_type not in ["euler", "heun", "flow_match_euler"]:
            logger.warning(f"Invalid scheduler_type '{scheduler_type}'. Defaulting to 'euler'.")
            scheduler_type = "euler"
        
        # Run the ACE-Step pipeline in repaint mode
        pipeline(
        audio_duration=audio_duration,
            task="repaint",  # Use repaint task mode
            src_audio_path=source_audio_path,  # Source audio
            repaint_start=repaint_start_int,  # Start time in seconds, as integer
            repaint_end=repaint_end_int,  # End time in seconds, as integer
            retake_variance=repaint_strength,  # Control repaint strength
            manual_seeds=str(seed),  # Use specific seed
            retake_seeds=str(seed),  # Use specific seed for repaint
            infer_step=infer_step,
            guidance_scale=guidance_scale,
            scheduler_type=scheduler_type,  # Explicitly use the validated scheduler_type
            cfg_type=cfg_type,
            omega_scale=omega_scale,
            save_path=output_path_file
        )
        
        # Progress update
        if progress_callback is not None:
            progress_callback(1.0, "Repainting complete!")
        
        section_info = f"from {start_time:.1f}s to {end_time:.1f}s"
        return output_path_file, f"Successfully repainted audio section {section_info} with strength {repaint_strength} using seed {seed}"
    
    except Exception as e:
        logger.error(f"Error repainting audio: {e}")
        import traceback
        traceback.print_exc()
        return "", f"Error repainting audio: {e}"

def process_edit(
    # Source audio
    source_audio_path: str,
    
    # Edit settings
    start_time: float = 0.0,
    end_time: float = 0.0,
    current_lyrics: str = "",
    new_lyrics: str = "",
    edit_mode: str = "only_lyrics",  # "only_lyrics" or "remix"
    
    # Model parameters
    model_name: str = DEFAULT_MODEL,
    bf16: bool = True,
    torch_compile: bool = False,
    cpu_offload: bool = False,
    device_id: int = 0,
    
    # Generation parameters
    infer_step: int = 27,
    guidance_scale: float = 7.5,
    scheduler_type: str = "flow_match_euler",
    cfg_type: str = "apg",
    omega_scale: float = 10.0,
    seed: Optional[int] = None,
    
    # Output parameters
    output_filename: Optional[str] = None,
    progress_callback: Optional[callable] = None
) -> Tuple[str, str]:
    """
    Edit lyrics in existing music using ACE-Step model.
    
    Args:
        source_audio_path: Path to the source audio file
        start_time: Start time of the section to edit (in seconds)
        end_time: End time of the section to edit (in seconds)
        current_lyrics: Current lyrics in the audio (for reference)
        new_lyrics: New lyrics to replace in the selected section
        edit_mode: Mode of editing ("only_lyrics" or "remix")
        model_name: Name of the ACE-Step model to use
        bf16: Whether to use bfloat16 precision
        torch_compile: Whether to use torch compile for faster inference
        cpu_offload: Whether to offload models to CPU to save VRAM
        device_id: GPU device ID to use
        infer_step: Number of inference steps
        guidance_scale: Guidance scale for generation
        scheduler_type: Scheduler type (flow_match_euler, euler, dpm)
        cfg_type: CFG type (apg, cfg)
        omega_scale: Omega scale parameter
        seed: Random seed for reproducibility
        output_filename: Custom output filename (optional)
        progress_callback: Callback function for progress updates
        
    Returns:
        Tuple[str, str]: Path to the generated audio file, message
    """
    # Ensure source audio exists
    if not source_audio_path or not os.path.exists(source_audio_path):
        return "", "Source audio file not found. Please provide a valid audio file."
    
    # Validate parameters
    if not new_lyrics or new_lyrics.strip() == "":
        return "", "New lyrics cannot be empty."
    
    if edit_mode not in ["only_lyrics", "remix"]:
        return "", "Invalid edit mode. Must be 'only_lyrics' or 'remix'."
    
    if end_time <= start_time and end_time != 0:
        return "", "End time must be greater than start time."
    
    # Ensure ACE-Step is available
    if not check_acestep_installed():
        return "", "ACE-Step is not installed. Please install required dependencies."
    
    # Progress update
    if progress_callback is not None:
        progress_callback(0.1, "Checking model availability...")
    
    # Download/retrieve model
    try:
        checkpoint_path = download_acestep_model(model_name)
    except Exception as e:
        logger.error(f"Failed to download/retrieve model: {e}")
        return "", f"Failed to download/retrieve model: {e}"
    
    # Progress update
    if progress_callback is not None:
        progress_callback(0.2, "Initializing pipeline...")
    
    # Initialize pipeline
    try:
        pipeline = initialize_acestep_pipeline(
            checkpoint_path=checkpoint_path,
        bf16=bf16,
        torch_compile=torch_compile,
        cpu_offload=cpu_offload,
            overlapped_decode=False,  # Disable overlapped decoding for edit
            device_id=device_id
        )
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        return "", f"Failed to initialize pipeline: {e}"
    
    # Progress update
    if progress_callback is not None:
        progress_callback(0.3, "Analyzing source audio...")
    
    # Get audio duration from source file
    try:
        import torchaudio
        audio, sr = torchaudio.load(source_audio_path)
        audio_duration = audio.shape[1] / sr
        
        # If end_time is 0 or exceeds duration, use the full duration
        if end_time <= 0:
            end_time = audio_duration
            
        # Validate time range against audio length
        if end_time > audio_duration:
            logger.warning(f"End time {end_time} exceeds audio duration {audio_duration}. Clamping to {audio_duration}.")
            end_time = audio_duration
            
    except Exception as e:
        logger.error(f"Failed to get audio duration: {e}")
        audio_duration = 60.0
        if end_time <= 0:
            end_time = audio_duration
    
    # Generate output path
    os.makedirs(os.path.join(output_path, "acestep"), exist_ok=True)
    timestamp = uuid.uuid4().hex[:8]
    
    # Set up seed
    if seed is None:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    
    # Generate filename
    if output_filename:
        # Use provided filename but ensure it has .wav extension
        if not output_filename.endswith(".wav"):
            output_filename = f"{output_filename}.wav"
        output_path_file = os.path.join(output_path, "acestep", output_filename)
    else:
        # Generate filename based on timestamp and seed
        output_path_file = os.path.join(output_path, "acestep", f"acestep_edit_{timestamp}_{seed}.wav")
    
    # Progress update
    if progress_callback is not None:
        progress_callback(0.4, f"Editing lyrics in section from {start_time:.1f}s to {end_time:.1f}s...")
    
    # Set edit parameters based on mode
    edit_n_min = 0.0
    edit_n_max = 1.0
    if edit_mode == "only_lyrics":
        # Only change the lyrics, preserve most of the music
        edit_n_min = 0.0
        edit_n_max = 0.3
    else:  # remix mode
        # More aggressive change that affects both lyrics and music
        edit_n_min = 0.3
        edit_n_max = 0.8
    
    try:
        # Convert start and end times to integers for the pipeline
        repaint_start_int = int(start_time)
        repaint_end_int = int(end_time)
        
        # Ensure scheduler_type is one of the valid values to prevent undefined scheduler
        if scheduler_type not in ["euler", "heun", "flow_match_euler"]:
            logger.warning(f"Invalid scheduler_type '{scheduler_type}'. Defaulting to 'euler'.")
            scheduler_type = "euler"
        
        # Run the ACE-Step pipeline in edit mode
        pipeline(
            audio_duration=audio_duration,
            task="edit",  # Use edit task mode
            src_audio_path=source_audio_path,  # Source audio
            prompt="",  # Original prompt (empty for now)
            lyrics=current_lyrics,  # Current lyrics
            edit_target_prompt="",  # Target prompt (empty for now)
            edit_target_lyrics=new_lyrics,  # New lyrics
            edit_n_min=edit_n_min,  # Control how much is preserved
            edit_n_max=edit_n_max,  # Control how much is changed
            edit_n_avg=1,  # Number of samples to average
            repaint_start=repaint_start_int,  # Start time in seconds (use for edit range), as integer
            repaint_end=repaint_end_int,  # End time in seconds (use for edit range), as integer
            manual_seeds=str(seed),  # Use specific seed
            retake_seeds=str(seed),  # Use specific seed for edit
        infer_step=infer_step,
        guidance_scale=guidance_scale,
            scheduler_type=scheduler_type,  # Explicitly use the validated scheduler_type
        cfg_type=cfg_type,
        omega_scale=omega_scale,
            save_path=output_path_file
        )
        
        # Progress update
        if progress_callback is not None:
            progress_callback(1.0, "Lyric editing complete!")
        
        mode_text = "only modifying lyrics" if edit_mode == "only_lyrics" else "remixing the section"
        section_info = f"from {start_time:.1f}s to {end_time:.1f}s"
        return output_path_file, f"Successfully edited lyrics {section_info} in {mode_text} mode using seed {seed}"
    
    except Exception as e:
        logger.error(f"Error editing lyrics: {e}")
        import traceback
        traceback.print_exc()
        return "", f"Error editing lyrics: {e}" 