"""
Model utilities for the cloning module.

This module provides utilities for handling model downloads and loading,
ensuring all models are stored in the configured model_path directory.
"""

import os
import logging
import tempfile
import shutil
import requests
from pathlib import Path
from typing import Optional, Dict, Any, Union, List, Tuple
from tqdm import tqdm

from handlers.config import model_path

logger = logging.getLogger(__name__)

# Ensure model directory exists
os.makedirs(model_path, exist_ok=True)


def download_file(url: str, destination: str) -> bool:
    """
    Download a file from a URL to a destination with progress tracking.
    
    Args:
        url: URL to download from
        destination: Path to save the file to
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        # Download with progress bar
        response = requests.get(url, stream=True)
        if not response.ok:
            logger.error(f"Failed to download from {url}, status code: {response.status_code}")
            return False
            
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        
        logger.info(f"Downloading {url} to {destination}")
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        
        with open(destination, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
                
        progress_bar.close()
        
        # Verify file was downloaded correctly
        if total_size_in_bytes > 0 and progress_bar.n != total_size_in_bytes:
            logger.error("Downloaded file size doesn't match expected size")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        return False


def check_model(model_type: str, model_name: str, files_to_download: Dict[str, str]) -> str:
    """
    Check if a model exists in the model_path directory, download it if not.
    
    Args:
        model_type: Type of model (e.g., 'openvoice', 'tts', 'diarization')
        model_name: Name of the model or model directory
        files_to_download: Dictionary mapping filenames to download URLs
        
    Returns:
        Path to the model directory
    """
    # Create model-specific directory
    model_dir = os.path.join(model_path, model_type, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Check if all files exist
    all_exist = True
    for filename in files_to_download:
        file_path = os.path.join(model_dir, filename)
        if not os.path.exists(file_path):
            all_exist = False
            break
    
    # Download missing files
    if not all_exist:
        logger.info(f"Downloading model files for {model_type}/{model_name}")
        for filename, url in files_to_download.items():
            file_path = os.path.join(model_dir, filename)
            if not os.path.exists(file_path):
                success = download_file(url, file_path)
                if not success:
                    logger.error(f"Failed to download {filename} for {model_type}/{model_name}")
                    raise RuntimeError(f"Failed to download required model file: {filename}")
    
    return model_dir


def get_huggingface_model_path(model_id: str, model_type: str, use_auth_token: Optional[str] = None) -> str:
    """
    Get the path to a HuggingFace model, downloading it to model_path if needed.
    Uses the Huggingface API to get model files but stores them in our model directory.
    
    Args:
        model_id: HuggingFace model ID (e.g., 'pyannote/speaker-diarization-3.1')
        model_type: Type of model for local storage classification
        use_auth_token: HuggingFace authentication token for private models
        
    Returns:
        Path to the model directory
    """
    try:
        from huggingface_hub import snapshot_download, HfApi
        
        # Replace slashes with underscores for the directory name
        safe_model_name = model_id.replace('/', '_')
        model_dir = os.path.join(model_path, model_type, safe_model_name)
        
        # Check if model is already downloaded
        if os.path.exists(model_dir) and len(os.listdir(model_dir)) > 0:
            logger.info(f"Model {model_id} already exists at {model_dir}")
            return model_dir
            
        # Create a temporary directory to download the model
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger.info(f"Downloading model {model_id} to temporary directory")
            
            # Download the model to the temporary directory
            snapshot_path = snapshot_download(
                repo_id=model_id,
                local_dir=tmp_dir,
                token=use_auth_token,
                local_dir_use_symlinks=False
            )
            
            # Create the model directory
            os.makedirs(model_dir, exist_ok=True)
            
            # Move files from temporary directory to model directory
            for item in os.listdir(snapshot_path):
                src_path = os.path.join(snapshot_path, item)
                dst_path = os.path.join(model_dir, item)
                
                if os.path.isfile(src_path):
                    shutil.copy2(src_path, dst_path)
                elif os.path.isdir(src_path):
                    if os.path.exists(dst_path):
                        shutil.rmtree(dst_path)
                    shutil.copytree(src_path, dst_path)
            
            logger.info(f"Model {model_id} downloaded to {model_dir}")
            return model_dir
            
    except Exception as e:
        logger.error(f"Error getting HuggingFace model: {e}")
        raise RuntimeError(f"Failed to download model {model_id}: {str(e)}")


def load_pyannote_pipeline(model_id: str, use_auth_token: Optional[str] = None):
    """
    Load a pyannote.audio Pipeline model from the model_path directory.
    
    Args:
        model_id: Model ID (e.g., 'pyannote/speaker-diarization-3.1')
        use_auth_token: HuggingFace authentication token
        
    Returns:
        Loaded pyannote.audio Pipeline object
    """
    try:
        from pyannote.audio import Pipeline
        
        # Get the model path
        model_path_dir = get_huggingface_model_path(model_id, "diarization", use_auth_token)
        
        # Load the model from the local path
        pipeline = Pipeline.from_pretrained(model_path_dir)
        
        return pipeline
        
    except Exception as e:
        logger.error(f"Error loading pyannote.audio Pipeline: {e}")
        raise RuntimeError(f"Failed to load pyannote model {model_id}: {str(e)}")


def initialize_openvoice_model() -> Tuple[str, str]:
    """
    Initialize the OpenVoice v2 model, downloading and extracting it to model_path if needed.

    Returns:
        Tuple of (config_path, checkpoint_path)
    """
    v2_zip_url = "https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip"

    # Extract to model_path/openvoice/ (the zip contains checkpoints_v2/ subdirectory)
    model_dir = os.path.join(model_path, "openvoice")
    os.makedirs(model_dir, exist_ok=True)

    # Check if the checkpoints_v2/converter directory exists (indicates successful extraction)
    checkpoints_v2_dir = os.path.join(model_dir, "checkpoints_v2")
    converter_dir = os.path.join(checkpoints_v2_dir, "converter")

    if not os.path.exists(converter_dir) or not os.listdir(converter_dir):
        logger.info("Downloading OpenVoice v2 model...")
        success = download_and_extract_zip(v2_zip_url, model_dir)
        if not success:
            raise RuntimeError("Failed to download OpenVoice v2 model")

    # Verify the model files exist
    config_path = os.path.join(converter_dir, "config.json")
    checkpoint_path = os.path.join(converter_dir, "checkpoint.pth")

    if not os.path.exists(config_path) or not os.path.exists(checkpoint_path):
        raise RuntimeError("OpenVoice v2 model files not found after extraction")

    return config_path, checkpoint_path


def download_and_extract_zip(url: str, extract_to: str) -> bool:
    """
    Download a zip file and extract it to the specified directory.

    Args:
        url: URL of the zip file to download
        extract_to: Directory to extract the zip contents to

    Returns:
        True if successful, False otherwise
    """
    try:
        import zipfile
        import tempfile
        import requests

        # Create temporary file for download
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_file:
            tmp_path = tmp_file.name

        # Download the zip file
        logger.info(f"Downloading OpenVoice v2 from {url}")
        response = requests.get(url, stream=True)
        if not response.ok:
            logger.error(f"Failed to download from {url}, status code: {response.status_code}")
            return False

        # Save to temporary file
        with open(tmp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        # Extract to destination
        logger.info(f"Extracting OpenVoice v2 to {extract_to}")
        with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

        # Clean up temporary file
        os.unlink(tmp_path)

        logger.info("OpenVoice v2 extraction complete")
        return True

    except Exception as e:
        logger.error(f"Error downloading/extracting OpenVoice v2: {e}")
        return False 