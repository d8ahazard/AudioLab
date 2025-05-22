"""
DiffRhythm inference utility functions

This implementation is adapted from the original DiffRhythm project:
https://github.com/ASLP-lab/DiffRhythm
"""

import torch
import librosa
import random
import json
from muq import MuQMuLan
from mutagen.mp3 import MP3
import os
import shutil
import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download
from mutagen.mp3 import MP3
from handlers.config import model_path, app_path

from modules.diffrythm.model import DiT, CFM

def check_download_model(repo_id):
    """
    Check if the model exists, download it if not, and return its path.
    
    Args:
        repo_id: Repository ID with or without 'ASLP-lab/' prefix
        
    Returns:
        Path to the model file
    """
    # Model file mapping
    model_files = {
        "DiffRhythm-vae": "vae_model.pt",
        "DiffRhythm-full": "cfm_model.pt",
        "DiffRhythm-base": "cfm_model.pt"
    }
    
    # Handle repo_id with or without ASLP-lab/ prefix
    repo_name = repo_id.replace("ASLP-lab/", "")
    
    # Get the correct model filename
    if repo_name not in model_files:
        raise ValueError(f"Unknown repository ID: {repo_id}")
    
    model_file = model_files[repo_name]
    
    # Define paths
    model_dir = os.path.join(model_path, "diffrythm", repo_name)
    model_file_path = os.path.join(model_dir, model_file)
    
    # If model already exists, return its path
    if os.path.exists(model_file_path):
        return model_file_path
    
    # Ensure model directory exists
    os.makedirs(model_dir, exist_ok=True)
    
    # Define direct download URLs
    download_urls = {
        "DiffRhythm-vae": f"https://huggingface.co/ASLP-lab/DiffRhythm-vae/resolve/main/{model_file}",
        "DiffRhythm-full": f"https://huggingface.co/ASLP-lab/DiffRhythm-full/resolve/main/{model_file}",
        "DiffRhythm-base": f"https://huggingface.co/ASLP-lab/DiffRhythm-base/resolve/main/{model_file}"
    }
    
    # Get the direct download URL
    download_url = download_urls[repo_name]
    
    try:
        # Try direct download first (avoiding HF Hub)
        import requests
        print(f"Downloading {repo_name} model from {download_url}...")
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        
        # Save the file
        with open(model_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Model downloaded successfully to {model_file_path}")
        return model_file_path
        
    except Exception as e:
        print(f"Direct download failed: {e}")
        print("Falling back to huggingface_hub...")
        
        try:
            # Fall back to hf_hub_download if direct download fails
            full_repo_id = f"ASLP-lab/{repo_name}"
            downloaded_path = hf_hub_download(repo_id=full_repo_id, filename=model_file, cache_dir=model_dir)
            
            # Ensure the file is at the expected location
            if downloaded_path != model_file_path and os.path.exists(downloaded_path):
                shutil.copy(downloaded_path, model_file_path)
            
            # Clean up by deleting any extra files in the model directory
            for file in os.listdir(model_dir):
                file_path = os.path.join(model_dir, file)
                if file_path != model_file_path and os.path.isfile(file_path):
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        print(f"Warning: Could not remove {file_path}: {e}")
            
            return model_file_path
            
        except Exception as e:
            raise ValueError(f"Failed to download model {repo_name}: {e}")
        

def fetch_model(repo_id):
    base_model_dir = os.path.join(model_path, "diffrythm")
    repo_name = repo_id.replace("/", "_")
    model_dir = os.path.join(base_model_dir, repo_name)
    
    if os.path.exists(model_dir):
        return model_dir
    
    model_dir = snapshot_download(repo_id, cache_dir=model_dir)
    return model_dir


def decode_audio(latents, vae_model, chunked=False, overlap=32, chunk_size=128):
    """
    Decode audio from latents using VAE model
    
    Args:
        latents: Audio latents
        vae_model: VAE model for decoding
        chunked: Whether to use chunked decoding
        overlap: Overlap between chunks when using chunked decoding
        chunk_size: Size of chunks when using chunked decoding
        
    Returns:
        Tensor: Decoded audio
    """
    # Ensure latents are in the correct dtype for the VAE model
    downsampling_ratio = 2048
    io_channels = 2
    
    if not chunked:
        # Direct decoding
        return vae_model.decode_export(latents)
    else:
        # chunked decoding
        hop_size = chunk_size - overlap
        total_size = latents.shape[2]
        batch_size = latents.shape[0]
        chunks = []
        i = 0
        for i in range(0, total_size - chunk_size + 1, hop_size):
            chunk = latents[:, :, i : i + chunk_size]
            chunks.append(chunk)
        if i + chunk_size != total_size:
            # Final chunk
            chunk = latents[:, :, -chunk_size:]
            chunks.append(chunk)
        chunks = torch.stack(chunks)
        num_chunks = chunks.shape[0]
        # samples_per_latent is just the downsampling ratio
        samples_per_latent = downsampling_ratio
        # Create an empty waveform, we will populate it with chunks as decode them
        y_size = total_size * samples_per_latent
        y_final = torch.zeros((batch_size, io_channels, y_size)).to(latents.device)
        for i in range(num_chunks):
            x_chunk = chunks[i, :]
            # decode the chunk
            y_chunk = vae_model.decode_export(x_chunk)
            # figure out where to put the audio along the time domain
            if i == num_chunks - 1:
                # final chunk always goes at the end
                t_end = y_size
                t_start = t_end - y_chunk.shape[2]
            else:
                t_start = i * hop_size * samples_per_latent
                t_end = t_start + chunk_size * samples_per_latent
            #  remove the edges of the overlaps
            ol = (overlap // 2) * samples_per_latent
            chunk_start = 0
            chunk_end = y_chunk.shape[2]
            if i > 0:
                # no overlap for the start of the first chunk
                t_start += ol
                chunk_start += ol
            if i < num_chunks - 1:
                # no overlap for the end of the last chunk
                t_end -= ol
                chunk_end -= ol
            # paste the chunked audio into our y_final output audio
            y_final[:, :, t_start:t_end] = y_chunk[:, :, chunk_start:chunk_end]
            
        return y_final


def prepare_model(max_frames, device, repo_id="ASLP-lab/DiffRhythm-base"):
    """
    Prepare DiffRhythm models for inference
    
    Args:
        max_frames: Maximum number of frames to generate
        device: Device to run inference on
        repo_id: HuggingFace repository ID
        
    Returns:
        Tuple: (CFM model, tokenizer, MuQ model, VAE model)
    """
    # prepare cfm model
    dit_ckpt_path = check_download_model(repo_id=repo_id)
    dit_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "diffrhythm-1b.json")
    with open(dit_config_path) as f:
        model_config = json.load(f)
    dit_model_cls = DiT
    
    # Create model with original precision
    cfm = CFM(
        transformer=dit_model_cls(**model_config["model"], max_frames=max_frames),
        num_channels=model_config["model"]["mel_dim"],
        max_frames=max_frames
    )
    
    # Move to device first
    cfm = cfm.to(device)
    cfm = load_checkpoint(cfm, dit_ckpt_path, device=device, use_ema=False)

    # prepare tokenizer
    tokenizer = CNENTokenizer()

    muq_path = fetch_model("OpenMuQ/MuQ-MuLan-large")
    # prepare muq
    muq = MuQMuLan.from_pretrained(muq_path)
    muq = muq.to(device).eval()

    # prepare vae
    vae_ckpt_path = check_download_model(repo_id="ASLP-lab/DiffRhythm-vae")
    vae = torch.jit.load(vae_ckpt_path, map_location="cpu").to(device)

    return cfm, tokenizer, muq, vae


# for song edit, will be added in the future
def get_reference_latent(device, max_frames):
    return torch.zeros(1, max_frames, 64).to(device)


def get_negative_style_prompt(device):
    file_path = os.path.join(app_path, "modules", "diffrythm", "infer", "example", "vocal.npy")
    vocal_stlye = np.load(file_path)

    vocal_stlye = torch.from_numpy(vocal_stlye).to(device)  # [1, 512]
    vocal_stlye = vocal_stlye.half()

    return vocal_stlye


@torch.no_grad()
def get_style_prompt(model, wav_path=None, prompt=None):
    """
    Get a style prompt for song generation
    
    Args:
        model: MuQ model for extracting style from audio or text
        wav_path: Path to reference audio file (optional)
        prompt: Text prompt (optional)
        
    Returns:
        Tensor: Style prompt embedding
    """
    mulan = model

    if prompt is not None:
        return mulan(texts=prompt).half()

    ext = os.path.splitext(wav_path)[-1].lower()
    if ext == ".mp3":
        meta = MP3(wav_path)
        audio_len = meta.info.length
    elif ext in [".wav", ".flac"]:
        audio_len = librosa.get_duration(path=wav_path)
    else:
        raise ValueError("Unsupported file format: {}".format(ext))

    if audio_len < 10:
        print(
            f"Warning: The audio file {wav_path} is too short ({audio_len:.2f} seconds). Expected at least 10 seconds."
        )

    assert audio_len >= 10

    mid_time = audio_len // 2
    start_time = mid_time - 5
    wav, _ = librosa.load(wav_path, sr=24000, offset=start_time, duration=10)

    wav = torch.tensor(wav).unsqueeze(0).to(model.device)

    with torch.no_grad():
        audio_emb = mulan(wavs=wav)  # [1, 512]

    audio_emb = audio_emb
    audio_emb = audio_emb.half()
    return audio_emb


def parse_lyrics(lyrics: str):
    """
    Parse lyrics with timestamps
    
    Args:
        lyrics: Lyrics with timestamps in the format [MM:SS.xx]lyrics
        
    Returns:
        List: List of (timestamp, lyrics) tuples
    """
    lyrics_with_time = []
    lyrics = lyrics.strip()
    for line in lyrics.split("\n"):
        try:
            time, lyric = line[1:9], line[10:]
            lyric = lyric.strip()
            mins, secs = time.split(":")
            secs = int(mins) * 60 + float(secs)
            lyrics_with_time.append((secs, lyric))
        except:
            continue
    return lyrics_with_time


class CNENTokenizer:
    def __init__(self):
        vocab_path = os.path.join(app_path, "modules", "diffrythm", "g2p", "g2p", "vocab.json")
        with open(vocab_path, "r", encoding='utf-8') as file:
            self.phone2id: dict = json.load(file)["vocab"]
        self.id2phone = {v: k for (k, v) in self.phone2id.items()}
        from modules.diffrythm.g2p.g2p_generation import chn_eng_g2p

        self.tokenizer = chn_eng_g2p

    def encode(self, text):
        phone, token = self.tokenizer(text)
        token = [x + 1 for x in token]
        return token

    def decode(self, token):
        return "|".join([self.id2phone[x - 1] for x in token])


def get_lrc_token(max_frames, text, tokenizer, device):
    """
    Convert lyrics to tokens
    
    Args:
        max_frames: Maximum number of frames
        text: Lyrics with timestamps
        tokenizer: Tokenizer for converting text to token IDs
        device: Device to place tensors on
        
    Returns:
        Tuple: (lyrics tokens, normalized start time)
    """
    lyrics_shift = 0
    sampling_rate = 44100
    downsample_rate = 2048
    max_secs = max_frames / (sampling_rate / downsample_rate)

    comma_token_id = 1
    period_token_id = 2

    lrc_with_time = parse_lyrics(text)

    modified_lrc_with_time = []
    for i in range(len(lrc_with_time)):
        time, line = lrc_with_time[i]
        line_token = tokenizer.encode(line)
        modified_lrc_with_time.append((time, line_token))
    lrc_with_time = modified_lrc_with_time

    lrc_with_time = [
        (time_start, line)
        for (time_start, line) in lrc_with_time
        if time_start < max_secs
    ]
    if max_frames == 2048:
        lrc_with_time = lrc_with_time[:-1] if len(lrc_with_time) >= 1 else lrc_with_time

    normalized_start_time = 0.0

    lrc = torch.zeros((max_frames,), dtype=torch.long)

    tokens_count = 0
    last_end_pos = 0
    for time_start, line in lrc_with_time:
        tokens = [
            token if token != period_token_id else comma_token_id for token in line
        ] + [period_token_id]
        tokens = torch.tensor(tokens, dtype=torch.long)
        num_tokens = tokens.shape[0]

        gt_frame_start = int(time_start * sampling_rate / downsample_rate)

        frame_shift = random.randint(int(lyrics_shift), int(lyrics_shift))

        frame_start = max(gt_frame_start - frame_shift, last_end_pos)
        frame_len = min(num_tokens, max_frames - frame_start)

        lrc[frame_start : frame_start + frame_len] = tokens[:frame_len]

        tokens_count += num_tokens
        last_end_pos = frame_start + frame_len

    lrc_emb = lrc.unsqueeze(0).to(device)

    normalized_start_time = torch.tensor(normalized_start_time).unsqueeze(0).to(device)
    normalized_start_time = normalized_start_time.half()

    return lrc_emb, normalized_start_time


def load_checkpoint(model, ckpt_path, device, use_ema=True):
    """
    Load model checkpoint
    
    Args:
        model: Model to load weights into
        ckpt_path: Path to checkpoint file
        device: Device to load model on
        use_ema: Whether to use EMA weights
        
    Returns:
        Model with loaded weights
    """
    # Ensure model is already in half-precision to match original weights
    model = model.half()

    ckpt_type = ckpt_path.split(".")[-1]
    if ckpt_type == "safetensors":
        from safetensors.torch import load_file
        checkpoint = load_file(ckpt_path)
    else:
        # Load with weights_only to avoid mismatch issues
        checkpoint = torch.load(ckpt_path, weights_only=True, map_location=device)

    if use_ema:
        if ckpt_type == "safetensors":
            checkpoint = {"ema_model_state_dict": checkpoint}
        checkpoint["model_state_dict"] = {
            k.replace("ema_model.", ""): v
            for k, v in checkpoint["ema_model_state_dict"].items()
            if k not in ["initted", "step"]
        }
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        if ckpt_type == "safetensors":
            checkpoint = {"model_state_dict": checkpoint}
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    return model.to(device) 