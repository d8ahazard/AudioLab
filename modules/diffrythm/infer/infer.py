"""
DiffRhythm inference functions

This implementation is adapted from the original DiffRhythm project:
https://github.com/ASLP-lab/DiffRhythm
"""

import torch
from einops import rearrange

from modules.diffrythm.infer.infer_utils import decode_audio


def inference(
    cfm_model,
    vae_model,
    cond,
    text,
    duration,
    style_prompt,
    negative_style_prompt,
    start_time,
    chunked=False,
):
    """
    Generate audio using DiffRhythm model
    
    Args:
        cfm_model: The Conditional Flow Matching model
        vae_model: The VAE model for audio decoding
        cond: Conditioning tensor
        text: Text tokens
        duration: Duration of audio to generate
        style_prompt: Style prompt tensor
        negative_style_prompt: Negative style prompt tensor
        start_time: Start time tensor
        chunked: Whether to use chunked decoding (for memory efficiency)
        
    Returns:
        Tensor: Generated audio
    """
    with torch.inference_mode():
        # Ensure all inputs have matching dtypes before generation
        device = cfm_model.device
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        # Make sure these tensors match the model's dtype
        cond = cond.to(dtype=dtype)
        text = text.to(device)  # Keep text as LongTensor
        style_prompt = style_prompt.to(dtype=dtype)
        negative_style_prompt = negative_style_prompt.to(dtype=dtype)
        start_time = start_time.to(dtype=dtype)
        
        # Generate the latents
        generated, _ = cfm_model.sample(
            cond=cond,
            text=text,
            duration=duration,
            style_prompt=style_prompt,
            negative_style_prompt=negative_style_prompt,
            steps=32,
            cfg_strength=4.0,
            start_time=start_time,
        )

        # Convert to float32 for VAE decoding
        generated = generated.to(torch.float32)
        latent = generated.transpose(1, 2)  # [b d t]

        # Ensure VAE is in appropriate dtype
        with torch.amp.autocast('cuda' if device == "cuda" else 'cpu'):
            output = decode_audio(latent, vae_model, chunked=chunked)

        # Rearrange audio batch to a single sequence
        output = rearrange(output, "b d n -> d (b n)")
        
        # Peak normalize, clip, convert to int16, and save to file
        output = (
            output.to(torch.float32)  # Ensure float32 for audio processing
            .div(torch.max(torch.abs(output)))
            .clamp(-1, 1)
            .mul(32767)
            .to(torch.int16)
            .cpu()
        )

        return output 