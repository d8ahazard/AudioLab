"""
Expand RVC v2 pretrained weights to v3 architecture.

This utility loads v2 generator and discriminator weights and adapts them
to the v3 architecture by mapping compatible layers and initializing new ones.
"""

import logging
import os
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

from handlers.config import model_path
from modules.rvc_v3.models.generator import RVCV3Generator
from modules.rvc_v3.models.text_encoder import TextEncoder
from modules.rvc.lib.discriminator import MultiPeriodDiscriminator

logger = logging.getLogger(__name__)


def load_v2_checkpoint(checkpoint_path: str) -> Dict:
    """
    Load v2 checkpoint.
    
    Args:
        checkpoint_path: Path to v2 .pth file
        
    Returns:
        Checkpoint dictionary
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"V2 checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading v2 checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # V2 checkpoints have structure: {'model': state_dict, 'iteration': ..., 'learning_rate': ...}
    if 'model' in checkpoint:
        return checkpoint
    else:
        # If no 'model' key, assume the whole thing is the state dict
        return {'model': checkpoint}


def expand_generator_weights(
    v2_checkpoint: Dict,
    v3_generator: RVCV3Generator,
    v3_text_encoder: TextEncoder
) -> Tuple[Dict, Dict]:
    """
    Expand v2 generator weights to v3 architecture.
    
    Args:
        v2_checkpoint: V2 checkpoint dictionary
        v3_generator: Initialized v3 generator model
        v3_text_encoder: Initialized v3 text encoder model
        
    Returns:
        Tuple of (v3_generator_state_dict, v3_text_encoder_state_dict)
    """
    v2_state = v2_checkpoint['model']
    v3_gen_state = v3_generator.state_dict()
    v3_text_state = v3_text_encoder.state_dict()
    
    # Track what we copy vs initialize
    copied_layers = []
    initialized_layers = []
    
    # Mapping from v2 to v3 generator layers
    # V2 structure: emb_g, enc_p, enc_q, flow, dec
    # V3 structure: (in generator) emb_g, enc_q, flow, dec, enc_p (new with cross-attn)
    
    logger.info("Mapping v2 weights to v3 architecture...")
    
    # 1. Speaker embedding (emb_g) - direct copy
    for k in v3_gen_state.keys():
        if k.startswith('emb_g.'):
            if k in v2_state:
                v3_gen_state[k] = v2_state[k]
                copied_layers.append(k)
            else:
                initialized_layers.append(k)
    
    # 2. Posterior encoder (enc_q) - should be identical, direct copy
    for k in v3_gen_state.keys():
        if k.startswith('enc_q.'):
            if k in v2_state:
                v3_gen_state[k] = v2_state[k]
                copied_layers.append(k)
            else:
                initialized_layers.append(k)
    
    # 3. Normalizing flow - should be identical, direct copy
    for k in v3_gen_state.keys():
        if k.startswith('flow.'):
            if k in v2_state:
                v3_gen_state[k] = v2_state[k]
                copied_layers.append(k)
            else:
                initialized_layers.append(k)
    
    # 4. Decoder/Vocoder (dec) - mostly compatible, attempt copy
    for k in v3_gen_state.keys():
        if k.startswith('dec.'):
            if k in v2_state:
                # Check shape compatibility
                if v3_gen_state[k].shape == v2_state[k].shape:
                    v3_gen_state[k] = v2_state[k]
                    copied_layers.append(k)
                else:
                    logger.warning(f"Shape mismatch for {k}: v2={v2_state[k].shape}, v3={v3_gen_state[k].shape}")
                    initialized_layers.append(k)
            else:
                initialized_layers.append(k)
    
    # 5. Prior encoder (enc_p) - V3 has new architecture with cross-attention
    # Try to copy base layers from v2's enc_p, but new cross-attention layers stay random
    v2_enc_p_prefix = 'enc_p.'
    v3_enc_p_prefix = 'enc_p.'
    
    for k in v3_gen_state.keys():
        if k.startswith(v3_enc_p_prefix):
            # Try to find corresponding v2 layer
            v2_key = k  # Same key structure initially
            
            if v2_key in v2_state:
                # Check for shape compatibility
                if v3_gen_state[k].shape == v2_state[v2_key].shape:
                    v3_gen_state[k] = v2_state[v2_key]
                    copied_layers.append(k)
                else:
                    # Shape mismatch - this is expected for new v3 layers
                    initialized_layers.append(k)
            else:
                # New v3 layer (e.g., cross-attention)
                initialized_layers.append(k)
    
    # Text encoder is entirely new in v3, so all layers stay randomly initialized
    for k in v3_text_state.keys():
        initialized_layers.append(f"text_encoder.{k}")
    
    logger.info(f"Weight mapping complete:")
    logger.info(f"  Copied from v2: {len(copied_layers)} layers")
    logger.info(f"  Randomly initialized: {len(initialized_layers)} layers")
    logger.info(f"  Copy ratio: {len(copied_layers) / (len(copied_layers) + len(initialized_layers)) * 100:.1f}%")
    
    return v3_gen_state, v3_text_state


def expand_discriminator_weights(
    v2_checkpoint: Dict,
    v3_discriminator: MultiPeriodDiscriminator
) -> Dict:
    """
    Expand v2 discriminator weights to v3.
    
    V2 and V3 use the same MultiPeriodDiscriminator, so this is a direct copy.
    
    Args:
        v2_checkpoint: V2 checkpoint dictionary
        v3_discriminator: Initialized v3 discriminator model
        
    Returns:
        V3 discriminator state dict
    """
    v2_state = v2_checkpoint['model']
    v3_disc_state = v3_discriminator.state_dict()
    
    logger.info("Mapping v2 discriminator weights to v3...")
    
    copied = 0
    initialized = 0
    
    for k in v3_disc_state.keys():
        if k in v2_state:
            if v3_disc_state[k].shape == v2_state[k].shape:
                v3_disc_state[k] = v2_state[k]
                copied += 1
            else:
                logger.warning(f"Shape mismatch for discriminator {k}")
                initialized += 1
        else:
            initialized += 1
    
    logger.info(f"Discriminator mapping: copied {copied}, initialized {initialized}")
    
    return v3_disc_state


def expand_v2_to_v3(
    v2_g_path: str,
    v2_d_path: str,
    output_dir: str,
    sample_rate: int = 48000,
    if_f0: bool = True
) -> Tuple[str, str]:
    """
    Expand v2 pretrained weights to v3 architecture.
    
    Args:
        v2_g_path: Path to v2 generator checkpoint
        v2_d_path: Path to v2 discriminator checkpoint
        output_dir: Directory to save expanded v3 weights
        sample_rate: Sample rate for the model
        if_f0: Whether model uses F0
        
    Returns:
        Tuple of (v3_generator_path, v3_discriminator_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load v2 checkpoints
    v2_g_checkpoint = load_v2_checkpoint(v2_g_path)
    v2_d_checkpoint = load_v2_checkpoint(v2_d_path)
    
    # Load v3 config for 48k (as reference)
    from modules.rvc_v3.configs.v3_config import RVCV3Config
    from modules.rvc.configs.config import Config
    
    config_obj = Config()
    if sample_rate == 48000:
        config_key = 'v3/48k.json'
    elif sample_rate == 40000:
        config_key = 'v3/40k.json'
    else:
        config_key = 'v3/32k.json'
    
    json_config = config_obj.json_config[config_key]
    
    # Create v3 config
    v3_config = RVCV3Config(
        spec_channels=json_config['data']['n_mel_channels'],
        segment_size=json_config['train']['segment_size'],
        inter_channels=json_config['model']['inter_channels'],
        hidden_channels=json_config['model']['hidden_channels'],
        filter_channels=json_config['model']['filter_channels'],
        n_heads=json_config['model']['n_heads'],
        n_layers=json_config['model']['n_layers'],
        kernel_size=json_config['model']['kernel_size'],
        p_dropout=json_config['model']['p_dropout'],
        resblock=json_config['model']['resblock'],
        resblock_kernel_sizes=json_config['model']['resblock_kernel_sizes'],
        resblock_dilation_sizes=json_config['model']['resblock_dilation_sizes'],
        upsample_rates=json_config['model']['upsample_rates'],
        upsample_initial_channel=json_config['model']['upsample_initial_channel'],
        upsample_kernel_sizes=json_config['model']['upsample_kernel_sizes'],
        gin_channels=json_config['model']['gin_channels'],
        spk_embed_dim=json_config['model']['spk_embed_dim'],
        sampling_rate=json_config['data']['sampling_rate'],
        vocoder_type='bigvgan',  # Default to bigvgan for v3
        use_dual_encoder=False,  # Single HuBERT for now
        hubert_dim=768,
    )
    
    # Initialize v3 models
    logger.info("Initializing v3 models...")
    
    # Text encoder
    v3_text_encoder = TextEncoder(
        vocab_size=200,
        d_model=v3_config.text_encoder_dim,
        nhead=v3_config.text_encoder_heads,
        num_layers=v3_config.text_encoder_layers,
        dim_feedforward=v3_config.text_encoder_ff_dim,
        dropout=v3_config.text_dropout
    )
    
    # Generator
    v3_generator = RVCV3Generator(
        spec_channels=v3_config.spec_channels,
        segment_size=v3_config.segment_size,
        inter_channels=v3_config.inter_channels,
        hidden_channels=v3_config.hidden_channels,
        filter_channels=v3_config.filter_channels,
        n_heads=v3_config.n_heads,
        n_layers=v3_config.n_layers,
        kernel_size=v3_config.kernel_size,
        p_dropout=v3_config.p_dropout,
        resblock=v3_config.resblock,
        resblock_kernel_sizes=v3_config.resblock_kernel_sizes,
        resblock_dilation_sizes=v3_config.resblock_dilation_sizes,
        upsample_rates=v3_config.upsample_rates,
        upsample_initial_channel=v3_config.upsample_initial_channel,
        upsample_kernel_sizes=v3_config.upsample_kernel_sizes,
        spk_embed_dim=v3_config.spk_embed_dim,
        gin_channels=v3_config.gin_channels,
        sr=v3_config.sampling_rate,
        vocoder_type=v3_config.vocoder_type,
        text_encoder_dim=v3_config.text_encoder_dim,
        n_cross_attn_layers=v3_config.n_cross_attn_layers,
        ppg_dim=v3_config.hubert_dim
    )
    
    # Discriminator
    v3_discriminator = MultiPeriodDiscriminator()
    
    # Expand weights
    v3_gen_state, v3_text_state = expand_generator_weights(
        v2_g_checkpoint, v3_generator, v3_text_encoder
    )
    v3_disc_state = expand_discriminator_weights(
        v2_d_checkpoint, v3_discriminator
    )
    
    # Combine generator and text encoder into single checkpoint
    # (V3 trains them together)
    v3_gen_checkpoint = {
        'model': v3_gen_state,
        'text_encoder': v3_text_state,
        'iteration': v2_g_checkpoint.get('iteration', 0),
        'learning_rate': v2_g_checkpoint.get('learning_rate', 0.0001),
        'version': 'v3',
        'config': v3_config.__dict__
    }
    
    v3_disc_checkpoint = {
        'model': v3_disc_state,
        'iteration': v2_d_checkpoint.get('iteration', 0),
        'learning_rate': v2_d_checkpoint.get('learning_rate', 0.0001),
        'version': 'v3'
    }
    
    # Save expanded weights
    f0_str = 'f0' if if_f0 else ''
    sr_str = f"{sample_rate // 1000}k"
    
    v3_g_path = os.path.join(output_dir, f"{f0_str}G{sr_str}.pth")
    v3_d_path = os.path.join(output_dir, f"{f0_str}D{sr_str}.pth")
    
    logger.info(f"Saving expanded v3 generator to {v3_g_path}")
    torch.save(v3_gen_checkpoint, v3_g_path)
    
    logger.info(f"Saving expanded v3 discriminator to {v3_d_path}")
    torch.save(v3_disc_checkpoint, v3_d_path)
    
    logger.info("Weight expansion complete!")
    
    return v3_g_path, v3_d_path


def expand_all_v2_pretrains():
    """
    Expand all v2 pretrained models to v3.
    
    Looks for v2 pretrains in models/rvc/pretrained_v2/
    and creates v3 versions in models/rvc/pretrained_v3/
    """
    v2_pretrain_dir = os.path.join(model_path, "rvc", "pretrained_v2")
    v3_pretrain_dir = os.path.join(model_path, "rvc", "pretrained_v3")
    
    if not os.path.exists(v2_pretrain_dir):
        logger.warning(f"V2 pretrain directory not found: {v2_pretrain_dir}")
        logger.warning("Skipping automatic expansion. Please ensure v2 pretrained models are available.")
        return
    
    logger.info(f"Expanding v2 pretrains from {v2_pretrain_dir} to {v3_pretrain_dir}")
    
    # Find all v2 pretrained models
    sample_rates = [40000, 48000]
    f0_options = [True, False]
    
    for sr in sample_rates:
        for if_f0 in f0_options:
            f0_str = 'f0' if if_f0 else ''
            sr_str = f"{sr // 1000}k"
            
            v2_g_path = os.path.join(v2_pretrain_dir, f"{f0_str}G{sr_str}.pth")
            v2_d_path = os.path.join(v2_pretrain_dir, f"{f0_str}D{sr_str}.pth")
            
            if os.path.exists(v2_g_path) and os.path.exists(v2_d_path):
                logger.info(f"Expanding {f0_str}{sr_str} model...")
                try:
                    expand_v2_to_v3(
                        v2_g_path,
                        v2_d_path,
                        v3_pretrain_dir,
                        sample_rate=sr,
                        if_f0=if_f0
                    )
                except Exception as e:
                    logger.error(f"Failed to expand {f0_str}{sr_str}: {e}")
            else:
                logger.info(f"Skipping {f0_str}{sr_str} (not found)")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Expand all v2 pretrains
    expand_all_v2_pretrains()

