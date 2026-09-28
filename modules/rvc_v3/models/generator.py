"""
RVC V3 Generator with cross-attention to text.

Extended generator that conditions on both audio content and text/lyric features.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import base modules from RVC v2
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from modules.rvc.lib.models import TextEncoder as RVCTextEncoder, PosteriorEncoder, ResidualCouplingBlock

logger = logging.getLogger(__name__)


class CrossAttentionLayer(nn.Module):
    """
    Cross-attention layer for attending text features from content features.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Multi-head cross-attention
        self.cross_attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Cross-attend from query (content) to key_value (text).
        
        Args:
            query: Content features (batch, seq_len_q, d_model)
            key_value: Text features (batch, seq_len_kv, d_model)
            key_padding_mask: Mask for text padding (batch, seq_len_kv)
        
        Returns:
            Enhanced features (batch, seq_len_q, d_model)
        """
        # Cross-attention
        attn_out, _ = self.cross_attn(
            query, key_value, key_value,
            key_padding_mask=key_padding_mask
        )
        
        # Residual and norm
        query = self.norm1(query + attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(query)
        query = self.norm2(query + ffn_out)
        
        return query


class TextConditionedTextEncoder(nn.Module):
    """
    Extended TextEncoder with cross-attention to lyric text.
    
    Wraps the RVC v2 TextEncoder and adds cross-attention layers.
    """
    
    def __init__(
        self,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        text_d_model: int = 256,
        n_cross_attn_layers: int = 2,
        ppg_dim: Optional[int] = None,
        backbone: str = 'legacy',
    ):
        super().__init__()
        
        # Base RVC TextEncoder
        self.backbone = backbone
        if backbone == 'v2_compatible':
            from modules.rvc.infer.lib.infer_pack.models import TextEncoder as CompatibleEncoder
            self.base_encoder = CompatibleEncoder(768, inter_channels, hidden_channels,
                filter_channels, n_heads, n_layers, kernel_size, p_dropout)
        else:
            self.base_encoder = RVCTextEncoder(
                inter_channels, hidden_channels, filter_channels, n_heads,
                n_layers, kernel_size, p_dropout, ppg_dim=ppg_dim,
            )
        
        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionLayer(
                inter_channels,
                n_heads=n_heads,
                dropout=p_dropout
            )
            for _ in range(n_cross_attn_layers)
        ])
        
        # Project text features to hidden_channels if needed
        if text_d_model != inter_channels:
            self.text_projection = nn.Linear(text_d_model, inter_channels)
        else:
            self.text_projection = None
        
        logger.info(
            f"TextConditionedTextEncoder: {n_cross_attn_layers} cross-attention layers"
        )
    
    def forward(
        self,
        phone: torch.Tensor,
        pitch: torch.Tensor,
        lengths: torch.Tensor,
        text_features: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        ppg: Optional[torch.Tensor] = None,
        text_strength: float = 1.0,
    ):
        """
        Forward pass with optional text conditioning.
        
        Args:
            phone: Content features (batch, feature_dim, time)
            pitch: Pitch features (batch, 1, time)
            lengths: Sequence lengths
            text_features: Text encoder output (batch, text_len, text_d_model)
            text_mask: Text padding mask (batch, text_len)
            ppg: Optional PPG features
        
        Returns:
            Encoded features, log scales, and mask
        """
        # Base encoding
        if self.backbone == 'v2_compatible':
            if ppg is not None:
                raise ValueError('PPG conditioning is not supported by the V2-compatible backbone')
            m_p, logs_p, x_mask = self.base_encoder(phone, pitch, lengths)
        else:
            m_p, logs_p, x_mask = self.base_encoder(phone, pitch, lengths, ppg=ppg)
        
        # If text features provided, apply cross-attention with controllable strength.
        if text_features is not None and text_strength > 0.0:
            valid_rows = (torch.ones(phone.size(0), dtype=torch.bool, device=phone.device)
                          if text_mask is None else ~text_mask.all(dim=1))
            if not valid_rows.any():
                return m_p, logs_p, x_mask
            text_features = text_features[valid_rows]
            if text_mask is not None:
                text_mask = text_mask[valid_rows]
            # Project text if needed
            if self.text_projection is not None:
                text_features = self.text_projection(text_features)
            
            # Transpose for cross-attention (batch, time, channels)
            content = m_p[valid_rows].transpose(1, 2)
            
            # Apply cross-attention layers
            for layer in self.cross_attn_layers:
                content = layer(content, text_features, text_mask)
            
            # Transpose back and blend with base encoding.
            m_p_text = content.transpose(1, 2)  # (batch, channels, time)
            ts = float(max(0.0, min(1.0, text_strength)))
            m_p = m_p.clone()
            m_p[valid_rows] = (1.0 - ts) * m_p[valid_rows] + ts * m_p_text
            m_p = m_p * x_mask
        
        return m_p, logs_p, x_mask


class RVCV3Generator(nn.Module):
    """
    RVC V3 Generator with text conditioning.
    
    Extends RVC v2 model with cross-attention to text/lyric features.
    """
    
    def __init__(
        self,
        spec_channels: int,
        segment_size: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        resblock: str,
        resblock_kernel_sizes: list,
        resblock_dilation_sizes: list,
        upsample_rates: list,
        upsample_initial_channel: int,
        upsample_kernel_sizes: list,
        spk_embed_dim: int,
        gin_channels: int,
        sr: int,
        vocoder_type: str = 'hifigan',
        text_encoder_dim: int = 256,
        n_cross_attn_layers: int = 2,
        ppg_dim: Optional[int] = None,
        backbone: str = 'legacy',
        **kwargs
    ):
        """
        Initialize RVC V3 generator.
        
        Most parameters are same as RVC v2, with additions for text conditioning.
        
        Args:
            text_encoder_dim: Dimension of text encoder output
            n_cross_attn_layers: Number of cross-attention layers
        """
        super().__init__()
        
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        self.spk_embed_dim = spk_embed_dim
        
        # Speaker embedding
        self.emb_g = nn.Embedding(spk_embed_dim, gin_channels)
        
        # Text-conditioned encoder
        self.enc_p = TextConditionedTextEncoder(
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            text_d_model=text_encoder_dim,
            n_cross_attn_layers=n_cross_attn_layers,
            ppg_dim=ppg_dim,
            backbone=backbone,
        )
        
        # Posterior encoder (unchanged from v2)
        posterior_cls, flow_cls = PosteriorEncoder, ResidualCouplingBlock
        if backbone == 'v2_compatible':
            from modules.rvc.infer.lib.infer_pack.models import (
                PosteriorEncoder as posterior_cls, ResidualCouplingBlock as flow_cls)
        self.enc_q = posterior_cls(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels
        )
        
        # Normalizing flow (unchanged from v2)
        self.flow = flow_cls(
            inter_channels, hidden_channels, 5, 1, 3, gin_channels=gin_channels
        )
        
        # Decoder (vocoder) - will be replaced with stereo version
        # For now, use the standard decoder
        from modules.rvc.lib.models import GeneratorNSF, GeneratorBigVgan
        if backbone == 'v2_compatible':
            from modules.rvc.infer.lib.infer_pack.models import GeneratorNSF
        
        if vocoder_type == 'hifigan':
            self.dec = GeneratorNSF(
                inter_channels,
                resblock,
                resblock_kernel_sizes,
                resblock_dilation_sizes,
                upsample_rates,
                upsample_initial_channel,
                upsample_kernel_sizes,
                gin_channels=gin_channels,
                sr=sr,
                is_half=False
            )
        elif vocoder_type == 'bigvgan':
            # Match our bundled BigVGAN wrapper signature.
            # See: modules/rvc/lib/models_bigvgan.py::GeneratorBigVgan
            self.dec = GeneratorBigVgan(
                resblock_kernel_sizes=resblock_kernel_sizes,
                resblock_dilation_sizes=resblock_dilation_sizes,
                upsample_rates=upsample_rates,
                upsample_kernel_sizes=upsample_kernel_sizes,
                upsample_input=inter_channels,
                upsample_initial_channel=upsample_initial_channel,
                sampling_rate=sr,
                spk_dim=gin_channels,
            )
        else:
            raise ValueError(f"Unknown vocoder type: {vocoder_type}")
        
        logger.info("RVCV3Generator initialized with text conditioning")
    
    def forward(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: torch.Tensor,
        pitchf: torch.Tensor,
        y: torch.Tensor,
        y_lengths: torch.Tensor,
        ds: torch.Tensor,
        text_features: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        ppg: Optional[torch.Tensor] = None,
        enable_perturbation: bool = False,
        text_strength: float = 1.0,
    ):
        """
        Forward pass for training.
        
        Args:
            phone: Content features (batch, feature_dim, time)
            phone_lengths: Lengths of content sequences
            pitch: Coarse pitch (batch, time)
            pitchf: Fine pitch (batch, time)
            y: Target mel spectrogram (batch, mel_bins, time)
            y_lengths: Lengths of mel spectrograms
            ds: Speaker IDs
            text_features: Text encoder output (batch, text_len, text_dim)
            text_mask: Text padding mask (batch, text_len)
            ppg: Optional PPG features
            enable_perturbation: Whether to add perturbation for regularization
        """
        if enable_perturbation:
            if ppg is not None:
                ppg = ppg + torch.randn_like(ppg) * 1
            phone = phone + torch.randn_like(phone) * 2
        
        # Speaker embedding
        g = self.emb_g(ds).unsqueeze(-1)  # (batch, gin_channels, 1)
        
        # Encode with text conditioning
        m_p, logs_p, x_mask = self.enc_p(
            phone, pitch, phone_lengths,
            text_features=text_features,
            text_mask=text_mask,
            ppg=ppg,
            text_strength=text_strength,
        )
        
        # Posterior encoding
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        
        # Flow
        z_p = self.flow(z, y_mask, g=g)
        
        # Random segment slicing for training
        from modules.rvc.lib.models import rand_slice_segments, slice_segments2
        
        z_slice, ids_slice = rand_slice_segments(z, y_lengths, self.segment_size)
        pitchf = slice_segments2(pitchf, ids_slice, self.segment_size)
        
        # Decode
        o = self.dec(z_slice, pitchf, g=g)
        
        return o, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)
    
    def infer(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: torch.Tensor,
        nsff0: torch.Tensor,
        sid: torch.Tensor,
        text_features: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        rate: Optional[torch.Tensor] = None,
        ppg: Optional[torch.Tensor] = None,
        text_strength: float = 1.0,
        noise_scale: float = 0.0,
    ):
        """
        Inference with optional text conditioning.
        
        Args:
            phone: Content features
            phone_lengths: Lengths
            pitch: Coarse pitch
            nsff0: Fine pitch for NSF
            sid: Speaker ID
            text_features: Text encoder output
            text_mask: Text padding mask
            rate: Optional rate adjustment
            ppg: Optional PPG features
        """
        # Speaker embedding
        g = self.emb_g(sid).unsqueeze(-1)
        
        # Encode with text conditioning
        m_p, logs_p, x_mask = self.enc_p(
            phone, pitch, phone_lengths,
            text_features=text_features,
            text_mask=text_mask,
            ppg=ppg,
            text_strength=text_strength,
        )
        
        # Sample from latent distribution.
        # V3 can produce modulating buzz in silence when stochastic sampling is high.
        # Keep this controllable and default to deterministic inference.
        ns = float(max(0.0, noise_scale))
        if ns > 0.0:
            z_noise = torch.randn_like(m_p) * ns
            z_p = (m_p + torch.exp(logs_p) * z_noise) * x_mask
        else:
            z_p = m_p * x_mask
        
        # Rate adjustment if provided
        if rate is not None:
            head = int(z_p.shape[2] * (1.0 - rate.item()))
            z_p = z_p[:, :, head:]
            x_mask = x_mask[:, :, head:]
            nsff0 = nsff0[:, head:]
        
        # Flow (reverse)
        z = self.flow(z_p, x_mask, g=g, reverse=True)
        
        # Decode
        o = self.dec(z * x_mask, nsff0, g=g)
        
        return o, x_mask, (z, z_p, m_p, logs_p)

