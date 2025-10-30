"""
Text/lyric encoder for RVC V3.

Transformer-based encoder for phoneme sequences with style tags.
"""

import logging
import math
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        
        Returns:
            x with positional encoding added
        """
        x = x + self.pe[:x.size(1), :]
        return x


class TextEncoder(nn.Module):
    """
    Text encoder for phoneme sequences with style tags.
    
    Uses Transformer encoder to produce context-enriched embeddings.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 5000
    ):
        """
        Initialize text encoder.
        
        Args:
            vocab_size: Size of phoneme/tag vocabulary
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feed-forward dimension
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.dropout = nn.Dropout(dropout)
        
        logger.info(
            f"TextEncoder initialized: vocab={vocab_size}, d_model={d_model}, "
            f"layers={num_layers}, heads={nhead}"
        )
    
    def forward(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode phoneme/tag tokens.
        
        Args:
            tokens: Token IDs (batch, seq_len)
            mask: Attention mask (batch, seq_len) - True for padding
        
        Returns:
            Encoded features (batch, seq_len, d_model)
        """
        # Embed tokens
        x = self.embedding(tokens) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        # Generate attention mask for transformer
        # PyTorch transformer expects False for valid positions
        if mask is not None:
            attn_mask = mask  # (batch, seq_len)
        else:
            attn_mask = None
        
        # Encode
        encoded = self.transformer_encoder(x, src_key_padding_mask=attn_mask)
        
        return encoded
    
    def get_output_dim(self) -> int:
        """Get output dimension."""
        return self.d_model


class BiLSTMTextEncoder(nn.Module):
    """
    Alternative text encoder using Bi-LSTM.
    
    Lighter weight alternative to Transformer encoder.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize Bi-LSTM text encoder.
        
        Args:
            vocab_size: Size of phoneme/tag vocabulary
            embedding_dim: Embedding dimension
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Bi-LSTM
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Project bidirectional output back to desired dimension
        self.output_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        logger.info(
            f"BiLSTMTextEncoder initialized: vocab={vocab_size}, "
            f"hidden={hidden_dim}, layers={num_layers}"
        )
    
    def forward(
        self,
        tokens: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode phoneme/tag tokens.
        
        Args:
            tokens: Token IDs (batch, seq_len)
            lengths: Actual lengths of sequences (batch,)
        
        Returns:
            Encoded features (batch, seq_len, hidden_dim)
        """
        # Embed tokens
        x = self.embedding(tokens)
        x = self.dropout(x)
        
        # Pack sequence if lengths provided
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        # LSTM encoding
        lstm_out, _ = self.lstm(x)
        
        # Unpack if we packed
        if lengths is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True
            )
        
        # Project to hidden_dim
        output = self.output_projection(lstm_out)
        
        return output
    
    def get_output_dim(self) -> int:
        """Get output dimension."""
        return self.hidden_dim

