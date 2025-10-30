"""
RVC V3 Inference Pipeline.

Complete pipeline for voice conversion with text conditioning.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple, Dict

import torch
import numpy as np
import librosa
import soundfile as sf

from modules.rvc_v3.models.generator import RVCV3Generator
from modules.rvc_v3.models.text_encoder import TextEncoder
from modules.rvc_v3.models.content_encoders import DualContentEncoder, HuBERTEncoder
from modules.rvc_v3.models.retrieval import RetrievalIndex
from modules.rvc_v3.data_prep.phonemizer import Phonemizer
from handlers.config import model_path

logger = logging.getLogger(__name__)


class RVCV3Pipeline:
    """
    Complete inference pipeline for RVC V3.
    
    Handles:
    - Content feature extraction
    - Pitch extraction
    - Text tokenization and encoding
    - Feature retrieval and mixing
    - Audio generation
    """
    
    def __init__(
        self,
        model_path_or_checkpoint: str,
        config,
        device: str = "cuda"
    ):
        """
        Initialize inference pipeline.
        
        Args:
            model_path_or_checkpoint: Path to model checkpoint
            config: RVCV3Config
            device: Device to run on
        """
        self.config = config
        self.device = device
        
        # Initialize content encoder
        self._init_content_encoder()
        
        # Initialize pitch extractor
        self._init_pitch_extractor()
        
        # Initialize phonemizer
        self.phonemizer = Phonemizer(language='en-us')
        
        # Initialize text encoder
        self._init_text_encoder()
        
        # Initialize generator
        self._init_generator()
        
        # Load model weights
        self.load_model(model_path_or_checkpoint)
        
        # Retrieval index (loaded separately)
        self.retrieval_index = None
        
        logger.info("RVCV3Pipeline initialized")
    
    def _init_content_encoder(self):
        """Initialize content encoder."""
        hubert_path = os.path.join(model_path, "rvc", "hubert_base.pt")
        
        if self.config.use_dual_encoder:
            self.content_encoder = DualContentEncoder(
                hubert_path=hubert_path,
                whisper_model=self.config.whisper_model,
                fusion_method=self.config.fusion_method,
                output_dim=self.config.content_output_dim,
                device=self.device,
                is_half=False
            )
        else:
            self.content_encoder = HuBERTEncoder(
                model_path=hubert_path,
                device=self.device,
                is_half=False
            )
    
    def _init_pitch_extractor(self):
        """Initialize pitch extractor."""
        from modules.rvc.infer.lib.rmvpe import RMVPE
        
        rmvpe_path = os.path.join(model_path, "rvc", "rmvpe.pt")
        self.pitch_extractor = RMVPE(rmvpe_path, is_half=False, device=self.device)
    
    def _init_text_encoder(self):
        """Initialize text encoder."""
        self.text_encoder = TextEncoder(
            vocab_size=200,  # Will be updated from phonemizer vocab
            d_model=self.config.text_encoder_dim,
            nhead=self.config.text_encoder_heads,
            num_layers=self.config.text_encoder_layers,
            dim_feedforward=self.config.text_encoder_ff_dim,
            dropout=self.config.text_dropout
        ).to(self.device)
        self.text_encoder.eval()
    
    def _init_generator(self):
        """Initialize generator."""
        self.generator = RVCV3Generator(
            spec_channels=self.config.spec_channels,
            segment_size=self.config.segment_size,
            inter_channels=self.config.inter_channels,
            hidden_channels=self.config.hidden_channels,
            filter_channels=self.config.filter_channels,
            n_heads=self.config.n_heads,
            n_layers=self.config.n_layers,
            kernel_size=self.config.kernel_size,
            p_dropout=self.config.p_dropout,
            resblock=self.config.resblock,
            resblock_kernel_sizes=self.config.resblock_kernel_sizes,
            resblock_dilation_sizes=self.config.resblock_dilation_sizes,
            upsample_rates=self.config.upsample_rates,
            upsample_initial_channel=self.config.upsample_initial_channel,
            upsample_kernel_sizes=self.config.upsample_kernel_sizes,
            spk_embed_dim=self.config.spk_embed_dim,
            gin_channels=self.config.gin_channels,
            sr=self.config.sampling_rate,
            vocoder_type=self.config.vocoder_type,
            text_encoder_dim=self.config.text_encoder_dim,
            n_cross_attn_layers=self.config.n_cross_attn_layers,
            ppg_dim=self.config.get_content_feature_dim()
        ).to(self.device)
        self.generator.eval()
    
    def load_model(self, checkpoint_path: str):
        """Load model weights from checkpoint."""
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator'])
        self.text_encoder.load_state_dict(checkpoint['text_encoder'])
        
        logger.info(f"Model loaded from {checkpoint_path}")
    
    def load_retrieval_index(self, index_path: str):
        """Load retrieval index."""
        feature_dim = self.config.get_content_feature_dim()
        
        self.retrieval_index = RetrievalIndex(
            feature_dim=feature_dim,
            index_type="IVF",
            use_gpu=True
        )
        
        self.retrieval_index.load(index_path)
        logger.info(f"Retrieval index loaded from {index_path}")
    
    @torch.no_grad()
    def convert(
        self,
        audio_path: str,
        lyrics: Optional[str] = None,
        output_path: Optional[str] = None,
        index_rate: float = 0.75,
        pitch_shift: int = 0,
        speaker_id: int = 0
    ) -> Tuple[np.ndarray, int]:
        """
        Convert audio to target voice.
        
        Args:
            audio_path: Path to input audio
            lyrics: Optional lyrics text with tags
            output_path: Optional path to save output
            index_rate: Retrieval mixing ratio (0-1)
            pitch_shift: Pitch shift in semitones
            speaker_id: Target speaker ID
        
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        # Load and preprocess audio
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(self.device)
        
        # Extract content features
        content_features = self.content_encoder.extract_features(audio_tensor)
        content_features = content_features.squeeze(0)  # (T, D)
        
        # Extract pitch
        f0 = self.pitch_extractor.infer_from_audio(audio, thred=0.03)
        
        # Apply pitch shift if requested
        if pitch_shift != 0:
            f0 = f0 * (2 ** (pitch_shift / 12.0))
        
        # Convert to tensors
        content_np = content_features.cpu().numpy()
        
        # Retrieval mixing
        if self.retrieval_index is not None and index_rate > 0:
            content_mixed = self.retrieval_index.mix_features(
                content_np,
                alpha=1.0 - index_rate,  # Convert to retrieval weight
                k=1
            )
            content_features = torch.FloatTensor(content_mixed).to(self.device)
        
        # Text encoding
        text_features = None
        text_mask = None
        
        if lyrics:
            # Phonemize lyrics
            phonemes = self.phonemizer.phonemize(lyrics, preserve_tags=True)
            token_ids = self.phonemizer.encode(phonemes)
            
            # Convert to tensors
            text_tokens = torch.LongTensor(token_ids).unsqueeze(0).to(self.device)
            text_mask = torch.zeros_like(text_tokens, dtype=torch.bool)
            
            # Encode text
            text_features = self.text_encoder(text_tokens, text_mask)
        
        # Prepare inputs for generator
        content_features = content_features.unsqueeze(0).transpose(1, 2)  # (1, D, T)
        pitch_tensor = torch.FloatTensor(f0).unsqueeze(0).to(self.device)  # (1, T)
        
        # Create lengths tensor
        content_len = content_features.shape[2]
        lengths = torch.LongTensor([content_len]).to(self.device)
        
        # Create speaker ID tensor
        sid = torch.LongTensor([speaker_id]).to(self.device)
        
        # Generate audio
        audio_out, _, _ = self.generator.infer(
            phone=content_features,
            phone_lengths=lengths,
            pitch=pitch_tensor[:, :content_len].long(),  # Coarse pitch
            nsff0=pitch_tensor[:, :content_len],  # Fine pitch
            sid=sid,
            text_features=text_features,
            text_mask=text_mask
        )
        
        # Convert to numpy
        audio_out = audio_out.squeeze().cpu().numpy()
        
        # Save if output path provided
        if output_path:
            sf.write(output_path, audio_out, self.config.sampling_rate)
            logger.info(f"Output saved to {output_path}")
        
        return audio_out, self.config.sampling_rate

