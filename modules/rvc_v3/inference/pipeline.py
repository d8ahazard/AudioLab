"""
RVC V3 Inference Pipeline.

Complete pipeline for voice conversion with text conditioning.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

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
        self.last_convert_debug: Dict[str, Any] = {}
        
        logger.info("RVCV3Pipeline initialized")

    def _encode_lyrics_tokens(self, lyrics: str) -> Tuple[list[int], str]:
        """
        Encode lyrics text to token IDs.

        NOTE: Current V3 training path uses SimplePhonemizer (character-level IDs).
        For inference, default to the same tokenizer to avoid train/infer mismatch.
        """
        mode = os.environ.get("AUDIOCLONE_V3_TEXT_TOKENIZER", "simple").strip().lower()
        max_tokens = int(os.environ.get("AUDIOCLONE_V3_MAX_TEXT_TOKENS", "4096"))
        text = (lyrics or "").strip()
        if not text:
            return [], "none"

        # Match modules/rvc_v3/training/train_wrapper.py::SimplePhonemizer
        if mode in {"simple", "char", "character", "auto"}:
            vocab = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?'\"")
            char_to_id = {c: i for i, c in enumerate(vocab)}
            token_ids = [char_to_id.get(ch, 0) for ch in text]
            if max_tokens > 0:
                token_ids = token_ids[:max_tokens]
            return token_ids, "simple_char"

        # Optional explicit phonemizer path
        phonemes = self.phonemizer.phonemize(text, preserve_tags=True)
        token_ids = self.phonemizer.encode(phonemes)
        if max_tokens > 0:
            token_ids = token_ids[:max_tokens]
        return token_ids, "phonemizer"
    
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
        self.pitch_method = os.environ.get("AUDIOCLONE_V3_PITCH_METHOD", "rmvpe").strip().lower()
        if self.pitch_method == "rmvpe+":
            from modules.rvc.pitch_extraction import FeatureExtractor
            from modules.rvc.configs.config import Config as RVCConfig
            self.pitch_extractor = FeatureExtractor(self.config.sampling_rate, RVCConfig(), onnx=False)
        else:
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
        """Load model weights from .safetensors or .pth checkpoint."""
        from modules.rvc_v3.io.checkpoint_io import load_inference_model

        p = Path(checkpoint_path)
        base = p.parent / p.stem if p.suffix else Path(checkpoint_path)
        if not base.with_suffix(".safetensors").exists() and not base.with_suffix(".pth").exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        gen_sd, text_sd, _ = load_inference_model(checkpoint_path, device=self.device)
        if not gen_sd:
            raise RuntimeError(f"Invalid checkpoint (missing generator state): {checkpoint_path}")
        self.generator.load_state_dict(gen_sd)
        if text_sd:
            self.text_encoder.load_state_dict(text_sd)
        else:
            logger.warning("Checkpoint has no text_encoder state; continuing with randomly initialized text encoder")
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
        if getattr(self, "pitch_method", "rmvpe") == "rmvpe+":
            _, f0 = self.pitch_extractor.get_f0(
                audio,
                f0_up_key=0,
                f0_method="rmvpe+",
                merge_type="median",
                filter_radius=3,
                crepe_hop_length=160,
                rmvpe_onnx=False,
                f0_min=50,
                f0_max=1100,
            )
        else:
            f0 = self.pitch_extractor.infer_from_audio(audio, thred=0.03)
        f0_raw_len = int(len(f0))
        
        # Apply pitch shift if requested
        if pitch_shift != 0:
            f0 = f0 * (2 ** (pitch_shift / 12.0))
        
        # Convert to numpy for retrieval/protect path first (HuBERT timebase)
        content_np = content_features.cpu().numpy()
        content_np_base = content_np.copy()
        content_len_hubert = int(content_np.shape[0])

        # Retrieval mixing: keep conservative defaults to preserve intelligibility.
        # High index rates can over-impose target timbre and collapse consonants into humming.
        effective_rate = float(index_rate)
        effective_rate = min(effective_rate, 0.35)
        if self.retrieval_index is not None and getattr(
            self.retrieval_index, "is_v2_reconstructed", False
        ):
            effective_rate = min(effective_rate, 0.25)
        if self.retrieval_index is not None and effective_rate > 0:
            content_mixed = self.retrieval_index.mix_features(
                content_np,
                alpha=1.0 - effective_rate,  # Convert to retrieval weight
                k=1
            )
            content_np = content_mixed.astype(np.float32, copy=False)

        # RVC contract in many v2 paths uses ~100 Hz phone timeline (x2 vs HuBERT).
        # Keep this switchable per-checkpoint because some V3 runs are trained on native rate.
        phone_x2 = os.environ.get("AUDIOCLONE_V3_PHONE_X2", "1").strip() not in {"0", "false", "False"}
        if phone_x2:
            content_np = np.repeat(content_np, 2, axis=0)
            content_np_base = np.repeat(content_np_base, 2, axis=0)
        content_len_model = int(content_np.shape[0])

        # Convert f0 (Hz) to coarse pitch indices 0-255 for emb_pitch
        f0_mel = 1127.0 * np.log(1.0 + np.maximum(f0, 0.0) / 700.0)
        f0_min, f0_max = 50.0, 1100.0
        f0_mel_min = 1127.0 * np.log(1.0 + f0_min / 700.0)
        f0_mel_max = 1127.0 * np.log(1.0 + f0_max / 700.0)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254.0 / (f0_mel_max - f0_mel_min) + 1.0
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(np.int64)
        
        # Align pitch to model content length (~100 Hz after x2 upsample)
        if len(f0) != content_len_model:
            x_old = np.linspace(0, 1, len(f0))
            x_new = np.linspace(0, 1, content_len_model)
            voiced = f0 > 1.0  # Unvoiced frames
            f0_interp = np.interp(x_new, x_old, f0)
            f0_coarse_interp = np.interp(x_new, x_old, f0_coarse.astype(np.float64))
            # Preserve unvoiced: mask interpolated values where source was unvoiced
            voiced_interp = np.interp(x_new, x_old, voiced.astype(np.float64)) > 0.5
            f0 = np.where(voiced_interp, f0_interp, 0.0)
            f0_coarse = np.where(voiced_interp, np.rint(f0_coarse_interp).astype(np.int64), 1)
            f0_coarse = np.clip(f0_coarse, 1, 255)

        # V2-style protect blend: in unvoiced regions, preserve more source content features.
        # This helps reduce sustained humming artifacts after retrieval mixing.
        protect = float(os.environ.get("AUDIOCLONE_V3_PROTECT", "0.20"))
        protect = float(np.clip(protect, 0.0, 0.5))
        if self.retrieval_index is not None and effective_rate > 0.0 and protect < 0.5:
            voiced_mask = (f0 > 1.0).astype(np.float32).reshape(-1, 1)
            blend = voiced_mask + (1.0 - voiced_mask) * protect
            if content_np_base.shape[0] == content_np.shape[0]:
                content_np = content_np * blend + content_np_base * (1.0 - blend)
        
        # Text encoding
        text_features = None
        text_mask = None
        token_ids: list[int] = []
        tokenizer_used = "none"
        
        if lyrics:
            token_ids, tokenizer_used = self._encode_lyrics_tokens(lyrics)
            # Convert to tensors
            if token_ids:
                text_tokens = torch.LongTensor(token_ids).unsqueeze(0).to(self.device)
                text_mask = torch.zeros_like(text_tokens, dtype=torch.bool)
                # Encode text
                text_features = self.text_encoder(text_tokens, text_mask)

        # Keep text conditioning conservative by default; full-strength tends to over-smooth.
        if text_features is not None:
            text_strength = float(os.environ.get("AUDIOCLONE_V3_TEXT_STRENGTH", "0.35"))
        else:
            text_strength = 0.0
        text_strength = float(np.clip(text_strength, 0.0, 1.0))
        noise_scale = float(np.clip(float(os.environ.get("AUDIOCLONE_V3_NOISE_SCALE", "0.0")), 0.0, 1.0))
        
        # Prepare inputs for generator (base_encoder expects phone as B,T,768)
        content_features = torch.FloatTensor(content_np).unsqueeze(0).to(self.device)  # (1, T, 768)
        pitch_coarse = torch.LongTensor(f0_coarse).unsqueeze(0).to(self.device)  # (1, T)
        pitch_fine = torch.FloatTensor(f0).unsqueeze(0).to(self.device)  # (1, T)
        
        # Create lengths tensor
        content_len = content_features.shape[1]
        if pitch_coarse.shape[1] != content_len:
            raise RuntimeError(
                f"Pitch/content length mismatch: pitch={pitch_coarse.shape[1]} content={content_len}"
            )
        if content_features.shape[2] != 768:
            raise RuntimeError(
                f"Unexpected content feature dim: got {content_features.shape[2]}, expected 768"
            )
        lengths = torch.LongTensor([content_len]).to(self.device)
        
        # Create speaker ID tensor
        sid = torch.LongTensor([speaker_id]).to(self.device)
        
        # Generate audio (chunked on long clips to avoid long-context humming/oscillation)
        hop = int(getattr(self.config, "hop_length", 480))
        frames_per_sec = float(self.config.sampling_rate) / float(max(1, hop))
        chunk_seconds = float(os.environ.get("AUDIOCLONE_V3_CHUNK_SECONDS", "12.0"))
        overlap_seconds = float(os.environ.get("AUDIOCLONE_V3_CHUNK_OVERLAP_SECONDS", "0.30"))
        chunk_frames = max(1, int(chunk_seconds * frames_per_sec))
        overlap_frames = max(1, int(overlap_seconds * frames_per_sec))
        overlap_frames = min(overlap_frames, max(1, chunk_frames // 4))

        if content_len <= chunk_frames:
            audio_out, _, _ = self.generator.infer(
                phone=content_features,
                phone_lengths=lengths,
                pitch=pitch_coarse[:, :content_len],  # Coarse pitch 0-255
                nsff0=pitch_fine[:, :content_len],   # Fine pitch for NSF
                sid=sid,
                text_features=text_features,
                text_mask=text_mask,
                text_strength=text_strength,
                noise_scale=noise_scale,
            )
        else:
            chunks: list[np.ndarray] = []
            start = 0
            n_chunks = 0
            while start < content_len:
                end = min(content_len, start + chunk_frames)
                phone_seg = content_features[:, start:end, :]
                pitch_seg = pitch_coarse[:, start:end]
                pitchf_seg = pitch_fine[:, start:end]
                seg_len = torch.LongTensor([phone_seg.shape[1]]).to(self.device)
                # Slice text by relative position so each chunk attends local lyrics.
                seg_text_features = text_features
                seg_text_mask = text_mask
                if text_features is not None and text_features.shape[1] > 8:
                    text_len = int(text_features.shape[1])
                    margin = int(os.environ.get("AUDIOCLONE_V3_TEXT_MARGIN_TOKENS", "24"))
                    t0 = max(0, int((start / max(1, content_len)) * text_len) - margin)
                    t1 = min(text_len, int((end / max(1, content_len)) * text_len) + margin)
                    if t1 > t0:
                        seg_text_features = text_features[:, t0:t1, :]
                        seg_text_mask = text_mask[:, t0:t1] if text_mask is not None else None
                seg_audio, _, _ = self.generator.infer(
                    phone=phone_seg,
                    phone_lengths=seg_len,
                    pitch=pitch_seg,
                    nsff0=pitchf_seg,
                    sid=sid,
                    text_features=seg_text_features,
                    text_mask=seg_text_mask,
                    text_strength=text_strength,
                    noise_scale=noise_scale,
                )
                seg_np = seg_audio.squeeze().detach().cpu().numpy()
                if chunks:
                    # Crossfade overlap to suppress chunk seams.
                    fade_samples = min(overlap_frames * hop, len(chunks[-1]), len(seg_np))
                    if fade_samples > 0:
                        fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
                        fade_in = 1.0 - fade_out
                        tail = chunks[-1][-fade_samples:] * fade_out + seg_np[:fade_samples] * fade_in
                        chunks[-1] = np.concatenate([chunks[-1][:-fade_samples], tail], axis=0)
                        seg_np = seg_np[fade_samples:]
                chunks.append(seg_np)
                n_chunks += 1
                if end >= content_len:
                    break
                start = max(0, end - overlap_frames)
            audio_out = np.concatenate(chunks, axis=0) if chunks else np.zeros(0, dtype=np.float32)
            audio_out = torch.from_numpy(audio_out).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Convert to numpy
        audio_out = audio_out.squeeze().cpu().numpy()
        if not np.isfinite(audio_out).all():
            raise RuntimeError("V3 inference produced NaN/Inf audio output")

        # Match output loudness closer to source to avoid "empty/whispery" clones.
        src_rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64))) if audio.size else 0.0
        out_rms_pre = float(np.sqrt(np.mean(np.square(audio_out), dtype=np.float64))) if audio_out.size else 0.0
        if src_rms > 1e-6 and out_rms_pre > 1e-6:
            target_rms = max(src_rms * 0.8, out_rms_pre)
            gain = float(np.clip(target_rms / out_rms_pre, 0.5, 3.0))
            audio_out = audio_out * gain

        # Source-guided silence gate: suppress synthetic HF modulation in silent spots.
        gate_db = float(os.environ.get("AUDIOCLONE_V3_SILENCE_GATE_DB", "-48.0"))
        gate_strength = float(np.clip(float(os.environ.get("AUDIOCLONE_V3_SILENCE_GATE_STRENGTH", "0.90")), 0.0, 1.0))
        gate_min = float(np.clip(float(os.environ.get("AUDIOCLONE_V3_SILENCE_MIN_GAIN", "0.05")), 0.0, 1.0))
        if audio_out.size and gate_strength > 0.0:
            frame_len = 320  # 20 ms @ 16k
            hop_len = 160    # 10 ms @ 16k
            src_rms_frames = librosa.feature.rms(y=audio.astype(np.float32), frame_length=frame_len, hop_length=hop_len, center=True)[0]
            src_rms_frames = np.maximum(src_rms_frames, 1e-8)
            thr = float(10.0 ** (gate_db / 20.0))
            frame_gain = np.ones_like(src_rms_frames, dtype=np.float32)
            below = src_rms_frames < thr
            if np.any(below):
                norm = np.clip(src_rms_frames[below] / max(thr, 1e-8), 0.0, 1.0)
                frame_gain[below] = np.maximum(gate_min, np.power(norm, 2.0).astype(np.float32))
            frame_gain = (1.0 - gate_strength) + gate_strength * frame_gain
            t_src = np.linspace(0.0, 1.0, num=frame_gain.shape[0], dtype=np.float32)
            t_out = np.linspace(0.0, 1.0, num=audio_out.shape[0], dtype=np.float32)
            sample_gain = np.interp(t_out, t_src, frame_gain).astype(np.float32)
            audio_out = audio_out * sample_gain

        # Prevent hard clipping crackle on save path.
        peak_abs = float(np.max(np.abs(audio_out))) if audio_out.size else 0.0
        if peak_abs > 1.0:
            audio_out = audio_out / peak_abs * 0.995
        audio_out = np.clip(audio_out, -1.0, 1.0)

        clipped_frac = float(np.mean(np.abs(audio_out) > 0.99)) if audio_out.size else 0.0
        voiced_ratio = float(np.mean(f0 > 1.0)) if len(f0) else 0.0
        self.last_convert_debug = {
            "audio_path": audio_path,
            "input_samples_16k": int(len(audio)),
            "content_frames_hubert": content_len_hubert,
            "content_frames_model": content_len_model,
            "f0_frames_raw": f0_raw_len,
            "f0_frames_aligned": int(len(f0)),
            "upsample_factor": int(content_len_model / max(1, content_len_hubert)),
            "phone_x2": bool(phone_x2),
            "index_rate_requested": float(index_rate),
            "index_rate_effective": float(effective_rate),
            "protect": float(protect),
            "text_strength": float(text_strength),
            "noise_scale": float(noise_scale),
            "pitch_method": str(getattr(self, "pitch_method", "rmvpe")),
            "tokenizer_used": tokenizer_used,
            "text_tokens": int(len(token_ids)),
            "has_retrieval_index": bool(self.retrieval_index is not None),
            "retrieval_index_v2_style": bool(
                self.retrieval_index is not None and getattr(self.retrieval_index, "is_v2_reconstructed", False)
            ),
            "voiced_ratio": voiced_ratio,
            "output_peak_abs": float(np.max(np.abs(audio_out))) if audio_out.size else 0.0,
            "output_rms": float(np.sqrt(np.mean(np.square(audio_out), dtype=np.float64))) if audio_out.size else 0.0,
            "output_clipped_frac": clipped_frac,
            "output_duration_sec": float(audio_out.shape[0] / float(self.config.sampling_rate)) if audio_out.size else 0.0,
            "chunk_frames": int(chunk_frames),
            "overlap_frames": int(overlap_frames),
        }
        logger.info("V3 convert debug: %s", self.last_convert_debug)
        
        # Save if output path provided
        if output_path:
            sf.write(output_path, audio_out, self.config.sampling_rate)
            logger.info(f"Output saved to {output_path}")
        
        return audio_out, self.config.sampling_rate

