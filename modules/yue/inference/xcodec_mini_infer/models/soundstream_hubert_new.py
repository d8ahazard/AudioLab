import math
import os
from typing import Sequence, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

import modules.yue.inference.xcodec_mini_infer.descriptaudiocodec.dac.model.dac as dac2
from handlers.config import model_path, app_path
from modules.yue.inference.xcodec_mini_infer.RepCodec.repcodec.modules.decoder import Decoder
from modules.yue.inference.xcodec_mini_infer.RepCodec.repcodec.modules.encoder import Encoder
from modules.yue.inference.xcodec_mini_infer.quantization import ResidualVectorQuantizer


def get_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    model_size_bytes = total_params
    model_size_mb = model_size_bytes / (1024 ** 2)
    return total_params, model_size_mb


class SoundStream(nn.Module):
    def __init__(
            self,
            n_filters: int = 32,
            D: int = 128,
            target_bandwidths: Sequence[Union[int, float]] = [1, 1.5, 2, 4, 6],
            ratios: Sequence[int] = [8, 5, 4, 2],
            sample_rate: int = 16000,
            bins: int = 1024,
            normalize: bool = False,
            causal: bool = False,
    ):
        super().__init__()
        self.hop_length = np.prod(ratios)
        n_q = int(1000 * target_bandwidths[-1] // (math.ceil(sample_rate / self.hop_length) * 10))
        self.frame_rate = math.ceil(sample_rate / np.prod(ratios))
        self.bits_per_codebook = int(math.log2(bins))
        self.target_bandwidths = target_bandwidths
        self.n_q = n_q
        self.sample_rate = sample_rate

        self.encoder = dac2.Encoder(64, ratios, D)
        self.encoder_semantic = Encoder(input_channels=768, encode_channels=768)
        self.decoder_semantic = Decoder(code_dim=768, output_channels=768, decode_channels=768)
        self.quantizer = ResidualVectorQuantizer(dimension=D + 768, n_q=n_q, bins=bins)
        self.decoder_2 = dac2.Decoder(D, 1024, ratios)

        new_xcodec_path = os.path.join(model_path, "YuE", "hf_1_325000")
        xcodec_path = new_xcodec_path if os.path.exists(new_xcodec_path) else os.path.join(
            app_path, "modules", "yue", "inference", "xcodec_mini_infer", "semantic_ckpts", "hf_1_325000"
        )
        self.semantic_model = AutoModel.from_pretrained(xcodec_path)
        self.semantic_model.eval()

        self.fc_prior = nn.Linear(D + 768, D + 768)
        self.fc_post1 = nn.Linear(D + 768, 768)
        self.fc_post2 = nn.Linear(D + 768, D)

    def get_last_layer(self):
        return self.decoder_2.layers[-1].weight

    def calculate_rec_loss(self, rec, target):
        target = target / target.norm(dim=-1, keepdim=True)
        rec = rec / rec.norm(dim=-1, keepdim=True)
        return (1 - (target * rec).sum(-1)).mean()

    @torch.no_grad()
    def get_regress_target(self, x):
        x = x[:, 0, :]
        x = F.pad(x, (160, 160))
        target = self.semantic_model(x, output_hidden_states=True).hidden_states
        return torch.stack(target, dim=1).mean(1)

    def forward(self, x: torch.Tensor, bw: int):
        e_semantic_input = self.get_regress_target(x).detach()
        e_semantic = self.encoder_semantic(e_semantic_input.transpose(1, 2))
        e_acoustic = self.encoder(x)
        e = torch.cat([e_acoustic, e_semantic], dim=1)
        e = self.fc_prior(e.transpose(1, 2)).transpose(1, 2)

        quantized, codes, bandwidth, commit_loss = self.quantizer(e, self.frame_rate, bw)
        quantized_semantic = self.fc_post1(quantized.transpose(1, 2)).transpose(1, 2)
        quantized_acoustic = self.fc_post2(quantized.transpose(1, 2)).transpose(1, 2)

        o = self.decoder_2(quantized_acoustic)
        o_semantic = self.decoder_semantic(quantized_semantic)
        semantic_recon_loss = F.mse_loss(e_semantic_input.transpose(1, 2).detach(), o_semantic)

        return o, commit_loss, semantic_recon_loss, None

    def encode(self, x: torch.Tensor, target_bw: Optional[int] = None) -> torch.Tensor:
        bw = target_bw
        e_semantic_input = self.get_regress_target(x).detach()
        e_semantic = self.encoder_semantic(e_semantic_input.transpose(1, 2))
        e_acoustic = self.encoder(x)

        if e_acoustic.shape[2] != e_semantic.shape[2]:
            e_acoustic = self.encoder(torch.transpose(F.pad(x[:, 0, :], (160, 160)).unsqueeze(0), 0, 1))

        e = torch.cat([e_acoustic, e_semantic], dim=1)
        e = self.fc_prior(e.transpose(1, 2)).transpose(1, 2)

        quantized, codes, bandwidth, commit_loss = self.quantizer(e, self.frame_rate, bw)
        return codes

    def get_embed(self, codes: torch.Tensor) -> torch.Tensor:
        return self.quantizer.decode(codes)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        quantized = self.quantizer.decode(codes)
        quantized_acoustic = self.fc_post2(quantized.transpose(1, 2)).transpose(1, 2)
        return self.decoder_2(quantized_acoustic)
