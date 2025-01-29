import math
import logging
from typing import Optional

logger = logging.getLogger(__name__)

import torch
from torch import nn

from modules.rvc.lib.commons import sequence_mask, rand_slice_segments, slice_segments2
from modules.rvc.lib.attentions import Encoder
from modules.rvc.lib.modules import WN, ResidualCouplingBlock
from modules.rvc.lib.models_hifigan import GeneratorNSF
from modules.rvc.lib.models_bigvgan import GeneratorBigVgan

sr2sr = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}


# Class representing a text encoder with 768-dimensional input.
class TextEncoder(nn.Module):
    def __init__(
            self,
            out_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            f0=True,
            ppg_dim=None
    ):
        super(TextEncoder, self).__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = float(p_dropout)
        self.emb_phone = nn.Linear(768, hidden_channels)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.encoder = Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
        if f0 is True:
            # coarse f0 could take 256 values
            self.emb_pitch = nn.Embedding(256, hidden_channels)
        if ppg_dim is not None:
            self.ppg_dim = ppg_dim
            self.ppg_pre = nn.Conv1d(self.ppg_dim, hidden_channels, kernel_size=5, padding=2)

    def forward(self, phone: torch.Tensor, pitch: torch.Tensor, lengths: torch.Tensor, ppg=None):
        if pitch is None:
            x = self.emb_phone(phone)
        else:
            x = self.emb_phone(phone) + self.emb_pitch(pitch)

        if ppg is not None and hasattr(self, 'ppg_pre'):
            # we transpose before input to conv1d
            # later we transpose back
            ppg = torch.transpose(ppg, 1, 2)  # [b, h, t]
            ppg = self.ppg_pre(ppg)
            ppg = torch.transpose(ppg, 1, 2)
            x = x + ppg
        x = x * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = self.lrelu(x)
        # x shape is [bs, seq_len, hs]

        # [bs, seq_len]
        x_mask = sequence_mask(lengths, x.size(1)).to(x.dtype)

        # Adjust the forward pass call for the encoder.
        # Note: If nn.MultiheadAttention expects a different attn_mask shape or type, adjust accordingly.
        # Here, attn_mask is used directly if the attention layer can handle boolean masks.
        # Otherwise, convert it to the expected format or values.
        x = self.encoder(x, x_mask)
        x_mask = x_mask.unsqueeze(-1)

        # since proj is another conv, we have to transpose again
        stats = self.proj(x.transpose(1, 2)).transpose(1, 2) * x_mask
        stats = stats.transpose(1, 2)

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return m, logs, x_mask


class PosteriorEncoder(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=0,
    ):
        super(PosteriorEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
            self, x: torch.Tensor, x_lengths: torch.Tensor, g: Optional[torch.Tensor] = None
    ):
        """
        Perform forward pass of network
        Specifically, perform operations to calculate a mean and log variance vector
        representing a latent distribution conditioned on x
        """
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()

    def __prepare_scriptable__(self):
        for hook in self.enc._forward_pre_hooks.values():
            if (
                    hook.__module__ == "torch.nn.utils.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
            ):
                torch.nn.utils.remove_weight_norm(self.enc)
        return self


class RVCModel(nn.Module):
    def __init__(
            self,
            spec_channels,
            segment_size,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            spk_embed_dim,
            gin_channels,
            sr,
            vocoder_type='hifigan',
            ppg_dim=None,
            **kwargs
    ):
        super(RVCModel, self).__init__()
        if isinstance(sr, str):
            sr = sr2sr[sr]
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = float(p_dropout)
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        # self.hop_length = hop_length
        self.spk_embed_dim = spk_embed_dim

        # speaker embedding
        self.emb_g = nn.Embedding(self.spk_embed_dim, gin_channels)

        # text encoder
        self.enc_p = TextEncoder(
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            float(p_dropout),
            ppg_dim=ppg_dim,
        )

        # posterior encoder
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )

        # normalizing flow
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 3, gin_channels=gin_channels
        )

        if vocoder_type == 'hifigan':
            # decoder (nsf-hifigan)
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
                is_half=True,
            )
        elif vocoder_type == 'bigvgan':
            # decoder (big-vgan)
            self.dec = GeneratorBigVgan(
                resblock_kernel_sizes=resblock_kernel_sizes,
                resblock_dilation_sizes=resblock_dilation_sizes,
                upsample_rates=upsample_rates,
                upsample_kernel_sizes=upsample_kernel_sizes,
                upsample_input=inter_channels,
                upsample_initial_channel=upsample_initial_channel,
                spk_dim=gin_channels,
                sampling_rate=sr,
            )

        logger.debug(
            "gin_channels: "
            + str(gin_channels)
            + ", self.spk_embed_dim: "
            + str(self.spk_embed_dim)
        )

    def remove_weight_norm(self):
        self.dec.remove_weight_norm()
        self.flow.remove_weight_norm()
        self.enc_q.remove_weight_norm()

    def __prepare_scriptable__(self):
        for hook in self.dec._forward_pre_hooks.values():
            # The hook we want to remove is an instance of WeightNorm class, so
            # normally we would do `if isinstance(...)` but this class is not accessible
            # because of shadowing, so we check the module name directly.
            # https://github.com/pytorch/pytorch/blob/be0ca00c5ce260eb5bcec3237357f7a30cc08983/torch/nn/utils/__init__.py#L3
            if (
                    hook.__module__ == "torch.nn.utils.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
            ):
                torch.nn.utils.remove_weight_norm(self.dec)
        for hook in self.flow._forward_pre_hooks.values():
            if (
                    hook.__module__ == "torch.nn.utils.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
            ):
                torch.nn.utils.remove_weight_norm(self.flow)
        if hasattr(self, "enc_q"):
            for hook in self.enc_q._forward_pre_hooks.values():
                if (
                        hook.__module__ == "torch.nn.utils.weight_norm"
                        and hook.__class__.__name__ == "WeightNorm"
                ):
                    torch.nn.utils.remove_weight_norm(self.enc_q)
        return self

    # MAIN FORWARD PASS
    @torch.jit.ignore
    def forward(
            self,
            phone, phone_lengths, pitch, pitchf, y, y_lengths, ds,
            ppg=None,
            enable_perturbation: bool = False
    ):
        # phone, phone_lengths – output of contentvec (hubert)
        # ppg, ppg_lengths – output of whisper ppg
        # pitch, pitchf – pitch values
        # y, y_lengths – mel spectrogram for posterior
        # ds – speaker ids, transformed to speaker embedding

        if enable_perturbation:
            # Whisper ppg perturbation
            ppg = ppg + torch.randn_like(ppg) * 1
            # VAE perturbation
            phone = phone + torch.randn_like(phone) * 2  # Perturbation

        g = self.emb_g(ds).unsqueeze(-1)  # [b, 256, 1]
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths, ppg=ppg)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)
        z_slice, ids_slice = rand_slice_segments(
            z, y_lengths, self.segment_size
        )
        # print(-1,pitchf.shape,ids_slice,self.segment_size,self.hop_length,self.segment_size//self.hop_length)
        pitchf = slice_segments2(pitchf, ids_slice, self.segment_size)
        # print(-2,pitchf.shape,z_slice.shape)
        o = self.dec(z_slice, pitchf, g=g)
        return o, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    @torch.jit.export
    def infer(
            self,
            phone: torch.Tensor,
            phone_lengths: torch.Tensor,
            pitch: torch.Tensor,
            nsff0: torch.Tensor,
            sid: torch.Tensor,
            rate: Optional[torch.Tensor] = None,
            ppg=None,
            enable_perturbation: bool = False
    ):
        # we expect than phone lengths and ppg_lengths are the same
        if enable_perturbation:
            ppg = ppg + torch.randn_like(ppg) * 0.0001  # Perturbation

        g = self.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths, ppg=ppg)
        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
        if rate is not None:
            head = int(z_p.shape[2] * (1.0 - rate.item()))
            z_p = z_p[:, :, head:]
            x_mask = x_mask[:, :, head:]
            nsff0 = nsff0[:, head:]
        z = self.flow(z_p, x_mask, g=g, reverse=True)
        o = self.dec(z * x_mask, nsff0, g=g)
        return o, x_mask, (z, z_p, m_p, logs_p)
