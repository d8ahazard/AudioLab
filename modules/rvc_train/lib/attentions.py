import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size=1, p_dropout=0.0,
                 window_size=10, **kwargs):
        super(Encoder, self).__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = int(n_layers)
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size

        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for i in range(self.n_layers):
            self.attn_layers.append(
                nn.MultiheadAttention(hidden_channels, n_heads, dropout=p_dropout, batch_first=True))
            self.norm_layers_1.append(nn.LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout))
            self.norm_layers_2.append(nn.LayerNorm(hidden_channels))

    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(1) * x_mask.unsqueeze(2)
        # [bs, seq_len, seq_len] -> [seq_len, seq_len]
        attn_mask = attn_mask[0]
        # Assuming x_mask is [seq_length, seq_length] with 1s for valid positions and 0s for masked
        attn_mask = attn_mask == 0  # Flip mask: now 1s indicate positions to mask out
        x = x * x_mask.unsqueeze(-1)
        for attn_layer, norm_layer_1, ffn_layer, norm_layer_2 in zip(self.attn_layers, self.norm_layers_1,
                                                                     self.ffn_layers, self.norm_layers_2):
            y, _ = attn_layer(x, x, x, attn_mask=attn_mask)
            y = self.drop(y)
            x = norm_layer_1(x + y)

            y = ffn_layer(x, x_mask)
            y = self.drop(y)
            x = norm_layer_2(x + y)
        x = x * x_mask.unsqueeze(-1)
        return x


class FFN(nn.Module):
    def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0.0, activation: str = None,
                 causal=False):
        super(FFN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = activation
        self.causal = causal
        self.is_activation = True if activation == "gelu" else False

        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.drop = nn.Dropout(p_dropout)

    def apply_padding(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        if self.causal:
            padding = self._causal_padding(x * x_mask.unsqueeze(-1))
        else:
            padding = self._same_padding(x * x_mask.unsqueeze(-1))
        return padding

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor):
        x_padded = self.apply_padding(x, x_mask)

        x = x_padded.transpose(1, 2)  # input for conv
        x = self.conv_1(x)
        x = x.transpose(1, 2)  # input for conv revert
        if self.is_activation:
            x = F.gelu(x)
        else:
            x = F.relu(x)
        x = self.drop(x)

        x_padded = self.apply_padding(x, x_mask)

        x = x_padded.transpose(1, 2)  # input for conv
        x = self.conv_2(x)
        x = x.transpose(1, 2)  # input for conv revert
        return x * x_mask.unsqueeze(-1)

    def _causal_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l: int = self.kernel_size - 1
        pad_r: int = 0
        # padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        # padding is done from the end
        x = F.pad(x, [0, 0, pad_l, pad_r, 0, 0])
        return x

    def _same_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l: int = (self.kernel_size - 1) // 2
        pad_r: int = self.kernel_size // 2
        # padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        # padding is done from the end
        x = F.pad(x, [0, 0, pad_l, pad_r, 0, 0])
        return x
