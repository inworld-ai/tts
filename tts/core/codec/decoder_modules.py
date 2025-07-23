"""Codec decoder Implementation."""

# This module is adapted from the following repository:
# URL: https://github.com/zhenye234/X-Codec-2.0
# License: MIT License

import einops
import torch
import torchtune.modules as torchtune_module
import vector_quantize_pytorch as vq


def init_weights(m):
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.trunc_normal_(m.weight, std=0.02)
        torch.nn.init.constant_(m.bias, 0)


class ISTFT(torch.nn.Module):
    """Custom implementation of ISTFT."""

    def __init__(
        self, n_fft: int, hop_length: int, win_length: int, padding: str = "same"
    ):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Compute the Inverse Short Time Fourier Transform (ISTFT) of a complex
        spectrogram.

        Args:
            spec (Tensor): Input complex spectrogram of shape (B, N, T), where B is the
                batch size,
                N is the number of frequency bins, and T is the number of time frames.

        Returns:
            Tensor: Reconstructed time-domain signal of shape (B, L), where L is the
                length of the output signal.
        """
        if self.padding == "center":
            # Fallback to pytorch native implementation
            return torch.istft(
                spec,
                self.n_fft,
                self.hop_length,
                self.win_length,
                self.window,
                center=True,
            )
        elif self.padding == "same":
            pad = (self.win_length - self.hop_length) // 2
        else:
            raise ValueError("Padding must be 'center' or 'same'.")

        assert spec.dim() == 3, "Expected a 3D tensor as input"
        B, N, T = spec.shape

        # Inverse FFT
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        ifft = ifft * self.window[None, :, None]

        # Overlap and Add
        output_size = (T - 1) * self.hop_length + self.win_length
        y = torch.nn.functional.fold(
            ifft,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, pad:-pad]

        # Window envelope
        window_sq = self.window.square().expand(1, T, -1).transpose(1, 2)
        window_envelope = torch.nn.functional.fold(
            window_sq,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        ).squeeze()[pad:-pad]

        # Normalize
        assert (window_envelope > 1e-11).all()
        y = y / window_envelope

        return y


class ISTFTHead(torch.nn.Module):
    """
    ISTFT Head module for predicting STFT complex coefficients.

    Args:
        dim (int): Hidden dimension of the model.
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames,
            which should align with
                          the resolution of the input features.
        padding (str, optional): Type of padding. Options are "center" or "same".
            Defaults to "same".
    """

    def __init__(self, dim: int, n_fft: int, hop_length: int, padding: str = "same"):
        super().__init__()
        out_dim = n_fft + 2
        self.out = torch.nn.Linear(dim, out_dim)
        self.istft = ISTFT(
            n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ISTFTHead module.

        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is
                the length of the output signal.
        """
        x_pred = self.out(x)
        # x_pred = x
        x_pred = x_pred.transpose(1, 2)
        mag, p = x_pred.chunk(2, dim=1)
        mag = torch.exp(mag)
        mag = torch.clip(
            mag, max=1e2
        )  # safeguard to prevent excessively large magnitudes
        # wrapping happens here. These two lines produce real and imaginary value
        x = torch.cos(p)
        y = torch.sin(p)
        # recalculating phase here does not produce anything new
        # only costs time
        # phase = torch.atan2(y, x)
        # S = mag * torch.exp(phase * 1j)
        # better directly produce the complex value
        S = mag * (x + 1j * y)
        audio = self.istft(S)
        return audio.unsqueeze(1)


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )


class ResnetBlock(torch.nn.Module):
    """Resnet block."""

    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv1d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = torch.nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(
        self, x: torch.Tensor, temb: torch.Tensor | None = None
    ) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        r"""https://github.com/meta-llama/llama/blob/main/llama/model.py"""
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = torch.mean(x**2, dim=-1, keepdim=True)
        output = x * torch.rsqrt(norm_x + self.eps) * self.weight
        return output


class MLP(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()

        self.fc1 = torch.nn.Linear(dim, 4 * dim, bias=False)
        self.silu = torch.nn.SiLU()
        self.fc2 = torch.nn.Linear(4 * dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.silu(x)
        x = self.fc2(x)
        return x


class Attention(torch.nn.Module):
    """Attention module."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        rotary_embed: torchtune_module.RotaryPositionalEmbeddings,
    ):
        super().__init__()

        if dim % n_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by n_heads {n_heads}")

        self.n_heads = n_heads
        self.dim = dim
        self.rotary_embed = rotary_embed

        self.c_attn = torch.nn.Linear(dim, 3 * dim, bias=False)
        self.c_proj = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = einops.rearrange(
            self.c_attn(x), "b t (r h d) -> r b h t d", r=3, h=self.n_heads
        )

        q = self.rotary_embed(q)
        k = self.rotary_embed(k)

        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0, is_causal=False
        )

        y = einops.rearrange(y, "b h t d -> b t (h d)")
        y = self.c_proj(y)

        return y


class TransformerBlock(torch.nn.Module):
    """Transformer block."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        rotary_embed: torchtune_module.RotaryPositionalEmbeddings,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads

        self.att_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        self.att = Attention(dim=dim, n_heads=n_heads, rotary_embed=rotary_embed)
        self.mlp = MLP(dim=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.att(self.att_norm(x))
        x = x + self.mlp(self.ffn_norm(x))
        return x


class VocosBackbone(torch.nn.Module):
    """
    Vocos backbone module built with ConvNeXt blocks. Supports additional conditioning
    with Adaptive Layer Normalization

    Args:
        input_channels (int): Number of input features channels.
        dim (int): Hidden dimension of the model.
        intermediate_dim (int): Intermediate dimension used in ConvNeXtBlock.
        num_layers (int): Number of ConvNeXtBlock layers.
        layer_scale_init_value (float, optional): Initial value for layer scaling.
            Defaults to `1 / num_layers`.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        depth: int = 12,
        heads: int = 16,
        pos_meb_dim: int = 64,
    ):
        super().__init__()

        self.embed = torch.nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3)

        self.temb_ch = 0
        block_in = hidden_dim
        dropout = 0.1

        prior_net: list[torch.nn.Module] = [
            ResnetBlock(
                in_channels=block_in,
                out_channels=block_in,
                temb_channels=self.temb_ch,
                dropout=dropout,
            ),
            ResnetBlock(
                in_channels=block_in,
                out_channels=block_in,
                temb_channels=self.temb_ch,
                dropout=dropout,
            ),
        ]
        self.prior_net = torch.nn.Sequential(*prior_net)

        depth = depth
        time_rotary_embed = torchtune_module.RotaryPositionalEmbeddings(dim=pos_meb_dim)

        transformer_blocks = [
            TransformerBlock(
                dim=hidden_dim, n_heads=heads, rotary_embed=time_rotary_embed
            )
            for _ in range(depth)
        ]

        self.transformers = torch.nn.Sequential(*transformer_blocks)
        self.final_layer_norm = torch.nn.LayerNorm(hidden_dim, eps=1e-6)
        post_net: list[torch.nn.Module] = [
            ResnetBlock(
                in_channels=block_in,
                out_channels=block_in,
                temb_channels=self.temb_ch,
                dropout=dropout,
            ),
            ResnetBlock(
                in_channels=block_in,
                out_channels=block_in,
                temb_channels=self.temb_ch,
                dropout=dropout,
            ),
        ]
        self.post_net = torch.nn.Sequential(*post_net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.embed(x)
        x = self.prior_net(x)
        x = x.transpose(1, 2)
        x = self.transformers(x)
        x = x.transpose(1, 2)
        x = self.post_net(x)
        x = x.transpose(1, 2)
        x = self.final_layer_norm(x)
        return x


class Generator(torch.nn.Module):
    """Codec decoder module."""

    def __init__(
        self,
        hidden_dim=1024,
        depth=12,
        heads=16,
        pos_meb_dim=64,
        hop_length=320,
        vq_dim=2048,
    ):
        super().__init__()
        self.hop_length = hop_length

        self.quantizer = vq.ResidualFSQ(
            dim=vq_dim, levels=[4, 4, 4, 4, 4, 4, 4, 4], num_quantizers=1
        )

        self.backbone = VocosBackbone(
            hidden_dim=hidden_dim, depth=depth, heads=heads, pos_meb_dim=pos_meb_dim
        )

        self.head = ISTFTHead(
            dim=hidden_dim,
            n_fft=self.hop_length * 4,
            hop_length=self.hop_length,
            padding="same",
        )

        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.head(x)

        return x

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(
                m, torch.nn.ConvTranspose1d
            ):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        self.apply(init_weights)
