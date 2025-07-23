"""Implementation of the encoder modules."""

# This module is adapted from the following repository:
# URL: https://github.com/zhenye234/X-Codec-2.0
# License: MIT License

import numpy as np
import torch

from tts.core.codec import activations


def init_weights(m: torch.nn.Module):
    """Initializes the weights of the model."""
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.trunc_normal_(m.weight, std=0.02)
        torch.nn.init.constant_(m.bias, 0)


class ResidualUnit(torch.nn.Module):
    """Residual block."""

    def __init__(self, dim: int = 16, kernel_size: int = 7, dilation: int = 1):
        super().__init__()
        pad = ((kernel_size - 1) * dilation) // 2
        self.block = torch.nn.Sequential(
            activations.Activation1d(
                activation=activations.SnakeBeta(dim, alpha_logscale=True)
            ),
            torch.nn.utils.weight_norm(
                torch.nn.Conv1d(
                    dim, dim, kernel_size=kernel_size, dilation=dilation, padding=pad
                )
            ),
            activations.Activation1d(
                activation=activations.SnakeBeta(dim, alpha_logscale=True)
            ),
            torch.nn.utils.weight_norm(torch.nn.Conv1d(dim, dim, kernel_size=1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class EncoderBlock(torch.nn.Module):
    """Encoder block."""

    def __init__(self, dim: int = 16, stride: int = 1, dilations=(1, 3, 9)):
        super().__init__()
        runits = [ResidualUnit(dim // 2, dilation=d) for d in dilations]
        self.block = torch.nn.Sequential(
            *runits,
            activations.Activation1d(
                activation=activations.SnakeBeta(dim // 2, alpha_logscale=True)
            ),
            torch.nn.utils.weight_norm(
                torch.nn.Conv1d(
                    dim // 2,
                    dim,
                    kernel_size=2 * stride,
                    stride=stride,
                    padding=stride // 2 + stride % 2,
                )
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SemanticEncoder(torch.nn.Module):
    """Semantic encoder model to encode the wav2vec features."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        encode_channels: int,
        kernel_size: int,
        bias: bool = True,
    ):
        super().__init__()

        self.initial_conv = torch.nn.Conv1d(
            in_channels=input_channels,
            out_channels=encode_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )

        self.residual_blocks = torch.nn.Sequential(
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(
                encode_channels,
                encode_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                bias=bias,
            ),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(
                encode_channels,
                encode_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                bias=bias,
            ),
        )

        self.final_conv = torch.nn.Conv1d(
            in_channels=encode_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial_conv(x)
        x = self.residual_blocks(x) + x
        x = self.final_conv(x)
        return x


class AcousticEncoder(torch.nn.Module):
    """Acoustic encoder model to encode the audio waveform."""

    def __init__(
        self,
        num_generator_features: int,
        initial_conv_kernel_size: int,
        final_conv_kernel_size: int,
        up_ratios: list[int],
        dilations: tuple[int],
        output_dim: int,
    ):
        super().__init__()
        self.hop_length = np.prod(up_ratios)
        self.num_generator_features = num_generator_features
        self.up_ratios = up_ratios

        d_model = num_generator_features

        self.conv_blocks = [
            torch.nn.utils.weight_norm(
                torch.nn.Conv1d(
                    1,
                    d_model,
                    kernel_size=initial_conv_kernel_size,
                    padding=(initial_conv_kernel_size - 1) // 2,
                )
            )
        ]

        for _, stride in enumerate(up_ratios):
            d_model *= 2
            self.conv_blocks += [
                EncoderBlock(d_model, stride=stride, dilations=dilations)
            ]

        self.conv_blocks = torch.nn.Sequential(*self.conv_blocks)

        self.conv_final_block = [
            activations.Activation1d(
                activation=activations.SnakeBeta(d_model, alpha_logscale=True)
            ),
            torch.nn.utils.weight_norm(
                torch.nn.Conv1d(
                    d_model,
                    output_dim,
                    kernel_size=final_conv_kernel_size,
                    padding=(final_conv_kernel_size - 1) // 2,
                )
            ),
        ]
        self.conv_final_block = torch.nn.Sequential(*self.conv_final_block)

        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_blocks(x)
        x = self.conv_final_block(x)
        x = x.permute(0, 2, 1)
        return x

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m: torch.nn.Module):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m: torch.nn.Module):
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        self.apply(init_weights)
