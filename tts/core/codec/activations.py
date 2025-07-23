"""Activation functions for codec models."""

import torch

from tts.core.codec import filters


class Snake(torch.nn.Module):
    """Implementation of a sine-based periodic activation function."""

    def __init__(
        self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False
    ):
        super().__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # log scale alphas initialized to zeros
            self.alpha = torch.nn.Parameter(torch.zeros(in_features) * alpha)
        else:  # linear scale alphas initialized to ones
            self.alpha = torch.nn.Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable

        self.no_div_by_zero = 1e-9

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        Snake ∶= x + 1/a * sin^2 (xa)
        """
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # line up with x to [B, C, T]
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
        x = x + (1.0 / (alpha + self.no_div_by_zero)) * torch.pow(
            torch.sin(x * alpha), 2
        )

        return x


class SnakeBeta(torch.nn.Module):
    """A modified Snake function."""

    def __init__(
        self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False
    ):
        super().__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # log scale alphas initialized to zeros
            self.alpha = torch.nn.Parameter(torch.zeros(in_features) * alpha)
            self.beta = torch.nn.Parameter(torch.zeros(in_features) * alpha)
        else:  # linear scale alphas initialized to ones
            self.alpha = torch.nn.Parameter(torch.ones(in_features) * alpha)
            self.beta = torch.nn.Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 1e-9

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta ∶= x + 1/b * sin^2 (xa)
        """
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = x + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(
            torch.sin(x * alpha), 2
        )

        return x


class Activation1d(torch.nn.Module):
    """
    A 1D activation function that applies an activation
    function to the input and then downsamples the output.
    """

    def __init__(
        self,
        activation,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = filters.UpSample1d(up_ratio, up_kernel_size)
        self.downsample = filters.DownSample1d(down_ratio, down_kernel_size)

    # x: [B,C,T]
    def forward(self, x):
        x = self.upsample(x)
        x = self.act(x)
        return self.downsample(x)
