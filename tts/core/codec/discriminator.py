"""Discriminators for codec training."""

# This module is adapted from the following repository:
# URL: https://github.com/zhenye234/X-Codec-2.0
# License: MIT License

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def stft(
    x: torch.Tensor,
    fft_size: int,
    hop_size: int,
    win_length: int,
    window: str,
    use_complex: bool = False,
) -> torch.Tensor:
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """

    x_stft = torch.stft(
        x, fft_size, hop_size, win_length, window.to(x.device), return_complex=True
    )

    # clamp is needed to avoid nan or inf
    if not use_complex:
        return torch.sqrt(
            torch.clamp(x_stft.real**2 + x_stft.imag**2, min=1e-7, max=1e3)
        ).transpose(2, 1)
    else:
        res = torch.cat([x_stft.real.unsqueeze(1), x_stft.imag.unsqueeze(1)], dim=1)
        res = res.transpose(2, 3)  # [B, 2, T, F]
        return res


class HiFiGANPeriodDiscriminator(torch.nn.Module):
    """HiFiGAN period discriminator module."""

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        period=3,
        kernel_sizes: list[int] | None = None,
        channels=32,
        downsample_scales: list[int] | None = None,
        channel_increasing_factor=4,
        max_downsample_channels=1024,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params: dict | None = None,
        use_weight_norm=True,
    ):
        """Initialize HiFiGANPeriodDiscriminator module.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            period (int): Period.
            kernel_sizes (list): Kernel sizes of initial conv layers and the final
                conv layer.
            channels (int): Number of initial channels.
            downsample_scales (list): List of downsampling scales.
            max_downsample_channels (int): Number of maximum downsampling channels.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
        """
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [5, 3]
        if downsample_scales is None:
            downsample_scales = [3, 3, 3, 3, 1]
        if nonlinear_activation_params is None:
            nonlinear_activation_params = {"negative_slope": 0.1}
        assert len(kernel_sizes) == 2
        assert kernel_sizes[0] % 2 == 1, "Kernel size must be odd number."
        assert kernel_sizes[1] % 2 == 1, "Kernel size must be odd number."

        self.period = period
        self.convs = torch.nn.ModuleList()
        in_chs = in_channels
        out_chs = channels
        for downsample_scale in downsample_scales:
            self.convs += [
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_chs,
                        out_chs,
                        (kernel_sizes[0], 1),
                        (downsample_scale, 1),
                        padding=((kernel_sizes[0] - 1) // 2, 0),
                    ),
                    getattr(torch.nn, nonlinear_activation)(
                        **nonlinear_activation_params
                    ),
                )
            ]
            in_chs = out_chs
            out_chs = min(out_chs * channel_increasing_factor, max_downsample_channels)
        self.output_conv = torch.nn.Conv2d(
            in_chs,
            out_channels,
            (kernel_sizes[1] - 1, 1),
            1,
            padding=((kernel_sizes[1] - 1) // 2, 0),
        )

        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Calculate forward propagation.
        Args:
            c (Tensor): Input tensor (B, in_channels, T).
        Returns:
            list: List of each layer's tensors.
        """
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t += n_pad
        x = x.view(b, c, t // self.period, self.period)

        outs = []
        for layer in self.convs:
            x = layer(x)
            outs += [x]
        x = self.output_conv(x)
        x = torch.flatten(x, 1, -1)
        outs += [x]

        return outs

    def apply_weight_norm(self):
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)


class HiFiGANMultiPeriodDiscriminator(torch.nn.Module):
    """HiFiGAN multi-period discriminator module."""

    def __init__(self, periods: list[int] | None = None, **kwargs):
        """Initialize HiFiGANMultiPeriodDiscriminator module.
        Args:
            periods (list): List of periods.
            discriminator_params (dict): Parameters for hifi-gan period discriminator
                module.
                The period parameter will be overwritten.
        """
        super().__init__()
        if periods is None:
            periods = [2, 3, 5, 7, 11]
        self.discriminators = torch.nn.ModuleList()
        for period in periods:
            params = copy.deepcopy(kwargs)
            params["period"] = period
            self.discriminators += [HiFiGANPeriodDiscriminator(**params)]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Calculate forward propagation.
        Args:
            x (Tensor): Input noise signal (B, 1, T).
        Returns:
            List: List of list of each discriminator outputs, which consists of each
                layer output tensors.
        """
        outs = []
        for f in self.discriminators:
            outs += [f(x)]

        return outs


class SpecDiscriminator(nn.Module):
    """Spec discriminator."""

    def __init__(
        self,
        stft_params=None,
        in_channels=1,
        out_channels=1,
        kernel_sizes=(7, 3),
        channels=32,
        max_downsample_channels=512,
        downsample_scales=(2, 2, 2),
        use_weight_norm=True,
    ):
        super().__init__()

        if stft_params is None:
            stft_params = {
                "fft_sizes": [1024, 2048, 512],
                "hop_sizes": [120, 240, 50],
                "win_lengths": [600, 1200, 240],
                "window": "hann_window",
            }

        self.stft_params = stft_params

        self.model = nn.ModuleDict()
        for i in range(len(stft_params["fft_sizes"])):
            self.model["disc_" + str(i)] = NLayerSpecDiscriminator(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_sizes=kernel_sizes,
                channels=channels,
                max_downsample_channels=max_downsample_channels,
                downsample_scales=downsample_scales,
            )

        if use_weight_norm:
            self.apply_weight_norm()
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        results = []
        i = 0
        x = x.squeeze(1)
        for _, disc in self.model.items():
            spec = stft(
                x,
                self.stft_params["fft_sizes"][i],
                self.stft_params["hop_sizes"][i],
                self.stft_params["win_lengths"][i],
                window=getattr(torch, self.stft_params["window"])(
                    self.stft_params["win_lengths"][i]
                ),
            )  # [B, T, F]
            spec = spec.transpose(1, 2).unsqueeze(1)  # [B, 1, F, T]
            results.append(disc(spec))
            i += 1
        return results

    def remove_weight_norm(self):
        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        def _apply_weight_norm(m):
            if (
                isinstance(m, nn.Conv1d)
                or isinstance(m, nn.ConvTranspose1d)
                or isinstance(m, nn.Conv2d)
                or isinstance(m, nn.ConvTranspose2d)
            ):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        def _reset_parameters(m):
            if (
                isinstance(m, nn.Conv1d)
                or isinstance(m, nn.ConvTranspose1d)
                or isinstance(m, nn.Conv2d)
                or isinstance(m, nn.ConvTranspose2d)
            ):
                m.weight.data.normal_(0.0, 0.02)

        self.apply(_reset_parameters)


class NLayerSpecDiscriminator(nn.Module):
    """N-layer spec discriminator."""

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        kernel_sizes=(5, 3),
        channels=32,
        max_downsample_channels=512,
        downsample_scales=(2, 2, 2),
    ):
        super().__init__()

        # check kernel size is valid
        assert kernel_sizes[0] % 2 == 1
        assert kernel_sizes[1] % 2 == 1

        model = nn.ModuleDict()

        model["layer_0"] = nn.Sequential(
            nn.Conv2d(
                in_channels,
                channels,
                kernel_size=kernel_sizes[0],
                stride=2,
                padding=kernel_sizes[0] // 2,
            ),
            nn.LeakyReLU(0.2, True),
        )

        in_chs = channels
        for i, downsample_scale in enumerate(downsample_scales):
            out_chs = min(in_chs * downsample_scale, max_downsample_channels)

            model[f"layer_{i + 1}"] = nn.Sequential(
                nn.Conv2d(
                    in_chs,
                    out_chs,
                    kernel_size=downsample_scale * 2 + 1,
                    stride=downsample_scale,
                    padding=downsample_scale,
                ),
                nn.LeakyReLU(0.2, True),
            )
            in_chs = out_chs

        out_chs = min(in_chs * 2, max_downsample_channels)
        model[f"layer_{len(downsample_scales) + 1}"] = nn.Sequential(
            nn.Conv2d(
                in_chs,
                out_chs,
                kernel_size=kernel_sizes[1],
                padding=kernel_sizes[1] // 2,
            ),
            nn.LeakyReLU(0.2, True),
        )

        model[f"layer_{len(downsample_scales) + 2}"] = nn.Conv2d(
            out_chs,
            out_channels,
            kernel_size=kernel_sizes[1],
            padding=kernel_sizes[1] // 2,
        )

        self.model = model

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        results = []
        for _, layer in self.model.items():
            x = layer(x)
            results.append(x)
        return results
