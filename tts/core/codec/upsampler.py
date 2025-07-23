"""Upsampler Module that expand features for decoder."""

import torch
from absl import logging

from tts.core.codec import decoder_modules


class UpSamplerBlock(torch.nn.Module):
    """Transpose Conv plus Resnet Blocks to upsample feature embedding."""

    def __init__(
        self, in_channels: int, upsample_factors: list[int], kernel_sizes: list[int]
    ):
        """
        Args:
            num_channels (int): Number of channels..
            upsample_factors (List[int]): List of upsampling factors.
            kernel_sizes (List[int]): List of kernel sizes.
        """
        super().__init__()

        self.in_channels = in_channels
        self.upsample_factors = upsample_factors
        self.kernel_sizes = kernel_sizes

        self.upsample_layers = torch.nn.ModuleList()
        self.resnet_blocks = torch.nn.ModuleList()
        self.out_proj = torch.nn.Linear(
            self.in_channels // (2 ** len(upsample_factors)), self.in_channels
        )

        logging.info(
            f"UpsamplerBlock: in_channels: {self.in_channels}"
            f"upsample_factors: {self.upsample_factors}"
        )

        # List of upsamples followed by Resnet Blocks.
        # reference: skylark vocoder hiftnet.
        for i, (k, u) in enumerate(
            zip(self.kernel_sizes, self.upsample_factors, strict=False)
        ):
            self.upsample_layers.append(
                torch.nn.utils.weight_norm(
                    torch.nn.ConvTranspose1d(
                        self.in_channels // (2**i),
                        self.in_channels // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )
            self.resnet_blocks.append(
                decoder_modules.ResnetBlock(
                    in_channels=self.in_channels // (2 ** (i + 1)),
                    out_channels=self.in_channels // (2 ** (i + 1)),
                    dropout=0.0,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input/output shape: [batch_size, in_channels, seq_len]
        # output [batch, 1, audio_sample_len]
        for up, rsblk in zip(self.upsample_layers, self.resnet_blocks, strict=False):
            x = up(x)
            x = rsblk(x)

        return decoder_modules.nonlinearity(self.out_proj(x.transpose(1, 2)))
