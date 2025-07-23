"""Loss functions for the for vocoder training."""

import torch
import torchaudio


class GANLoss(torch.nn.Module):
    """GAN loss."""

    def __init__(self):
        super().__init__()

    def disc_loss(self, real, fake):
        real_loss = torch.nn.functional.mse_loss(real, torch.ones_like(real))
        fake_loss = torch.nn.functional.mse_loss(fake, torch.zeros_like(fake))
        return real_loss, fake_loss

    def gen_loss(self, fake):
        gen_loss = torch.nn.functional.mse_loss(fake, torch.ones_like(fake))
        return gen_loss


class MultiResolutionMelSpectrogramLoss(torch.nn.Module):
    """Multi-resolution mel spectrogram loss."""

    def __init__(
        self,
        sample_rate=16000,
        n_mels: list[int] | None = None,
        window_lengths: list[int] | None = None,
        clamp_eps: float = 1e-5,
        pow: float = 1.0,
        mel_fmin: list[float] | None = None,
        mel_fmax: list[float | None] | None = None,
    ):
        super().__init__()
        if n_mels is None:
            n_mels = [5, 10, 20, 40, 80, 160, 320]
        if window_lengths is None:
            window_lengths = [32, 64, 128, 256, 512, 1024, 2048]
        if mel_fmin is None:
            mel_fmin = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if mel_fmax is None:
            mel_fmax = [None, None, None, None, None, None, None]
        self.mel_transforms = torch.nn.ModuleList(
            [
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=window_length,
                    hop_length=window_length // 4,
                    n_mels=n_mel,
                    power=1.0,
                    center=True,
                    norm="slaney",
                    mel_scale="slaney",
                )
                for n_mel, window_length in zip(n_mels, window_lengths, strict=False)
            ]
        )
        self.n_mels = n_mels
        self.loss_fn = torch.nn.L1Loss()
        self.clamp_eps = clamp_eps
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.pow = pow

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        for mel_transform in self.mel_transforms:
            x_mel = mel_transform(x)
            y_mel = mel_transform(y)
            log_x_mel = x_mel.clamp(self.clamp_eps).pow(self.pow).log10()
            log_y_mel = y_mel.clamp(self.clamp_eps).pow(self.pow).log10()
            loss += self.loss_fn(log_x_mel, log_y_mel)
        return loss


class STFTLoss(torch.nn.Module):
    """STFT loss."""

    def __init__(self, fft_size: int, hop_size: int, win_size: int):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = win_size
        self.register_buffer("window", torch.hann_window(win_size))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_stft = torch.stft(
            x,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.window,
            return_complex=True,
        )
        y_stft = torch.stft(
            y,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.window,
            return_complex=True,
        )

        x_mag = torch.abs(x_stft)
        y_mag = torch.abs(y_stft)

        # Spectral Convergence Loss
        sc_loss = torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")

        # Log STFT Magnitude Loss
        log_x_mag = torch.log(x_mag + 1e-7)
        log_y_mag = torch.log(y_mag + 1e-7)
        mag_loss = torch.nn.functional.l1_loss(log_x_mag, log_y_mag)

        return sc_loss + mag_loss


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi-resolution STFT loss."""

    def __init__(
        self,
        fft_sizes: list[int] | None = None,
        hop_sizes: list[int] | None = None,
        win_sizes: list[int] | None = None,
    ):
        super().__init__()
        if fft_sizes is None:
            fft_sizes = [1024, 2048, 512]
        if hop_sizes is None:
            hop_sizes = [120, 240, 50]
        if win_sizes is None:
            win_sizes = [600, 1200, 240]
        self.loss_funcs = torch.nn.ModuleList(
            [
                STFTLoss(fft_size, hop_size, win_size)
                for fft_size, hop_size, win_size in zip(
                    fft_sizes, hop_sizes, win_sizes, strict=False
                )
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        losses = [loss_fn(x, y) for loss_fn in self.loss_funcs]
        return sum(losses) / len(losses)
