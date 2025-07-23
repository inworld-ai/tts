"""Decoder model for audio waveform generation."""

import collections
import itertools
import math
from typing import Any

import torch
from absl import logging

from tts.core.codec import criterion, decoder_modules, discriminator, upsampler


class Decoder(torch.nn.Module):
    """The decoder model for audio waveform generation."""

    def __init__(
        self,
        sample_rate: int,
        hop_length: int,
        upsample_factors: list[int],
        kernel_sizes: list[int],
        checkpoint_path: str | None = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.upsample_factors = upsample_factors
        self.kernel_sizes = kernel_sizes

        total_ups = math.prod(self.upsample_factors) if self.upsample_factors else 1
        if self.sample_rate // self.hop_length // total_ups != 50:
            raise ValueError(
                f"Current hop length {self.hop_length} and upsample "
                f"factors {self.upsample_factors} do not match the target "
                f"sample rate {self.sample_rate}."
            )

        logging.info(
            "Creating audio decoder with output sample_rate"
            f" {self.sample_rate}, upsample factors {self.upsample_factors}, "
            f"kernel sizes {self.kernel_sizes}, hop_length {self.hop_length}."
        )

        # decoder includes the quantizer and the vocoder.
        self.decoder = decoder_modules.Generator(hop_length=hop_length)

        if self.upsample_factors:
            self.upsampler = upsampler.UpSamplerBlock(
                in_channels=1024,
                upsample_factors=self.upsample_factors,
                kernel_sizes=self.kernel_sizes,
            )
            logging.info(
                "Upsampler created with upsample factors %s and kernel sizes %s",
                self.upsample_factors,
                self.kernel_sizes,
            )
        else:
            logging.info("No upsampler created.")
            self.upsampler = None

        self.fc_post_a = torch.nn.Linear(2048, 1024)

        if checkpoint_path is not None:
            logging.info("Loading codec checkpoint from %s", checkpoint_path)
            self.load_from_checkpoint(checkpoint_path)

    def forward(self, vq_codes: torch.Tensor) -> torch.Tensor:
        """Computes VQ codes for a batch of audio files."""

        # vq_codes: (batch_size, codes_length) or (batch_size, 1, code_length)
        if len(vq_codes.shape) == 2:
            vq_codes = vq_codes.unsqueeze(1)

        vq_codes = vq_codes.transpose(1, 2).to(self.fc_post_a.weight.device)
        vq_post_emb = self.decoder.quantizer.get_output_from_indices(vq_codes)

        vq_post_emb_fc = self.fc_post_a(vq_post_emb)

        # vocos_backbone -> upsampler -> istft_head
        decoder_hidden = self.decoder.backbone(vq_post_emb_fc)
        if self.upsampler:
            upsampled_hidden = self.upsampler(decoder_hidden.transpose(1, 2))
        else:
            upsampled_hidden = decoder_hidden

        recon_audio = self.decoder.head(upsampled_hidden)
        return recon_audio

    def load_from_checkpoint(self, checkpoint_path: str) -> None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if "state_dict" in ckpt.keys():
            # to provide compatibility for the checkpoint in
            # https://huggingface.co/HKUSTAudio/xcodec2/tree/main/ckpt
            ckpt = ckpt["state_dict"]

            filtered_state_dict_generator = collections.OrderedDict()
            filtered_state_dict_fc_post = collections.OrderedDict()
            for key, value in ckpt.items():
                if key.startswith("generator."):
                    new_key = key[len("generator.") :]
                    filtered_state_dict_generator[new_key] = value
                elif key.startswith("fc_post_a."):
                    new_key = key[len("fc_post_a.") :]
                    filtered_state_dict_fc_post[new_key] = value

            self.decoder.load_state_dict(filtered_state_dict_generator)
            self.fc_post_a.load_state_dict(filtered_state_dict_fc_post)

        else:
            ckpt = ckpt["model"]
            ckpt = {
                k.replace("generator.", ""): v
                for k, v in ckpt.items()
                if k.startswith("generator.")
            }
            self.load_state_dict(ckpt, strict=True)


class TrainableDecoder(torch.nn.Module):
    """A training wrapper for the decoder model."""

    def __init__(
        self,
        generator: Decoder,
        mpd: discriminator.HiFiGANMultiPeriodDiscriminator,
        msd: discriminator.SpecDiscriminator,
    ):
        super().__init__()

        self.sample_rate = generator.sample_rate

        # generator / discriminators
        self.generator = generator
        self.mpd = mpd
        self.msd = msd

        # gan training related losses
        self.gan_loss = criterion.GANLoss()
        self.mel_loss = criterion.MultiResolutionMelSpectrogramLoss(
            sample_rate=self.sample_rate
        )
        self.fm_loss = torch.nn.L1Loss()

        # Hyperparameters
        # TODO: add them to config json instead of hardcoding here.
        self.lambda_disc = 1.0
        self.lambda_fm = 1.0
        self.lambda_mel = 15.0
        self.lambda_adv = 1.0
        self.lambda_rms = 1.0

        # gradient clip norm
        self.grad_clip_disc = 1.0
        self.grad_clip_gen = 1.0

    def set_discriminator_gradients(self, flag=True):
        """utility function to disable discriminators gradients."""
        for p in self.mpd.parameters():
            p.requires_grad = flag

        for p in self.msd.parameters():
            p.requires_grad = flag

    def forward(self, vq_codes: torch.Tensor) -> torch.Tensor:
        return self.generator(vq_codes)

    def compute_discriminator_loss(
        self, y_true: torch.Tensor, y_gen: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """discriminator loss."""
        # no gradient for generator when training discriminator
        y_gen = y_gen.detach()

        # MPD
        p_mpd_true = self.mpd(y_true)
        p_mpd_gen = self.mpd(y_gen)

        real_loss_list, fake_loss_list = [], []
        for i in range(len(p_mpd_true)):
            real_loss, fake_loss = self.gan_loss.disc_loss(
                p_mpd_true[i][-1], p_mpd_gen[i][-1]
            )
            real_loss_list.append(real_loss)
            fake_loss_list.append(fake_loss)

        # MSD
        p_msd_true = self.msd(y_true)
        p_msd_gen = self.msd(y_gen)

        for i in range(len(p_msd_true)):
            real_loss, fake_loss = self.gan_loss.disc_loss(
                p_msd_true[i][-1], p_msd_gen[i][-1]
            )
            real_loss_list.append(real_loss)
            fake_loss_list.append(fake_loss)

        # sum up all the losses
        real_loss = sum(real_loss_list)
        fake_loss = sum(fake_loss_list)

        disc_loss = real_loss + fake_loss
        disc_loss = self.lambda_disc * disc_loss

        return {"disc_loss": disc_loss}

    def compute_generator_loss(
        self, y_true: torch.Tensor, y_gen: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """generator loss."""

        self.set_discriminator_gradients(False)
        gen_loss = 0.0
        output_dict = {}

        # Mel spectrogram loss
        mel_loss = self.mel_loss(y_gen.squeeze(1), y_true.squeeze(1))

        gen_loss += mel_loss * self.lambda_mel
        output_dict["mel_loss"] = mel_loss

        # RMS loss
        # Input audio RMS per track.
        input_rms_per_track = torch.sqrt(torch.mean(y_true.squeeze(1) ** 2, dim=-1))
        input_rms_per_track_db = 20 * torch.log10(input_rms_per_track + 1e-10)

        # Generated audio RMS per track.
        rms = torch.sqrt(torch.mean(y_gen.squeeze(1) ** 2, dim=-1))
        current_rms_db = 20 * torch.log10(rms + 1e-10)

        # Difference between generated and input audio RMS per track.
        rms_diff = current_rms_db - input_rms_per_track_db
        rms_loss = torch.mean(rms_diff**2)

        gen_loss += rms_loss * self.lambda_rms
        output_dict["rms_loss"] = rms_loss

        # MPD generator loss
        p_mpd_gen = self.mpd(y_gen)
        adv_loss_list = []
        for i in range(len(p_mpd_gen)):
            adv_loss_list.append(self.gan_loss.gen_loss(p_mpd_gen[i][-1]))

        # MSD generator loss
        p_msd_gen = self.msd(y_gen)
        for i in range(len(p_msd_gen)):
            adv_loss_list.append(self.gan_loss.gen_loss(p_msd_gen[i][-1]))

        adv_loss = sum(adv_loss_list)
        gen_loss += adv_loss * self.lambda_adv

        # Feature Matching loss
        mpd_fm_loss = 0.0
        with torch.no_grad():
            p_mpd_true = self.mpd(y_true)
        for i in range(len(p_mpd_gen)):
            for j in range(len(p_mpd_gen[i]) - 1):
                mpd_fm_loss += self.fm_loss(p_mpd_gen[i][j], p_mpd_true[i][j].detach())
        gen_loss += mpd_fm_loss * self.lambda_fm

        msd_fm_loss = 0.0
        with torch.no_grad():
            p_msd_true = self.msd(y_true)
        for i in range(len(p_msd_gen)):
            for j in range(len(p_msd_gen[i]) - 1):
                msd_fm_loss += self.fm_loss(p_msd_gen[i][j], p_msd_true[i][j].detach())
        gen_loss += msd_fm_loss * self.lambda_fm

        output_dict["adv_loss"] = adv_loss
        output_dict["fm_loss"] = mpd_fm_loss + msd_fm_loss
        output_dict["gen_loss"] = gen_loss

        self.set_discriminator_gradients(True)
        return output_dict

    def training_step(self, fabric: Any, batch: dict, gradient_accumulation_steps: int):
        """GAN training step, called as forward during training."""
        y_true = batch["wav"].unsqueeze(1)
        y_gen = self.forward(batch["audio_codes"])

        # training discriminators
        disc_losses = self.compute_discriminator_loss(y_true, y_gen)
        disc_loss = disc_losses["disc_loss"]
        disc_loss_step = disc_loss / gradient_accumulation_steps
        fabric.backward(disc_loss_step)

        # training generator
        gen_losses = self.compute_generator_loss(y_true, y_gen)
        gen_loss = gen_losses["gen_loss"]
        gen_loss_step = gen_loss / gradient_accumulation_steps
        fabric.backward(gen_loss_step)

        return {
            "disc_loss": disc_loss,
            "gen_loss": gen_loss,
            "adv_loss": gen_losses["adv_loss"],
            "fm_loss": gen_losses["fm_loss"],
            "mel_loss": gen_losses["mel_loss"],
            "rms_loss": gen_losses["rms_loss"],
        }

    def validation_step(self, batch: dict):
        """validation step, same as training but no backprops and weight updates."""

        y_true = batch["wav"].unsqueeze(1)
        y_gen = self.forward(batch["audio_codes"])

        # eval discriminators
        disc_losses = self.compute_discriminator_loss(y_true, y_gen)
        disc_loss = disc_losses["disc_loss"]

        # eval generator
        gen_losses = self.compute_generator_loss(y_true, y_gen)
        gen_loss = gen_losses["gen_loss"]

        return {
            "disc_loss": disc_loss,
            "gen_loss": gen_loss,
            "adv_loss": gen_losses["adv_loss"],
            "fm_loss": gen_losses["fm_loss"],
            "mel_loss": gen_losses["mel_loss"],
            "rms_loss": gen_losses["rms_loss"],
        }

    def quality_validation(self, batch: dict):
        """compute audio for quality evaluation samples."""
        with torch.no_grad():
            y_gen = self.forward(batch["audio_codes"])

        return {"generated_wavs": y_gen}


def create(
    sample_rate: int,
    hop_length: int,
    upsample_factors: list[int] | None = None,
    kernel_sizes: list[int] | None = None,
):
    generator = Decoder(
        sample_rate=sample_rate,
        hop_length=hop_length,
        upsample_factors=upsample_factors,
        kernel_sizes=kernel_sizes,
    )

    mpd = discriminator.HiFiGANMultiPeriodDiscriminator(
        periods=[2, 3, 5, 7, 11],
        max_downsample_channels=512,
        channels=16,
        channel_increasing_factor=4,
    )

    msd = discriminator.SpecDiscriminator(
        stft_params={
            "fft_sizes": [78, 126, 206, 334, 542, 876, 1418, 2296],
            "hop_sizes": [39, 63, 103, 167, 271, 438, 709, 1148],
            "win_lengths": [78, 126, 206, 334, 542, 876, 1418, 2296],
            "window": "hann_window",
        },
        in_channels=1,
        out_channels=1,
        kernel_sizes=[5, 3],
        channels=32,
        max_downsample_channels=512,
        downsample_scales=[2, 2, 2],
        use_weight_norm=True,
    )

    # build model
    return TrainableDecoder(generator, mpd, msd)


def create_optimizer(
    model: TrainableDecoder,
    learning_rate: float,
    betas: tuple[float, float],
    weight_decay: float,
):
    """Create optimizers for generator and discriminator."""
    # in the generator, keep vq quantizer frozen.

    gen_params = itertools.chain(
        model.generator.decoder.backbone.parameters(),
        model.generator.upsampler.parameters() if model.generator.upsampler else [],
        model.generator.decoder.head.parameters(),
        model.generator.fc_post_a.parameters(),
    )
    gen_optimizer = torch.optim.AdamW(
        gen_params, lr=learning_rate, betas=betas, weight_decay=weight_decay
    )

    # discriminators.
    disc_params = itertools.chain(model.mpd.parameters(), model.msd.parameters())
    disc_optimizer = torch.optim.AdamW(
        disc_params, lr=learning_rate, betas=betas, weight_decay=weight_decay
    )
    return gen_optimizer, disc_optimizer
