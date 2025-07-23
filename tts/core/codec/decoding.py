"""Decoding tokens to audio."""

import abc
import dataclasses
import json
import os

import torch

from tts.core.codec import decoder


@dataclasses.dataclass(frozen=True)
class DecoderConfig:
    """Model config for the codec decoder model."""

    model_type: str
    sample_rate: int
    token_rate: int
    hop_length: int
    upsample_factors: list[int]
    kernel_sizes: list[int]

    @staticmethod
    def from_json(file: str | os.PathLike) -> "DecoderConfig":
        with open(file) as f:
            config = json.load(f)
        return DecoderConfig(
            model_type=config["model_type"],
            sample_rate=config["sample_rate"],
            token_rate=config["token_rate"],
            hop_length=config["hop_length"],
            upsample_factors=config["upsample_factors"],
            kernel_sizes=config["kernel_sizes"],
        )


class AudioDecoderInterface(metaclass=abc.ABCMeta):
    """Abstract interface class for audio decoders."""

    @abc.abstractmethod
    def decode(self, speech_ids: torch.Tensor) -> torch.Tensor:
        """Decodes a batch of speech IDs into audio waveforms."""
        raise NotImplementedError("Subclasses must implement this decode method.")

    @property
    @abc.abstractmethod
    def sample_rate(self) -> int:
        """Returns the sample rate of the audio decoder."""
        raise NotImplementedError("Subclasses must implement this property.")

    @property
    @abc.abstractmethod
    def token_rate(self) -> int:
        """Returns the input token rate of the audio decoder."""
        raise NotImplementedError("Subclasses must implement this property.")


class AudioDecoder(AudioDecoderInterface):
    """A wrapper around the audio decoder model for batch decoding."""

    def __init__(
        self,
        model_path: str,
        config: DecoderConfig,
        device: torch.device | str | None = "cpu",
    ):
        super().__init__()

        self._device = device
        self._decoder = decoder.Decoder(
            sample_rate=config.sample_rate,
            hop_length=config.hop_length,
            upsample_factors=config.upsample_factors,
            kernel_sizes=config.kernel_sizes,
            checkpoint_path=model_path,
        )
        self._decoder.to(self._device)
        self._decoder.eval()

        self._sample_rate = config.sample_rate
        self._token_rate = config.token_rate

    @torch.no_grad()
    def decode(self, speech_ids: torch.Tensor) -> torch.Tensor:
        """Decodes a batch of speech ids into audio waveforms."""
        vq_codes = speech_ids.unsqueeze(0).unsqueeze(0)
        vq_codes = vq_codes.to(self._device)
        return self._decoder(vq_codes).detach().cpu().squeeze(0)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def token_rate(self) -> int:
        return self._token_rate


def create(
    model_path: str, device: torch.device | str | None = "cpu"
) -> AudioDecoderInterface:
    """Create audio decoder with model path and optionally a config file."""

    ckpt_dir = os.path.dirname(model_path)
    config_path = os.path.join(ckpt_dir, "model_config.json")

    if not os.path.exists(config_path):
        raise ValueError("No model_config.json found in the provided path.")

    model_config = DecoderConfig.from_json(config_path)
    return AudioDecoder(model_path, model_config, device=device)
