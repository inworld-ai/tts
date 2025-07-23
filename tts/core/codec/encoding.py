import abc

import torch

from tts.core.codec import encoder


class AudioEncoderInterface(metaclass=abc.ABCMeta):
    """Abstract interface class for audio encoders."""

    @abc.abstractmethod
    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        """Encodes a waveform into a sequence of tokens."""
        raise NotImplementedError("Subclasses must implement this encode method.")

    @property
    @abc.abstractmethod
    def sample_rate(self) -> int:
        """Returns the input sample rate of the audio decoder."""
        raise NotImplementedError("Subclasses must implement this property.")

    @property
    @abc.abstractmethod
    def token_rate(self) -> int:
        """Returns the output token rate of the audio encoder."""
        raise NotImplementedError("Subclasses must implement this property.")


class AudioEncoder(AudioEncoderInterface):
    """Audio encoder class."""

    def __init__(self, model_path: str, device: torch.device):
        super().__init__()

        self._device = device
        self._encoder = encoder.Encoder(model_path=model_path)
        self._encoder.to(self._device)
        self._encoder.eval()

    @torch.no_grad()
    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        """Encodes a waveform into a sequence of tokens."""
        return self._encoder.encode(wav)

    @property
    def sample_rate(self) -> int:
        """Returns the input sample rate of the audio encoder."""
        return self._encoder.sample_rate

    @property
    def token_rate(self) -> int:
        """Returns the output token rate of the audio encoder."""
        return self._encoder.token_rate


class CachingAudioEncoder:
    """Encodes audios and caches the results."""

    def __init__(self, model_path: str, device: torch.device):
        super().__init__()

        self._encoder = create(model_path=model_path, device=device)
        self._prompt_encoding_cache = {}

    @torch.no_grad()
    def encode(self, prompt_id: str, prompt_wav: torch.Tensor) -> list[int]:
        if prompt_id in self._prompt_encoding_cache:
            return self._prompt_encoding_cache[prompt_id]

        codes = self._encoder.encode(prompt_wav).cpu().tolist()
        self._prompt_encoding_cache[prompt_id] = codes
        return codes


def create(
    model_path: str, device: torch.device | str | None = "cpu"
) -> AudioEncoderInterface:
    """Create audio decoder with model path and optionally a config file."""

    return AudioEncoder(model_path, device=device)
