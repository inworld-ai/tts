import torch
import torchaudio
from absl import logging

from tts.inference import quality_validation


class FixedBatchCodecValidator(quality_validation.QualityValidator):
    """
    Quality validator for Audio Codec, always validate on a
    fixed val batch to monitor the audio synthesize quality progress.
    """

    def __init__(
        self,
        batched_samples: dict[str, torch.Tensor],
        checkpointing_dir: str,
        overwrite_previous: bool | None = True,
    ) -> None:
        super().__init__()
        self.batch = batched_samples
        self._checkpointing_dir = checkpointing_dir

        # overwrite historical samples or keep all of them
        self._overwrite_previous = overwrite_previous

    def validate(self, model: torch.nn.Module, step: int | None) -> None:
        generated_wavs = model.quality_validation(self.batch)["generated_wavs"]

        for i in range(len(generated_wavs)):
            gen_wav = generated_wavs[i].detach().cpu()
            true_wav = self.batch["wav"][i].unsqueeze(0).detach().cpu()

            idf = str(i) if self._overwrite_previous else f"{i}_step_{step}"
            torchaudio.save(
                f"{self._checkpointing_dir}/generated_wav_{idf}.wav",
                gen_wav,
                model.sample_rate,
            )
            torchaudio.save(
                f"{self._checkpointing_dir}/true_wav_{idf}.wav",
                true_wav,
                model.sample_rate,
            )
            logging.info(
                f"Quality validation sample {i} saved to {self._checkpointing_dir}"
            )


def create_codec_quality_validator(
    batched_samples: dict[str, torch.Tensor], checkpointing_dir: str
):
    return FixedBatchCodecValidator(
        batched_samples=batched_samples, checkpointing_dir=checkpointing_dir
    )
