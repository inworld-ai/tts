import numpy as np
import torch
import transformers
from absl import logging

from tts.core import constants, prompting
from tts.data import data_sample, text_normalization


class TtsRLHFDataset(torch.utils.data.Dataset):
    """Implements a TTS RLHF dataset."""

    def __init__(
        self,
        dataset_name: str,
        samples: list[data_sample.Sample],
        codes: np.ndarray,
        indexes: list[tuple[int, int]],
        tokenizer: transformers.PreTrainedTokenizer,
        prompt_compiler: prompting.PromptCompiler,  # should use InferencePromptCompiler
        text_normalizer: text_normalization.TextNormalizer,
    ):
        self.dataset_name = dataset_name
        self.samples = samples
        self.codes = codes
        self.indexes = indexes
        self.number_of_codes = sum(end - start for start, end in self.indexes)
        self.length = len(self.samples)
        self._text_normalizer = text_normalizer

        if len(self.indexes) != self.length:
            raise ValueError("The number of samples and codes must match!")
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.speech_start_id = tokenizer.convert_tokens_to_ids(
            constants.SPEECH_START_TOKEN
        )
        self.speech_end_id = tokenizer.convert_tokens_to_ids(constants.SPEECH_END_TOKEN)
        logging.info(
            "Loadded RLHF [%s]-dataset with [%d] samples / [%d] codes.",
            dataset_name,
            self.length,
            self.number_of_codes,
        )
        self.prompt_compiler = prompt_compiler

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        speech_ids = self.codes[self.indexes[idx][0] : self.indexes[idx][1]]
        sample: data_sample.Sample = self.samples[idx]
        sample.transcript = self._text_normalizer.normalize_with_language(
            sample.transcript, sample.language
        )
        # use the next sample as the text to synthesize
        next_idx = (idx + 1) % self.length
        next_sample: data_sample.Sample = self.samples[next_idx]
        text_to_synthesize = self._text_normalizer.normalize_with_language(
            next_sample.transcript, next_sample.language
        )
        prompt = self.prompt_compiler.compile_prompt(
            audio_prompt_transcription=sample.transcript,
            text_to_synthesize=text_to_synthesize,
            speech_ids=speech_ids,
            voice_description="",
        )

        result = {
            "prompt": prompt,
            "prompt_speech_ids": torch.tensor(speech_ids),
            "completion_truth": text_to_synthesize,
            "prompt_wav_path": sample.wav_path,
        }

        return result
