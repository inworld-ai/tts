import os
from collections.abc import Sequence

import numpy as np
import torch
import transformers
from absl import logging

from tts.core import constants

# Memmap is recreated after `SAMPLES_TO_RECREATE` getitem invokes to free RAM.
_SAMPLES_TO_RECREATE = 10**5


class TtsPretrainingDataset(torch.utils.data.Dataset):
    """Implements a TTS pretraining dataset."""

    def __init__(
        self,
        dataset_dir: str,
        split: str,
        max_seq_len: int,
        tokenizer: transformers.PreTrainedTokenizer,
    ):
        codes_path = os.path.join(dataset_dir, f"{split}_pretraining_codes.npy")
        self._codes_path = codes_path
        self._codes = np.memmap(self._codes_path, dtype=np.int32, mode="r")
        self._max_seq_len = max_seq_len
        self._num_codes = len(self._codes)
        if self._num_codes < self._max_seq_len:
            raise ValueError(
                f"Dataset [{codes_path}] size [{self._num_codes}] is too small for "
                f"the requested max sequence length [{self._max_seq_len}]."
            )
        self._samples_gave = 0
        self._tokenizer_vocab = tokenizer.vocab

        logging.info("Ready to use %d samples from %s.", len(self), self._codes_path)

    def __len__(self) -> int:
        return self._num_codes // self._max_seq_len - 1

    def _convert_codes_to_speech_tokens(self, codes: Sequence[int]) -> torch.Tensor:
        speech_tokens = []
        for code in codes:
            token = constants.SPEECH_TOKEN_PATTERN.format(code)
            speech_tokens.append(self._tokenizer_vocab.get(token, code))
        return torch.tensor(speech_tokens)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        self._samples_gave += 1
        if (self._samples_gave % _SAMPLES_TO_RECREATE) == 0:
            self._codes = np.memmap(self._codes_path, dtype=np.int32, mode="r")
            logging.info("Recreated memmap for [%s] dataset.", self._codes_path)

        index = idx * self._max_seq_len
        input_ids = self._codes[index : index + self._max_seq_len]
        input_ids = self._convert_codes_to_speech_tokens(input_ids)
        labels = input_ids.clone()

        audio_processed_sec = self._max_seq_len / constants.CODEC_TOKENS_RATE
        return {
            "input_ids": input_ids,
            "labels": labels,
            "tokens_processed": self._max_seq_len,
            "generated_audio_duration_sec": audio_processed_sec,
            "audio_processed_sec": audio_processed_sec,
        }


class TextPretrainingDataset(torch.utils.data.Dataset):
    """Implements a text pretraining dataset."""

    def __init__(self, dataset_dir: str, split: str, max_seq_len: int):
        tokens_path = os.path.join(dataset_dir, f"{split}_pretraining_tokens.npy")
        self._tokens_path = tokens_path
        self._tokens = np.memmap(self._tokens_path, dtype=np.int32, mode="r")
        self._max_seq_len = max_seq_len
        self._num_tokens = len(self._tokens)
        if self._num_tokens < self._max_seq_len:
            raise ValueError(
                f"Text dataset [{tokens_path}] size [{self._num_tokens}] is too small "
                f"for the requested max sequence length [{self._max_seq_len}]."
            )
        self._samples_gave = 0

        logging.info("Ready to use %d samples from %s.", len(self), self._tokens_path)

    def __len__(self) -> int:
        return self._num_tokens // self._max_seq_len - 1

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        self._samples_gave += 1
        if (self._samples_gave % _SAMPLES_TO_RECREATE) == 0:
            self._tokens = np.memmap(self._tokens_path, dtype=np.int32, mode="r")
            logging.info("Recreated memmap for [%s] dataset.", self._tokens_path)

        index = idx * self._max_seq_len
        input_ids = torch.tensor(
            self._tokens[index : index + self._max_seq_len], dtype=torch.long
        )
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "labels": labels,
            "tokens_processed": self._max_seq_len,
            "generated_audio_duration_sec": 0.0,
            "audio_processed_sec": 0.0,
        }
