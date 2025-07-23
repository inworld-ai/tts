from typing import Any

import datasets
import numpy as np
import torch
import transformers
from absl import logging

from tts.core import constants, prompting
from tts.data import data_sample, text_normalization


class TtsFineTuningDataset(torch.utils.data.Dataset):
    """Implements a TTS fine-tuning dataset."""

    def __init__(
        self,
        dataset_name: str,
        samples: list[data_sample.Sample],
        codes: np.ndarray,
        indexes: list[tuple[int, int]],
        tokenizer: transformers.PreTrainedTokenizer,
        max_seq_len: int,
        prompt_compiler: prompting.PromptCompiler,
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
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.speech_start_id = tokenizer.convert_tokens_to_ids(
            constants.SPEECH_START_TOKEN
        )
        self.speech_end_id = tokenizer.convert_tokens_to_ids(constants.SPEECH_END_TOKEN)
        logging.info(
            "Loadded [%s]-dataset with [%d] samples / [%d] codes. "
            "Pad/speech start/speed end tokens: [%d, %d, %d]",
            dataset_name,
            self.length,
            self.number_of_codes,
            self.pad_token_id,
            self.speech_start_id,
            self.speech_end_id,
        )
        self.prompt_compiler = prompt_compiler

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        speech_ids = self.codes[self.indexes[idx][0] : self.indexes[idx][1]]
        sample: data_sample.Sample = self.samples[idx]
        transcript = self._text_normalizer.normalize_with_language(
            sample.transcript, sample.language
        )
        prompt = self.prompt_compiler.compile_prompt(
            audio_prompt_transcription=transcript,
            text_to_synthesize="",
            speech_ids=speech_ids,
            voice_description=sample.voice_description,
        )

        input_ids = self.tokenizer(
            prompt, add_special_tokens=True, return_tensors="pt"
        )["input_ids"][0]
        input_ids = input_ids[: self.max_seq_len]

        try:
            separator_idx = (
                (input_ids == self.speech_start_id).nonzero(as_tuple=True)[0].item()
            )
        except Exception:  # pylint: disable=broad-except
            logging.warning("Speech start not found in the input ids.")
            separator_idx = -1

        num_tokens_to_generate = speech_ids.shape[0]
        audio_processed_sec = num_tokens_to_generate / constants.CODEC_TOKENS_RATE

        # Create the labels and attention mask by masking out all the tokens before
        # the separator token as for traditional supervised LLM fine-tuning.
        labels = torch.full_like(input_ids, constants.LOSS_IGNORE_TOKEN_ID)
        if separator_idx != -1:
            labels[separator_idx:] = input_ids[separator_idx:]
        labels[input_ids == self.pad_token_id] = constants.LOSS_IGNORE_TOKEN_ID
        attention_mask = (input_ids != self.pad_token_id).long()

        result = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "tokens_processed": input_ids.shape[0],
            "generated_audio_duration_sec": num_tokens_to_generate
            / constants.CODEC_TOKENS_RATE,
            "audio_processed_sec": audio_processed_sec,
        }

        return result


class TextFineTuningDataset(torch.utils.data.Dataset):
    """Implements a text fine-tuning dataset. Only supports LLaMA 3.1 models."""

    def __init__(
        self,
        dataset: datasets.Dataset,
        tokenizer: transformers.PreTrainedTokenizer,
        max_seq_len: int,
    ):
        self._dataset = dataset
        self._tokenizer = tokenizer
        self._max_seq_len = max_seq_len
        self._end_header_id = tokenizer.convert_tokens_to_ids(constants.END_HEADER_ID)

    def __len__(self) -> int:
        return len(self._dataset)

    def _parse_oig_sample(self, raw_text: str) -> list[dict[str, str]]:
        """Parses a sample from the OIG dataset into a list of message objects."""
        messages = []
        raw_text = raw_text.strip()
        if not raw_text.startswith("<human>:"):
            raise ValueError("Sample does not start with <human>:")

        parts = raw_text.split("<human>:")
        parts = parts[1:] if parts else []
        for part in parts:
            if "<bot>:" in part:
                human_text, bot_part = part.split("<bot>:", 1)
                messages.append({"role": "user", "content": human_text.strip()})

                if "<human>:" in bot_part:
                    bot_text = bot_part.split("<human>:", 1)[0].strip()
                else:
                    bot_text = bot_part.strip()

                messages.append({"role": "assistant", "content": bot_text})
            else:
                messages.append({"role": "user", "content": part.strip()})
                break

        return messages

    def _compile_messages(self, sample: Any) -> list[Any]:
        if "messages" in sample:
            return sample["messages"]
        return self._parse_oig_sample(sample["text"])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        messages = self._compile_messages(self._dataset[idx])
        input_ids = self._tokenizer.apply_chat_template(
            messages, tokenize=True, return_tensors="pt"
        ).squeeze()

        try:
            right_most_end_header_id = (
                (input_ids == self._end_header_id).nonzero()[-1].item()
            )
        except Exception:  # pylint: disable=broad-except
            right_most_end_header_id = input_ids.shape[0] - 1
        response_start_idx = min(right_most_end_header_id + 1, self._max_seq_len - 1)

        # Ignore loss on all the non-assistant final response tokens.
        input_ids = input_ids[: self._max_seq_len]
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        labels[:response_start_idx] = constants.LOSS_IGNORE_TOKEN_ID

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "tokens_processed": len(input_ids),
            "generated_audio_duration_sec": 0.0,
            "audio_processed_sec": 0.0,
        }
