import dataclasses
import math
import os
from collections.abc import Callable
from typing import Any

import datasets
import torch
import transformers
from absl import logging

from tts.core import constants, prompting
from tts.data import data_utils, text_normalization
from tts.data.datasets import finetuning, pretraining, rlhf
from tts.utils import configuration


def _get_sft_dataset(dataset_path: str, split: str) -> datasets.Dataset:
    """Handles creation of a text fine-tuning dataset from a dataset path."""
    dataset = datasets.load_from_disk(dataset_path.replace("[text]", ""))
    subset = split if split == constants.TRAIN_SPLIT else "test"
    return dataset.train_test_split(test_size=0.001)[subset]


def _build_dataset(
    *,
    tokenizer: transformers.PreTrainedTokenizer,
    dataset_path: str,
    max_seq_len: int,
    split: str,
    pretraining_mode: bool,
    text_normalizer: text_normalization.TextNormalizer,
    dataset_config: configuration.DatasetConfig,
) -> tuple[torch.utils.data.Dataset, str]:
    """Builds a dataset from a dataset path."""
    dataset_name = os.path.basename(dataset_path)
    text_dataset = dataset_name.endswith("[text]")
    if pretraining_mode:
        if text_dataset:
            dataset_dir = dataset_path.replace("[text]", "")
            return pretraining.TextPretrainingDataset(
                dataset_dir=dataset_dir, split=split, max_seq_len=max_seq_len
            ), dataset_name
        return pretraining.TtsPretrainingDataset(
            dataset_dir=dataset_path,
            split=split,
            max_seq_len=max_seq_len,
            tokenizer=tokenizer,
        ), dataset_name
    if text_dataset:
        dataset = _get_sft_dataset(dataset_path, split)
        return finetuning.TextFineTuningDataset(
            dataset, tokenizer, max_seq_len
        ), dataset_name

    codes, samples, indexes, _ = data_utils.load_and_filter_audio_codes_and_samples(
        dataset_dir=dataset_path, split=split, dataset_config=dataset_config
    )

    if dataset_config.enable_rlhf_training:
        return rlhf.TtsRLHFDataset(
            dataset_name=dataset_name,
            samples=samples,
            codes=codes,
            indexes=indexes,
            tokenizer=tokenizer,
            prompt_compiler=prompting.InferencePromptCompiler(),
            text_normalizer=text_normalizer,
        ), dataset_name
    return finetuning.TtsFineTuningDataset(
        dataset_name=dataset_name,
        samples=samples,
        codes=codes,
        indexes=indexes,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        prompt_compiler=prompting.TrainingPromptCompiler(),
        text_normalizer=text_normalizer,
    ), dataset_name


@dataclasses.dataclass
class WeightedDataset:
    """A torch dataset with a weight.

    Attrs:
        name: Name of the dataset.
        dataset: The dataset instance.
        epochs: The number of epochs to run on the dataset.
    """

    name: str
    dataset: torch.utils.data.Dataset
    epochs: float


class CombinedDataset(torch.utils.data.Dataset):
    """A dataset that combines multiple datasets.

    Datasets are virtually arranged as (being already weighted):

    | < -- dataset_1 -- > | < ----- dataset_2 ---- > | ... | < dataset_n > |
    | < ------------ effective_total_number_samples -------------------- > |
                                              ^
                                              |
    |--------------------------------------->idx
                          |-------------->relative_idx (might need an offset)

    Their total length is given by the sum of the effective lengths
    which is |self._effective_total_number_samples|.

    Note: this class always returns batches with all samples having same context length,
    a client can choose to trim it based on usage,
    (see curriculum learning, for example).

    Use enable/disable fast forwarding methods to avoid doing any data processing
    on the samples which have to be skipped. WARNING: this isn't a thread safe method!
    """

    def __init__(self, weighted_datasets: list[WeightedDataset]):
        self._original_lengths, self._effective_lengths = [], []
        self._datasets = sorted(weighted_datasets, key=lambda x: x.name)
        for weighted_dataset in self._datasets:
            num_samples = len(weighted_dataset.dataset)
            self._original_lengths.append(num_samples)
            self._effective_lengths.append(
                math.floor(num_samples * weighted_dataset.epochs)
            )

        self._effective_total_number_samples = sum(self._effective_lengths)
        self._sources = [weighted_dataset.name for weighted_dataset in self._datasets]
        self._fast_forward_mode = False

    @property
    def sources(self) -> list[str]:
        return self._sources

    def enable_fast_forwarding(self):
        self._fast_forward_mode = True

    def disable_fast_forwarding(self):
        self._fast_forward_mode = False

    def __len__(self) -> int:
        return self._effective_total_number_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self._fast_forward_mode:
            return {}

        # |idx| ranges from 0 to |self._effective_total_number_samples| - 1.
        if idx >= self._effective_total_number_samples or idx < 0:
            raise IndexError(f"Index {idx} is out of range.")

        # Map |idx| to the corresponding dataset.
        dataset_idx = 0
        relative_idx = idx
        while relative_idx >= self._effective_lengths[dataset_idx]:
            relative_idx -= self._effective_lengths[dataset_idx]
            dataset_idx += 1

        epoch_offseted_relative_idx = relative_idx % self._original_lengths[dataset_idx]
        weighted_dataset = self._datasets[dataset_idx]
        sample = weighted_dataset.dataset[epoch_offseted_relative_idx]
        sample["source"] = weighted_dataset.name
        return sample


def get_collate_fn(pad_token_id: int) -> Callable:
    """Returns a collate function for the dataset."""

    # NOTE: the reason why we don't pad everything to max sequence length is to
    #       save on processing unneeded tokens. Each batch gets on average shorter
    #       thereby increasing the training speed.
    def collate_fn(features: list[Any]) -> dict[str, Any]:
        if sum(len(x) for x in features) == 0:
            return {}

        input_ids = torch.nn.utils.rnn.pad_sequence(
            [f["input_ids"] for f in features],
            batch_first=True,
            padding_value=pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [f["labels"] for f in features],
            batch_first=True,
            padding_value=constants.LOSS_IGNORE_TOKEN_ID,
        )

        attention_mask = None
        if "attention_mask" in next(iter(features)):
            attention_mask = torch.nn.utils.rnn.pad_sequence(
                [f["attention_mask"] for f in features],
                batch_first=True,
                padding_value=0,
            )

        generated_audio_duration_sec = None
        if "generated_audio_duration_sec" in next(iter(features)):
            generated_audio_duration_sec = [
                f["generated_audio_duration_sec"] for f in features
            ]

        tokens_processed = torch.tensor([f["tokens_processed"] for f in features])
        audio_processed_sec = torch.tensor([f["audio_processed_sec"] for f in features])

        result = {
            "source": [f["source"] for f in features],
            "input_ids": input_ids,
            "labels": labels,
            "tokens_processed": tokens_processed,
            "audio_processed_sec": audio_processed_sec,
        }
        if attention_mask is not None:
            result["attention_mask"] = attention_mask
        if generated_audio_duration_sec is not None:
            result["generated_audio_duration_sec"] = torch.tensor(
                generated_audio_duration_sec
            )

        return result

    return collate_fn


def merge_datasets(
    tokenizer: transformers.PreTrainedTokenizer,
    weighted_datasets: dict[str, float],
    max_seq_len: int,
    split: str,
    pretraining_mode: bool,
    text_normalizer: text_normalization.TextNormalizer,
    dataset_config: configuration.DatasetConfig,
) -> torch.utils.data.Dataset:
    """Merges multiple datasets into a single dataset."""

    logging.info("-*-" * 10 + " %s " + "-*-" * 10, split)
    datasets = []
    for dataset_path, dataset_weight in weighted_datasets.items():
        dataset, dataset_name = _build_dataset(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            max_seq_len=max_seq_len,
            split=split,
            pretraining_mode=pretraining_mode,
            text_normalizer=text_normalizer,
            dataset_config=dataset_config,
        )
        datasets.append(
            WeightedDataset(name=dataset_name, dataset=dataset, epochs=dataset_weight)
        )
        logging.info(
            f"Dataset [{dataset_name}] has [{len(dataset)}] samples and "
            f"epochs [{dataset_weight:.2f}]."
        )

    merged_dataset = CombinedDataset(datasets)
    if len(datasets) > 1:
        logging.info(
            "Merged dataset has %d sources and %d samples total.",
            len(merged_dataset.sources),
            len(merged_dataset),
        )
    logging.info("-*-" * (18 + len(split)))
    return merged_dataset


def get_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    collate_fn: Callable,
    shuffle: bool,
    num_workers: int,
) -> torch.utils.data.DataLoader:
    """Creates a dataloader for the dataset."""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )


# TODO: find a more elegant solution to avoid using this.
def prettify_data_sample(data_sample: dict[str, Any]) -> dict[str, Any]:
    """Removes finetuning-specific fields from the data sample."""
    for field in [
        "tokens_processed",
        "generated_audio_duration_sec",
        "audio_processed_sec",
        "source",
    ]:
        if field in data_sample:
            data_sample.pop(field)
    return data_sample
