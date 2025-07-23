import json
import os
import random
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from absl import logging

from tts.data import data_sample, data_utils, tts_datasets


class CodecTrainingDataset(torch.utils.data.Dataset):
    """Implements a TTS fine-tuning dataset."""

    def __init__(
        self,
        dataset_dir: str,
        split: str,
        audio_window_size: int,
        sample_rate: int,
        minimum_data_sample_rate: int,
    ):
        dataset_name = os.path.basename(dataset_dir) + "_" + split
        samples_path = os.path.join(dataset_dir, f"{split}_samples.jsonl")
        self.samples = []
        self.sample_rate = sample_rate
        self.audio_window_size = audio_window_size
        self.minimum_data_sample_rate = minimum_data_sample_rate
        self.code_window_size = int(self.audio_window_size / self.sample_rate * 50)
        self.hop_length = self.sample_rate // 50
        self.half_hop_length = self.hop_length // 2

        self.filtered_samples_index = []
        self.original_length = 0
        with open(samples_path) as f:
            for i, line in enumerate(f):
                sp = data_sample.Sample.from_json(json.loads(line), dataset_dir)
                self.original_length += 1
                if sp.sample_rate >= self.minimum_data_sample_rate:
                    self.samples.append(sp)
                    self.filtered_samples_index.append(i)

        codes_path = os.path.join(dataset_dir, f"{split}_codes.npy")
        codes_index_path = os.path.join(dataset_dir, f"{split}_codes_index.npy")
        self.codes = np.memmap(codes_path, dtype=np.int32, mode="r")
        self.number_of_codes = self.codes.shape[0]
        self.codes_index = np.load(codes_index_path)

        self.length = len(self.filtered_samples_index)
        if self.original_length != self.codes_index.shape[0]:
            raise ValueError("The number of samples and codes indices must match!")

        logging.info(
            "Loadded dataset [%s] with [%d] samples in total, [%d] samples "
            "removed due to low sample rate.",
            dataset_name,
            self.length,
            self.original_length - self.length,
        )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # convert filtered idx to the original index
        c_idx = self.filtered_samples_index[idx]

        left_boundary = self.codes_index[c_idx]
        right_boundary = (
            self.codes_index[c_idx + 1]
            if c_idx < self.original_length - 1
            else self.number_of_codes
        )
        audio_codes = self.codes[left_boundary:right_boundary]
        audio_codes = torch.from_numpy(audio_codes).unsqueeze(0)

        # load and pad audio
        wav, _ = data_utils.load_wav(
            self.samples[idx].wav_path, target_sample_rate=self.sample_rate
        )

        wav = torch.nn.functional.pad(
            wav, (0, self.hop_length - (wav.shape[1] % self.hop_length))
        )

        # TODO: figure out whether extra padding is needed
        # wav = torch.nn.functional.pad(
        #     wav, (self.half_hop_length, self.half_hop_length))

        # repeat wav and audio_codes if it's too short
        if wav.shape[0] < self.audio_window_size:
            n_repeat = self.audio_window_size // wav.shape[1] + 1
            wav = wav.repeat(1, n_repeat)

            audio_codes = audio_codes.repeat(1, n_repeat)
            wav = wav[:, : self.audio_window_size]

            audio_codes = audio_codes[:, : self.code_window_size]

        st = random.randint(0, audio_codes.shape[1] - self.code_window_size)
        audio_codes = audio_codes[:, st : st + self.code_window_size]
        wav = wav[
            :, (st * self.hop_length) : st * self.hop_length + self.audio_window_size
        ]

        return {
            "audio_codes": audio_codes,
            "wav": wav,
            "audio_processed_sec": wav.shape[1] / self.sample_rate,
        }


def collate_fn(features: list[Any]) -> dict[str, Any]:
    """collate_fn for codec datasets."""
    audio_codes = torch.cat([f["audio_codes"] for f in features], dim=0)
    wav = torch.cat([f["wav"] for f in features], dim=0)
    audio_processed_sec = torch.tensor([f["audio_processed_sec"] for f in features])

    return {
        "audio_codes": audio_codes,
        "wav": wav,
        "audio_processed_sec": audio_processed_sec,
    }


def merge_datasets(
    weighted_datasets: dict[str, float],
    audio_window_size: int,
    split: str,
    sample_rate: int,
    minimum_data_sample_rate: int,
) -> torch.utils.data.Dataset:
    """Merges multiple datasets into a single dataset."""
    logging.info("-*-" * 10 + " %s " + "-*-" * 10, split)

    # Normalize the dataset weights for non-training splits.
    normalized_dataset_weights = dict(weighted_datasets.items())
    if split != "train":
        max_dataset_weight = max(normalized_dataset_weights.values())
        normalized_dataset_weights = {
            dataset_path: dataset_weight / max_dataset_weight
            for dataset_path, dataset_weight in normalized_dataset_weights.items()
        }

    datasets = []
    for dataset_path in weighted_datasets:
        dataset_name = os.path.basename(dataset_path)
        dataset = CodecTrainingDataset(
            dataset_path,
            split,
            audio_window_size,
            sample_rate,
            minimum_data_sample_rate,
        )
        datasets.append(
            tts_datasets.WeightedDataset(
                name=dataset_name,
                dataset=dataset,
                epochs=normalized_dataset_weights[dataset_path],
            )
        )
        logging.info(
            f"Dataset [{dataset_name}] has [{len(dataset)}] samples and "
            f"epochs [{normalized_dataset_weights[dataset_path]:.1f}]."
        )

    merged_dataset = tts_datasets.CombinedDataset(datasets)
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
    # pin_memory must be False to avoid error.
    # num_workers must be > 0 for normal audio loading speed.
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
