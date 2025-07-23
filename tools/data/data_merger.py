"""Merges shards into a single dataset for the actual use by trainer."""

import dataclasses
import fnmatch
import json
import os
import time
from typing import List, Sequence, Tuple

import numpy as np
from absl import app, flags, logging

from tts.core import constants

FLAGS = flags.FLAGS

_DATASET_PATH = flags.DEFINE_string("dataset_path", None, "Path to the dataset")
_REMOVE_SHARDS = flags.DEFINE_boolean("remove_shards", False,
                                      "Whether to remove shards.")
_ONLY_GLUE_SAMPLES = flags.DEFINE_boolean("only_glue_samples", False,
                                          "Whether to only glue raw samples.")


@dataclasses.dataclass(frozen=True)
class Shard:
    id: int
    split: str

    codes_path: str
    codes_index_path: str
    samples_path: str


def sort_filenames(filenames: List[str]) -> List[str]:
    """Sorts the filenames by the shard ID."""

    def _get_shard_id(filename: str) -> int:
        shard_id, _ = filename.split('.')
        return int(shard_id.split('_')[-1])

    return sorted(filenames, key=_get_shard_id)


def read_and_glue_samples(dataset_path: str) -> dict:
    """Reads and glues samples from the dataset."""
    sample_files = [
        f for f in os.listdir(dataset_path) if fnmatch.fnmatch(f, "samples_*.jsonl")
    ]
    logging.info("Found %d sample files in %s", len(sample_files), dataset_path)

    all_samples = []
    for sample_file in sample_files:
        with open(os.path.join(dataset_path, sample_file), "r", encoding="utf-8") as f:
            for line in f:
                all_samples.append(json.loads(line))

    logging.info("Found %d samples in %s.", len(all_samples), dataset_path)
    with open(os.path.join(dataset_path, "samples.jsonl"), "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + "\n")


def read_the_directory(dataset_path: str) -> tuple[dict, dict, dict]:
    """Validates the dataset path."""
    files = os.listdir(dataset_path)
    if not files:
        raise ValueError(f"Dataset path {dataset_path} is empty")

    num_shards = 0
    codes, codes_index, samples = {}, {}, {}
    for split in ['train', 'val']:
        codes[split] = sort_filenames(
            [f for f in files if fnmatch.fnmatch(f, f'{split}_codes_[0-9]*.npy')])
        codes_index[split] = sort_filenames(
            [f for f in files if fnmatch.fnmatch(f, f'{split}_codes_index_*.npy')])
        samples[split] = sort_filenames(
            [f for f in files if fnmatch.fnmatch(f, f'{split}_samples_*.jsonl')])

        number_of_codes = len(codes[split])
        number_of_codes_index = len(codes_index[split])
        number_of_samples = len(samples[split])
        if number_of_codes != number_of_codes_index:
            raise ValueError(
                "Number of codes [{}] does not match number of codes_index [{}]".format(
                    number_of_codes, number_of_codes_index))

        if number_of_samples != number_of_codes:
            raise ValueError(
                "Number of samples [{}] does not match number of codes [{}]".format(
                    number_of_samples, number_of_codes))

        if not num_shards:
            num_shards = number_of_samples
        else:
            if num_shards != number_of_samples:
                raise ValueError("Number of train/val shards  does not match!")

    logging.info("Found %d valid shard(s) in %s.", num_shards, dataset_path)
    return codes, codes_index, samples


def create_shards(dataset_path: str, codes: dict, codes_index: dict,
                  samples: dict) -> Tuple[List[Shard], List[Shard]]:
    """Creates shards for train and val splits."""
    shards = []
    for split in ['train', 'val']:
        for shard_id in range(len(codes[split])):
            codes_path = os.path.join(dataset_path, codes[split][shard_id])
            codes_index_path = os.path.join(dataset_path, codes_index[split][shard_id])
            samples_path = os.path.join(dataset_path, samples[split][shard_id])

            # Double check that file names are correct.
            codes_path_shard_id, _ = codes_path.split('.')
            codes_index_path_shard_id, _ = codes_index_path.split('.')
            samples_path_shard_id, _ = samples_path.split('.')
            codes_path_shard_id = int(codes_path_shard_id.split('_')[-1])
            codes_index_path_shard_id = int(codes_index_path_shard_id.split('_')[-1])
            samples_path_shard_id = int(samples_path_shard_id.split('_')[-1])
            if codes_path_shard_id != shard_id:
                raise ValueError(
                    "Shard ID mismatch in codes_path [{}] and shard_id [{}]".format(
                        codes_path_shard_id, shard_id))
            if codes_index_path_shard_id != shard_id:
                raise ValueError(
                    "Shard ID mismatch in codes_index_path [{}] and shard_id [{}]".
                    format(codes_index_path_shard_id, shard_id))
            if samples_path_shard_id != shard_id:
                raise ValueError(
                    "Shard ID mismatch in samples_path [{}] and shard_id [{}]".format(
                        samples_path_shard_id, shard_id))

            shards.append(
                Shard(id=shard_id,
                      split=split,
                      codes_path=codes_path,
                      codes_index_path=codes_index_path,
                      samples_path=samples_path))

    train_shards = [shard for shard in shards if shard.split == 'train']
    val_shards = [shard for shard in shards if shard.split == 'val']

    if len(train_shards) != len(val_shards):
        raise ValueError(
            "Number of train shards [{}] does not match number of val shards [{}]".
            format(len(train_shards), len(val_shards)))

    return train_shards, val_shards


def merge_shards(shards: List[Shard], output_path: str, split: str):
    """Merges a list of shard files into a single set of three files."""
    logging.info("Merging %d %s-shards into %s", len(shards), split, output_path)
    start_time = time.time()

    current_offset = 0
    all_codes, all_codes_index, all_samples = [], [], []
    for shard in shards:
        shard_codes = np.memmap(shard.codes_path, dtype=np.int32, mode="r")
        shard_index = np.load(shard.codes_index_path)  # shape: [num_samples_in_shard]

        shard_samples = []
        with open(shard.samples_path, "r", encoding="utf-8") as sf:
            for line in sf:
                shard_samples.append(line.rstrip("\n"))

        if len(shard_index) != len(shard_samples):
            raise ValueError("Shard {} mismatch: codes_index has {} entries "
                             "but samples file has {} lines.".format(
                                 shard.id, len(shard_index), len(shard_samples)))

        shard_index += current_offset  # shift the indices to the 'right'
        all_codes.append(shard_codes.copy())
        all_codes_index.append(shard_index)
        all_samples.extend(shard_samples)

        shard_length = shard_codes.shape[0]
        current_offset += shard_length

        logging.info("Shard %d has %d/%d codes/examples", shard.id, shard_length,
                     len(shard_index))

    total_examples = sum(len(x) for x in all_codes_index)
    if len(all_samples) != total_examples:
        raise ValueError(
            "Number of samples [{}] does not match number of codes_index [{}]. "
            "Probably the shards are not merged correctly.".format(
                len(all_samples), total_examples))

    logging.info("Concatenating final arrays for split=%s...", split)
    merged_codes = np.concatenate(all_codes, axis=0)
    merged_index = np.concatenate(all_codes_index, axis=0)

    merged_codes_path = os.path.join(output_path, f"{split}_codes.npy")
    merged_codes_index_path = os.path.join(output_path, f"{split}_codes_index.npy")
    merged_samples_path = os.path.join(output_path, f"{split}_samples.jsonl")

    total_number_of_codes = merged_codes.shape[0]
    merged_codes_arr = np.memmap(merged_codes_path,
                                 dtype=np.int32,
                                 mode="w+",
                                 shape=(total_number_of_codes,))
    merged_codes_arr[:] = merged_codes
    merged_codes_arr.flush()
    np.save(merged_codes_index_path, merged_index, allow_pickle=False)

    with open(merged_samples_path, "w", encoding="utf-8") as f_out:
        for line in all_samples:
            f_out.write(line + "\n")

    logging.info(
        "Done merging %d shards into: %s for split=%s in %.2f seconds. "
        "Total codes: %d, total samples: %d, total hours: %.2f", len(shards),
        merged_codes_path, split,
        time.time() - start_time, total_number_of_codes, total_examples,
        total_number_of_codes / (3600.0 * constants.CODEC_TOKENS_RATE))


def validate_merged_shards(dataset_path: str, split: str) -> None:
    codes_path = os.path.join(dataset_path, f"{split}_codes.npy")
    codes_index_path = os.path.join(dataset_path, f"{split}_codes_index.npy")
    samples_path = os.path.join(dataset_path, f"{split}_samples.jsonl")

    codes = np.memmap(codes_path, dtype=np.int32, mode="r")
    codes_index = np.load(codes_index_path)
    samples = []

    with open(samples_path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(line.rstrip("\n"))

    if len(samples) != len(codes_index):
        raise ValueError(
            "Number of samples [{}] does not match number of codes_index [{}]. "
            "Probably the shards are not merged correctly.".format(
                len(samples), len(codes_index)))

    # Perform traversal and confirm codes are contiguous.
    for i in range(len(samples)):
        try:
            if i == len(samples) - 1:
                _ = codes[codes_index[i]:]
            else:
                _ = codes[codes_index[i]:codes_index[i + 1]]
        except Exception as e:
            raise ValueError("Codes are malformed [{}] at index [{}]. Probably the "
                             "shards are not merged correctly.".format(e, i))


def main(argv: Sequence[str]) -> None:
    del argv  # Unused.

    dataset_path = _DATASET_PATH.value
    if _ONLY_GLUE_SAMPLES.value:
        read_and_glue_samples(dataset_path)
        return

    codes, codes_index, samples = read_the_directory(dataset_path)
    train_shards, val_shards = create_shards(dataset_path, codes, codes_index, samples)

    merge_shards(train_shards, dataset_path, "train")
    validate_merged_shards(dataset_path, "train")

    merge_shards(val_shards, dataset_path, "val")
    validate_merged_shards(dataset_path, "val")

    if _REMOVE_SHARDS.value:
        for shard in train_shards + val_shards:
            os.remove(shard.codes_path)
            os.remove(shard.codes_index_path)
            os.remove(shard.samples_path)
        logging.info("Removed %d shards from %s",
                     len(train_shards) + len(val_shards), dataset_path)


if __name__ == "__main__":
    flags.mark_flags_as_required(["dataset_path"])
    app.run(main)
