import collections
import gc
import json
import os
import time
from typing import Any

import numpy as np
import torch
import torchaudio
from absl import logging

from tts.data import data_sample, filtering
from tts.utils import configuration


def chunk_work(work_items: list[Any], worker_id: int, num_workers: int) -> list[Any]:
    """Cuts the global work_items into a chunk for this worker_id."""
    if num_workers <= 1:
        return work_items

    total = len(work_items)
    chunk_size = total // num_workers
    start_idx = worker_id * chunk_size
    end_idx = total if (worker_id == num_workers - 1) else (start_idx + chunk_size)
    chunk = work_items[start_idx:end_idx]
    logging.info(
        "Worker %d will process indices [%d, %d) => %d items.",
        worker_id,
        start_idx,
        end_idx,
        len(chunk),
    )
    return chunk


def load_samples(
    dataset_path: str, max_samples: int = -1
) -> tuple[list[data_sample.Sample], float]:
    """Loads samples from a dataset."""
    filters = [
        filtering.filter_empty_transcript,
        filtering.filter_non_english,
        filtering.filter_long_duration,
        filtering.filter_punct_or_space_only_transcript,
    ]

    if not dataset_path.endswith(".jsonl"):
        raise ValueError("Dataset path must end with .jsonl")

    dataset_name = os.path.basename(os.path.dirname(dataset_path))
    samples = []
    start_time = time.time()
    logging.info("Starting to load dataset %s", dataset_path)

    num_filtered = 0
    durations = []
    total_duration = 0.0
    with open(dataset_path, encoding="utf-8") as file:
        for line in file:
            if max_samples > 0 and len(samples) >= max_samples:
                break

            sample = data_sample.Sample.from_json(json.loads(line), dataset_name)
            # Short-circuit filtering: stop at first filter that applies
            filter_reason = None
            for f in filters:
                filter_reason = f(sample)
                if filter_reason:
                    num_filtered += 1
                    break
            if filter_reason:
                continue

            samples.append(sample)
            durations.append(sample.duration)
            total_duration += sample.duration
            if samples and len(samples) % 100000 == 0:
                logging.info(
                    "Loaded %d sample metadatas. Average sample duration: %.2f seconds",
                    len(samples),
                    np.mean(durations),
                )
                durations = []

    logging.info(
        "Loaded %d samples from %s in %.2fsec. Total audio duration: %.2f hours. "
        "Filtered %d samples.",
        len(samples),
        dataset_path,
        time.time() - start_time,
        total_duration / 3600.0,
        num_filtered,
    )
    return samples, total_duration


def load_and_filter_audio_codes_and_samples(
    dataset_dir: str, split: str, dataset_config: configuration.DatasetConfig
) -> tuple[np.ndarray, list[data_sample.Sample], list[tuple[int, int]], dict[str, int]]:
    """Loads samples from a dataset and filters them along with audio codes."""
    dataset_name = os.path.basename(dataset_dir) + "_" + split
    samples_path = os.path.join(dataset_dir, f"{split}_samples.jsonl")
    codes_path = os.path.join(dataset_dir, f"{split}_codes.npy")
    codes_index_path = os.path.join(dataset_dir, f"{split}_codes_index.npy")
    codes = np.memmap(codes_path, dtype=np.int32, mode="r")
    codes_index = np.load(codes_index_path)
    number_of_codes = codes.shape[0]

    filtered_samples, filtered_indexes = [], []
    sample_status = collections.Counter()

    # Build composable filters using dataset_config.
    filters = [
        filtering.filter_allowed_languages(dataset_config.allowed_languages),
        filtering.filter_min_sample_rate(dataset_config.min_sample_rate),
        filtering.filter_min_dnsmos_score(dataset_config.min_dnsmos_score),
        filtering.filter_min_audio_duration(dataset_config.min_audio_duration),
        filtering.filter_empty_transcript,
        filtering.filter_long_duration,
        filtering.filter_punct_or_space_only_transcript,
    ]

    with open(samples_path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            sample = data_sample.Sample.from_json(json.loads(line), dataset_name)
            sample_status["total"] += 1

            # Apply composable filters and stop at the first that applies.
            filter_reason = None
            for f in filters:
                filter_reason = f(sample)
                if filter_reason:
                    sample_status[f"filtered_by_{filter_reason}"] += 1
                    sample_status["total_filtered"] += 1
                    break
            if filter_reason:
                continue

            sample_status[f"{sample.language}"] += 1
            filtered_samples.append(sample)
            left = codes_index[idx]
            right = (
                codes_index[idx + 1] if idx < len(codes_index) - 1 else number_of_codes
            )
            filtered_indexes.append((left, right))

    del codes_index
    gc.collect()

    logging.info(f"[{dataset_name}]-dataset stats: {dict(sample_status)}")
    return codes, filtered_samples, filtered_indexes, sample_status


def find_all_wavs_recursively(root_dir: str) -> list[str]:
    """Finds all wav files recursively in a directory."""
    wav_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if not file.endswith(".wav"):
                continue

            wav_files.append(os.path.join(root, file))
            if len(wav_files) % 10000 == 0:
                logging.info("Already found %d wav files", len(wav_files))

    return wav_files


def load_wav(
    wav_path: str, target_sample_rate: int | None = None
) -> tuple[torch.Tensor, int]:
    """Loads and preprocesses a WAV file.

    Args:
        wav_path: Path to the WAV file to load
        target_sample_rate: Optional target sample rate to resample to. If None,
            keeps original.

    Returns:
        Tuple of (preprocessed audio tensor with shape [1, num_samples], sample rate)
    """
    wav, sr = torchaudio.load(wav_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if target_sample_rate and sr != target_sample_rate:
        wav = torchaudio.functional.resample(wav, sr, target_sample_rate)
        sr = target_sample_rate
    return wav, sr
