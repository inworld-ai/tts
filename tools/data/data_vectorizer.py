"""Batch encoder of audio files using audio encoder."""

import json
import os
import time
from typing import Sequence

import numpy as np
import torch
import torchaudio
import transformers
import wandb
from absl import app, flags, logging

from tts.core import constants
from tts.core.codec import encoding
from tts.data import data_sample, data_utils
from tts.training import environment
from tts.utils import custom_logging

FLAGS = flags.FLAGS

_BATCH_SIZE = flags.DEFINE_integer('batch_size', 8, 'Batch size for processing.')
_CODEC_MODEL_PATH = flags.DEFINE_string(
    'codec_model_path', "/path/to/some-982760.pt",
    'Path to the codec model checkpoint.')
_COMPILE_MODEL = flags.DEFINE_boolean('compile_model', False,
                                      'Whether to compile the model.')
_DATASET_PATH = flags.DEFINE_string('dataset_path', None, 'Path to the dataset.')
_DRY_RUN = flags.DEFINE_boolean('dry_run', False, 'Dry run')
_NUM_WORKERS = flags.DEFINE_integer('num_workers', 1,
                                    'Number of worker threads for the DataLoader.')
_OUTPUT_DIR = flags.DEFINE_string('output_dir', None, 'Path to the output directory.')
_RUN_NAME = flags.DEFINE_string('run_name', None, 'Run name.')
_SLURM_DISTRIBUTED = flags.DEFINE_boolean('slurm_distributed', False,
                                          'Whether to run in SLURM distributed mode.')
_USE_WANDB = flags.DEFINE_boolean('use_wandb', False, 'Whether to use wandb.')
_VAL_SPLIT = flags.DEFINE_float('val_split', 0.001, 'Validation split.')

_LOG_EVERY_N_BATCHES = 20

_BATCH_ITEM = tuple[torch.Tensor, torch.Tensor, int, data_sample.Sample]
_BATCH = list[_BATCH_ITEM]


def pad_audio_batch(
    batch: _BATCH
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[data_sample.Sample]]:
    """Pads a batch of audio files to the same length."""
    filtered_batch = [item for item in batch if item[0] is not None]
    if len(filtered_batch) == 0:
        return None, None, None, None

    audio_list, feat_list, codes_lengths, samples = zip(*filtered_batch)
    feat_list = list(feat_list)
    max_length_feat = max([feat.shape[1] for feat in feat_list])
    max_length = max_length_feat * 320

    padded_audios = []
    for audio in audio_list:
        padding = max_length - audio.shape[1]
        if padding > 0:

            padded_audio = torch.nn.functional.pad(audio, (0, padding),
                                                   mode='constant',
                                                   value=0)
        else:
            padded_audio = audio[:, :max_length]
        padded_audios.append(padded_audio)

    padded_feat_list = []
    for feat in feat_list:
        padding = max_length_feat - feat.shape[1]
        padded_feat = torch.nn.functional.pad(feat, (0, 0, 0, padding),
                                              mode='constant',
                                              value=0)
        padded_feat_list.append(padded_feat)

    padded_audios = torch.stack(padded_audios)
    padded_feat_list = torch.stack(padded_feat_list)
    return padded_audios, padded_feat_list, codes_lengths, samples


class WaveDataset(torch.utils.data.Dataset):
    """A dataset to prepare features for the audio encoder."""

    def __init__(self, samples: list[data_sample.Sample], sampling_rate: int):
        self.samples = samples
        self.sampling_rate = sampling_rate
        self.hop_length = 320  # 50 codes / sec of sample rate
        self.half_hop_length = self.hop_length // 2
        self.feature_extractor = transformers.AutoFeatureExtractor.from_pretrained(
            "facebook/w2v-bert-2.0")

    def __getitem__(self, index: int) -> _BATCH_ITEM:
        sample = self.samples[index]
        fname = sample.wav_path

        try:
            audio, sr = data_utils.load_wav(fname,
                                            target_sample_rate=self.sampling_rate)
        except Exception as e:
            logging.warning("Skipping sample [%s] because: %s", fname, e)
            return None, None, None, None

        audio = torch.nn.functional.pad(audio, (0, self.hop_length -
                                                (audio.shape[1] % self.hop_length)))
        audio_pad = torch.nn.functional.pad(
            audio, (self.half_hop_length, self.half_hop_length))
        feat = self.feature_extractor(audio_pad,
                                      sampling_rate=self.sampling_rate,
                                      return_tensors="pt").data['input_features']

        # Note that the audio is padded by half a hop length on each side.
        # so the number of codes doesn't always match the number of frames.
        codes_length = int(audio.shape[1] / self.hop_length)
        return audio, feat, codes_length, sample

    def __len__(self) -> int:
        return len(self.samples)

def save_data(samples: list[data_sample.Sample], codes: np.ndarray,
              codes_index: np.ndarray, output_dir: str, split: str, rank: int):
    """Save codes, codes index, and samples to files."""
    start_time = time.time()

    codes_index_file = os.path.join(output_dir, f'{split}_codes_index_{rank}.npy')
    codes_file = os.path.join(output_dir, f'{split}_codes_{rank}.npy')
    samples_file = os.path.join(output_dir, f'{split}_samples_{rank}.jsonl')

    np.save(codes_index_file, codes_index)
    # TODO: consider trying unit16 instead.
    codes_arr = np.memmap(codes_file,
                          dtype=np.int32,
                          mode="w+",
                          shape=(codes.shape[0],))
    codes_arr[:] = codes
    codes_arr.flush()

    with open(samples_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample.to_json(), ensure_ascii=False) + '\n')

    logging.info("Saved [%s] codes/index/samples  to %s in %.2f seconds", split,
                 codes_file,
                 time.time() - start_time)


def main(argv: Sequence[str]) -> None:
    del argv  # Unused.

    # ------------------ Hardware initialization. ------------------ #
    env_context = environment.initialize_distributed_environment_context(
        slurm_distributed=_SLURM_DISTRIBUTED.value)
    custom_logging.reconfigure_absl_logging_handler(global_rank=env_context.global_rank)
    if not env_context.is_main_process():
        logging.set_verbosity(logging.ERROR)

    logging.info("Vectorizer initialized with world size: [%s]. Flags: [%s]",
                 env_context.world_size, FLAGS.flags_into_string())

    # ------------------ Wandb initialization. ------------------ #
    use_wandb = _USE_WANDB.value
    if use_wandb:
        wandb.init(project=os.environ["WANDB_PROJECT"],
                   name=_RUN_NAME.value,
                   group=_RUN_NAME.value)

    # ------------------ Model initialization. ------------------ #
    start_time = time.time()
    model = encoding.create(_CODEC_MODEL_PATH.value, env_context.device)
    num_params = sum(p.numel() for p in model._encoder.parameters())
    logging.info("Model loaded in %.2f seconds. Number of parameters: %.2fM",
                 time.time() - start_time, num_params / 1e6)

    if _COMPILE_MODEL.value:
        logging.info("Compiling the model...")
        model._encoder = torch.compile(model._encoder)

    # Before getting into expensive computations, confirm there's write access in place.
    output_dir = _OUTPUT_DIR.value
    os.makedirs(output_dir, exist_ok=True)

    # Before getting into expensive computations, confirm there's write access in place.
    output_dir = _OUTPUT_DIR.value
    os.makedirs(output_dir, exist_ok=True)

    # ------------------ Data loading. ------------------ #
    batch_size = _BATCH_SIZE.value
    max_samples = -1
    if _DRY_RUN.value:
        max_samples = batch_size * env_context.world_size * 50
    original_samples, _ = data_utils.load_samples(_DATASET_PATH.value,
                                                  max_samples=max_samples)
    original_samples = data_utils.chunk_work(work_items=original_samples,
                                             worker_id=env_context.global_rank,
                                             num_workers=env_context.world_size)
    dataset = WaveDataset(samples=original_samples,
                          sampling_rate=constants.CODEC_SAMPLE_RATE)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=True,
                                             num_workers=_NUM_WORKERS.value,
                                             pin_memory=True,
                                             collate_fn=pad_audio_batch)

    # ------------------ Data processing. ------------------ #
    logging.info("Starting to process data...")
    all_samples, all_codes, codes_index = [], [], []
    index_counter = 0
    total_batches = len(dataloader)

    total_batch_times, inference_times = [], []
    batch_start = time.time()
    for batch_idx, batch in enumerate(dataloader):
        if batch is None:
            logging.warning("Skipping batch [%d] because it is empty.", batch_idx)
            continue

        wavs, feats, codes_lengths, batch_samples = batch
        wavs = wavs.to(env_context.device)
        feats = feats.to(env_context.device)

        inference_start = time.time()
        with torch.no_grad():
            vq_code = model._encoder(wavs, feats).cpu().squeeze(1)  # shape [B, T_codes]
        inference_time = time.time() - inference_start
        inference_times.append(inference_time)

        all_samples.extend(batch_samples)
        for idx, code_length in enumerate(codes_lengths):
            all_codes.append(vq_code[idx, :code_length].tolist())
            codes_index.append(index_counter)
            index_counter += code_length

        batch_time = time.time() - batch_start
        total_batch_times.append(batch_time)

        if batch_idx and batch_idx % _LOG_EVERY_N_BATCHES == 0:
            avg_batch_time = sum(total_batch_times) / len(total_batch_times)
            avg_inference_time = sum(inference_times) / len(inference_times)
            logging.info(
                "Processed %d/%d batches. Avg batch latency: %.2fs, "
                "avg inference latency: %.2fs", batch_idx, total_batches,
                avg_batch_time, avg_inference_time)
            if use_wandb:
                wandb.log({
                    "batches_processed": batch_idx,
                    "avg_batch_latency": avg_batch_time,
                    "avg_inference_latency": avg_inference_time
                })
            total_batch_times = []
            inference_times = []

        # Reset the batch start time to cover data loading time.
        batch_start = time.time()

    if len(all_samples) != len(all_codes):
        raise ValueError("Number of samples [%d] does not match number of codes [%d]",
                         len(all_samples), len(all_codes))

    # ------------------ Data saving. ------------------ #
    logging.info("Starting to save codes...")

    # Compute train/val split.
    val_split = _VAL_SPLIT.value
    total_num_samples = len(all_codes)
    train_num_samples = int(total_num_samples * (1 - val_split))
    logging.info("Train / val split: %d / %d (total: %d)", train_num_samples,
                 total_num_samples - train_num_samples, total_num_samples)

    # Create train codes vector.
    train_codes = np.concatenate(all_codes[:train_num_samples])
    train_codes_index = np.array(codes_index[:train_num_samples])

    # Create val codes vector.
    val_codes = np.concatenate(all_codes[train_num_samples:])
    val_codes_index = np.array(codes_index[train_num_samples:])
    val_codes_index -= np.min(val_codes_index)

    # Split samples into train/val as well.
    train_samples = all_samples[:train_num_samples]
    val_samples = all_samples[train_num_samples:]

    save_data(train_samples,
              train_codes,
              train_codes_index,
              output_dir,
              split='train',
              rank=env_context.global_rank)
    save_data(val_samples,
              val_codes,
              val_codes_index,
              output_dir,
              split='val',
              rank=env_context.global_rank)

    # ------------------ Finalization. ------------------ #
    logging.info("Finished encoding. Exiting.")


if __name__ == '__main__':
    flags.mark_flags_as_required(['dataset_path', 'output_dir'])
    app.run(main)
