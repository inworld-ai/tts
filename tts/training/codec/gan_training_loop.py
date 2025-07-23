import time
from collections.abc import Iterator

import lightning.fabric as lightning_fabric
import torch
import wandb
from absl import logging

from tts.core import constants
from tts.inference import quality_validation
from tts.training import checkpointing
from tts.utils import configuration, custom_logging

_PERFORMANCE_METRICS_STEP_OFFSET = 5


def _running_average(old: float, new: float) -> float:
    """Computes a running average of the old and new values."""
    if old == -1.0:
        return new
    else:
        return 0.9 * old + 0.1 * new


def _train_micro_batch(
    statistics: custom_logging.Statistics,
    fabric: lightning_fabric.Fabric,
    model: torch.nn.Module,
    optimizers: dict[str, torch.optim.Optimizer],
    train_data_loader: torch.utils.data.DataLoader,
    train_data_loader_iterator: Iterator,
    gradient_clip_value: float,
    gradient_accumulation_steps: int,
) -> tuple[Iterator, float]:
    """Performs a micro-batch of training."""
    statistics.start_micro_batch_training()
    average_duration_sec = 0.0

    # Note: resettings the grads here, not right after the backward pass because
    #       the health check eval needs reading the grads.
    optimizers["gen_optimizer"].zero_grad()
    optimizers["disc_optimizer"].zero_grad()

    for micro_step in range(gradient_accumulation_steps):
        with custom_logging.Timer() as t:
            try:
                batch = next(train_data_loader_iterator)
            except StopIteration:
                logging.info(
                    "Reached the end of the training dataset. "
                    "Restarting the iterator."
                )
                train_data_loader_iterator = iter(train_data_loader)
                batch = next(train_data_loader_iterator)

        data_reading_time = t.get_duration()
        is_accumulating = micro_step < gradient_accumulation_steps - 1
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            audio_processed_sec = torch.sum(batch.pop("audio_processed_sec")).item()
            samples_processed = batch["audio_codes"].shape[0]

            output = model.training_step(
                fabric=fabric,
                batch=batch,
                gradient_accumulation_steps=gradient_accumulation_steps,
            )
            del batch

            # Update statistics.
            metrics = {
                k: output[k].detach().item()
                for k in [
                    "disc_loss",
                    "gen_loss",
                    "fm_loss",
                    "mel_loss",
                    "adv_loss",
                    "rms_loss",
                ]
            }
            # TODO: add source to the statistics.
            statistics.record(
                metrics=metrics,
                sources=[constants.TOTAL_SOURCE],
                stats_to_sum={
                    "samples_processed": samples_processed,
                    "audio_processed_sec": audio_processed_sec,
                },
            )
            del output

    # optimizer step
    fabric.clip_gradients(
        model,
        optimizers["gen_optimizer"],
        clip_val=gradient_clip_value,
        error_if_nonfinite=True,
    )
    fabric.clip_gradients(
        model,
        optimizers["disc_optimizer"],
        clip_val=gradient_clip_value,
        error_if_nonfinite=True,
    )
    optimizers["gen_optimizer"].step()
    optimizers["disc_optimizer"].step()

    average_duration_sec /= gradient_accumulation_steps
    return train_data_loader_iterator, data_reading_time, average_duration_sec


def run(
    *,
    fabric: lightning_fabric.Fabric,
    model: torch.nn.Module,
    train_data_loader: torch.utils.data.DataLoader,
    train_data_loader_iterator: Iterator[tuple[torch.Tensor, torch.Tensor]],
    config: configuration.ExperimentConfig,
    optimizers: dict[str, torch.optim.Optimizer],
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    val_data_loader: torch.utils.data.DataLoader,
    quality_validator: quality_validation.QualityValidator,
    train_dataset_names: list[str],
    use_wandb: bool,
) -> custom_logging.Statistics:
    """Implements training loop with loss values computed per batch."""

    total_training_steps = config.dataset.total_training_steps
    steps_per_epoch = config.dataset.steps_per_epoch
    logging_steps = config.training.logging_steps
    save_steps = config.checkpointing.save_steps
    gradient_accumulation_steps = config.training.gradient_accumulation_steps
    gradient_clip_value = config.training.gradient_clip_value
    total_samples_per_step = (
        train_data_loader.batch_size * gradient_accumulation_steps * fabric.world_size
    )

    running_step_time, running_data_reading_time = -1.0, -1.0
    statistics = custom_logging.Statistics(train_dataset_names)

    # ------------------ Load checkpoint if needed. ------------------ #
    checkpoint_file_to_resume_from = config.checkpointing.checkpoint_file_to_resume_from
    only_load_model_weights = config.checkpointing.only_load_model_weights
    if checkpoint_file_to_resume_from:
        model, old_statistics, optimizers = checkpointing.load_from_checkpoint(
            fabric=fabric,
            model=model,
            optimizer=optimizers,
            checkpoint_file_to_resume_from=checkpoint_file_to_resume_from,
            load_full_checkpoint=not only_load_model_weights,
        )
        if old_statistics:
            statistics = old_statistics
            logging.info(
                "Loaded checkpoint from [%s] at step [%d] with learning rate: %f.",
                checkpoint_file_to_resume_from,
                statistics.step,
                lr_scheduler.get_lr(statistics.step),
            )
        else:
            logging.info(
                "Loaded model weights from [%s].", checkpoint_file_to_resume_from
            )

    keep_training = True
    model.train()
    while keep_training:
        # No evaluation for now,
        # the vocoder should be trained for as long time as possible.

        timestamp = time.perf_counter()

        # schedule learning rates for gen and disc optimizers
        learning_rate = lr_scheduler.get_lr(statistics.step)
        for param_group in optimizers["gen_optimizer"].param_groups:
            param_group["lr"] = learning_rate
        for param_group in optimizers["disc_optimizer"].param_groups:
            param_group["lr"] = learning_rate

        runtime_error = False
        try:
            # Perform a micro-batch of training.
            train_data_loader_iterator, data_reading_time, average_duration_sec = (
                _train_micro_batch(
                    statistics,
                    fabric,
                    model,
                    optimizers,
                    train_data_loader,
                    train_data_loader_iterator,
                    gradient_clip_value,
                    gradient_accumulation_steps,
                )
            )
        except RuntimeError as e:
            runtime_error = True
            logging.info("Train batch runtime error: %s, stopping training.", str(e))
            keep_training = False

        # Stop if needed.
        statistics.step += 1
        if statistics.step >= total_training_steps:
            logging.info("Maximum number of steps reached. Stopping the training.")
            keep_training = False

        if statistics.step >= _PERFORMANCE_METRICS_STEP_OFFSET:
            step_time = time.perf_counter() - timestamp
            running_step_time = _running_average(old=running_step_time, new=step_time)
            running_data_reading_time = _running_average(
                old=running_data_reading_time, new=data_reading_time
            )

        # Perform logging.
        if statistics.step % logging_steps == 0:
            logs = custom_logging.get_logging_stats(
                fabric,
                statistics,
                steps_per_epoch=steps_per_epoch,
                total_samples_per_step=total_samples_per_step,
                learning_rate=learning_rate,
                running_data_reading_time=running_data_reading_time,
                running_step_time=running_step_time,
            )
            logs["train_average_duration_sec"] = average_duration_sec
            logging.info("Training step %d: %s", statistics.step, logs)
            if use_wandb and fabric.is_global_zero:
                wandb.log(
                    {**custom_logging.rewrite_logs_for_wandb(logs)},
                    step=statistics.step,
                )

        # TODO: best eval loss won't be saved if step isn't a multiple of eval_steps.
        #       also save the latest checkpoint if runtime error occurs

        if (save_steps > 0 and statistics.step % save_steps == 0) or runtime_error:
            with custom_logging.Timer() as t:
                checkpoint_file = checkpointing.save_to_checkpoint(
                    fabric, model, config, optimizers, statistics
                )
            logging.info(
                "Step [%d]: checkpointing to %s took %.2f s.",
                statistics.step,
                checkpoint_file,
                t.get_duration(),
            )

            if quality_validator is not None and fabric.is_global_zero:
                with custom_logging.Timer() as t:
                    quality_validator.validate(model.module, statistics.step)
                logging.info(
                    "Step [%d]: quality validation took %.2f s.",
                    statistics.step,
                    t.get_duration(),
                )
            fabric.barrier()

    return statistics
