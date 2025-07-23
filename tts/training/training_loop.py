import contextlib
import time
from collections.abc import Iterator

import lightning.fabric as lightning_fabric
import torch
import wandb
from absl import logging

from tts.core import constants
from tts.inference import quality_validation
from tts.training import checkpointing, evaluation
from tts.utils import configuration, custom_logging

_PERFORMANCE_METRICS_STEP_OFFSET = 5


def _running_average(old: float, new: float) -> float:
    """Computes a running average of the old and new values."""
    if old == -1.0:
        return new
    else:
        return 0.9 * old + 0.1 * new


def _resume_from_checkpoint(
    fabric: lightning_fabric.Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: configuration.ExperimentConfig,
    train_data_loader: torch.utils.data.DataLoader,
    train_data_loader_iterator: Iterator,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    statistics: custom_logging.Statistics,
) -> tuple[torch.nn.Module, torch.optim.Optimizer, custom_logging.Statistics, Iterator]:
    """Resumes training from a checkpoint if specified in the config."""
    checkpoint_file_to_resume_from = config.checkpointing.checkpoint_file_to_resume_from
    if checkpoint_file_to_resume_from:
        logging.info("Loading checkpoint from [%s]...", checkpoint_file_to_resume_from)
        only_load_model_weights = config.checkpointing.only_load_model_weights
        model, loaded_statistics, optimizer = checkpointing.load_from_checkpoint(
            fabric=fabric,
            model=model,
            optimizer=optimizer,
            checkpoint_file_to_resume_from=checkpoint_file_to_resume_from,
            load_full_checkpoint=not only_load_model_weights,
        )

        if loaded_statistics:
            statistics = loaded_statistics
            logging.info(
                "Checkpoint [%s] was loaded. Advancing dataloader iterator...",
                checkpoint_file_to_resume_from,
            )

            # TODO: find a more elegant solution to avoid using this.
            # pylint: disable=protected-access
            train_data_loader._dataloader.dataset.enable_fast_forwarding()
            train_data_loader_iterator = iter(train_data_loader)

            for step_advanced in range(statistics.step):
                if step_advanced and step_advanced % 10000 == 0:
                    logging.info(
                        "Already advanced dataloader iterator to step [%d]...",
                        step_advanced,
                    )
                next(train_data_loader_iterator)

            train_data_loader._dataloader.dataset.disable_fast_forwarding()
            train_data_loader_iterator = iter(train_data_loader)
            # pylint: enable=protected-access

            logging.info(
                "Dataloader iterator was advanced to step [%d] with learning rate: %f.",
                statistics.step,
                lr_scheduler.get_lr(statistics.step),
            )

        else:
            logging.info(
                "Loaded model weights from [%s].", checkpoint_file_to_resume_from
            )

    return model, optimizer, statistics, train_data_loader_iterator


def _get_no_backward_sync_ctx(
    fabric: lightning_fabric.Fabric, deepspeed: bool
) -> contextlib.AbstractContextManager:
    """Returns a context manager for gradient synchronization during backward pass."""
    if deepspeed:

        @contextlib.contextmanager
        def _null_context(*args, **kwargs):
            yield

        return _null_context

    return fabric.no_backward_sync


def _train_micro_batch(
    statistics: custom_logging.Statistics,
    fabric: lightning_fabric.Fabric,
    no_backward_sync_ctx: contextlib.AbstractContextManager,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data_loader: torch.utils.data.DataLoader,
    train_data_loader_iterator: Iterator,
    gradient_clip_value: float,
    gradient_accumulation_steps: int,
    deepspeed: bool,
) -> tuple[Iterator, float]:
    """Performs a micro-batch of training."""
    statistics.start_micro_batch_training()
    average_duration_sec = 0.0

    # Note: resettings the grads here, not right after the backward pass because
    #       the health check eval needs reading the grads.
    optimizer.zero_grad()

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
        with no_backward_sync_ctx(model, enabled=is_accumulating):
            average_duration_sec += torch.mean(
                batch.pop("generated_audio_duration_sec")
            ).item()
            tokens_processed = torch.sum(batch.pop("tokens_processed")).item()
            audio_processed_sec = torch.sum(batch.pop("audio_processed_sec")).item()
            sources = [constants.TOTAL_SOURCE] + batch.pop("source")

            output = model(**batch)
            del batch

            loss = output.loss / gradient_accumulation_steps
            fabric.backward(loss)

            # Update statistics.
            statistics.record(
                metrics={"loss": loss.detach().item()},
                sources=sources,
                stats_to_sum={
                    "tokens_processed": tokens_processed,
                    "audio_processed_sec": audio_processed_sec,
                },
            )

            del output

    if not deepspeed:
        fabric.clip_gradients(
            model, optimizer, clip_val=gradient_clip_value, error_if_nonfinite=True
        )
    optimizer.step()

    average_duration_sec /= gradient_accumulation_steps
    return train_data_loader_iterator, data_reading_time, average_duration_sec


def run(
    *,
    fabric: lightning_fabric.Fabric,
    model: torch.nn.Module,
    train_data_loader: torch.utils.data.DataLoader,
    train_data_loader_iterator: Iterator,
    config: configuration.ExperimentConfig,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    quality_validator: quality_validation.QualityValidator,
    val_data_loader: torch.utils.data.DataLoader,
    train_dataset_names: list[str],
    use_wandb: bool,
) -> custom_logging.Statistics:
    """Implements training loop with loss values computed per batch."""
    total_training_steps = config.dataset.total_training_steps
    steps_per_epoch = config.dataset.steps_per_epoch
    eval_steps = config.training.eval_steps
    logging_steps = config.training.logging_steps
    save_steps = config.checkpointing.save_steps
    gradient_accumulation_steps = config.training.gradient_accumulation_steps
    gradient_clip_value = config.training.gradient_clip_value
    total_samples_per_step = (
        train_data_loader.batch_size * gradient_accumulation_steps * fabric.world_size
    )
    deepspeed = config.training.strategy == configuration.TrainingStrategy.DEEPSPEED
    no_backward_sync_ctx = _get_no_backward_sync_ctx(fabric, deepspeed)

    step_time, data_reading_time = -1.0, -1.0
    running_step_time, running_data_reading_time = -1.0, -1.0
    statistics = custom_logging.Statistics(train_dataset_names)

    # ------------------ Load checkpoint if needed. ------------------ #
    model, optimizer, statistics, train_data_loader_iterator = _resume_from_checkpoint(
        fabric=fabric,
        model=model,
        optimizer=optimizer,
        config=config,
        train_data_loader=train_data_loader,
        train_data_loader_iterator=train_data_loader_iterator,
        lr_scheduler=lr_scheduler,
        statistics=statistics,
    )

    torch.cuda.empty_cache()
    fabric.barrier()
    keep_training = True
    model.train()
    while keep_training:
        # Perform evaluation at the beginning to see the default performance.
        # TODO: consider doing it only for the main rank, because evals only
        #       main rank's stats will be reported anyway.
        if val_data_loader is not None:
            if statistics.step == 0 or statistics.step % eval_steps == 0:
                logging.info("Eval step %d: %s", statistics.step, "Starting evaluation")
                fabric.barrier()
                model.eval()
                metrics = evaluation.compute_metrics(
                    fabric=fabric,
                    model=model,
                    val_data_loader=val_data_loader,
                    optimizer=optimizer,
                    collect_health_stats=config.checkpointing.collect_health_stats,
                )
                logging.info("Eval step %d: %s", statistics.step, metrics)
                if use_wandb and fabric.is_global_zero:
                    wandb.log(
                        {**custom_logging.rewrite_logs_for_wandb(metrics)},
                        step=statistics.step,
                    )
                model.train()
                fabric.barrier()
                logging.info(f"Back to training step: {statistics.step}")

        timestamp = time.perf_counter()
        learning_rate = lr_scheduler.get_lr(statistics.step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate

        runtime_error = False
        try:
            # Perform a micro-batch of training.
            train_data_loader_iterator, data_reading_time, average_duration_sec = (
                _train_micro_batch(
                    statistics,
                    fabric,
                    no_backward_sync_ctx,
                    model,
                    optimizer,
                    train_data_loader,
                    train_data_loader_iterator,
                    gradient_clip_value,
                    gradient_accumulation_steps,
                    deepspeed,
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
            fabric.barrier()

        # TODO: best eval loss won't be saved if step isn't a multiple of eval_steps.
        #       also save the latest checkpoint if runtime error occurs
        if (save_steps > 0 and statistics.step % save_steps == 0) or runtime_error:
            logging.info("Step [%d]: starting checkpointing...", statistics.step)
            with custom_logging.Timer() as t:
                checkpoint_file = checkpointing.save_to_checkpoint(
                    fabric, model, config, optimizer, statistics
                )
            logging.info(
                "Step [%d]: checkpointing to %s took %.2f s.",
                statistics.step,
                checkpoint_file,
                t.get_duration(),
            )
            fabric.barrier()

            with custom_logging.Timer() as t:
                quality_validator.validate(model, statistics.step)
            logging.info(
                "Step [%d]: quality validation took %.2f s.",
                statistics.step,
                t.get_duration(),
            )
            fabric.barrier()

    return statistics
