"""Main entry point for the Inworld Codec training pipeline."""

import json
import math
import os
import time
from collections.abc import Sequence

import lightning.fabric as lightning_fabric
import wandb
from absl import app, flags, logging

from tts.core import constants, optimization
from tts.core.codec import decoder
from tts.training import checkpointing, environment
from tts.training.codec import (
    codec_datasets,
    codec_quality_validation,
    gan_training_loop,
)
from tts.utils import configuration, custom_logging

FLAGS = flags.FLAGS

_CONFIG_PATH = flags.DEFINE_string(
    "config_path", "configs/decoder.json", "Path to the training config json file."
)
_SLURM_DISTRIBUTED = flags.DEFINE_bool(
    "slurm_distributed", False, "Whether the script is running via SLURM."
)
_RUN_NAME = flags.DEFINE_string("run_name", "test", "Run name for wandb logging.")
_DRY_RUN = flags.DEFINE_bool("dry_run", False, "Whether to run in dry run mode.")
_EXPERIMENT_DIR = flags.DEFINE_string(
    "experiment_dir", "./experiments", "Experiment directory."
)
_SEED = flags.DEFINE_integer("seed", 777, "Random seed.")

# Wandb logging flags.
_PROJECT_NAME = flags.DEFINE_string(
    "project_name", None, "Project name for wandb logging."
)
_USE_WANDB = flags.DEFINE_bool("use_wandb", False, "Whether to use wandb for logging.")

_QUALITY_VALIDATION_SAMPLE_SIZE = 4


def setup_directories(
    experiment_dir: str, checkpoint_dir: str, is_global_zero: bool
) -> tuple[str, str]:
    """Makes path absolute and creates directory if it doesn't exist."""
    experiment_dir = os.path.abspath(experiment_dir)
    checkpoint_dir = os.path.join(experiment_dir, checkpoint_dir)
    if not os.path.exists(checkpoint_dir) and is_global_zero:
        os.makedirs(checkpoint_dir)
        logging.info("Checkpoint directory was created: %s", checkpoint_dir)

    return experiment_dir, checkpoint_dir


def save_codec_config(config: configuration.CodecTrainingConfig, checkpoint_dir: str):
    """Saves model config to a file, which is used to initialize the trained model."""

    config_file = os.path.join(checkpoint_dir, "model_config.json")
    total_ups = math.prod(config.upsample_factors) if config.upsample_factors else 1
    model_config = {
        "model_type": _RUN_NAME.value,
        "sample_rate": config.sample_rate,
        "token_rate": config.sample_rate // (config.hop_length * total_ups),
        "hop_length": config.hop_length,
        "upsample_factors": config.upsample_factors,
        "kernel_sizes": config.kernel_sizes,
    }
    with open(config_file, "w") as f:
        json.dump(model_config, f, indent=4)


def run_training(
    fabric: lightning_fabric.Fabric,
    config: configuration.ExperimentConfig,
    use_wandb: bool,
    dry_run: bool,
) -> None:
    """Launches the training pipeline."""

    # ------------------ Model ------------------ #

    with custom_logging.Timer() as t:
        model = decoder.create(
            sample_rate=config.codec.sample_rate,
            hop_length=config.codec.hop_length,
            upsample_factors=config.codec.upsample_factors,
            kernel_sizes=config.codec.kernel_sizes,
        )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    config.model_size = num_params
    logging.info(
        "Model with [%.4f]M parameters has been loaded in %.2f seconds",
        num_params / 1e6,
        t.get_duration(),
    )

    # ------------------ Data setup. ------------------ #
    logging.info("Starting data setup...")
    with custom_logging.Timer() as t:
        collate_fn = codec_datasets.collate_fn

        train_dataset = codec_datasets.merge_datasets(
            weighted_datasets=config.train_weighted_datasets,
            split=constants.TRAIN_SPLIT,
            audio_window_size=config.codec.audio_window_size,
            sample_rate=config.codec.sample_rate,
            minimum_data_sample_rate=config.codec.minimum_data_sample_rate,
        )
        val_dataset = codec_datasets.merge_datasets(
            weighted_datasets=config.val_weighted_datasets,
            split=constants.VAL_SPLIT,
            audio_window_size=config.codec.audio_window_size,
            sample_rate=config.codec.sample_rate,
            minimum_data_sample_rate=config.codec.minimum_data_sample_rate,
        )
        train_data_loader = codec_datasets.get_dataloader(
            train_dataset,
            config.training.batch_size,
            collate_fn,
            shuffle=True,
            num_workers=config.training.num_workers,
        )
        val_data_loader = codec_datasets.get_dataloader(
            val_dataset,
            config.training.batch_size,
            collate_fn,
            shuffle=False,
            num_workers=config.training.num_workers,
        )

        # A single epoch will be used for logging. Number of steps already has
        # all the information for how many actual epochs each dataset will be processed.
        steps_per_epoch = int(
            len(train_data_loader)
            / fabric.world_size
            / config.training.gradient_accumulation_steps
        )
        config.dataset.total_training_steps = steps_per_epoch
        config.dataset.steps_per_epoch = steps_per_epoch
    logging.info(
        "Datasets were loaded in %.2f seconds. Config: %s",
        t.get_duration(),
        config.dataset,
    )

    # ------------------ Save config. ------------------ #
    if fabric.is_global_zero:
        config.wandb_url = wandb.run.url if use_wandb else None
        checkpointing.save_config(config, config.checkpointing.directory, use_wandb)
        save_codec_config(config.codec, config.checkpointing.directory)
        logging.info(
            "Model config [%s] has been saved to [%s].",
            config,
            config.checkpointing.directory,
        )

    # ------------------ Optimizer and scheduler. ------------------ #
    warmup_steps = int(
        config.training.warmup_ratio * config.dataset.total_training_steps
    )
    lr_scheduler = optimization.CosineLrScheduler(
        learning_rate=config.training.learning_rate,
        warmup_steps=warmup_steps,
        lr_decay_steps=config.dataset.total_training_steps,
    )

    gen_optimizer, disc_optimizer = decoder.create_optimizer(
        model=model,
        learning_rate=config.training.learning_rate,
        betas=config.training.betas,
        weight_decay=config.training.weight_decay,
    )

    logging.info(
        "%s optimizer and cosine LR scheduler created with %d warmup steps.",
        gen_optimizer.__class__.__name__,
        warmup_steps,
    )

    train_data_loader = fabric.setup_dataloaders(train_data_loader)
    val_data_loader = fabric.setup_dataloaders(val_data_loader)
    quality_validation_samples = codec_datasets.collate_fn(
        [val_dataset[i] for i in range(_QUALITY_VALIDATION_SAMPLE_SIZE)]
    )
    # TODO: investigate torch.compile for training.
    # model = torch.compile(model)
    model, gen_optimizer, disc_optimizer = fabric.setup(
        model, gen_optimizer, disc_optimizer
    )
    model.mark_forward_method("training_step")

    if dry_run:
        data_sample = codec_datasets.prettify_data_sample(next(iter(train_data_loader)))
        print(model(**data_sample).loss)
        print("Dry run completed successfully!")
        return

    # ------------------ Quality validator. ------------------ #
    with custom_logging.Timer() as t:
        quality_validator = codec_quality_validation.create_codec_quality_validator(
            batched_samples=quality_validation_samples,
            checkpointing_dir=config.checkpointing.directory,
        )
    logging.info("Quality validator created in %.2f seconds.", t.get_duration())

    # ------------------ Training. ------------------ #
    logging.info("Starting training...")
    final_statistics = gan_training_loop.run(
        fabric=fabric,
        model=model,
        train_data_loader=train_data_loader,
        train_data_loader_iterator=iter(train_data_loader),
        config=config,
        optimizers={"gen_optimizer": gen_optimizer, "disc_optimizer": disc_optimizer},
        lr_scheduler=lr_scheduler,
        quality_validator=quality_validator,
        val_data_loader=val_data_loader,
        train_dataset_names=train_dataset.sources,
        use_wandb=use_wandb,
    )

    # ------------------ Save final model. ------------------ #
    checkpointing.save_to_checkpoint(
        fabric,
        model,
        config,
        {"gen_optimizer": gen_optimizer, "disc_optimizer": disc_optimizer},
        statistics=final_statistics,
        checkpoint_name="final_model.pt",
    )


def main(argv: Sequence[str]) -> None:
    del argv  # Unused.

    # ------------------ Config. ------------------ #
    config = configuration.ExperimentConfig.from_json(file=_CONFIG_PATH.value)
    config.seed = _SEED.value

    # ------------------ Hardware initialization. ------------------ #
    env_context = environment.initialize_distributed_environment_context(
        slurm_distributed=_SLURM_DISTRIBUTED.value
    )
    fabric = environment.initialize_fabric(
        env_context,
        strategy_name=config.training.strategy.value,
        training_precision=config.training.precision,
        find_unused_parameters=True,
    )
    config.world_size = fabric.world_size
    fabric.seed_everything(config.seed)
    fabric.barrier()
    custom_logging.reconfigure_absl_logging_handler(global_rank=fabric.global_rank)
    if not fabric.is_global_zero:
        logging.set_verbosity(logging.ERROR)
    logging.info(
        "Fabric initialized with world size: [%s]. Flags: [%s]. Seed: [%s]",
        fabric.world_size,
        FLAGS.flags_into_string(),
        config.seed,
    )

    # ------------------ Temporary directories. ------------------ #
    run_name = _RUN_NAME.value or str(round(time.time() * 1000))
    experiments_dir, checkpoint_dir = setup_directories(
        _EXPERIMENT_DIR.value, run_name, fabric.is_global_zero
    )
    config.checkpointing.directory = checkpoint_dir

    # ------------------ Experimental setup. ------------------ #
    use_wandb = _USE_WANDB.value
    # TODO: add sweeps support.
    config = configuration.maybe_setup_wandb_and_update_config(
        config=config,
        global_rank=fabric.global_rank,
        use_wandb=use_wandb,
        experiments_dir=experiments_dir,
        run_name=run_name,
        project_name=_PROJECT_NAME.value,
        log_all_ranks=False,
    )

    # ------------------ Launch training pipeline. ------------------ #
    with custom_logging.Timer() as t:
        run_training(
            fabric=fabric, config=config, use_wandb=use_wandb, dry_run=_DRY_RUN.value
        )
    logging.info(
        "{} finished in {:.2f} seconds.".format(
            "Dry run" if _DRY_RUN.value else "Training", t.get_duration()
        )
    )


if __name__ == "__main__":
    flags.mark_flags_as_required(["config_path"])
    app.run(main)
