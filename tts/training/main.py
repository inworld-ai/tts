"""Main entry point for the Inworld Text-to-Speech training pipeline."""

import os
import time
from collections.abc import Sequence

import lightning.fabric as lightning_fabric
import torch
import wandb
from absl import app, flags, logging

from tts.core import constants, lora, modeling, optimization, tokenization
from tts.data import text_normalization, tts_datasets
from tts.inference import quality_validation
from tts.training import checkpointing, environment, training_loop
from tts.utils import configuration, custom_logging

FLAGS = flags.FLAGS

_CONFIG_PATH = flags.DEFINE_string(
    "config_path", None, "Path to the training config json file."
)
_SLURM_DISTRIBUTED = flags.DEFINE_bool(
    "slurm_distributed", False, "Whether the script is running via SLURM."
)
_RUN_NAME = flags.DEFINE_string("run_name", None, "Run name for wandb logging.")
_DRY_RUN = flags.DEFINE_bool("dry_run", False, "Whether to run in dry run mode.")
_EXPERIMENT_DIR = flags.DEFINE_string(
    "experiment_dir", "./experiments", "Experiment directory."
)
_PRETRAINING_MODE = flags.DEFINE_bool(
    "pretraining_mode", False, "Whether to run in pretraining mode."
)
_COMPILE_MODEL = flags.DEFINE_bool("compile_model", False, "Whether to compile model.")
_SEED = flags.DEFINE_integer("seed", 777, "Random seed.")

# Wandb logging flags.
_PROJECT_NAME = flags.DEFINE_string(
    "project_name", None, "Project name for wandb logging."
)
_USE_WANDB = flags.DEFINE_bool("use_wandb", False, "Whether to use wandb for logging.")


def _setup_directories(
    experiment_dir: str, checkpoint_dir: str, is_global_zero: bool
) -> tuple[str, str]:
    """Makes path absolute and creates directory if it doesn't exist."""
    experiment_dir = os.path.abspath(experiment_dir)
    checkpoint_dir = os.path.join(experiment_dir, checkpoint_dir)
    if not os.path.exists(checkpoint_dir) and is_global_zero:
        os.makedirs(checkpoint_dir)
        logging.info("Checkpoint directory was created: %s.", checkpoint_dir)

    return experiment_dir, checkpoint_dir


def run_training(
    fabric: lightning_fabric.Fabric,
    config: configuration.ExperimentConfig,
    use_wandb: bool,
    pretraining_mode: bool,
    dry_run: bool,
) -> None:
    """Launches the training pipeline."""
    modeling_params = config.modeling.parameters
    max_seq_len = modeling_params["max_seq_len"]
    is_lora = config.lora is not None

    # ------------------ Tokenizer ------------------ #
    tokenizer = tokenization.build_tokenizer(
        model_name=modeling_params["model_name"],
        max_seq_len=max_seq_len,
        codebook_size=modeling_params["codebook_size"],
        is_lora=is_lora,
    )

    if fabric.is_global_zero:
        with custom_logging.Timer() as t:
            tokenizer.save_pretrained(config.checkpointing.directory)
        logging.info(
            "Tokenizer saved to %s in %.2f seconds.",
            config.checkpointing.directory,
            t.get_duration(),
        )

    # ------------------ Model ------------------ #
    vocab_size = len(tokenizer)
    with custom_logging.Timer() as t:
        deepspeed = config.training.strategy == configuration.TrainingStrategy.DEEPSPEED
        model = modeling.build_model(
            fabric,
            modeling_params["model_name"],
            config.training.precision,
            vocab_size,
            deepspeed=deepspeed,
            gradient_checkpointing=config.training.gradient_checkpointing,
        )
        if is_lora:
            model = lora.apply_lora(model, config.lora)
            logging.info("Applied LoRA adapter to the base model.")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    config.model_size = num_params
    config.vocab_size = vocab_size
    logging.info(
        "Model with [%.4f]M parameters has been loaded in %.2f seconds.",
        num_params / 1e6,
        t.get_duration(),
    )

    # ------------------ Data setup. ------------------ #
    logging.info("Starting data setup. Max seq len: %d.", max_seq_len)
    enable_text_normalization = modeling_params.get("enable_text_normalization", False)
    text_normalizer = text_normalization.create_text_normalizer(
        enable_text_normalization
    )
    with custom_logging.Timer() as t:
        collate_fn = tts_datasets.get_collate_fn(tokenizer.pad_token_id)
        train_dataset = tts_datasets.merge_datasets(
            tokenizer=tokenizer,
            weighted_datasets=config.train_weighted_datasets,
            max_seq_len=max_seq_len,
            split=constants.TRAIN_SPLIT,
            pretraining_mode=pretraining_mode,
            text_normalizer=text_normalizer,
            dataset_config=config.dataset,
        )
        val_dataset = tts_datasets.merge_datasets(
            tokenizer=tokenizer,
            weighted_datasets=config.val_weighted_datasets,
            max_seq_len=max_seq_len,
            split=constants.VAL_SPLIT,
            pretraining_mode=pretraining_mode,
            text_normalizer=text_normalizer,
            dataset_config=config.dataset,
        )
        train_data_loader = tts_datasets.get_dataloader(
            dataset=train_dataset,
            batch_size=config.training.batch_size,
            collate_fn=collate_fn,
            shuffle=True,
            num_workers=config.training.num_workers,
        )
        val_data_loader = tts_datasets.get_dataloader(
            dataset=val_dataset,
            batch_size=config.training.batch_size,
            collate_fn=collate_fn,
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
        "Datasets were loaded in %.2f seconds. Config: %s.",
        t.get_duration(),
        config.dataset,
    )

    # ------------------ Save config. ------------------ #
    if fabric.is_global_zero:
        config.wandb_url = wandb.run.url if use_wandb else None
        checkpointing.save_config(config, config.checkpointing.directory, use_wandb)
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
    optimizer = optimization.create_optimizer(
        model=model,
        learning_rate=config.training.learning_rate,
        betas=config.training.betas,
        weight_decay=config.training.weight_decay,
    )
    logging.info(
        "%s optimizer and cosine LR scheduler created with %d warmup steps.",
        optimizer.__class__.__name__,
        warmup_steps,
    )

    train_data_loader = fabric.setup_dataloaders(train_data_loader)
    val_data_loader = fabric.setup_dataloaders(val_data_loader)
    if _COMPILE_MODEL.value:
        model = torch.compile(model)
    model, optimizer = fabric.setup(model, optimizer)
    model.mark_forward_method("generate")
    if dry_run:
        data_sample = tts_datasets.prettify_data_sample(next(iter(train_data_loader)))
        print(model(**data_sample).loss)
        print("Dry run completed successfully!")
        return

    # ------------------ Quality validator. ------------------ #
    with custom_logging.Timer() as t:
        quality_validator = quality_validation.create_quality_validator(
            tokenizer=tokenizer,
            checkpointing_dir=config.checkpointing.directory,
            save_intermediate_generations=config.checkpointing.save_intermediate_generations,
            global_rank=fabric.global_rank,
            world_size=fabric.world_size,
            device=fabric.device,
            validation_type=config.checkpointing.validation_type,
        )
    logging.info("Quality validator created in %.2f seconds.", t.get_duration())

    # ------------------ Training. ------------------ #
    logging.info("Starting training...")
    final_statistics = training_loop.run(
        fabric=fabric,
        model=model,
        train_data_loader=train_data_loader,
        train_data_loader_iterator=iter(train_data_loader),
        config=config,
        optimizer=optimizer,
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
        optimizer,
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
    )
    config.world_size = fabric.world_size
    fabric.seed_everything(config.seed)
    fabric.barrier()
    custom_logging.reconfigure_absl_logging_handler(global_rank=fabric.global_rank)
    if not fabric.is_global_zero:
        logging.set_verbosity(logging.ERROR)
    logging.info(
        "Fabric initialized with world size: [%s]. Flags: [%s]. Seed: [%s].",
        fabric.world_size,
        FLAGS.flags_into_string(),
        config.seed,
    )

    # ------------------ Temporary directories. ------------------ #
    run_name = _RUN_NAME.value or str(round(time.time() * 1000))
    experiments_dir, checkpoint_dir = _setup_directories(
        _EXPERIMENT_DIR.value, run_name, fabric.is_global_zero
    )
    config.checkpointing.directory = checkpoint_dir

    # ------------------ Experimental setup. ------------------ #
    use_wandb = _USE_WANDB.value
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
            fabric=fabric,
            config=config,
            use_wandb=use_wandb,
            pretraining_mode=_PRETRAINING_MODE.value,
            dry_run=_DRY_RUN.value,
        )
    logging.info(
        "{} finished in {:.2f} seconds.".format(
            "Dry run" if _DRY_RUN.value else "Training", t.get_duration()
        )
    )


if __name__ == "__main__":
    flags.mark_flags_as_required(["config_path"])
    app.run(main)
