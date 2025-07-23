"""Main entry point for the Inworld Text-to-Speech RLHF training pipeline."""

import os
import time
from collections.abc import Sequence

import torch
from absl import app, flags, logging
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from tts.core import constants
from tts.data import text_normalization, tts_datasets
from tts.training import checkpointing
from tts.training.rlhf import rewards
from tts.utils import configuration, custom_logging

FLAGS = flags.FLAGS

_CONFIG_PATH = flags.DEFINE_string(
    "config_path", None, "Path to the training config json file."
)
_RUN_NAME = flags.DEFINE_string("run_name", None, "Run name for wandb logging.")
_EXPERIMENT_DIR = flags.DEFINE_string(
    "experiment_dir", "./rlhf_experiments", "Experiment directory."
)
# Wandb logging flags.
_PROJECT_NAME = flags.DEFINE_string(
    "project_name", None, "Project name for wandb logging."
)
_USE_WANDB = flags.DEFINE_bool("use_wandb", False, "Whether to use wandb for logging.")
_VLLM_SERVER_HOST = flags.DEFINE_string(
    "vllm_server_host", None, "Hostname or IP address of the vLLM server."
)
_VLLM_SERVER_TIMEOUT = flags.DEFINE_integer(
    "vllm_server_timeout", 1800, "Timeout for the vLLM server."
)

_COMPLETION_SAVE_SUB_DIR = "completions"


def setup_directories(
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
    config: configuration.ExperimentConfig, use_wandb: bool, run_name: str, rank: int
) -> None:
    """Launches the training pipeline."""
    modeling_params = config.modeling.parameters
    max_seq_len = modeling_params["max_seq_len"]

    # ------------------ Tokenizer ------------------ #
    tokenizer = AutoTokenizer.from_pretrained(config.rlhf_training.base_model_dir)
    if rank == 0:
        tokenizer.save_pretrained(config.checkpointing.directory)
        logging.info(
            "Tokenizer saved to [%s] with vocab size [%d]",
            config.checkpointing.directory,
            len(tokenizer),
        )

    # ------------------ Data setup. ------------------ #
    logging.info("Starting data setup. Max seq len: %d.", max_seq_len)
    enable_text_normalization = modeling_params.get("enable_text_normalization", False)
    text_normalizer = text_normalization.create_text_normalizer(
        enable_text_normalization
    )
    with custom_logging.Timer():
        train_dataset = tts_datasets.merge_datasets(
            tokenizer=tokenizer,
            weighted_datasets=config.train_weighted_datasets,
            max_seq_len=max_seq_len,
            split=constants.TRAIN_SPLIT,
            pretraining_mode=False,
            text_normalizer=text_normalizer,
            dataset_config=config.dataset,
        )
        _ = tts_datasets.merge_datasets(
            tokenizer=tokenizer,
            weighted_datasets=config.val_weighted_datasets,
            max_seq_len=max_seq_len,
            split=constants.VAL_SPLIT,
            pretraining_mode=False,
            text_normalizer=text_normalizer,
            dataset_config=config.dataset,
        )

    # ------------------ Save config. ------------------ #
    checkpointing.save_config(
        config, config.checkpointing.directory, use_wandb, is_global_zero=rank == 0
    )
    logging.info(
        "Model config [%s] has been saved to [%s].",
        config,
        config.checkpointing.directory,
    )

    # ------------------ Training. ------------------ #
    logging.info("Starting training...")

    grpo_config = GRPOConfig(
        output_dir=config.checkpointing.directory,
        run_name=run_name,
        learning_rate=config.training.learning_rate,
        warmup_ratio=config.training.warmup_ratio,
        adam_beta1=config.training.betas[0],
        adam_beta2=config.training.betas[1],
        weight_decay=config.training.weight_decay,
        logging_steps=config.training.logging_steps,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        save_steps=config.checkpointing.save_steps,
        top_p=config.rlhf_training.top_p,
        top_k=config.rlhf_training.top_k,
        repetition_penalty=config.rlhf_training.repetition_penalty,
        temperature=config.rlhf_training.temperature,
        num_generations=config.rlhf_training.num_generations,
        max_prompt_length=config.rlhf_training.max_prompt_length,
        max_completion_length=config.rlhf_training.max_completion_length,
        min_completion_length=config.rlhf_training.min_completion_length,
        per_device_train_batch_size=config.rlhf_training.per_device_train_batch_size,
        reward_weights=config.rlhf_training.reward_weights,
        num_iterations=config.rlhf_training.num_iterations,
        scale_rewards=config.rlhf_training.scale_rewards,
        beta=config.rlhf_training.kl_beta,
        use_vllm=config.rlhf_training.use_vllm,
        vllm_server_timeout=_VLLM_SERVER_TIMEOUT.value,
        vllm_server_host=_VLLM_SERVER_HOST.value,
    )

    reward_funcs = rewards.create_reward_funcs(
        reward_func_names=config.rlhf_training.reward_funcs,
        tokenizer=tokenizer,
        device=torch.device(f"cuda:{rank%8}"),
        save_completions_steps=config.rlhf_training.save_completions_steps,
        save_dir=os.path.join(config.checkpointing.directory, _COMPLETION_SAVE_SUB_DIR),
        logging_steps=config.training.logging_steps,
    )
    if len(config.rlhf_training.reward_weights) != len(
        config.rlhf_training.reward_funcs
    ):
        logging.error(
            "reward_weights length (%d) does not match reward_funcs length "
            "(%d). Adjusting weights.",
            len(config.rlhf_training.reward_weights),
            len(config.rlhf_training.reward_funcs),
        )

    # Both base model and reference model are loaded from base_model_dir
    trainer = GRPOTrainer(
        model=config.rlhf_training.base_model_dir,
        args=grpo_config,
        reward_funcs=reward_funcs,
        processing_class=tokenizer,
        train_dataset=train_dataset,
    )
    trainer.train()


def main(argv: Sequence[str]) -> None:
    del argv  # Unused.

    # ------------------ Config. ------------------ #
    config = configuration.ExperimentConfig.from_json(file=_CONFIG_PATH.value)

    # ------------------ Accelerator Ranks. ------------------ #
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    logging.info(f"accelerator rank: {rank}, world size: {world_size}")
    # ------------------ Temporary directories. ------------------ #
    run_name = _RUN_NAME.value or str(round(time.time() * 1000))
    experiments_dir, checkpoint_dir = setup_directories(
        _EXPERIMENT_DIR.value, run_name, is_global_zero=rank == 0
    )
    config.checkpointing.directory = checkpoint_dir

    # ------------------ Experimental setup. ------------------ #
    use_wandb = _USE_WANDB.value
    config = configuration.maybe_setup_wandb_and_update_config(
        config=config,
        global_rank=rank,
        use_wandb=use_wandb,
        experiments_dir=experiments_dir,
        run_name=run_name,
        project_name=_PROJECT_NAME.value,
        log_all_ranks=False,
    )

    # ------------------ Launch training pipeline. ------------------ #
    with custom_logging.Timer() as t:
        run_training(config=config, use_wandb=use_wandb, run_name=run_name, rank=rank)
    logging.info(f"Finished RLHF training in {t.get_duration():.2f} seconds.")


if __name__ == "__main__":
    flags.mark_flags_as_required(["config_path"])
    app.run(main)
