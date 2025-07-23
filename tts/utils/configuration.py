import dataclasses
import enum
import json
import os
from typing import Any

import cattr
import wandb

from tts.training import environment

_REQUIRED_CONFIG_KEYS = [
    "train_weighted_datasets",
    "val_weighted_datasets",
    "training",
    "modeling",
    "checkpointing",
]


class TrainingStrategy(enum.Enum):
    """Training strategy."""

    # Distributed Data Parallel (DDP). Used by default.
    DDP = "ddp"

    # Fully Sharded Data Parallel (FSDP).
    # When using FSDP with model compilation (e.g., torch.compile),
    # evaluation must run before training.
    # To support this, make sure checkpoints are saved just before evaluation steps,
    # so that after loading a checkpoint, evaluation runs immediately.
    FSDP = "fsdp"

    # DeepSpeed Stage 2 that uses ZeRO-2 (aka gradient partitioning).
    DEEPSPEED = "deepspeed"


@dataclasses.dataclass(frozen=False)
class DatasetConfig:
    """Defines variable dataset parameters computed upon training start."""

    # List of allowed languages for dataset filtering.
    # If empty or None, no filtering will be done.
    allowed_languages: list[str]
    # Minimum DNSMOS score for dataset filtering.
    min_dnsmos_score: float
    # Minimum sample rate for audio in the dataset filtering.
    min_sample_rate: int
    # Whether to enable RLHF (Reinforcement Learning from Human Feedback) training.
    enable_rlhf_training: bool
    # Minimum audio duration for dataset filtering.
    min_audio_duration: float = 0.0
    # List of allowed instruction finetuning annotations.
    # If empty or None, no instruction finetuning will be done.
    allowed_ift_annotations: list[str] | None = None
    # The total number of training steps needed to traverse the dataset.
    # To be set when loading the dataset.
    total_training_steps: int | None = None
    # The number of steps needed to traverse the dataset once.
    # To be set when loading the dataset.
    steps_per_epoch: int | None = None


@dataclasses.dataclass(frozen=True)
class ModelingConfig:
    # Model architecture parameters.
    parameters: dict[str, Any]


@dataclasses.dataclass(frozen=False)
class CheckpointingConfig:
    # The number of steps to save the model for.
    save_steps: int

    # Checkpoint dir.
    directory: str

    # Collect model health statistics every |save_steps| steps.
    collect_health_stats: bool

    # Whether to save intermediate generations of voice given random
    # validation dataset samples.
    save_intermediate_generations: bool

    # Type of validation to perform during checkpointing ("random_phrases" or
    # "prompt_continuation")
    validation_type: str = "random_phrases"

    # Whether to keep only the last N checkpoints.
    keep_only_last_n_checkpoints: int | None = None

    # If set, the model will resume training from the given checkpoint.
    checkpoint_file_to_resume_from: str | None = None

    # If set, the model will only load the model weights from the given checkpoint.
    only_load_model_weights: bool = False


@dataclasses.dataclass(frozen=True)
class TrainingConfig:
    """Configuration for training."""

    # The global seed for all ranks.
    seed: int

    # The number of steps to perform logging for.
    logging_steps: int

    # The number of steps to do periodic evaluation for.
    eval_steps: int

    # The number of steps to accumulate gradients for.
    gradient_accumulation_steps: int

    # The value to clip gradients to.
    gradient_clip_value: float

    # The initial learning rate.
    learning_rate: float

    # AdamW betas.
    betas: tuple[float, float]

    # The number of warmup steps as a fraction of the total number of steps.
    warmup_ratio: float

    # The batch size.
    batch_size: int

    # The weight decay.s
    weight_decay: float

    # The precision to use for training.
    precision: str

    # The training strategy to use.
    strategy: TrainingStrategy

    # If true, the model will use gradient checkpointing.
    gradient_checkpointing: bool

    # The number of workers to use for data loading.
    num_workers: int = 0


@dataclasses.dataclass(frozen=True)
class RLHFConfig:
    # The directory of the base model.
    base_model_dir: str

    # Float that controls the cumulative probability of the top tokens to consider.
    top_p: float

    # Integer that controls the number of top tokens to consider.
    top_k: int

    # Float that controls the repetition penalty.
    repetition_penalty: float

    # Float that controls the temperature of the logits.
    temperature: float

    # Number of generations per prompt to sample.
    num_generations: int

    # Maximum length of the prompt. If the prompt is longer than this value, it
    # will be truncated left.
    max_prompt_length: int

    # Maximum length of the generated completion.
    max_completion_length: int

    # Minimum length of the generated completion.
    min_completion_length: int

    # Whether to use vllm for training.
    use_vllm: bool

    # The reward functions to use for training.
    reward_funcs: list[str]

    # The weights for the reward functions.
    reward_weights: list[float]

    # The number of steps to save generated audios from completions in reward functions.
    # If set to <=0, no audios will be saved.
    save_completions_steps: int

    # The per-device train batch size.
    per_device_train_batch_size: int

    # The number of iterations per batch.
    num_iterations: int

    # Whether to scale the rewards by dividing them by their standard deviation.
    scale_rewards: bool

    # The KL coefficient with the reference model.
    # If 0.0, the reference model is not loaded,
    # reducing memory usage and improving training speed.
    kl_beta: float


@dataclasses.dataclass(frozen=True)
class LoraConfig:
    # The task type for LoRA.
    task_type: str
    # The rank of the LoRA attention dimension.
    r: int
    # The alpha parameter for the LoRA scaling.
    lora_alpha: int
    # The target modules to apply LoRA to.
    target_modules: list[str]
    # The dropout probability for the LoRA layers.
    lora_dropout: float
    # The bias type for LoRA.
    bias: str
    # The path to the adapter weights.
    # If not provided, the adapter will be initialized randomly.
    adapter_path: str | None = None

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "LoraConfig":
        return cattr.structure(data, LoraConfig)


@dataclasses.dataclass(frozen=True)
class CodecTrainingConfig:
    # the size of the window in vocoder training.
    audio_window_size: int

    # sample rate of the codec decoder.
    sample_rate: int

    # hop_length of the codec decoder.
    hop_length: int

    # minimum sample rate of the data to use for training.
    minimum_data_sample_rate: int = 24000

    # upsample factors of the codec decoder.
    upsample_factors: list[int] | None = None

    # kernel sizes of the codec decoder.
    kernel_sizes: list[int] | None = None


@dataclasses.dataclass(frozen=False)
class ExperimentConfig:
    """Configuration for an experiment."""

    # Static variables.
    training: TrainingConfig
    modeling: ModelingConfig
    checkpointing: CheckpointingConfig

    # The list of datasets to use for training with the associated
    # number of epochs to run on each dataset.
    train_weighted_datasets: dict[str, float]
    val_weighted_datasets: dict[str, float]

    # Dynamic variables.
    rlhf_training: RLHFConfig | None = None
    lora: LoraConfig | None = None
    codec: CodecTrainingConfig | None = None
    dataset: DatasetConfig | None = None
    seed: int | None = None
    world_size: int | None = None
    model_size: int | None = None
    vocab_size: int | None = None
    wandb_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return cattr.unstructure(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ExperimentConfig":
        return cattr.structure(data, ExperimentConfig)

    @staticmethod
    def from_json(file: str | os.PathLike) -> "ExperimentConfig":
        """Read config from json file."""

        with open(file) as f:
            config = json.load(f)

        for key in _REQUIRED_CONFIG_KEYS:
            if key not in config:
                # ignore missing modeling config if codec config is specified.
                if key == "modeling" and "codec" in config:
                    continue
                raise ValueError(f"Missing {key} specification in the config file!")

        # Reset dynamic variables.
        config["seed"] = None
        config["world_size"] = None
        config["model_size"] = None
        config["vocab_size"] = None
        config["wandb_url"] = None
        config["checkpointing"]["directory"] = None

        return ExperimentConfig.from_dict(config)

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=4, default=str)


def maybe_setup_wandb_and_update_config(
    config: ExperimentConfig,
    global_rank: int,
    use_wandb: bool,
    experiments_dir: str,
    run_name: str | None = None,
    project_name: str | None = None,
    log_all_ranks: bool = False,
) -> ExperimentConfig:
    """Setup wandb for logging if enabled."""
    if use_wandb:
        if not project_name:
            if "WANDB_PROJECT" in os.environ:
                project_name = os.environ["WANDB_PROJECT"]
            else:
                project_name = "inworld_{}".format(os.environ["USER"])

        # Even in a multi-processing setup each process need to initialize WandB
        # client and assign a unique run name. One can always single-node
        # runs in the UI and keep information about each node/process health.
        name, group = run_name, None
        if log_all_ranks and global_rank != environment.EnvironmentContext.DEFAULT_RANK:
            name = f"{run_name}_{global_rank}"
            group = run_name
        if global_rank == 0 or log_all_ranks:
            wandb.init(
                dir=experiments_dir,
                project=project_name,
                name=name,
                group=group,
                config=config.to_dict(),
            )

    return config
