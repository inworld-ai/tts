import os

import lightning.fabric as lightning_fabric
import torch
import wandb
from absl import logging

from tts.core import constants
from tts.utils import configuration, custom_logging


# TODO: consider loading the tokenizer too to minimize human mistakes.
# TODO: investigate why using this method with DDP reduces VRAM usage a bit.
def load_from_checkpoint(
    fabric: lightning_fabric.Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_file_to_resume_from: str,
    load_full_checkpoint: bool = True,
) -> tuple[torch.nn.Module, custom_logging.Statistics, torch.optim.Optimizer]:
    """Loads the most appropriate checkpoint file and updates model state."""
    checkpoint = {"model": model}
    if load_full_checkpoint:
        checkpoint.update({"optimizer": optimizer, "loss_statistics": {}})

    fabric.load(checkpoint_file_to_resume_from, checkpoint, strict=True)
    statistics = None
    if load_full_checkpoint:
        statistics = custom_logging.Statistics.from_dict(checkpoint["loss_statistics"])

    return model, statistics, optimizer


def save_to_checkpoint(
    fabric: lightning_fabric.Fabric,
    model: torch.nn.Module,
    config: configuration.ExperimentConfig,
    optimizer: torch.optim.Optimizer,
    statistics: custom_logging.Statistics,
    checkpoint_name: str | None = None,
) -> str:
    """Saves the model and training state to a checkpoint."""
    checkpoint_name = checkpoint_name or f"checkpoint_{statistics.step}.pt"
    checkpoint_file = os.path.join(config.checkpointing.directory, checkpoint_name)

    checkpoint = {
        "model": model,
        "loss_statistics": statistics.as_dict(),
        "optimizer": optimizer,
        "config": config.to_dict(),
    }
    fabric.save(path=checkpoint_file, state=checkpoint)

    if fabric.is_global_zero:
        keep_only_last_n_checkpoints = config.checkpointing.keep_only_last_n_checkpoints
        if keep_only_last_n_checkpoints is not None:
            checkpoint_files = [
                f
                for f in os.listdir(config.checkpointing.directory)
                if f.startswith("checkpoint_") and f.endswith(".pt")
            ]
            checkpoint_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            for f in checkpoint_files[:-keep_only_last_n_checkpoints]:
                logging.info("Removing too old checkpoint %s...", f)
                os.remove(os.path.join(config.checkpointing.directory, f))

    return checkpoint_file


def save_config(
    experiment_config: configuration.ExperimentConfig,
    checkpoint_dir: str,
    use_wandb: bool,
):
    """Saves model config to a file."""
    config_file = os.path.join(checkpoint_dir, constants.CONFIG_FILE_NAME)
    with open(config_file, "w") as f:
        f.write(str(experiment_config))

    # Config might be with new values after first initialized, to ensure
    # consistency, the config here should be updated with wandb.
    #
    # TODO: sweep run set value can be overriden by python's training code
    #       leading to config here and one shown in the W&B UI being different.
    if use_wandb:
        wandb.config.update(experiment_config.to_dict(), allow_val_change=True)
