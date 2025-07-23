import dataclasses
import os
import subprocess
import sys
from typing import ClassVar

import lightning.fabric as lightning_fabric
import regex as re
import torch
from absl import logging
from lightning.fabric import strategies
from lightning.fabric.plugins.precision import fsdp
from transformers.utils import import_utils as transformers_utils


def _get_slurm_config() -> tuple[int, int, int]:
    """Returns the configuration of the SLURM job."""
    if not torch.cuda.is_available():
        raise RuntimeError("SLURM distributed training requires CUDA.")

    global_rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NPROCS"])
    local_rank = global_rank % torch.cuda.device_count()
    return global_rank, local_rank, world_size


def _get_cuda_version_from_nvidia_smi() -> str:
    """Returns the CUDA version as reported by nvidia-smi."""
    try:
        output = subprocess.check_output(["nvidia-smi"]).decode("utf-8")
        match = re.search(r"CUDA Version:\s+(\d+\.\d+)", output)
        if match:
            return match.group(1)
        else:
            return "N/A"
    except Exception:  # pylint: disable=broad-exception-caught
        return "Unavailable"


def _init_hardware(local_rank: int) -> torch.device:
    """Configures training hardware and returns its python representation."""
    if not torch.cuda.is_available():
        logging.warning("CUDA is not available. Using CPU.")
        return torch.device("cpu")

    # Use tf32 where possible as per recommendations from
    # https://huggingface.co/docs/transformers/v4.15.0/performance#tf32
    torch.cuda.empty_cache()
    torch.backends.cudnn.enabled = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Use the same device as the current process.
    local_rank = local_rank if local_rank >= 0 else 0
    return torch.device("cuda", index=local_rank)


def _get_fabric_precision(training_precision: str) -> str:
    """Returns the precision to use for the lightning fabric."""
    if training_precision == "bf16":
        return "bf16-true"

    raise ValueError(f"Unsupported precision: {training_precision}")


def _get_strategy(
    training_precision: str, strategy_name: str, find_unused_parameters: bool
) -> strategies.Strategy:
    """Returns the strategy to use for training."""
    if strategy_name == "ddp":
        if torch.cuda.device_count() == 1:
            logging.info(
                "Using SingleDeviceStrategy strategy for single-GPU training..."
            )
            return strategies.SingleDeviceStrategy(
                device="cuda:0", precision=training_precision
            )
        logging.info("Using DDPStrategy strategy for multi-GPU training...")
        return strategies.DDPStrategy(
            precision=training_precision, find_unused_parameters=find_unused_parameters
        )

    elif strategy_name == "fsdp":
        logging.info("Using FSDP strategy for multi-GPU training...")
        # TODO: make settings configurable from the config file.
        precision = fsdp.FSDPPrecision(
            precision=_get_fabric_precision(training_precision)
        )
        return strategies.FSDPStrategy(
            precision=precision,
            cpu_offload=False,
            backward_prefetch=torch.distributed.fsdp.BackwardPrefetch.BACKWARD_POST,
            use_orig_params=True,
            limit_all_gathers=True,
            state_dict_type="full",
            sync_module_states=True,
        )

    elif strategy_name == "deepspeed":
        logging.info("Using DeepSpeed strategy for multi-GPU training...")
        return strategies.DeepSpeedStrategy(
            precision=_get_fabric_precision(training_precision)
        )

    raise ValueError(f"Unsupported strategy: {strategy_name}")


@dataclasses.dataclass(frozen=True)
class EnvironmentContext:
    """Encapsulates the configuration of a distributed training/serving run.

    Args:
        local_rank: The local rank of the process.
        global_rank: The global rank of the process.
        world_size: The total number of processes.
        device: The device to use for the process.
        slurm_distributed: If the run is distributed via SLURM.
        dry_run: If the run is "dry" in which no heavy-weight operations are performed.
    """

    local_rank: int
    global_rank: int
    world_size: int
    device: torch.device
    slurm_distributed: bool
    dry_run: bool
    DEFAULT_RANK: ClassVar[int] = -1

    def is_main_process(self) -> bool:
        """Returns true if this process is the main process."""
        return self.global_rank in {self.DEFAULT_RANK, 0}


def initialize_distributed_environment_context(
    local_rank: int | None = None,
    slurm_distributed: bool = False,
    dry_run: bool = False,
) -> EnvironmentContext:
    """Initializes the distributed training context."""
    cuda_build_version, cuda_runtime_version = "N/A", "N/A"
    if torch.cuda.is_available():
        cuda_build_version = torch.version.cuda
        cuda_runtime_version = _get_cuda_version_from_nvidia_smi()
        if not transformers_utils.is_flash_attn_2_available():
            raise ValueError("Flash attention 2 is not available! Install it!")

    python_version = str(sys.version).replace("\n", "")
    logging.info(
        "Initializing environment with: torch [%s], "
        "cuda-built [%s], cuda-runtime [%s], python [%s]",
        torch.__version__,
        cuda_build_version,
        cuda_runtime_version,
        python_version,
    )

    if local_rank is None:
        if slurm_distributed:
            global_rank, local_rank, world_size = _get_slurm_config()
        else:
            local_rank = EnvironmentContext.DEFAULT_RANK
            global_rank = EnvironmentContext.DEFAULT_RANK

            # Support lightning CLI distributed training on a single node.
            if (
                os.environ.get("LOCAL_RANK", EnvironmentContext.DEFAULT_RANK)
                != local_rank
            ):
                local_rank = int(os.environ["LOCAL_RANK"])
            if os.environ.get("RANK", EnvironmentContext.DEFAULT_RANK) != global_rank:
                global_rank = int(os.environ["RANK"])

            # Try to find world size from environment variables.
            world_size = os.environ.get("PMI_SIZE", None) or os.environ.get(
                "WORLD_SIZE", None
            )
            if world_size is None:
                logging.warning("World size is not set. Assuming 1.")
                world_size = 1
            else:
                world_size = int(world_size)
    else:
        global_rank = local_rank

    return EnvironmentContext(
        local_rank=local_rank,
        global_rank=global_rank,
        world_size=world_size,
        device=_init_hardware(local_rank),
        slurm_distributed=slurm_distributed,
        dry_run=dry_run,
    )


def initialize_fabric(
    env_context: EnvironmentContext,
    strategy_name: str,
    training_precision: str = "32-true",
    find_unused_parameters: bool = False,
) -> lightning_fabric.Fabric:
    """Initializes lightning's fabric."""
    strategy = _get_strategy(training_precision, strategy_name, find_unused_parameters)
    if env_context.slurm_distributed:
        for env_var in ["SLURM_JOB_NUM_NODES", "SLURM_NTASKS_PER_NODE"]:
            if env_var not in os.environ:
                raise ValueError(f"Environment variable {env_var} is not set.")

        logging.info("Start syncing distributed SLURM processes via fabric...")
        logging.info(f"SLURM_JOB_NUM_NODES: {os.environ['SLURM_JOB_NUM_NODES']}")
        logging.info(f"SLURM_NTASKS_PER_NODE: {os.environ['SLURM_NTASKS_PER_NODE']}")
        fabric = lightning_fabric.Fabric(
            precision=training_precision,
            accelerator="gpu",
            strategy=strategy,
            devices=int(os.environ["SLURM_NTASKS_PER_NODE"]),
            num_nodes=int(os.environ["SLURM_JOB_NUM_NODES"]),
        )
        fabric.launch()
    else:
        # Use lightning CLI to run on one node and several GPUs like
        # `fabric run --accelerator=gpu --devices=2 ...`
        fabric = lightning_fabric.Fabric(accelerator="gpu", strategy=strategy)

    return fabric
