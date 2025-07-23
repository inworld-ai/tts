import peft
import torch
from absl import logging

from tts.utils import configuration


def _find_linear_modules(model: torch.nn.Module) -> list[str]:
    """Finds all linear components within a model."""
    suffixes = set()
    for name, param in model.named_modules():
        if isinstance(param, torch.nn.Linear):
            suffixes.add(name.split(".")[-1])
    if len(suffixes) == 0:
        raise ValueError(
            f"Model {type(model)} contains no linear component, "
            "LoRA can't be applied."
        )
    return list(suffixes)


def apply_lora(
    model: torch.nn.Module, config: configuration.LoraConfig
) -> torch.nn.Module:
    """Applies LoRA to the model."""
    if config.adapter_path is not None:
        model = load_adapter(model, config.adapter_path)
        for name, param in model.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True
        return model

    if config.target_modules:
        target_modules = list(config.target_modules)
    else:
        # When target_modules aren't provided, automatically find all linear components.
        target_modules = _find_linear_modules(model)
        logging.info("No |target_modules| provided, using all linear layers instead.")
    peft_config = peft.LoraConfig(
        task_type=config.task_type,
        r=config.r,
        lora_alpha=config.lora_alpha,
        target_modules=target_modules,
        lora_dropout=config.lora_dropout,
        bias=config.bias,
    )
    return peft.get_peft_model(model, peft_config)


def load_adapter(base_model: torch.nn.Module, adapter_path: str) -> torch.nn.Module:
    """Loads adapter weights from directory."""
    return peft.PeftModel.from_pretrained(base_model, adapter_path, device_map="auto")


def save_lora_adapter(model: torch.nn.Module, adapter_path: str):
    """Saves the LoRA adapter to a directory."""
    model.save_pretrained(adapter_path)
