import contextlib
import json
import os
import time

import lightning.fabric as lightning_fabric
import torch
import transformers
from absl import logging

from tts.core import constants, lora
from tts.data import caching
from tts.utils import configuration

_ATTN_IMPLEMENTATION = "flash_attention_2"


def _str_to_torch_dtype(dtype: str) -> torch.dtype:
    """Converts a string to a torch dtype."""
    str_to_dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "int8": torch.int8,
    }
    if dtype not in str_to_dtype:
        raise ValueError(f"Unknown dtype [{dtype}].")
    return str_to_dtype[dtype]


def _construct_model(
    model_name: str, torch_dtype: torch.dtype | str, vocab_size: int, cache_dir: str
) -> torch.nn.Module:
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        attn_implementation=_ATTN_IMPLEMENTATION,
        cache_dir=cache_dir,
    )

    current_model_vocab_size = model.model.embed_tokens.weight.size(0)
    logging.info("[%s] vocab size: [%d].", model_name, current_model_vocab_size)

    if current_model_vocab_size != vocab_size:
        logging.info(
            "Resizing model from [%d] to [%d]...", current_model_vocab_size, vocab_size
        )
        model.resize_token_embeddings(vocab_size)
        model.vocab_size = vocab_size
        if vocab_size != model.model.embed_tokens.weight.size(0):
            raise ValueError("Embedding size mismatch!")
        if vocab_size != model.lm_head.weight.size(0):
            raise ValueError("LM head size mismatch!")

    return model


def load_tokenizer_config_and_model(
    checkpoint_path: str,
) -> tuple[
    transformers.PreTrainedTokenizerBase,
    dict,
    torch.nn.Module,
    configuration.LoraConfig | None,
]:
    """Loads tokenizer, config, and model from a checkpoint.

    This is a common pattern used across multiple scripts that load models
    from checkpoints. It handles loading the tokenizer, training config,
    extracting LoRA config if present, and loading the model.

    Args:
        checkpoint_path: Path to the checkpoint file or directory.
        set_pad_token: Whether to set the tokenizer's pad_token to eos_token.

    Returns:
        Tuple of (tokenizer, config, model, lora_config).
    """
    checkpoint_dir = (
        checkpoint_path
        if os.path.isdir(checkpoint_path)
        else os.path.dirname(checkpoint_path)
    )

    # Load tokenizer
    start_time = time.time()
    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint_dir)
    logging.info(
        "Tokenizer with size: %d loaded in %.2f seconds.",
        len(tokenizer),
        time.time() - start_time,
    )

    # Load training config
    training_config_file = os.path.join(checkpoint_dir, constants.CONFIG_FILE_NAME)
    with open(training_config_file) as f:
        config = json.load(f)

    # Extract LoRA config if present
    lora_config = None
    if "lora" in config and config["lora"] is not None:
        lora_config = configuration.LoraConfig.from_dict(config["lora"])

    # Load model
    start_time = time.time()
    model = load_model_from_checkpoint(
        model_name=config["modeling"]["parameters"]["model_name"],
        vocab_size=len(tokenizer),
        checkpoint_path=checkpoint_path,
        precision=config["training"]["precision"],
        lora_config=lora_config,
    )
    logging.info("Model loaded in %.2f seconds.", time.time() - start_time)

    return tokenizer, config, model, lora_config


def build_model(
    fabric: lightning_fabric.Fabric,
    model_name: str,
    precision: str,
    vocab_size: int,
    deepspeed: bool = False,
    gradient_checkpointing: bool = False,
) -> torch.nn.Module:
    """Builds a TTS model on top of a pre-trained LLM."""
    cache_dir = caching.get_hf_cache_dir()
    logging.info(
        "Loading pretrained model [%s] using the following HF cache directory: [%s]...",
        model_name,
        cache_dir,
    )

    # TODO: debug why using DDP multi-GPU training leaks memory.
    model_init_context = contextlib.nullcontext()
    if fabric.world_size == 1 or deepspeed:
        model_init_context = fabric.init_module()
    with model_init_context:
        model = _construct_model(
            model_name, _str_to_torch_dtype(precision), vocab_size, cache_dir
        )
    if gradient_checkpointing:
        logging.info("Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable()

    return model


def load_model_from_checkpoint(
    model_name: str,
    vocab_size: int,
    checkpoint_path: str,
    precision: str,
    lora_config: dict | None = None,
) -> torch.nn.Module:
    """Loads a model from a checkpoint file to CPU."""
    cache_dir = caching.get_hf_cache_dir()
    torch_dtype = _str_to_torch_dtype(precision)
    logging.info(
        "Loading model from [%s] using the following HF cache directory: [%s]...",
        checkpoint_path,
        cache_dir,
    )
    model = _construct_model(model_name, torch_dtype, vocab_size, cache_dir)
    if lora_config is not None:
        model = lora.apply_lora(model, lora_config)
        logging.info("Applied LoRA adapter [%s] to the base model.", lora_config)
    checkpoint = torch.load(checkpoint_path, weights_only=False)["model"]
    model.load_state_dict(checkpoint)
    return model
