import transformers
from absl import logging

from tts.core import constants
from tts.data import caching

# Llama 3.1 8B Instruct and Llama 3.2 1B Instruct with speech tokens.
_EXPECTED_VOCAB_SIZE = 193856


def build_tokenizer(
    model_name: str, max_seq_len: int, codebook_size: int, is_lora: bool
) -> transformers.AutoTokenizer:
    """Creates a tokenizer for the model."""
    cache_dir = caching.get_hf_cache_dir()
    logging.info(
        "Loading tokenizer using the following HF cache directory: [%s]...", cache_dir
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=max_seq_len,
        padding_side="right",
        cache_dir=cache_dir,
    )
    tokenizer.pad_token = tokenizer.eos_token
    original_vocab_size = len(tokenizer)
    logging.info(
        "Loaded [%s] tokenizer with [%d] tokens.", model_name, original_vocab_size
    )

    expected_vocab_size = _EXPECTED_VOCAB_SIZE
    if original_vocab_size == expected_vocab_size:
        logging.info("Tokenizer already has the correct size.")
        return tokenizer

    new_tokens = [
        constants.SPEECH_START_TOKEN,
        constants.SPEECH_END_TOKEN,
        constants.TEXT_PROMPT_START_TOKEN,
        constants.TEXT_PROMPT_END_TOKEN,
        constants.VOICE_DESCRIPTION_START_TOKEN,
        constants.VOICE_DESCRIPTION_END_TOKEN,
        constants.SOUND_EFFECT_START_TOKEN,
        constants.SOUND_EFFECT_END_TOKEN,
    ]
    new_tokens.extend(
        [constants.SPEECH_TOKEN_PATTERN.format(i) for i in range(codebook_size)]
    )
    num_added_tokens = tokenizer.add_tokens(sorted(new_tokens))
    new_vocab_size = len(tokenizer)

    if new_vocab_size < expected_vocab_size:
        num_extra_tokens = expected_vocab_size - new_vocab_size
        extra_tokens = [f"<|extra_token_{i}|>" for i in range(num_extra_tokens)]
        num_added_tokens += tokenizer.add_tokens(extra_tokens)
        new_vocab_size = len(tokenizer)
        logging.info(
            "Added another [%d] token(s) to the tokenizer so the vocab size is [%d].",
            num_extra_tokens,
            new_vocab_size,
        )

    if new_vocab_size != expected_vocab_size:
        raise ValueError(
            f"Expected tokenizer size to be {expected_vocab_size}, "
            f"but got {new_vocab_size}!"
        )
    logging.info(
        "Added [%d] speech tokens to the tokenizer total. Final vocab size: [%d].",
        num_added_tokens,
        new_vocab_size,
    )

    return tokenizer
