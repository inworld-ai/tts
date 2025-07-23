import abc
import os

import torch
import torchaudio
import transformers
from absl import logging

from tts.core import constants, prompting
from tts.core.codec import decoding, encoding
from tts.data import data_utils, text_normalization
from tts.inference import inferencing

_DEFAULT_ENCODER_CHECKPOINT_PATH = "/path/to/some-982760.pt"
_DEFAULT_DECODER_CHECKPOINT_PATH = "/path/to/some-9a5f5d.pt"
_DEFAULT_PROMPT_WAVS = {
    "/path/to/some-91f247.wav": (
        "It was extremely dark, this passage, after the blinding sunlight "
        "reflected from the sulfurous ground."
    ),
    "/path/to/some-438d0c.wav": (
        "Unveiling our summer collection, dive into vibrant colors and "
        "unparalleled designs."
    ),
    "/path/to/some-c3f992.wav": "Dogs are sitting by the door.",
}

_DEFAULT_PHRASES = [
    "Hello, how are you?",  #
    "I'm doing really well, thank you!",  #
    "Wow!",  #
    "Aha",
    "No",
    "Yes",
    "Once upon a time, there was a cat.",
    "Hmmm, ok...",  #
    "Ahah hahah hahaha hahah",  #
    (
        "Uh, honestly, I'm not too sure about that, but, like, I kinda remember "
        "hearing something about it on the radio last week or so."
    ),  #
    (
        "Yeah, so, you know, I was thinking maybe we could try that new pizza "
        "place on, uh, what was it—Main Street, I think?"
    ),  #
    (
        "Dude, I swear, it was, like, the funniest thing ever. I mean, I can't "
        "even explain it, you just had to be there."
    ),  #
    (
        "Wait, hold up a second, lemme think... oh yeah, that was totally the "
        "restaurant I was talking about earlier."
    ),  #
    (
        "I guess, I mean, it's probably fine, right? Like, I don't really see a "
        "huge problem with it or anything—unless you do, or whatever."
    ),  #
    "Can you please call +1 650 999 9999?",  #
    "The total amount is going to be around $1,456.79.",  #
    "We haven't seen each other in a while. How have you been, huh!?",  #
    (
        "She sells seashells by the seashore. The shells she sells are surely "
        "seashells. So if she sells shells on the seashore, I'm sure she sells "
        "seashore shells"
    ),  #
    (
        "Fuzzy Wuzzy was a bear. Fuzzy Wuzzy had no hair. Fuzzy Wuzzy wasn't very "
        "fuzzy, was he?"
    ),  #
    (
        "Peter Piper picked a peck of pickled peppers. A peck of pickled peppers "
        "Peter Piper picked. If Peter Piper picked a peck of pickled peppers, "
        "where's the peck of pickled peppers Peter Piper picked?"
    ),  #
    (
        "I was just a normal person,  walking down the street, minding my own "
        "business. And then! All of a sudden, a truck - a huge truck - hit me! "
        "Now I am a superhero... the one... the only... Truck Man!"
    ),
]

_TEST_COMBINATION = tuple[str, str, str]


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Unwraps the model to use for quality validation."""
    result = model

    if hasattr(result, "_forward_module"):
        # FSDP requires the model to be wrapped in a module.
        if isinstance(
            result._forward_module, torch.distributed.fsdp.FullyShardedDataParallel
        ):
            return result
        result = result._forward_module

    if hasattr(result, "_orig_mod"):
        if hasattr(result._orig_mod, "module"):
            result = result._orig_mod.module
        else:
            result = result._orig_mod

    # Unwrap DDP.
    if isinstance(result, torch.nn.parallel.DistributedDataParallel):
        result = result.module

    return result


def _get_all_test_combinations() -> list[_TEST_COMBINATION]:
    """Returns all test combinations of prompt wavs and phrases."""
    result = []
    prompt_wavs = sorted(_DEFAULT_PROMPT_WAVS.items(), key=lambda x: x[0])
    for prompt_wav_path, prompt_text in prompt_wavs:
        for phrase in _DEFAULT_PHRASES:
            result.append((prompt_wav_path, prompt_text, phrase))
    return result


class QualityValidator(metaclass=abc.ABCMeta):
    """Abstract base class for computing quality validation artifacts/metrics."""

    @abc.abstractmethod
    def validate(self, model: torch.nn.Module, step: int):
        raise NotImplementedError("|validate| must be implemented by subclasses.")


class NoOpQualityValidator(QualityValidator):
    """Quality validator that does nothing."""

    def validate(self, model: torch.nn.Module, step: int):
        del model, step  # Unused.


# TODO: improve and cover more use cases (eg voice description).
class RandomPhrasesSynthesizer(QualityValidator):
    """Quality validator that synthesizes random phrases with the codec on CPU."""

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        checkpointing_dir: str,
        global_rank: int,
        world_size: int,
        device: torch.device,
        prompt_compiler: prompting.PromptCompiler,
        codec_encoder_checkpoint_path: str = _DEFAULT_ENCODER_CHECKPOINT_PATH,
        codec_decoder_checkpoint_path: str = _DEFAULT_DECODER_CHECKPOINT_PATH,
        enable_text_normalization: bool = True,
    ):
        self._tokenizer = tokenizer
        self._audio_encoder = encoding.CachingAudioEncoder(
            codec_encoder_checkpoint_path, device
        )
        self._audio_decoder = decoding.create(codec_decoder_checkpoint_path, device)
        self._checkpointing_dir = checkpointing_dir
        self._global_rank = global_rank
        self._world_size = world_size
        self._device = device
        self._prompt_compiler = prompt_compiler
        # By default, we enable text normalization.
        self._text_normalizer = text_normalization.create_text_normalizer(
            enable_text_normalization
        )

    def _load_prompt_wav(self, prompt_wav_path: str) -> torch.Tensor:
        prompt_wav, _ = data_utils.load_wav(
            prompt_wav_path, target_sample_rate=constants.CODEC_SAMPLE_RATE
        )
        return prompt_wav

    def _select_test_combinations(
        self, test_combinations: list[_TEST_COMBINATION]
    ) -> list[_TEST_COMBINATION]:
        if self._world_size == 1:
            return test_combinations

        n_items = len(test_combinations)
        left = (self._global_rank * n_items) // self._world_size
        right = ((self._global_rank + 1) * n_items) // self._world_size
        right = min(right, n_items)
        return test_combinations[left:right]

    def _get_model(self, model: torch.nn.Module) -> inferencing.LocalTtsModel:
        return inferencing.LocalTtsModel(
            model=_unwrap_model(model),
            device=self._device,
            tokenizer=self._tokenizer,
            audio_encoder=self._audio_encoder,
            audio_decoder=self._audio_decoder,
            prompt_compiler=self._prompt_compiler,
        )

    # TODO: consider reusing more local inferencing code.
    def validate(self, model: torch.nn.Module, step: int) -> None:
        generation_dir = os.path.join(self._checkpointing_dir, f"generations/{step}/")
        os.makedirs(generation_dir, exist_ok=True)
        logging.info(
            "Starting to synthesize test phrases for step %d. They will be saved in %s",
            step,
            generation_dir,
        )

        test_combinations = self._select_test_combinations(_get_all_test_combinations())
        logging.info("Synthesizing %d phrases...", len(test_combinations))

        # TODO: support batch inference.
        for idx, (prompt_wav_path, prompt_text, phrase) in enumerate(test_combinations):
            prompt_wav = self._load_prompt_wav(prompt_wav_path)
            phrase = self._text_normalizer.normalize(phrase)
            inference_result = self._get_model(model).synthesize_speech(
                inference_settings=inferencing.DEFAULT_INFERENCE_SETTINGS,
                text_to_synthesize=phrase,
                prompt_id=prompt_wav_path,
                prompt_wav=prompt_wav,
                audio_prompt_transcription=prompt_text,
                voice_description="",
            )
            wav_path = os.path.join(
                generation_dir, f"rank_{self._global_rank}_{idx}.wav"
            )
            torchaudio.save(
                wav_path, inference_result.wav, self._audio_decoder.sample_rate
            )
            logging.info(
                "Synthesized %d/%d phrases...", idx + 1, len(test_combinations)
            )


class PromptContinuationValidator(QualityValidator):
    """Quality validator that continues audio prompts."""

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        checkpointing_dir: str,
        global_rank: int,
        world_size: int,
        device: torch.device,
        prompt_wav_paths: list[str] | None = None,
        codec_encoder_checkpoint_path: str = _DEFAULT_ENCODER_CHECKPOINT_PATH,
        codec_decoder_checkpoint_path: str = _DEFAULT_DECODER_CHECKPOINT_PATH,
    ):
        """Initializes the prompt continuation validator."""
        self._tokenizer = tokenizer

        # Note: This class instantiates a copy of the encoder/decoder, so if someone
        # accidentally uses it with the random phrases for TTS - it'll own a copy of it.
        self._audio_encoder = encoding.create(codec_encoder_checkpoint_path, device)
        self._audio_decoder = decoding.create(codec_decoder_checkpoint_path, device)

        self._checkpointing_dir = checkpointing_dir
        self._global_rank = global_rank
        self._world_size = world_size
        self._device = device

        # Use provided prompt paths or default ones
        self._prompt_wav_paths = prompt_wav_paths or list(_DEFAULT_PROMPT_WAVS.keys())

    def validate(self, model: torch.nn.Module, step: int):
        """Validates the model by continuing audio prompts."""
        if self._global_rank != 0:
            return

        generation_dir = os.path.join(
            self._checkpointing_dir, f"prompt_continuations/{step}/"
        )
        os.makedirs(generation_dir, exist_ok=True)
        logging.info(
            "Starting prompt continuation validation for step %d. "
            "Results will be saved in %s",
            step,
            generation_dir,
        )

        logging.info("Processing %d prompt(s)...", len(self._prompt_wav_paths))
        for idx, prompt_wav_path in enumerate(self._prompt_wav_paths):
            prompt_wav, _ = data_utils.load_wav(
                prompt_wav_path, target_sample_rate=constants.CODEC_SAMPLE_RATE
            )
            logging.info(
                "Processing prompt %d/%d: %.2fs long audio from %s",
                idx + 1,
                len(self._prompt_wav_paths),
                prompt_wav.shape[1] / constants.CODEC_SAMPLE_RATE,
                prompt_wav_path,
            )

            gen_wav = inferencing.complete_prompt(
                model=_unwrap_model(model),
                encoder=self._audio_encoder,
                tokenizer=self._tokenizer,
                decoder=self._audio_decoder,
                prompt_wav=prompt_wav,
                model_device=self._device,
                inference_settings=inferencing.DEFAULT_INFERENCE_SETTINGS,
            )

            # Save the continuation.
            base_name = f"prompt_{idx}"
            torchaudio.save(
                os.path.join(generation_dir, f"{base_name}_continuation.wav"),
                gen_wav,
                self._audio_decoder.sample_rate,
            )
            logging.info(
                "Completed validation for prompt %d/%d",
                idx + 1,
                len(self._prompt_wav_paths),
            )


def create_quality_validator(
    tokenizer: transformers.PreTrainedTokenizer,
    checkpointing_dir: str,
    save_intermediate_generations: bool,
    global_rank: int,
    world_size: int,
    device: torch.device,
    validation_type: str,
) -> QualityValidator:
    """Creates a quality validator for master process based on the provided settings."""
    if not save_intermediate_generations:
        return NoOpQualityValidator()

    if validation_type == "prompt_continuation":
        return PromptContinuationValidator(
            tokenizer=tokenizer,
            checkpointing_dir=checkpointing_dir,
            global_rank=global_rank,
            world_size=world_size,
            device=device,
        )
    prompt_compiler = prompting.InferencePromptCompiler()
    return RandomPhrasesSynthesizer(
        tokenizer=tokenizer,
        checkpointing_dir=checkpointing_dir,
        global_rank=global_rank,
        world_size=world_size,
        device=device,
        prompt_compiler=prompt_compiler,
    )
