"""Implements inference logic for local TTS models."""

import dataclasses
from typing import Any

import torch
import transformers
from absl import logging

from tts.core import constants, prompting
from tts.core.codec import decoding, encoding
from tts.utils import custom_logging


class InferenceSettings:
    """Settings for a TTS inference."""

    def __init__(
        self,
        temperature=0.8,
        max_tokens=1792,
        min_tokens=10,
        top_p=1.0,
        top_k=50,
        repetition_penalty=1.1,
        frequency_penalty=0.3,
        seed=42,
    ):
        """Initialize inference settings with default or provided values."""
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.frequency_penalty = frequency_penalty
        self.seed = seed


DEFAULT_INFERENCE_SETTINGS = InferenceSettings()


@dataclasses.dataclass(frozen=True)
class InferenceResult:
    """Result of a TTS inference."""

    wav: torch.Tensor
    encoding_time: float
    decoding_time: float
    inference_time: float


def extract_speech_ids(speech_tokens_str: list[str]) -> list[int]:
    """Extracts speech ids from speech tokens strings."""
    speech_ids = []
    for token_str in speech_tokens_str:
        if token_str.startswith("<|s_") and token_str.endswith("|>"):
            num_str = token_str[4:-2]
            num = int(num_str)
            speech_ids.append(num)
        else:
            logging.error("Unexpected token: %s", token_str)
    return speech_ids


@torch.no_grad()
def _generate_speech_tokens(
    model: transformers.AutoModelForCausalLM | Any,
    input_ids: torch.Tensor,
    inference_settings: InferenceSettings,
    speech_end_id: int,
    use_vllm: bool = False,
) -> torch.Tensor:
    """Implements actual speech token generation logic."""
    if use_vllm:
        import vllm

        sampling_params = vllm.SamplingParams(
            max_tokens=inference_settings.max_tokens,
            min_tokens=inference_settings.min_tokens,
            stop_token_ids=[speech_end_id],
            repetition_penalty=inference_settings.repetition_penalty,
            top_p=inference_settings.top_p,
            top_k=inference_settings.top_k,
            frequency_penalty=inference_settings.frequency_penalty,
            temperature=inference_settings.temperature,
            detokenize=False,
        )
        outputs = model.generate(
            prompt_token_ids=input_ids[0].tolist(), sampling_params=sampling_params
        )
        return outputs[0].outputs[0].token_ids

    return (
        model.generate(
            input_ids=input_ids,
            max_length=inference_settings.max_tokens,
            min_new_tokens=inference_settings.min_tokens,
            eos_token_id=speech_end_id,
            do_sample=True if inference_settings.temperature > 0.0 else False,
            repetition_penalty=inference_settings.repetition_penalty,
            top_p=inference_settings.top_p,
            temperature=inference_settings.temperature,
        )
        .cpu()
        .squeeze(0)
    )


@torch.no_grad()
def _synthesize_audio(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    audio_decoder: decoding.AudioDecoderInterface,
    speech_ids: list[int],
    prompt: str,
    model_device: torch.device,
    inference_settings: InferenceSettings,
    use_vllm: bool,
) -> tuple[torch.Tensor, float]:
    """Synthesizes audio from text using a finetuned FinchTTS model."""
    input_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")[
        "input_ids"
    ]
    input_ids = input_ids.to(model_device)
    speech_end_id = tokenizer.convert_tokens_to_ids(constants.SPEECH_END_TOKEN)

    # Generate the speech autoregressively the classic way.
    transformers.set_seed(inference_settings.seed)
    generated_ids = _generate_speech_tokens(
        model=model,
        input_ids=input_ids,
        inference_settings=inference_settings,
        speech_end_id=speech_end_id,
        use_vllm=use_vllm,
    )

    # Convert string speech tokens to speech token ids.
    if use_vllm:
        speech_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        # append the prompt speech ids to the generated speech ids
        speech_tokens = torch.tensor(speech_ids + extract_speech_ids(speech_tokens))
    else:
        # Keep only the audio tokens.
        generated_ids = generated_ids[input_ids.shape[1] - len(speech_ids) : -1]

        # Extract the speech token strings. Direct use of the output tokens
        # is dangerous, as there might be non-speech tokens in the output.
        speech_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        speech_tokens = torch.tensor(extract_speech_ids(speech_tokens))

    # Decode the speech tokens to speech waveform.
    with custom_logging.Timer() as timer:
        gen_wav = audio_decoder.decode(speech_tokens)
    decoding_time = timer.get_duration()

    prompt_wav_length = len(speech_ids) / audio_decoder.token_rate
    prompt_wav_length = int(prompt_wav_length * audio_decoder.sample_rate)
    return gen_wav[:, prompt_wav_length:], decoding_time


class LocalTtsModel:
    """Implements local inference for a TTS model."""

    def __init__(
        self,
        model: transformers.AutoModelForCausalLM | Any,
        device: torch.device,
        tokenizer: transformers.AutoTokenizer,
        audio_encoder: encoding.CachingAudioEncoder,
        audio_decoder: decoding.AudioDecoderInterface,
        prompt_compiler: prompting.PromptCompiler,
        use_vllm: bool = False,
    ):
        self._model = model
        self._device = device
        self._tokenizer = tokenizer
        self._audio_encoder = audio_encoder
        self._audio_decoder = audio_decoder
        self._prompt_compiler = prompt_compiler
        self._use_vllm = use_vllm

    def synthesize_speech(
        self,
        inference_settings: InferenceSettings,
        text_to_synthesize: str,
        prompt_id: str,
        prompt_wav: torch.Tensor,
        audio_prompt_transcription: str,
        voice_description: str = "",
        enable_instruction: bool = True,
    ) -> InferenceResult:
        """Synthesizes speech from text using a finetuned FinchTTS model."""
        speech_ids = []
        encoding_time = 0.0
        if not voice_description or enable_instruction:
            with custom_logging.Timer() as timer:
                speech_ids = self._audio_encoder.encode(prompt_id, prompt_wav)
            encoding_time = timer.get_duration()

        prompt = self._prompt_compiler.compile_prompt(
            audio_prompt_transcription=audio_prompt_transcription,
            text_to_synthesize=text_to_synthesize,
            speech_ids=speech_ids,
            voice_description=voice_description,
            enable_instruction=enable_instruction,
        )
        logging.info("Prompt: [%s].", prompt)
        with custom_logging.Timer() as timer:
            wav, decoding_time = _synthesize_audio(
                model=self._model,
                tokenizer=self._tokenizer,
                audio_decoder=self._audio_decoder,
                speech_ids=speech_ids,
                prompt=prompt,
                model_device=self._device,
                inference_settings=inference_settings,
                use_vllm=self._use_vllm,
            )
        inference_time = timer.get_duration()
        logging.info("Local inference time: %.2fs", inference_time)

        return InferenceResult(
            wav=wav,
            encoding_time=encoding_time,
            decoding_time=decoding_time,
            inference_time=inference_time,
        )


@torch.no_grad()
def complete_prompt(
    model: transformers.AutoModelForCausalLM,
    encoder: encoding.AudioEncoderInterface,
    tokenizer: transformers.AutoTokenizer,
    decoder: decoding.AudioDecoderInterface,
    prompt_wav: torch.Tensor,
    model_device: torch.device,
    inference_settings: InferenceSettings,
) -> torch.Tensor:
    """Performs autdio prompt completion inference."""
    speech_start_id = tokenizer.convert_tokens_to_ids(constants.SPEECH_START_TOKEN)
    speech_end_id = tokenizer.convert_tokens_to_ids(constants.SPEECH_END_TOKEN)

    # Encode prompt.
    input_ids = encoder.encode(prompt_wav)
    input_ids = input_ids.cpu().squeeze(0).tolist()
    prompt_ids_len = len(input_ids)
    input_ids = [
        tokenizer.vocab[constants.SPEECH_TOKEN_PATTERN.format(code)]
        for code in input_ids
    ]
    input_ids = [speech_start_id] + input_ids
    input_ids = torch.tensor(input_ids, device=model_device).unsqueeze(0)

    # Generate continuation using common logic.
    generated_ids = _generate_speech_tokens(
        model=model,
        input_ids=input_ids,
        inference_settings=inference_settings,
        speech_end_id=speech_end_id,
    )

    # Remove the speech start and end tokens.
    generated_ids = generated_ids[1:-1]

    # Extract generated tokens.
    speech_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    speech_tokens = extract_speech_ids(speech_tokens)
    speech_tokens = torch.tensor(speech_tokens)

    # Decode to audio.
    entire_wav = decoder.decode(speech_tokens)
    prompt_wav_length = prompt_ids_len / decoder.token_rate
    prompt_wav_length = int(prompt_wav_length * decoder.sample_rate)
    return entire_wav[:, prompt_wav_length:]
