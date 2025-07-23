import abc
from collections.abc import Sequence
from typing import Any

from tts.core import constants


def _format_transcript(transcript: str) -> str:
    """Formats a transcript with the appropriate tokens."""
    return (
        f"{constants.TEXT_PROMPT_START_TOKEN}{transcript}"
        f"{constants.TEXT_PROMPT_END_TOKEN}"
    )


def _format_voice_description(voice_description: str) -> str:
    """Formats a voice description with the appropriate tokens."""
    return (
        f"{constants.VOICE_DESCRIPTION_START_TOKEN}{voice_description}"
        f"{constants.VOICE_DESCRIPTION_END_TOKEN}"
    )


def _create_message_with_voice_description(
    voice_description: str, transcript: str
) -> str:
    """Creates a message with voice description."""
    formatted_voice_description = _format_voice_description(voice_description)
    return (
        "Given the following voice description "
        + formatted_voice_description
        + " convert the text to speech:"
        + transcript
    )


def _format_speech_tokens(speech_ids: list[int]) -> str:
    """Formats speech IDs into speech tokens."""
    speech_tokens_str = []
    for speech_id in speech_ids:
        speech_tokens_str.append(constants.SPEECH_TOKEN_PATTERN.format(speech_id))
    return "".join(speech_tokens_str)


class PromptCompiler(abc.ABC):
    """Abstract base class for prompt compilers."""

    @abc.abstractmethod
    def compile_prompt(
        self,
        audio_prompt_transcription: str,
        text_to_synthesize: str,
        speech_ids: Sequence[int],
        voice_description: str = "",
        enable_instruction: bool = True,
    ) -> Any:
        """Compiles the chat prompt for the model."""
        raise NotImplementedError("Subclasses must implement this method.")


class TrainingPromptCompiler(PromptCompiler):
    """Prompt compiler for training/fine-tuning."""

    def compile_prompt(
        self,
        audio_prompt_transcription: str,
        text_to_synthesize: str,
        speech_ids: Sequence[int],
        voice_description: str = "",
        enable_instruction: bool = True,
    ) -> Any:
        """Compiles the chat prompt for training."""
        del text_to_synthesize  # Not used in training.
        del enable_instruction  # Not used in training.

        user_message = self._compile_user_message(
            audio_prompt_transcription, voice_description
        )
        assistant_message = self._compile_assistant_message(speech_ids)
        return self._compile_prompt(user_message, assistant_message)

    def _compile_user_message(
        self, audio_prompt_transcription: str, voice_description: str = ""
    ) -> str:
        """Compiles a user message for finetuning."""
        transcript = _format_transcript(audio_prompt_transcription)
        if voice_description:
            return _create_message_with_voice_description(voice_description, transcript)
        return constants.DEFAULT_MODEL_INSTRUCTION + transcript

    def _compile_assistant_message(self, speech_ids: Sequence[int]) -> str:
        """Compiles an assistant message for finetuning."""
        if len(speech_ids) == 0:
            raise ValueError("Speech IDs are empty!")
        speech_tokens = _format_speech_tokens(speech_ids)
        return constants.SPEECH_START_TOKEN + speech_tokens + constants.SPEECH_END_TOKEN

    def _compile_prompt(self, user_message: str, assistant_message: str) -> Any:
        """Compiles prompt for the model."""
        return user_message + "\n" + assistant_message


class InferencePromptCompiler(PromptCompiler):
    """Prompt compiler for inference."""

    def compile_prompt(
        self,
        audio_prompt_transcription: str,
        text_to_synthesize: str,
        speech_ids: Sequence[int],
        voice_description: str = "",
        enable_instruction: bool = True,
    ) -> Any:
        """Compiles the chat prompt for inference."""
        user_message = self._compile_user_message(
            audio_prompt_transcription,
            text_to_synthesize,
            voice_description,
            enable_instruction,
        )
        assistant_message = self._compile_assistant_message(speech_ids)
        return self._compile_prompt(user_message, assistant_message)

    def _compile_user_message(
        self,
        audio_prompt_transcription: str,
        text_to_synthesize: str,
        voice_description: str = "",
        enable_instruction: bool = True,
    ) -> str:
        """Compiles a user message for inference."""
        if audio_prompt_transcription and (not voice_description or enable_instruction):
            transcript = f"{audio_prompt_transcription} {text_to_synthesize}"
        else:
            transcript = text_to_synthesize

        transcript = _format_transcript(transcript)

        if voice_description:
            return _create_message_with_voice_description(voice_description, transcript)

        return constants.DEFAULT_MODEL_INSTRUCTION + transcript

    def _compile_assistant_message(self, speech_ids: list[int]) -> str:
        """Compiles an assistant message for inference."""
        if len(speech_ids) == 0:
            return constants.SPEECH_START_TOKEN

        speech_tokens = _format_speech_tokens(speech_ids)
        return constants.SPEECH_START_TOKEN + speech_tokens

    def _compile_prompt(self, user_message: str, assistant_message: str) -> Any:
        """Compiles prompt for the model."""
        return user_message + "\n" + assistant_message
