import abc
import os
import time
import uuid
from collections.abc import Sequence
from typing import Any

import torch
import torchaudio
import transformers
import whisper
from absl import logging

from tts.core import constants
from tts.core.codec import decoding
from tts.data import data_utils
from tts.inference import inferencing
from tts.training.rlhf import ecapa_tdnn, reward_utils

_DEFAULT_CODEC_CHECKPOINT_PATH = (
    "/path/to/some-9a5f5d.pt"
)
_DEFAULT_SIM_CHECKPOINT_PATH = (
    "/path/to/some-3aac85.pth"
)


class RewardFunc(abc.ABC):
    def __init__(
        self,
        tokenizer: transformers.AutoTokenizer,
        device: torch.device,
        save_completions_steps: int,
        save_dir: str,
        logging_steps: int,
        **kwargs,
    ):
        self._tokenizer = tokenizer
        self._device = device
        self._audio_decoder = decoding.create(
            model_path=_DEFAULT_CODEC_CHECKPOINT_PATH, device=self._device
        )
        self._save_completions_steps = save_completions_steps
        self._save_dir = save_dir
        self.steps = 0
        self.logging_steps = logging_steps
        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir, exist_ok=True)

    @abc.abstractmethod
    def __call__(self, completions: Sequence[str], **kwargs: Any) -> Sequence[float]:
        pass

    @property
    def __name__(self) -> str:
        return "RewardFunc"

    def _save_completion(self, audio: torch.Tensor):
        if (
            self._save_completions_steps > 0
            and self.steps % self._save_completions_steps == 0
        ):
            # Generate a unique filename with completion_step as prefix.
            unique_id = str(uuid.uuid4())
            audio_path = os.path.join(
                self._save_dir,
                f"completion_{self.steps}_rank{self._device.index}_{unique_id}.wav",
            )
            torchaudio.save(audio_path, audio, self._audio_decoder.sample_rate)

    def _decode_audio(
        self, prompt_speech_ids: torch.Tensor, completion: str
    ) -> torch.Tensor:
        speech_tokens = self._tokenizer.tokenize(completion)
        generated_speech_ids = torch.tensor(
            inferencing.extract_speech_ids(speech_tokens)
        ).to(self._device)
        # If generated_speech_ids is empty, return an empty tensor to avoid
        # downstream errors.
        if generated_speech_ids.shape[0] == 0:
            logging.warning(
                f"Warning: empty generated_speech_ids from "
                f"{len(speech_tokens)} speech tokens, returning empty audio."
            )
            # Return empty audio tensor with correct dimensions (channels, samples).
            return torch.zeros((1, 0), device=prompt_speech_ids.device)
        prompt_speech_ids = prompt_speech_ids.to(self._device)

        speech_ids = torch.cat([prompt_speech_ids, generated_speech_ids])
        try:
            gen_wav = self._audio_decoder.decode(speech_ids)
            prompt_wav_length = int(
                len(prompt_speech_ids)
                / self._audio_decoder.token_rate
                * self._audio_decoder.sample_rate
            )
            final_wav = gen_wav[:, prompt_wav_length:]
            self._save_completion(final_wav)
        except Exception as e:
            logging.error(f"Error decoding audio for speech_ids: {speech_ids}, {e}.")
            return torch.zeros((1, 0), device=prompt_speech_ids.device)
        return final_wav.to(self._device)


class WERRewardFunc(RewardFunc):
    def __init__(
        self,
        tokenizer: transformers.AutoTokenizer,
        device: torch.device,
        save_completions_steps: int,
        save_dir: str,
        **kwargs,
    ):
        super().__init__(tokenizer, device, save_completions_steps, save_dir, **kwargs)
        self.whisper_model = whisper.load_model("large-v3", device=self._device)

    def __call__(self, completions: Sequence[str], **kwargs: Any) -> Sequence[float]:
        reward_scores = []
        wer_scores = []
        start_time = time.time()
        for (
            prompt_speech_ids,
            completion,
            completion_truth,
            language,
            prompt_transcript,
        ) in zip(
            kwargs["prompt_speech_ids"],
            completions,
            kwargs["completion_truth"],
            kwargs["language"],
            kwargs["prompt_transcript"],
            strict=False,
        ):
            gen_wav = self._decode_audio(prompt_speech_ids, completion)
            if gen_wav.shape[1] == 0:
                logging.warning(
                    f"Warning: empty generated audio for completion: "
                    f"language:{language}, prompt_transcript:{prompt_transcript}, "
                    f"completion_truth:{completion_truth}, completion:{completion}."
                )
            wer = reward_utils.eval_wer(
                self.whisper_model,
                gen_wav,
                self._audio_decoder.sample_rate,
                completion_truth,
                language,
                prompt_transcript,
            )
            # Use negative WER as reward (higher reward for lower WER).
            # Normalize WER to [0, 1] range.
            reward = reward_utils.normalize_wer(wer)
            reward_scores.append(reward)
            wer_scores.append(wer)
        self.steps += 1
        end_time = time.time()
        if self.steps % self.logging_steps == 0:
            logging.info(
                f"WERRewardFunc called with wer_scores: {wer_scores}, "
                f"reward_scores: {reward_scores}, "
                f"time: {end_time - start_time:.2f} seconds."
            )
        return reward_scores

    @property
    def __name__(self) -> str:
        return "WERRewardFunc"


class DNSMOSRewardFunc(RewardFunc):
    def __init__(
        self,
        tokenizer: transformers.AutoTokenizer,
        device: torch.device,
        save_completions_steps: int,
        save_dir: str,
        **kwargs,
    ):
        super().__init__(tokenizer, device, save_completions_steps, save_dir, **kwargs)

    def __call__(self, completions: Sequence[str], **kwargs: Any) -> Sequence[float]:
        dnsmos_scores = []
        raw_dnsmos_scores = []
        start_time = time.time()
        for prompt_speech_ids, completion in zip(
            kwargs["prompt_speech_ids"], completions, strict=False
        ):
            gen_wav = self._decode_audio(prompt_speech_ids, completion)
            dnsmos = reward_utils.eval_dnsmos(
                gen_wav, self._audio_decoder.sample_rate, self._device
            )
            # Normalizes DNSMOS score to [0, 1] range.
            normalized_dnsmos = reward_utils.normalize_dnsmos(dnsmos)
            raw_dnsmos_scores.append(dnsmos)
            dnsmos_scores.append(normalized_dnsmos)
        self.steps += 1
        end_time = time.time()
        if self.steps % self.logging_steps == 0:
            logging.info(
                f"DNSMOSRewardFunc called with raw_dnsmos_scores: "
                f"{raw_dnsmos_scores}, normalized_dnsmos_scores: {dnsmos_scores}, "
                f"time: {end_time - start_time:.2f} seconds."
            )
        return dnsmos_scores

    @property
    def __name__(self) -> str:
        return "DNSMOSRewardFunc"


class SimilarityRewardFunc(RewardFunc):
    def __init__(
        self,
        tokenizer: transformers.AutoTokenizer,
        device: torch.device,
        save_completions_steps: int,
        save_dir: str,
        **kwargs,
    ):
        super().__init__(tokenizer, device, save_completions_steps, save_dir, **kwargs)
        # Import ECAPA-TDNN model for speaker similarity.
        # Initialize the speaker similarity model.
        self.similarity_model = ecapa_tdnn.ECAPA_TDNN_SMALL(
            feat_dim=1024, feat_type="wavlm_large", config_path=None
        ).to(device)

        state_dict = torch.load(
            _DEFAULT_SIM_CHECKPOINT_PATH,
            weights_only=True,
            map_location=lambda storage, loc: storage,
        )
        self.similarity_model.load_state_dict(state_dict["model"], strict=False)

        # Set model to evaluation mode.
        self.similarity_model.to(self._device).eval()

    def __call__(self, completions: Sequence[str], **kwargs: Any) -> Sequence[float]:
        similarity_scores = []
        raw_similarity_scores = []
        start_time = time.time()

        for prompt_speech_ids, completion, prompt_wav_path in zip(
            kwargs["prompt_speech_ids"],
            completions,
            kwargs["prompt_wav_path"],
            strict=False,
        ):
            completion_audio = self._decode_audio(prompt_speech_ids, completion)
            # Load prompt audio from the provided wav path.
            prompt_audio, prompt_sample_rate = data_utils.load_wav(prompt_wav_path)
            prompt_audio = prompt_audio.to(self._device)

            # Calculate similarity between prompt and completion.
            similarity = reward_utils.eval_similarity(
                self.similarity_model,
                prompt_audio,
                prompt_sample_rate,
                completion_audio,
                self._audio_decoder.sample_rate,
            )
            # Normalize similarity score to [0, 1] range.
            normalized_similarity = reward_utils.normalize_similarity(similarity)
            raw_similarity_scores.append(similarity)
            similarity_scores.append(normalized_similarity)

        self.steps += 1
        end_time = time.time()
        if self.steps % self.logging_steps == 0:
            logging.info(
                f"SimilarityRewardFunc called with raw_similarity_scores: "
                f"{raw_similarity_scores}, normalized_similarity_scores: "
                f"{similarity_scores}, time: {end_time - start_time:.2f} seconds."
            )
        return similarity_scores

    @property
    def __name__(self) -> str:
        return "SimilarityRewardFunc"


def create_reward_funcs(
    reward_func_names: list[str],
    tokenizer: transformers.AutoTokenizer,
    device: torch.device,
    save_completions_steps: int,
    save_dir: str,
    **kwargs,
) -> list[RewardFunc]:
    """Creates reward functions based on given reward function names.

    Args:
        reward_func_names: List of reward function names to create.
        tokenizer: Tokenizer to use for the reward function.
        device: Device to run the reward function on.
        **kwargs: Additional arguments to pass to the reward function.

    Returns:
        An instance of the specified reward function.

    Raises:
        ValueError: If the reward function name is not recognized.
    """
    reward_funcs = []
    for idx, reward_func_name in enumerate(reward_func_names):
        # Only the first reward function saves completions to avoid duplicate
        # audio files.
        save_steps = save_completions_steps if idx == 0 else 0
        if reward_func_name == constants.WER_REWARD_FUNC:
            reward_funcs.append(
                WERRewardFunc(
                    tokenizer=tokenizer,
                    device=device,
                    save_completions_steps=save_steps,
                    save_dir=save_dir,
                    **kwargs,
                )
            )
        elif reward_func_name == constants.DNSMOS_REWARD_FUNC:
            reward_funcs.append(
                DNSMOSRewardFunc(
                    tokenizer=tokenizer,
                    device=device,
                    save_completions_steps=save_steps,
                    save_dir=save_dir,
                    **kwargs,
                )
            )
        elif reward_func_name == constants.SIMILARITY_REWARD_FUNC:
            reward_funcs.append(
                SimilarityRewardFunc(
                    tokenizer=tokenizer,
                    device=device,
                    save_completions_steps=save_steps,
                    save_dir=save_dir,
                    **kwargs,
                )
            )
        else:
            raise ValueError(f"Unknown reward function: {reward_func_name}.")
    return reward_funcs
