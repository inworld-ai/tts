import math
import string

import jiwer
import torch
import torchaudio
import torchmetrics
import zhconv
from absl import logging
from zhon import hanzi

EVAL_SAMPLE_RATE = 16000
# Default values for reward functions that return if the audio is empty
_DEFAULT_WER = 5.0
_DEFAULT_DNSMOS = 0
_DEFAULT_SIMILARITY = 0

# Language list for CER (Character Error Rate) instead of WER (Word Error Rate).
_CER_LANG_LIST = ["zh", "ja", "ko"]


def _transcribe_audio(
    whisper_model: torch.nn.Module, audio: torch.Tensor, language: str
) -> str:
    # Transcribe the audio using the Whisper model
    try:
        result = whisper_model.transcribe(audio.squeeze(), language=language)
        transcription = result["text"]
        return transcription
    except RuntimeError as e:
        logging.error(f"Runtime error: {audio.shape}, {e}. Try to empty the cache.")
        torch.cuda.empty_cache()
        return ""
    except Exception as e:
        logging.error(f"Unexpected exception: {audio.shape}, {e}.")
        return ""


def _normalize_transcript(transcript: str, language: str) -> str:
    """Normalizes a transcript by removing punctuation, extra spaces, and
    converting to lowercase."""
    punctuation_all = hanzi.punctuation + string.punctuation

    normalized = transcript.lower().strip()
    normalized = normalized.translate(str.maketrans("", "", punctuation_all))
    normalized = " ".join(normalized.split())
    if language == "zh":
        normalized = zhconv.convert(normalized, "zh-cn")
    if language in _CER_LANG_LIST:
        normalized = normalized.replace(" ", "")
    return normalized


def normalize_wer(wer: float) -> float:
    """Normalizes a WER to (0, 1] range as reward signal."""
    k = 2.5
    return math.exp(-k * wer)


def normalize_dnsmos(dnsmos: float) -> float:
    """Normalizes a DNSMOS score from [1, 5] range to [0, 1] range as
    reward signal."""
    return (dnsmos - 1) / 4


def normalize_similarity(similarity: float) -> float:
    """Normalizes a cosine similarity score from [-1, 1] range to [0, 1]
    range as reward signal."""
    return (similarity + 1) / 2


def eval_wer(
    whisper_model: torch.nn.Module,
    audio: torch.Tensor,
    sample_rate: int,
    ground_truth: str,
    language: str,
    prompt_transcript: str,
) -> float:
    """
    Calculate the Word Error Rate (WER) between the ground truth and transcription.

    Args:
        ground_truth (str): The correct transcription.
        transcription (str): The transcription to evaluate.

    Returns:
        float: The WER score.
    """
    # If the audio is empty, return the default WER score.
    if audio.shape[1] == 0:
        return _DEFAULT_WER

    if sample_rate != EVAL_SAMPLE_RATE:
        audio = torchaudio.functional.resample(
            audio, orig_freq=sample_rate, new_freq=EVAL_SAMPLE_RATE
        )

    # Transcribe the audio using the existing helper function
    transcription = _transcribe_audio(whisper_model, audio, language)

    if not transcription:
        return _DEFAULT_WER

    # Remove punctuation and normalize whitespace
    ground_truth_clean = _normalize_transcript(ground_truth, language)
    transcription_clean = _normalize_transcript(transcription, language)

    if language in _CER_LANG_LIST:
        wer = jiwer.cer(ground_truth_clean, transcription_clean)
    else:
        wer = jiwer.wer(ground_truth_clean, transcription_clean)

    # Edge case logging purpose.
    logging.info(
        f"WER is {wer} [reward: {normalize_wer(wer)}]: "
        f"original truth[{ground_truth}], asr[{transcription}], "
        f"language[{language}], prompt_transcript[{prompt_transcript}]"
    )
    return wer


def eval_dnsmos(audio: torch.Tensor, sample_rate: int, device: torch.device) -> float:
    # If the audio is empty, return the default DNSMOS score.
    if audio.shape[1] == 0:
        return _DEFAULT_DNSMOS
    dnsmos_tensor = (
        torchmetrics.functional.audio.dnsmos.deep_noise_suppression_mean_opinion_score(
            preds=audio, fs=sample_rate, personalized=True, device=device, num_threads=4
        )
        .cpu()
        .numpy()
    )
    if len(dnsmos_tensor[0]) < 4:
        logging.info(f"dnsmos_tensor length is less than 4: {dnsmos_tensor}")
        return _DEFAULT_DNSMOS
    return dnsmos_tensor[0][3]


def eval_similarity(
    model: torch.nn.Module,
    prompt_audio: torch.Tensor,
    prompt_sample_rate: int,
    completion_audio: torch.Tensor,
    completion_sample_rate: int,
) -> float:
    """
    Calculate the speaker similarity between the prompt and completion audio.

    Args:
        model: The model to use for speaker similarity calculation.
        prompt_audio: The prompt audio tensor.
        prompt_sample_rate: The sample rate of the prompt audio.
        completion_audio: The completion audio tensor.
        completion_sample_rate: The sample rate of the completion audio.

    Returns:
        float: The speaker similarity score.
    """
    # If the completion audio is empty, return the default similarity score.
    if completion_audio.shape[1] == 0:
        return _DEFAULT_SIMILARITY

    device = completion_audio.device
    if prompt_sample_rate != EVAL_SAMPLE_RATE:
        prompt_audio = torchaudio.functional.resample(
            prompt_audio, prompt_sample_rate, EVAL_SAMPLE_RATE
        ).to(device)

    if completion_sample_rate != EVAL_SAMPLE_RATE:
        completion_audio = torchaudio.functional.resample(
            completion_audio, completion_sample_rate, EVAL_SAMPLE_RATE
        ).to(device)

    # Ensure model is in evaluation mode
    model.eval()

    try:
        with torch.no_grad():
            # Extract speaker embeddings
            prompt_embedding = model(prompt_audio)
            completion_embedding = model(completion_audio)

            # Calculate cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                prompt_embedding, completion_embedding
            )[0].item()
            return similarity
    except Exception as e:
        logging.error(f"Error calculating speaker similarity: {e}")
        return _DEFAULT_SIMILARITY
