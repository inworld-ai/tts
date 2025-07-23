import string

import torch
import torchaudio
import torchmetrics
from absl import logging
from jiwer import compute_measures

EVAL_SAMPLE_RATE = 16000
# Default values for reward functions that return if the audio is empty
_DEFAULT_WER = 5.0
_DEFAULT_DNSMOS = 0
_DEFAULT_SIMILARITY = 0


def _transcribe_audio(
    whisper_model: torch.nn.Module, audio: torch.Tensor, sample_rate: int
) -> str:
    # Transcribe the audio using the Whisper model
    try:
        result = whisper_model.transcribe(audio.squeeze(), language="en")
        transcription = result["text"]
        return transcription
    except RuntimeError as e:
        logging.error(f"Runtime error: {audio.shape}, {e}. Try to empty the cache.")
        torch.cuda.empty_cache()
        return ""
    except Exception as e:
        logging.error(f"Unexpected exception: {audio.shape}, {e}.")
        return ""


def eval_wer(
    whisper_model: torch.nn.Module,
    audio: torch.Tensor,
    sample_rate: int,
    ground_truth: str,
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
    transcription = _transcribe_audio(whisper_model, audio, EVAL_SAMPLE_RATE)

    if not transcription:
        return _DEFAULT_WER

    # Define punctuation characters to remove
    punctuation_all = string.punctuation

    # Remove punctuation and normalize whitespace
    ground_truth_clean = ground_truth.lower()
    transcription_clean = transcription.lower()

    for x in punctuation_all:
        ground_truth_clean = ground_truth_clean.replace(x, "")
        transcription_clean = transcription_clean.replace(x, "")

    # Replace multiple spaces with a single space
    ground_truth_clean = ground_truth_clean.replace("  ", " ").strip()
    transcription_clean = transcription_clean.replace("  ", " ").strip()
    measures = compute_measures(ground_truth_clean, transcription_clean)
    wer = measures["wer"]
    # Edge case logging purpose.
    if wer == 1.0:
        logging.info(
            f"WER is 1.0: original truth[{ground_truth}], asr[{transcription}]"
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
