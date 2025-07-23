"""Filtering functions for the data pipeline."""

import string

from tts.data import data_sample


def filter_empty_transcript(sample: data_sample.Sample):
    """Filter if transcript is empty."""
    return "empty_transcript" if sample.transcript == "" else None


def filter_non_english(sample: data_sample.Sample):
    """Filter if language is not English."""
    return "non_english" if sample.language != "en" else None


def filter_long_duration(sample: data_sample.Sample):
    """Filter if duration is greater than 30.0 seconds."""
    return "long_duration" if sample.duration > 30.0 else None


def filter_punct_or_space_only_transcript(sample: data_sample.Sample):
    """Filter if transcript is only punctuation or spaces."""
    transcript = sample.transcript
    if bool(transcript) and all(
        char in string.punctuation or char == " " for char in transcript
    ):
        return "punct_or_space_only_transcript"
    return None


def filter_allowed_languages(allowed_languages: list[str]):
    """Factory: filter if language is not in allowed_languages."""

    def _filter(sample: data_sample.Sample):
        if allowed_languages and sample.language not in allowed_languages:
            return f"languages-{sample.language}"
        return None

    return _filter


def filter_min_sample_rate(min_sample_rate: int):
    """Factory: filter if sample_rate is less than min_sample_rate."""

    def _filter(sample: data_sample.Sample):
        if sample.sample_rate < min_sample_rate:
            return f"sampling_rate-{sample.sample_rate}"
        return None

    return _filter


def filter_min_dnsmos_score(min_dnsmos_score: float):
    """Factory: filter if dnsmos_mos_ovr is less than min_dnsmos_score."""

    def _filter(sample: data_sample.Sample):
        if sample.dnsmos_mos_ovr < min_dnsmos_score:
            return "dnsmos"
        return None

    return _filter


def filter_min_audio_duration(min_audio_duration: float):
    """Factory: filter if duration is less than min_audio_duration."""

    def _filter(sample: data_sample.Sample):
        if sample.duration < min_audio_duration:
            return "audio_duration"
        return None

    return _filter
