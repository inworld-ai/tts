import dataclasses
import uuid
from typing import Any

_DEFAULT_VALUES = {
    "speaker_id": "",
    "emotion": "",
    "language": "unknown",
    "dnsmos_mos_ovr": 0.0,
    "style": "",
}
_REQUIRED_FIELDS = frozenset(("wav_path",))


@dataclasses.dataclass
class Sample:
    """A TTS sample. Can and should be adjusted for the specific use-cases.

    Args:
        id: The id of the sample.
        wav_path: The path to the wav file.
        speaker_id: The speaker id.

        language: The language of the sample.
        emotion: The emotion of the sample.
        transcript: The transcript of the sample.
        voice_description: The voice description of the sample.
        sound_effect: The sound effect of the sample.

        duration: The duration of the sample in seconds.
        sample_rate: Original sample rate of the wav file.
        dataset_name: The name of the dataset the sample is from.
        dnsmos_mos_ovr: The dnsmos mos ovr of the sample.
        style: The style of the sample (e.g., whispering, shouting, laughing, etc.).
        original_data: The original sample data used to support flexible formatting.
    """

    # Metadata.
    id: str
    wav_path: str
    speaker_id: str

    # Annotations.
    language: str
    emotion: str
    transcript: str
    voice_description: str
    sound_effect: str

    # Properties.
    duration: float
    sample_rate: int
    dataset_name: str
    dnsmos_mos_ovr: float
    style: str
    original_data: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        """Validates sample fields."""
        if not self.transcript and not self.voice_description and not self.sound_effect:
            raise ValueError(
                "At least one of transcript, voice_description, "
                "or sound_effect must be set."
            )

    def to_json(self) -> dict[str, str]:
        return {k: v for k, v in dataclasses.asdict(self).items() if v}

    @classmethod
    def from_json(cls, json_data: dict[str, Any], dataset_name: str) -> "Sample":
        if not dataset_name:
            raise ValueError("dataset_name is required")
        for field in _REQUIRED_FIELDS:
            if json_data.get(field) is None:
                raise ValueError(f"{field} is required for sample: {json_data}")

        return Sample(
            id=json_data.get("id", str(uuid.uuid4())),
            wav_path=json_data["wav_path"],
            speaker_id=json_data.get("speaker_id", _DEFAULT_VALUES["speaker_id"]),
            emotion=json_data.get("emotion", _DEFAULT_VALUES["emotion"]).lower(),
            transcript=json_data.get("transcript", ""),
            voice_description=json_data.get("voice_description", ""),
            sound_effect=json_data.get("sound_effect", ""),
            language=json_data.get("language", _DEFAULT_VALUES["language"]),
            duration=json_data.get("duration", -1.0),
            sample_rate=json_data.get("sample_rate", -1),
            dataset_name=dataset_name,
            dnsmos_mos_ovr=json_data.get(
                "dnsmos_mos_ovr", _DEFAULT_VALUES["dnsmos_mos_ovr"]
            ),
            style=json_data.get("style", _DEFAULT_VALUES["style"]).lower(),
            original_data=json_data.get("original_data", {}),
        )
