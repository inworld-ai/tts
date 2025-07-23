# Training-time constants.
LOSS_IGNORE_TOKEN_ID = -100

# Tokenization.
SPEECH_TOKEN_PATTERN = "<|s_{}|>"
SPEECH_START_TOKEN = "<|speech_start|>"
SPEECH_END_TOKEN = "<|speech_end|>"
TEXT_PROMPT_START_TOKEN = "<|text_prompt_start|>"
TEXT_PROMPT_END_TOKEN = "<|text_prompt_end|>"
VOICE_DESCRIPTION_START_TOKEN = "<|voice_description_start|>"
VOICE_DESCRIPTION_END_TOKEN = "<|voice_description_end|>"
SOUND_EFFECT_START_TOKEN = "<|sound_effect_start|>"
SOUND_EFFECT_END_TOKEN = "<|sound_effect_end|>"
END_HEADER_ID = "<|end_header_id|>"

# File names.
CONFIG_FILE_NAME = "training_config.json"

# Audio.
CODEC_TOKENS_RATE = 50
CODEC_SAMPLE_RATE = 16000

# Splits.
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"

# vLLM related constants.
# TODO: use this instruction from prompt building across repos.
DEFAULT_MODEL_INSTRUCTION = "Convert the text to speech:"
# TODO: get rid of this name.
DEFAULT_MODEL_NAME = "/model/finch"

# Metrics.
TOTAL_SOURCE = "total"

# Reward functions.
WER_REWARD_FUNC = "WERRewardFunc"
DNSMOS_REWARD_FUNC = "DNSMOSRewardFunc"
SIMILARITY_REWARD_FUNC = "SimilarityRewardFunc"

# Instruction finetuning.
IFT_EMOTION = "emotion"
IFT_ACCENT = "accent"
IFT_AGE = "age"
IFT_GENDER = "gender"
IFT_PITCH = "pitch"
IFT_LANGUAGE = "language"
IFT_PACE = "pace"
IFT_SPEAKER = "speaker"
IFT_STYLE = "style"

# 21 nonverbal tokens.
# TODO: make i18n friendly.
NONVERBAL_TOKENS = [
    "<breathe>",
    "<burp>",
    "<chew>",
    "<clear_throat>",
    "<cough>",
    "<cry>",
    "<gasp>",
    "<grunt>",
    "<hiccup>",
    "<laugh>",
    "<moan>",
    "<pant>",
    "<scream>",
    "<sigh>",
    "<sing>",
    "<slurp>",
    "<sneeze>",
    "<sniff>",
    "<snort>",
    "<whistle>",
    "<yawn>",
]
