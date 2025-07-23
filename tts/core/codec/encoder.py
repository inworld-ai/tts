"""Codec encoder implementation."""

import collections

import torch
import transformers
import vector_quantize_pytorch as vq
from absl import logging

from tts.core import constants
from tts.core.codec import encoder_modules

_HOP_LENGTH = 320
_HALF_HOP_LENGTH = _HOP_LENGTH // 2


class Encoder(torch.nn.Module):
    """The audio encoder model."""

    def __init__(self, model_path: str):
        super().__init__()

        # model properties
        self.sample_rate = constants.CODEC_SAMPLE_RATE
        self.token_rate = constants.CODEC_TOKENS_RATE

        # encoders
        self.semantic_encoder = encoder_modules.SemanticEncoder(
            input_channels=1024,
            output_channels=1024,
            encode_channels=1024,
            kernel_size=3,
        )
        self.acoustic_encoder = encoder_modules.AcousticEncoder(
            num_generator_features=48,
            initial_conv_kernel_size=7,
            final_conv_kernel_size=3,
            up_ratios=[2, 2, 4, 4, 5],
            dilations=(1, 3, 9),
            output_dim=1024,
        )
        self.fusion_layer = torch.nn.Linear(2048, 2048)

        # vector quantization.
        self.quantizer = vq.ResidualFSQ(
            dim=2048, levels=[4, 4, 4, 4, 4, 4, 4, 4], num_quantizers=1
        )
        self.load_from_checkpoint(model_path)

        # wav2vec model for semantic features extraction.
        self.wav2vec_feature_extractor = (
            transformers.AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
        )
        self.wav2vec_model = transformers.Wav2Vec2BertModel.from_pretrained(
            "facebook/w2v-bert-2.0", output_hidden_states=True
        )

    def forward(self, wavs: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        """Computes VQ codes for a batch of audio files."""
        acoustic_hidden_states = self.acoustic_encoder(wavs)
        acoustic_hidden_states = acoustic_hidden_states.transpose(1, 2)

        wav2vec_features = self.wav2vec_model(feats[:, 0, :, :]).hidden_states[16]
        semantic_hidden_states = self.semantic_encoder(wav2vec_features.transpose(1, 2))

        hidden_states = torch.cat(
            [semantic_hidden_states, acoustic_hidden_states], dim=1
        )
        hidden_states = self.fusion_layer(hidden_states.transpose(1, 2)).transpose(1, 2)

        return self.quantize(hidden_states)

    def quantize(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Quantize the hidden states to the VQ codes."""
        hidden_states = hidden_states.permute(0, 2, 1)
        _, vq_code = self.quantizer(hidden_states)
        vq_code = vq_code.permute(0, 2, 1)
        return vq_code

    def load_from_checkpoint(self, checkpoint_path: str):
        """Loads the encoder from a checkpoint file."""
        logging.info("Loading encoder checkpoint from %s", checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if "state_dict" in ckpt.keys():
            # to provide compatibility for the checkpoint in
            # https://huggingface.co/HKUSTAudio/xcodec2/tree/main/ckpt
            ckpt = ckpt["state_dict"]

            filtered_state_dict_codec = collections.OrderedDict()
            filtered_state_dict_semantic_encoder = collections.OrderedDict()
            filtered_state_dict_gen = collections.OrderedDict()
            filtered_state_dict_fc_prior = collections.OrderedDict()
            for key, value in ckpt.items():
                if key.startswith("CodecEnc."):
                    new_key = key[len("CodecEnc.") :]
                    filtered_state_dict_codec[new_key] = value
                elif key.startswith("generator.quantizer."):
                    new_key = key[len("generator.quantizer.") :]
                    filtered_state_dict_gen[new_key] = value
                elif key.startswith("SemanticEncoder_module."):
                    new_key = key[len("SemanticEncoder_module.") :]
                    filtered_state_dict_semantic_encoder[new_key] = value
                elif key.startswith("fc_prior."):
                    new_key = key[len("fc_prior.") :]
                    filtered_state_dict_fc_prior[new_key] = value

            self.semantic_encoder.load_state_dict(filtered_state_dict_semantic_encoder)
            self.acoustic_encoder.load_state_dict(filtered_state_dict_codec)
            self.fusion_layer.load_state_dict(filtered_state_dict_fc_prior)
            self.quantizer.load_state_dict(filtered_state_dict_gen)
        else:
            self.load_state_dict(ckpt)

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        """Encodes a waveform into a sequence of tokens."""
        audio = torch.nn.functional.pad(
            wav.cpu(), (0, _HOP_LENGTH - (wav.shape[1] % _HOP_LENGTH))
        )
        audio_pad = torch.nn.functional.pad(audio, (_HALF_HOP_LENGTH, _HALF_HOP_LENGTH))
        feat = self.wav2vec_feature_extractor(
            audio_pad, sampling_rate=constants.CODEC_SAMPLE_RATE, return_tensors="pt"
        ).data["input_features"]

        device = self.wav2vec_model.device
        return self.forward(
            audio.unsqueeze(0).to(device), feat.unsqueeze(0).to(device)
        ).squeeze()
