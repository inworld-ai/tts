#!/usr/bin/env python3
"""Simple inference script for TTS model.

This script demonstrates how to load a trained TTS model checkpoint and generate
speech from text using an audio prompt.

Usage:
    python inference.py \
        --model_checkpoint_path /path/to/your/trained_model.pt \
        --audio_encoder_path /path/to/encoder.pt \
        --audio_decoder_path /path/to/decoder.pt \
        --prompt_wav_path /path/to/your_prompt.wav \
        --prompt_transcription "This is what the speaker says in the prompt." \
        --text "Hello, this is a test of the text-to-speech system." \
        --output_path output.wav
"""

import argparse
import torch
import torchaudio
from absl import logging

from tts.core import constants, modeling, prompting
from tts.core.codec import decoding, encoding
from tts.data import data_utils
from tts.inference import inferencing


def main():
    parser = argparse.ArgumentParser(description="Simple TTS inference")
    parser.add_argument("--model_checkpoint_path", required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--audio_encoder_path", required=True,
                       help="Path to audio encoder checkpoint")
    parser.add_argument("--audio_decoder_path", required=True,
                       help="Path to audio decoder checkpoint (must be in same directory as model_config.json)")
    parser.add_argument("--prompt_wav_path", required=True,
                       help="Path to audio prompt (.wav file)")
    parser.add_argument("--prompt_transcription", required=True,
                       help="Transcription of the audio prompt")
    parser.add_argument("--text", required=True,
                       help="Text to synthesize")
    parser.add_argument("--output_path", default="output.wav",
                       help="Output audio file path")

    args = parser.parse_args()

    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load model components
    logging.info("Loading model...")
    tokenizer, _, model, _ = modeling.load_tokenizer_config_and_model(
        args.model_checkpoint_path)
    model.eval().to(device)

    # Initialize audio codec components
    logging.info("Loading audio encoder...")
    audio_encoder = encoding.CachingAudioEncoder(
        model_path=args.audio_encoder_path, device=device)

    logging.info("Loading audio decoder...")
    audio_decoder = decoding.create(
        model_path=args.audio_decoder_path, device=device)

    # Create TTS model
    prompt_compiler = prompting.InferencePromptCompiler()
    tts_model = inferencing.LocalTtsModel(
        model=model,
        device=device,
        tokenizer=tokenizer,
        audio_encoder=audio_encoder,
        audio_decoder=audio_decoder,
        prompt_compiler=prompt_compiler,
        use_vllm=False
    )

    # Load audio prompt
    logging.info(f"Loading audio prompt: {args.prompt_wav_path}")
    prompt_wav, _ = data_utils.load_wav(
        args.prompt_wav_path, target_sample_rate=constants.CODEC_SAMPLE_RATE)

    # Inference settings
    inference_settings = inferencing.InferenceSettings(
        temperature=0.8,
        max_tokens=1792,
        min_tokens=10,
        top_p=1.0,
        top_k=50,
        repetition_penalty=1.4,
        frequency_penalty=0.4,
        seed=42,
    )

    text_to_synthesize = args.text
    audio_prompt_transcription = args.prompt_transcription

    # Generate speech
    logging.info(f"Generating speech for text: '{args.text}'")
    inference_result = tts_model.synthesize_speech(
        inference_settings=inference_settings,
        text_to_synthesize=text_to_synthesize,
        prompt_id="simple_inference",
        prompt_wav=prompt_wav,
        audio_prompt_transcription=audio_prompt_transcription,
        voice_description="",
        enable_instruction=True
    )

    # Save output audio
    output_wav = inference_result.wav
    torchaudio.save(args.output_path, output_wav, audio_decoder.sample_rate)

    # Print results
    logging.info(f"Generated audio saved to: {args.output_path}")

if __name__ == "__main__":
    main()
