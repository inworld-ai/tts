# Inworld Text-To-Speech Trainer

[![Lint Code](https://github.com/inworld-ai/tts/actions/workflows/linter.yaml/badge.svg)](https://github.com/inworld-ai/tts/actions/workflows/linter.yaml)
[![Python 3.10](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA Support](https://img.shields.io/badge/CUDA-12.4%20%7C%2012.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%20%7C%202.7-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the training and modeling code used to create [Inworld TTS-1 and TTS-1-Max models](https://inworld.ai/blog/introducing-inworld-tts).

You can use this code to pre-train, fine-tune, or align with RL your arbitrary SpeechLM-based TTS model, no matter if you're using a single GPU machine or a multi-GPU cluster.

## Inworld TTS

* [Playground](https://inworld-ai.github.io/tts/playground)
* [Examples](https://inworld-ai.github.io/tts/)
* [Technical Report](./tech_report.pdf)

## TTS Trainer Features

* **Modeling**: SpeechLM and 1D audio-codecs
* **Distributed Training**: DDP, DeepSpeed and FSDP for training arbitrary SpeechLM
* **Data Pipeline**: Ready-to-use scripts to vectorize and prepare your audio data for training

## Requirements

The code is only tested on Ubuntu 22.04.

| Component | Version | Notes |
|-----------|---------|-------|
| **Python** | 3.10 | Required for all features |
| **CUDA** | 12.4 or 12.8 | Auto-detected |
| **PyTorch** | 2.6 (CUDA 12.4) or 2.7 (CUDA 12.8) | Auto-installed |

## Quick Start

### Prerequisites

This project depends on **Python 3.10** and **uv** for package management.

#### Install uv

Install [uv](https://docs.astral.sh/uv/) for fast Python package management:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### One-Command Setup

**Default setup (CUDA 12.8 + PyTorch 2.7):**

```bash
make install
```

**Specify CUDA version:**

```bash
# For CUDA 12.4 + PyTorch 2.6
make install CUDA_VERSION=12.4

# For CUDA 12.8 + PyTorch 2.7
make install CUDA_VERSION=12.8
```

This automatically:
* Creates Python 3.10 virtual environment
* Installs CUDA-optimized PyTorch with the proper [flash attention](https://github.com/Dao-AILab/flash-attention) implementation
* Sets up all project dependencies

## Training Example

In order to train a SpeechLM you need to first vectorize your audio-data into the audio-codes.
Combined together with the transcript, the model learns how being conditioned on it generates the audio.
Below example shows how to get started with a simple SFT training.

### 1. Data Preparation

Process your raw audio dataset into a JSONL file where each line contains a sample with the following format:

```json
{
  "transcript": "Then they would swiftly dart at their prey and bear it to the ground.",
  "language": "en",
  "wav_path": "/path/to/audio.wav",
  "duration": 3.42,
  "sample_rate": 24000
}
```

**Required fields:**
* `transcript`: Text transcription of the audio
* `language`: Language code (e.g., "en" for English)
* `wav_path`: Absolute path to the audio file
* `duration`: Audio duration in seconds
* `sample_rate`: Audio sample rate in Hz

**Example dataset:** You can reference the [LibriTTS dataset](https://huggingface.co/datasets/mythicinfinity/libritts) which contains ~585 hours of English speech from 2, 456 speakers at 24kHz sampling rate.

**Sample files:** We provide real example data from LibriTTS in this repository:
* [`./example/configs/samples.jsonl`](./example/configs/samples.jsonl) - 100 real LibriTTS samples with proper JSONL format
* `./example/wavs/` - Corresponding audio files (audio_1.wav through audio_100.wav)

This gives you a working example to test the data vectorization and training pipeline with actual audio data.

### 2. Data Vectorization

Vectorize audio data using codec's encoder (We also made the codec compatible with [xcodec2](https://huggingface.co/HKUSTAudio/xcodec2/tree/main/ckpt), so you can use the publicly available checkpoint if you prefer not to train the codec from scratch):

**Test with provided samples:**
```bash
WANDB_PROJECT="your_project" \
torchrun --nproc_per_node 8 ./tools/data/data_vectorizer.py \
    --codec_model_path=/path/to/codec/model.pt \
    --batch_size=16 \
    --dataset_path=./example/configs/samples.jsonl \
    --output_dir=/path/to/output_directory \
    --use_wandb \
    --run_name=test_vectorization
```

**With your own dataset:**
```bash
WANDB_PROJECT="your_project" \
torchrun --nproc_per_node 8 ./tools/data/data_vectorizer.py \
    --codec_model_path=/path/to/codec/model.pt \
    --batch_size=16 \
    --dataset_path=/path/to/your_data.jsonl \
    --output_dir=/path/to/output_directory \
    --use_wandb \
    --run_name=vectorization_run
```

After vectorization completes, you'll have multiple shard files in your output directory like below:

```
train_codes_{0..n}.npy         # Vectorized audio codes for training
train_codes_index_{0..n}.npy   # Index mappings for training codes
train_samples_{0..n}.jsonl     # Training sample metadata
val_codes_{0..n}.npy           # Vectorized audio codes for validation
val_codes_index_{0..n}.npy     # Index mappings for validation codes
val_samples_{0..n}.jsonl       # Validation sample metadata
```

Feel free to customize [filtering logic](./tts/data/data_utils.py) to implement your custom filtering logic.

### 3. Merge Shards

Combine vectorized shards into unified dataset to save space:

```bash
python tools/data/data_merger.py \
    --dataset_path /path/to/your/vectorized_dataset \
    --remove_shards
```

After merging, your dataset folder will contain:

```
train_codes.npy        # Merged training codes
train_codes_index.npy  # Merged training code indices
train_samples.jsonl    # Merged training samples
val_codes.npy          # Merged validation codes
val_codes_index.npy    # Merged validation code indices
val_samples.jsonl      # Merged validation samples
```

### 4. Configuration

Create training config ( `./example/configs/sft.json` ). Below shows key configuration sections - see the full configuration file at `./example/configs/sft.json` for all available options:

```json
{
    "training": {
        "seed": 777,
        "logging_steps": 150,
        "eval_steps": 300,
        "learning_rate": 1e-04,
        "batch_size": 4,
        "precision": "bf16",
        "strategy": "ddp"
    },
    "modeling": {
        "parameters": {
            "codebook_size": 65536,
            "max_seq_len": 2048,
            "model_name": "meta-llama/Llama-3.2-1B-Instruct",
            "enable_text_normalization": true
        }
    },
    "checkpointing": {
        "save_steps": 10000,
        "keep_only_last_n_checkpoints": 10
    },
    "train_weighted_datasets": {
        "/path/to/your/vectorized_dataset": 1.0
    },
    "val_weighted_datasets": {
        "/path/to/your/vectorized_dataset": 1.0
    },
    "dataset": {
        "enable_rlhf_training": false
    }
}
```

**Important**:
* Update dataset paths to point to your vectorized data directory
* This shows only key parameters - refer to `./example/configs/sft.json` for the complete configuration with all available options
* To resume from a checkpoint: Add `"checkpoint_file_to_resume_from": "/path/to/your/checkpoint.pt"` to the `checkpointing` section

### 5. Training

**SFT training:**

```bash
fabric run --devices=$NUM_GPU tts/training/main.py \
    --config_path=./example/configs/sft.json \
    --use_wandb \
    --run_name=my_tts_training
```

After training completes, you'll find the trained model at `./experiments/my_tts_training/final_model.pt` along with model and tokenizer configuration files.

**Additional options:**
* `--dry_run`: Test pipeline without training
* `--compile_model`: Enable torch.compile optimization (works well only if all your batch' samples have the same length)

### 6. Monitoring

Track progress via:
* **Weights & Biases**: Loss curves and training metrics
* **Checkpoints**: Saved every `save_steps` iterations
* **Console logs**: Real-time training information

## Inference

Once you have a trained model, you can use it for inference to generate speech from text and an audio prompt.

Use the provided inference script for easy speech generation:

```bash
python tools/inference.py \
    --model_checkpoint_path /path/to/your/trained_model.pt \
    --audio_encoder_path /path/to/encoder.pt \
    --audio_decoder_path /path/to/decoder.pt \
    --prompt_wav_path /path/to/your_prompt.wav \
    --prompt_transcription "This is what the speaker says in the prompt." \
    --text "Hello, this is a test of the text-to-speech system." \
    --output_path output.wav
```

**Required components:**
- Trained model checkpoint (`.pt` file from training)
- Audio encoder checkpoint (codec `.pt`/`.ckpt` file)
- Audio decoder checkpoint (codec `.pt`/`.ckpt` file + `model_config.json` in same directory)
- Audio prompt file (`.wav` format)
- Prompt transcription (text of what's spoken in the audio prompt)

**Note:** If you don't want to retrain the decoder, you can use the same checkpoint from [xcodec2](https://huggingface.co/HKUSTAudio/xcodec2/tree/main/ckpt) for both encoder and decoder paths. We provided a xcodec2 compatible `model_config.json` file in `./example/codec`.

## Development

### Available Commands

```bash
make help           # Show all available commands
make install        # Install development environment
make test           # Run test suite
make test-coverage  # Run tests with coverage report
make lint           # Run code linting
make lint-fix       # Auto-fix linting issues
make version        # Show current version or bump version
make version patch  # Bump patch version (1.0.0 → 1.0.1), also support `minor`, `major`
```

### Development Workflow

```bash
git clone https://github.com/inworld-ai/tts.git
cd tts

# Set up development environment
make install

# Install pre-commit hooks
pre-commit install

# Make your changes and test
make lint-fix
make test-coverage
```

## Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Run tests**: `make test`
4. **Run linting**: `make lint-fix`
5. **Commit changes**: `git commit -m 'Add amazing feature'`
6. **Push to branch**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**

## Acknowledgments

* Meta AI team for open-sourcing [LLaMA LLMs](https://www.llama.com/)
* The PyTorch and Hugging Face communities
* Codec architecture inspired by [Llasa](https://arxiv.org/abs/2502.04128)

## Support

* **Bug Reports**: [GitHub Issues](https://github.com/inworld-ai/tts/issues)
* **General Questions**: For general inquiries and support, please [email us](mailto:opensource@inworld.ai)
