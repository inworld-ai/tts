# Inworld Text-to-speech Trainer
### *Python 3.10+ | CUDA 12.4/12.8 | PyTorch 2.6/2.7*

[![Lint Code](https://github.com/inworld-ai/tts/actions/workflows/linter.yaml/badge.svg)](https://github.com/inworld-ai/tts/actions/workflows/linter.yaml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA Support](https://img.shields.io/badge/CUDA-12.4%20%7C%2012.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%20%7C%202.7-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance neural text-to-speech system using frozen audio codecs and large language models. This framework also supports training other speech-focused models including voice conversion, speech enhancement, and audio generation tasks. Built with automatic CUDA detection, optimized for both research and production environments.

## Demo

[Live Demo and Examples](https://inworld-ai.github.io/tts/)

## Features

- **High-Performance TTS**: State-of-the-art neural text-to-speech synthesis
- **Multi-Task Training**: Support for TTS, voice conversion, and other speech models
- **Multi-Platform**: Supports Linux (CUDA) and macOS (CPU)
- **Auto CUDA Detection**: Automatically selects optimal PyTorch version for your CUDA setup
- **Production Ready**: Built for scalable deployment with deepspeed and lightning
- **Developer Friendly**: Comprehensive development tools and linting

## Requirements

| Component | Version | Notes |
|-----------|---------|-------|
| **Python** | 3.10+ | Required for all features |
| **CUDA** | 12.4 or 12.8 | Linux only, auto-detected |
| **PyTorch** | 2.6 (CUDA 12.4) or 2.7 (CUDA 12.8) | Auto-installed |
| **Platform** | Linux (CUDA) / macOS (CPU) | Full GPU acceleration on Linux |

## Quick Start

### Prerequisites

This project depends on **Python 3.10+** and **uv** for package management.

#### Install Python 3.10+

**Option 1: Using asdf or mise (Recommended for development)**
```bash
# Install asdf: https://asdf-vm.com/guide/getting-started.html
# or mise: https://mise.jdx.dev/getting-started.html

# With asdf
asdf plugin add python
asdf install python 3.10.12
asdf global python 3.10.12

# With mise
mise use python@3.10.12
```

**Option 2: Native OS installation**
```bash
# macOS (via Homebrew)
brew install python@3.10

# Ubuntu/Debian
sudo apt update && sudo apt install python3.10 python3.10-venv

# CentOS/RHEL/Fedora
sudo dnf install python3.10
```

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
- Creates Python 3.10 virtual environment
- Installs CUDA-optimized PyTorch
- Sets up all project dependencies

## Training Example

Complete end-to-end training pipeline using the LibriTTS dataset:

### 1. Data Preparation

Process your raw audio dataset into a JSONL file where each line contains a sample with the following format:

```json
{
  "speaker_id": "train-clean-360_2272",
  "transcript": "Then they would swiftly dart at their prey and bear it to the ground.",
  "language": "en",
  "wav_path": "/path/to/audio.wav",
  "duration": 3.42,
  "sample_rate": 24000
}
```

**Required fields:**
- `transcript`: Text transcription of the audio
- `language`: Language code (e.g., "en" for English)
- `wav_path`: Absolute path to the audio file
- `duration`: Audio duration in seconds
- `sample_rate`: Audio sample rate in Hz

**Optional field(s):**
- `speaker_id`: Unique identifier for the speaker

**Example dataset:** You can reference the [LibriTTS dataset](https://huggingface.co/datasets/mythicinfinity/libritts) which contains ~585 hours of English speech from 2,456 speakers at 24kHz sampling rate.

**Sample file:** See [`./example_configs/samples.jsonl`](./example_configs/samples.jsonl) for a demonstration file with 1000 samples showing the correct format (tiny subset for demonstration purposes only).

### 2. Data Vectorization

Vectorize audio data using codec encoder:

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

### 3. Merge Shards

Combine vectorized shards into unified dataset:

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

Create training config (`./example_configs/sft.json`). Below shows key configuration sections - see the full configuration file at `./example_configs/sft.json` for all available options:

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
- Update dataset paths to point to your vectorized data directory
- This shows only key parameters - refer to `./example_configs/sft.json` for the complete configuration with all available options

### 5. Training

**Multi-GPU training:**
```bash
torchrun --nproc_per_node=4 tts/training/main.py \
    --config_path=./example_configs/sft.json \
    --use_wandb \
    --run_name=my_tts_training
```

**Single GPU training:**
```bash
python tts/training/main.py \
    --config_path=./example_configs/sft.json \
    --use_wandb \
    --run_name=my_tts_training
```

After training completes, you'll find the trained model at `./experiments/my_tts_training/final_model.pt` along with model and tokenizer configuration files.

**Additional options:**
- `--dry_run`: Test pipeline without training
- `--compile_model`: Enable torch.compile optimization

### 6. Monitoring

Track progress via:
- **Weights & Biases**: Loss curves and training metrics
- **Checkpoints**: Saved every `save_steps` iterations
- **Console logs**: Real-time training information

## 💻 Development

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
# 1. Set up environment
make install

# 2. Make your changes
# ... edit code ...

# 3. Run linter and tests
make lint-fix
make test

# 4. Check coverage
make test-coverage
```

## Platform Support

### Linux (Recommended)
- **Full CUDA acceleration** with CUDA 12.4 or 12.8
- **All features supported**

### macOS
- **CPU-only mode** with optimized PyTorch
- **Most features supported** (some CUDA-specific optimizations unavailable)

## Advanced Configuration

### CUDA Version Selection

Set the `CUDA_VERSION` environment variable for automated setup:

```bash
export CUDA_VERSION=12.4    # Set globally
make install

# Or for single command
make install CUDA_VERSION=12.4
```

### Manual Installation

If you prefer manual control:

```bash
# Create environment
uv venv --python 3.10

# Activate environment
source .venv/bin/activate  # Linux/macOS
# or .venv\Scripts\activate  # Windows

# Install with specific CUDA version
uv sync --extra cu124      # CUDA 12.4
uv sync --extra cu128      # CUDA 12.8
uv sync                    # Default (CUDA 12.8 on Linux, CPU on macOS)
```

## Troubleshooting

### CUDA Issues
```bash
# Check your CUDA version
nvcc --version
nvidia-smi

# Only CUDA 12.4 and 12.8 are supported
# For other versions, install PyTorch manually
```

### Python Version Issues
```bash
# Ensure Python 3.10+
python --version

# Available Python versions: 3.9, 3.10, 3.11, 3.12
```

### Installation Problems
```bash
# Clean install
rm -rf .venv
make install
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

### Development Setup for Contributors

```bash
# Clone your fork
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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@article{inworld2024tts,
  title={Reserved for arXiv article citation},
  author={},
  journal={arXiv preprint arXiv:},
  year={2024}
}
```

## Acknowledgments

- [Inworld AI](https://inworld.ai) for open-sourcing this project
- The PyTorch and Hugging Face communities
- Codec architecture inspired by [Llasa](https://arxiv.org/abs/2502.04128)

## Support

- **Bug Reports**: [GitHub Issues](https://github.com/inworld-ai/tts/issues)
- **General Questions**: For general inquiries and support, please [email us](mailto:opensource@inworld.ai)

---

**Made with ❤️ by [Inworld AI](https://inworld.ai)**
