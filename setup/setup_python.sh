#!/bin/bash
# Setup script for tts-core project
# Deletes existing .venv and recreates it with all dependencies
#
# CUDA Version Support:
# Set CUDA_VERSION environment variable to specify CUDA version (12.4, 12.8)
# Default: 12.8
# Installs flash-attn prebuild wheels from https://github.com/mjun0812/flash-attention-prebuild-wheels
#
# Examples:
#   ./setup_python.sh                    # Uses CUDA 12.8 + PyTorch 2.7 + flash-attn 2.8.1
#   CUDA_VERSION=12.4 ./setup_python.sh  # Uses CUDA 12.4 + PyTorch 2.6 + flash-attn 2.8.0
#   CUDA_VERSION=12.8 ./setup_python.sh  # Uses CUDA 12.8 + PyTorch 2.7 + flash-attn 2.8.1

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Emojis for better UX
ROCKET="🚀"
PACKAGE="📦"
CHECK="✅"
CROSS="❌"
WARNING="⚠️"
TRASH="🗑️"
FOLDER="📁"
LIGHTBULB="💡"
PARTY="🎉"
CLIPBOARD="📋"

log_info() {
    echo -e "${BLUE}${PACKAGE} $1${NC}"
}

log_success() {
    echo -e "${GREEN}${CHECK} $1${NC}"
}

log_error() {
    echo -e "${RED}${CROSS} $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}${WARNING} $1${NC}"
}

log_header() {
    echo -e "${BLUE}${ROCKET} $1${NC}"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Get the project root directory (parent of setup directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PATH="$PROJECT_ROOT/.venv"

log_header "Starting tts-core setup..."
echo -e "${FOLDER} Working in: $PROJECT_ROOT"

# Show CUDA version that will be used
CUDA_VERSION="${CUDA_VERSION:-12.8}"
if [[ "$OSTYPE" == "darwin"* ]]; then
    log_info "Platform: macOS (will use CPU-only PyTorch)"
else
    log_info "Platform: Linux (will use CUDA $CUDA_VERSION)"
fi

# Check if uv is installed
if ! command_exists uv; then
    log_error "uv is not installed. Please install it first:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Change to project root
cd "$PROJECT_ROOT"

# Step 1: Delete existing .venv if it exists
if [ -d "$VENV_PATH" ]; then
    log_info "Removing existing virtual environment at $VENV_PATH"
    rm -rf "$VENV_PATH"
    log_success "Existing .venv removed"
else
    log_info "No existing .venv found"
fi

# Step 2: Create new virtual environment with Python 3.10
log_info "Creating virtual environment with Python 3.10..."
if uv venv --python 3.10; then
    log_success "Virtual environment created successfully"
else
    log_error "Failed to create virtual environment"
    exit 1
fi

# Step 3: Install project dependencies
log_info "Installing project dependencies..."

# Install with CUDA extra on Linux, without extra on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    log_info "Installing project dependencies (macOS - CPU only)..."
    INSTALL_CMD="uv pip install -e ."
else

    # Map CUDA version to extra (simplified to cu124 and cu128 only)
    case "$CUDA_VERSION" in
        "12.4")
            CUDA_EXTRA="cu124"
            TORCH_VERSION="2.6"
            ;;
        "12.8")
            CUDA_EXTRA="cu128"
            TORCH_VERSION="2.7"
            ;;
        *)
            log_warning "Unsupported CUDA version: $CUDA_VERSION. Only CUDA 12.4 and 12.8 are supported. Defaulting to CUDA 12.8"
            CUDA_EXTRA="cu128"
            TORCH_VERSION="2.7"
            ;;
    esac

    log_info "Installing project dependencies with CUDA extra: $CUDA_EXTRA..."
    INSTALL_CMD="uv pip install -e .[$CUDA_EXTRA]"
fi

if $INSTALL_CMD; then
    log_success "Project dependencies installed successfully"
else
    log_error "Failed to install project dependencies"
    exit 1
fi

# Step 4: Install flash-attn prebuild wheel (Linux only)
if [[ "$OSTYPE" != "darwin"* ]]; then
    log_info "Installing flash-attn prebuild wheel for CUDA $CUDA_VERSION + PyTorch $TORCH_VERSION..."

    # Determine Python version
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    PYTHON_VERSION_NODOT=$(echo "$PYTHON_VERSION" | tr -d '.')

    # Construct flash-attn wheel URL
    # Format: flash_attn-[Version]+cu[CUDA]torch[PyTorch]-cp[Python]-cp[Python]-linux_x86_64.whl
    case "$CUDA_VERSION" in
        "12.4")
            PREBUILD_VERSION="0.3.12"
            FLASH_ATTN_VERSION="2.8.0"
            ;;
        "12.8")
            PREBUILD_VERSION="0.3.13"
            FLASH_ATTN_VERSION="2.8.1"
            ;;
    esac

    FLASH_ATTN_WHEEL="flash_attn-${FLASH_ATTN_VERSION}+cu${CUDA_VERSION//./}torch${TORCH_VERSION}-cp${PYTHON_VERSION_NODOT}-cp${PYTHON_VERSION_NODOT}-linux_x86_64.whl"
    FLASH_ATTN_URL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v${PREBUILD_VERSION}/${FLASH_ATTN_WHEEL}"

    log_info "Installing: $FLASH_ATTN_WHEEL"

    if uv pip install "$FLASH_ATTN_URL"; then
        log_success "flash-attn prebuild wheel installed successfully"
    else
        log_warning "flash-attn prebuild wheel installation failed, but continuing setup..."
        echo -e "${LIGHTBULB} You can install it manually later with:"
        echo "   uv pip install $FLASH_ATTN_URL"
        echo -e "${LIGHTBULB} Prebuild wheels available at: https://github.com/mjun0812/flash-attention-prebuild-wheels"
    fi
else
    log_warning "Skipping NeMo-text-processing installation on macOS"
    log_warning "Skipping flash-attn prebuild wheel on macOS (use pip install flash-attn for CPU version if needed)"
fi

echo ""
log_success "Setup completed successfully!"
echo ""
echo -e "${CLIPBOARD} Next steps:"
echo "   1. Activate the virtual environment:"
echo "      source .venv/bin/activate"
echo "   2. Start developing!"
echo ""
echo -e "${LIGHTBULB} CUDA version info:"
echo "   • Current CUDA version: $CUDA_VERSION"
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "   • PyTorch version: $TORCH_VERSION"
    echo "   • Installed with extra: $CUDA_EXTRA"
    echo "   • flash-attn: prebuild wheel from mjun0812/flash-attention-prebuild-wheels"
    echo "   • Supported CUDA versions:"
    echo "     CUDA_VERSION=12.4 $0  # CUDA 12.4 + PyTorch 2.6 + flash-attn 2.8.0"
    echo "     CUDA_VERSION=12.8 $0  # CUDA 12.8 + PyTorch 2.7 + flash-attn 2.8.1"
else
    echo "   • macOS detected - using CPU-only PyTorch"
    echo "   • For flash-attn on macOS: pip install flash-attn (CPU version)"
fi
echo ""
echo -e "${LIGHTBULB} To activate in the current shell, run:"
echo "   source .venv/bin/activate"
echo ""
echo -e "${PARTY} Happy coding!"
