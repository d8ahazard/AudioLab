#!/bin/bash

# Check for --fix flag
FIX_MODE=false
if [ "$1" == "--fix" ]; then
    FIX_MODE=true
fi

# Use absolute simplest CUDA detection with fallback
echo "Detecting CUDA version..."

# Default to CUDA 12.1 as fallback
CUDA_VERSION="12.1"
CUDA_URL="https://download.pytorch.org/whl/cu121"

# Get CUDA version from nvidia-smi
if command -v nvidia-smi &> /dev/null; then
    DETECTED_VERSION=$(nvidia-smi | grep "CUDA Version" | sed 's/.*CUDA Version: \([0-9]*\.[0-9]*\).*/\1/')
    if [ ! -z "$DETECTED_VERSION" ]; then
        CUDA_VERSION=$DETECTED_VERSION
    fi
fi

echo "Using CUDA version: $CUDA_VERSION"
# If cuda version is 12.8, use 12.4
if [ "$CUDA_VERSION" == "12.8" ]; then
    CUDA_VERSION="12.4"
fi
CUDA_URL="https://download.pytorch.org/whl/cu${CUDA_VERSION//./}"
echo "PyTorch URL: $CUDA_URL"
read -p "Press enter to continue"

# Check if we're already in a virtual environment
IN_VENV=false
if [ -n "$VIRTUAL_ENV" ] || [ -n "$CONDA_PREFIX" ]; then
    echo "Already in a Python virtual environment, using the current environment."
    IN_VENV=true
else
    # Create and activate venv only if not already in a venv
    if [ ! -d "venv" ]; then
        echo "Creating new virtual environment..."
        python3 -m venv venv
    fi

    # Activate virtual environment
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Branch based on fix mode
if [ "$FIX_MODE" == "true" ]; then
    echo "Skipping to fix section..."
    goto_fix_section=true
else
    goto_fix_section=false
fi

if [ "$goto_fix_section" == "false" ]; then
    # Regular installation starts here
    # Update pip
    echo "Updating pip..."
    python -m pip install --upgrade pip==24.0 || {
        echo "Error updating pip. Exiting."
        exit 1
    }

    pip install ninja

    # Install Torch and related libraries
    echo "Installing PyTorch and related packages..."
    python -m pip install "torch>=2.4.0" "torchvision>=0.19.0" "torchaudio>=2.4.0" faiss-cpu fairseq --extra-index-url "$CUDA_URL" || {
        echo "Error installing PyTorch packages. Ensure CUDA version compatibility."
        exit 1
    }

    # Install flash-attn with exact version matching
    echo "Installing flash-attn..."
    # Get Python version (e.g., 3.10 -> 310)
    PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")
    # Get torch version (e.g., 2.4.0 -> 2.4)
    TORCH_VERSION=$(python -c "import torch; print('.'.join(torch.__version__.split('.')[:2]))")
    # Simplified CUDA version for URL (e.g., 12.1 -> 12)
    CUDA_SHORT=$(echo $CUDA_VERSION | cut -d. -f1)

    # Construct the flash-attention URL with appropriate versions
    FLASH_ATTN_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu${CUDA_SHORT}torch${TORCH_VERSION}cxx11abiFALSE-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-linux_x86_64.whl"

    echo "Installing flash-attn from: $FLASH_ATTN_URL"
    python -m pip install "$FLASH_ATTN_URL" || {
        echo "Error installing flash-attn. Trying alternative version..."
        # Try with abiTRUE as fallback
        FLASH_ATTN_URL_ALT="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu${CUDA_SHORT}torch${TORCH_VERSION}cxx11abiTRUE-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-linux_x86_64.whl"
        python -m pip install "$FLASH_ATTN_URL_ALT" || {
            echo "Error installing flash-attn. Please install manually."
            exit 1
        }
    }

    # Install torchlibrosa and librosa
    echo "Installing torchlibrosa and librosa..."
    python -m pip install "torchlibrosa>=0.0.9" "librosa>=0.10.2.post1" || {
        echo "Error installing torchlibrosa or librosa."
        exit 1
    }

    # Install orpheus-speech
    echo "Installing orpheus-speech..."
    pip install git+https://github.com/Deathdadev/Orpheus-Speech-PyPi --extra-index-url "https://download.pytorch.org/whl/cu124"
    pip install accelerate

    # Install remaining dependencies from requirements.txt
    if [ -f "requirements.txt" ]; then
        echo "Installing remaining dependencies..."
        while IFS= read -r package || [[ -n "$package" ]]; do
            # Skip empty lines and comments
            if [[ ! -z "$package" && ! "$package" =~ ^# ]]; then
                echo "Installing $package..."
                python -m pip install "$package" || {
                    echo "Error installing $package from requirements.txt"
                    exit 1
                }
            fi
        done < requirements.txt
    else
        echo "requirements.txt not found!"
        exit 1
    fi

    # Install Python-Wrapper-for-World-Vocoder
    echo "Cloning and installing Python-Wrapper-for-World-Vocoder..."
    if [ -d "Python-Wrapper-for-World-Vocoder" ]; then
        rm -rf Python-Wrapper-for-World-Vocoder
    fi
    git clone https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder.git && {
        cd Python-Wrapper-for-World-Vocoder
        git submodule update --init
        python -m pip install -r requirements.txt
        python -m pip install .
        cd ..
    } || {
        echo "Error cloning Python-Wrapper-for-World-Vocoder repository."
        exit 1
    }
fi

# Fix section
echo "Running fix section..."

echo "[1/8] Installing dependencies for Linux..."
sudo apt-get update && sudo apt-get install -y build-essential

echo "[2/8] Complete cleanup of problematic packages..."
pip uninstall -y onnx onnxruntime onnxruntime-gpu torch torchvision torchaudio audio-separator fairseq triton
pip cache purge
rm -rf ~/.cache/torch_extensions 2>/dev/null
rm -rf ~/.cache/huggingface 2>/dev/null

echo "[3/8] Installing PyTorch ecosystem with EXPLICIT CUDA support..."
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --extra-index-url "$CUDA_URL"

echo "[4/8] Verifying CUDA is available in PyTorch..."
python -c "import torch; print(f'PyTorch CUDA check: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}, Version: {torch.__version__}')"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available'"

echo "[5/8] Installing core packages and dependencies..."
pip install numpy==1.24.3 protobuf==4.25.3
pip install TTS fairseq wandb
pip install ./wheels/audiosr-0.0.8-py2.py3-none-any.whl
# Use Linux equivalents where necessary
pip install causal-conv1d
pip install triton
pip install mamba-ssm

echo "[6/8] Installing specific ONNX and ONNX Runtime versions..."
pip install onnx==1.15.0
pip install onnxruntime-gpu==1.16.3
pip install openvoice-cli

echo "[7/8] Installing final dependencies..."
pip install faiss-cpu

echo "[8/8] Installing audio-separator with specific dependency version..."
pip install audio-separator==0.30.1 --no-deps
pip install numpy==1.24.3

echo "[8/8] Installing PyTorch ecosystem with EXPLICIT CUDA support..."
pip install numpy==1.24.3 pandas numba torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 faiss-cpu fairseq onnxruntime-gpu wandb xformers gradio --extra-index-url "https://download.pytorch.org/whl/$CUDA_URL" --force-reinstall
#pip install numpy==1.24.3 pandas numba torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 faiss-cpu fairseq onnxruntime-gpu --extra-index-url "$CUDA_URL" --force-reinstall

echo "Performing final verification..."
python -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}, Version: {torch.__version__}')"
python -c "import onnxruntime as ort; print(f'ONNX Runtime version: {ort.__version__}, Providers: {ort.get_available_providers()}')"
python -c "import audio_separator; print(f'Audio Separator version: {audio_separator.__version__}')"

echo "Running comprehensive DLL check for troubleshooting..."
python -c "import os, sys; print('Python PATH:'); [print(p) for p in sys.path]"
python -c "import ctypes; print('Checking shared libraries...')"

# Deactivate venv only if we activated it in this script
if [ "$IN_VENV" == "false" ]; then
    echo "Deactivating virtual environment..."
    deactivate
fi

if [ "$FIX_MODE" == "true" ]; then
    echo "Fix complete! Dependencies have been reinstalled."
    read -p "Press enter to continue"
    exit 0
fi

# Install espeak-ng (Linux equivalent)
echo "Installing espeak-ng..."
sudo apt-get install -y espeak-ng

echo "Installation complete."
echo "All dependencies installed successfully!"
read -p "Press enter to continue"
