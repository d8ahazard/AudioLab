#!/bin/bash

sudo apt-get install espeak
# Auto-detect CUDA driver version
CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d. -f1,2)
CUDA_URL="https://download.pytorch.org/whl/cu${CUDA_VERSION//./}"

# Update pip
echo "Updating pip..."
python3 -m pip install --upgrade pip==24.0
python3 -m pip install ninja

# Install Torch and related libraries
echo "Installing PyTorch and related packages..."
if python3 -m pip install flash-attn torch>=2.4.0 torchvision>=0.19.0 torchaudio>=2.4.0 fairseq --extra-index-url "$CUDA_URL"; then
    echo "PyTorch packages installed successfully."
else
    echo "Error installing PyTorch packages. Ensure CUDA version compatibility." >&2
    exit 1
fi

# Install torchlibrosa and librosa
echo "Installing torchlibrosa and librosa..."
if ! python3 -m pip install torchlibrosa>=0.0.9 librosa>=0.10.2.post1; then
    echo "Error installing torchlibrosa or librosa." >&2
    exit 1
fi

# Install remaining dependencies from requirements.txt
echo "Installing remaining dependencies..."
if [[ -f requirements.txt ]]; then
    while IFS= read -r package || [[ -n $package ]]; do
        [[ -z "$package" || "$package" =~ ^# ]] && continue  # Skip empty lines and comments
        echo "Installing $package..."
        if ! python3 -m pip install "$package"; then
            echo "Error installing $package from requirements.txt" >&2
        fi
    done < requirements.txt
else
    echo "requirements.txt not found!" >&2
    exit 1
fi

# Install Python-Wrapper-for-World-Vocoder
echo "Cloning and installing Python-Wrapper-for-World-Vocoder..."
rm -rf Python-Wrapper-for-World-Vocoder
if git clone https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder.git; then
    cd Python-Wrapper-for-World-Vocoder || exit
    git submodule update --init
    python3 -m pip install -r requirements.txt
    python3 -m pip install .
    cd ..
else
    echo "Error cloning Python-Wrapper-for-World-Vocoder repository." >&2
    exit 1
fi

# Ensure these are re-installed/correctly
pip install mamba-ssm[causal-conv1d] --no-build-isolation
pip install TTS
pip install mamba-ssm[causal-conv1d] torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 faiss-cpu --extra-index-url "${CUDA_URL}"
#pip install mamba-ssm[causal-conv1d] torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 faiss-cpu --extra-index-url "https://download.pytorch.org/whl/cu124"
pip install omegaconf==2.2.3
pip install fairseq
pip install ./wheels/audiosr-0.0.8-py2.py3-none-any.whl
pip uninstall onnxruntime -y
pip uninstall onnxruntime-gpu -y
pip install onnxruntime-gpu --extra-index-url "${CUDA_URL}"

# Install Orpheus TTS dependencies by cloning the repository
echo "Installing Orpheus TTS..."
ORPHEUS_DIR="models/orpheus/Orpheus-TTS"
mkdir -p models/orpheus

if [ -d "$ORPHEUS_DIR" ]; then
    echo "Orpheus repository already exists, updating..."
    cd "$ORPHEUS_DIR" || exit
    git pull
    cd ../../..
else
    echo "Cloning Orpheus repository..."
    git clone https://github.com/canopyai/Orpheus-TTS.git "$ORPHEUS_DIR" || {
        echo "Error cloning Orpheus-TTS repository." >&2
        exit 1
    }
fi

# Install orpheus-speech from the cloned repository
cd "$ORPHEUS_DIR" || exit
echo "Installing orpheus-speech package..."
pip install orpheus-speech || {
    echo "Error installing orpheus-speech package." >&2
    cd ../../..
    exit 1
}

# Fix vllm version
echo "Installing specific vllm version to avoid bugs..."
pip install vllm==0.7.3 || {
    echo "Error installing specific vllm version." >&2
    cd ../../..
    exit 1
}
cd ../../..

echo "All dependencies installed successfully!"
