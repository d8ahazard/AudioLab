#!/bin/bash

# Auto-detect CUDA driver version
CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d. -f1,2)
CUDA_URL="https://download.pytorch.org/whl/cu${CUDA_VERSION//./}"

# Update pip
echo "Updating pip..."
python3 -m pip install --upgrade pip==24.0

# Install Torch and related libraries
echo "Installing PyTorch and related packages..."
if python3 -m pip install torch>=2.4.0 torchvision>=0.19.0 torchaudio>=2.4.0 faiss-gpu fairseq --extra-index-url "$CUDA_URL"; then
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
            exit 1
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
pip install TTS
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --extra-index-url https://download.pytorch.org/whl/cu121
pip install omegaconf==2.2.3
pip install fairseq
pip install ./wheels/audiosr-0.0.8-py2.py3-none-any.whl


echo "All dependencies installed successfully!"
