#!/bin/bash

# Auto-detect CUDA driver version
CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d. -f1,2)
CUDA_URL="https://download.pytorch.org/whl/cu${CUDA_VERSION//./}"

# Update pip
echo "Updating pip..."
python3 -m pip install --upgrade pip==24.0

# Install Torch and related libraries
echo "Installing PyTorch and related packages..."
python3 -m pip install torch>=2.4.1 torchvision>=0.19.1 torchaudio>=2.4.1 --extra-index-url "$CUDA_URL"

# Install torchlibrosa and librosa
echo "Installing torchlibrosa and librosa..."
python3 -m pip install torchlibrosa>=0.0.9 librosa>=0.10.2.post1

# Install remaining dependencies from requirements.txt
echo "Installing remaining dependencies..."
while IFS= read -r package; do
    echo "Installing $package..."
    python3 -m pip install "$package"
done < requirements.txt

git clone https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder.git
cd Python-Wrapper-for-World-Vocoder
git submodule update --init
pip install -r requirements.txt
pip install .
cd ..


echo "All dependencies installed successfully!"
