@echo off
:: Auto-detect CUDA driver version
for /f "tokens=2 delims==" %%A in ('nvidia-smi --query-gpu=driver_version --format=csv,noheader') do set CUDA_VERSION=%%A
set CUDA_URL=https://download.pytorch.org/whl/cu%CUDA_VERSION%

:: Update pip
echo Updating pip...
python -m pip install --upgrade pip==24.0

:: Install remaining dependencies from requirements.txt
echo Installing remaining dependencies...
for /f "delims=" %%P in (requirements.txt) do (
    echo Installing %%P...
    python -m pip install %%P
)

:: Install Torch and related libraries
echo Installing PyTorch and related packages...
python -m pip install torch>=2.4.1 torchvision>=0.19.1 torchaudio>=2.4.1 --extra-index-url %CUDA_URL%

:: Install torchlibrosa and librosa
echo Installing torchlibrosa and librosa...
python -m pip install torchlibrosa>=0.0.9 librosa>=0.10.2.post1

echo All dependencies installed successfully!
pause
