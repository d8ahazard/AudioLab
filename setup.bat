@echo off
REM Download the espeak-ng installer
echo Installing espeak-ng...
curl -L -o espeak-ng.msi https://github.com/espeak-ng/espeak-ng/releases/download/1.52.0/espeak-ng.msi || (
    echo Error downloading espeak-ng.msi. Exiting.
    exit /b 1
)

REM Install espeak-ng silently
msiexec /i espeak-ng.msi /quiet /norestart || (
    echo Error installing espeak-ng. Exiting.
    exit /b 1
)

REM Add "C:\Program Files\eSpeak NG\" to the *system* PATH
REM (Requires running as Administrator; won't take effect in the current session)
setx /M PATH "%PATH%;C:\Program Files\eSpeak NG\"

echo Installation complete. You may need to restart your terminal or log out and back in for the PATH change to be recognized.

REM Auto-detect CUDA driver version
FOR /F "tokens=1-2 delims=." %%A IN ('nvidia-smi --query-gpu=driver_version --format=csv,noheader') DO (
    SET CUDA_VERSION=%%A.%%B
)
SET CUDA_URL=https://download.pytorch.org/whl/cu%CUDA_VERSION:.=%

REM Update pip
echo Updating pip...
python -m pip install --upgrade pip==24.0 || (
    echo Error updating pip. Exiting.
    exit /b 1
)

pip install ninja

REM Install Torch and related libraries
echo Installing PyTorch and related packages...
REM pip install torch>=2.4.0 torchvision>=0.19.0 torchaudio>=2.4.0 faiss-cpu fairseq --extra-index-url https://download.pytorch.org/whl/cu121
python -m pip install torch>=2.4.0 torchvision>=0.19.0 torchaudio>=2.4.0 faiss-cpu fairseq --extra-index-url %CUDA_URL% || (
    echo Error installing PyTorch packages. Ensure CUDA version compatibility.
    exit /b 1
)

REM Collecting flash-attn https://github.com/bdashore3/flash-attention/releases/download/v2.7.0.post2/flash_attn-2.7.0.post2+cu124torch2.5.1cxx11abiFALSE-cp310-cp310-win_amd64.whl
echo Installing flash-attn...
python -m pip install https://github.com/bdashore3/flash-attention/releases/download/v2.7.0.post2/flash_attn-2.7.0.post2+cu124torch2.5.1cxx11abiFALSE-cp310-cp310-win_amd64.whl || (
    echo Error installing flash-attn.
    exit /b 1
)


REM Install torchlibrosa and librosa
echo Installing torchlibrosa and librosa...
python -m pip install torchlibrosa>=0.0.9 librosa>=0.10.2.post1 || (
    echo Error installing torchlibrosa or librosa.
    exit /b 1
)

REM Install remaining dependencies from requirements.txt
IF EXIST requirements.txt (
    echo Installing remaining dependencies...
    FOR /F "usebackq tokens=* delims=" %%A IN (requirements.txt) DO (
        REM Skip empty lines and comments
        IF NOT "%%A"=="" (
            echo Installing %%A...
            python -m pip install %%A || (
                echo Error installing %%A from requirements.txt
                exit /b 1
            )
        )
    )
) ELSE (
    echo requirements.txt not found!
    exit /b 1
)

REM Install Python-Wrapper-for-World-Vocoder
echo Cloning and installing Python-Wrapper-for-World-Vocoder...
IF EXIST Python-Wrapper-for-World-Vocoder (
    RMDIR /S /Q Python-Wrapper-for-World-Vocoder
)
git clone https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder.git && (
    CD Python-Wrapper-for-World-Vocoder
    git submodule update --init
    python -m pip install -r requirements.txt
    python -m pip install .
    CD ..
) || (
    echo Error cloning Python-Wrapper-for-World-Vocoder repository.
    exit /b 1
)

pip install TTS
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --extra-index-url %CUDA_URL% --force-reinstall
pip install omegaconf==2.2.3
pip install fairseq
pip install ./wheels/audiosr-0.0.8-py2.py3-none-any.whl
pip install https://github.com/d8ahazard/AudioLab/releases/download/1.0.0/causal_conv1d-1.5.0.post8-cp310-cp310-win_amd64.whl
pip install https://github.com/woct0rdho/triton-windows/releases/download/v3.2.0-windows.post9/triton-3.2.0-cp310-cp310-win_amd64.whl
pip install https://github.com/d8ahazard/AudioLab/releases/download/1.0.0/mamba_ssm-2.2.4-cp310-cp310-win_amd64.whl

echo All dependencies installed successfully!
pause
