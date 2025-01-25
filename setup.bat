@echo off

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

REM Install Torch and related libraries
echo Installing PyTorch and related packages...
python -m pip install torch>=2.4.0 torchvision>=0.19.0 torchaudio>=2.4.0 faiss-gpu fairseq --extra-index-url %CUDA_URL% || (
    echo Error installing PyTorch packages. Ensure CUDA version compatibility.
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

REM Ensure these are re-installed/correctly
pip install TTS torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --extra-index-url https://download.pytorch.org/whl/cu121 --force-reinstall
pip install omegaconf==2.2.3
pip install fairseq
pip install ./wheels/audiosr-0.0.8-py2.py3-none-any.whl

echo All dependencies installed successfully!
pause
