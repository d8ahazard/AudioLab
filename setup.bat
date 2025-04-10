@echo off

REM Check for --fix flag
SET FIX_MODE=false
IF "%1"=="--fix" SET FIX_MODE=true

REM Use absolute simplest CUDA detection with fallback
echo Detecting CUDA version...

REM Default to CUDA 12.1 as fallback
SET CUDA_VERSION=12.1
SET CUDA_URL=https://download.pytorch.org/whl/cu121

for /f "tokens=3 delims=:" %%a in ('nvidia-smi ^| findstr "CUDA Version"') do @for /f "tokens=1" %%b in ("%%a") do @set CUDA_VERSION=%%b



echo Using CUDA version: %CUDA_VERSION%
REM If cuda version is 12.8, use 12.6
if %CUDA_VERSION%==12.8 set CUDA_VERSION=12.4
SET CUDA_URL=https://download.pytorch.org/whl/cu%CUDA_VERSION:.=%
echo PyTorch URL: %CUDA_URL%
pause

REM Check if we're already in a virtual environment
SET IN_VENV=false

REM Check for VIRTUAL_ENV (standard venv) or CONDA_PREFIX (conda env) environment variables
IF DEFINED VIRTUAL_ENV (
    echo Already in a Python virtual environment, using the current environment.
    SET IN_VENV=true
    GOTO :skip_venv_creation
)

IF DEFINED CONDA_PREFIX (
    echo Already in a Conda environment, using the current environment.
    SET IN_VENV=true
    GOTO :skip_venv_creation
)

REM Create venv only if not already in a venv and venv folder doesn't exist
IF NOT EXIST venv (
    echo Creating new virtual environment...
    python -m venv venv
)

REM Activate venv
echo Activating virtual environment...
call venv\Scripts\activate.bat

:skip_venv_creation

REM Branch based on fix mode
IF "%FIX_MODE%"=="true" (
    echo Skipping to fix section...
    GOTO :fix_section
)

REM Regular installation starts here
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
python -m pip install torch^>=2.4.0 torchvision^>=0.19.0 torchaudio^>=2.4.0 faiss-cpu fairseq --extra-index-url %CUDA_URL% || (
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
python -m pip install torchlibrosa^>=0.0.9 librosa^>=0.10.2.post1 || (
    echo Error installing torchlibrosa or librosa.
    exit /b 1
)

REM Install orpheus-speech
echo Installing orpheus-speech...
pip install git+https://github.com/Deathdadev/Orpheus-Speech-PyPi --extra-index-url https://download.pytorch.org/whl/cu124
pip install accelerate

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

:fix_section
REM Important fix section - uses the CUDA_VERSION already detected
echo Running fix section...

echo [1/8] Installing Visual C++ Redistributable (required for DLLs)...
powershell -Command "& {Invoke-WebRequest -Uri 'https://aka.ms/vs/17/release/vc_redist.x64.exe' -OutFile 'vc_redist.x64.exe'; Start-Process -FilePath 'vc_redist.x64.exe' -ArgumentList '/install', '/quiet', '/norestart' -Wait; Remove-Item -Path 'vc_redist.x64.exe'}"

echo [2/8] Complete cleanup of problematic packages...
pip uninstall -y onnx onnxruntime onnxruntime-gpu torch torchvision torchaudio audio-separator fairseq triton
pip cache purge
rmdir /s /q %USERPROFILE%\.cache\torch_extensions 2>nul
rmdir /s /q %USERPROFILE%\.cache\huggingface 2>nul

echo [3/8] Installing PyTorch ecosystem with EXPLICIT CUDA support...
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --extra-index-url %CUDA_URL%

echo [4/8] Verifying CUDA is available in PyTorch...
python -c "import torch; print(f'PyTorch CUDA check: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}, Version: {torch.__version__}')"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available'"

echo [5/8] Installing core packages and dependencies...
pip install numpy==1.24.3 protobuf==4.25.3
pip install TTS fairseq wandb
pip install ./wheels/audiosr-0.0.8-py2.py3-none-any.whl
pip install https://github.com/d8ahazard/AudioLab/releases/download/1.0.0/causal_conv1d-1.5.0.post8-cp310-cp310-win_amd64.whl
pip install https://github.com/woct0rdho/triton-windows/releases/download/v3.2.0-windows.post9/triton-3.2.0-cp310-cp310-win_amd64.whl
pip install https://github.com/d8ahazard/AudioLab/releases/download/1.0.0/mamba_ssm-2.2.4-cp310-cp310-win_amd64.whl

echo [6/8] Installing specific ONNX and ONNX Runtime versions...
pip install onnx==1.15.0
pip install onnxruntime-gpu==1.16.3

echo [7/8] Installing final dependencies...
pip install faiss-cpu

echo [8/8] Installing audio-separator with specific dependency version...
pip install audio-separator==0.30.1 --no-deps
pip install numpy==1.24.3  

echo [8/8] Installing PyTorch ecosystem with EXPLICIT CUDA support...
pip install numpy==1.24.3 pandas numba torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 faiss-cpu fairseq onnxruntime-gpu wandb xformers== 0.0.27.post2 gradio --extra-index-url "%CUDA_URL%" --force-reinstall


echo Performing final verification...
python -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}, Version: {torch.__version__}')"
python -c "import onnxruntime as ort; print(f'ONNX Runtime version: {ort.__version__}, Providers: {ort.get_available_providers()}')"
python -c "import audio_separator; print(f'Audio Separator version: {audio_separator.__version__}')"

echo Running comprehensive DLL check for troubleshooting...
python -c "import os, sys; print('Python PATH:'); [print(p) for p in sys.path]"
python -c "import ctypes; print('Loading C++ Runtime DLLs...'); ctypes.cdll.LoadLibrary('vcruntime140.dll'); ctypes.cdll.LoadLibrary('vcruntime140_1.dll'); print('C++ Runtime loaded successfully')"

REM Deactivate Venv only if we activated it in this script
IF "%IN_VENV%"=="false" (
    echo Deactivating virtual environment...
    call venv\Scripts\deactivate.bat
)

IF "%FIX_MODE%"=="true" (
    echo Fix complete! Dependencies have been reinstalled.
    pause
    exit /b 0
)

REM Download the espeak-ng installer
echo Installing espeak-ng...
curl -L -o espeak-ng.msi https://github.com/espeak-ng/espeak-ng/releases/download/1.52.0/espeak-ng.msi || (
    echo Error downloading espeak-ng.msi. Exiting.
    exit /b 1
)

REM Install espeak-ng silently
msiexec /i espeak-ng.msi /quiet /norestart || (
    echo Error installing espeak-ng.
)

REM Add "C:\Program Files\eSpeak NG\" to the *system* PATH
REM (Requires running as Administrator; won't take effect in the current session)
setx /M PATH "%PATH%;C:\Program Files\eSpeak NG\"

echo Installation complete. You may need to restart your terminal or log out and back in for the PATH change to be recognized.

echo All dependencies installed successfully!
pause
