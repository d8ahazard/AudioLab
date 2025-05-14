#!/bin/bash
# Requires bash shell
set -e # Exit on error (equivalent to PowerShell's $ErrorActionPreference = 'Stop')

# Define color output functions to match PowerShell's Write-* functions
function write_info() {
    echo -e "\e[36m[INFO]  $1\e[0m"  # Cyan color
}

function write_success() {
    echo -e "\e[32m[OK]    $1\e[0m"  # Green color
}

function write_error() {
    echo -e "\e[31m[ERROR] $1\e[0m"  # Red color
}

function write_warning() {
    echo -e "\e[33m[WARN]  $1\e[0m"  # Yellow color
}

# Check for --fix flag
FIX_MODE=false
if [ "$1" == "--fix" ]; then
    FIX_MODE=true
    write_info "Fix mode enabled"
fi

# Detect platform and python version
IS_LINUX=false
if [[ "$(uname)" == "Linux" ]]; then
    IS_LINUX=true
fi

# Check for Python
if ! command -v python &> /dev/null; then
    if command -v python3 &> /dev/null; then
        # Create a symlink or alias if needed
        write_info "Found python3, using it as python"
        alias python=python3
    else
        write_error "Python not found in PATH. Aborting."
        exit 1
    fi
fi

# Get Python version
PY_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
write_info "Python version: $PY_VERSION"

# 1) CUDA detection with fallback (from original setup.sh)
write_info "Detecting CUDA version..."

# Default to CUDA 12.4 as fallback (aligned with Windows script)
CUDA_VERSION="12.4"
CUDA_URL="https://download.pytorch.org/whl/cu124"

# Get CUDA version from nvidia-smi
if command -v nvidia-smi &> /dev/null; then
    DETECTED_VERSION=$(nvidia-smi | grep "CUDA Version" | sed 's/.*CUDA Version: \([0-9]*\.[0-9]*\).*/\1/')
    if [ ! -z "$DETECTED_VERSION" ]; then
        CUDA_VERSION=$DETECTED_VERSION
        # If cuda version is 12.8, use 12.4
        if [ "$CUDA_VERSION" == "12.8" ]; then
            CUDA_VERSION="12.4"
        fi
        CUDA_URL="https://download.pytorch.org/whl/cu${CUDA_VERSION//./}"
    fi
fi

write_info "Using CUDA version: $CUDA_VERSION (URL: $CUDA_URL)"

# 2) Create or activate venv/conda
IN_VENV=false
if [ -n "$VIRTUAL_ENV" ]; then
    write_info "Detected existing virtualenv at $VIRTUAL_ENV"
    IN_VENV=true
elif [ -n "$CONDA_PREFIX" ]; then
    write_info "Detected existing Conda environment at $CONDA_PREFIX"
    IN_VENV=true
else
    if [ ! -d "./venv" ]; then
        write_info "Creating virtual environment in ./venv..."
        python -m venv ./venv
        if [ $? -ne 0 ]; then 
            write_error "Failed to create venv"
            exit 1
        fi
    fi
    write_info "Activating virtual environment..."
    source ./venv/bin/activate
fi

# 3) Core installer function (now supports extra pip args)
function install_packages() {
    local packages=("$@")
    local use_pytorch_cuda=true
    
    # Process flags if the last argument is a flag
    if [[ "${packages[-1]}" == "--no-cuda" ]]; then
        use_pytorch_cuda=false
        unset 'packages[-1]'
    fi
    
    for pkg in "${packages[@]}"; do
        echo -n -e "\e[36m[INFO]  $pkg -> \e[0m"
        
        CMD_ARGS=("install" "$pkg" "--quiet" "--disable-pip-version-check")
        
        # If our package name contains './wheels/', force-reinstall
        if [[ "$pkg" =~ ^\.\/wheels\/ ]]; then
            CMD_ARGS+=("--force-reinstall")
        fi
        
        if [ "$use_pytorch_cuda" = true ]; then
            CMD_ARGS+=("--extra-index-url" "$CUDA_URL")
        fi
        
        python -m pip "${CMD_ARGS[@]}" 2>/dev/null
        
        if [ $? -eq 0 ]; then
            # Print check mark (✓)
            echo -e "\e[32m✓\e[0m"
        else
            # Print cross mark (✗)
            echo -e "\e[31m✗\e[0m"
            write_error "Failed to install $pkg; continuing..."
        fi
    done
}

# Bypass the fix mode if it's enabled
if [ "$FIX_MODE" == "true" ]; then
    write_info "Running fix section..."
    
    write_info "Installing dependencies for Linux..."
    sudo apt-get update && sudo apt-get install -y build-essential
    
    write_info "Complete cleanup of problematic packages..."
    pip uninstall -y onnx onnxruntime onnxruntime-gpu torch torchvision torchaudio audio-separator fairseq triton
    pip cache purge
    rm -rf ~/.cache/torch_extensions 2>/dev/null
    rm -rf ~/.cache/huggingface 2>/dev/null
    
    write_info "Installing PyTorch ecosystem with EXPLICIT CUDA support..."
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --extra-index-url "$CUDA_URL"
    
    write_info "Verifying CUDA is available in PyTorch..."
    python -c "import torch; print(f'PyTorch CUDA check: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}, Version: {torch.__version__}')"
    python -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available'"
    
    write_info "Installing core packages and dependencies..."
    pip install numpy==2.0.2 protobuf==4.25.3
    pip install TTS fairseq wandb
    if [ -f "./wheels/audiosr-0.0.8-py2.py3-none-any.whl" ]; then
        pip install ./wheels/audiosr-0.0.8-py2.py3-none-any.whl
    fi
    # Use Linux equivalents where necessary
    pip install causal-conv1d
    pip install triton
    pip install mamba-ssm
    
    write_info "Installing specific ONNX and ONNX Runtime versions..."
    pip install onnx==1.15.0
    pip install onnxruntime-gpu==1.16.3
    pip install openvoice-cli
    
    write_info "Installing final dependencies..."
    pip install faiss-cpu
    
    write_info "Installing audio-separator with GPU support..."
    pip install audio-separator[gpu]>=0.32.0
    
    write_info "Installing PyTorch ecosystem with EXPLICIT CUDA support..."
    pip install numpy==2.0.2 pandas numba torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 faiss-cpu fairseq onnxruntime-gpu wandb xformers gradio --extra-index-url "$CUDA_URL" --force-reinstall
    
    write_success "Fix complete! Dependencies have been reinstalled."
    
    # Deactivate venv only if we activated it in this script
    if [ "$IN_VENV" == "false" ]; then
        write_info "Deactivating virtual environment..."
        deactivate
    fi
    
    read -p "Press enter to continue"
    exit 0
fi

# 4) Install fundamental build tools
write_info "Installing build tools (ninja, wheel, setuptools)..."
install_packages "ninja" "wheel" "setuptools"

# Linux-specific essentials
if [ "$IS_LINUX" = true ]; then
    write_info "Installing Linux build essentials..."
    sudo apt-get update && sudo apt-get install -y build-essential
fi

# 5) PyTorch & audio-separator (with extra-index-url)
write_info "Installing PyTorch and related packages..."
install_packages "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" "numpy==2.0.2" "wandb>=0.17.2"

# For Linux, we need triton not triton-windows
if [ "$IS_LINUX" = true ]; then
    write_info "Installing triton (Linux)..."
    install_packages "triton"
else
    write_info "Installing triton-windows..."
    install_packages "triton-windows"
fi

write_info "Installing audio-separator..."
install_packages "audio-separator[gpu]>=0.32.0"

# 6) Process requirements.txt with platform-specific conditions
if [ -f "./requirements.txt" ]; then
    write_info "Processing requirements.txt..."
    while IFS= read -r line || [ -n "$line" ]; do
        # Trim the line
        line=$(echo "$line" | xargs)
        # Skip empty lines and comments
        if [ -z "$line" ] || [[ "$line" =~ ^# ]]; then
            continue
        fi
        
        # Split by semicolon for conditions
        if [[ "$line" == *";"* ]]; then
            pkg_spec=$(echo "$line" | cut -d';' -f1 | xargs)
            cond_text=$(echo "$line" | cut -d';' -f2- | xargs)
            
            # Process conditions (and/or logic)
            ok=true
            IFS='and' read -ra conditions <<< "$cond_text"
            for c in "${conditions[@]}"; do
                c=$(echo "$c" | xargs)
                if [[ "$c" =~ sys_platform\ *==\ *\"win32\" ]]; then
                    if [ "$IS_LINUX" = true ]; then ok=false; break; fi
                elif [[ "$c" =~ sys_platform\ *!=\ *\"win32\" ]]; then
                    if [ "$IS_LINUX" = false ]; then ok=false; break; fi
                elif [[ "$c" =~ python_version\ *==\ *\"([^\"]+)\" ]]; then
                    py_req="${BASH_REMATCH[1]}"
                    if [ "$PY_VERSION" != "$py_req" ]; then ok=false; break; fi
                elif [[ "$c" =~ python_version\ *!=\ *\"([^\"]+)\" ]]; then
                    py_req="${BASH_REMATCH[1]}"
                    if [ "$PY_VERSION" = "$py_req" ]; then ok=false; break; fi
                else
                    write_warning "Unrecognized condition '$c'; skipping $pkg_spec"
                    ok=false
                    break
                fi
            done
            
            if [ "$ok" = true ]; then
                install_packages "$pkg_spec"
            else
                write_info "Skipping $pkg_spec (condition: $cond_text)"
            fi
        else
            install_packages "$line"
        fi
    done < "./requirements.txt"
    write_success "Finished processing requirements.txt"
else
    write_warning "requirements.txt not found; skipping."
fi

# 7) Re-install torch and stuff (for consistency)
write_info "Reinstalling PyTorch and related packages..."
install_packages "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" "numpy>=2.0.2" "wandb>=0.17.2"

# Flash-attention for Linux (from original setup.sh)
if [ "$IS_LINUX" = true ]; then
    write_info "Installing flash-attn (Linux specific)..."
    # Get Python version (e.g., 3.10 -> 310)
    PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")
    # Get torch version (e.g., 2.6.0 -> 2.6)
    TORCH_VERSION=$(python -c "import torch; print('.'.join(torch.__version__.split('.')[:2]))")
    # Simplified CUDA version for URL (e.g., 12.4 -> 12)
    CUDA_SHORT=$(echo $CUDA_VERSION | cut -d. -f1)

    # Construct the flash-attention URL with appropriate versions
    FLASH_ATTN_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu${CUDA_SHORT}torch${TORCH_VERSION}cxx11abiFALSE-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-linux_x86_64.whl"

    echo -n -e "\e[36m[INFO]  flash-attn -> \e[0m"
    python -m pip install "$FLASH_ATTN_URL" 2>/dev/null
    
    if [ $? -ne 0 ]; then
        echo -e "\e[31m✗\e[0m"
        write_info "Trying alternative flash-attn version..."
        # Try with abiTRUE as fallback
        FLASH_ATTN_URL_ALT="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu${CUDA_SHORT}torch${TORCH_VERSION}cxx11abiTRUE-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-linux_x86_64.whl"
        
        echo -n -e "\e[36m[INFO]  flash-attn (alt) -> \e[0m"
        python -m pip install "$FLASH_ATTN_URL_ALT" 2>/dev/null
        
        if [ $? -eq 0 ]; then
            echo -e "\e[32m✓\e[0m"
        else
            echo -e "\e[31m✗\e[0m"
            write_warning "Failed to install flash-attn; continuing..."
        fi
    else
        echo -e "\e[32m✓\e[0m"
    fi
fi

# 8) Verification
write_info "Verifying installations..."
{
    TORCH_INFO=$(python -c "import torch; print('PyTorch CUDA:', torch.cuda.is_available(), 'Device count:', torch.cuda.device_count(), 'Version:', torch.__version__)")
    echo "$TORCH_INFO"
    
    ORT_INFO=$(python -c "import onnxruntime as ort; print('ONNX Runtime:', ort.__version__, 'Providers:', ort.get_available_providers())")
    echo "$ORT_INFO"
    
    SEP_INFO=$(python -c "import audio_separator; print('Audio Separator:', audio_separator.__version__)")
    echo "$SEP_INFO"
    
    WANDB_INFO=$(python -c "import wandb; print('wandb:', wandb.__version__)")
    echo "$WANDB_INFO"
    
    write_success "All verifications passed."
} || {
    write_error "Verification step failed: $?"
}

# 9) Shared library check (platform specific)
if [ "$IS_LINUX" = true ]; then
    write_info "Performing shared library check..."
    {
        python -c "import ctypes; print('Shared libraries: OK')"
        write_success "Library check passed."
    } || {
        write_warning "Library check failed."
    }
else
    write_info "Performing DLL load check..."
    {
        python -c "import ctypes; [ctypes.cdll.LoadLibrary(n) for n in ('vcruntime140.dll','vcruntime140_1.dll')]; print('C++ runtimes loaded OK')"
        write_success "DLL check passed."
    } || {
        write_warning "DLL check failed."
    }
fi

# 10) Deactivate if we activated the venv
if [ "$IN_VENV" == "false" ]; then
    write_info "Deactivating virtual environment..."
    deactivate
fi

# 11) Install espeak-ng 
if [ "$IS_LINUX" = true ]; then
    write_info "Installing espeak-ng (Linux)..."
    sudo apt-get install -y espeak-ng
    write_success "espeak-ng installed."
fi

write_success "All dependencies installed successfully!"
read -p "Press Enter to exit"
