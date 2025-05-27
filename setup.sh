#!/bin/bash
# Requires bash shell
set -e # Exit on error

# Set to true to enable verbose debug output for requirements.txt condition matching
DEBUG_REQUIREMENTS=false

# Define color output functions
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

function write_debug() {
    if [ "$DEBUG_REQUIREMENTS" = true ]; then
        echo -e "\e[35m[DEBUG] $1\e[0m"  # Magenta color
    fi
}

# Detect python version
if ! command -v python &> /dev/null; then
    if command -v python3 &> /dev/null; then
        alias python=python3
    else
        write_error "Python not found in PATH. Aborting."
        exit 1
    fi
fi

# Get Python version
PY_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
write_info "Python version: $PY_VERSION"

# 1) Create or activate venv
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
            write_error "Failed to create venv"; exit 1
        fi
    fi
    write_info "Activating virtual environment..."
    source ./venv/bin/activate
fi

# 2) Core installer function
function install_packages() {
    local pkg="$1"
    write_info "Installing $pkg..."
    
    # Handle local wheel files
    if [[ "$pkg" == ./wheels/* ]]; then
        if [ -f "$pkg" ]; then
            pip install "$pkg"
        else
            write_warning "Local wheel not found: $pkg"
            return 1
        fi
    # Handle direct URLs
    elif [[ "$pkg" == http* ]]; then
        pip install "$pkg"
    # Handle regular packages
    else
        pip install "$pkg"
    fi
}

# 3) Install fundamental build tools
write_info "Installing build tools (ninja, wheel, setuptools)..."
install_packages "ninja" "wheel" "setuptools"

# 4) PyTorch & audio-separator
write_info "Installing PyTorch and related packages..."
install_packages "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" "triton" "numpy==2.0.2" "wandb>=0.17.2"

write_info "Installing audio-separator..."
install_packages "audio-separator[gpu]>=0.32.0"

# 5) Process requirements.txt
if [ -f "./requirements.txt" ]; then
    write_info "Processing requirements.txt..."
    
    # Read requirements.txt line by line
    while IFS= read -r line || [ -n "$line" ]; do
        # Skip empty lines and comments
        [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
        
        # Remove leading/trailing whitespace
        line=$(echo "$line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
        
        # Check if line has platform/version conditions
        if [[ "$line" == *";"* ]]; then
            pkg_spec=$(echo "$line" | cut -d';' -f1 | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
            conditions=$(echo "$line" | cut -d';' -f2- | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
            
            # Process conditions
            install=true
            
            # Check Windows condition
            if [[ "$conditions" == *"sys_platform == \"win32\""* ]]; then
                if [[ ! "$OSTYPE" == "msys"* && ! "$OSTYPE" == "win"* && ! "$OSTYPE" == "cygwin"* ]]; then
                    install=false
                fi
            fi
            
            if [[ "$conditions" == *"sys_platform != \"win32\""* ]]; then
                if [[ "$OSTYPE" == "msys"* || "$OSTYPE" == "win"* || "$OSTYPE" == "cygwin"* ]]; then
                    install=false
                fi
            fi
            
            # Check Python version condition
            if [[ "$conditions" == *"python_version == "* ]]; then
                required_version=$(echo "$conditions" | grep -o 'python_version == "[^"]*"' | cut -d'"' -f2)
                if [[ "$PY_VERSION" != "$required_version" ]]; then
                    install=false
                fi
            fi
            
            # Install if all conditions are met
            if [[ "$install" == true ]]; then
                write_info "Installing (conditions met): $pkg_spec"
                install_packages "$pkg_spec"
            else
                write_info "Skipping (conditions not met): $pkg_spec"
            fi
        else
            # No conditions, install directly
            install_packages "$line"
        fi
    done < "./requirements.txt"
    write_success "Finished processing requirements.txt"
else
    write_warning "requirements.txt not found; skipping."
fi

# 6) Re-install torch and related packages
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 numpy>=2.0.2 wandb>=0.17.2 flash_attn --extra-index-url https://download.pytorch.org/whl/cu124 --force-reinstall

# 7) Verification
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

# 8) Deactivate if we activated
if [ "$IN_VENV" = false ]; then
    write_info "Deactivating virtual environment..."
    deactivate
fi

# 9) Install espeak-ng
write_info "Installing espeak-ng..."
sudo apt-get install -y espeak-ng
write_success "espeak-ng installed."

write_success "All dependencies installed successfully!"
