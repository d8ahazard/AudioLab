#!/bin/bash
# Requires bash shell
set -e # Exit on error

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
    local packages=("$@")
    for pkg in "${packages[@]}"; do
        echo -n -e "\e[36m[INFO]  $pkg -> \e[0m"
        
        CMD_ARGS=("install" "$pkg" "--quiet" "--disable-pip-version-check")
        
        # If our package name contains './wheels/', force-reinstall
        if [[ "$pkg" =~ ^\.\/wheels\/ ]]; then
            # Skip Windows-specific wheels
            if [[ "$pkg" =~ win_amd64 ]] && [[ ! "$pkg" =~ none-any ]]; then
                echo -e "\e[33mskipped (Windows-only)\e[0m"
                continue
            fi
            CMD_ARGS+=("--force-reinstall")
        fi
        
        # Add PyTorch CUDA URL
        CMD_ARGS+=("--extra-index-url" "https://download.pytorch.org/whl/cu124")
        
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
            
            # Process conditions
            ok=true
            IFS='and' read -ra conditions <<< "$cond_text"
            for c in "${conditions[@]}"; do
                c=$(echo "$c" | xargs)
                if [[ "$c" =~ sys_platform\ *==\ *\"win32\" ]]; then
                    # Skip Windows-only packages
                    ok=false; break
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

# 6) Re-install torch and related packages
install_packages "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" "numpy>=2.0.2" "wandb>=0.17.2"

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
