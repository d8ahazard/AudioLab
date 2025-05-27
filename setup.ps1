# Requires PowerShell 5.1+
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Set to $true to enable verbose debug output for requirements.txt condition matching
$DEBUG_REQUIREMENTS = $false 

function Write-Info    { param($m) Write-Host "[INFO]  $m"     -ForegroundColor Cyan }
function Write-Success { param($m) Write-Host "[OK]    $m"     -ForegroundColor Green }
function Write-ErrorLog{ param($m) Write-Host "[ERROR] $m"     -ForegroundColor Red }
function Write-Debug   { param($m) if ($DEBUG_REQUIREMENTS) { Write-Host "[DEBUG] $m" -ForegroundColor Magenta } }

# Detect platform and python version
$isWindows = $env:OS -eq 'Windows_NT'
try {
    $pyVersion = (& python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
} catch {
    Write-ErrorLog "Python not found in PATH. Aborting."
    exit 1
}

# 1) Create or activate venv/conda
$inVenv = $false
if ($env:VIRTUAL_ENV) {
    Write-Info "Detected existing virtualenv at $env:VIRTUAL_ENV"
    $inVenv = $true
} elseif ($env:CONDA_PREFIX) {
    Write-Info "Detected existing Conda environment at $env:CONDA_PREFIX"
    $inVenv = $true
} else {
    if (-not (Test-Path .\venv)) {
        Write-Info "Creating virtual environment in .\venv..."
        python -m venv .\venv
        if ($LASTEXITCODE -ne 0) { Write-ErrorLog "Failed to create venv"; exit 1 }
    }
    Write-Info "Activating virtual environment..."
    . .\venv\Scripts\Activate.ps1
}

# 2) Core installer function (now supports extra pip args)
function Install-Packages {
    param(
        [string[]] $packages,
        [string]   $context,
        [switch]   $UsePytorchCuda = $true
    )
    #Write-Info "Installing $context..."
    foreach ($pkg in $packages) {
        Write-Host -NoNewline "[INFO]  $pkg -> "
        $cmdArgs = @('install', $pkg, '--quiet', '--disable-pip-version-check')
        # If our package name contains './wheels/', force-reinstall
        if ($pkg -match '^\.\/wheels\/') {
            $cmdArgs += '--force-reinstall'
        }
        if ($UsePytorchCuda) {
            $cmdArgs += '--extra-index-url'
            $cmdArgs += 'https://download.pytorch.org/whl/cu124'
        }
        pip @cmdArgs
        if ($LASTEXITCODE -eq 0) {
            # U+2713 is a check mark
            $checkMark = [char]0x2713
            Write-Host $checkMark -ForegroundColor Green
        } else {
            # U+2717 is a cross mark
            $crossMark = [char]0x2717
            Write-Host $crossMark -ForegroundColor Red
            Write-ErrorLog "Failed to install $pkg; continuing..."
        }
    }
}

# 3) Install fundamental build tools
Write-Info "Installing build tools (ninja, wheel, setuptools)..."
Install-Packages -packages @(
    'ninja','wheel','setuptools'
) -context 'build tools (ninja, wheel, setuptools)'

# 4) PyTorch & audio-separator (with extra-index-url)
Write-Info "Installing PyTorch and related packages..."
Install-Packages -packages @(
    'torch==2.6.0','torchvision==0.21.0','torchaudio==2.6.0',
    'triton-windows','numpy==2.0.2', 'wandb>=0.17.2'
) -context 'PyTorch and related packages'

Write-Info "Installing audio-separator..."
Install-Packages -packages @(
    'audio-separator[gpu]>=0.32.0'
) -context 'audio-separator'

if (Test-Path '.\requirements.txt') {
    Write-Info "Processing requirements.txt..."
    Get-Content .\requirements.txt | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line -match '^\s*#') { return }
        $parts = $line -split ';',2
        $pkgSpec = $parts[0].Trim()
        if ($parts.Count -eq 1) {
            Install-Packages -packages @($pkgSpec) -context "package"
        } else {
            $condText  = $parts[1].Trim()
            $conditions= $condText -split '\s+and\s+'
            $ok = $true
            
            Write-Debug "Processing conditions for $pkgSpec"
            Write-Debug "Found $($conditions.Count) condition(s): $condText"
            
            foreach ($c in $conditions) {
                $c = $c.Trim()
                Write-Debug "  Checking condition: '$c'"
                
                if ($c -match 'sys_platform\s*==\s*[''"]win32[''"]') {
                    Write-Debug "    - sys_platform == win32 match"
                    if (-not $isWindows) { 
                        Write-Debug "    - SKIP: Not on Windows"
                        $ok = $false; break 
                    }
                } elseif ($c -match 'sys_platform\s*!=\s*[''"]win32[''"]') {
                    Write-Debug "    - sys_platform != win32 match"
                    if ($isWindows) { 
                        Write-Debug "    - SKIP: On Windows"
                        $ok = $false; break 
                    }
                } elseif ($c -match 'python_version\s*==\s*[''"]([0-9\.]+)[''"]') {
                    $requiredVersion = $Matches[1]
                    Write-Debug "    - python_version == $requiredVersion match"
                    if ($pyVersion -ne $requiredVersion) { 
                        Write-Debug "    - SKIP: Python $pyVersion != $requiredVersion"
                        $ok = $false; break 
                    }
                } elseif ($c -match 'python_version\s*!=\s*[''"]([0-9\.]+)[''"]') {
                    $excludedVersion = $Matches[1]
                    Write-Debug "    - python_version != $excludedVersion match"
                    if ($pyVersion -eq $excludedVersion) { 
                        Write-Debug "    - SKIP: Python $pyVersion == $excludedVersion"
                        $ok = $false; break 
                    }
                } else {
                    Write-Warning "Unrecognized condition '$c'; skipping $pkgSpec"
                    $ok = $false; break
                }
            }
            
            if ($ok) {
                Write-Debug "  All conditions passed, installing $pkgSpec"
                Install-Packages -packages @($pkgSpec) -context "conditional package"
            } else {
                Write-Info "Skipping $pkgSpec (condition: $condText)"
            }
        }
    }
    Write-Success "Finished processing requirements.txt"
} else {
    Write-Warning "requirements.txt not found; skipping."
}

# Re-install torch and stuff

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 numpy>=2.0.2 wandb>=0.17.2 flash_attn --extra-index-url https://download.pytorch.org/whl/cu124 --force-reinstall

Write-Info "Verifying installations..."
try {
    $torchInfo = & python -c "import torch; print('PyTorch CUDA:', torch.cuda.is_available(), 'Device count:', torch.cuda.device_count(), 'Version:', torch.__version__)"
    Write-Host $torchInfo
    $ortInfo   = & python -c "import onnxruntime as ort; print('ONNX Runtime:', ort.__version__, 'Providers:', ort.get_available_providers())"
    Write-Host $ortInfo
    $sepInfo   = & python -c "import audio_separator; print('Audio Separator:', audio_separator.__version__)"
    Write-Host $sepInfo
    $wandbInfo = & python -c "import wandb; print('wandb:', wandb.__version__)"
    Write-Host $wandbInfo
    Write-Success "All verifications passed."
} catch {
    Write-ErrorLog "Verification step failed: $_"
}

# 8) DLL troubleshooting (optional)
Write-Info "Performing DLL load check..."
try {
    & python -c "import ctypes; [ctypes.cdll.LoadLibrary(n) for n in ('vcruntime140.dll','vcruntime140_1.dll')]; print('C++ runtimes loaded OK')"
    Write-Success "DLL check passed."
} catch {
    Write-Warning "DLL check failed: $_"
}

# 9) Deactivate if we activated
if (-not $inVenv) {
    Write-Info "Deactivating virtual environment..."
    . .\venv\Scripts\Deactivate.ps1
}

# 10) Install espeak-ng if missing
if (-not (Test-Path '.\espeak-ng.msi')) {
    Write-Info "Downloading espeak-ng..."
    try {
        Invoke-WebRequest -Uri 'https://github.com/espeak-ng/espeak-ng/releases/download/1.52.0/espeak-ng.msi' `
                          -OutFile '.\espeak-ng.msi' -UseBasicParsing
        Write-Info "Installing espeak-ng..."
        Start-Process msiexec -ArgumentList '/i espeak-ng.msi /quiet /norestart' -Wait -NoNewWindow
        Write-Info "Updating system PATH..."
        setx /M PATH "$($env:PATH);C:\Program Files\eSpeak NG\" | Out-Null
        Write-Success "espeak-ng installed."
    } catch {
        Write-ErrorLog "espeak-ng installation failed: $_"
    }
}

Write-Success "All dependencies installed successfully!"
