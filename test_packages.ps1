# test_install_sequence.ps1
param(
    [switch]$cleanUp = $false
)

# 1. Remove existing venv and create a fresh one (only if cleanUp is true)
if ($cleanUp) {
    if (Test-Path ".\test_venv") {
        Write-Host "Removing existing virtual environment..."
        Remove-Item -Recurse -Force ".\test_venv"
    }
    Write-Host "Creating new virtual environment..."
    python -m venv .\test_venv
} else {
    Write-Host "Using existing virtual environment (use -cleanUp to start fresh)..."
    if (-not (Test-Path ".\test_venv")) {
        Write-Host "Virtual environment not found. Creating new one..."
        python -m venv .\test_venv
    }
}

# 2. Activate venv
Write-Host "Activating virtual environment..."
& .\test_venv\Scripts\Activate.ps1

# 3. Pin pip to avoid omegaconf issues
Write-Host "Downgrading pip to 23.1.2..."
pip install pip==23.1.2

# 4. Determine Python minor version (e.g. "310")
$pyVer = & python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')" 

# 5. Set your CUDA index (CU124)
$cudaIndex = "https://download.pytorch.org/whl/cu124"

$requiredVersions = @{
    "torch" = "2.6.0+cu124"
    "torchvision" = "0.21.0+cu124"
    "torchaudio" = "2.6.0+cu124"
    "torchlibrosa" = "0.1.0"
    "omegaconf" = "2.0.6"
    "numpy" = "1.23.5"
    "pandas" = "2.0.2"
    "scipy" = "1.11.4"
    "audio-separator[gpu]" = "0.28.5"
    "onnxruntime-gpu" = "1.20.1"
}

# Function to restore base packages to specific versions
function RestoreBasePackages {
    Write-Host "Installing/restoring base packages to required versions..."
    pip install --extra-index-url $cudaIndex `
        torch==2.6.0 `
        torchvision==0.21.0 `
        torchaudio==2.6.0 `
        torchlibrosa>=0.1.0 `
        omegaconf==2.0.6 `
        numpy==1.23.5 `
        pandas==2.0.2 `
        scipy==1.11.4 `
        onnxruntime-gpu>=1.20.1
}

# 6. Install base packages initially
RestoreBasePackages

# 7. Hard‑coded full list of everything you want tested
$pkgList = @(
    "torchlibrosa>=0.1.0",
    "fairseq",
    "faiss-cpu",
    "audio-separator[gpu]>=0.28.5",
    "descript-audiotools==0.7.2",
    "descript-audio-codec==1.0.0",
    "demucs>=4.0.1",
    "librosa>=0.11.0",
    "matchering>=2.0.6",
    "pyloudnorm>=0.1.1",
    "soundfile>=0.13.1",
    "torchcrepe",
    "whisperx",
    "diffusers>=0.32.2",
    "einops>=0.8.0",
    "onnxruntime-gpu>=1.20.1",
    "onnxsim",
    "transformers>=4.51.3",
    "x-transformers>=2.3.1",
    "pydub>=0.25.1",
    "wave",
    "kanjize",
    "phonemizer>=3.3.0",
    "sudachidict-full>=20241021",
    "sudachipy>=0.6.10",
    "hf_xet",
    "ml_collections",
    "muq>=0.1.0",
    "mutagen>=1.47.0",
    "praat-parselmouth>=0.4.4",
    "pyworld",
    "reathon",
    "segmentation_models_pytorch>=0.3.4",
    "stable-audio-tools>=0.0.19",
    "torchdiffeq>=0.2.5",
    "webvtt-py",
    "./wheels/audiosr-0.0.9-py3-none-any.whl"
)

# 8. Flash‑attention wheel URLs and select the one matching Py version
$flashUrls = @(
    "https://github.com/kingbri1/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu124torch2.6.0cxx11abiFALSE-cp310-cp310-win_amd64.whl",
    "https://github.com/kingbri1/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu124torch2.6.0cxx11abiFALSE-cp311-cp311-win_amd64.whl",
    "https://github.com/kingbri1/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu124torch2.6.0cxx11abiFALSE-cp312-cp312-win_amd64.whl",
    "https://github.com/kingbri1/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu124torch2.6.0cxx11abiFALSE-cp313-cp313-win_amd64.whl"
)
$selectedFlash = $flashUrls | Where-Object { $_ -match "cp$pyVer-cp$pyVer-win_amd64" }
if ($selectedFlash) {
    $pkgList += $selectedFlash
}

# Simple function to get ALL package versions at once and check them
function Check-BasePackages {
    # Run pip list once
    $pipOutput = pip list
    
    # Extract versions for key packages
    $torch = ($pipOutput | Select-String "^torch\s+" -ErrorAction SilentlyContinue) -split '\s+' | Select-Object -Skip 1 -First 1
    $torchvision = ($pipOutput | Select-String "^torchvision\s+" -ErrorAction SilentlyContinue) -split '\s+' | Select-Object -Skip 1 -First 1
    $numpy = ($pipOutput | Select-String "^numpy\s+" -ErrorAction SilentlyContinue) -split '\s+' | Select-Object -Skip 1 -First 1
    $pandas = ($pipOutput | Select-String "^pandas\s+" -ErrorAction SilentlyContinue) -split '\s+' | Select-Object -Skip 1 -First 1
    $omegaconf = ($pipOutput | Select-String "^omegaconf\s+" -ErrorAction SilentlyContinue) -split '\s+' | Select-Object -Skip 1 -First 1
    
    # Default to "NA" for any packages not found
    if ($null -eq $torch) { $torch = "NA" }
    if ($null -eq $torchvision) { $torchvision = "NA" }
    if ($null -eq $numpy) { $numpy = "NA" }
    if ($null -eq $pandas) { $pandas = "NA" }
    if ($null -eq $omegaconf) { $omegaconf = "NA" }
    
    $needsRestore = $false
    
    # Check versions against expected values
    if ($torch -ne "2.6.0+cu124") {
        Write-Host "torch version mismatch: expected 2.6.0+cu124, got $torch"
        $needsRestore = $true
    }
    if ($torchvision -ne "0.21.0+cu124") {
        Write-Host "torchvision version mismatch: expected 0.21.0, got $torchvision"
        $needsRestore = $true
    }
    if ($numpy -ne "1.23.5") {
        Write-Host "numpy version mismatch: expected 1.23.5, got $numpy"
        $needsRestore = $true
    }
    if ($pandas -ne "2.0.2") {
        Write-Host "pandas version mismatch: expected 2.0.2, got $pandas"
        $needsRestore = $true
    }
    if ($omegaconf -ne "2.0.6") {
        Write-Host "omegaconf version mismatch: expected 2.0.6, got $omegaconf"
        $needsRestore = $true
    }
    
    # Restore packages if needed
    if ($needsRestore) {
        Write-Host "Base package versions changed - restoring..."
        RestoreBasePackages
        return $true
    }
    
    return $false
}

# Function to get all package versions at once for reporting
function Get-AllPackageVersions {
    # Run pip list once
    $pipOutput = pip list
    
    # Extract versions for packages we care about
    $versions = @{}
    
    $versions["torch"] = ($pipOutput | Select-String "^torch\s+" -ErrorAction SilentlyContinue) -split '\s+' | Select-Object -Skip 1 -First 1
    $versions["torchvision"] = ($pipOutput | Select-String "^torchvision\s+" -ErrorAction SilentlyContinue) -split '\s+' | Select-Object -Skip 1 -First 1
    $versions["numba"] = ($pipOutput | Select-String "^numba\s+" -ErrorAction SilentlyContinue) -split '\s+' | Select-Object -Skip 1 -First 1
    $versions["numpy"] = ($pipOutput | Select-String "^numpy\s+" -ErrorAction SilentlyContinue) -split '\s+' | Select-Object -Skip 1 -First 1
    $versions["pandas"] = ($pipOutput | Select-String "^pandas\s+" -ErrorAction SilentlyContinue) -split '\s+' | Select-Object -Skip 1 -First 1
    $versions["onnxruntime-gpu"] = ($pipOutput | Select-String "^onnxruntime-gpu\s+" -ErrorAction SilentlyContinue) -split '\s+' | Select-Object -Skip 1 -First 1
    
    # Default to "NA" for any packages not found
    foreach ($key in @("torch", "torchvision", "numba", "numpy", "pandas", "onnxruntime-gpu")) {
        if ($null -eq $versions[$key]) { $versions[$key] = "NA" }
    }
    
    return $versions
}

# 9. Iterate installs, capture before/after versions, build report
$report = @()
foreach ($pkg in $pkgList) {
    Write-Host "`n=== Installing $pkg ==="

    # Extract base package name (removing version specifiers)
    $basePkg = $pkg -replace '(\[.*\]|>=.*|==.*|<=.*|>.*|<.*|~=.*|\+.*)', ''
    $basePkg = $basePkg.Trim()
    
    # For wheel files, extract name from filename
    if ($pkg -like "*.whl") {
        $basePkg = [System.IO.Path]::GetFileNameWithoutExtension($pkg) -replace '-\d+.*', ''
    }

    # Get versions before installation
    $beforeVersions = Get-AllPackageVersions

    # Install the package
    pip install --extra-index-url $cudaIndex $pkg

    # Get versions after installation/restoration
    $afterVersions = Get-AllPackageVersions

    # Enumerate each key in before/after versions, if the package versions don't match, force-reinstall the package
    foreach ($key in $beforeVersions.Keys) {
        if ($beforeVersions[$key] -ne $afterVersions[$key]) {
            Write-Host "Package $key version mismatch: expected $($beforeVersions[$key]), got $($afterVersions[$key])"
            $requiredVersion = $requiredVersions[$key]
            pip install --extra-index-url $cudaIndex $key==$requiredVersion --force-reinstall
        }
    }


    
    # Append to report
    $report += [PSCustomObject]@{
        Package             = $pkg
        RequiredRestore     = $wasRestored
        Torch_Before        = $beforeVersions["torch"]
        Torch_After         = $afterVersions["torch"]
        Torchvision_Before  = $beforeVersions["torchvision"]
        Torchvision_After   = $afterVersions["torchvision"]
        Numba_Before        = $beforeVersions["numba"]
        Numba_After         = $afterVersions["numba"]
        Numpy_Before        = $beforeVersions["numpy"]
        Numpy_After         = $afterVersions["numpy"]
        Pandas_Before       = $beforeVersions["pandas"]
        Pandas_After        = $afterVersions["pandas"]
        ORT_Before       = $beforeVersions["onnxruntime-gpu"]
        ORT_After        = $afterVersions["onnxruntime-gpu"]
    }
}

# 10. Export CSV report
$csvPath = "install_report.csv"
$report | Export-Csv -Path $csvPath -NoTypeInformation
Write-Host "`nInstallation sequence complete. Report at: $csvPath"

# 11. Deactivate venv
Deactivate
