#!/usr/bin/env python3
"""
AudioLab Unified Installer
==========================

A cross-platform installer that automatically detects your system configuration
and installs the appropriate dependencies.

Usage:
    python install.py              # Full installation with auto-detection
    python install.py --cpu        # CPU-only installation (no CUDA)
    python install.py --dev        # Include development dependencies
    python install.py --minimal    # Core dependencies only
    python install.py --help       # Show help message
"""

import argparse
import logging
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

# Configure logging with rich formatting if available
try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.panel import Panel
    from rich.table import Table
    
    console = Console()
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )
    RICH_AVAILABLE = True
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s"
    )
    console = None
    RICH_AVAILABLE = False

logger = logging.getLogger("AudioLab Installer")

# Version requirements
MIN_PYTHON_VERSION = (3, 10)
MAX_PYTHON_VERSION = (3, 13)
CUDA_VERSION = "12.8"
TORCH_VERSION = "2.7.1"

# Wheel URLs for platform-specific packages
WHEEL_URLS = {
    "flash_attn": {
        "win32": {
            "3.10": "https://github.com/kingbri1/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu128torch2.7.0cxx11abiFALSE-cp310-cp310-win_amd64.whl",
            "3.11": "https://github.com/kingbri1/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu128torch2.7.0cxx11abiFALSE-cp311-cp311-win_amd64.whl",
            "3.12": "https://github.com/kingbri1/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu128torch2.7.0cxx11abiFALSE-cp312-cp312-win_amd64.whl",
            "3.13": "https://github.com/kingbri1/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu128torch2.7.0cxx11abiFALSE-cp313-cp313-win_amd64.whl",
        }
    },
    "causal_conv1d": {
        "win32": {
            "3.10": "https://github.com/d8ahazard/AudioLab/releases/download/1.0.0/causal_conv1d-1.5.0.post8-cp310-cp310-win_amd64.whl",
        }
    },
    "mamba_ssm": {
        "win32": {
            "3.10": "https://github.com/d8ahazard/AudioLab/releases/download/1.0.0/mamba_ssm-2.2.4-cp310-cp310-win_amd64.whl",
        },
        "linux": {
            "3.10": "https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl",
            "3.11": "https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl",
            "3.12": "https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl",
            "3.13": "https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu12torch2.6cxx11abiFALSE-cp313-cp313-linux_x86_64.whl",
        }
    }
}


class SystemInfo:
    """Detect and store system information."""
    
    def __init__(self):
        self.os_name = platform.system().lower()
        self.os_version = platform.version()
        self.architecture = platform.machine()
        self.python_version = sys.version_info[:2]
        self.python_version_str = f"{self.python_version[0]}.{self.python_version[1]}"
        self.is_windows = self.os_name == "windows" or sys.platform == "win32"
        self.is_linux = self.os_name == "linux"
        self.is_macos = self.os_name == "darwin"
        self.cuda_available = False
        self.cuda_version: Optional[str] = None
        self.gpu_name: Optional[str] = None
        self.gpu_memory: Optional[int] = None
        
        self._detect_gpu()
    
    def _detect_gpu(self):
        """Detect CUDA GPU availability and properties."""
        # Try nvidia-smi first
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split("\n")
                if lines:
                    parts = lines[0].split(",")
                    self.gpu_name = parts[0].strip()
                    if len(parts) > 1:
                        self.gpu_memory = int(parts[1].strip())
                    self.cuda_available = True
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            pass
        
        # Try nvcc for CUDA version
        try:
            result = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "release" in line.lower():
                        # Extract version like "12.4" from "Cuda compilation tools, release 12.4, V12.4.131"
                        import re
                        match = re.search(r"release\s+(\d+\.\d+)", line)
                        if match:
                            self.cuda_version = match.group(1)
                            self.cuda_available = True
                            break
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            pass
    
    def to_dict(self) -> dict:
        return {
            "os": self.os_name,
            "os_version": self.os_version,
            "architecture": self.architecture,
            "python_version": self.python_version_str,
            "cuda_available": self.cuda_available,
            "cuda_version": self.cuda_version,
            "gpu_name": self.gpu_name,
            "gpu_memory_mb": self.gpu_memory,
        }


def print_banner():
    """Print the AudioLab installation banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║     █████╗ ██╗   ██╗██████╗ ██╗ ██████╗ ██╗      █████╗ ██████╗  ║
    ║    ██╔══██╗██║   ██║██╔══██╗██║██╔═══██╗██║     ██╔══██╗██╔══██╗ ║
    ║    ███████║██║   ██║██║  ██║██║██║   ██║██║     ███████║██████╔╝ ║
    ║    ██╔══██║██║   ██║██║  ██║██║██║   ██║██║     ██╔══██║██╔══██╗ ║
    ║    ██║  ██║╚██████╔╝██████╔╝██║╚██████╔╝███████╗██║  ██║██████╔╝ ║
    ║    ╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═════╝  ║
    ║                                                               ║
    ║              Unified Installer v2.0.0                         ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    if RICH_AVAILABLE:
        console.print(Panel(banner, style="bold blue"))
    else:
        print(banner)


def print_system_info(info: SystemInfo):
    """Print detected system information."""
    if RICH_AVAILABLE:
        table = Table(title="System Information", show_header=False, box=None)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Operating System", f"{info.os_name.capitalize()} {info.os_version[:30]}...")
        table.add_row("Architecture", info.architecture)
        table.add_row("Python Version", info.python_version_str)
        table.add_row("CUDA Available", "Yes" if info.cuda_available else "No")
        if info.cuda_version:
            table.add_row("CUDA Version", info.cuda_version)
        if info.gpu_name:
            table.add_row("GPU", info.gpu_name)
        if info.gpu_memory:
            table.add_row("GPU Memory", f"{info.gpu_memory} MB")
        
        console.print(table)
    else:
        print("\n=== System Information ===")
        print(f"  OS: {info.os_name.capitalize()}")
        print(f"  Python: {info.python_version_str}")
        print(f"  CUDA: {'Yes' if info.cuda_available else 'No'}")
        if info.gpu_name:
            print(f"  GPU: {info.gpu_name}")
        print()


def check_python_version() -> bool:
    """Check if Python version is compatible."""
    version = sys.version_info[:2]
    if version < MIN_PYTHON_VERSION:
        logger.error(f"Python {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}+ required. You have {version[0]}.{version[1]}")
        return False
    if version > MAX_PYTHON_VERSION:
        logger.warning(f"Python {version[0]}.{version[1]} may not be fully tested. Recommended: 3.10-3.12")
    return True


def run_pip(args: list, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    """Run pip with the given arguments."""
    cmd = [sys.executable, "-m", "pip"] + args
    try:
        return subprocess.run(
            cmd,
            check=check,
            capture_output=capture,
            text=True
        )
    except subprocess.CalledProcessError as e:
        if capture:
            logger.error(f"pip command failed: {e.stderr}")
        raise


def install_package(package: str, extra_args: list = None, quiet: bool = True) -> bool:
    """Install a single package with pip."""
    args = ["install"]
    if quiet:
        args.append("--quiet")
    args.append(package)
    if extra_args:
        args.extend(extra_args)
    
    try:
        run_pip(args)
        return True
    except subprocess.CalledProcessError:
        logger.warning(f"Failed to install {package}")
        return False


def install_requirements_file(filepath: str, extra_args: list = None) -> bool:
    """Install packages from a requirements file."""
    if not os.path.exists(filepath):
        logger.warning(f"Requirements file not found: {filepath}")
        return False
    
    args = ["install", "-r", filepath, "--quiet"]
    if extra_args:
        args.extend(extra_args)
    
    try:
        run_pip(args)
        return True
    except subprocess.CalledProcessError:
        logger.error(f"Failed to install requirements from {filepath}")
        return False


def install_pytorch(info: SystemInfo, cpu_only: bool = False) -> bool:
    """Install PyTorch with appropriate CUDA support."""
    logger.info("Installing PyTorch...")
    
    packages = [f"torch=={TORCH_VERSION}", f"torchvision==0.22.1", f"torchaudio=={TORCH_VERSION}"]
    
    if cpu_only or not info.cuda_available:
        logger.info("Installing CPU-only PyTorch...")
        extra_args = ["--index-url", "https://download.pytorch.org/whl/cpu"]
    else:
        logger.info(f"Installing PyTorch with CUDA {CUDA_VERSION} support...")
        extra_args = ["--index-url", f"https://download.pytorch.org/whl/cu{CUDA_VERSION.replace('.', '')}"]
    
    args = ["install"] + packages + extra_args + ["--quiet"]
    
    try:
        run_pip(args)
        logger.info("PyTorch installed successfully")
        return True
    except subprocess.CalledProcessError:
        logger.error("Failed to install PyTorch")
        return False


def install_platform_wheels(info: SystemInfo, cpu_only: bool = False) -> bool:
    """Install platform-specific wheel packages."""
    if cpu_only:
        logger.info("Skipping GPU-specific wheels (CPU-only mode)")
        return True
    
    platform_key = "win32" if info.is_windows else "linux"
    py_ver = info.python_version_str
    
    logger.info("Installing platform-specific GPU packages...")
    
    for package_name, platforms in WHEEL_URLS.items():
        if platform_key in platforms and py_ver in platforms[platform_key]:
            url = platforms[platform_key][py_ver]
            logger.info(f"  Installing {package_name}...")
            if not install_package(url, quiet=False):
                logger.warning(f"  Failed to install {package_name} (non-critical)")
    
    return True


def install_espeak(info: SystemInfo) -> bool:
    """Install espeak-ng for phonemizer support."""
    logger.info("Checking espeak-ng installation...")
    
    if info.is_windows:
        # Check if espeak-ng is already installed
        espeak_paths = [
            r"C:\Program Files\eSpeak NG\espeak-ng.exe",
            r"C:\Program Files (x86)\eSpeak NG\espeak-ng.exe",
        ]
        
        for path in espeak_paths:
            if os.path.exists(path):
                logger.info("espeak-ng already installed")
                return True
        
        # Download and install espeak-ng
        logger.info("Downloading espeak-ng installer...")
        msi_url = "https://github.com/espeak-ng/espeak-ng/releases/download/1.52.0/espeak-ng.msi"
        msi_path = "espeak-ng.msi"
        
        try:
            import urllib.request
            urllib.request.urlretrieve(msi_url, msi_path)
            
            logger.info("Installing espeak-ng (requires admin privileges)...")
            subprocess.run(
                ["msiexec", "/i", msi_path, "/quiet", "/norestart"],
                check=True
            )
            os.remove(msi_path)
            logger.info("espeak-ng installed successfully")
            return True
        except Exception as e:
            logger.warning(f"Failed to auto-install espeak-ng: {e}")
            logger.info("Please install manually from: https://github.com/espeak-ng/espeak-ng/releases")
            return False
    
    elif info.is_linux:
        # Check if espeak-ng is available
        if shutil.which("espeak-ng"):
            logger.info("espeak-ng already installed")
            return True
        
        logger.info("Installing espeak-ng via package manager...")
        try:
            # Try apt (Debian/Ubuntu)
            subprocess.run(["sudo", "apt-get", "install", "-y", "espeak-ng"], check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            try:
                # Try dnf (Fedora)
                subprocess.run(["sudo", "dnf", "install", "-y", "espeak-ng"], check=True)
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.warning("Could not auto-install espeak-ng. Please install manually.")
                return False
    
    elif info.is_macos:
        if shutil.which("espeak-ng"):
            logger.info("espeak-ng already installed")
            return True
        
        try:
            subprocess.run(["brew", "install", "espeak-ng"], check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("Could not install espeak-ng. Please run: brew install espeak-ng")
            return False
    
    return True


def verify_installation() -> bool:
    """Verify that key packages are installed correctly."""
    logger.info("Verifying installation...")
    
    checks = [
        ("torch", "import torch; print(f'PyTorch {torch.__version__}')"),
        ("torchaudio", "import torchaudio; print(f'TorchAudio {torchaudio.__version__}')"),
        ("gradio", "import gradio; print(f'Gradio {gradio.__version__}')"),
        ("transformers", "import transformers; print(f'Transformers {transformers.__version__}')"),
    ]
    
    all_passed = True
    for name, check_code in checks:
        try:
            result = subprocess.run(
                [sys.executable, "-c", check_code],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                logger.info(f"  ✓ {result.stdout.strip()}")
            else:
                logger.warning(f"  ✗ {name} check failed")
                all_passed = False
        except Exception as e:
            logger.warning(f"  ✗ {name} check failed: {e}")
            all_passed = False
    
    # Check CUDA availability
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            logger.info(f"  {result.stdout.strip()}")
    except Exception:
        pass
    
    return all_passed


def normalize_separator_runtime(cpu_only=False):
    """Stage one ONNX namespace owner without replacing DLLs in a live app."""
    import json
    import tempfile
    import site
    runtime = "onnxruntime==1.22.0" if cpu_only else "onnxruntime-gpu==1.22.0"
    runtime_root = Path(sys.prefix) / "audio_separator_runtime"
    runtime_root.mkdir(exist_ok=True)
    target = Path(tempfile.mkdtemp(prefix="0.47.0-", dir=runtime_root))
    logger.info("Staging AudioSeparator runtime in %s", target)
    try:
        run_pip(["install", "--target", str(target), "--no-deps", "audio-separator==0.47.0",
                 "onnx-weekly==1.20.0.dev20251005", "onnx2torch-py313==1.6.0",
                 "protobuf==4.25.8", "ml_dtypes==0.6.0", "tensorboardX==2.6.5", runtime])
        probe = "import sys; sys.path.insert(0, " + repr(str(target)) + "); import torch, onnx, onnxruntime, onnx2torch, tensorboardX; from audio_separator.separator import Separator"
        subprocess.run([sys.executable, "-c", probe], check=True)
        site_dir = next(Path(x) for x in site.getsitepackages() if Path(x).name == "site-packages")
        (site_dir / "audiolab_separator_runtime.pth").write_text(
            "import sys; sys.path.insert(0, " + repr(str(target)) + ")\n", encoding="utf-8")
        (target / "audiolab-runtime.json").write_text(json.dumps({"audio_separator": "0.47.0", "runtime": runtime}))
        logger.info("AudioSeparator runtime activated for new processes; restart running AudioLab sessions.")
        return True
    except (subprocess.CalledProcessError, OSError):
        logger.exception("Runtime staging failed; the previous runtime remains selected")
        return False


def main():
    """Main installation function."""
    parser = argparse.ArgumentParser(
        description="AudioLab Unified Installer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python install.py              # Full installation with GPU support
    python install.py --cpu        # CPU-only installation
    python install.py --dev        # Include development dependencies
    python install.py --minimal    # Core dependencies only
        """
    )
    parser.add_argument("--cpu", action="store_true", help="Install CPU-only version (no CUDA)")
    parser.add_argument("--dev", action="store_true", help="Include development dependencies")
    parser.add_argument("--minimal", action="store_true", help="Install minimal core dependencies only")
    parser.add_argument("--skip-espeak", action="store_true", help="Skip espeak-ng installation")
    parser.add_argument("--skip-verify", action="store_true", help="Skip installation verification")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Print banner
    print_banner()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Detect system information
    info = SystemInfo()
    print_system_info(info)
    
    # Get script directory
    script_dir = Path(__file__).parent.resolve()
    os.chdir(script_dir)
    
    # Upgrade pip and install build tools
    logger.info("Upgrading pip and installing build tools...")
    run_pip(["install", "--upgrade", "pip", "wheel", "setuptools", "--quiet"])
    
    # Install PyTorch
    if not install_pytorch(info, cpu_only=args.cpu):
        logger.error("PyTorch installation failed. Aborting.")
        sys.exit(1)
    
    # Install custom wheels first
    wheels_file = script_dir / "requirements-wheels.txt"
    if wheels_file.exists():
        logger.info("Installing custom wheel packages...")
        install_requirements_file(str(wheels_file))
    
    # Install core requirements
    if args.minimal:
        logger.info("Installing minimal core dependencies...")
        # Install from pyproject.toml dependencies
        run_pip(["install", "-e", ".", "--quiet"])
    else:
        core_file = script_dir / "requirements-core.txt"
        if core_file.exists():
            logger.info("Installing core dependencies...")
            install_requirements_file(str(core_file))
        else:
            # Fallback to main requirements
            logger.info("Installing dependencies from requirements.txt...")
            install_requirements_file(str(script_dir / "requirements.txt"))
    
    # Install CUDA-specific packages
    if not args.cpu and info.cuda_available:
        cuda_file = script_dir / "requirements-cuda.txt"
        if cuda_file.exists():
            logger.info("Installing CUDA dependencies...")
            install_requirements_file(
                str(cuda_file),
                extra_args=["--extra-index-url", f"https://download.pytorch.org/whl/cu{CUDA_VERSION.replace('.', '')}"]
            )
        
        # Install platform-specific wheels
        install_platform_wheels(info, cpu_only=args.cpu)
    
    # Install development dependencies
    if args.dev:
        dev_file = script_dir / "requirements-dev.txt"
        if dev_file.exists():
            logger.info("Installing development dependencies...")
            install_requirements_file(str(dev_file))
    
    if not normalize_separator_runtime(cpu_only=args.cpu):
        logger.error("AudioSeparator runtime normalization failed. Aborting.")
        sys.exit(1)

    # Install espeak-ng
    if not args.skip_espeak:
        install_espeak(info)
    
    # Verify installation
    if not args.skip_verify:
        print()
        if verify_installation():
            logger.info("\n✓ Installation completed successfully!")
        else:
            logger.warning("\n⚠ Installation completed with warnings. Some packages may need manual attention.")
    else:
        logger.info("\n✓ Installation completed!")
    
    # Print next steps
    print("\n" + "="*60)
    print("Next Steps:")
    print("  1. Activate your virtual environment (if not already active)")
    print("  2. Run AudioLab: python main.py")
    print("  3. Open http://127.0.0.1:7860 in your browser")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
