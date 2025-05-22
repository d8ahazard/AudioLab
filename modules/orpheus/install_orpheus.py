#!/usr/bin/env python
"""
Installation script for Orpheus TTS dependencies.
"""

import os
import sys
import subprocess
import logging
import platform

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OrpheusInstaller")

def install_package(package, version=None):
    """Install a package using pip."""
    package_str = f"{package}=={version}" if version else package
    
    try:
        logger.info(f"Installing {package_str}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_str])
        logger.info(f"Successfully installed {package_str}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install {package_str}: {e}")
        return False

def main():
    """Main installation function."""
    logger.info("Starting Orpheus TTS installation...")
    
    # Upgrade pip first
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    except:
        logger.warning("Failed to upgrade pip. Continuing with installation...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        logger.error("Python 3.8 or higher is required for Orpheus TTS")
        sys.exit(1)
    
    # Install orpheus-tts
    success = install_package("orpheus-tts")
    
    if success:
        logger.info("Orpheus TTS installation completed successfully")
        
        # Test the installation
        try:
            import orpheus_tts
            logger.info(f"Orpheus TTS version: {orpheus_tts.__version__}")
            logger.info("You can now use the Orpheus TTS module in AudioLab")
        except ImportError as e:
            logger.error(f"Failed to import orpheus_tts: {e}")
            logger.error("Please check the installation and try again")
    else:
        logger.error("Failed to install Orpheus TTS. Please check the logs for details")
        sys.exit(1)

if __name__ == "__main__":
    main() 