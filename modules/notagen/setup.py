"""
Setup script for NotaGen module.

This script checks for MuseScore installation and provides instructions if needed.
"""

import os
import subprocess
import shutil
import sys
import logging
import platform

logger = logging.getLogger("ADLB.NotaGen.setup")

def check_musescore_installation():
    """
    Check if MuseScore is installed and properly configured
    
    Returns:
        Tuple of (is_installed, executable_path, instructions)
    """
    # Check if MuseScore is in the PATH
    musescore_path = shutil.which('MuseScore4') or shutil.which('mscore')
    
    if musescore_path:
        # Test if MuseScore can be executed
        try:
            result = subprocess.run(
                [musescore_path, '--version'], 
                check=True,
                capture_output=True,
                text=True
            )
            version = result.stdout.strip() if result.stdout else "Unknown version"
            return True, musescore_path, f"MuseScore is installed at: {musescore_path} ({version})"
        except Exception as e:
            return False, musescore_path, f"MuseScore is found at {musescore_path} but execution failed: {str(e)}"
    
    # MuseScore is not in PATH, provide installation instructions
    system = platform.system()
    if system == "Windows":
        instructions = """
        MuseScore is not installed or not found in your PATH. Please follow these steps:
        
        1. Download MuseScore 4 from https://musescore.org/
        2. Install MuseScore 4 with the default options
        3. Add the MuseScore bin directory to your PATH:
           - Typically located at: C:\\Program Files\\MuseScore 4\\bin
           - Open System Properties > Advanced > Environment Variables
           - Edit the PATH variable and add the MuseScore bin directory
        4. Restart your computer or terminal
        """
    elif system == "Linux":
        instructions = """
        MuseScore is not installed. Please install it using your package manager:
        
        For Ubuntu/Debian:
        ```
        sudo apt update
        sudo apt install musescore
        
        # For headless systems (servers) also install:
        sudo apt install xvfb
        sudo apt install libxcb-xinerama0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libxcb-xkb1 libxkbcommon-x11-0
        ```
        
        For Fedora:
        ```
        sudo dnf install musescore
        ```
        """
    elif system == "Darwin":  # macOS
        instructions = """
        MuseScore is not installed. Please follow these steps:
        
        1. Download MuseScore 4 from https://musescore.org/
        2. Install MuseScore 4 by dragging it to your Applications folder
        3. Add MuseScore to your PATH:
           - Open Terminal
           - Run: echo 'export PATH="$PATH:/Applications/MuseScore 4.app/Contents/MacOS"' >> ~/.zshrc
           - Run: source ~/.zshrc
        """
    else:
        instructions = """
        MuseScore is not installed. Please download and install MuseScore 4 from https://musescore.org/
        and make sure it's available in your PATH.
        """
    
    return False, None, instructions

def setup():
    """Run the setup process for NotaGen"""
    print("Setting up NotaGen module...")
    
    # Check for MuseScore
    is_installed, executable_path, message = check_musescore_installation()
    
    if is_installed:
        print(f"✅ {message}")
    else:
        print(f"❌ MuseScore not found")
        print(message)
    
    # Check for required Python packages
    try:
        import transformers
        print("✅ transformers package is installed")
    except ImportError:
        print("❌ transformers package is missing. Please install it with: pip install transformers")
    
    return is_installed

if __name__ == "__main__":
    setup() 