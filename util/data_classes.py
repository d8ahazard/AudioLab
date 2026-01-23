"""
AudioLab Data Classes
=====================

Core data structures used throughout AudioLab for project and file management.
"""

from __future__ import annotations

import os
from pathlib import Path
from shutil import copyfile
from typing import Dict, List, Optional, Union

import xxhash

from handlers.config import output_path


class ProjectFiles:
    """
    Manages file organization for an audio processing project.
    
    Creates a project directory structure based on a hash of the input file,
    allowing for caching and organizing outputs from multiple processing steps.
    
    Attributes:
        src_file: Path to the source file (copied to project directory)
        file_hash: Unique 8-character hash identifying this project
        project_dir: Root directory for all project files
        last_outputs: Most recent output files from processing
        video_sources: Mapping of video source paths
        file_dict: Dictionary mapping process names to their output files
        output_dict: Dictionary mapping process names to outputs (excluding source)
    
    Example:
        >>> project = ProjectFiles("/path/to/song.mp3")
        >>> print(project.project_dir)
        /outputs/process/song_a1b2c3d4
        >>> project.add_output("separate", ["vocals.wav", "instrumental.wav"])
        >>> print(project.all_outputs())
        ['/outputs/process/song_a1b2c3d4/separate/vocals.wav', ...]
    """
    
    src_file: str
    file_hash: str
    project_dir: str
    last_outputs: List[str]
    video_sources: Dict[str, str]
    file_dict: Dict[str, List[str]]
    output_dict: Dict[str, List[str]]
    
    def __init__(self, input_file: Union[str, Path]) -> None:
        """
        Initialize a ProjectFiles instance for the given input file.
        
        Args:
            input_file: Path to the input audio/video file
        
        Raises:
            FileNotFoundError: If input_file does not exist
            IOError: If unable to read the input file
        """
        input_file = str(input_file)
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Generate unique hash for this file
        hash_gen = xxhash.xxh64()
        with open(input_file, 'rb') as f:
            while chunk := f.read(8192):
                hash_gen.update(chunk)
        file_hash = hash_gen.hexdigest()[:8]
        
        # Create project directory structure
        project_name, _ = os.path.splitext(os.path.basename(input_file))
        project_dir = os.path.join(output_path, "process", f"{project_name}_{file_hash}")
        os.makedirs(project_dir, exist_ok=True)
        
        source_dir = os.path.join(project_dir, 'source')
        os.makedirs(source_dir, exist_ok=True)
        
        src_file = os.path.join(source_dir, os.path.basename(input_file))
        if not os.path.exists(src_file):
            copyfile(input_file, src_file)
        
        self.src_file = src_file
        self.file_hash = file_hash
        self.project_dir = project_dir
        self.last_outputs: List[str] = []
        self.video_sources: Dict[str, str] = {}
        
        self.file_dict: Dict[str, List[str]] = {
            'source': [src_file]
        }
        self.output_dict: Dict[str, List[str]] = {}
        
        # Enumerate existing files in project directory
        for root, dirs, files in os.walk(project_dir):
            if root == project_dir:
                continue
            folder_name = os.path.basename(root)
            if folder_name not in self.file_dict:
                self.file_dict[folder_name] = []
            for file in files:
                self.file_dict[folder_name].append(os.path.join(root, file))
    
    def add_output(self, process: str, outputs: Union[List[str], str]) -> None:
        """
        Register output files from a processing step.
        
        Args:
            process: Name of the processing step (e.g., 'separate', 'clone')
            outputs: Single file path or list of file paths
        """
        if isinstance(outputs, str):
            outputs = [outputs]
        
        self.last_outputs = outputs
        
        if process not in self.file_dict:
            self.file_dict[process] = []
        if process not in self.output_dict:
            self.output_dict[process] = []
        
        self.file_dict[process].extend(outputs)
        self.output_dict[process].extend(outputs)
    
    def all_outputs(self) -> List[str]:
        """
        Get all output files from processing steps.
        
        Excludes merge, convert, and export outputs as these are typically
        final deliverables rather than intermediate files.
        
        Returns:
            List of file paths that exist on disk
        """
        excluded_processes = {"merge", "convert", "export"}
        output_list: List[str] = []
        
        for process_name, files in self.output_dict.items():
            if process_name in excluded_processes:
                continue
            for file_path in files:
                if os.path.exists(file_path) and file_path not in output_list:
                    output_list.append(file_path)
        
        return output_list
    
    def get_outputs(self, process: str) -> List[str]:
        """
        Get output files from a specific processing step.
        
        Args:
            process: Name of the processing step
            
        Returns:
            List of file paths for the specified process
        """
        return self.output_dict.get(process, [])
    
    def get_latest_vocals(self) -> Optional[str]:
        """
        Find the most recent vocal track from processing.
        
        Returns:
            Path to the latest vocals file, or None if not found
        """
        for process in reversed(list(self.output_dict.keys())):
            for file_path in self.output_dict[process]:
                if "vocal" in file_path.lower() and os.path.exists(file_path):
                    return file_path
        return None
    
    def get_latest_instrumental(self) -> Optional[str]:
        """
        Find the most recent instrumental track from processing.
        
        Returns:
            Path to the latest instrumental file, or None if not found
        """
        for process in reversed(list(self.output_dict.keys())):
            for file_path in self.output_dict[process]:
                if "instrumental" in file_path.lower() and os.path.exists(file_path):
                    return file_path
        return None
    
    def output_dir(self, process: str) -> str:
        """
        Get or create the output directory for a processing step.
        
        Args:
            process: Name of the processing step
            
        Returns:
            Path to the output directory
        """
        output_path = os.path.join(self.project_dir, process)
        os.makedirs(output_path, exist_ok=True)
        return output_path
    
    def __repr__(self) -> str:
        return f"ProjectFiles(hash={self.file_hash}, dir={self.project_dir})"
