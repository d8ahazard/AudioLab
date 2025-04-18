import os
from shutil import copyfile
from typing import Union, List

import xxhash

from handlers.config import output_path


class ProjectFiles:
    def __init__(self, input_file):
        hash_gen = xxhash.xxh64()
        with open(input_file, 'rb') as f:
            while chunk := f.read(8192):  # Read in chunks for large files
                hash_gen.update(chunk)
        file_hash = hash_gen.hexdigest()[:8]  # Shorten the hash to 8 characters for brevity
        # Create a project directory for the input file and hash
        project_name, _ = os.path.splitext(os.path.basename(input_file))
        project_dir = os.path.join(output_path, "process", f"{project_name}_{file_hash}")
        os.makedirs(project_dir, exist_ok=True)
        source_dir = os.path.join(project_dir, 'source')
        os.makedirs(source_dir, exist_ok=True)
        src_file = os.path.join(source_dir, os.path.basename(input_file))
        if not os.path.exists(src_file):
            # Copy the file
            copyfile(input_file, src_file)

        self.src_file = src_file
        self.file_hash = file_hash
        self.project_dir = project_dir
        self.last_outputs = []
        self.video_sources = {}

        self.file_dict = {
            'source': [src_file]
        }
        self.output_dict = {}

        # Enumerate other folders/files in the project_dir
        for root, dirs, files in os.walk(project_dir):
            if root == project_dir:
                continue
            folder_name = os.path.basename(root)
            if folder_name not in self.file_dict:
                self.file_dict[folder_name] = []
            for file in files:
                self.file_dict[folder_name].append(os.path.join(root, file))

    def add_output(self, process: str, outputs: Union[List[str], str]):
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
        output_list = []
        for key in self.output_dict:
            if key != "merge" and key != "convert" and key != "export":
                for file in self.output_dict[key]:
                    if os.path.exists(file) and file not in output_list:
                        output_list.append(file)
        return output_list


