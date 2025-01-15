import os
from typing import Any, List, Callable, Dict

from pydub import AudioSegment

from handlers.config import output_path
from wrappers.base_wrapper import BaseWrapper


class Merge(BaseWrapper):
    title = "Merge"
    priority = 5

    def process_audio(self, inputs: List[str], callback: Callable = None, **kwargs: Dict[str, Any]) -> List[str]:
        output_folder = os.path.join(output_path, "merged")
        os.makedirs(output_folder, exist_ok=True)
        inputs, outputs = self.filter_inputs(inputs, "audio")

        if not inputs:
            raise ValueError("No valid audio files provided for merging.")

        # Set up output file details
        first_file = inputs[0]
        file_name, file_ext = os.path.splitext(os.path.basename(first_file))
        output_file = os.path.join(output_folder, f"{file_name}_merged{file_ext}")

        # Initialize progress tracking
        total_steps = len(inputs)
        current_step = 0
        if callback:
            callback(0, f"Starting merge of {len(inputs)} tracks", total_steps)

        # Load the first audio file
        stem_0 = AudioSegment.from_file(first_file)
        current_step += 1
        if callback:
            callback(current_step, f"Loaded {os.path.basename(first_file)}", total_steps)

        # Merge remaining audio files
        for i in range(1, len(inputs)):
            stem_i = AudioSegment.from_file(inputs[i])
            stem_0 = stem_0.overlay(stem_i)
            current_step += 1
            if callback:
                callback(current_step, f"Merged {os.path.basename(inputs[i])}", total_steps)

        # Export merged track
        stem_0.export(output_file, format=file_ext[1:])
        if callback:
            callback(total_steps, f"Exported merged track to {output_file}", total_steps)

        return [output_file] + outputs

    def register_api_endpoint(self, api) -> Any:
        pass
