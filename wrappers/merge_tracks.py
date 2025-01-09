import os
from typing import Any, List, Callable, Dict

from pydub import AudioSegment

from handlers.config import output_path
from wrappers.base_wrapper import BaseWrapper


class MergeTracks(BaseWrapper):
    priority = 5000

    def process_audio(self, inputs: List[str], callback: Callable = None, **kwargs: Dict[str, Any]) -> List[str]:
        output_folder = os.path.join(output_path, "merged")
        os.makedirs(output_folder, exist_ok=True)
        first_file = inputs[0]
        file_name, file_ext = os.path.splitext(os.path.basename(first_file))
        output_file = os.path.join(output_folder, f"{file_name}_merged{file_ext}")
        stem_0 = AudioSegment.from_file(first_file)
        for i in range(1, len(inputs)):
            stem_i = AudioSegment.from_file(inputs[i])
            stem_0 = stem_0.overlay(stem_i)
        stem_0.export(output_file, format=file_ext[1:])
        return [output_file]

    def register_api_endpoint(self, api) -> Any:
        pass
