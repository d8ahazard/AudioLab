import os
from typing import List, Dict, Any, Callable

from handlers.config import output_path
from wrappers.base_wrapper import BaseWrapper, TypedInput


class ConvertToMp3(BaseWrapper):
    priority = 10000
    allowed_kwargs = {
        "bitrate": TypedInput(
            description="Bitrate for the output MP3 file",
            default="192k",  # Default bitrate used by FFMPEG when unspecified
            type=str,
            gradio_type="Dropdown",
            choices=["64k", "96k", "128k", "160k", "192k", "224k", "256k", "320k"],
        ),
    }

    def register_api_endpoint(self, api) -> Any:
        pass

    def process_audio(self, inputs: List[str], callback: Callable = None, **kwargs: Dict[str, Any]) -> List[str]:
        output_folder = os.path.join(output_path, "converted")
        os.makedirs(output_folder, exist_ok=True)
        bitrate = kwargs.get("bitrate", "192k")  # Use the default bitrate if not specified
        for i, file in enumerate(inputs):
            file_name, ext = os.path.splitext(os.path.basename(file))
            if ext == ".wav":
                out_file = os.path.join(output_folder, f"{file_name}.mp3")
                os.system(f'ffmpeg -i "{file}" -b:a {bitrate} "{out_file}"')
                inputs[i] = out_file
                os.remove(file)
        return inputs
