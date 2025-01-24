import os
from typing import List, Dict, Any

from util.data_classes import ProjectFiles
from wrappers.base_wrapper import BaseWrapper, TypedInput


class Convert(BaseWrapper):
    priority = 10
    title = "Convert"
    default = True
    description = "Convert audio files to MP3 format."
    allowed_kwargs = {
        "bitrate": TypedInput(
            description="Bitrate for the output MP3 file",
            default="320k",  # Default bitrate used by FFMPEG when unspecified
            type=str,
            gradio_type="Dropdown",
            choices=["64k", "96k", "128k", "160k", "192k", "224k", "256k", "320k"],
        ),
    }

    def register_api_endpoint(self, api) -> Any:
        pass

    def process_audio(self, inputs: List[ProjectFiles], callback=None, **kwargs: Dict[str, Any]) -> List[ProjectFiles]:
        bitrate = kwargs.get("bitrate", "192k")  # Default bitrate

        # Filter inputs and initialize progress tracking
        pj_outputs = []
        for project in inputs:
            outputs = []
            input_files, _ = self.filter_inputs(project, "audio")
            non_mp3_inputs = [i for i in input_files if not i.endswith(".mp3")]
            if not non_mp3_inputs:
                continue
            output_folder = os.path.join(project.project_dir)
            os.makedirs(output_folder, exist_ok=True)
            for input_file in non_mp3_inputs:
                file_name, ext = os.path.splitext(os.path.basename(input_file))
                output_file = os.path.join(output_folder, f"{file_name}.mp3")
                if os.path.exists(output_file):
                    os.remove(output_file)
                # Convert to MP3
                os.system(f'ffmpeg -i "{input_file}" -b:a {bitrate} "{output_file}"')
                outputs.append(output_file)
            project.add_output("converted", outputs)
            pj_outputs.append(project)
        return pj_outputs
