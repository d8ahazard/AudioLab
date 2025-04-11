import os
import subprocess
from typing import List, Dict, Any, Optional

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
        """
        Register FastAPI endpoint for audio format conversion.
        
        Args:
            api: FastAPI application instance
            
        Returns:
            The registered endpoint route
        """
        from fastapi import Body
        
        # Create models for JSON API
        FileData, JsonRequest = self.create_json_models()

        @api.post("/api/v1/process/convert", tags=["Audio Processing"])
        async def process_convert_json(
            request: JsonRequest = Body(...)
        ):
            """
            Convert audio files to MP3 format.
            
            This endpoint converts audio files to MP3 format with configurable bitrate settings.
            It provides a simple way to standardize your audio collection to a consistent format
            while maintaining quality control through bitrate selection.
            
            ## Request Body
            
            ```json
            {
              "files": [
                {
                  "filename": "audio.wav",
                  "content": "base64_encoded_file_content..."
                }
              ],
              "settings": {
                "bitrate": "192k"
              }
            }
            ```
            
            ## Parameters
            
            - **files**: Array of file objects, each containing:
              - **filename**: Name of the file (with extension)
              - **content**: Base64-encoded file content
            - **settings**: Conversion settings with the following options:
              - **bitrate**: Bitrate for the output MP3 file (default: "320k")
                - Options: "64k", "96k", "128k", "160k", "192k", "224k", "256k", "320k"
                - Higher bitrates provide better audio quality but larger file sizes
            
            ## Response
            
            ```json
            {
              "files": [
                {
                  "filename": "audio.mp3",
                  "content": "base64_encoded_file_content..."
                }
              ]
            }
            ```
            
            The API returns an object containing an array of files, each with filename and Base64-encoded content.
            """
            # Use the handle_json_request helper from BaseWrapper
            return self.handle_json_request(request, self.process_audio)

        return process_convert_json

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
            for idx, input_file in enumerate(non_mp3_inputs):
                if callback is not None:
                    pct_done = int((idx + 1) / len(non_mp3_inputs))
                    callback(pct_done, f"Converting {os.path.basename(input_file)}", len(non_mp3_inputs))
                file_name, ext = os.path.splitext(os.path.basename(input_file))
                output_file = os.path.join(output_folder, f"{file_name}.mp3")
                if os.path.exists(output_file):
                    os.remove(output_file)
                # Convert to MP3
                subprocess.run(
                    f'ffmpeg -i "{input_file}" -b:a {bitrate} "{output_file}"',
                    shell=True,
                    stdout=subprocess.DEVNULL,  # Suppress stdout
                    stderr=subprocess.PIPE,  # Redirect stderr to capture errors (optional)
                )
                outputs.append(output_file)
            project.add_output("converted", outputs)
            pj_outputs.append(project)
        return pj_outputs
