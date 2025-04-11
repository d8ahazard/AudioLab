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
        from fastapi import File, UploadFile, HTTPException
        from fastapi.responses import FileResponse
        from pydantic import BaseModel, create_model
        from pathlib import Path
        import tempfile

        # Create Pydantic model for settings
        fields = {}
        for key, value in self.allowed_kwargs.items():
            field_type = value.type
            if value.field.default == ...:
                field_type = Optional[field_type]
            fields[key] = (field_type, value.field)
        
        SettingsModel = create_model(f"{self.__class__.__name__}Settings", **fields)

        @api.post("/api/v1/process/convert")
        async def process_convert(
            files: List[UploadFile] = File(...),
            settings: Optional[SettingsModel] = None
        ):
            """
            Convert audio files to MP3 format.
            
            This endpoint converts audio files to MP3 format with configurable bitrate settings.
            It provides a simple way to standardize your audio collection to a consistent format
            while maintaining quality control through bitrate selection.
            
            ## Parameters
            
            - **files**: Audio files to convert (WAV, FLAC, AAC, OGG, etc.)
            - **settings**: Conversion settings with the following options:
              - **bitrate**: Bitrate for the output MP3 file (default: "320k")
                - Options: "64k", "96k", "128k", "160k", "192k", "224k", "256k", "320k"
                - Higher bitrates provide better audio quality but larger file sizes
            
            ## Example Request
            
            ```python
            import requests
            
            url = "http://localhost:7860/api/v1/process/convert"
            
            # Upload audio files
            files = [
                ('files', ('audio.wav', open('audio.wav', 'rb'), 'audio/wav')),
                ('files', ('audio2.flac', open('audio2.flac', 'rb'), 'audio/flac'))
            ]
            
            # Configure conversion parameters
            data = {
                'bitrate': '192k'  # Medium quality, good balance of size and quality
            }
            
            # Send request
            response = requests.post(url, files=files, data=data)
            
            # Save the converted files
            for i, file_response in enumerate(response.json()):
                file_url = file_response['url']
                file_data = requests.get(file_url)
                with open(f'converted_{i}.mp3', 'wb') as f:
                    f.write(file_data.content)
            ```
            
            ## How It Works
            
            The conversion process uses FFmpeg to transcode audio files with these steps:
            
            1. Input files are read and processed one by one
            2. FFmpeg is called with the specified bitrate parameter
            3. The output is saved as an MP3 file with the same basename as the input
            4. The converted files are returned to the client
            
            ## Common Use Cases
            
            1. **Archive Optimization**: Convert high-quality lossless files to MP3 for storage efficiency
            2. **Compatibility**: Ensure audio files work on all devices and players
            3. **Bandwidth Saving**: Create smaller files for streaming or sharing
            
            ## Response
            
            The API returns the converted audio files as attachments.
            """
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Save uploaded files
                    input_files = []
                    for file in files:
                        file_path = Path(temp_dir) / file.filename
                        with file_path.open("wb") as f:
                            content = await file.read()
                            f.write(content)
                        input_files.append(ProjectFiles(str(file_path)))
                    
                    # Process files
                    settings_dict = settings.dict() if settings else {}
                    processed_files = self.process_audio(input_files, **settings_dict)
                    
                    # Return processed files
                    output_files = []
                    for project in processed_files:
                        for output in project.last_outputs:
                            output_path = Path(output)
                            if output_path.exists():
                                output_files.append(FileResponse(output))
                    
                    return output_files
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        return process_convert

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
