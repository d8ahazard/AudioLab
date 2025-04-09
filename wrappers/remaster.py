import os
from typing import Any, List, Dict

import matchering as mg

from util.data_classes import ProjectFiles
from wrappers.base_wrapper import BaseWrapper, TypedInput
import logging
logger = logging.getLogger(__name__)


class Remaster(BaseWrapper):
    title = "Remaster"
    description = "Remaster audio files using a reference track. Uses Matchering."
    priority = 7
    allowed_kwargs = {
        "use_source_track_as_reference": TypedInput(
            description="Use the source audio as the reference track. (Overrides the reference track input)",
            default=True,
            type=bool,
            gradio_type="Checkbox"
        ),
        "reference_track": TypedInput(
            description="The reference track to use for the remastering process.",
            default=None,
            type=str,
            gradio_type="File"
        )
    }

    def process_audio(self, inputs: List[ProjectFiles], callback=None, **kwargs: Dict[str, Any]) -> List[ProjectFiles]:
        reference_file = kwargs.get("reference_track")
        use_source_track_as_reference = kwargs.get("use_source_track_as_reference")
        if not reference_file and not use_source_track_as_reference:
            raise ValueError("Reference track not provided")
        callback(0, f"Remastering {len(inputs)} audio files", len(inputs))
        callback_step = 0
        pj_outputs = []
        try:
            for project in inputs:
                outputs = []
                if use_source_track_as_reference:
                    reference_file = project.src_file
                output_folder = os.path.join(project.project_dir, "remastered")
                os.makedirs(output_folder, exist_ok=True)
                input_files, _ = self.filter_inputs(project, "audio")
                for input_file in input_files:
                    logger.info(f"Remastering {input_file}")
                    callback(callback_step, f"Remastering {input_file}", len(inputs))
                    inputs_name, inputs_ext = os.path.splitext(os.path.basename(input_file))
                    output_file = os.path.join(output_folder, f"{inputs_name}(Remastered){inputs_ext}")
                    mg.process(
                        # The track you want to master
                        target=input_file,
                        # Some "wet" reference track
                        reference=reference_file,
                        # Where and how to save your results
                        results=[
                            mg.pcm24(output_file),
                        ],
                    )
                    outputs.append(output_file)
                    callback_step += 1
                project.add_output("remaster", outputs)
                pj_outputs.append(project)
        except Exception as e:
            logger.error(f"Error remastering audio: {e}")
            if callback is not None:
                callback(1, "Error remastering audio")
            raise e
        return pj_outputs

    def register_api_endpoint(self, api) -> Any:
        """
        Register FastAPI endpoint for audio remastering.
        
        Args:
            api: FastAPI application instance
            
        Returns:
            The registered endpoint route
        """
        from fastapi import File, UploadFile, HTTPException
        from fastapi.responses import FileResponse
        from pydantic import BaseModel, create_model
        from typing import List, Optional
        import tempfile
        from pathlib import Path

        # Create Pydantic model for settings
        fields = {}
        for key, value in self.allowed_kwargs.items():
            field_type = value.type
            if value.field.default == ...:
                field_type = Optional[field_type]
            fields[key] = (field_type, value.field)
        
        SettingsModel = create_model(f"{self.__class__.__name__}Settings", **fields)

        @api.post("/api/v1/process/remaster")
        async def process_remaster(
            files: List[UploadFile] = File(...),
            reference_file: Optional[UploadFile] = File(None),
            settings: Optional[SettingsModel] = None
        ):
            """
            Remaster audio files using Matchering.
            
            This endpoint applies professional-grade mastering to audio files by matching their 
            sonic characteristics to a reference track. It uses the Matchering library to analyze 
            the frequency spectrum, dynamics, and loudness of a reference track and applies those 
            characteristics to your input files.
            
            ## Parameters
            
            - **files**: Audio files to remaster (WAV, MP3, FLAC)
            - **reference_file**: Optional reference track to use as the sonic target
              - If not provided and use_source_track_as_reference=true, the input file itself will be used
            - **settings**: Remaster settings with the following options:
              - **use_source_track_as_reference**: Use the source file as the reference (default: true)
                - When true, any uploaded reference_file is ignored
                - When false, a reference_file must be provided
            
            ## Example Request
            
            ```python
            import requests
            
            url = "http://localhost:7860/api/v1/process/remaster"
            
            # Upload audio file to remaster
            files = [
                ('files', ('my_track.wav', open('my_track.wav', 'rb'), 'audio/wav'))
            ]
            
            # Upload reference track for mastering target (optional)
            reference = [
                ('reference_file', ('reference.wav', open('reference.wav', 'rb'), 'audio/wav'))
            ]
            
            # Configure remaster parameters
            data = {
                'use_source_track_as_reference': 'false'  # Use the provided reference file
            }
            
            # Send request
            response = requests.post(url, files=files + reference, data=data)
            
            # Save the remastered audio
            with open('remastered_track.wav', 'wb') as f:
                f.write(response.content)
            ```
            
            ## How Matchering Works
            
            The remastering process performs these sophisticated steps:
            
            1. **Analysis**: Both target and reference audio are analyzed for their spectral balance, 
              dynamics, and loudness profiles
            2. **Matching**: A series of processing steps are applied to match the characteristics:
               - Frequency spectrum equalization to match tonal balance
               - Dynamic range processing to match compression profile
               - Loudness normalization to match perceived volume
            3. **Rendering**: The processed audio is exported as high-quality 24-bit audio
            
            ## Use Cases
            
            1. **Consistent Album Sound**: Make all tracks on an album sound cohesive
            2. **Professional Polish**: Give your mixes a commercial-grade sound
            3. **Genre Adaptation**: Make a track fit the sonic profile of a specific genre
            4. **Loudness Optimization**: Achieve competitive loudness while preserving dynamics
            
            ## Response
            
            The API returns the remastered audio files as attachments.
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
                    
                    # Save reference file if provided
                    settings_dict = settings.dict() if settings else {}
                    if reference_file:
                        ref_path = Path(temp_dir) / reference_file.filename
                        with ref_path.open("wb") as f:
                            content = await reference_file.read()
                            f.write(content)
                        settings_dict["reference_track"] = str(ref_path)
                        settings_dict["use_source_track_as_reference"] = False
                    
                    # Process files
                    processed_files = self.process_audio(input_files, **settings_dict)
                    
                    # Return remastered files
                    output_files = []
                    for project in processed_files:
                        for output in project.last_outputs:
                            output_path = Path(output)
                            if output_path.exists():
                                output_files.append(FileResponse(output))
                    
                    return output_files
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        return process_remaster