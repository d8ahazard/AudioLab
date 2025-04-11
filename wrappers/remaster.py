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
        from fastapi import Body
        
        # Create models for JSON API
        FileData, JsonRequest = self.create_json_models()

        @api.post("/api/v1/process/remaster", tags=["Audio Processing"])
        async def process_remaster_json(
            request: JsonRequest = Body(...)
        ):
            """
            Remaster audio files using Matchering.
            
            This endpoint applies professional-grade mastering to audio files by matching their 
            sonic characteristics to a reference track. It uses the Matchering library to analyze 
            the frequency spectrum, dynamics, and loudness of a reference track and applies those 
            characteristics to your input files.
            
            ## Request Body
            
            ```json
            {
              "files": [
                {
                  "filename": "my_track.wav",
                  "content": "base64_encoded_file_content..."
                },
                {
                  "filename": "reference.wav",  
                  "content": "base64_encoded_file_content..."
                }
              ],
              "settings": {
                "use_source_track_as_reference": false
              }
            }
            ```
            
            ## Parameters
            
            - **files**: Array of file objects, each containing:
              - **filename**: Name of the file (with extension)
              - **content**: Base64-encoded file content
              - The first file is always the track to remaster
              - If a second file is provided and use_source_track_as_reference=false, it's used as the reference
            - **settings**: Remaster settings with the following options:
              - **use_source_track_as_reference**: Use the source file as the reference (default: true)
                - When true, any reference file is ignored
                - When false, a reference_file must be provided as the second file
            
            ## Response
            
            ```json
            {
              "files": [
                {
                  "filename": "remastered_track.wav",
                  "content": "base64_encoded_file_content..."
                }
              ]
            }
            ```
            
            The API returns an object containing the remastered audio file as a Base64-encoded string.
            """
            # Custom handling for reference file
            if len(request.files) > 1 and request.settings and request.settings.get("use_source_track_as_reference", True) == False:
                # Use the second file as reference
                # This will be handled by the process_audio method
                pass
                
            # Use the handle_json_request helper from BaseWrapper
            return self.handle_json_request(request, self.process_audio)

        return process_remaster_json