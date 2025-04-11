import os
import zipfile
from typing import Any, List, Dict

import librosa
import logging

import numpy as np

from handlers.ableton import create_ableton_project
from handlers.reaper import create_reaper_project
from util.data_classes import ProjectFiles
from wrappers.base_wrapper import BaseWrapper, TypedInput

logger = logging.getLogger(__name__)


def detect_bpm(audio_path: str, start_time: float = 0, end_time: float = None):
    """
    Optional helper to detect BPM of a WAV via librosa.
    """
    bpm = 0
    beat_times = []
    duration = None
    if end_time:
        duration = end_time - start_time

    try:
        y, sr = librosa.load(audio_path, offset=start_time, duration=duration)
        # Convert to real if the signal is complex
        if np.iscomplexobj(y):
            y = y.real
        bpm, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        beat_times = [float(time) for time in beat_times]
        print(f"Estimated tempo for {audio_path}: {bpm}")
    except Exception as e:
        print(f"Error detecting BPM for {audio_path}: {e}")

    return bpm, beat_times

def zip_folder(folder_path: str):
    """
    Optional helper to zip a folder.
    The resulting zip archive will have the project folder (i.e. the base folder)
    as the top-level directory in the archive.
    """

    def zipdir(path, ziph):
        base_folder = os.path.basename(path)
        for root, dirs, files in os.walk(path):
            for file in files:
                abs_file_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_file_path, start=path)
                arcname = os.path.join(base_folder, rel_path)
                ziph.write(abs_file_path, arcname)

    folder_base = os.path.basename(folder_path)
    out_zip = os.path.join(os.path.dirname(folder_path), f"{folder_base}.zip")
    zipf = zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED)
    zipdir(folder_path, zipf)
    zipf.close()
    return out_zip


class Export(BaseWrapper):
    """
    Example class hooking into your existing pipeline,
    detecting BPM if needed, then creating the .als with pylive.
    """

    def register_api_endpoint(self, api) -> Any:
        """
        Register FastAPI endpoint for project export.
        
        Args:
            api: FastAPI application instance
            
        Returns:
            The registered endpoint route
        """
        from fastapi import File, UploadFile, HTTPException, Body
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
        
        # Create models for JSON API
        FileData, JsonRequest = self.create_json_models()

        @api.post("/api/v1/process/export", tags=["Audio Processing"])
        async def process_export(
            files: List[UploadFile] = File(...),
            settings: Optional[SettingsModel] = None
        ):
            """
            Export audio files to DAW project.
            
            Args:
                files: List of audio files to process
                settings: Export settings including project format
                
            Returns:
                Exported project file (zip)
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
                    
                    # Return project file
                    output_files = []
                    for project in processed_files:
                        for output in project.last_outputs:
                            output_path = Path(output)
                            if output_path.exists() and output_path.suffix == '.zip':
                                output_files.append(FileResponse(output))
                    
                    if not output_files:
                        raise HTTPException(status_code=500, detail="No project file generated")
                    
                    return output_files[0]  # Return the first (and should be only) project file
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @api.post("/api/v2/process/export", tags=["Audio Processing"])
        async def process_export_json(
            request: JsonRequest = Body(...)
        ):
            """
            Export audio files to DAW project.
            
            This endpoint takes multiple audio files (typically stems) and creates a Digital Audio Workstation (DAW) 
            project file that includes all the audio tracks properly arranged. The resulting project can be opened 
            directly in Ableton Live or Reaper, depending on the format you choose.
            
            ## Request Body
            
            ```json
            {
              "files": [
                {
                  "filename": "vocals.wav",
                  "content": "base64_encoded_file_content..."
                },
                {
                  "filename": "instrumental.wav",
                  "content": "base64_encoded_file_content..."
                }
              ],
              "settings": {
                "project_format": "Ableton",
                "export_all_stems": true
              }
            }
            ```
            
            ## Parameters
            
            - **files**: Array of file objects, each containing:
              - **filename**: Name of the file (with extension)
              - **content**: Base64-encoded file content
            - **settings**: Export settings with the following options:
              - **project_format**: DAW format to create (default: "Ableton")
                - Options: "Ableton", "Reaper"
              - **export_all_stems**: Include all available stems, not just processed ones (default: true)
              - **pitch_shift**: Apply pitch shift in semitones to non-cloned tracks (default: 0)
            
            ## Response
            
            ```json
            {
              "files": [
                {
                  "filename": "project.zip",
                  "content": "base64_encoded_file_content..."
                }
              ]
            }
            ```
            
            The API returns an object containing the zipped project as a Base64-encoded file.
            """
            # Use the handle_json_request helper from BaseWrapper
            return self.handle_json_request(request, self.process_audio)

        return process_export_json

    title = "Export to Ableton Live"
    description = "Export stems to an Ableton Live project file (.als)"
    priority = 5
    allowed_kwargs = {
        "project_format": TypedInput(
            default="Ableton",
            description="Project format to export.",
            choices=["Ableton", "Reaper"],
            gradio_type="Dropdown",
            type=str,
        ),
        "export_all_stems": TypedInput(
            default=True,
            description="Export all stems, including non-cloned vocals, etc.",
            type=bool,
            gradio_type="Checkbox",
        ),
        "pitch_shift": TypedInput(
            default=0,
            description="Pitch shift in semitones.",
            type=int,
            gradio_type="Slider",
            render=False
        ),
    }

    def process_audio(self, inputs: List[ProjectFiles], callback=None, **kwargs: Dict[str, Any]) -> List[ProjectFiles]:
        try:
            for project_file in inputs:
                # The previously processed stems might be in last_outputs
                stems = project_file.last_outputs  # or wherever your actual file list is
                # Default tempo
                bpm = 120

                filtered_kwargs = {key: value for key, value in kwargs.items() if key in self.allowed_kwargs}
                project_format = filtered_kwargs.get("project_format", "Ableton")
                export_all_stems = filtered_kwargs.get("export_all_stems", True)
                pitch_shift = filtered_kwargs.get("pitch_shift", 0)
                if export_all_stems:
                    stems = project_file.all_outputs()
                    stems_dir = os.path.join(project_file.project_dir, "stems")
                    all_stem_files = [os.path.join(stems_dir, stem) for stem in os.listdir(stems_dir)]
                    for stem in all_stem_files:
                        if stem not in stems:
                            stems.append(stem)

                existing_stems = [stem for stem in stems if os.path.exists(stem)]
                # Remove non-wav files
                existing_stems = [stem for stem in existing_stems if stem.endswith(".wav")]
                for stem in stems:
                    if stem not in existing_stems:
                        logger.error(f"Stem {stem} not found.")
                stems = existing_stems

                # OPTIONAL: If we want to detect BPM from the "Instrumental" track
                for stem in stems:
                    if "(Instrumental)" in stem:
                        # detect_bpm returns (bpm, beat_times)
                        found_bpm, _ = detect_bpm(stem)
                        if found_bpm > 0:
                            print(f"Detected BPM: {found_bpm}")
                            bpm = found_bpm
                        break
                    if "(Drums)" in stem:
                        found_bpm, _ = detect_bpm(stem)
                        if found_bpm > 0:
                            print(f"Detected BPM: {found_bpm}")
                            bpm = found_bpm
                        break
                out_zip = None
                if project_format == "Ableton":
                    als_path = create_ableton_project(project_file, stems, bpm, pitch_shift)
                    out_zip = zip_folder(als_path)
                    print(f"Saved Ableton project to: {als_path}")
                elif project_format == "Reaper":
                    reaper_path = create_reaper_project(project_file, stems, bpm)
                    out_zip = zip_folder(reaper_path)
                    print(f"Saved Reaper project to: {reaper_path}")
                last_outputs = project_file.last_outputs
                project_file.add_output("export", out_zip)
                last_outputs.append(out_zip)
                project_file.last_outputs = last_outputs
        except Exception as e:
            logger.error(f"Error exporting to project: {e}")
            if callback is not None:
                callback(1, "Error exporting to project")
            raise e
        return inputs
