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
        from fastapi import Body
        
        # Create models for JSON API
        FileData, JsonRequest = self.create_json_models()

        @api.post("/api/v1/process/export", tags=["Audio Processing"])
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
              - **export_videos**: Reconstruct videos with processed audio (default: true)
            
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
        "export_videos": TypedInput(
            default=True,
            description="Reconstruct videos with processed audio.",
            type=bool,
            gradio_type="Checkbox",
        ),
        "pitch_shift": TypedInput(
            default=0,
            description="Pitch shift in semitones.",
            type=int,
            gradio_type="Slider",
            ge=-12,
            le=12,
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
                export_videos = filtered_kwargs.get("export_videos", True)
                pitch_shift = filtered_kwargs.get("pitch_shift", 0)
                original_videos = filtered_kwargs.get("original_videos", {})
                
                # Check if we have video sources associated with this project
                video_sources = {}
                if hasattr(project_file, "video_sources"):
                    video_sources.update(project_file.video_sources)
                if original_videos:
                    for original_video_path, video_source_path in original_videos.items():
                        if os.path.exists(video_source_path):
                            video_sources[original_video_path] = video_source_path
                
                if export_all_stems:
                    stems = project_file.all_outputs()
                    stems_dir = os.path.join(project_file.project_dir, "stems")
                    if os.path.exists(stems_dir):
                        all_stem_files = [os.path.join(stems_dir, stem) for stem in os.listdir(stems_dir)
                                          if os.path.isfile(os.path.join(stems_dir, stem)) and not stem.endswith(".json")]
                        for stem in all_stem_files:
                            if stem not in stems:
                                stems.append(stem)
                    else:
                        logger.warning(f"Stems directory not found: {stems_dir}")

                # Filter existing stems
                existing_stems = []
                for stem in stems:
                    if os.path.exists(stem):
                        if stem.endswith(".wav"):
                            existing_stems.append(stem)
                        else:
                            logger.warning(f"Ignoring non-WAV file: {stem}")
                    else:
                        logger.error(f"Stem file not found: {stem}")

                if not existing_stems:
                    logger.warning("No valid audio stems found for export")
                    if callback:
                        callback(1.0, "No valid audio stems found for export")
                    return inputs

                stems = existing_stems

                # Process video files if requested
                video_outputs = []
                if export_videos and video_sources:
                    
                    if callback:
                        callback(0.6, "Processing video files...")
                    
                    # Find the main processed audio output
                    main_audio = None
                    for stem in stems:
                        # Prefer vocal stems for cloned audio
                        if "(Vocals)" in stem.lower() or "voice" in stem.lower():
                            main_audio = stem
                            break
                    
                    # If no specific vocal stem is found, use the first WAV
                    if not main_audio and stems:
                        main_audio = stems[0]
                    
                    if main_audio:
                        # Process each video file
                        for original_video_path, video_path in video_sources.items():
                            if os.path.exists(video_path):
                                try:
                                    if callback:
                                        callback(0.7, f"Recombining audio with video: {os.path.basename(video_path)}")

                                    # Find the extracted audio that corresponds to this video
                                    audio_for_video = None

                                    # Look for extracted audio files in the project directory
                                    if hasattr(project_file, "project_dir") and project_file.project_dir:
                                        extracted_audio_pattern = f"{os.path.splitext(os.path.basename(original_video_path))[0]}_extracted.wav"
                                        potential_audio = os.path.join(project_file.project_dir, extracted_audio_pattern)

                                        # Check if this extracted audio exists and is in the stems
                                        if os.path.exists(potential_audio) and potential_audio in stems:
                                            audio_for_video = potential_audio
                                        else:
                                            # Look for any processed version of the extracted audio
                                            for stem in stems:
                                                stem_name = os.path.basename(stem).lower()
                                                if extracted_audio_pattern.lower().replace("_extracted.wav", "") in stem_name:
                                                    audio_for_video = stem
                                                    break

                                    # If we couldn't find a specific audio for this video, use the main audio
                                    if not audio_for_video:
                                        audio_for_video = main_audio
                                    
                                    # Create output video filename in project root
                                    video_name = os.path.splitext(os.path.basename(video_path))[0]
                                    audio_name = os.path.splitext(os.path.basename(audio_for_video))[0]
                                    output_video = os.path.join(project_file.project_dir, f"{video_name}_with_{audio_name}.mp4")

                                    # Recombine video with audio
                                    result_video = self.recombine_audio_with_video(video_path, audio_for_video, output_video)
                                    if result_video:
                                        video_outputs.append(result_video)
                                        logger.info(f"Created video with processed audio: {result_video}")
                                except Exception as e:
                                    logger.error(f"Error processing video {video_path}: {e}")

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
                    try:
                        # Pass video files to the Ableton project creation if available
                        if video_outputs and export_videos:
                            logger.info(f"Including {len(video_outputs)} video files in Ableton project")
                            als_path = create_ableton_project(project_file, stems, bpm, pitch_shift, videos=video_outputs)
                        else:
                            als_path = create_ableton_project(project_file, stems, bpm, pitch_shift)
                        
                        out_zip = zip_folder(als_path)
                        logger.info(f"Saved Ableton project to: {als_path}")
                    except Exception as e:
                        logger.error(f"Error creating Ableton project: {e}")
                        if callback:
                            callback(0.9, f"Error creating Ableton project: {str(e)}")
                        # Continue without breaking - still try to add outputs
                elif project_format == "Reaper":
                    try:
                        reaper_path = create_reaper_project(project_file, stems, bpm)
                        out_zip = zip_folder(reaper_path)
                        logger.info(f"Saved Reaper project to: {reaper_path}")
                    except Exception as e:
                        logger.error(f"Error creating Reaper project: {e}")
                        if callback:
                            callback(0.9, f"Error creating Reaper project: {str(e)}")
                
                # Add outputs to the project
                last_outputs = project_file.last_outputs
                if out_zip:
                    project_file.add_output("export", out_zip)
                    last_outputs.append(out_zip)
                
                # Add any video outputs to the project's last_outputs
                for video_output in video_outputs:
                    if os.path.exists(video_output):
                        project_file.add_output("video", video_output)
                        last_outputs.append(video_output)
                    
                project_file.last_outputs = last_outputs
        except Exception as e:
            logger.error(f"Error exporting to project: {e}")
            if callback is not None:
                callback(1, "Error exporting to project")
            raise e
        return inputs
