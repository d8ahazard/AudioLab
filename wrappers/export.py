import os
import zipfile
from typing import Any, List, Dict

import librosa

from handlers.ableton import create_ableton_project
from handlers.reaper import create_reaper_project
from util.data_classes import ProjectFiles
from wrappers.base_wrapper import BaseWrapper, TypedInput


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
    """

    def zipdir(path, ziph):
        # ziph is zipfile handle
        for root, dirs, files in os.walk(path):
            for file in files:
                ziph.write(os.path.join(root, file))

    folder_base = os.path.basename(folder_path)
    zipf = zipfile.ZipFile(f"{folder_base}.zip", "w", zipfile.ZIP_DEFLATED)
    zipdir(folder_path, zipf)
    zipf.close()
    return f"{folder_base}.zip"


class Export(BaseWrapper):
    """
    Example class hooking into your existing pipeline,
    detecting BPM if needed, then creating the .als with pylive.
    """

    def register_api_endpoint(self, api) -> Any:
        pass

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
        )
    }

    def process_audio(self, inputs: List[ProjectFiles], callback=None, **kwargs: Dict[str, Any]) -> List[ProjectFiles]:
        for project_file in inputs:
            # The previously processed stems might be in last_outputs
            stems = project_file.last_outputs  # or wherever your actual file list is
            # Default tempo
            bpm = 120

            filtered_kwargs = {key: value for key, value in kwargs.items() if key in self.allowed_kwargs}
            project_format = filtered_kwargs.get("project_format", "Ableton")

            # OPTIONAL: If we want to detect BPM from the "Instrumental" track
            for stem in stems:
                if "(Instrumental)" in stem:
                    # detect_bpm returns (bpm, beat_times)
                    found_bpm, _ = detect_bpm(stem)
                    if found_bpm > 0:
                        bpm = found_bpm
                    break
                if "(Drums)" in stem:
                    found_bpm, _ = detect_bpm(stem)
                    if found_bpm > 0:
                        bpm = found_bpm
                    break
            out_zip = None
            if project_format == "Ableton":
                als_path = create_ableton_project(project_file, bpm)
                out_zip = zip_folder(als_path)
                print(f"Saved Ableton project to: {als_path}")
            elif project_format == "Reaper":
                reaper_path = create_reaper_project(project_file, bpm)
                out_zip = zip_folder(reaper_path)
                print(f"Saved Reaper project to: {reaper_path}")
            last_outputs = project_file.last_outputs
            project_file.add_output("export", out_zip)
            project_file.last_outputs = last_outputs

        return inputs
