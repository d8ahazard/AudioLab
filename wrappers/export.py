import os
import shutil
import librosa
from typing import Any, List, Dict

# If your local pylive install is from "live", import like so:
import live
from live import Set, Track, Clip

from handlers.ableton import create_ableton_project
from handlers.config import app_path
from util.data_classes import ProjectFiles
from wrappers.base_wrapper import BaseWrapper


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


class Export(BaseWrapper):
    """
    Example class hooking into your existing pipeline,
    detecting BPM if needed, then creating the .als with pylive.
    """
    title = "Export to Ableton Live"
    description = "Export stems to an Ableton Live project file (.als)"
    priority = 5

    def process_audio(self, inputs: List[ProjectFiles], callback=None, **kwargs: Dict[str, Any]) -> List[ProjectFiles]:
        for project_file in inputs:
            # The previously processed stems might be in last_outputs
            stems = project_file.last_outputs  # or wherever your actual file list is
            # Default tempo
            bpm = 120

            # OPTIONAL: If we want to detect BPM from the "Instrumental" track
            for stem in stems:
                if "(Instrumental)" in stem:
                    # detect_bpm returns (bpm, beat_times)
                    found_bpm, _ = detect_bpm(stem)
                    if found_bpm > 0:
                        bpm = found_bpm
                    break

            als_path = create_ableton_project(project_file, bpm)
            print(f"Saved Ableton project to: {als_path}")

        return inputs
