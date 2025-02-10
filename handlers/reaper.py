import os
import shutil
import wave
from reathon.nodes import Project, Track, Item, Source
from handlers.config import output_path
from util.data_classes import ProjectFiles


def create_reaper_project(project: ProjectFiles, bpm: int = None):
    project_name = os.path.basename(project.project_dir)
    stems = project.last_outputs

    # Prepare the Reaper project directory: /reaper/<project_name>/
    reaper_project_dir = os.path.join(project.project_dir, "export", "reaper", project_name)
    os.makedirs(reaper_project_dir, exist_ok=True)

    # Prepare the Media folder where audio files will be stored
    media_path = os.path.join(reaper_project_dir, "Media")
    os.makedirs(media_path, exist_ok=True)

    # Determine track length from the first stem
    first_stem = stems[0]
    with wave.open(first_stem, 'rb') as w:
        frames = w.getnframes()
        samplerate = w.getframerate()
        track_length_secs = frames / samplerate

    tracks = []

    # Create one track per stem with an audio item
    for idx, stem_path in enumerate(stems):
        stem_basename = os.path.basename(stem_path)
        dest_stem_path = os.path.join(media_path, stem_basename)
        shutil.copy2(stem_path, dest_stem_path)

        # Create an effective track name, e.g. "1-MySound"
        track_basename_no_ext = os.path.splitext(stem_basename)[0]
        effective_name = f"{idx + 1}-{track_basename_no_ext}"

        # Create a Source node pointing to the relative path within the project folder
        relative_path = os.path.join("Media", stem_basename)
        source = Source(file=relative_path)

        # Create an Item node with the audio source, starting at 0.0 seconds
        item = Item(source, position=0.0, length=track_length_secs)

        # Create a Track node with the item; if supported, set the track name
        track = Track(item, name=effective_name)
        tracks.append(track)

    # Create the Reaper project using reathon.
    # Pass BPM as a "tempo" property if provided.
    if bpm is not None:
        rpp_project = Project(*tracks, tempo=bpm)
    else:
        rpp_project = Project(*tracks)

    # Write the project to a .rpp file in the project directory
    project_file_path = os.path.join(reaper_project_dir, f"{project_name}.rpp")
    rpp_project.write(project_file_path)

    print(f"Created Reaper project: {project_file_path}")
    return reaper_project_dir
