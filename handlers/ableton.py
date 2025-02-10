import gzip
import os
import shutil
import wave
import binascii
import xml.etree.ElementTree as ET

from handlers.config import output_path, app_path
from util.audio_track import AudioTrack
from util.data_classes import ProjectFiles


def create_ableton_project(project: ProjectFiles, bpm: int = None):
    project_name = os.path.basename(project.project_dir)
    stems = project.last_outputs

    # Prepare the Ableton project directory: /ableton/<project_name>/
    als_project_dir = os.path.join(output_path, "ableton", project_name)
    os.makedirs(als_project_dir, exist_ok=True)

    # Where we'll copy stems:
    samples_path = os.path.join(als_project_dir, "Samples", "Imported")
    os.makedirs(samples_path, exist_ok=True)

    # Load template XML
    template_xml = os.path.join(app_path, "res", "template.xml")
    tree = ET.parse(template_xml)
    root = tree.getroot()

    # Find <LiveSet> and <Tracks>
    live_set_elem = root.find("LiveSet")
    if live_set_elem is None:
        raise ValueError("Template XML missing <LiveSet> element.")

    tracks_elem = live_set_elem.find("Tracks")
    if tracks_elem is None:
        tracks_elem = ET.SubElement(live_set_elem, "Tracks")

    # -------------------------------------------------------------------------
    # 1) ENSURE EVERY EXISTING TRACK (Master/Return) HAS AN "Id"
    # -------------------------------------------------------------------------
    used_ids = set()
    for tnode in tracks_elem.findall("*Track"):
        old_id = tnode.get("Id")
        if old_id is not None:
            try:
                used_ids.add(int(old_id))
            except ValueError:
                pass  # If not numeric, ignore

    # If no track IDs found, start from 10
    next_id = (max(used_ids) + 1) if used_ids else 10

    # Assign IDs to leftover track-like nodes that have none
    for tnode in tracks_elem.findall("*Track"):
        if "Id" not in tnode.attrib:
            tnode.set("Id", str(next_id))
            next_id += 1

    # -------------------------------------------------------------------------
    # 2) RETAIN ONLY <ReturnTrack> ELEMENTS; DELETE EVERYTHING ELSE IN <Tracks>
    # -------------------------------------------------------------------------
    return_tracks = list(tracks_elem.findall("ReturnTrack"))
    for child in list(tracks_elem):
        tracks_elem.remove(child)

    # -------------------------------------------------------------------------
    # 3) SET PROJECT BPM IF PROVIDED
    # -------------------------------------------------------------------------
    if bpm is not None:
        current_tempo_node = live_set_elem.find("CurrentSongTempo")
        if current_tempo_node is not None:
            current_tempo_node.set("Value", str(bpm))
        else:
            song_tempo_manual = root.find(".//SongTempo/Manual")
            if song_tempo_manual is not None:
                song_tempo_manual.set("Value", str(bpm))

    # -------------------------------------------------------------------------
    # 4) DETERMINE CLIP LENGTH FROM THE FIRST STEM
    # -------------------------------------------------------------------------
    first_stem = stems[0]
    with wave.open(first_stem, 'rb') as w:
        frames = w.getnframes()
        samplerate = w.getframerate()
        track_length_secs = frames / samplerate

    clip_start = 16.0
    clip_end = clip_start + track_length_secs

    # -------------------------------------------------------------------------
    # Prepare global next pointee ID from <NextPointeeId> element or use a default
    # -------------------------------------------------------------------------
    next_pointee_node = live_set_elem.find("NextPointeeId")
    if next_pointee_node is not None:
        try:
            global_next_pointee_id = int(next_pointee_node.get("Value", "0"))
        except ValueError:
            global_next_pointee_id = 1000
    else:
        global_next_pointee_id = 1000

    # -------------------------------------------------------------------------
    # 5) BUILD & APPEND NEW AUDIO TRACKS
    # -------------------------------------------------------------------------
    for idx, stem_path in enumerate(stems):
        stem_basename = os.path.basename(stem_path)
        dest_stem_path = os.path.join(samples_path, stem_basename)
        shutil.copy2(stem_path, dest_stem_path)

        original_file_size = os.path.getsize(dest_stem_path)
        with open(dest_stem_path, 'rb') as f:
            crc_val = binascii.crc32(f.read()) & 0xFFFFFFFF

        track_id = next_id
        next_id += 1

        color_val = idx
        track_basename_no_ext = os.path.splitext(stem_basename)[0]
        effective_name = f"{idx + 1}-{track_basename_no_ext}"
        clip_name = track_basename_no_ext

        # Pass the current global_next_pointee_id to the AudioTrack instance
        audio_track_obj = AudioTrack(
            track_id=track_id,
            next_pointee_id=global_next_pointee_id,
            effective_name=effective_name,
            clip_name=clip_name,
            color=color_val,
            clip_start=clip_start,
            clip_end=clip_end,
            relative_path=f"Samples/Imported/{stem_basename}",
            absolute_path=dest_stem_path,
            original_file_size=original_file_size,
            original_crc=crc_val,
            default_duration=frames,
            default_sample_rate=samplerate
        )

        track_elem = audio_track_obj.to_element()
        # Update the global next pointee ID for subsequent tracks
        global_next_pointee_id = audio_track_obj.next_pointee_id
        tracks_elem.append(track_elem)

    # Append the previously stored <ReturnTrack> elements at the end
    for rt in return_tracks:
        tracks_elem.append(rt)

    # -------------------------------------------------------------------------
    # 6) UPDATE <NextPointeeId> TO AVOID “invalid pointee ID”
    # -------------------------------------------------------------------------
    # Gather all numeric IDs in the entire <LiveSet>
    def gather_all_ids(elem, all_ids):
        if "Id" in elem.attrib:
            val_str = elem.attrib["Id"]
            try:
                val_num = int(val_str)
                all_ids.add(val_num)
            except ValueError:
                pass
        for child in elem:
            gather_all_ids(child, all_ids)

    all_ids = set()
    gather_all_ids(live_set_elem, all_ids)
    max_id_used = max(all_ids) if all_ids else 10

    # Ensure <NextPointeeId Value="..."/> is at least the larger of our global next pointee ID
    # and (max existing ID + 1)
    new_next_pointee_val = max(global_next_pointee_id, max_id_used + 1)
    next_pointee_node = live_set_elem.find("NextPointeeId")
    if next_pointee_node is not None:
        next_pointee_node.set("Value", str(new_next_pointee_val))
    else:
        ET.SubElement(live_set_elem, "NextPointeeId", {"Value": str(new_next_pointee_val)})

    # -------------------------------------------------------------------------
    # 7) PRETTY-PRINT & GZIP THE RESULTING XML AS .als
    # -------------------------------------------------------------------------
    try:
        ET.indent(tree, space="  ")
    except AttributeError:
        pass  # For Python < 3.9, skip

    temp_xml_path = os.path.join(als_project_dir, f"{project_name}_temp.xml")
    tree.write(temp_xml_path, encoding='utf-8', xml_declaration=True)

    als_file_path = os.path.join(als_project_dir, f"{project_name}.als")
    with open(temp_xml_path, "rb") as f_in, gzip.open(als_file_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    os.remove(temp_xml_path)

    # Copy res/Ableton Project Info folder into project
    info_folder = os.path.join(app_path, "res", "Ableton Project Info")
    shutil.copytree(info_folder, os.path.join(als_project_dir, "Ableton Project Info"), dirs_exist_ok=True)
    desktop_ini_path = os.path.join(als_project_dir, "desktop.ini")

    if os.name == 'nt' and not os.path.exists(desktop_ini_path):
        try:
            with open(desktop_ini_path, "w") as f:
                f.write(
                    "[.ShellClassInfo]\n"
                    "ConfirmFileOp=0\n"
                    "NoSharing=0\n"
                    "IconResource=Ableton Project Info\\AProject.ico,0\n"
                )
            # Set desktop.ini file attributes to hidden and system
            os.system(f'attrib +h +s "{desktop_ini_path}"')
            # Mark the folder as a system folder so Windows uses the desktop.ini settings
            os.system(f'attrib +s "{als_project_dir}"')
        except:
            pass

    print(f"Created Ableton project: {als_file_path}")
    return als_project_dir
