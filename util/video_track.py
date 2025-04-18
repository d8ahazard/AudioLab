import os
import xml.etree.ElementTree as ET


class VideoTrack:
    def __init__(
            self,
            track_id,
            next_pointee_id,
            effective_name,
            clip_name,
            color=16,  # Different default color from audio
            # Clip timing (where in the Arrangement we place the clip)
            clip_start=16.0,
            clip_end=238.75,
            # File references
            relative_path="Samples/Imported/YourVideo.mp4",
            absolute_path="C:/Absolute/Path/To/YourVideo.mp4",
            # Optional "file info" placeholders
            original_file_size=0,
            original_crc=0
    ):
        """
        VideoTrack for Ableton Live that includes a video clip.
        Automatically assigns unique sub-IDs for automation so tracks won't collide.

        :param track_id: The integer ID for <AudioTrack Id="..."> (must be unique).
        :param next_pointee_id: The starting pointee ID to be used and incremented for every new pointee.
        :param effective_name: The track name (e.g., "1-MyVideoTrack").
        :param clip_name: The name that appears on the clip itself.
        :param color: Ableton color index.
        :param clip_start: Where the clip starts (in beats).
        :param clip_end: Where it ends (in beats).
        :param relative_path: "Samples/Imported/..."
        :param absolute_path: Full path on disk to the video.
        :param original_file_size: Byte size of file.
        :param original_crc: CRC32 of file.
        """
        self.track_id = track_id
        self.next_pointee_id = next_pointee_id
        self.effective_name = effective_name
        self.clip_name = clip_name
        self.color = color
        self.clip_start = clip_start
        self.clip_end = clip_end
        self.relative_path = relative_path
        self.absolute_path = absolute_path
        self.original_file_size = original_file_size
        self.original_crc = original_crc

        # We'll define a base for automation IDs so each track has unique sub-IDs.
        # Example: if track_id=15, base_automation_id=1500, so no collisions with track_id=14 => 1400.
        self.base_automation_id = track_id * 100

    def get_next_pointee_id(self):
        current_id = self.next_pointee_id
        self.next_pointee_id += 1
        return current_id

    def to_element(self):
        """
        Create an XML element representing a video track in Ableton Live.
        """
        # Video tracks in Ableton are similar to audio tracks with specific settings
        video_track_elem = ET.Element("AudioTrack", {"Id": str(self.track_id)})

        # Standard top-level stuff
        ET.SubElement(video_track_elem, "LomId", {"Value": "0"})
        ET.SubElement(video_track_elem, "LomIdView", {"Value": "0"})
        ET.SubElement(video_track_elem, "IsContentSelectedInDocument", {"Value": "false"})
        ET.SubElement(video_track_elem, "PreferredContentViewMode", {"Value": "0"})

        track_delay_elem = ET.SubElement(video_track_elem, "TrackDelay")
        ET.SubElement(track_delay_elem, "Value", {"Value": "0"})
        ET.SubElement(track_delay_elem, "IsValueSampleBased", {"Value": "false"})

        name_elem = ET.SubElement(video_track_elem, "Name")
        ET.SubElement(name_elem, "EffectiveName", {"Value": self.effective_name})
        ET.SubElement(name_elem, "UserName", {"Value": ""})
        ET.SubElement(name_elem, "Annotation", {"Value": "Video Track"})
        ET.SubElement(name_elem, "MemorizedFirstClipName", {"Value": self.clip_name})

        ET.SubElement(video_track_elem, "Color", {"Value": str(self.color)})
        auto_env_elem = ET.SubElement(video_track_elem, "AutomationEnvelopes")
        ET.SubElement(auto_env_elem, "Envelopes")

        ET.SubElement(video_track_elem, "TrackGroupId", {"Value": "-1"})
        ET.SubElement(video_track_elem, "TrackUnfolded", {"Value": "true"})
        ET.SubElement(video_track_elem, "DevicesListWrapper", {"LomId": "0"})
        ET.SubElement(video_track_elem, "ClipSlotsListWrapper", {"LomId": "0"})
        ET.SubElement(video_track_elem, "ViewData", {"Value": "{}"})

        take_lanes_elem = ET.SubElement(video_track_elem, "TakeLanes")
        ET.SubElement(take_lanes_elem, "TakeLanes")
        ET.SubElement(take_lanes_elem, "AreTakeLanesFolded", {"Value": "true"})

        ET.SubElement(video_track_elem, "LinkedTrackGroupId", {"Value": "-1"})
        ET.SubElement(video_track_elem, "SavedPlayingSlot", {"Value": "-1"})
        ET.SubElement(video_track_elem, "SavedPlayingOffset", {"Value": "0"})
        ET.SubElement(video_track_elem, "Freeze", {"Value": "false"})
        ET.SubElement(video_track_elem, "VelocityDetail", {"Value": "0"})
        ET.SubElement(video_track_elem, "NeedArrangerRefreeze", {"Value": "true"})
        ET.SubElement(video_track_elem, "PostProcessFreezeClips", {"Value": "0"})

        # ---- DeviceChain ----
        device_chain_elem = ET.SubElement(video_track_elem, "DeviceChain")

        # Minimal AutomationLanes
        auto_lanes_elem = ET.SubElement(device_chain_elem, "AutomationLanes")
        nested_auto_lanes = ET.SubElement(auto_lanes_elem, "AutomationLanes")
        automation_lane0 = ET.SubElement(nested_auto_lanes, "AutomationLane", {"Id": "0"})
        ET.SubElement(automation_lane0, "SelectedDevice", {"Value": "0"})
        ET.SubElement(automation_lane0, "SelectedEnvelope", {"Value": "0"})
        ET.SubElement(automation_lane0, "IsContentSelectedInDocument", {"Value": "false"})
        ET.SubElement(automation_lane0, "LaneHeight", {"Value": "68"})
        ET.SubElement(auto_lanes_elem, "AreAdditionalAutomationLanesFolded", {"Value": "false"})

        clip_env_view = ET.SubElement(device_chain_elem, "ClipEnvelopeChooserViewState")
        ET.SubElement(clip_env_view, "SelectedDevice", {"Value": "0"})
        ET.SubElement(clip_env_view, "SelectedEnvelope", {"Value": "0"})
        ET.SubElement(clip_env_view, "PreferModulationVisible", {"Value": "false"})

        # Add standard mixer setup
        mixer_elem = ET.SubElement(device_chain_elem, "Mixer")
        ET.SubElement(mixer_elem, "LomId", {"Value": "0"})
        ET.SubElement(mixer_elem, "LomIdView", {"Value": "0"})
        ET.SubElement(mixer_elem, "IsExpanded", {"Value": "true"})

        on_elem = ET.SubElement(mixer_elem, "On")
        ET.SubElement(on_elem, "LomId", {"Value": "0"})
        ET.SubElement(on_elem, "Manual", {"Value": "true"})
        on_auto_target_id = self.base_automation_id + 0
        at_on = ET.SubElement(on_elem, "AutomationTarget", {"Id": str(on_auto_target_id)})
        ET.SubElement(at_on, "LockEnvelope", {"Value": "0"})
        midi_cc_on_off = ET.SubElement(on_elem, "MidiCCOnOffThresholds")
        ET.SubElement(midi_cc_on_off, "Min", {"Value": "64"})
        ET.SubElement(midi_cc_on_off, "Max", {"Value": "127"})

        # Set volume to 0 (muted) for video tracks
        volume_elem = ET.SubElement(mixer_elem, "Volume")
        ET.SubElement(volume_elem, "LomId", {"Value": "0"})
        ET.SubElement(volume_elem, "Manual", {"Value": "0.0"})  # Muted
        volume_auto_target_id = self.base_automation_id + 1
        at_volume = ET.SubElement(volume_elem, "AutomationTarget", {"Id": str(volume_auto_target_id)})
        ET.SubElement(at_volume, "LockEnvelope", {"Value": "0"})
        midi_cc_volume = ET.SubElement(volume_elem, "MidiCCOnOffThresholds")
        ET.SubElement(midi_cc_volume, "Min", {"Value": "64"})
        ET.SubElement(midi_cc_volume, "Max", {"Value": "127"})

        ET.SubElement(mixer_elem, "ModulationSourceCount", {"Value": "0"})
        ET.SubElement(mixer_elem, "ParametersListWrapper", {"LomId": "0"})

        # For "Pointee," using the next available pointee ID
        pointee_id = self.get_next_pointee_id()
        ET.SubElement(mixer_elem, "Pointee", {"Id": str(pointee_id)})

        ET.SubElement(mixer_elem, "LastSelectedTimeableIndex", {"Value": "0"})
        ET.SubElement(mixer_elem, "LastSelectedClipEnvelopeIndex", {"Value": "0"})
        last_preset_elem = ET.SubElement(mixer_elem, "LastPresetRef")
        ET.SubElement(last_preset_elem, "Value")
        ET.SubElement(mixer_elem, "LockedScripts")
        ET.SubElement(mixer_elem, "IsFolded", {"Value": "false"})
        ET.SubElement(mixer_elem, "ShouldShowPresetName", {"Value": "false"})
        ET.SubElement(mixer_elem, "UserName", {"Value": ""})

        # Add arrangement clips section
        arrangement_clips_elem = ET.SubElement(device_chain_elem, "ArrangerAutomation")
        ET.SubElement(arrangement_clips_elem, "Events")
        clip_time_elem = ET.SubElement(arrangement_clips_elem, "ClipTimeable")
        events_elem = ET.SubElement(clip_time_elem, "ArrangerClipSlotEvents")
        
        # Create video clip entry
        video_clip_elem = ET.SubElement(events_elem, "AudioClip", {
            "Time": str(self.clip_start),
            "Duration": str(self.clip_end - self.clip_start),
            "RelativePathType": "3",
            "IsWarped": "false",
            "WarpMode": "0",
            "IsLoop": "false",
            "Start": "0",
            "End": str(self.clip_end - self.clip_start),
            "LoopStart": "0",
            "LoopEnd": str(self.clip_end - self.clip_start),
            "OutMarker": str(self.clip_end - self.clip_start),
            "HiddenLoopStart": "0",
            "HiddenLoopEnd": str(self.clip_end - self.clip_start)
        })
        
        # Add name and color to clip
        ET.SubElement(video_clip_elem, "Name", {"Value": self.clip_name})
        ET.SubElement(video_clip_elem, "LaunchMode", {"Value": "0"})
        ET.SubElement(video_clip_elem, "LaunchQuantisation", {"Value": "0"})
        ram_mode_elem = ET.SubElement(video_clip_elem, "CurrentStart", {"Value": "0"})
        ET.SubElement(video_clip_elem, "CurrentEnd", {"Value": str(self.clip_end - self.clip_start)})
        ET.SubElement(video_clip_elem, "IsLooping", {"Value": "false"})
        ET.SubElement(video_clip_elem, "LoopStart", {"Value": "0"})
        ET.SubElement(video_clip_elem, "LoopEnd", {"Value": str(self.clip_end - self.clip_start)})
        ET.SubElement(video_clip_elem, "PlayFromSelection", {"Value": "true"})
        ET.SubElement(video_clip_elem, "TimeSelection", {"Value": "0"})
        ET.SubElement(video_clip_elem, "PlayCount", {"Value": "0"})
        
        # Add sample reference for the video file
        sample_ref_elem = ET.SubElement(video_clip_elem, "SampleRef")
        file_ref_elem = ET.SubElement(sample_ref_elem, "FileRef")
        ET.SubElement(file_ref_elem, "RelativePath", {"Value": self.relative_path})
        ET.SubElement(file_ref_elem, "Path", {"Value": self.absolute_path})
        ET.SubElement(file_ref_elem, "Type", {"Value": "4"})  # Type 4 is Video
        ET.SubElement(file_ref_elem, "LivePackName", {"Value": ""})
        ET.SubElement(file_ref_elem, "LivePackId", {"Value": ""})
        ET.SubElement(file_ref_elem, "OriginalFileSize", {"Value": str(self.original_file_size)})
        ET.SubElement(file_ref_elem, "OriginalCrc", {"Value": str(self.original_crc)})
        
        # Add additional video track specific settings
        ET.SubElement(video_clip_elem, "ColorIndex", {"Value": str(self.color)})
        ET.SubElement(video_clip_elem, "VideoWindowRect", {"Value": "29 81 1053 657"})
        ET.SubElement(video_clip_elem, "IsVideoWindowOpen", {"Value": "true"})
        
        return video_track_elem 