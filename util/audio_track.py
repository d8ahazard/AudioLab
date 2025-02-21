import xml.etree.ElementTree as ET

from pydub import AudioSegment


class AudioTrack:
    def __init__(
            self,
            track_id,
            next_pointee_id,
            effective_name,
            clip_name,
            color=7,
            # Clip timing (where in the Arrangement we place the clip)
            clip_start=16.0,
            clip_end=238.75,
            # File references
            relative_path="Samples/Imported/YourFile.wav",
            absolute_path="C:/Absolute/Path/To/YourFile.wav",
            # Optional “file info” placeholders
            original_file_size=0,
            original_crc=0,
            default_duration=44100,
            default_sample_rate=44100,
            pitch_shift=0
    ):
        """
        AudioTrack that includes 2 Send knobs referencing Return A (Id=0) and B (Id=1).
        Automatically assigns unique sub-IDs for automation so tracks won't collide.

        :param track_id: The integer ID for <AudioTrack Id="..."> (must be unique).
        :param next_pointee_id: The starting pointee ID to be used and incremented for every new pointee.
        :param effective_name: The track name (e.g., "1-MyTrack").
        :param clip_name: The name that appears on the clip itself.
        :param color: Ableton color index.
        :param clip_start: Where the clip starts (in beats).
        :param clip_end: Where it ends (in beats).
        :param relative_path: "Samples/Imported/..."
        :param absolute_path: Full path on disk to the sample.
        :param original_file_size: Byte size of file.
        :param original_crc: CRC32 of file.
        :param default_duration: Number of audio frames (for some versions of Live).
        :param default_sample_rate: Sample rate of the file (e.g. 44100).
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
        self.default_duration = default_duration
        self.default_sample_rate = default_sample_rate
        self.pitch_shift = pitch_shift
        self.is_warped = self.pitch_shift != 0

        # We'll define a base for automation IDs so each track has unique sub-IDs.
        # Example: if track_id=15, base_automation_id=1500, so no collisions with track_id=14 => 1400.
        self.base_automation_id = track_id * 100

    def get_next_pointee_id(self):
        current_id = self.next_pointee_id
        self.next_pointee_id += 1
        return current_id

    def to_element(self):
        audio_track_elem = ET.Element("AudioTrack", {"Id": str(self.track_id)})

        # Standard top-level stuff
        ET.SubElement(audio_track_elem, "LomId", {"Value": "0"})
        ET.SubElement(audio_track_elem, "LomIdView", {"Value": "0"})
        ET.SubElement(audio_track_elem, "IsContentSelectedInDocument", {"Value": "false"})
        ET.SubElement(audio_track_elem, "PreferredContentViewMode", {"Value": "0"})

        track_delay_elem = ET.SubElement(audio_track_elem, "TrackDelay")
        ET.SubElement(track_delay_elem, "Value", {"Value": "0"})
        ET.SubElement(track_delay_elem, "IsValueSampleBased", {"Value": "false"})

        name_elem = ET.SubElement(audio_track_elem, "Name")
        ET.SubElement(name_elem, "EffectiveName", {"Value": self.effective_name})
        ET.SubElement(name_elem, "UserName", {"Value": ""})
        ET.SubElement(name_elem, "Annotation", {"Value": ""})
        ET.SubElement(name_elem, "MemorizedFirstClipName", {"Value": self.clip_name})

        ET.SubElement(audio_track_elem, "Color", {"Value": str(self.color)})
        auto_env_elem = ET.SubElement(audio_track_elem, "AutomationEnvelopes")
        ET.SubElement(auto_env_elem, "Envelopes")

        ET.SubElement(audio_track_elem, "TrackGroupId", {"Value": "-1"})
        ET.SubElement(audio_track_elem, "TrackUnfolded", {"Value": "true"})
        ET.SubElement(audio_track_elem, "DevicesListWrapper", {"LomId": "0"})
        ET.SubElement(audio_track_elem, "ClipSlotsListWrapper", {"LomId": "0"})
        ET.SubElement(audio_track_elem, "ViewData", {"Value": "{}"})

        take_lanes_elem = ET.SubElement(audio_track_elem, "TakeLanes")
        ET.SubElement(take_lanes_elem, "TakeLanes")
        ET.SubElement(take_lanes_elem, "AreTakeLanesFolded", {"Value": "true"})

        ET.SubElement(audio_track_elem, "LinkedTrackGroupId", {"Value": "-1"})
        ET.SubElement(audio_track_elem, "SavedPlayingSlot", {"Value": "-1"})
        ET.SubElement(audio_track_elem, "SavedPlayingOffset", {"Value": "0"})
        ET.SubElement(audio_track_elem, "Freeze", {"Value": "false"})
        ET.SubElement(audio_track_elem, "VelocityDetail", {"Value": "0"})
        ET.SubElement(audio_track_elem, "NeedArrangerRefreeze", {"Value": "true"})
        ET.SubElement(audio_track_elem, "PostProcessFreezeClips", {"Value": "0"})

        # ---- DeviceChain ----
        device_chain_elem = ET.SubElement(audio_track_elem, "DeviceChain")

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

        # Input Routings
        audio_in_elem = ET.SubElement(device_chain_elem, "AudioInputRouting")
        ET.SubElement(audio_in_elem, "Target", {"Value": "AudioIn/External/S0"})
        ET.SubElement(audio_in_elem, "UpperDisplayString", {"Value": "Ext. In"})
        ET.SubElement(audio_in_elem, "LowerDisplayString", {"Value": "1/2"})
        mpe_audio_in = ET.SubElement(audio_in_elem, "MpeSettings")
        ET.SubElement(mpe_audio_in, "ZoneType", {"Value": "0"})
        ET.SubElement(mpe_audio_in, "FirstNoteChannel", {"Value": "1"})
        ET.SubElement(mpe_audio_in, "LastNoteChannel", {"Value": "15"})

        midi_in_elem = ET.SubElement(device_chain_elem, "MidiInputRouting")
        ET.SubElement(midi_in_elem, "Target", {"Value": "MidiIn/External.All/-1"})
        ET.SubElement(midi_in_elem, "UpperDisplayString", {"Value": "Ext: All Ins"})
        ET.SubElement(midi_in_elem, "LowerDisplayString", {"Value": ""})
        mpe_midi_in = ET.SubElement(midi_in_elem, "MpeSettings")
        ET.SubElement(mpe_midi_in, "ZoneType", {"Value": "0"})
        ET.SubElement(mpe_midi_in, "FirstNoteChannel", {"Value": "1"})
        ET.SubElement(mpe_midi_in, "LastNoteChannel", {"Value": "15"})

        audio_out_elem = ET.SubElement(device_chain_elem, "AudioOutputRouting")
        ET.SubElement(audio_out_elem, "Target", {"Value": "AudioOut/Master"})
        ET.SubElement(audio_out_elem, "UpperDisplayString", {"Value": "Master"})
        ET.SubElement(audio_out_elem, "LowerDisplayString", {"Value": ""})
        mpe_audio_out = ET.SubElement(audio_out_elem, "MpeSettings")
        ET.SubElement(mpe_audio_out, "ZoneType", {"Value": "0"})
        ET.SubElement(mpe_audio_out, "FirstNoteChannel", {"Value": "1"})
        ET.SubElement(mpe_audio_out, "LastNoteChannel", {"Value": "15"})

        midi_out_elem = ET.SubElement(device_chain_elem, "MidiOutputRouting")
        ET.SubElement(midi_out_elem, "Target", {"Value": "MidiOut/None"})
        ET.SubElement(midi_out_elem, "UpperDisplayString", {"Value": "None"})
        ET.SubElement(midi_out_elem, "LowerDisplayString", {"Value": ""})
        mpe_midi_out = ET.SubElement(midi_out_elem, "MpeSettings")
        ET.SubElement(mpe_midi_out, "ZoneType", {"Value": "0"})
        ET.SubElement(mpe_midi_out, "FirstNoteChannel", {"Value": "1"})
        ET.SubElement(mpe_midi_out, "LastNoteChannel", {"Value": "15"})

        # ---- Mixer with 2 Sends (Id=0, Id=1) referencing Return A/B ----
        mixer_elem = ET.SubElement(device_chain_elem, "Mixer")
        ET.SubElement(mixer_elem, "LomId", {"Value": "0"})
        ET.SubElement(mixer_elem, "LomIdView", {"Value": "0"})
        ET.SubElement(mixer_elem, "IsExpanded", {"Value": "true"})

        on_elem = ET.SubElement(mixer_elem, "On")
        ET.SubElement(on_elem, "LomId", {"Value": "0"})
        ET.SubElement(on_elem, "Manual", {"Value": "true"})
        # Example AutomationTarget: each track uses base_automation_id + offset.
        # For "On," let's do offset=0.
        on_auto_target_id = self.base_automation_id + 0
        at_on = ET.SubElement(on_elem, "AutomationTarget", {"Id": str(on_auto_target_id)})
        ET.SubElement(at_on, "LockEnvelope", {"Value": "0"})
        midi_cc_on_off = ET.SubElement(on_elem, "MidiCCOnOffThresholds")
        ET.SubElement(midi_cc_on_off, "Min", {"Value": "64"})
        ET.SubElement(midi_cc_on_off, "Max", {"Value": "127"})

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
        ET.SubElement(mixer_elem, "Annotation", {"Value": ""})
        source_context_elem = ET.SubElement(mixer_elem, "SourceContext")
        ET.SubElement(source_context_elem, "Value")

        # <Sends>
        sends_elem = ET.SubElement(mixer_elem, "Sends")

        # TrackSendHolder Id=0 (Send A)
        # We'll define offsets for each automation / modulation target.
        # For example: offset=2,3 => 2 for automation, 3 for modulation.
        # Then offset=4,5 for the second send, etc.
        sendA_auto_id = self.base_automation_id + 2
        sendA_mod_id = self.base_automation_id + 3
        tshA = ET.SubElement(sends_elem, "TrackSendHolder", {"Id": "0"})
        sendA_elem = ET.SubElement(tshA, "Send")
        ET.SubElement(sendA_elem, "LomId", {"Value": "0"})
        ET.SubElement(sendA_elem, "Manual", {"Value": "0.0003162277571"})
        midi_range_A = ET.SubElement(sendA_elem, "MidiControllerRange")
        ET.SubElement(midi_range_A, "Min", {"Value": "0.0003162277571"})
        ET.SubElement(midi_range_A, "Max", {"Value": "1"})
        atA = ET.SubElement(sendA_elem, "AutomationTarget", {"Id": str(sendA_auto_id)})
        ET.SubElement(atA, "LockEnvelope", {"Value": "0"})
        mtA = ET.SubElement(sendA_elem, "ModulationTarget", {"Id": str(sendA_mod_id)})
        ET.SubElement(mtA, "LockEnvelope", {"Value": "0"})
        ET.SubElement(tshA, "Active", {"Value": "true"})

        # TrackSendHolder Id=1 (Send B)
        sendB_auto_id = self.base_automation_id + 4
        sendB_mod_id = self.base_automation_id + 5
        tshB = ET.SubElement(sends_elem, "TrackSendHolder", {"Id": "1"})
        sendB_elem = ET.SubElement(tshB, "Send")
        ET.SubElement(sendB_elem, "LomId", {"Value": "0"})
        ET.SubElement(sendB_elem, "Manual", {"Value": "0.0003162277571"})
        midi_range_B = ET.SubElement(sendB_elem, "MidiControllerRange")
        ET.SubElement(midi_range_B, "Min", {"Value": "0.0003162277571"})
        ET.SubElement(midi_range_B, "Max", {"Value": "1"})
        atB = ET.SubElement(sendB_elem, "AutomationTarget", {"Id": str(sendB_auto_id)})
        ET.SubElement(atB, "LockEnvelope", {"Value": "0"})
        mtB = ET.SubElement(sendB_elem, "ModulationTarget", {"Id": str(sendB_mod_id)})
        ET.SubElement(mtB, "LockEnvelope", {"Value": "0"})
        ET.SubElement(tshB, "Active", {"Value": "true"})

        # Speaker (offset=6)
        speaker_elem = ET.SubElement(mixer_elem, "Speaker")
        ET.SubElement(speaker_elem, "LomId", {"Value": "0"})
        ET.SubElement(speaker_elem, "Manual", {"Value": "true"})
        speaker_auto_id = self.base_automation_id + 6
        speaker_auto = ET.SubElement(speaker_elem, "AutomationTarget", {"Id": str(speaker_auto_id)})
        ET.SubElement(speaker_auto, "LockEnvelope", {"Value": "0"})
        sp_midi_cc = ET.SubElement(speaker_elem, "MidiCCOnOffThresholds")
        ET.SubElement(sp_midi_cc, "Min", {"Value": "64"})
        ET.SubElement(sp_midi_cc, "Max", {"Value": "127"})

        ET.SubElement(mixer_elem, "SoloSink", {"Value": "false"})
        ET.SubElement(mixer_elem, "PanMode", {"Value": "0"})

        # Pan (offset=7,8)
        pan_elem = ET.SubElement(mixer_elem, "Pan")
        ET.SubElement(pan_elem, "LomId", {"Value": "0"})
        ET.SubElement(pan_elem, "Manual", {"Value": "0"})
        midi_range_pan = ET.SubElement(pan_elem, "MidiControllerRange")
        ET.SubElement(midi_range_pan, "Min", {"Value": "-1"})
        ET.SubElement(midi_range_pan, "Max", {"Value": "1"})
        pan_auto_id = self.base_automation_id + 7
        pan_mod_id = self.base_automation_id + 8
        pan_auto = ET.SubElement(pan_elem, "AutomationTarget", {"Id": str(pan_auto_id)})
        ET.SubElement(pan_auto, "LockEnvelope", {"Value": "0"})
        pan_mod = ET.SubElement(pan_elem, "ModulationTarget", {"Id": str(pan_mod_id)})
        ET.SubElement(pan_mod, "LockEnvelope", {"Value": "0"})

        # SplitStereoPanL (offset=9,10)
        splitL_elem = ET.SubElement(mixer_elem, "SplitStereoPanL")
        ET.SubElement(splitL_elem, "LomId", {"Value": "0"})
        ET.SubElement(splitL_elem, "Manual", {"Value": "-1"})
        midi_range_L = ET.SubElement(splitL_elem, "MidiControllerRange")
        ET.SubElement(midi_range_L, "Min", {"Value": "-1"})
        ET.SubElement(midi_range_L, "Max", {"Value": "1"})
        splitL_auto_id = self.base_automation_id + 9
        splitL_mod_id = self.base_automation_id + 10
        splitL_auto = ET.SubElement(splitL_elem, "AutomationTarget", {"Id": str(splitL_auto_id)})
        ET.SubElement(splitL_auto, "LockEnvelope", {"Value": "0"})
        splitL_mod = ET.SubElement(splitL_elem, "ModulationTarget", {"Id": str(splitL_mod_id)})
        ET.SubElement(splitL_mod, "LockEnvelope", {"Value": "0"})

        # SplitStereoPanR (offset=11,12)
        splitR_elem = ET.SubElement(mixer_elem, "SplitStereoPanR")
        ET.SubElement(splitR_elem, "LomId", {"Value": "0"})
        ET.SubElement(splitR_elem, "Manual", {"Value": "1"})
        midi_range_R = ET.SubElement(splitR_elem, "MidiControllerRange")
        ET.SubElement(midi_range_R, "Min", {"Value": "-1"})
        ET.SubElement(midi_range_R, "Max", {"Value": "1"})
        splitR_auto_id = self.base_automation_id + 11
        splitR_mod_id = self.base_automation_id + 12
        splitR_auto = ET.SubElement(splitR_elem, "AutomationTarget", {"Id": str(splitR_auto_id)})
        ET.SubElement(splitR_auto, "LockEnvelope", {"Value": "0"})
        splitR_mod = ET.SubElement(splitR_elem, "ModulationTarget", {"Id": str(splitR_mod_id)})
        ET.SubElement(splitR_mod, "LockEnvelope", {"Value": "0"})

        # Volume (offset=13,14)
        volume_elem = ET.SubElement(mixer_elem, "Volume")
        ET.SubElement(volume_elem, "LomId", {"Value": "0"})
        ET.SubElement(volume_elem, "Manual", {"Value": "1"})
        midi_range_vol = ET.SubElement(volume_elem, "MidiControllerRange")
        ET.SubElement(midi_range_vol, "Min", {"Value": "0.0003162277571"})
        ET.SubElement(midi_range_vol, "Max", {"Value": "1.99526238"})
        volume_auto_id = self.base_automation_id + 13
        volume_mod_id = self.base_automation_id + 14
        vol_auto = ET.SubElement(volume_elem, "AutomationTarget", {"Id": str(volume_auto_id)})
        ET.SubElement(vol_auto, "LockEnvelope", {"Value": "0"})
        vol_mod = ET.SubElement(volume_elem, "ModulationTarget", {"Id": str(volume_mod_id)})
        ET.SubElement(vol_mod, "LockEnvelope", {"Value": "0"})

        ET.SubElement(mixer_elem, "ViewStateSesstionTrackWidth", {"Value": "93"})

        # CrossFadeState (offset=15)
        cross_elem = ET.SubElement(mixer_elem, "CrossFadeState")
        ET.SubElement(cross_elem, "LomId", {"Value": "0"})
        ET.SubElement(cross_elem, "Manual", {"Value": "1"})
        cross_auto_id = self.base_automation_id + 15
        cross_auto = ET.SubElement(cross_elem, "AutomationTarget", {"Id": str(cross_auto_id)})
        ET.SubElement(cross_auto, "LockEnvelope", {"Value": "0"})

        ET.SubElement(mixer_elem, "SendsListWrapper", {"LomId": "0"})

        # ---- MainSequencer ----
        main_seq_elem = ET.SubElement(device_chain_elem, "MainSequencer")
        ET.SubElement(main_seq_elem, "LomId", {"Value": "0"})
        ET.SubElement(main_seq_elem, "LomIdView", {"Value": "0"})
        ET.SubElement(main_seq_elem, "IsExpanded", {"Value": "true"})

        on2_elem = ET.SubElement(main_seq_elem, "On")
        ET.SubElement(on2_elem, "LomId", {"Value": "0"})
        ET.SubElement(on2_elem, "Manual", {"Value": "true"})

        # offset=16
        on2_auto_id = self.base_automation_id + 16
        at_on2 = ET.SubElement(on2_elem, "AutomationTarget", {"Id": str(on2_auto_id)})
        ET.SubElement(at_on2, "LockEnvelope", {"Value": "0"})

        midi_cc_on_off2 = ET.SubElement(on2_elem, "MidiCCOnOffThresholds")
        ET.SubElement(midi_cc_on_off2, "Min", {"Value": "64"})
        ET.SubElement(midi_cc_on_off2, "Max", {"Value": "127"})

        ET.SubElement(main_seq_elem, "ModulationSourceCount", {"Value": "0"})
        ET.SubElement(main_seq_elem, "ParametersListWrapper", {"LomId": "0"})

        # Use next available pointee ID for MainSequencer
        main_seq_pointee_id = self.get_next_pointee_id()
        ET.SubElement(main_seq_elem, "Pointee", {"Id": str(main_seq_pointee_id)})

        ET.SubElement(main_seq_elem, "LastSelectedTimeableIndex", {"Value": "0"})
        ET.SubElement(main_seq_elem, "LastSelectedClipEnvelopeIndex", {"Value": "0"})
        lsr = ET.SubElement(main_seq_elem, "LastPresetRef")
        ET.SubElement(lsr, "Value")
        ET.SubElement(main_seq_elem, "LockedScripts")
        ET.SubElement(main_seq_elem, "IsFolded", {"Value": "false"})
        ET.SubElement(main_seq_elem, "ShouldShowPresetName", {"Value": "true"})
        ET.SubElement(main_seq_elem, "UserName", {"Value": ""})
        ET.SubElement(main_seq_elem, "Annotation", {"Value": ""})
        sc = ET.SubElement(main_seq_elem, "SourceContext")
        ET.SubElement(sc, "Value")

        clip_slot_list_elem = ET.SubElement(main_seq_elem, "ClipSlotList")
        for i in range(8):
            slot = ET.SubElement(clip_slot_list_elem, "ClipSlot", {"Id": str(i)})
            ET.SubElement(slot, "LomId", {"Value": "0"})
            cslot = ET.SubElement(slot, "ClipSlot")
            ET.SubElement(cslot, "Value")
            ET.SubElement(slot, "HasStop", {"Value": "true"})
            ET.SubElement(slot, "NeedRefreeze", {"Value": "true"})

        ET.SubElement(main_seq_elem, "MonitoringEnum", {"Value": "1"})

        # Sample => ArrangerAutomation => AudioClip
        sample_elem = ET.SubElement(main_seq_elem, "Sample")
        arranger_automation_elem = ET.SubElement(sample_elem, "ArrangerAutomation")
        events_elem = ET.SubElement(arranger_automation_elem, "Events")

        # For the main AudioClip, we can set Id="0" or omit it. We'll keep "Id=0" for consistency.
        audio_clip_elem = ET.SubElement(events_elem, "AudioClip", {
            "Id": "0",
            "Time": str(self.clip_start),
        })
        ET.SubElement(audio_clip_elem, "LomId", {"Value": "0"})
        ET.SubElement(audio_clip_elem, "LomIdView", {"Value": "0"})
        ET.SubElement(audio_clip_elem, "CurrentStart", {"Value": str(self.clip_start)})
        ET.SubElement(audio_clip_elem, "CurrentEnd", {"Value": str(self.clip_end)})

        loop_elem = ET.SubElement(audio_clip_elem, "Loop")
        ET.SubElement(loop_elem, "LoopStart", {"Value": "0"})
        ET.SubElement(loop_elem, "LoopEnd", {"Value": str(self.clip_end)})
        ET.SubElement(loop_elem, "StartRelative", {"Value": "0"})
        ET.SubElement(loop_elem, "LoopOn", {"Value": "false"})
        ET.SubElement(loop_elem, "OutMarker", {"Value": "131.232"})
        ET.SubElement(loop_elem, "HiddenLoopStart", {"Value": "0"})
        ET.SubElement(loop_elem, "HiddenLoopEnd", {"Value": "131.232"})

        ET.SubElement(audio_clip_elem, "Name", {"Value": self.clip_name})
        ET.SubElement(audio_clip_elem, "Annotation", {"Value": ""})
        ET.SubElement(audio_clip_elem, "Color", {"Value": str(self.color)})
        ET.SubElement(audio_clip_elem, "LaunchMode", {"Value": "0"})
        ET.SubElement(audio_clip_elem, "LaunchQuantisation", {"Value": "0"})

        # TimeSignature
        time_sig_elem = ET.SubElement(audio_clip_elem, "TimeSignature")
        rts_elem = ET.SubElement(time_sig_elem, "TimeSignatures")
        rts0 = ET.SubElement(rts_elem, "RemoteableTimeSignature", {"Id": "0"})
        ET.SubElement(rts0, "Numerator", {"Value": "4"})
        ET.SubElement(rts0, "Denominator", {"Value": "4"})
        ET.SubElement(rts0, "Time", {"Value": "0"})

        # Envelopes
        envs = ET.SubElement(audio_clip_elem, "Envelopes")
        envs.append(ET.Element("Envelopes"))

        scroller_elem = ET.SubElement(audio_clip_elem, "ScrollerTimePreserver")
        ET.SubElement(scroller_elem, "LeftTime", {"Value": "0"})
        ET.SubElement(scroller_elem, "RightTime", {"Value": "131.232"})

        time_sel_elem = ET.SubElement(audio_clip_elem, "TimeSelection")
        ET.SubElement(time_sel_elem, "AnchorTime", {"Value": "0"})
        ET.SubElement(time_sel_elem, "OtherTime", {"Value": "0"})

        ET.SubElement(audio_clip_elem, "Legato", {"Value": "false"})
        ET.SubElement(audio_clip_elem, "Ram", {"Value": "false"})

        groove_elem = ET.SubElement(audio_clip_elem, "GrooveSettings")
        ET.SubElement(groove_elem, "GrooveId", {"Value": "-1"})

        ET.SubElement(audio_clip_elem, "Disabled", {"Value": "false"})
        ET.SubElement(audio_clip_elem, "VelocityAmount", {"Value": "0"})

        # FollowAction
        follow_elem = ET.SubElement(audio_clip_elem, "FollowAction")
        ET.SubElement(follow_elem, "FollowTime", {"Value": "4"})
        ET.SubElement(follow_elem, "IsLinked", {"Value": "true"})
        ET.SubElement(follow_elem, "LoopIterations", {"Value": "1"})
        ET.SubElement(follow_elem, "FollowActionA", {"Value": "4"})
        ET.SubElement(follow_elem, "FollowActionB", {"Value": "0"})
        ET.SubElement(follow_elem, "FollowChanceA", {"Value": "100"})
        ET.SubElement(follow_elem, "FollowChanceB", {"Value": "0"})
        ET.SubElement(follow_elem, "JumpIndexA", {"Value": "1"})
        ET.SubElement(follow_elem, "JumpIndexB", {"Value": "1"})
        ET.SubElement(follow_elem, "FollowActionEnabled", {"Value": "false"})

        # Grid
        grid_elem = ET.SubElement(audio_clip_elem, "Grid")
        ET.SubElement(grid_elem, "FixedNumerator", {"Value": "1"})
        ET.SubElement(grid_elem, "FixedDenominator", {"Value": "16"})
        ET.SubElement(grid_elem, "GridIntervalPixel", {"Value": "20"})
        ET.SubElement(grid_elem, "Ntoles", {"Value": "2"})
        ET.SubElement(grid_elem, "SnapToGrid", {"Value": "true"})
        ET.SubElement(grid_elem, "Fixed", {"Value": "false"})

        ET.SubElement(audio_clip_elem, "FreezeStart", {"Value": "0"})
        ET.SubElement(audio_clip_elem, "FreezeEnd", {"Value": "0"})
        ET.SubElement(audio_clip_elem, "IsWarped", {"Value": "false" if not self.is_warped else "true"})
        ET.SubElement(audio_clip_elem, "TakeId", {"Value": "1"})

        # SampleRef/FileRef
        sample_ref_elem = ET.SubElement(audio_clip_elem, "SampleRef")
        file_ref_elem = ET.SubElement(sample_ref_elem, "FileRef")
        ET.SubElement(file_ref_elem, "RelativePathType", {"Value": "3"})
        ET.SubElement(file_ref_elem, "RelativePath", {"Value": self.relative_path})
        ET.SubElement(file_ref_elem, "Path", {"Value": self.absolute_path})
        ET.SubElement(file_ref_elem, "Type", {"Value": "1"})
        ET.SubElement(file_ref_elem, "LivePackName", {"Value": ""})
        ET.SubElement(file_ref_elem, "LivePackId", {"Value": ""})
        ET.SubElement(file_ref_elem, "OriginalFileSize", {"Value": str(self.original_file_size)})
        ET.SubElement(file_ref_elem, "OriginalCrc", {"Value": str(self.original_crc)})

        ET.SubElement(sample_ref_elem, "LastModDate", {"Value": "1739070310"})
        ET.SubElement(sample_ref_elem, "SourceContext")
        ET.SubElement(sample_ref_elem, "SampleUsageHint", {"Value": "0"})
        ET.SubElement(sample_ref_elem, "DefaultDuration", {"Value": str(self.default_duration)})
        ET.SubElement(sample_ref_elem, "DefaultSampleRate", {"Value": str(self.default_sample_rate)})

        onsets_elem = ET.SubElement(audio_clip_elem, "Onsets")
        ET.SubElement(onsets_elem, "UserOnsets")
        ET.SubElement(onsets_elem, "HasUserOnsets", {"Value": "false"})

        ET.SubElement(audio_clip_elem, "WarpMode", {"Value": "6"})
        ET.SubElement(audio_clip_elem, "GranularityTones", {"Value": "30"})
        ET.SubElement(audio_clip_elem, "GranularityTexture", {"Value": "65"})
        ET.SubElement(audio_clip_elem, "FluctuationTexture", {"Value": "25"})
        ET.SubElement(audio_clip_elem, "TransientResolution", {"Value": "6"})
        ET.SubElement(audio_clip_elem, "TransientLoopMode", {"Value": "2"})
        ET.SubElement(audio_clip_elem, "TransientEnvelope", {"Value": "100"})
        ET.SubElement(audio_clip_elem, "ComplexProFormants", {"Value": "100"})
        ET.SubElement(audio_clip_elem, "ComplexProEnvelope", {"Value": "128"})
        ET.SubElement(audio_clip_elem, "Sync", {"Value": "true"})
        ET.SubElement(audio_clip_elem, "HiQ", {"Value": "true"})
        ET.SubElement(audio_clip_elem, "Fade", {"Value": "true"})

        # Fades
        fades_elem = ET.SubElement(audio_clip_elem, "Fades")
        ET.SubElement(fades_elem, "FadeInLength", {"Value": "0"})
        ET.SubElement(fades_elem, "FadeOutLength", {"Value": "0"})
        ET.SubElement(fades_elem, "ClipFadesAreInitialized", {"Value": "true"})
        ET.SubElement(fades_elem, "CrossfadeInState", {"Value": "0"})
        ET.SubElement(fades_elem, "FadeInCurveSkew", {"Value": "0"})
        ET.SubElement(fades_elem, "FadeInCurveSlope", {"Value": "0"})
        ET.SubElement(fades_elem, "FadeOutCurveSkew", {"Value": "0"})
        ET.SubElement(fades_elem, "FadeOutCurveSlope", {"Value": "0"})
        ET.SubElement(fades_elem, "IsDefaultFadeIn", {"Value": "true"})
        ET.SubElement(fades_elem, "IsDefaultFadeOut", {"Value": "true"})

        ET.SubElement(audio_clip_elem, "PitchCoarse", {"Value": str(self.pitch_shift)})
        ET.SubElement(audio_clip_elem, "PitchFine", {"Value": "0"})
        ET.SubElement(audio_clip_elem, "SampleVolume", {"Value": "1"})

        # WarpMarkers
        warp_markers_elem = ET.SubElement(audio_clip_elem, "WarpMarkers")
        warp_marker_id1 = self.get_next_pointee_id()
        warp_marker_id2 = self.get_next_pointee_id()
        ET.SubElement(warp_markers_elem, "WarpMarker", {"SecTime": "0", "BeatTime": "0", "Id": str(warp_marker_id1)})
        ET.SubElement(warp_markers_elem, "WarpMarker",
                      {"SecTime": "0.015625", "BeatTime": "0.03125", "Id": str(warp_marker_id2)})

        ET.SubElement(audio_clip_elem, "SavedWarpMarkersForStretched")
        ET.SubElement(audio_clip_elem, "MarkersGenerated", {"Value": "true"})
        ET.SubElement(audio_clip_elem, "IsSongTempoMaster", {"Value": "false"})

        # ArrangerAutomation transform state
        atvs = ET.SubElement(arranger_automation_elem, "AutomationTransformViewState")
        ET.SubElement(atvs, "IsTransformPending", {"Value": "false"})
        ET.SubElement(atvs, "TimeAndValueTransforms")

        # ---- FreezeSequencer ----
        freeze_seq_elem = ET.SubElement(device_chain_elem, "FreezeSequencer")
        ET.SubElement(freeze_seq_elem, "LomId", {"Value": "0"})
        ET.SubElement(freeze_seq_elem, "LomIdView", {"Value": "0"})
        ET.SubElement(freeze_seq_elem, "IsExpanded", {"Value": "true"})
        on3_elem = ET.SubElement(freeze_seq_elem, "On")
        ET.SubElement(on3_elem, "LomId", {"Value": "0"})
        ET.SubElement(on3_elem, "Manual", {"Value": "true"})
        ET.SubElement(on3_elem, "MidiCCOnOffThresholds")

        ET.SubElement(freeze_seq_elem, "ModulationSourceCount", {"Value": "0"})
        ET.SubElement(freeze_seq_elem, "ParametersListWrapper", {"LomId": "0"})
        freeze_seq_pointee_id = self.get_next_pointee_id()
        ET.SubElement(freeze_seq_elem, "Pointee", {"Id": str(freeze_seq_pointee_id)})
        ET.SubElement(freeze_seq_elem, "LastSelectedTimeableIndex", {"Value": "0"})
        ET.SubElement(freeze_seq_elem, "LastSelectedClipEnvelopeIndex", {"Value": "0"})
        lsr2 = ET.SubElement(freeze_seq_elem, "LastPresetRef")
        ET.SubElement(lsr2, "Value")
        ET.SubElement(freeze_seq_elem, "LockedScripts")
        ET.SubElement(freeze_seq_elem, "IsFolded", {"Value": "false"})
        ET.SubElement(freeze_seq_elem, "ShouldShowPresetName", {"Value": "true"})
        ET.SubElement(freeze_seq_elem, "UserName", {"Value": ""})
        ET.SubElement(freeze_seq_elem, "Annotation", {"Value": ""})
        sc2 = ET.SubElement(freeze_seq_elem, "SourceContext")
        ET.SubElement(sc2, "Value")

        fslot_list = ET.SubElement(freeze_seq_elem, "ClipSlotList")
        for i in range(8):
            slot = ET.SubElement(fslot_list, "ClipSlot", {"Id": str(i)})
            ET.SubElement(slot, "LomId", {"Value": "0"})
            ET.SubElement(slot, "ClipSlot").append(ET.Element("Value"))
            ET.SubElement(slot, "HasStop", {"Value": "true"})
            ET.SubElement(slot, "NeedRefreeze", {"Value": "true"})

        ET.SubElement(freeze_seq_elem, "MonitoringEnum", {"Value": "1"})
        sample2_elem = ET.SubElement(freeze_seq_elem, "Sample")
        arr_automation2 = ET.SubElement(sample2_elem, "ArrangerAutomation")
        ET.SubElement(arr_automation2, "Events")
        atvs2 = ET.SubElement(arr_automation2, "AutomationTransformViewState")
        ET.SubElement(atvs2, "IsTransformPending", {"Value": "false"})
        ET.SubElement(atvs2, "TimeAndValueTransforms")

        # offset=18..22 for these final targets
        vol_mod_id2 = self.base_automation_id + 18
        trans_mod_id = self.base_automation_id + 19
        grain_mod_id = self.base_automation_id + 20
        flux_mod_id = self.base_automation_id + 21
        sampoffs_mod_id = self.base_automation_id + 22

        ET.SubElement(freeze_seq_elem, "VolumeModulationTarget", {"Id": str(vol_mod_id2)}).append(
            ET.Element("LockEnvelope", {"Value": "0"}))
        ET.SubElement(freeze_seq_elem, "TranspositionModulationTarget", {"Id": str(trans_mod_id)}).append(
            ET.Element("LockEnvelope", {"Value": "0"}))
        ET.SubElement(freeze_seq_elem, "GrainSizeModulationTarget", {"Id": str(grain_mod_id)}).append(
            ET.Element("LockEnvelope", {"Value": "0"}))
        ET.SubElement(freeze_seq_elem, "FluxModulationTarget", {"Id": str(flux_mod_id)}).append(
            ET.Element("LockEnvelope", {"Value": "0"}))
        ET.SubElement(freeze_seq_elem, "SampleOffsetModulationTarget", {"Id": str(sampoffs_mod_id)}).append(
            ET.Element("LockEnvelope", {"Value": "0"}))

        ET.SubElement(freeze_seq_elem, "PitchViewScrollPosition", {"Value": "-1073741824"})
        ET.SubElement(freeze_seq_elem, "SampleOffsetModulationScrollPosition", {"Value": "-1073741824"})
        recorder2_elem = ET.SubElement(freeze_seq_elem, "Recorder")
        ET.SubElement(recorder2_elem, "IsArmed", {"Value": "false"})
        ET.SubElement(recorder2_elem, "TakeCounter", {"Value": "1"})

        # A second <DeviceChain> under FreezeSequencer
        device_chain2_elem = ET.SubElement(device_chain_elem, "DeviceChain")
        ET.SubElement(device_chain2_elem, "Devices")
        ET.SubElement(device_chain2_elem, "SignalModulations")

        return audio_track_elem


from typing import Union, Tuple
import numpy as np
import torch
import torchaudio.functional as AF
from pydub import AudioSegment


def shift_pitch(audio: Union[AudioSegment, Tuple[np.ndarray, int]], pitch_shift: int) -> Union[
    AudioSegment, Tuple[np.ndarray, int]]:
    """
    Pitch shift audio by a given number of semitones WITHOUT altering its speed,
    using torchaudio.functional.pitch_shift.

    If the input is an AudioSegment, the output will be an AudioSegment.
    If the input is a tuple (ndarray, sample_rate), the output will be a tuple (ndarray, sample_rate).

    Requirements:
     - torchaudio >= 2.1 (which includes pitch_shift)
     - PyTorch installed (version matching torchaudio)
    """
    if pitch_shift == 0:
        return audio

    # Branch based on input type
    if isinstance(audio, AudioSegment):
        # === AudioSegment branch ===
        sample_rate = audio.frame_rate
        channels = audio.channels
        sample_width = audio.sample_width

        # Convert AudioSegment to float32 NumPy array
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        max_val = float(1 << (8 * sample_width - 1))
        samples = samples / max_val

        # Reshape to [channels, num_frames] for torchaudio
        if channels > 1:
            samples = samples.reshape(-1, channels).T  # shape: (channels, n_frames)
        else:
            samples = np.expand_dims(samples, axis=0)  # shape: (1, n_frames)

        waveform = torch.from_numpy(samples)

        # Pitch shift with torchaudio.functional.pitch_shift
        pitched_wf = AF.pitch_shift(
            waveform,
            sample_rate=sample_rate,
            n_steps=pitch_shift,
            bins_per_octave=12
        )

        # Convert back to NumPy and transpose to (n_frames, channels)
        pitched_np = pitched_wf.numpy().transpose(1, 0)
        pitched_np = pitched_np * max_val

        # Determine dtype based on sample_width
        if sample_width == 1:
            dtype = np.int8
        elif sample_width == 2:
            dtype = np.int16
        elif sample_width == 4:
            dtype = np.int32
        else:
            dtype = np.int16  # fallback

        min_val, max_allow = -max_val, max_val - 1
        pitched_np = np.clip(pitched_np, min_val, max_allow).astype(dtype)

        # Flatten for pydub’s raw byte layout and spawn a new AudioSegment
        pitched_flat = pitched_np.flatten()
        return audio._spawn(pitched_flat.tobytes())

    elif isinstance(audio, tuple) and len(audio) == 2:
        # === Tuple branch: (ndarray, sample_rate) ===
        samples, sample_rate = audio

        # Determine audio shape: assume 1D array is mono, 2D is (n_frames, channels)
        if samples.ndim == 1:
            channels = 1
            waveform_np = np.expand_dims(samples, axis=0)  # shape: (1, n_frames)
        elif samples.ndim == 2:
            channels = samples.shape[1]
            waveform_np = samples.T  # shape: (channels, n_frames)
        else:
            raise ValueError("Input ndarray must be 1D or 2D.")

        # Check if data is integer type; if so, normalize it
        if np.issubdtype(samples.dtype, np.integer):
            sample_width = samples.dtype.itemsize
            max_val = float(1 << (8 * sample_width - 1))
            waveform_np = waveform_np.astype(np.float32) / max_val
            is_integer = True
        else:
            # Assume float data already in [-1, 1]
            max_val = 1.0
            sample_width = 4
            is_integer = False

        waveform = torch.from_numpy(waveform_np)

        # Pitch shift
        pitched_wf = AF.pitch_shift(
            waveform,
            sample_rate=sample_rate,
            n_steps=pitch_shift,
            bins_per_octave=12
        )

        # Convert back to NumPy and transpose to (n_frames, channels)
        pitched_np = pitched_wf.numpy().transpose(1, 0)

        if is_integer:
            pitched_np = pitched_np * max_val
            if sample_width == 1:
                dtype = np.int8
            elif sample_width == 2:
                dtype = np.int16
            elif sample_width == 4:
                dtype = np.int32
            else:
                dtype = np.int16
            min_val, max_allow = -max_val, max_val - 1
            pitched_np = np.clip(pitched_np, min_val, max_allow).astype(dtype)

        # If original was mono (1D), return 1D array
        if samples.ndim == 1:
            pitched_np = pitched_np.flatten()

        return pitched_np, sample_rate

    else:
        raise TypeError("Input audio must be either an AudioSegment or a tuple of (ndarray, sample_rate).")
