import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple


class MidiTrack:
    def __init__(
        self,
        track_id: int,
        next_pointee_id: int,
        effective_name: str,
        clip_name: str,
        color: int,
        clip_start: float,
        clip_end: float,
        notes_by_key: Dict[int, List[Tuple[float, float, float]]],
    ):
        self.track_id = track_id
        self.next_pointee_id = next_pointee_id
        self.effective_name = effective_name
        self.clip_name = clip_name
        self.color = color
        self.clip_start = clip_start
        self.clip_end = clip_end
        self.notes_by_key = notes_by_key
        self.base_automation_id = track_id * 100

    def get_next_pointee_id(self) -> int:
        current_id = self.next_pointee_id
        self.next_pointee_id += 1
        return current_id

    def to_element(self) -> ET.Element:
        midi_track_elem = ET.Element(
            "MidiTrack",
            {
                "Id": str(self.track_id),
                "SelectedToolPanel": "2",
                "SelectedTransformationName": "Arpeggiate",
                "SelectedGeneratorName": "Rhythm",
            },
        )

        ET.SubElement(midi_track_elem, "LomId", {"Value": "0"})
        ET.SubElement(midi_track_elem, "LomIdView", {"Value": "0"})
        ET.SubElement(midi_track_elem, "IsContentSelectedInDocument", {"Value": "false"})
        ET.SubElement(midi_track_elem, "PreferredContentViewMode", {"Value": "0"})

        track_delay_elem = ET.SubElement(midi_track_elem, "TrackDelay")
        ET.SubElement(track_delay_elem, "Value", {"Value": "0"})
        ET.SubElement(track_delay_elem, "IsValueSampleBased", {"Value": "false"})

        name_elem = ET.SubElement(midi_track_elem, "Name")
        ET.SubElement(name_elem, "EffectiveName", {"Value": self.effective_name})
        ET.SubElement(name_elem, "UserName", {"Value": ""})
        ET.SubElement(name_elem, "Annotation", {"Value": ""})
        ET.SubElement(name_elem, "MemorizedFirstClipName", {"Value": ""})

        ET.SubElement(midi_track_elem, "Color", {"Value": str(self.color)})
        auto_env_elem = ET.SubElement(midi_track_elem, "AutomationEnvelopes")
        ET.SubElement(auto_env_elem, "Envelopes")

        ET.SubElement(midi_track_elem, "TrackGroupId", {"Value": "-1"})
        ET.SubElement(midi_track_elem, "TrackUnfolded", {"Value": "true"})
        ET.SubElement(midi_track_elem, "DevicesListWrapper", {"LomId": "0"})
        ET.SubElement(midi_track_elem, "ClipSlotsListWrapper", {"LomId": "0"})
        ET.SubElement(midi_track_elem, "ArrangementClipsListWrapper", {"LomId": "0"})
        ET.SubElement(midi_track_elem, "ViewData", {"Value": "{}"})

        take_lanes_elem = ET.SubElement(midi_track_elem, "TakeLanes")
        ET.SubElement(take_lanes_elem, "TakeLanes")
        ET.SubElement(take_lanes_elem, "AreTakeLanesFolded", {"Value": "true"})

        ET.SubElement(midi_track_elem, "LinkedTrackGroupId", {"Value": "-1"})
        ET.SubElement(midi_track_elem, "SavedPlayingSlot", {"Value": "-1"})
        ET.SubElement(midi_track_elem, "SavedPlayingOffset", {"Value": "0"})
        ET.SubElement(midi_track_elem, "Freeze", {"Value": "false"})
        ET.SubElement(midi_track_elem, "NeedArrangerRefreeze", {"Value": "true"})
        ET.SubElement(midi_track_elem, "PostProcessFreezeClips", {"Value": "0"})

        device_chain_elem = ET.SubElement(midi_track_elem, "DeviceChain")

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

        audio_in_elem = ET.SubElement(device_chain_elem, "AudioInputRouting")
        ET.SubElement(audio_in_elem, "Target", {"Value": "AudioIn/External/S0"})
        ET.SubElement(audio_in_elem, "UpperDisplayString", {"Value": "Ext. In"})
        ET.SubElement(audio_in_elem, "LowerDisplayString", {"Value": "1/2"})
        mpe_audio_in = ET.SubElement(audio_in_elem, "MpeSettings")
        ET.SubElement(mpe_audio_in, "ZoneType", {"Value": "0"})
        ET.SubElement(mpe_audio_in, "FirstNoteChannel", {"Value": "1"})
        ET.SubElement(mpe_audio_in, "LastNoteChannel", {"Value": "15"})
        ET.SubElement(audio_in_elem, "MpePitchBendUsesTuning", {"Value": "true"})

        midi_in_elem = ET.SubElement(device_chain_elem, "MidiInputRouting")
        ET.SubElement(midi_in_elem, "Target", {"Value": "MidiIn/External.All/-1"})
        ET.SubElement(midi_in_elem, "UpperDisplayString", {"Value": "Ext: All Ins"})
        ET.SubElement(midi_in_elem, "LowerDisplayString", {"Value": ""})
        mpe_midi_in = ET.SubElement(midi_in_elem, "MpeSettings")
        ET.SubElement(mpe_midi_in, "ZoneType", {"Value": "0"})
        ET.SubElement(mpe_midi_in, "FirstNoteChannel", {"Value": "1"})
        ET.SubElement(mpe_midi_in, "LastNoteChannel", {"Value": "15"})
        ET.SubElement(midi_in_elem, "MpePitchBendUsesTuning", {"Value": "true"})

        audio_out_elem = ET.SubElement(device_chain_elem, "AudioOutputRouting")
        ET.SubElement(audio_out_elem, "Target", {"Value": "AudioOut/Main"})
        ET.SubElement(audio_out_elem, "UpperDisplayString", {"Value": "Main"})
        ET.SubElement(audio_out_elem, "LowerDisplayString", {"Value": ""})
        mpe_audio_out = ET.SubElement(audio_out_elem, "MpeSettings")
        ET.SubElement(mpe_audio_out, "ZoneType", {"Value": "0"})
        ET.SubElement(mpe_audio_out, "FirstNoteChannel", {"Value": "1"})
        ET.SubElement(mpe_audio_out, "LastNoteChannel", {"Value": "15"})
        ET.SubElement(audio_out_elem, "MpePitchBendUsesTuning", {"Value": "true"})

        midi_out_elem = ET.SubElement(device_chain_elem, "MidiOutputRouting")
        ET.SubElement(midi_out_elem, "Target", {"Value": "MidiOut/None"})
        ET.SubElement(midi_out_elem, "UpperDisplayString", {"Value": "None"})
        ET.SubElement(midi_out_elem, "LowerDisplayString", {"Value": ""})
        mpe_midi_out = ET.SubElement(midi_out_elem, "MpeSettings")
        ET.SubElement(mpe_midi_out, "ZoneType", {"Value": "0"})
        ET.SubElement(mpe_midi_out, "FirstNoteChannel", {"Value": "1"})
        ET.SubElement(mpe_midi_out, "LastNoteChannel", {"Value": "15"})
        ET.SubElement(midi_out_elem, "MpePitchBendUsesTuning", {"Value": "true"})

        mixer_elem = ET.SubElement(device_chain_elem, "Mixer")
        ET.SubElement(mixer_elem, "LomId", {"Value": "0"})
        ET.SubElement(mixer_elem, "LomIdView", {"Value": "0"})
        ET.SubElement(mixer_elem, "IsExpanded", {"Value": "true"})
        ET.SubElement(mixer_elem, "BreakoutIsExpanded", {"Value": "false"})

        on_elem = ET.SubElement(mixer_elem, "On")
        ET.SubElement(on_elem, "LomId", {"Value": "0"})
        ET.SubElement(on_elem, "Manual", {"Value": "true"})
        on_auto_target_id = self.base_automation_id + 0
        at_on = ET.SubElement(on_elem, "AutomationTarget", {"Id": str(on_auto_target_id)})
        ET.SubElement(at_on, "LockEnvelope", {"Value": "0"})
        midi_cc_on_off = ET.SubElement(on_elem, "MidiCCOnOffThresholds")
        ET.SubElement(midi_cc_on_off, "Min", {"Value": "64"})
        ET.SubElement(midi_cc_on_off, "Max", {"Value": "127"})

        ET.SubElement(mixer_elem, "ModulationSourceCount", {"Value": "0"})
        ET.SubElement(mixer_elem, "ParametersListWrapper", {"LomId": "0"})
        ET.SubElement(mixer_elem, "Pointee", {"Id": str(self.get_next_pointee_id())})
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

        sends_elem = ET.SubElement(mixer_elem, "Sends")
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

        ET.SubElement(mixer_elem, "ViewStateSessionTrackWidth", {"Value": "93"})

        cross_elem = ET.SubElement(mixer_elem, "CrossFadeState")
        ET.SubElement(cross_elem, "LomId", {"Value": "0"})
        ET.SubElement(cross_elem, "Manual", {"Value": "1"})
        cross_auto_id = self.base_automation_id + 15
        cross_auto = ET.SubElement(cross_elem, "AutomationTarget", {"Id": str(cross_auto_id)})
        ET.SubElement(cross_auto, "LockEnvelope", {"Value": "0"})

        ET.SubElement(mixer_elem, "SendsListWrapper", {"LomId": "0"})

        main_seq_elem = ET.SubElement(device_chain_elem, "MainSequencer")
        ET.SubElement(main_seq_elem, "LomId", {"Value": "0"})
        ET.SubElement(main_seq_elem, "LomIdView", {"Value": "0"})
        ET.SubElement(main_seq_elem, "IsExpanded", {"Value": "true"})
        ET.SubElement(main_seq_elem, "BreakoutIsExpanded", {"Value": "false"})

        on2_elem = ET.SubElement(main_seq_elem, "On")
        ET.SubElement(on2_elem, "LomId", {"Value": "0"})
        ET.SubElement(on2_elem, "Manual", {"Value": "true"})
        on2_auto_id = self.base_automation_id + 16
        at_on2 = ET.SubElement(on2_elem, "AutomationTarget", {"Id": str(on2_auto_id)})
        ET.SubElement(at_on2, "LockEnvelope", {"Value": "0"})
        midi_cc_on_off2 = ET.SubElement(on2_elem, "MidiCCOnOffThresholds")
        ET.SubElement(midi_cc_on_off2, "Min", {"Value": "64"})
        ET.SubElement(midi_cc_on_off2, "Max", {"Value": "127"})

        ET.SubElement(main_seq_elem, "ModulationSourceCount", {"Value": "0"})
        ET.SubElement(main_seq_elem, "ParametersListWrapper", {"LomId": "0"})
        ET.SubElement(main_seq_elem, "Pointee", {"Id": str(self.get_next_pointee_id())})
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
        ET.SubElement(main_seq_elem, "MpePitchBendUsesTuning", {"Value": "true"})

        clip_slot_list_elem = ET.SubElement(main_seq_elem, "ClipSlotList")
        for i in range(8):
            slot = ET.SubElement(clip_slot_list_elem, "ClipSlot", {"Id": str(i)})
            ET.SubElement(slot, "LomId", {"Value": "0"})
            cslot = ET.SubElement(slot, "ClipSlot")
            ET.SubElement(cslot, "Value")
            ET.SubElement(slot, "HasStop", {"Value": "true"})
            ET.SubElement(slot, "NeedRefreeze", {"Value": "true"})

        ET.SubElement(main_seq_elem, "MonitoringEnum", {"Value": "1"})
        ET.SubElement(main_seq_elem, "KeepRecordMonitoringLatency", {"Value": "true"})

        clip_timeable = ET.SubElement(main_seq_elem, "ClipTimeable")
        arranger_automation_elem = ET.SubElement(clip_timeable, "ArrangerAutomation")
        events_elem = ET.SubElement(arranger_automation_elem, "Events")

        midi_clip_elem = ET.SubElement(events_elem, "MidiClip", {"Id": "1", "Time": str(self.clip_start)})
        ET.SubElement(midi_clip_elem, "LomId", {"Value": "0"})
        ET.SubElement(midi_clip_elem, "LomIdView", {"Value": "0"})
        ET.SubElement(midi_clip_elem, "CurrentStart", {"Value": str(self.clip_start)})
        ET.SubElement(midi_clip_elem, "CurrentEnd", {"Value": str(self.clip_end)})

        loop_elem = ET.SubElement(midi_clip_elem, "Loop")
        ET.SubElement(loop_elem, "LoopStart", {"Value": "0"})
        ET.SubElement(loop_elem, "LoopEnd", {"Value": str(self.clip_end - self.clip_start)})
        ET.SubElement(loop_elem, "StartRelative", {"Value": "0"})
        ET.SubElement(loop_elem, "LoopOn", {"Value": "true"})
        ET.SubElement(loop_elem, "OutMarker", {"Value": str(self.clip_end - self.clip_start)})
        ET.SubElement(loop_elem, "HiddenLoopStart", {"Value": "0"})
        ET.SubElement(loop_elem, "HiddenLoopEnd", {"Value": str(self.clip_end - self.clip_start)})

        ET.SubElement(midi_clip_elem, "Name", {"Value": self.clip_name})
        ET.SubElement(midi_clip_elem, "Annotation", {"Value": ""})
        ET.SubElement(midi_clip_elem, "Color", {"Value": str(self.color)})
        ET.SubElement(midi_clip_elem, "LaunchMode", {"Value": "0"})
        ET.SubElement(midi_clip_elem, "LaunchQuantisation", {"Value": "0"})

        time_sig_elem = ET.SubElement(midi_clip_elem, "TimeSignature")
        rts_elem = ET.SubElement(time_sig_elem, "TimeSignatures")
        rts0 = ET.SubElement(rts_elem, "RemoteableTimeSignature", {"Id": "0"})
        ET.SubElement(rts0, "Numerator", {"Value": "4"})
        ET.SubElement(rts0, "Denominator", {"Value": "4"})
        ET.SubElement(rts0, "Time", {"Value": "0"})

        envs = ET.SubElement(midi_clip_elem, "Envelopes")
        envs.append(ET.Element("Envelopes"))

        scroller_elem = ET.SubElement(midi_clip_elem, "ScrollerTimePreserver")
        ET.SubElement(scroller_elem, "LeftTime", {"Value": str(self.clip_start)})
        ET.SubElement(scroller_elem, "RightTime", {"Value": str(self.clip_start + 8)})

        time_sel_elem = ET.SubElement(midi_clip_elem, "TimeSelection")
        ET.SubElement(time_sel_elem, "AnchorTime", {"Value": "0"})
        ET.SubElement(time_sel_elem, "OtherTime", {"Value": "0"})

        ET.SubElement(midi_clip_elem, "Legato", {"Value": "false"})
        ET.SubElement(midi_clip_elem, "Ram", {"Value": "false"})

        groove_elem = ET.SubElement(midi_clip_elem, "GrooveSettings")
        ET.SubElement(groove_elem, "GrooveId", {"Value": "-1"})

        ET.SubElement(midi_clip_elem, "Disabled", {"Value": "false"})
        ET.SubElement(midi_clip_elem, "VelocityAmount", {"Value": "0"})

        follow_elem = ET.SubElement(midi_clip_elem, "FollowAction")
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

        grid_elem = ET.SubElement(midi_clip_elem, "Grid")
        ET.SubElement(grid_elem, "FixedNumerator", {"Value": "1"})
        ET.SubElement(grid_elem, "FixedDenominator", {"Value": "16"})
        ET.SubElement(grid_elem, "GridIntervalPixel", {"Value": "20"})
        ET.SubElement(grid_elem, "Ntoles", {"Value": "2"})
        ET.SubElement(grid_elem, "SnapToGrid", {"Value": "true"})
        ET.SubElement(grid_elem, "Fixed", {"Value": "false"})

        ET.SubElement(midi_clip_elem, "FreezeStart", {"Value": "0"})
        ET.SubElement(midi_clip_elem, "FreezeEnd", {"Value": "0"})
        ET.SubElement(midi_clip_elem, "IsWarped", {"Value": "true"})
        ET.SubElement(midi_clip_elem, "TakeId", {"Value": "1"})
        ET.SubElement(midi_clip_elem, "IsInKey", {"Value": "true"})
        scale_info = ET.SubElement(midi_clip_elem, "ScaleInformation")
        ET.SubElement(scale_info, "Root", {"Value": "0"})
        ET.SubElement(scale_info, "Name", {"Value": "0"})

        notes_elem = ET.SubElement(midi_clip_elem, "Notes")
        key_tracks_elem = ET.SubElement(notes_elem, "KeyTracks")

        note_id = 1
        for key_idx, midi_key in enumerate(sorted(self.notes_by_key.keys())):
            key_track = ET.SubElement(key_tracks_elem, "KeyTrack", {"Id": str(key_idx)})
            notes_list = ET.SubElement(key_track, "Notes")
            for start, duration, velocity in self.notes_by_key[midi_key]:
                ET.SubElement(
                    notes_list,
                    "MidiNoteEvent",
                    {
                        "Time": str(start),
                        "Duration": str(duration),
                        "Velocity": str(velocity),
                        "VelocityDeviation": "0",
                        "OffVelocity": "64",
                        "Probability": "1",
                        "IsEnabled": "true",
                        "NoteId": str(note_id),
                    },
                )
                note_id += 1
            ET.SubElement(key_track, "MidiKey", {"Value": str(midi_key)})

        per_note_store = ET.SubElement(notes_elem, "PerNoteEventStore")
        ET.SubElement(per_note_store, "EventLists")
        ET.SubElement(notes_elem, "NoteProbabilityGroups")
        group_id_gen = ET.SubElement(notes_elem, "ProbabilityGroupIdGenerator")
        ET.SubElement(group_id_gen, "NextId", {"Value": "1"})
        note_id_gen = ET.SubElement(notes_elem, "NoteIdGenerator")
        ET.SubElement(note_id_gen, "NextId", {"Value": str(note_id)})

        ET.SubElement(midi_clip_elem, "BankSelectCoarse", {"Value": "-1"})
        ET.SubElement(midi_clip_elem, "BankSelectFine", {"Value": "-1"})
        ET.SubElement(midi_clip_elem, "ProgramChange", {"Value": "-1"})
        ET.SubElement(midi_clip_elem, "NoteEditorFoldInZoom", {"Value": "-1"})
        ET.SubElement(midi_clip_elem, "NoteEditorFoldInScroll", {"Value": "0"})
        ET.SubElement(midi_clip_elem, "NoteEditorFoldOutZoom", {"Value": "289"})
        ET.SubElement(midi_clip_elem, "NoteEditorFoldOutScroll", {"Value": "-97"})
        ET.SubElement(midi_clip_elem, "NoteEditorFoldScaleZoom", {"Value": "-1"})
        ET.SubElement(midi_clip_elem, "NoteEditorFoldScaleScroll", {"Value": "0"})
        ET.SubElement(midi_clip_elem, "NoteSpellingPreference", {"Value": "0"})
        ET.SubElement(midi_clip_elem, "AccidentalSpellingPreference", {"Value": "3"})
        ET.SubElement(midi_clip_elem, "PreferFlatRootNote", {"Value": "false"})

        expr_grid = ET.SubElement(midi_clip_elem, "ExpressionGrid")
        ET.SubElement(expr_grid, "FixedNumerator", {"Value": "1"})
        ET.SubElement(expr_grid, "FixedDenominator", {"Value": "16"})
        ET.SubElement(expr_grid, "GridIntervalPixel", {"Value": "20"})
        ET.SubElement(expr_grid, "Ntoles", {"Value": "2"})
        ET.SubElement(expr_grid, "SnapToGrid", {"Value": "false"})
        ET.SubElement(expr_grid, "Fixed", {"Value": "false"})

        return midi_track_elem
