import os
import traceback
from typing import Any, List, Dict, Union

from pydub import AudioSegment, effects
from handlers.reverb import apply_reverb
from util.data_classes import ProjectFiles
from wrappers.base_wrapper import BaseWrapper, TypedInput

class Merge(BaseWrapper):
    title = "Merge"
    description = "Merge multiple audio files into a single track."
    priority = 6
    default = True

    allowed_kwargs = {
        "pitch_shift": TypedInput(
            default=0,
            description="Pitch shift in semitones (+12 for an octave up, -12 for an octave down).",
            type=int,
            gradio_type="Number",
            render=False
        ),
    }

    @staticmethod
    def shift_pitch(audio: AudioSegment, pitch_shift: int) -> AudioSegment:
        """
        Simple pitch shifting by changing frame_rate (low fidelity for large shifts).
        For better quality, consider an external library (e.g. rubberband).
        """
        if pitch_shift == 0:
            return audio
        return audio._spawn(
            audio.raw_data,
            overrides={"frame_rate": int(audio.frame_rate * 2 ** (pitch_shift / 12))}
        )

    def process_audio(self, pj_inputs: List[ProjectFiles], callback=None, **kwargs: Dict[str, Any]) -> List[ProjectFiles]:
        pj_outputs = []

        # Choose desired sample rate & channels for the final mix:
        TARGET_SAMPLE_RATE = 44100
        TARGET_CHANNELS = 2  # stereo

        # Gain reduction for each stem before overlay (in dB).
        STEM_ATTENUATION_DB = 0  # Try -3 or -6 if you have many stems.

        # Pull relevant kwargs
        filtered_kwargs = {key: value for key, value in kwargs.items() if key in self.allowed_kwargs}
        pitch_shift = filtered_kwargs.get("pitch_shift", 0)

        for project in pj_inputs:
            src_stem = project.src_file
            src_name, _ = os.path.splitext(os.path.basename(src_stem))
            output_folder = os.path.join(project.project_dir, "merged")
            os.makedirs(output_folder, exist_ok=True)

            # Gather stems that we want to merge:
            inputs, _ = self.filter_inputs(project, "audio")

            # Handle multiple Drums
            drum_inputs = [stem for stem in inputs if "(Drums" in stem]
            if len(drum_inputs) > 1:
                # remove base (Drums) stem
                inputs = [stem for stem in inputs if "(Drums)" not in stem]

            # Handle specialized vs. generic "Instrumental"
            instrumental_keys = ["(Piano)", "(Guitar)", "(Bass)", "(Woodwinds)"]
            instrumental_inputs = [stem for stem in inputs if any(key in stem for key in instrumental_keys)]
            if len(instrumental_inputs) > 1:
                inputs = [stem for stem in inputs if "(Instrumental)" not in stem]

            # Optional Reverb on main vocals if impulse_response file found
            ir_file = os.path.join(project.project_dir, "impulse_response.ir")
            if os.path.exists(ir_file):
                new_inputs = []
                for stem_path in inputs:
                    # If it's main vocals, apply reverb (and pitch shift if needed)
                    if "(Vocals)" in stem_path and "(BG_Vocals)" not in stem_path and "(Reverb)" not in stem_path:
                        print(f"Applying impulse response to vocal file {stem_path}...")
                        try:
                            stem_name, ext = os.path.splitext(os.path.basename(stem_path))
                            reverb_stem_path = os.path.join(
                                project.project_dir, "stems", f"{stem_name}(Re-Reverb){ext}"
                            )
                            # apply reverb on disk
                            reverb_stem = apply_reverb(stem_path, ir_file, reverb_stem_path)
                            # load as AudioSegment, do pitch shift if needed
                            seg = AudioSegment.from_file(reverb_stem)
                            if pitch_shift != 0:
                                seg = self.shift_pitch(seg, pitch_shift)
                            new_inputs.append(seg)
                        except Exception as e:
                            print(f"Error applying reverb to {stem_path}: {e}")
                            traceback.print_exc()
                            # fallback: just load the original as an AudioSegment
                            seg = AudioSegment.from_file(stem_path)
                            if pitch_shift != 0:
                                seg = self.shift_pitch(seg, pitch_shift)
                            new_inputs.append(seg)
                    else:
                        seg = AudioSegment.from_file(stem_path)
                        if pitch_shift != 0:
                            seg = self.shift_pitch(seg, pitch_shift)
                        new_inputs.append(seg)
                inputs = new_inputs
            else:
                # If no IR file, we can still do pitch shift on all if needed
                new_inputs = []
                for stem_path in inputs:
                    seg = AudioSegment.from_file(stem_path)
                    if pitch_shift != 0:
                        seg = self.shift_pitch(seg, pitch_shift)
                    new_inputs.append(seg)
                inputs = new_inputs

            if not inputs:
                print(f"No audio inputs found for project: {project.project_dir}")
                pj_outputs.append(project)
                continue

            # Determine extension from the first file (original logic).
            # If your preference is always mp3 or always wav, you can force it here.
            first_ext = ".wav"
            if isinstance(inputs[0], AudioSegment):
                # We can't guess extension from an in-memory segment
                # So let's just default to WAV or do your own logic
                first_ext = ".wav"
            else:
                _, file_ext = os.path.splitext(os.path.basename(inputs[0]))
                first_ext = file_ext.lower() if file_ext else ".wav"

            output_file = os.path.join(output_folder, f"{src_name}_(Merged){first_ext}")
            if os.path.exists(output_file):
                os.remove(output_file)

            total_steps = len(inputs)
            current_step = 0
            if callback:
                callback(0, f"Starting merge of {len(inputs)} tracks", total_steps)

            # Helper to load/convert either a file path or an in-memory AudioSegment
            def load_and_convert(item: Union[str, AudioSegment]) -> AudioSegment:
                if isinstance(item, AudioSegment):
                    seg_ = item
                else:
                    seg_ = AudioSegment.from_file(item)
                seg_ = seg_.set_frame_rate(TARGET_SAMPLE_RATE)
                seg_ = seg_.set_channels(TARGET_CHANNELS)
                if STEM_ATTENUATION_DB != 0:
                    seg_ = seg_.apply_gain(STEM_ATTENUATION_DB)
                return seg_

            # Start with first item
            merged_segment = load_and_convert(inputs[0])
            current_step += 1
            if callback:
                callback(current_step / total_steps, f"Loaded first track", total_steps)

            # Overlay remaining
            for i in range(1, len(inputs)):
                seg_i = load_and_convert(inputs[i])
                merged_segment = merged_segment.overlay(seg_i)
                current_step += 1
                if callback:
                    callback(current_step / total_steps, f"Merged track {i+1}", total_steps)

            # Optional normalization
            merged_segment = effects.normalize(merged_segment)

            # Final export (only once!)
            # If you have a known extension like .mp3, set format="mp3", etc.
            # For WAV, you could do: format="wav", parameters=["-acodec", "pcm_s16le"]
            export_format = first_ext.lstrip(".")
            merged_segment.export(
                output_file,
                format=export_format,
                bitrate="320k" if export_format == "mp3" else None
            )

            if callback:
                callback(total_steps, f"Exported final merged track to {os.path.basename(output_file)}", total_steps)

            project.add_output("merged", output_file)
            pj_outputs.append(project)

        return pj_outputs

    def register_api_endpoint(self, api) -> Any:
        pass
