import os
from typing import Any, List, Dict
import logging

from pydub import AudioSegment, effects

from handlers.reverb import apply_reverb
from util.audio_track import shift_pitch
from util.data_classes import ProjectFiles
from wrappers.base_wrapper import BaseWrapper, TypedInput

logger = logging.getLogger(__name__)


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

    def process_audio(self, pj_inputs: List[ProjectFiles], callback=None, **kwargs: Dict[str, Any]) -> List[
        ProjectFiles]:
        pj_outputs = []

        filtered_kwargs = {key: value for key, value in kwargs.items() if key in self.allowed_kwargs}
        pitch_shift = filtered_kwargs.get("pitch_shift", 0)

        for project in pj_inputs:
            logger.info(f"Processing project: {os.path.basename(project.project_dir)}")
            src_stem = project.src_file
            src_name, _ = os.path.splitext(os.path.basename(src_stem))
            output_folder = os.path.join(project.project_dir, "merged")
            os.makedirs(output_folder, exist_ok=True)

            inputs, _ = self.filter_inputs(project, "audio")

            ir_file = os.path.join(project.project_dir, "stems", "impulse_response.ir")

            new_inputs = []
            for i, stem_path in enumerate(inputs):
                if callback is not None:
                    callback(i / len(inputs), f"Processing stem: {os.path.basename(stem_path)}", len(inputs))
                logger.info(f"Processing stem: {os.path.basename(stem_path)}")
                if "(Vocals)" in stem_path and "(BG_Vocals" not in stem_path:
                    if os.path.exists(ir_file):
                        logger.info(f"Applying reverb to {os.path.basename(stem_path)}")
                        stem_name, ext = os.path.splitext(os.path.basename(stem_path))
                        reverb_stem_path = os.path.join(project.project_dir, "stems", f"{stem_name}(Re-Reverb){ext}")
                        reverb_stem = apply_reverb(stem_path, ir_file, reverb_stem_path)
                        seg = AudioSegment.from_file(reverb_stem)
                    else:
                        seg = AudioSegment.from_file(stem_path)
                else:
                    seg = AudioSegment.from_file(stem_path)

                if pitch_shift != 0 and "(Cloned)" not in stem_path:
                    logger.info(f"Applying pitch shift to {os.path.basename(stem_path)}")
                    seg = shift_pitch(seg, pitch_shift)

                new_inputs.append(seg)

            # Determine output file extension.
            if isinstance(new_inputs[0], AudioSegment):
                first_ext = ".wav"
            else:
                _, file_ext = os.path.splitext(os.path.basename(new_inputs[0]))
                first_ext = file_ext.lower() if file_ext else ".wav"

            output_file = os.path.join(output_folder, f"{src_name}_(Merged){first_ext}")
            if os.path.exists(output_file):
                os.remove(output_file)

            merged_segment = new_inputs[0]
            for seg in new_inputs[1:]:
                merged_segment = merged_segment.overlay(seg)

            # Normalize final mix.
            merged_segment = effects.normalize(merged_segment)

            # Export final output.
            export_format = first_ext.lstrip(".")
            merged_segment.export(
                output_file,
                format=export_format,
                bitrate="320k" if export_format == "mp3" else None
            )

            project.add_output("merged", output_file)
            pj_outputs.append(project)

        return pj_outputs

    def register_api_endpoint(self, api) -> Any:
        pass
