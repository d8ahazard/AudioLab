import os
from typing import Any, List, Dict
import logging

from pydub import AudioSegment, effects

from handlers.reverb import apply_reverb
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

    @staticmethod
    def shift_pitch(audio: AudioSegment, pitch_shift: int) -> AudioSegment:
        """
        Pitch shift an AudioSegment by a given number of semitones WITHOUT altering its speed,
        using torchaudio.functional.pitch_shift (no SoX, no RubberBand).

        Requirements:
         - torchaudio >= 2.1 (which includes pitch_shift)
         - PyTorch installed (version matching torchaudio)
        """
        if pitch_shift == 0:
            return audio

        import torch
        import torchaudio.functional as AF
        import numpy as np

        # 1) Gather audio info
        sample_rate = audio.frame_rate
        channels = audio.channels
        sample_width = audio.sample_width

        # 2) Convert AudioSegment to float32 NumPy array
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)

        #   e.g. for 16-bit audio, max_val = 32768
        max_val = float(1 << (8 * sample_width - 1))
        # Normalize to [-1.0, 1.0]
        samples = samples / max_val

        # 3) Reshape to [channels, num_frames] for torchaudio
        if channels > 1:
            samples = samples.reshape(-1, channels).T  # shape: (channels, n_frames)
        else:
            # shape: (1, n_frames)
            samples = np.expand_dims(samples, axis=0)

        # 4) Convert to a PyTorch tensor
        waveform = torch.from_numpy(samples)

        # 5) Call torchaudio pitch_shift
        #    (n_steps is the # of semitones; bins_per_octave=12 means standard semitones)
        pitched_wf = AF.pitch_shift(
            waveform,
            sample_rate=sample_rate,
            n_steps=pitch_shift,
            bins_per_octave=12
        )

        # 6) Convert pitched audio back to NumPy, shape => (n_frames, channels)
        pitched_np = pitched_wf.numpy()
        pitched_np = pitched_np.transpose(1, 0)  # Now shape: (n_frames, channels)

        # 7) Denormalize from [-1,1] back to integer range
        pitched_np = pitched_np * max_val

        # Decide on integer dtype
        if sample_width == 1:
            dtype = np.int8
        elif sample_width == 2:
            dtype = np.int16
        elif sample_width == 4:
            dtype = np.int32
        else:
            dtype = np.int16  # fallback

        # Clip & convert
        min_val, max_allow = -max_val, max_val - 1
        pitched_np = np.clip(pitched_np, min_val, max_allow).astype(dtype)

        # 8) Flatten if multi-channel to match pydub’s raw byte layout
        pitched_flat = pitched_np.flatten()

        # 9) Spawn a new AudioSegment with the same metadata but pitched samples
        return audio._spawn(pitched_flat.tobytes())

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
            for stem_path in inputs:
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

                if pitch_shift != 0 and "(Vocals)" not in stem_path:
                    logger.info(f"Applying pitch shift to {os.path.basename(stem_path)}")
                    seg = self.shift_pitch(seg, pitch_shift)

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
