import os
from typing import Any, List, Dict
import logging

from pydub import AudioSegment, effects

from handlers.reverb import apply_reverb
from util.audio_track import shift_pitch
from util.data_classes import ProjectFiles
from wrappers.base_wrapper import BaseWrapper, TypedInput

logger = logging.getLogger(__name__)


def normalize_segment(src_track: str, segment: AudioSegment, prevent_clipping: bool = True) -> AudioSegment:
    import math
    # Load the original track to determine target loudness.
    original_segment = AudioSegment.from_file(src_track)
    target_dBFS = original_segment.dBFS

    # Normalize the merged segment.
    merged_segment = effects.normalize(segment)
    current_dBFS = merged_segment.dBFS

    # Calculate the gain change required to match the original's dBFS.
    gain_change = target_dBFS - current_dBFS

    if prevent_clipping:
        # Calculate maximum possible amplitude for the segment.
        max_possible_amplitude = float(1 << ((merged_segment.sample_width * 8) - 1))
        # Get the current peak amplitude.
        peak_amp = merged_segment.max
        # Compute the current peak level in dBFS.
        if peak_amp == 0:
            current_peak_dBFS = -float('inf')
        else:
            current_peak_dBFS = 20 * math.log10(peak_amp / max_possible_amplitude)
        # Determine how much gain we can add without clipping (i.e. peak should not exceed 0 dBFS).
        allowed_gain = -current_peak_dBFS  # Since 0 dBFS is the maximum
        # Limit the gain change if it would cause clipping.
        if gain_change > allowed_gain:
            gain_change = allowed_gain

    # Apply the calculated gain change.
    return merged_segment.apply_gain(gain_change)


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
        "prevent_clipping": TypedInput(
            default=True,
            description="Prevent clipping in the output audio by normalizing the final mix.",
            type=bool,
            gradio_type="Checkbox"
        ),
        "selected_voice": TypedInput(
            default="Vocals",
            description="Select the voice to be processed.",
            type=str,
            gradio_type="Text",
            render=False
        ),
        "pitch_extraction_method": TypedInput(
            default="rmvpe+",
            description="Select the pitch extraction method.",
            type=str,
            gradio_type="Text",
            render=False
        ),
    }

    def process_audio(self, pj_inputs: List[ProjectFiles], callback=None, **kwargs: Dict[str, Any]) -> List[ProjectFiles]:
        pj_outputs = []

        filtered_kwargs = {key: value for key, value in kwargs.items() if key in self.allowed_kwargs}
        pitch_shift = filtered_kwargs.get("pitch_shift", 0)
        selected_voice = filtered_kwargs.get("selected_voice", None)
        pitch_extraction_method = filtered_kwargs.get("pitch_extraction_method", "rmvpe+")
        try:
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
                            src_name = stem_name.replace("(Vocals)", "")
                            reverb_stem_path = os.path.join(project.project_dir, "stems",
                                                            f"{stem_name}(Re-Reverb){ext}")
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
                name_str = ""
                if selected_voice is not None and selected_voice != "":
                    name_str = f"({selected_voice}_{pitch_extraction_method})"
                if name_str in src_name:
                    name_str = ""
                output_file = os.path.join(output_folder, f"{src_name}{name_str}(Merged){first_ext}")
                if os.path.exists(output_file):
                    os.remove(output_file)

                merged_segment = new_inputs[0]
                for seg in new_inputs[1:]:
                    merged_segment = merged_segment.overlay(seg)

                merged_segment = normalize_segment(project.src_file, merged_segment,
                                                   prevent_clipping=filtered_kwargs.get("prevent_clipping", True))
                # Export final output.
                export_format = first_ext.lstrip(".")
                merged_segment.export(
                    output_file,
                    format=export_format,
                    bitrate="320k" if export_format == "mp3" else None
                )

                project.add_output("merged", output_file)
                pj_outputs.append(project)
        except Exception as e:
            logger.exception("Error merging audio files.")
            if callback is not None:
                callback(1.0, "Error merging audio files.", 1)
            raise e
        return pj_outputs

    def register_api_endpoint(self, api) -> Any:
        """
        Register FastAPI endpoint for audio merging.
        
        Args:
            api: FastAPI application instance
            
        Returns:
            The registered endpoint route
        """
        from fastapi import File, UploadFile, HTTPException
        from fastapi.responses import FileResponse
        from pydantic import BaseModel, create_model
        from typing import List, Optional
        import tempfile
        from pathlib import Path

        # Create Pydantic model for settings
        fields = {}
        for key, value in self.allowed_kwargs.items():
            field_type = value.type
            if value.field.default == ...:
                field_type = Optional[field_type]
            fields[key] = (field_type, value.field)
        
        SettingsModel = create_model(f"{self.__class__.__name__}Settings", **fields)

        @api.post("/api/v1/process/merge")
        async def process_merge(
            files: List[UploadFile] = File(...),
            settings: Optional[SettingsModel] = None
        ):
            """
            Merge multiple audio files into a single track.
            
            Args:
                files: List of audio files to merge
                settings: Merge settings including pitch shift and clipping prevention
                
            Returns:
                Merged audio file
            """
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Save uploaded files
                    input_files = []
                    for file in files:
                        file_path = Path(temp_dir) / file.filename
                        with file_path.open("wb") as f:
                            content = await file.read()
                            f.write(content)
                        input_files.append(ProjectFiles(str(file_path)))
                    
                    # Process files
                    settings_dict = settings.dict() if settings else {}
                    processed_files = self.process_audio(input_files, **settings_dict)
                    
                    # Return merged file
                    output_files = []
                    for project in processed_files:
                        for output in project.last_outputs:
                            output_path = Path(output)
                            if output_path.exists():
                                output_files.append(FileResponse(output))
                    
                    if not output_files:
                        raise HTTPException(status_code=500, detail="No merged file generated")
                    
                    return output_files[0]  # Return the first (and should be only) merged file
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        return process_merge
