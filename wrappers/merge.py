import os
from typing import Any, List, Dict

from pydub import AudioSegment

from handlers.reverb import apply_reverb
from util.data_classes import ProjectFiles
from wrappers.base_wrapper import BaseWrapper


class Merge(BaseWrapper):
    title = "Merge"
    priority = 5
    default = True

    def process_audio(self, inputs: List[ProjectFiles], callback=None, **kwargs: Dict[str, Any]) -> List[ProjectFiles]:
        pj_outputs = []
        for project in inputs:
            src_stem = project.src_file
            src_name, _ = os.path.splitext(os.path.basename(src_stem))
            output_folder = os.path.join(project.project_dir, "merged")
            os.makedirs(output_folder, exist_ok=True)
            inputs, _ = self.filter_inputs(project, "audio")
            ir_file = os.path.join(project.project_dir, "impulse_response.ir")
            if os.path.exists(ir_file):
                new_inputs = []
                for stem in inputs:
                    if "(Vocals)" in stem and "(BG_Vocals)" not in stem:
                        print(f"Applying impulse response (Reverb) to vocal file {stem}...")
                        try:
                            stem_name, ext = os.path.splitext(os.path.basename(stem))
                            reverb_stem_path = os.path.join(output_folder, f"{stem_name}_(Reverb){ext}")
                            reverb_stem = apply_reverb(stem, ir_file, reverb_stem_path)
                            new_inputs.append(reverb_stem_path)
                        except Exception as e:
                            print(f"Error applying reverb to {stem}: {e}")
                            new_inputs.append(stem)
                    else:
                        new_inputs.append(stem)
                inputs = new_inputs
            # Set up output file details
            first_file = inputs[0]
            file_name, file_ext = os.path.splitext(os.path.basename(first_file))
            output_file = os.path.join(output_folder, f"{src_name}_(Merged){file_ext}")

            # Initialize progress tracking
            total_steps = len(inputs)
            current_step = 0
            if callback:
                callback(0, f"Starting merge of {len(inputs)} tracks", total_steps)

            # Load the first audio file
            stem_0 = AudioSegment.from_file(first_file)
            current_step += 1
            if callback:
                callback(current_step, f"Loaded {os.path.basename(first_file)}", total_steps)

            # Merge remaining audio files
            for i in range(1, len(inputs)):
                stem_i = AudioSegment.from_file(inputs[i])
                stem_0 = stem_0.overlay(stem_i)
                current_step += 1
                if callback:
                    callback(current_step, f"Merged {os.path.basename(inputs[i])}", total_steps)

            # Export merged track
            stem_0.export(output_file, format=file_ext[1:])
            if callback:
                callback(total_steps, f"Exported merged track to {output_file}", total_steps)
            project.add_output("merged", output_file)
            pj_outputs.append(project)

        return pj_outputs

    def register_api_endpoint(self, api) -> Any:
        pass
