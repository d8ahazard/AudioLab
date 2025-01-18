import gc
import json
import os
from typing import Any, List, Dict

import whisperx

from util.data_classes import ProjectFiles
from wrappers.base_wrapper import BaseWrapper


class Transcribe(BaseWrapper):
    title = "Transcribe"
    description = "Transcribe audio files"
    priority = 100
    model = None
    model_a = None
    diarize_model = None

    def process_audio(self, inputs: List[ProjectFiles], callback=None, **kwargs: Dict[str, Any]) -> List[ProjectFiles]:
        device = "cuda"
        batch_size = 16
        compute_type = "float16"

        # 1. Transcribe with original whisper (batched)
        self.model = whisperx.load_model("large-v3", device, compute_type=compute_type)
        pj_outputs = []
        for project in inputs:
            last_outputs = project.last_outputs.copy()
            output_folder = os.path.join(project.project_dir, "transcriptions")
            os.makedirs(output_folder, exist_ok=True)
            filtered_inputs, _ = self.filter_inputs(project.last_outputs, "audio")
            json_outputs = []  # Separate list for JSON files
            instrum_keys = ["(Instrumental)", "(Bass)", "(Drum)", "(Other)"]
            # Filter out any files with instrum_keys in the name
            filtered_inputs = [i for i in filtered_inputs if not any([k in i for k in instrum_keys])]

            if len(filtered_inputs):
                callback(0, f"Transcribing {len(filtered_inputs)} audio files", len(filtered_inputs) * 4)
            audio_step = 1
            for audio_file in filtered_inputs:
                file_name, _ = os.path.splitext(os.path.basename(audio_file))
                # If any of the instrum_keys is in the file name, skip it

                audio = whisperx.load_audio(audio_file)
                callback(audio_step, f"Transcribing {file_name}", len(filtered_inputs) * 4)
                result = self.model.transcribe(audio, batch_size=batch_size)

                # 2. Align whisper output
                audio_step += 1
                callback(audio_step, f"Aligning {file_name}", len(filtered_inputs) * 4)
                self.model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
                result = whisperx.align(result["segments"], self.model_a, metadata, audio, device,
                                        return_char_alignments=False)

                # 3. Assign speaker labels
                audio_step += 1
                callback(audio_step, f"Assigning speakers {file_name}", len(filtered_inputs) * 4)
                self.diarize_model = whisperx.DiarizationPipeline(model_name="tensorlake/speaker-diarization-3.1",
                                                                  device=device)
                # add min/max number of speakers if known
                diarize_segments = self.diarize_model(audio)
                audio_step += 1
                callback(audio_step, f"Assigning speakers {file_name}", len(filtered_inputs) * 4)
                result = whisperx.assign_word_speakers(diarize_segments, result)

                out_file = os.path.join(output_folder, f"{file_name}.json")
                with open(out_file, 'w') as f:
                    json.dump(result, f, indent=4)
                json_outputs.append(out_file)  # Add to JSON outputs
            project.add_output("transcriptions", json_outputs)
            last_outputs.extend(json_outputs)
            project.last_outputs = last_outputs
            pj_outputs.append(project)

        self.clean()
        return pj_outputs

    def register_api_endpoint(self, api) -> Any:
        pass

    def clean(self):
        for model in [self.model, self.model_a, self.diarize_model]:
            if model is not None:
                try:
                    del model
                except Exception as e:
                    print(f"Error deleting model: {e}")
        gc.collect()
