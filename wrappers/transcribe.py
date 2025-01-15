import json
import os
from typing import Any, List, Dict

import whisperx
import gc

from handlers.config import output_path
from wrappers.base_wrapper import BaseWrapper


class Transcribe(BaseWrapper):
    title = "Transcribe"
    description = "Transcribe audio files"
    priority = 100
    model = None
    model_a = None
    diarize_model = None

    def process_audio(self, inputs: List[str], callback=None, **kwargs: Dict[str, Any]) -> List[str]:
        device = "cuda"
        batch_size = 16
        compute_type = "float16"
        output_folder = os.path.join(output_path, 'audio_transcribe')
        os.makedirs(output_folder, exist_ok=True)

        # 1. Transcribe with original whisper (batched)
        self.model = whisperx.load_model("large-v3", device, compute_type=compute_type)
        inputs, _ = self.filter_inputs(inputs, "audio")
        json_outputs = []  # Separate list for JSON files
        instrum_keys = ["_instrum", "_bass", "_drum", "_other"]
        # Filter out any files with instrum_keys in the name
        inputs = [i for i in inputs if not any([k in i for k in instrum_keys])]

        if len(inputs):
            callback(0, f"Transcribing {len(inputs)} audio files", len(inputs) * 4)
        audio_step = 1
        for audio_file in inputs:
            file_name, _ = os.path.splitext(os.path.basename(audio_file))
            # If any of the instrum_keys is in the file name, skip it

            audio = whisperx.load_audio(audio_file)
            callback(audio_step, f"Transcribing {file_name}", len(inputs) * 4)
            result = self.model.transcribe(audio, batch_size=batch_size)

            # 2. Align whisper output
            audio_step += 1
            callback(audio_step, f"Aligning {file_name}", len(inputs) * 4)
            self.model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
            result = whisperx.align(result["segments"], self.model_a, metadata, audio, device,
                                    return_char_alignments=False)

            # 3. Assign speaker labels
            audio_step += 1
            callback(audio_step, f"Assigning speakers {file_name}", len(inputs) * 4)
            self.diarize_model = whisperx.DiarizationPipeline(model_name="tensorlake/speaker-diarization-3.1",
                                                              device=device)
            # add min/max number of speakers if known
            diarize_segments = self.diarize_model(audio)
            audio_step += 1
            callback(audio_step, f"Assigning speakers {file_name}", len(inputs) * 4)
            result = whisperx.assign_word_speakers(diarize_segments, result)

            out_file = os.path.join(output_folder, f"{file_name}.json")
            with open(out_file, 'w') as f:
                json.dump(result, f, indent=4)
            json_outputs.append(out_file)  # Add to JSON outputs

        self.clean()
        callback(len(inputs) * 4, "Transcription complete", len(inputs) * 4)
        return inputs + json_outputs  # Combine original inputs with JSON files

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
