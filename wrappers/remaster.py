import os
from typing import Any, List, Callable, Dict
import matchering as mg

from handlers.config import output_path
from wrappers.base_wrapper import BaseWrapper, TypedInput


class Remaster(BaseWrapper):
    title = "Remaster"
    priority = 4
    allowed_kwargs = {
        "reference": TypedInput(
            description="Reference track",
            default=None,
            type=str,
            gradio_type="File"
        )
    }

    def process_audio(self, inputs: List[str], callback: Callable = None, **kwargs: Dict[str, Any]) -> List[str]:
        output_folder = os.path.join(output_path, "remastered")
        mg.log(print)
        inputs, outputs = self.filter_inputs(inputs, "audio")
        reference_file = kwargs.get("reference")
        if not reference_file:
            raise ValueError("Reference track not provided")
        callback(0, f"Remastering {len(inputs)} audio files", len(inputs))
        callback_step = 0
        for input_file in inputs:
            callback(callback_step, f"Remastering {input_file}", len(inputs))
            inputs_name, inputs_ext = os.path.splitext(os.path.basename(input_file))
            output_file = os.path.join(output_folder, f"{inputs_name}_remastered{inputs_ext}")
            mg.process(
                # The track you want to master
                target=input_file,
                # Some "wet" reference track
                reference=reference_file,
                # Where and how to save your results
                results=[
                    mg.pcm24(output_file),
                ],
            )
            outputs.append(output_file)
            callback_step += 1
        return outputs

    def register_api_endpoint(self, api) -> Any:
        pass