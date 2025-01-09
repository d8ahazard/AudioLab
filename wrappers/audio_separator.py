import os
from typing import List, Dict, Any, Callable

from modules.audio_separator.audio_separator import separate_music
from wrappers.base_wrapper import BaseWrapper, TypedInput
from handlers.config import output_path


class AudioSeparator(BaseWrapper):
    priority = 0
    allowed_kwargs = {
        "output_folder": TypedInput(
            description="Output folder",
            default=None,
            type=str,
            render=False
        ),
        "cpu": TypedInput(
            description="Use CPU for inference instead of GPU",
            default=False,
            type=bool
        ),
        "overlap_demucs": TypedInput(
            description="Overlap size for Demucs",
            default=0.1,
            type=float,
            ge=0.0,
            le=0.99,
            gradio_type="Slider"
        ),
        "overlap_VOCFT": TypedInput(
            description="Overlap size for VOCFT",
            default=0.1,
            type=float,
            ge=0.0,
            le=0.99,
            gradio_type="Slider"
        ),
        "overlap_VitLarge": TypedInput(
            description="Overlap size for VitLarge",
            default=1,
            type=int,
            ge=1,
            le=10,
            gradio_type="Slider"
        ),
        "overlap_InstVoc": TypedInput(
            description="Overlap size for InstVoc",
            default=1,
            type=int,
            ge=1,
            le=10,
            gradio_type="Slider"
        ),
        "weight_InstVoc": TypedInput(
            description="Weight for InstVoc",
            default=8.0,
            type=float,
            ge=0.1,
            le=20.0,
            gradio_type="Slider"
        ),
        "weight_VOCFT": TypedInput(
            description="Weight for VOCFT",
            default=1.0,
            type=float,
            ge=0.1,
            le=10.0,
            gradio_type="Slider"
        ),
        "weight_VitLarge": TypedInput(
            description="Weight for VitLarge",
            default=5.0,
            type=float,
            ge=0.1,
            le=10.0,
            gradio_type="Slider"
        ),
        "single_onnx": TypedInput(
            description="Use a single ONNX model for inference",
            default=False,
            type=bool
        ),
        "large_gpu": TypedInput(
            description="Enable optimizations for large GPUs",
            default=False,
            type=bool
        ),
        "BigShifts": TypedInput(
            description="Big shift value",
            default=7,
            type=int,
            ge=1,
            le=20,
            gradio_type="Slider"
        ),
        "vocals_only": TypedInput(
            description="Process vocals only",
            default=False,
            type=bool
        ),
        "use_VOCFT": TypedInput(
            description="Enable VOCFT processing",
            default=False,
            type=bool
        ),
        "output_format": TypedInput(
            description="Output format",
            default="FLOAT",
            type=str,
            render=False
        ),
        "return_all_stems": TypedInput(
            description="Return all stems",
            default=False,
            type=bool
        ),
    }

    def register_api_endpoint(self, api) -> Any:
        pass

    def process_audio(self, inputs: List[str], callback: Callable = None, **kwargs: Dict[str, Any]) -> List[str]:
        print(f"Separating audio from {inputs}")
        output_folder = os.path.join(output_path, "separated")
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in self.allowed_kwargs.keys()}
        # Pop return_all_stems, this is for us.
        return_all_stems = filtered_kwargs.pop("return_all_stems", False)
        outputs = separate_music(inputs, output_folder, **filtered_kwargs)
        all_keys = ["bass", "drums", "other", "vocals"]
        if not return_all_stems:
            all_keys = ["instrum", "vocals"]
        filtered_outputs = []
        for key in all_keys:
            for output in outputs:
                if f"_{key}" in output:
                    filtered_outputs.append(output)
                    break
        # Delete any files not in filtered_outputs
        for output in outputs:
            if output not in filtered_outputs:
                os.remove(output)
        return filtered_outputs
