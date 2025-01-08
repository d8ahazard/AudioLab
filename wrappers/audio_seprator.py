import os
from typing import List, Dict, Any

from modules.audio_separator.audio_separator import separate_music
from wrappers.base_wrapper import BaseWrapper
from handlers.config import output_path


class AudioSeparator(BaseWrapper):
    priority = 0
    allowed_kwargs = {
        "cpu": bool,
        "overlap_demucs": float,
        "overlap_VOCFT": float,
        "overlap_VitLarge": int,
        "overlap_InstVoc": int,
        "weight_InstVoc": float,
        "weight_VOCFT": float,
        "weight_VitLarge": float,
        "single_onnx": bool,
        "large_gpu": bool,
        "BigShifts": int,
        "vocals_only": bool,
        "use_VOCFT": bool,
        "output_format": str
    }

    def register_api_endpoint(self, api) -> Any:
        pass

    def process_audio(self, inputs: List[str], **kwargs: Dict[str, Any]) -> List[str]:
        print(f"Separating audio from {inputs}")
        output_folder = os.path.join(output_path, "separated")
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in self.allowed_kwargs.keys()}
        return separate_music(inputs, output_folder, **filtered_kwargs)