import os
import threading
from typing import Any, List, Dict
import logging

from modules.separator.stem_separator import separate_music
from util.data_classes import ProjectFiles
from wrappers.base_wrapper import BaseWrapper, TypedInput

logger = logging.getLogger(__name__)


class Separate(BaseWrapper):
    """
    A slimmed‐down wrapper that simply passes the user’s file and option settings
    to the main separation model.
    """
    title = "Separate"
    priority = 1
    default = True
    required = True
    description = (
        "Separate audio into distinct stems with optional background vocal splitting "
        "and audio transformations (reverb, echo, delay, crowd, noise removal)."
    )
    file_operation_lock = threading.Lock()

    allowed_kwargs = {
        "delete_extra_stems": TypedInput(
            default=True,
            description="Automatically delete intermediate stem files after processing.",
            type=bool,
            gradio_type="Checkbox"
        ),
        "separate_bg_vocals": TypedInput(
            default=True,
            description="Separate background vocals from main vocals.",
            type=bool,
            gradio_type="Checkbox"
        ),
        "separate_stems": TypedInput(
            default=False,
            description="Separate the audio into distinct stems.",
            type=bool,
            gradio_type="Checkbox"
        ),
        "store_reverb_ir": TypedInput(
            default=True,
            description="Store the impulse response for reverb removal.",
            type=bool,
            gradio_type="Checkbox"
        ),
        "separate_drums": TypedInput(
            default=False,
            description="Separate the drum track.",
            type=bool,
            gradio_type="Checkbox"
        ),
        "separate_woodwinds": TypedInput(
            default=False,
            description="Separate the woodwind instruments.",
            type=bool,
            gradio_type="Checkbox"
        ),
        "alt_bass_model": TypedInput(
            default=False,
            description="Use an alternative bass model.",
            type=bool,
            gradio_type="Checkbox"
        ),
        # Removal toggles
        "reverb_removal": TypedInput(
            default="Nothing",
            description="Apply reverb removal.",
            type=str,
            choices=["Nothing", "Main Vocals", "All Vocals", "All"],
            gradio_type="Dropdown"
        ),
        "echo_removal": TypedInput(
            default="Nothing",
            description="Apply echo removal.",
            type=str,
            choices=["Nothing", "Main Vocals", "All Vocals", "All"],
            gradio_type="Dropdown"
        ),
        "delay_removal": TypedInput(
            default="Nothing",
            description="Apply delay removal.",
            type=str,
            choices=["Nothing", "Main Vocals", "All Vocals", "All"],
            gradio_type="Dropdown"
        ),
        "crowd_removal": TypedInput(
            default="Nothing",
            description="Apply crowd noise removal.",
            type=str,
            choices=["Nothing", "Main Vocals", "All Vocals", "All"],
            gradio_type="Dropdown"
        ),
        "noise_removal": TypedInput(
            default="Nothing",
            description="Apply general noise removal.",
            type=str,
            choices=["Nothing", "Main Vocals", "All Vocals", "All"],
            gradio_type="Dropdown"
        ),
        # Model picks
        "delay_removal_model": TypedInput(
            default="UVR-DeEcho-DeReverb.pth",
            description="Which echo/delay removal model to use",
            type=str,
            choices=["UVR-DeEcho-DeReverb.pth", "UVR-De-Echo-Normal.pth"],
            gradio_type="Dropdown"
        ),
        "noise_removal_model": TypedInput(
            default="UVR-DeNoise.pth",
            description="Choose the model used for noise removal.",
            type=str,
            choices=["UVR-DeNoise.pth", "UVR-DeNoise-Lite.pth"],
            gradio_type="Dropdown"
        ),
        "crowd_removal_model": TypedInput(
            default="UVR-MDX-NET_Crowd_HQ_1.onnx",
            description="Select the model for crowd noise removal.",
            type=str,
            choices=["UVR-MDX-NET_Crowd_HQ_1.onnx", "mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt"],
            gradio_type="Dropdown"
        ),
    }

    def register_api_endpoint(self, api) -> Any:
        pass

    def process_audio(self, inputs: List[ProjectFiles], callback=None, **kwargs: Dict[str, any]) -> List[ProjectFiles]:
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in self.allowed_kwargs}
        outputs = []
        for project in inputs:
            out_dir = os.path.join(project.project_dir, "stems")
            os.makedirs(out_dir, exist_ok=True)
            # Call the main separation model
            stem_files = separate_music(
                input_audio=[project.src_file],
                output_folder=out_dir,
                callback=callback,
                **filtered_kwargs
            )
            project.add_output("stems", stem_files)
            outputs.append(project)
        # Optionally delete extra stems if requested
        if filtered_kwargs.get("delete_extra_stems", True):
            # Implement deletion logic as needed.
            pass
        return outputs

    def del_stem(self, path: str) -> bool:
        try:
            with self.file_operation_lock:
                if os.path.exists(path):
                    os.remove(path)
                    return True
        except Exception as e:
            print(f"Error deleting {path}: {e}")
        return False
