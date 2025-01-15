import os
from typing import Any, Dict, List

from wrappers.base_wrapper import BaseWrapper, TypedInput
from rvc.configs.config import Config
from rvc.infer.modules.vc.modules import VC


class Clone(BaseWrapper):
    title = "Clone"
    priority = 3
    allowed_kwargs = {
        "selected_voice": TypedInput(
            default="",
            description="The voice model to use.",
            type=str,
            gradio_type="Dropdown",
            required=True
        ),
        "speaker_id": TypedInput(
            default=0,
            description="Speaker ID to use.",
            type=int,
            gradio_type="Number",
        ),
        "separate_vocals": TypedInput(
            default=True,
            description="Separate vocals from the input audio.",
            type=bool,
            gradio_type="Checkbox",
        ),
        "pitch_shift": TypedInput(
            default=0,
            description="Pitch shift in semitones (+12 for an octave up, -12 for an octave down).",
            type=int,
            gradio_type="Number",
        ),
        "pitch_extraction_method": TypedInput(
            default="rmvpe",
            description="Pitch extraction algorithm.",
            type=str,
            choices=["pm", "harvest", "crepe", "rmvpe"],
            gradio_type="Dropdown",
        ),
        "export_format": TypedInput(
            default="wav",
            description="Output file format.",
            type=str,
            choices=["wav", "flac", "mp3", "m4a"],
            gradio_type="Dropdown",
        ),
        "resample_rate": TypedInput(
            default=0,
            description="Resample rate (0 for no resampling).",
            type=int,
            gradio_type="Slider",
            ge=0,
            le=48000,
            step=1,
        ),
        "volume_mix_rate": TypedInput(
            default=1,
            description="Mix ratio for volume envelope.",
            type=float,
            gradio_type="Slider",
            ge=0,
            le=1,
            step=0.01,
        ),
        "accent_strength": TypedInput(
            default=0.2,
            description="Protect clear consonants and breaths.",
            type=float,
            gradio_type="Slider",
            ge=0,
            le=0.5,
            step=0.01,
        ),
        "filter_radius": TypedInput(
            default=3,
            description="Median filter radius for 'harvest' pitch recognition.",
            type=int,
            gradio_type="Slider",
            ge=0,
            le=7,
            step=1,
        ),
        "index_rate": TypedInput(
            default=1,
            description="Feature search proportion.",
            type=float,
            gradio_type="Slider",
            ge=0,
            le=1,
            step=0.01,
        )
    }

    def setup(self):
        config = Config()
        self.vc = VC(config)

    def process_audio(self, inputs: List[str], **kwargs: Dict[str, Any]) -> List[str]:
        """
        Process audio inputs based on provided configurations.
        """
        self.setup()
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in self.allowed_kwargs}
        separate_vocals = filtered_kwargs.get("separate_vocals", True)
        selected_voice = filtered_kwargs.get("selected_voice", "")
        spk_id = filtered_kwargs.get("speaker_id", 0)
        pitch_shift = filtered_kwargs.get("pitch_shift", 0)
        f0method = filtered_kwargs.get("pitch_extraction_method", "rmvpe")
        resample_rate = filtered_kwargs.get("resample_rate", 0)
        rms_mix_rate = filtered_kwargs.get("volume_mix_rate", 1)
        protect = filtered_kwargs.get("accent_strength", 0.2)
        format_ = filtered_kwargs.get("export_format", "wav")
        index_rate = filtered_kwargs.get("index_rate", 1)
        filter_radius = filtered_kwargs.get("filter_radius", 3)

        outputs = []
        clone_outputs = self.vc.vc_multi(
            model=selected_voice,
            sid=spk_id,
            paths=inputs,
            f0_up_key=pitch_shift,
            format1=format_,
            f0_method=f0method,
            index_rate=index_rate,
            filter_radius=filter_radius,
            resample_sr=resample_rate,
            rms_mix_rate=rms_mix_rate,
            protect=protect
        )
        # for input_path in inputs:
        #     output = self.vc.vc_single(
        #         sid=selected_voice,
        #         input_audio_path=input_path,
        #         f0_method=f0method,
        #         index_rate=index_rate,
        #         filter_radius=filter_radius,
        #         resample_sr=resample_rate,
        #         rms_mix_rate=rms_mix_rate,
        #         protect=protect
        #     )
        #     outputs.append(output)

        return clone_outputs

    def change_choices(self) -> Dict[str, Any]:
        """
        Refresh available voices and indices.
        """
        weight_root = os.path.join(os.getenv("model_path", "models"), "cloned")
        voices = [name for name in os.listdir(weight_root) if name.endswith(".pth")]
        return {"choices": sorted(voices), "__type__": "update"}

    def clean(self):
        """
        Clean and reset states.
        """
        return {"value": "", "__type__": "update"}

    def register_api_endpoint(self, api):
        """
        Register the API endpoints for external usage.
        """
        api.add_route("/convert", self.process_audio, methods=["POST"])
        api.add_route("/change_choices", self.change_choices, methods=["GET"])
