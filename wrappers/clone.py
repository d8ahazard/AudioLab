import os
from typing import Any, Dict, List, Callable

from handlers.config import model_path
from util.data_classes import ProjectFiles
from wrappers.base_wrapper import BaseWrapper, TypedInput
from rvc.configs.config import Config
from rvc.infer.modules.vc.modules import VC


def list_speakers():
    speaker_dir = os.path.join(model_path, "trained")
    # List all .pth files in the speaker directory
    return [os.path.join(speaker_dir, f) for f in os.listdir(speaker_dir) if f.endswith(".pth")]


def list_speakers_ui():
    return {"choices": list_speakers(), "__type__": "update"}


class Clone(BaseWrapper):
    title = "Clone"
    priority = 3
    default = True
    vc = None
    description = "Clone vocals from one audio file to another using a pre-trained RVC voice model."
    all_speakers = list_speakers()
    first_speaker = all_speakers[0] if all_speakers else None
    allowed_kwargs = {
        "selected_voice": TypedInput(
            default=first_speaker,
            description="The voice model to use for cloning vocals.",
            choices=list_speakers(),
            type=str,
            gradio_type="Dropdown",
            refresh=list_speakers_ui,
            required=True
        ),
        "clone_bg_vocals": TypedInput(
            default=False,
            description="Clone background vocals in addition to the main vocals. (Not recommended with layred harmonies, will cause artifacts.)",
            type=bool,
            gradio_type="Checkbox",
        ),
        "speaker_id": TypedInput(
            default=0,
            description="The ID of the speaker to use if the model was trained on multiple speakers.",
            type=int,
            gradio_type="Number",
        ),
        "pitch_shift": TypedInput(
            default=0,
            description="Pitch shift in semitones (+12 for an octave up, -12 for an octave down). Note, background vocals or instrumentals will not currently be pitch-shifted.",
            type=int,
            gradio_type="Number",
        ),
        "pitch_extraction_method": TypedInput(
            default="rmvpe",
            description="Pitch extraction algorithm. 'rmvpe' is recommended for most cases.",
            type=str,
            choices=["pm", "harvest", "crepe", "rmvpe"],
            gradio_type="Dropdown",
        ),
        "export_format": TypedInput(
            default="wav",
            description="Output file format. (WAV is recommended for most cases.)",
            type=str,
            choices=["wav", "flac", "mp3", "m4a"],
            gradio_type="Dropdown",
        ),
        "resample_rate": TypedInput(
            default=0,
            description="Resample rate (0 for no resampling, i.e. - keep the original sample rate.).",
            type=int,
            gradio_type="Slider",
            ge=0,
            le=48000,
            step=1,
        ),
        "volume_mix_rate": TypedInput(
            default=1,
            description="Mix ratio for volume envelope. 1=original input audio volume.",
            type=float,
            gradio_type="Slider",
            ge=0,
            le=1,
            step=0.01,
        ),
        "accent_strength": TypedInput(
            default=0.5,
            description="A stronger accent strength will make the voice sound more like the target speaker, but may also introduce artifacts.",
            type=float,
            gradio_type="Slider",
            ge=0,
            le=1.0,
            step=0.01,
        ),
        "filter_radius": TypedInput(
            default=5,
            description="Median filter radius for 'harvest' pitch recognition. (Higher values may help reduce auto-tune like artifacts.)",
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

    def process_audio(self, inputs: List[ProjectFiles], callback=None, **kwargs: Dict[str, Any]) -> List[ProjectFiles]:
        """
        Process audio inputs based on provided configurations.
        """
        self.setup()
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in self.allowed_kwargs}
        clone_bg_vocals = filtered_kwargs.get("clone_bg_vocals", False)
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
        # Only inputs with (Vocals) in the name will be processed
        outputs = []
        for project in inputs:
            last_outputs = project.last_outputs
            filtered_inputs = [input_path for input_path in last_outputs if "(Vocals)" in input_path]
            if not len(filtered_inputs):
                filtered_inputs = [project.src_file]
            if not clone_bg_vocals:
                # Remove background vocals
                filtered_inputs = [input_path for input_path in filtered_inputs if "(BG_Vocals)" not in input_path]
            clone_outputs = self.vc.vc_multi(
                model=selected_voice,
                sid=spk_id,
                paths=filtered_inputs,
                f0_up_key=pitch_shift,
                format1=format_,
                f0_method=f0method,
                index_rate=index_rate,
                filter_radius=filter_radius,
                resample_sr=resample_rate,
                rms_mix_rate=rms_mix_rate,
                protect=protect,
                project_dir=project.project_dir
            )
            project.add_output("cloned", clone_outputs)
            project.last_outputs = clone_outputs + [input_path for input_path in last_outputs if input_path not in filtered_inputs]
            outputs.append(project)

        return outputs

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
