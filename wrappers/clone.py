import os
from typing import Any, Dict, List

from handlers.config import model_path
from modules.rvc.configs.config import Config
from modules.rvc.infer.modules.vc.pipeline import VC
from util.data_classes import ProjectFiles
from wrappers.base_wrapper import BaseWrapper, TypedInput


def list_speakers():
    """
    Scan the model_path/trained directory and return all .pth files (trained voice checkpoints).
    """
    speaker_dir = os.path.join(model_path, "trained")
    models = [f for f in os.listdir(speaker_dir) if f.endswith(".pth")]
    model_names = [os.path.splitext(f)[0] for f in models]
    return model_names


def list_speakers_ui():
    """
    Return a dictionary suitable for UI updates,
    containing the speaker checkpoint paths found by list_speakers().
    """
    return {"choices": list_speakers(), "__type__": "update"}


class Clone(BaseWrapper):
    """
    Clone vocals from one audio file to another using a pre-trained RVC voice model.
    """

    title = "Clone"
    priority = 2
    default = True
    vc = None
    description = (
        "Clone vocals from one audio file to another using a pre-trained RVC voice model."
    )

    # Detect all speaker .pth files
    all_speakers = list_speakers()
    first_speaker = all_speakers[0] if all_speakers else None

    # Allowed kwargs define the inputs accepted by 'process_audio'
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
            description="Clone background vocals in addition to the main vocals. "
                        "(Be aware that layered harmonies may cause artifacts.)",
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
            description="Pitch shift in semitones (+12 for an octave up, -12 for an octave down).",
            type=int,
            gradio_type="Number",
        ),
        "pitch_extraction_method": TypedInput(
            default="rmvpe",
            description="Pitch extraction algorithm. 'harvest' allows more features (e.g. smoothing).",
            type=str,
            choices=["pm", "harvest", "crepe", "rmvpe"],
            gradio_type="Dropdown",
        ),
        "volume_mix_rate": TypedInput(
            default=1,
            description="Mix ratio for volume envelope. 1=original input audio volume; "
                        "lower values blend with the new RMS shape.",
            type=float,
            gradio_type="Slider",
            ge=0,
            le=1,
            step=0.01,
        ),
        "accent_strength": TypedInput(
            default=0.25,
            description="A stronger accent strength makes the voice more like the target speaker, "
                        "but can introduce artifacts.",
            type=float,
            gradio_type="Slider",
            ge=0,
            le=1.0,
            step=0.01,
        ),
        "filter_radius": TypedInput(
            default=5,
            description="Median filter radius for 'harvest' pitch recognition. "
                        "Higher values reduce 'auto-tune' artifacts but may lose detail.",
            type=int,
            gradio_type="Slider",
            ge=0,
            le=7,
            step=1,
        ),
        "index_rate": TypedInput(
            default=1,
            description="Feature search proportion when using the vector index. 0=disable, 1=full usage.",
            type=float,
            gradio_type="Slider",
            ge=0,
            le=1,
            step=0.01,
        )
    }

    def process_audio(self, inputs: List[ProjectFiles], callback=None, **kwargs: Dict[str, Any]) -> List[ProjectFiles]:
        """
        Process one or more audio input(s) using the provided configurations.
        This method:
          1. Grabs RVC config arguments from kwargs.
          2. Identifies target vocal paths (e.g., main vocals or background vocals).
          3. Calls self.vc.vc_multi(...) to clone the vocals.
          4. Appends the cloned audio output to project outputs.
        """
        # Ensure VC is set up
        config = Config()
        self.vc = VC(config, True)

        # Filter out unexpected kwargs
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in self.allowed_kwargs}

        # Extract relevant configs
        clone_bg_vocals = filtered_kwargs.get("clone_bg_vocals", False)
        selected_voice = filtered_kwargs.get("selected_voice", "")
        speaker_dir = os.path.join(model_path, "trained")
        selected_voice = os.path.join(speaker_dir, f"{selected_voice}.pth")
        if not os.path.exists(selected_voice):
            if callback is not None:
                callback(0, "Selected voice model not found.")
            return []
        spk_id = filtered_kwargs.get("speaker_id", 0)
        pitch_shift = filtered_kwargs.get("pitch_shift", 0)
        f0method = filtered_kwargs.get("pitch_extraction_method", "rvmpe")
        rms_mix_rate = filtered_kwargs.get("volume_mix_rate", 1)
        protect = filtered_kwargs.get("accent_strength", 0.2)
        index_rate = filtered_kwargs.get("index_rate", 1)
        filter_radius = filtered_kwargs.get("filter_radius", 5)

        outputs = []
        total_steps = len(inputs)
        if clone_bg_vocals:
            total_steps *= 2
        # For the callback
        self.vc.total_steps = total_steps

        for project in inputs:

            project_name = os.path.basename(project.project_dir)

            def project_callback(step, message, steps=total_steps):
                if callback is not None:
                    callback(step, f"({project_name}) {message}", steps)

            last_outputs = project.last_outputs
            # Typically, we only clone from the path labeled "(Vocals)". If none, fallback to the src_file.
            filtered_inputs = [p for p in last_outputs if "(Vocals)" in p or "(BG_Vocals)" in p]
            if not filtered_inputs:
                filtered_inputs = [project.src_file]

            if not clone_bg_vocals:
                # Exclude any "(BG_Vocals)" if user doesn't want to clone them
                filtered_inputs = [p for p in filtered_inputs if "(BG_Vocals)" not in p]

            # Perform the voice conversion
            clone_outputs = self.vc.vc_multi(
                model=selected_voice,
                sid=spk_id,
                paths=filtered_inputs,
                f0_up_key=pitch_shift,
                f0_method=f0method,
                index_rate=index_rate,
                filter_radius=filter_radius,
                rms_mix_rate=rms_mix_rate,
                protect=protect,
                project_dir=project.project_dir,
                callback=project_callback
            )
            # Append (selected_voice) and (pitch_extraction_method) to the output file name
            for output in clone_outputs:
                base_name, ext = os.path.splitext(os.path.basename(output))
                selected_voice_base_name, _ = os.path.splitext(os.path.basename(selected_voice))
                new_name = os.path.join(os.path.dirname(output), f"{base_name}({selected_voice_base_name}_{f0method}){ext}")
                if os.path.exists(new_name):
                    os.remove(new_name)
                os.rename(output, new_name)
                clone_outputs[clone_outputs.index(output)] = new_name
            # Store results
            project.add_output("cloned", clone_outputs)
            # Update the last_outputs so we don't lose references to unprocessed files
            project.last_outputs = clone_outputs + [p for p in last_outputs if p not in filtered_inputs]
            outputs.append(project)

        return outputs

    @staticmethod
    def change_choices() -> Dict[str, Any]:
        """
        Refresh the available voice models by scanning the 'cloned' folder.
        """
        weight_root = os.path.join(os.getenv("model_path", "models"), "cloned")
        voices = [name for name in os.listdir(weight_root) if name.endswith(".pth")]
        return {"choices": sorted(voices), "__type__": "update"}

    @staticmethod
    def clean():
        """
        Clean and reset states for the UI.
        """
        return {"value": "", "__type__": "update"}

    def register_api_endpoint(self, api):
        """
        Register the API endpoints for external usage.
        """
        api.add_route("/convert", self.process_audio, methods=["POST"])
        api.add_route("/change_choices", self.change_choices, methods=["GET"])
