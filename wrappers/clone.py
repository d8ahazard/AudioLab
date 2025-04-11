import logging
import os
from typing import Any, Dict, List, Optional

from handlers.config import model_path
from modules.rvc.configs.config import Config
from modules.rvc.infer.modules.vc.pipeline import VC
from util.data_classes import ProjectFiles
from wrappers.base_wrapper import BaseWrapper, TypedInput

logger = logging.getLogger(__name__)


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
            description="Clone background vocals in addition to the main vocals. (Be aware that layered harmonies may cause artifacts.)",
            type=bool,
            gradio_type="Checkbox",
        ),
        "pitch_shift": TypedInput(
            default=0,
            ge=-24,
            le=24,
            description="Pitch shift in semitones (+12 for an octave up, -12 for an octave down). Adjust this if the cloned voice is higher or lower than the input vocals.",
            type=int,
            gradio_type="Slider",
        ),
        "pitch_shift_vocals_only": TypedInput(
            default=False,
            description="Apply pitch shift to vocals only, not the entire track. Not recommended for most use cases.",
            type=bool,
            gradio_type="Checkbox",
        ),
        "clone_stereo": TypedInput(
            default=True,
            description="When enabled, side-channel (audio panning) information is cloned. This can improve the stereo effect but may not work in all cases.",
            type=bool,
            gradio_type="Checkbox"
        ),

        "pitch_correction": TypedInput(
            default=False,
            description="Apply pitch correction (Auto-Tune) to the cloned vocals. Try this if there are artifacts in the cloned output.",
            type=bool,
            gradio_type="Checkbox",
        ),
        "pitch_correction_humanize": TypedInput(
            default=0.95,
            description="How much to humanize the pitch correction. 0=robotic, 1=human-like.",
            type=float,
            gradio_type="Slider",
            ge=0,
            le=1,
            step=0.01,
        ),
        "stereo_processing": TypedInput(
            default="mono",
            choices=["none", "mono", "stereo"],
            description="Stereo processing mode. None creates a single mono track, 'mono' converts stereo to mono and attempts to reconstruct; 'stereo' clones stereo audio and re-creates a similar width.",
            type=str,
            gradio_type="Dropdown",
            render=False
        ),
        "speaker_id": TypedInput(
            default=0,
            description="The ID of the speaker to use if the model was trained on multiple speakers.",
            type=int,
            gradio_type="Number",
        ),
        "pitch_extraction_method": TypedInput(
            default="rmvpe+",
            description="Pitch extraction algorithm. 'harvest' offers smoothing; 'rmvpe' is more accurate; 'hybrid' combines methods.",
            type=str,
            choices=["hybrid", "pm", "harvest", "dio", "rmvpe", "rmvpe_onnx", "rmvpe+", "crepe", "crepe-tiny",
                     "mangio-crepe", "mangio-crepe-tiny"],
            gradio_type="Dropdown",
        ),
        "volume_mix_rate": TypedInput(
            default=0.9,
            description="Mix ratio for volume envelope. 1=original input volume; lower values blend with the new RMS shape.",
            type=float,
            gradio_type="Slider",
            ge=0,
            le=1,
            step=0.01,
        ),
        "accent_strength": TypedInput(
            default=0.2,
            description="A stronger accent strength makes the voice more like the target speaker, but can introduce artifacts.",
            type=float,
            gradio_type="Slider",
            ge=0,
            le=1.0,
            step=0.01,
        ),
        "filter_radius": TypedInput(
            default=3,
            description="Median filter radius for 'harvest' pitch recognition. Higher values reduce auto-tune artifacts but may lose detail.",
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
        ),
        "merge_type": TypedInput(
            default="median",
            description="Merge strategy for hybrid pitch extraction. 'median' computes the median of multiple estimates, while 'mean' takes the average.",
            type=str,
            choices=["median", "mean"],
            gradio_type="Dropdown",
        ),
        "crepe_hop_length": TypedInput(
            default=160,
            description="Hop length for CREPE-based pitch extraction. Lower values improve temporal resolution at the cost of speed.",
            type=int,
            gradio_type="Number",
        ),
        "f0_autotune": TypedInput(
            default=False,
            description="Automatically apply autotune to the extracted pitch values to smooth rapid variations.",
            type=bool,
            gradio_type="Checkbox",
            render=False
        ),
        "rmvpe_onnx": TypedInput(
            default=False,
            description="Use the ONNX version of the RMVPE model for pitch extraction if available.",
            type=bool,
            gradio_type="Checkbox",
            render=False
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
        clone_stereo = filtered_kwargs.get("clone_stereo", False)
        pitch_correction = filtered_kwargs.get("pitch_correction", False)
        pitch_correction_humanize = filtered_kwargs.get("pitch_correction_humanize", 0.5)
        merge_type = filtered_kwargs.get("merge_type", "median")
        crepe_hop_length = filtered_kwargs.get("crepe_hop_length", 160)
        f0_autotune = filtered_kwargs.get("f0_autotune", False)
        rmvpe_onnx = filtered_kwargs.get("rmvpe_onnx", False)

        outputs = []
        total_steps = len(inputs)
        if clone_bg_vocals:
            total_steps *= 2
        if pitch_correction:
            total_steps *= 2
        # For the callback
        self.vc.total_steps = total_steps
        try:
            for project in inputs:

                project_name = os.path.basename(project.project_dir)

                def project_callback(step, message, steps=total_steps):
                    if callback is not None:
                        callback(step, f"({project_name}) {message}", steps)

                last_outputs = project.last_outputs
                
                # Check if we need to handle input directly (no prior separation)
                if not last_outputs:
                    # Look for vocal files in the stems directory first
                    stems_dir = os.path.join(project.project_dir, "stems")
                    if os.path.exists(stems_dir):
                        vocal_files = [os.path.join(stems_dir, f) for f in os.listdir(stems_dir) 
                                      if "(Vocals)" in f and os.path.isfile(os.path.join(stems_dir, f))]
                        if vocal_files:
                            filtered_inputs = vocal_files
                        else:
                            # If no vocals files found, try to use the source file directly
                            logger.info(f"No vocal files found in stems directory - using source file directly: {project.src_file}")
                            filtered_inputs = [project.src_file]
                    else:
                        # If no stems directory, use the source file
                        logger.info(f"No stems directory found - using source file directly: {project.src_file}")
                        filtered_inputs = [project.src_file]
                else:
                    # Typically, we only clone from the path labeled "(Vocals)". If none, fallback to the src_file.
                    filtered_inputs = [p for p in last_outputs if "(Vocals)" in p or "(BG_Vocals" in p]
                    if not filtered_inputs:
                        filtered_inputs = [project.src_file]

                if not clone_bg_vocals:
                    # Exclude any "(BG_Vocals" if user doesn't want to clone them
                    filtered_inputs = [p for p in filtered_inputs if "(BG_Vocals" not in p]

                # Perform the voice conversion with the new pitch extraction parameters
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
                    merge_type=merge_type,
                    crepe_hop_length=crepe_hop_length,
                    f0_autotune=f0_autotune,
                    rmvpe_onnx=rmvpe_onnx,
                    clone_stereo=clone_stereo,
                    pitch_correction=pitch_correction,
                    pitch_correction_humanize=pitch_correction_humanize,
                    project_dir=project.project_dir,
                    callback=project_callback
                )
                # Store results
                project.add_output("cloned", clone_outputs)
                # Update the last_outputs so we don't lose references to unprocessed files
                project.last_outputs = clone_outputs + [p for p in last_outputs if p not in filtered_inputs]
                outputs.append(project)
        except Exception as e:
            logger.error(f"Error cloning vocals: {e}")
            if callback is not None:
                callback(1, f"Error: {e}")
            raise e

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

    def register_api_endpoint(self, api) -> Any:
        """
        Register FastAPI endpoint for voice cloning.
        
        Args:
            api: FastAPI application instance
            
        Returns:
            The registered endpoint route
        """
        from fastapi import File, UploadFile, HTTPException
        from fastapi.responses import FileResponse
        from pydantic import BaseModel, create_model
        from typing import List, Optional
        import tempfile
        from pathlib import Path

        # Create Pydantic model for settings
        fields = {}
        for key, value in self.allowed_kwargs.items():
            field_type = value.type
            if value.field.default == ...:
                field_type = Optional[field_type]
            fields[key] = (field_type, value.field)
        
        SettingsModel = create_model(f"{self.__class__.__name__}Settings", **fields)

        @api.post("/api/v1/process/clone")
        async def process_clone(
            files: List[UploadFile] = File(...),
            settings: Optional[SettingsModel] = None
        ):
            """
            Clone vocals using RVC voice models.
            
            This endpoint transforms vocal characteristics in audio files using pre-trained RVC voice models.
            It allows you to change the timbre and characteristics of vocals while preserving the original
            pitch, timing, and emotion of the performance.
            
            ## Parameters
            
            - **files**: Audio files containing vocals to be transformed (WAV, MP3, FLAC)
            - **settings**: Voice cloning settings with the following options:
              - **selected_voice** (required): Voice model name from available models (get list from `/api/v1/clone/voices`)
              - **pitch_shift**: Pitch shift in semitones, from -24 to +24 (default: 0)
              - **clone_bg_vocals**: Whether to also clone background vocals (default: false)
              - **clone_stereo**: Preserve stereo information when cloning (default: true)
              - **pitch_correction**: Apply pitch correction to cloned vocals (default: false)
              - **pitch_correction_humanize**: How human-like the pitch correction should be (0-1, default: 0.95)
              - **volume_mix_rate**: Mix ratio for volume envelope (0-1, default: 0.9)
              - **accent_strength**: How strongly to apply the target voice characteristics (0-1, default: 0.2)
              - **speaker_id**: ID of the speaker for multi-speaker models (default: 0)
              - **pitch_extraction_method**: Algorithm for pitch extraction (default: "rmvpe+")
                - Options: "hybrid", "pm", "harvest", "dio", "rmvpe", "rmvpe_onnx", "rmvpe+", "crepe", etc.
              - **filter_radius**: Median filter radius for 'harvest' pitch recognition (0-7, default: 3)
              - **index_rate**: Feature search proportion when using the vector index (0-1, default: 1)
              - **merge_type**: Merge strategy for hybrid pitch extraction ("median" or "mean", default: "median")
              - **crepe_hop_length**: Hop length for CREPE-based pitch extraction (default: 160)
            
            ## Example Request
            
            ```python
            import requests
            
            url = "http://localhost:7860/api/v1/process/clone"
            
            # Get available voices first
            voices_response = requests.get("http://localhost:7860/api/v1/clone/voices")
            voice_model = voices_response.json()["voices"][0]  # Use first available voice
            
            # Upload audio file
            files = [
                ('files', ('vocals.wav', open('vocals.wav', 'rb'), 'audio/wav'))
            ]
            
            # Configure voice cloning parameters
            data = {
                'selected_voice': voice_model,
                'pitch_shift': 0,
                'accent_strength': 0.3,
                'pitch_extraction_method': 'rmvpe+',
                'volume_mix_rate': 0.9,
                'clone_stereo': 'true'
            }
            
            # Send request
            response = requests.post(url, files=files, data=data)
            
            # Save the cloned audio
            with open('cloned_vocals.wav', 'wb') as f:
                f.write(response.content)
            ```
            
            ## Best Practices
            
            1. Use isolated vocals for best results - separate your audio first
            2. Start with default settings and adjust as needed
            3. For challenging vocals, try different pitch extraction methods
            4. Adjust accent_strength to control how strongly the target voice is applied
            5. Use pitch_shift if the output is too high or low
            
            ## Response
            
            The API returns the cloned audio file as an attachment.
            """
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Save uploaded files
                    input_files = []
                    for file in files:
                        file_path = Path(temp_dir) / file.filename
                        with file_path.open("wb") as f:
                            content = await file.read()
                            f.write(content)
                        input_files.append(ProjectFiles(str(file_path)))
                    
                    # Process files
                    settings_dict = settings.dict() if settings else {}
                    processed_files = self.process_audio(input_files, **settings_dict)
                    
                    # Return processed files
                    output_files = []
                    for project in processed_files:
                        for output in project.last_outputs:
                            output_path = Path(output)
                            if output_path.exists():
                                output_files.append(FileResponse(output))
                    
                    return output_files
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        # Also register the voice list endpoint
        @api.get("/api/v1/clone/voices")
        async def list_available_voices():
            """
            Get list of available voice models for the Clone processor.
            
            This endpoint returns a list of all available pre-trained voice models that can be used with the Clone processor.
            These voice models are used to transform the vocal characteristics of audio files.
            
            ## Example Request
            
            ```python
            import requests
            
            url = "http://localhost:7860/api/v1/clone/voices"
            response = requests.get(url)
            voices = response.json()["voices"]
            
            print(f"Available voice models: {voices}")
            
            # Use a voice model with the Clone processor
            clone_url = "http://localhost:7860/api/v1/process/clone"
            files = [('files', ('vocals.wav', open('vocals.wav', 'rb'), 'audio/wav'))]
            data = {'selected_voice': voices[0]}  # Use the first available voice
            
            response = requests.post(clone_url, files=files, data=data)
            ```
            
            ## Response Format
            
            ```json
            {
                "voices": [
                    "voice_model_1",
                    "voice_model_2",
                    "celebrity_voice_1",
                    "..."
                ]
            }
            ```
            
            The returned voice model names can be used directly with the Clone processor's `selected_voice` parameter.
            """
            return {"voices": list_speakers()}

        return process_clone
