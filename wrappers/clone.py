import logging
import os
from typing import Any, Dict, List, Optional
import time

from handlers.config import model_path
from modules.rvc.configs.config import Config
from modules.rvc.infer.modules.vc.pipeline import VC
from modules.cloning import main as cloning
from util.data_classes import ProjectFiles
from wrappers.base_wrapper import BaseWrapper, TypedInput
import gradio as gr

logger = logging.getLogger(__name__)


def list_speakers():
    """
    Scan the model_path/trained directory and return all .pth files (trained voice checkpoints).
    """
    speaker_dir = os.path.join(model_path, "trained")
    os.makedirs(speaker_dir, exist_ok=True)
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
        "Clone vocals from one audio file to another using voice cloning models."
    )

    # Detect all speaker .pth files
    all_speakers = []
    first_speaker = None
    try:
        all_speakers = list_speakers()
        first_speaker = all_speakers[0] if all_speakers else None
    except Exception as e:
        logger.warning(f"Could not list speakers: {e}")
        print(f"Could not list speakers: {e}")

    allowed_kwargs = {
        "clone_method": TypedInput(
            default="RVC",
            description="The voice cloning method to use.",
            choices=["RVC", "OpenVoice", "TTS"],
            type=str,
            gradio_type="Dropdown",
            required=True
        ),
        # RVC-specific controls group
        "selected_voice": TypedInput(
            default=first_speaker,
            description="The voice model to use for RVC cloning.",
            choices=all_speakers,
            type=str,
            gradio_type="Dropdown",
            refresh=list_speakers_ui,
            required=False,
            render=True,
            group_name="RVC Controls"
        ),
        "pitch_shift": TypedInput(
            default=0,
            ge=-24,
            le=24,
            description="Pitch shift in semitones (+12 for an octave up, -12 for an octave down).",
            type=int,
            gradio_type="Slider",
            render=True,
            group_name="RVC Controls"
        ),
        "pitch_correction": TypedInput(
            default=False,
            description="Apply pitch correction (Auto-Tune) to the cloned vocals.",
            type=bool,
            gradio_type="Checkbox",
            render=True,
            group_name="RVC Controls"
        ),
        "pitch_correction_humanize": TypedInput(
            default=0.95,
            description="How much to humanize the pitch correction. 0=robotic, 1=human-like.",
            type=float,
            gradio_type="Slider",
            ge=0,
            le=1,
            step=0.01,
            render=True,
            group_name="RVC Controls"
        ),
        "clone_stereo": TypedInput(
            default=True,
            description="Preserve stereo information when cloning.",
            type=bool,
            gradio_type="Checkbox",
            render=True,
            group_name="RVC Controls"
        ),
        
        # OpenVoice/TTS shared controls
        "source_speaker": TypedInput(
            default=None,
            description="Reference audio file for voice cloning (for OpenVoice and TTS).",
            type=str,
            gradio_type="File",
            required=False,
            render=True,
            group_name="OpenVoice/TTS Controls"
        ),
        
        # OpenVoice specific controls
        "voice_strength": TypedInput(
            default=0.5,
            ge=0.0,
            le=1.0,
            description="Strength of voice characteristics to apply in OpenVoice cloning.",
            type=float,
            gradio_type="Slider",
            step=0.01,
            render=True,
            group_name="OpenVoice Controls"
        ),
        "custom_text": TypedInput(
            default="",
            description="Optional custom text for TTS voice cloning. If empty, text will be extracted from input audio.",
            type=str,
            gradio_type="Textbox",
            render=True,
            group_name="OpenVoice Controls"
        ),
        
        # Common controls
        "clone_bg_vocals": TypedInput(
            default=False,
            description="Clone background vocals in addition to the main vocals.",
            type=bool,
            gradio_type="Checkbox",
            render=True,
            group_name="Common Options"
        ),
        "diarize_speakers": TypedInput(
            default=False,
            description="Detect and separate multiple speakers in the audio before cloning.",
            type=bool,
            gradio_type="Checkbox",
            render=True,
            group_name="Common Options"
        ),
        "speaker_index": TypedInput(
            default=0,
            description="When diarization is enabled, which speaker to clone (0 is the first speaker).",
            type=int,
            gradio_type="Number",
            ge=0,
            render=True,
            group_name="Common Options"
        ),
        
        # Advanced RVC options
        "pitch_extraction_method": TypedInput(
            default="rmvpe+",
            description="Pitch extraction algorithm for RVC.",
            type=str,
            choices=["hybrid", "pm", "harvest", "dio", "rmvpe", "rmvpe_onnx", "rmvpe+", "crepe", "crepe-tiny",
                     "mangio-crepe", "mangio-crepe-tiny"],
            gradio_type="Dropdown",
            render=True,
            group_name="Advanced RVC Options"
        ),
        "volume_mix_rate": TypedInput(
            default=0.9,
            description="Mix ratio for volume envelope. 1=original input volume.",
            type=float,
            gradio_type="Slider",
            ge=0,
            le=1,
            step=0.01,
            render=True,
            group_name="Advanced RVC Options"
        ),
        "accent_strength": TypedInput(
            default=0.2,
            description="Strength of target voice characteristics (higher can introduce artifacts).",
            type=float,
            gradio_type="Slider",
            ge=0,
            le=1.0,
            step=0.01,
            render=True,
            group_name="Advanced RVC Options"
        ),
        "filter_radius": TypedInput(
            default=3,
            description="Median filter radius for 'harvest' pitch recognition.",
            type=int,
            gradio_type="Slider",
            ge=0,
            le=7,
            step=1,
            render=True,
            group_name="Advanced RVC Options"
        ),
        "index_rate": TypedInput(
            default=1,
            description="Feature search proportion when using the vector index. 0=disable, 1=full usage.",
            type=float,
            gradio_type="Slider",
            ge=0,
            le=1,
            step=0.01,
            render=True,
            group_name="Advanced RVC Options"
        ),
        "merge_type": TypedInput(
            default="median",
            description="Merge strategy for hybrid pitch extraction.",
            type=str,
            choices=["median", "mean"],
            gradio_type="Dropdown",
            render=True,
            group_name="Advanced RVC Options"
        ),
        "crepe_hop_length": TypedInput(
            default=160,
            description="Hop length for CREPE-based pitch extraction.",
            type=int,
            gradio_type="Number",
            render=True,
            group_name="Advanced RVC Options"
        ),
        "f0_autotune": TypedInput(
            default=False,
            description="Automatically apply autotune to extracted pitch values.",
            type=bool,
            gradio_type="Checkbox",
            render=True,
            group_name="Advanced RVC Options"
        ),
        "rmvpe_onnx": TypedInput(
            default=False,
            description="Use the ONNX version of the RMVPE model for pitch extraction if available.",
            type=bool,
            gradio_type="Checkbox",
            render=True,
            group_name="Advanced RVC Options"
        )
    }

    def process_audio(self, inputs: List[ProjectFiles], callback=None, **kwargs: Dict[str, Any]) -> List[ProjectFiles]:
        """
        Process one or more audio input(s) using the provided configurations.
        This method:
          1. Grabs config arguments from kwargs.
          2. Identifies target vocal paths (e.g., main vocals or background vocals).
          3. Calls the appropriate cloning method based on user selection.
          4. Appends the cloned audio output to project outputs.
        """
        # Filter out unexpected kwargs
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in self.allowed_kwargs}

        # Extract relevant configs
        clone_method = filtered_kwargs.get("clone_method", "RVC")
        clone_bg_vocals = filtered_kwargs.get("clone_bg_vocals", False)
        diarize_speakers = filtered_kwargs.get("diarize_speakers", False)
        speaker_index = filtered_kwargs.get("speaker_index", 0)
        source_speaker = filtered_kwargs.get("source_speaker", None)
        voice_strength = filtered_kwargs.get("voice_strength", 0.5)
        custom_text = filtered_kwargs.get("custom_text", "")
        
        # Use empty string as None for custom text
        if custom_text == "":
            custom_text = None

        # RVC-specific configs
        if clone_method == "RVC":
            # Ensure RVC is set up
            config = Config()
            self.vc = VC(config, True)
            
            selected_voice = filtered_kwargs.get("selected_voice", "")
            speaker_dir = os.path.join(model_path, "trained")
            selected_voice = os.path.join(speaker_dir, f"{selected_voice}.pth")
            if not os.path.exists(selected_voice):
                if callback is not None:
                    callback(0, "Selected voice model not found.")
                return []
                
            spk_id = filtered_kwargs.get("speaker_id", 0)
            f0method = filtered_kwargs.get("pitch_extraction_method", "rmvpe+")
            rms_mix_rate = filtered_kwargs.get("volume_mix_rate", 0.9)
            protect = filtered_kwargs.get("accent_strength", 0.2)
            index_rate = filtered_kwargs.get("index_rate", 1)
            filter_radius = filtered_kwargs.get("filter_radius", 3)
            clone_stereo = filtered_kwargs.get("clone_stereo", True)
            pitch_correction = filtered_kwargs.get("pitch_correction", False)
            pitch_correction_humanize = filtered_kwargs.get("pitch_correction_humanize", 0.95)
            merge_type = filtered_kwargs.get("merge_type", "median")
            crepe_hop_length = filtered_kwargs.get("crepe_hop_length", 160)
            f0_autotune = filtered_kwargs.get("f0_autotune", False)
            rmvpe_onnx = filtered_kwargs.get("rmvpe_onnx", False)

            total_steps = len(inputs)
            if clone_bg_vocals:
                total_steps *= 2
            if pitch_correction:
                total_steps *= 2
                
            # For RVC callback
            if self.vc:
                self.vc.total_steps = total_steps

        outputs = []
        try:
            for project in inputs:
                project_name = os.path.basename(project.project_dir)

                def project_callback(step, message, steps=total_steps if 'total_steps' in locals() else 1):
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

                clone_outputs = []
                
                # Process each input file
                for input_file in filtered_inputs:
                    if callback is not None:
                        callback(0, f"Processing {os.path.basename(input_file)}")
                
                    # Check if we need to perform speaker diarization
                    processed_inputs = [input_file]
                    if diarize_speakers:
                        if callback is not None:
                            callback(0.1, f"Detecting speakers in {os.path.basename(input_file)}")
                            
                        # Create a directory for speaker files
                        speakers_dir = os.path.join(project.project_dir, "speakers")
                        os.makedirs(speakers_dir, exist_ok=True)
                        
                        # Get one specific speaker
                        speaker_file = cloning.choose_speaker(input_file, speakers_dir, speaker_index)
                        if speaker_file:
                            processed_inputs = [speaker_file]
                            if callback is not None:
                                callback(0.2, f"Selected speaker {speaker_index}")
                        else:
                            logger.warning(f"Failed to separate speakers in {input_file}, using the original audio")
                
                    # Process each file with the appropriate cloning method
                    for proc_file in processed_inputs:
                        if clone_method == "RVC":
                            # Use RVC for cloning
                            if callback is not None:
                                callback(0.3, f"Cloning with RVC: {os.path.basename(proc_file)}")
                                
                            # Perform the voice conversion with RVC
                            file_outputs = self.vc.vc_multi(
                                model=selected_voice,
                                sid=spk_id,
                                paths=[proc_file],
                                f0_up_key=filtered_kwargs["pitch_shift"],
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
                            clone_outputs.extend(file_outputs)
                            
                        elif clone_method == "OpenVoice":
                            # Use OpenVoice for cloning
                            if callback is not None:
                                callback(0.3, f"Cloning with OpenVoice: {os.path.basename(proc_file)}")
                                
                            # Ensure source speaker file exists
                            if not source_speaker:
                                if callback is not None:
                                    callback(0.4, f"No reference audio provided for OpenVoice cloning")
                                continue
                                
                            if not os.path.exists(source_speaker):
                                if callback is not None:
                                    callback(0.4, f"Source speaker file not found: {source_speaker}")
                                continue
                                
                            # Perform the cloning
                            output_file = cloning.clone_voice_openvoice(
                                target_file=proc_file,
                                source_file=source_speaker,
                                output_dir=os.path.join(project.project_dir, "cloned"),
                                strength=voice_strength
                            )
                            
                            if output_file and os.path.exists(output_file):
                                clone_outputs.append(output_file)
                                if callback is not None:
                                    callback(1.0, f"Cloning complete: {os.path.basename(output_file)}")
                            else:
                                if callback is not None:
                                    callback(1.0, f"Cloning failed for {os.path.basename(proc_file)}")
                        
                        elif clone_method == "TTS":
                            # Use TTS for cloning
                            if callback is not None:
                                callback(0.3, f"Cloning with TTS: {os.path.basename(proc_file)}")
                                
                            # Ensure source speaker file exists
                            if not source_speaker:
                                if callback is not None:
                                    callback(0.4, f"No reference audio provided for TTS cloning")
                                continue
                                
                            if not os.path.exists(source_speaker):
                                if callback is not None:
                                    callback(0.4, f"Source speaker file not found: {source_speaker}")
                                continue
                                
                            # Perform the cloning
                            output_file = cloning.clone_voice_tts(
                                target_file=proc_file,
                                source_file=source_speaker,
                                output_dir=os.path.join(project.project_dir, "cloned"),
                                custom_text=custom_text
                            )
                            
                            if output_file and os.path.exists(output_file):
                                clone_outputs.append(output_file)
                                if callback is not None:
                                    callback(1.0, f"Cloning complete: {os.path.basename(output_file)}")
                            else:
                                if callback is not None:
                                    callback(1.0, f"Cloning failed for {os.path.basename(proc_file)}")
                
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
        finally:
            # Clean up any resources
            if clone_method != "RVC":
                cloning.cleanup()

        return outputs

    @staticmethod
    def change_choices() -> Dict[str, Any]:
        """
        Refresh the available voice models by scanning the 'trained' folder.
        """
        return {"choices": list_speakers(), "__type__": "update"}

    @staticmethod
    def clean():
        """
        Clean and reset states for the UI.
        """
        return {"value": "", "__type__": "update"}

    def register_api_endpoint(self, api) -> Any:
        """
        Register FastAPI endpoint for audio cloning.
        
        Args:
            api: FastAPI application instance
            
        Returns:
            The registered endpoint route
        """
        from fastapi import Body
        
        # Create models for JSON API
        FileData, JsonRequest = self.create_json_models()

        @api.post("/api/v1/process/clone", tags=["Audio Processing"])
        async def process_clone_json(
            request: JsonRequest = Body(...)
        ):
            """
            Clone vocals using voice models.
            
            This endpoint transforms vocal characteristics in audio files using various
            voice cloning methods (RVC, OpenVoice, or TTS).
            
            ## Request Body
            
            ```json
            {
              "files": [
                {
                  "filename": "vocals.wav",
                  "content": "base64_encoded_file_content..."
                }
              ],
              "settings": {
                "clone_method": "RVC",
                "selected_voice": "my_voice_model",
                "pitch_shift": 0,
                "clone_bg_vocals": false,
                "clone_stereo": true,
                "pitch_correction": false,
                "diarize_speakers": false,
                "speaker_index": 0
              }
            }
            ```
            
            ## Parameters
            
            - **files**: Array of file objects, each containing:
              - **filename**: Name of the file (with extension)
              - **content**: Base64-encoded file content
            - **settings**: Voice cloning settings with various options depending on the chosen method
              - For RVC: selected_voice, pitch_shift, etc.
              - For OpenVoice: source_speaker, voice_strength
              - For TTS: source_speaker, custom_text
            
            ## Response
            
            ```json
            {
              "files": [
                {
                  "filename": "cloned_vocals.wav",
                  "content": "base64_encoded_file_content..."
                }
              ]
            }
            ```
            
            The API returns an object containing the cloned audio file as a Base64-encoded string.
            """
            # Use the handle_json_request helper from BaseWrapper
            return self.handle_json_request(request, self.process_audio)

        @api.get("/api/v1/clone/voices", tags=["Audio Processing"])
        async def list_available_voices():
            """
            List available voice models for cloning.
            
            Returns a list of all available RVC voice models that can be used with the clone endpoint.
            Use these voice model names in the 'selected_voice' parameter of the clone request.
            
            ## Response
            
            ```json
            {
              "voices": [
                "Voice_Model_1",
                "Voice_Model_2",
                "Voice_Model_3"
              ]
            }
            ```
            """
            return {"voices": list_speakers()}

        @api.get("/api/v1/clone/methods", tags=["Audio Processing"])
        async def list_clone_methods():
            """
            List available voice cloning methods.
            
            Returns information about the available voice cloning methods.
            
            ## Response
            
            ```json
            {
              "methods": [
                {
                  "id": "RVC",
                  "name": "RVC",
                  "description": "Retrieval-based Voice Conversion using pre-trained models"
                },
                {
                  "id": "OpenVoice",
                  "name": "OpenVoice",
                  "description": "Zero-shot voice conversion using reference audio"
                },
                {
                  "id": "TTS",
                  "name": "TTS",
                  "description": "Text-to-speech voice cloning using reference audio"
                }
              ]
            }
            ```
            """
            methods = [
                {
                    "id": "RVC",
                    "name": "RVC",
                    "description": "Retrieval-based Voice Conversion using pre-trained models"
                },
                {
                    "id": "OpenVoice",
                    "name": "OpenVoice",
                    "description": "Zero-shot voice conversion using reference audio"
                },
                {
                    "id": "TTS",
                    "name": "TTS",
                    "description": "Text-to-speech voice cloning using reference audio"
                }
            ]
            return {"methods": methods}

        return [process_clone_json, list_available_voices, list_clone_methods]
