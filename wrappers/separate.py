import hashlib
import json
import os
import shutil
import threading
from typing import Any, List, Dict, Optional
import logging

from handlers.config import output_path
from modules.separator.stem_separator import separate_music
from util.data_classes import ProjectFiles
from wrappers.base_wrapper import BaseWrapper, TypedInput

logger = logging.getLogger(__name__)


class Separate(BaseWrapper):
    """
    A slimmed‐down wrapper that simply passes the user's file and option settings
    to the main separation model.
    """
    title = "Separate"
    priority = 1
    default = True
    required = False
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
        "bg_vocal_layers": TypedInput(
            default=1,
            le=10,
            ge=1,
            description="Number of background vocal layers to separate.",
            type=int,
            gradio_type="Slider",
            render=False
        ),
        "vocals_only": TypedInput(
            default=True,
            description="Enable to separate only the main vocals and instrumental, disable for additional stems.",
            type=bool,
            gradio_type="Checkbox"
        ),
        "store_reverb_ir": TypedInput(
            default=True,
            description="Store the impulse response for reverb removal. Will be used to re-apply reverb later.",
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
            default="Main Vocals",
            description="Apply reverb removal.",
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
        """
        Register FastAPI endpoint for audio separation.
        
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

        @api.post("/api/v1/process/separate")
        async def process_separate(
            files: List[UploadFile] = File(...),
            settings: Optional[SettingsModel] = None
        ):
            """
            Separate audio files into stems.
            
            This endpoint splits audio tracks into individual components (stems) such as vocals, instruments, 
            and other elements. It uses advanced AI models to isolate different parts of a mix with high quality.
            It also provides options for removing reverb, crowd noise, and general noise from the separated stems.
            
            ## Parameters
            
            - **files**: Audio files to separate (WAV, MP3, FLAC)
            - **settings**: Separation settings with the following options:
              - **vocals_only**: Only extract vocals and instrumental stems (default: true)
              - **separate_bg_vocals**: Separate background vocals from main vocals (default: true)
              - **delete_extra_stems**: Delete intermediate stem files after processing (default: true)
              - **separate_drums**: Separate the drum track (default: false)
              - **separate_woodwinds**: Separate woodwind instruments (default: false)
              - **alt_bass_model**: Use alternative bass separation model (default: false)
              - **store_reverb_ir**: Store impulse response for later reverb restoration (default: true)
              - **reverb_removal**: Apply reverb removal to specified stems (default: "Main Vocals")
                - Options: "Nothing", "Main Vocals", "All Vocals", "All"
              - **crowd_removal**: Apply crowd noise removal to specified stems (default: "Nothing")
                - Options: "Nothing", "Main Vocals", "All Vocals", "All"
              - **noise_removal**: Apply general noise removal to specified stems (default: "Nothing")
                - Options: "Nothing", "Main Vocals", "All Vocals", "All"
              - **noise_removal_model**: Model to use for noise removal (default: "UVR-DeNoise.pth")
                - Options: "UVR-DeNoise.pth", "UVR-DeNoise-Lite.pth"
              - **crowd_removal_model**: Model to use for crowd noise removal (default: "UVR-MDX-NET_Crowd_HQ_1.onnx")
                - Options: "UVR-MDX-NET_Crowd_HQ_1.onnx", "mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt"
            
            ## Example Request
            
            ```python
            import requests
            
            url = "http://localhost:7860/api/v1/process/separate"
            
            # Upload audio file
            files = [
                ('files', ('song.mp3', open('song.mp3', 'rb'), 'audio/mpeg'))
            ]
            
            # Configure separation parameters
            data = {
                'vocals_only': 'true',
                'separate_bg_vocals': 'true',
                'reverb_removal': 'Main Vocals',
                'noise_removal': 'Main Vocals'
            }
            
            # Send request
            response = requests.post(url, files=files, data=data)
            
            # The API returns a list of files, so we save each one
            if response.headers['Content-Type'] == 'application/json':
                # Multiple files returned as JSON with file information
                files_info = response.json()
                for i, file_info in enumerate(files_info):
                    file_response = requests.get(file_info['url'])
                    with open(f'stem_{i}.wav', 'wb') as f:
                        f.write(file_response.content)
            else:
                # Single file returned directly
                with open('vocals.wav', 'wb') as f:
                    f.write(response.content)
            ```
            
            ## Typical Output Files
            
            When using the default settings, the API will typically generate these files:
            
            1. **Main Vocals**: The lead vocals isolated from the mix
            2. **Background Vocals**: Any backing vocals or harmonies (if separate_bg_vocals=true)
            3. **Instrumental**: All non-vocal elements of the track
            
            With vocals_only=false, additional stems may include:
            
            4. **Bass**: Bass instruments
            5. **Drums**: Percussion and drum elements (if separate_drums=true)
            6. **Woodwinds**: Flute, saxophone, etc. (if separate_woodwinds=true)
            7. **Other**: Remaining instruments not covered by other stems
            
            ## Response
            
            The API returns the separated audio files as attachments.
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
                    
                    # Return separated files
                    output_files = []
                    for project in processed_files:
                        for output in project.last_outputs:
                            output_path = Path(output)
                            if output_path.exists():
                                output_files.append(FileResponse(output))
                    
                    return output_files
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        return process_separate

    def process_audio(self, inputs: List[ProjectFiles], callback=None, **kwargs: Dict[str, any]) -> List[ProjectFiles]:
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in self.allowed_kwargs}
        final_projects = []
        to_separate = []  # Projects that need separation (no valid cache)

        # First pass: Check cache for each project and handle special cases.
        for project in inputs:
            base_name = os.path.splitext(os.path.basename(project.src_file))[0]
            # Store base_name in project for mapping consistency
            project.base_name = base_name
            out_dir = os.path.join(project.project_dir, "stems")
            os.makedirs(out_dir, exist_ok=True)
            cache_file = os.path.join(out_dir, "separation_info.json")

            # Check if this is a special file that should skip separation:
            # 1. Files starting with TTS_ or ZONOS_
            # 2. Files from tts, zonos, or stable_audio directories
            file_basename = os.path.basename(project.src_file)
            file_dir = os.path.dirname(project.src_file)
            special_dirs = ["tts", "zonos", "stable_audio"]
            
            is_special_file = (
                file_basename.startswith("TTS_") or 
                file_basename.startswith("ZONOS_") or
                any(special_dir in file_dir for special_dir in special_dirs)
            )
            
            if is_special_file:
                # Handle like TTS files - copy to stems with (Vocals) suffix
                stem_dir = os.path.join(project.project_dir, "stems")
                base_name, ext = os.path.splitext(os.path.basename(project.src_file))
                new_name = f"{base_name}(Vocals){ext}"
                new_path = os.path.join(stem_dir, new_name)
                if not os.path.exists(new_path):
                    shutil.copyfile(project.src_file, new_path)
                project_stems = [new_path]
                project.add_output("stems", project_stems)
                final_projects.append(project)
                logger.info(f"Skipping separation for special file {project.src_file}")
                continue

            current_config = {
                "file": project.src_file,
                "vocals_only": filtered_kwargs.get("vocals_only", True),
                "separate_drums": filtered_kwargs.get("separate_drums", False),
                "separate_woodwinds": filtered_kwargs.get("separate_woodwinds", False),
                "alt_bass_model": filtered_kwargs.get("alt_bass_model", False),
                "separate_bg_vocals": filtered_kwargs.get("separate_bg_vocals", True),
                "bg_vocal_layers": filtered_kwargs.get("bg_vocal_layers", 1),
                "reverb_removal": filtered_kwargs.get("reverb_removal", "Nothing"),
                "echo_removal": filtered_kwargs.get("echo_removal", "Nothing"),
                "delay_removal": filtered_kwargs.get("delay_removal", "Nothing"),
                "crowd_removal": filtered_kwargs.get("crowd_removal", "Nothing"),
                "noise_removal": filtered_kwargs.get("noise_removal", "Nothing"),
                "delay_removal_model": filtered_kwargs.get("delay_removal_model", "UVR-DeEcho-DeReverb.pth"),
                "noise_removal_model": filtered_kwargs.get("noise_removal_model", "UVR-DeNoise.pth"),
                "crowd_removal_model": filtered_kwargs.get("crowd_removal_model", "UVR-MDX-NET_Crowd_HQ_1.onnx"),
                "store_reverb_ir": filtered_kwargs.get("store_reverb_ir", True)
            }

            valid_cache = False
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, "r") as f:
                        cached_data = json.load(f)
                    if cached_data.get("config") == current_config:
                        output_stems = []
                        all_stems_good = True
                        for stem_info in cached_data.get("stems", []):
                            path = stem_info.get("path")
                            digest = stem_info.get("hash")
                            if not os.path.exists(path) or self._hash_file(path) != digest:
                                all_stems_good = False
                                break
                            output_stems.append(path)
                        if all_stems_good:
                            project.add_output("stems", output_stems)
                            final_projects.append(project)
                            valid_cache = True
                except Exception as e:
                    logger.warning(f"Error reading cache file {cache_file}: {e}")
            if not valid_cache:
                to_separate.append((project, current_config))

        # Second pass: Process all projects needing separation together.
        if to_separate:
            # Gather input files and map them by base name.
            input_files = []
            input_dict = {}
            project_map = {}  # base_name -> (project, current_config)
            for proj, config in to_separate:
                stem_dir = os.path.join(proj.project_dir, "stems")
                os.makedirs(stem_dir, exist_ok=True)
                
                # TTS file handling has been moved to the first pass
                
                if stem_dir not in input_dict:
                    input_dict[stem_dir] = []
                input_dict[stem_dir].append(proj.src_file)
                input_files.append(proj.src_file)
                base_name = os.path.basename(proj.project_dir)
                project_map[base_name] = (proj, config)

            # Call separate_music once for all input files.
            combined_stems = separate_music(
                input_dict=input_dict,
                callback=callback,
                **filtered_kwargs
            )

            # Sort the outputs by base name. Using '__' as delimiter.
            separation_results = {}  # base_name -> list of stem files
            for stem in combined_stems:
                folder_parts = os.path.dirname(stem).split(os.path.sep)
                output_folder_parts = os.path.join(output_path, "process").split(os.path.sep)
                # Remove output_folder_parts from folder_parts
                folder_parts = [part for part in folder_parts if part not in output_folder_parts]
                base = folder_parts[0]
                separation_results.setdefault(base, []).append(stem)

            # For each project, move its outputs to its own stems folder and update cache.
            for base, (proj, config) in project_map.items():
                # TTS file handling has been moved to the first pass
                
                if base not in separation_results:
                    logger.warning(f"No separation results found for project {proj.src_file}")
                    continue
                project_stems = separation_results[base]
                proj.add_output("stems", project_stems)
                final_projects.append(proj)
                out_dir = os.path.join(proj.project_dir, "stems")
                cache_file = os.path.join(out_dir, "separation_info.json")
                cache_info = {"config": config, "stems": []}
                for p in project_stems:
                    hash_val = self._hash_file(p)
                    cache_info["stems"].append({"path": p, "hash": hash_val})
                try:
                    with open(cache_file, "w") as f:
                        json.dump(cache_info, f, indent=2)
                except Exception as e:
                    logger.warning(f"Error writing cache file {cache_file}: {e}")

        # Optionally delete extra stems if requested.
        if filtered_kwargs.get("delete_extra_stems", True):
            # For each project, delete any file in the stems folder that is not part of the final outputs
            for project in final_projects:
                out_dir = os.path.join(project.project_dir, "stems")
                final_stems = project.file_dict.get("stems", [])
                for fname in os.listdir(out_dir):
                    full_path = os.path.join(out_dir, fname)
                    if fname == "separation_info.json" or fname == "impulse_response.ir":
                        continue
                    if full_path not in final_stems:
                        self.del_stem(full_path)

        return final_projects

    def del_stem(self, path: str) -> bool:
        try:
            with self.file_operation_lock:
                if os.path.exists(path):
                    os.remove(path)
                    return True
        except Exception as e:
            print(f"Error deleting {path}: {e}")
        return False

    def _hash_file(self, filepath: str) -> str:
        """
        Compute a SHA-256 hash of the file contents.
        Useful for verifying the file hasn't changed between runs.
        """
        sha256_hash = hashlib.sha256()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    sha256_hash.update(chunk)
        except Exception as e:
            logger.warning(f"Error hashing file {filepath}: {e}")
        return sha256_hash.hexdigest()
