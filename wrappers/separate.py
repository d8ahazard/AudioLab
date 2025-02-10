import hashlib
import json
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
        final_projects = []
        to_separate = []  # Projects that need separation (no valid cache)

        # First pass: Check cache for each project.
        for project in inputs:
            base_name = os.path.splitext(os.path.basename(project.src_file))[0]
            # Store base_name in project for mapping consistency
            project.base_name = base_name
            out_dir = os.path.join(project.project_dir, "stems")
            os.makedirs(out_dir, exist_ok=True)
            cache_file = os.path.join(out_dir, "separation_info.json")

            current_config = {
                "file": project.src_file,
                "vocals_only": filtered_kwargs.get("vocals_only", True),
                "separate_drums": filtered_kwargs.get("separate_drums", False),
                "separate_woodwinds": filtered_kwargs.get("separate_woodwinds", False),
                "alt_bass_model": filtered_kwargs.get("alt_bass_model", False),
                "separate_bg_vocals": filtered_kwargs.get("separate_bg_vocals", True),
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
            # Create a temporary combined output folder.
            temp_out_dir = os.path.join(os.path.dirname(to_separate[0][0].project_dir), "combined_separation_temp")
            os.makedirs(temp_out_dir, exist_ok=True)

            # Gather input files and map them by base name.
            input_files = []
            project_map = {}  # base_name -> (project, current_config)
            for proj, config in to_separate:
                input_files.append(proj.src_file)
                base_name = os.path.basename(proj.project_dir)
                project_map[base_name] = (proj, config)

            # Call separate_music once for all input files.
            combined_stems = separate_music(
                input_audio=input_files,
                output_folder=temp_out_dir,
                callback=callback,
                **filtered_kwargs
            )

            # Sort the outputs by base name. Using '__' as delimiter.
            separation_results = {}  # base_name -> list of stem files
            for stem in combined_stems:
                filename = os.path.basename(stem)
                base = filename.split("__")[0]
                separation_results.setdefault(base, []).append(stem)

            # For each project, move its outputs to its own stems folder and update cache.
            for base, (proj, config) in project_map.items():
                if base not in separation_results:
                    logger.warning(f"No separation results found for project {proj.src_file}")
                    continue
                out_dir = os.path.join(proj.project_dir, "stems")
                os.makedirs(out_dir, exist_ok=True)
                project_stems = []
                for stem in separation_results[base]:
                    new_path = os.path.join(out_dir, os.path.basename(stem))
                    try:
                        if os.path.exists(new_path):
                            os.remove(new_path)
                        os.rename(stem, new_path)
                    except Exception as e:
                        logger.warning(f"Error moving {stem} to {new_path}: {e}")
                        new_path = stem  # fallback if move fails
                    project_stems.append(new_path)
                proj.add_output("stems", project_stems)
                final_projects.append(proj)
                # Write cache.
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

            # Optionally remove the temporary directory if empty.
            try:
                os.rmdir(temp_out_dir)
            except Exception:
                pass

        # Optionally delete extra stems if requested.
        if filtered_kwargs.get("delete_extra_stems", True):
            # For each project, delete any file in the stems folder that is not part of the final outputs
            for project in final_projects:
                out_dir = os.path.join(project.project_dir, "stems")
                final_stems = project.file_dict.get("stems", [])
                for fname in os.listdir(out_dir):
                    full_path = os.path.join(out_dir, fname)
                    if fname == "separation_info.json":
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
