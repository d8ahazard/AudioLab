import hashlib
import json
import logging
import os
from typing import Any, Dict, List

from handlers.tempo import adjust_tempo
from util.data_classes import ProjectFiles
from wrappers.base_wrapper import BaseWrapper, TypedInput

logger = logging.getLogger(__name__)


class AdjustTempo(BaseWrapper):
    title = "Adjust Tempo"
    priority = 0
    default = False
    prep_only = True
    description = "Adjust tempo with high-quality time-stretching."

    allowed_kwargs = {
        "tempo_change_pct": TypedInput(
            default=0,
            ge=-50,
            le=50,
            step=1,
            description="Tempo change percentage (negative to slow down).",
            type=float,
            gradio_type="Slider",
        )
    }

    def process_audio(self, inputs: List[ProjectFiles], callback=None, **kwargs: Dict[str, Any]) -> List[ProjectFiles]:
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in self.allowed_kwargs}
        tempo_change_pct = float(filtered_kwargs.get("tempo_change_pct", 0))
        tempo_factor = 1.0 + (tempo_change_pct / 100.0)

        pj_outputs = []
        for project in inputs:
            outputs = []
            input_files, _ = self.filter_inputs(project, "audio")
            if not input_files:
                continue

            output_folder = project.output_dir("tempo")
            for idx, input_file in enumerate(input_files):
                if callback is not None:
                    callback(idx / max(len(input_files), 1), f"Adjusting tempo for {os.path.basename(input_file)}")

                base_name = os.path.splitext(os.path.basename(input_file))[0]
                suffix = f"{tempo_change_pct:+.0f}pct"
                output_file = os.path.join(output_folder, f"{base_name}_tempo_{suffix}.wav")
                cache_path = os.path.join(output_folder, "tempo_info.json")
                cache_data = self._load_cache(cache_path)
                input_hash = self._hash_file(input_file)
                cache_key = f"{input_file}|{suffix}"

                if (
                    cache_data.get(cache_key, {}).get("input_hash") == input_hash
                    and cache_data.get(cache_key, {}).get("tempo_factor") == tempo_factor
                    and os.path.exists(output_file)
                ):
                    adjusted_path = output_file
                    logger.info(f"Adjust Tempo: using cached output for {os.path.basename(input_file)}")
                else:
                    adjusted_path = adjust_tempo(input_file, output_file, tempo_factor)
                    cache_data[cache_key] = {
                        "input_path": input_file,
                        "input_hash": input_hash,
                        "tempo_factor": tempo_factor,
                        "output_path": output_file,
                    }
                    self._save_cache(cache_path, cache_data)
                outputs.append(adjusted_path)

            project.add_output("tempo", outputs)
            pj_outputs.append(project)

        return pj_outputs

    @staticmethod
    def _hash_file(filepath: str) -> str:
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    @staticmethod
    def _load_cache(cache_path: str) -> Dict[str, Any]:
        if not os.path.exists(cache_path):
            return {}
        try:
            with open(cache_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    @staticmethod
    def _save_cache(cache_path: str, data: Dict[str, Any]) -> None:
        with open(cache_path, "w") as f:
            json.dump(data, f, indent=2)
