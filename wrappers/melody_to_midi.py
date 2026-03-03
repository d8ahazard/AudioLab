import hashlib
import json
import logging
import os
from typing import Any, Dict, List

from handlers.melody_to_midi import extract_melody_to_midi
from util.data_classes import ProjectFiles
from wrappers.base_wrapper import BaseWrapper, TypedInput

logger = logging.getLogger(__name__)


class MelodyToMidi(BaseWrapper):
    title = "Melody To Midi"
    priority = 2
    default = False
    prep_only = True
    description = "Extract a monophonic melody from vocals and save as MIDI."
    allowed_kwargs = {
        "onset_threshold": TypedInput(
            default=0.5,
            ge=0.0,
            le=1.0,
            step=0.01,
            description="Minimum energy required for a note onset (higher = fewer notes).",
            type=float,
            gradio_type="Slider",
        ),
        "frame_threshold": TypedInput(
            default=0.3,
            ge=0.0,
            le=1.0,
            step=0.01,
            description="Minimum energy for frames to count as a note (higher = fewer notes).",
            type=float,
            gradio_type="Slider",
        ),
        "minimum_note_length_ms": TypedInput(
            default=127.7,
            ge=20.0,
            le=1000.0,
            step=5.0,
            description="Minimum note length in milliseconds.",
            type=float,
            gradio_type="Slider",
        ),
    }

    def process_audio(self, inputs: List[ProjectFiles], callback=None, **kwargs: Dict[str, Any]) -> List[ProjectFiles]:
        pj_outputs = []

        for project in inputs:
            previous_last_outputs = list(project.last_outputs)
            vocals_path = project.get_latest_vocals() or project.src_file

            if not vocals_path or not os.path.exists(vocals_path):
                logger.warning("No vocals found for MIDI extraction.")
                continue

            output_folder = project.output_dir("midi")
            base_name = os.path.splitext(os.path.basename(vocals_path))[0]
            output_path = os.path.join(output_folder, f"{base_name}.mid")
            cache_path = os.path.join(output_folder, "midi_info.json")

            if callback is not None:
                callback(0, f"Extracting melody from {os.path.basename(vocals_path)}")
            logger.info(f"Melody To Midi: extracting from {vocals_path}")

            filtered_kwargs = {k: v for k, v in kwargs.items() if k in self.allowed_kwargs}
            onset_threshold = float(filtered_kwargs.get("onset_threshold", 0.5))
            frame_threshold = float(filtered_kwargs.get("frame_threshold", 0.3))
            minimum_note_length_ms = float(filtered_kwargs.get("minimum_note_length_ms", 127.7))

            cache_data = self._load_cache(cache_path)
            input_hash = self._hash_file(vocals_path)
            separation_hash = self._get_separation_hash(project)
            cache_key = vocals_path

            if (
                cache_data.get(cache_key, {}).get("input_hash") == input_hash
                and cache_data.get(cache_key, {}).get("separation_hash") == separation_hash
                and cache_data.get(cache_key, {}).get("onset_threshold") == onset_threshold
                and cache_data.get(cache_key, {}).get("frame_threshold") == frame_threshold
                and cache_data.get(cache_key, {}).get("minimum_note_length_ms") == minimum_note_length_ms
                and os.path.exists(output_path)
            ):
                midi_path = output_path
                logger.info("Melody To Midi: using cached output")
            else:
                midi_path = extract_melody_to_midi(
                    vocals_path,
                    output_path,
                    onset_threshold=onset_threshold,
                    frame_threshold=frame_threshold,
                    minimum_note_length_ms=minimum_note_length_ms,
                )
                cache_data[cache_key] = {
                    "input_path": vocals_path,
                    "input_hash": input_hash,
                    "separation_hash": separation_hash,
                    "onset_threshold": onset_threshold,
                    "frame_threshold": frame_threshold,
                    "minimum_note_length_ms": minimum_note_length_ms,
                    "output_path": output_path,
                }
                self._save_cache(cache_path, cache_data)
            project.add_output("midi", midi_path)
            project.last_outputs = previous_last_outputs if previous_last_outputs else [vocals_path]
            pj_outputs.append(project)
            logger.info(f"Melody To Midi: wrote {midi_path}")

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

    @staticmethod
    def _get_separation_hash(project: ProjectFiles) -> str:
        stems_dir = os.path.join(project.project_dir, "stems")
        cache_file = os.path.join(stems_dir, "separation_info.json")
        if not os.path.exists(cache_file):
            return ""
        sha256_hash = hashlib.sha256()
        with open(cache_file, "rb") as f:
            sha256_hash.update(f.read())
        return sha256_hash.hexdigest()
