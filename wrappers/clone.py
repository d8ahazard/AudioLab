import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import time
import json

from handlers.config import model_path, output_path
from handlers.weights_zip import list_weights_zip_models
from modules.rvc.configs.config import Config
from modules.rvc.infer.modules.vc.pipeline import VC
from modules.cloning import main as cloning
from util.data_classes import ProjectFiles
from wrappers.base_wrapper import BaseWrapper, TypedInput
import gradio as gr
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def _safe_track_key(track_ref: str) -> str:
    base = os.path.splitext(track_ref)[0].strip().lower()
    base = base.replace("\\", "/")
    return re.sub(r"[^a-z0-9/_\-\.]+", "_", base).replace("/", "__")


def _lyrics_text_from_json(path: str) -> Optional[str]:
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        segs = data.get("segments", []) if isinstance(data, dict) else []
        parts = []
        for seg in segs:
            txt = str(seg.get("text", "")).strip()
            if not txt:
                continue
            tags = seg.get("tags", [])
            if isinstance(tags, list) and tags:
                prefix = " ".join(f"[{str(t).strip()}]" for t in tags if str(t).strip())
                txt = f"{prefix} {txt}".strip() if prefix else txt
            parts.append(txt)
        return " ".join(parts).strip() or None
    except Exception as e:
        logger.debug("Could not parse lyrics json %s: %s", path, e)
        return None


def _candidate_track_lyrics_paths(project_dir: str, input_file: str) -> List[str]:
    lyrics_dir = os.path.join(project_dir, "lyrics")
    track_dir = os.path.join(lyrics_dir, "tracks")
    candidates: List[str] = []
    keys: List[str] = []
    abs_project = os.path.abspath(project_dir)
    abs_input = os.path.abspath(input_file)
    if abs_input.startswith(abs_project):
        rel = os.path.relpath(abs_input, abs_project).replace("\\", "/")
        keys.append(_safe_track_key(rel))
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    keys.append(_safe_track_key(base_name))
    normalized = re.sub(r"\((vocals|bg_vocals|vocals_full)\)", "", base_name, flags=re.IGNORECASE).strip(" _-")
    if normalized and normalized != base_name:
        keys.append(_safe_track_key(normalized))
    keys = list(dict.fromkeys([k for k in keys if k]))
    for key in keys:
        candidates.extend(
            [
                os.path.join(track_dir, f"{key}.annotated_lyrics.json"),
                os.path.join(track_dir, f"{key}.transcript.auto.json"),
                os.path.join(track_dir, f"{key}.transcript.json"),
            ]
        )
    candidates.extend(
        [
            os.path.join(lyrics_dir, "annotated_lyrics.json"),
            os.path.join(lyrics_dir, "transcript.auto.json"),
            os.path.join(lyrics_dir, "transcript.json"),
        ]
    )
    return candidates


def _wav_stats(path: str) -> Dict[str, Any]:
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if y.size == 0:
        return {"path": path, "sr": int(sr), "samples": 0}
    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(np.square(y), dtype=np.float64)))
    return {
        "path": path,
        "sr": int(sr),
        "samples": int(y.shape[0]),
        "duration_sec": float(y.shape[0] / max(1, int(sr))),
        "peak_abs": peak,
        "rms": rms,
        "clipped_frac": float(np.mean(np.abs(y) > 0.99)),
    }


def _append_ab_report(event: Dict[str, Any]) -> None:
    """
    Append JSONL event when AUDIOCLONE_AB_REPORT is set.
    This is used by deterministic V2/V3 A-B evaluations.
    """
    report_path = os.environ.get("AUDIOCLONE_AB_REPORT", "").strip()
    if not report_path:
        return
    try:
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("Failed writing AUDIOCLONE_AB_REPORT: %s", e)


def list_speakers():
    """
    Scan the model_path/trained directory and return all .pth files (trained voice checkpoints).
    """
    speaker_dir = os.path.join(model_path, "trained")
    os.makedirs(speaker_dir, exist_ok=True)
    v2_models = sorted(
        [os.path.splitext(f)[0] for f in os.listdir(speaker_dir) if f.endswith(".pth")]
    )
    v3_dir = os.path.join(speaker_dir, "v3")
    os.makedirs(v3_dir, exist_ok=True)
    v3_stems = {os.path.splitext(f)[0] for f in os.listdir(v3_dir) if f.endswith(".pth") or f.endswith(".safetensors")}
    v3_models = sorted(v3_stems)
    # Display v3 models explicitly so routing is unambiguous.
    v3_display = [f"{name} (v3)" for name in v3_models]
    # Merge in zip-backed models (weights.gg style)
    try:
        zip_models = list_weights_zip_models()
        zip_names = [m.display_name for m in zip_models]
    except Exception as e:
        logger.warning(f"Could not list zip models: {e}")
        zip_names = []
    # Keep it stable and avoid duplicates
    merged = list(dict.fromkeys(v2_models + v3_display + zip_names))
    return merged


def list_speakers_ui():
    """
    Return a dictionary suitable for UI updates,
    containing the speaker checkpoint paths found by list_speakers().
    """
    return {"choices": list_speakers(), "__type__": "update"}


def _resolve_v3_index_path(model_name: str) -> Optional[str]:
    candidates = [
        os.path.join(model_path, "trained", "v3", f"{model_name}_native_retrieval.index"),
        os.path.join(model_path, "trained", "v3", f"{model_name}.index"),
        os.path.join(output_path, "rvc_v3_data", model_name, "retrieval_index.index"),
    ]
    project_dir = os.path.join(output_path, "rvc_v3_data", model_name)
    if os.path.isdir(project_dir):
        candidates.extend(
            [
                os.path.join(project_dir, f)
                for f in os.listdir(project_dir)
                if f.endswith(".index")
            ]
        )
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def _resolve_voice_selection(selected_voice: str) -> Dict[str, Optional[str]]:
    """
    Resolve a UI-selected voice string to concrete model metadata.
    """
    if not selected_voice:
        return {
            "backend": None,
            "checkpoint_path": None,
            "index_path": None,
            "model_name": None,
            "display_name": selected_voice,
        }

    # Zip model selection
    if selected_voice.endswith(" (zip)"):
        try:
            for m in list_weights_zip_models():
                if m.display_name == selected_voice:
                    return {
                        "backend": "v2",
                        "checkpoint_path": m.pth_path,
                        "index_path": m.index_path,
                        "model_name": os.path.splitext(os.path.basename(m.pth_path))[0],
                        "display_name": selected_voice,
                    }
        except Exception as e:
            logger.warning(f"Failed resolving zip model '{selected_voice}': {e}")
        return {
            "backend": None,
            "checkpoint_path": None,
            "index_path": None,
            "model_name": None,
            "display_name": selected_voice,
        }

    selected_voice = selected_voice.strip()
    if selected_voice.endswith(" (v3)"):
        model_name = selected_voice[: -len(" (v3)")].strip()
        v3_dir = os.path.join(model_path, "trained", "v3")
        v3_safe = os.path.join(v3_dir, f"{model_name}.safetensors")
        v3_pth = os.path.join(v3_dir, f"{model_name}.pth")
        v3_path = v3_safe if os.path.exists(v3_safe) else v3_pth
        if os.path.exists(v3_path):
            return {
                "backend": "v3",
                "checkpoint_path": v3_path,
                "index_path": _resolve_v3_index_path(model_name),
                "model_name": model_name,
                "display_name": selected_voice,
            }
        return {
            "backend": None,
            "checkpoint_path": None,
            "index_path": None,
            "model_name": model_name,
            "display_name": selected_voice,
        }

    # Native checkpoint selection
    speaker_dir = os.path.join(model_path, "trained")
    pth_path = os.path.join(speaker_dir, f"{selected_voice}.pth")
    if os.path.exists(pth_path):
        index_path = None
        try:
            from modules.rvc.infer.modules.vc.utils import get_index_path_from_model

            idx_name = get_index_path_from_model(pth_path)
            if idx_name:
                candidate = os.path.join(speaker_dir, idx_name)
                if os.path.exists(candidate):
                    index_path = candidate
        except Exception:
            index_path = None

        return {
            "backend": "v2",
            "checkpoint_path": pth_path,
            "index_path": index_path,
            "model_name": selected_voice,
            "display_name": selected_voice,
        }

    # Fallback: allow selecting bare model name for v3 via API.
    v3_path = os.path.join(speaker_dir, "v3", f"{selected_voice}.pth")
    if os.path.exists(v3_path):
        return {
            "backend": "v3",
            "checkpoint_path": v3_path,
            "index_path": _resolve_v3_index_path(selected_voice),
            "model_name": selected_voice,
            "display_name": f"{selected_voice} (v3)",
        }

    return {
        "backend": None,
        "checkpoint_path": None,
        "index_path": None,
        "model_name": selected_voice,
        "display_name": selected_voice,
    }


def toggle_clone_elements(clone_method):
    """
    Toggle the visibility of the clone elements based on the clone method.
    "RVC Controls", "Advanced RVC Options", "OpenVoice Controls", "Source Speaker"
    """
    logger.info(f"Toggling clone elements for {clone_method}")
    show_src_speaker = clone_method == "OpenVoice" or clone_method == "TTS"
    logger.info(f"Showing source speaker: {show_src_speaker}")
    return [
        gr.update(visible=clone_method == "RVC"),
        gr.update(visible=clone_method == "RVC"),
        gr.update(visible=clone_method == "OpenVoice"), 
        gr.update(visible=show_src_speaker),
        gr.update(visible=clone_method == "Singing conversion"),
        gr.update(visible=clone_method == "Performer V3")
    ]

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
    hidden_groups = ["OpenVoice Controls", "TTS Controls", "Source Speaker", "Singing Conversion", "Performer V3"]
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
            choices=["RVC", "Performer V3", "Singing conversion", "OpenVoice", "TTS"],
            type=str,
            gradio_type="Dropdown",
            on_select=toggle_clone_elements,
            controls=["RVC Controls", "Advanced RVC Options", "OpenVoice Controls", "Source Speaker", "Singing Conversion", "Performer V3"],
            required=True
        ),
        "performer_profile": TypedInput(default="huxlxy", type=str, gradio_type="Dropdown",
            choices=["tupac", "huxlxy", "shinedown", "chester"], description="Experimental performer profile.", group_name="Performer V3"),
        "performer_mode": TypedInput(default="preserve", type=str, gradio_type="Dropdown",
            choices=["preserve", "performer"], description="Preserve uses existing RVC. Performer currently runs the untrained feasibility guide.", group_name="Performer V3"),
        "performer_lyrics": TypedInput(default="", type=str, gradio_type="Textbox",
            description="Desired words for a phrase preview up to 20 seconds; blank keeps source words.", group_name="Performer V3"),
        "performer_strength": TypedInput(default=1., type=float, gradio_type="Slider", ge=0, le=1, step=1,
            description="Delivery strength. This feasibility stage supports only 0 (preserve) or 1 (guide).", group_name="Performer V3"),
        "performer_seed": TypedInput(default=20260924, type=int, gradio_type="Number", render=False,
            description="Repeatable generation seed.", group_name="Performer V3"),
        "performer_source_transcript": TypedInput(default="", type=str, render=False,
            description="Corrected source words; disagreement with automatic timing requires realignment.", group_name="Performer V3"),
        "performer_edits": TypedInput(default=[], type=list, render=False,
            description="Phrase edits containing start, end, and desired text.", group_name="Performer V3"),
        # RVC-specific controls group
        "svc_backend": TypedInput(default="vevo2", type=str, gradio_type="Dropdown",
            choices=["seed_vc", "yingmusic", "vevo2"], description="Offline singing backend.", group_name="Singing Conversion"),
        "svc_reference": TypedInput(default="", type=str, gradio_type="Textbox",
            description="Target singer reference audio path (use a different song).", group_name="Singing Conversion"),
        "svc_checkpoint": TypedInput(default="", type=str, gradio_type="Textbox",
            description="Target checkpoint path; empty uses the official pretrained model. Seed-VC and YingMusic only.", group_name="Singing Conversion"),
        "svc_config": TypedInput(default="", type=str, gradio_type="Textbox",
            description="Matching model configuration path, if using a custom checkpoint.", group_name="Singing Conversion"),
        "svc_mode": TypedInput(default="target_style", type=str, gradio_type="Dropdown",
            choices=["preserve", "target_style"], description="Preserve source delivery, or use target delivery (Vevo2). Melody and song placement remain the goal.", group_name="Singing Conversion"),
        "svc_lyrics": TypedInput(default="", type=str, gradio_type="Textbox",
            description="Optional source lyrics; required for Vevo2 target delivery.", group_name="Singing Conversion"),
        "svc_reference_text": TypedInput(default="", type=str, gradio_type="Textbox",
            description="Words sung in the reference clip (Vevo2 target delivery).", group_name="Singing Conversion"),
        "svc_seed": TypedInput(default=20260924, type=int, gradio_type="Number",
            description="Random seed for repeatable comparisons.", group_name="Singing Conversion"),
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
            default=False,
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
            group_name="Source Speaker"
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
            choices=["hybrid", "pm", "harvest", "dio", "rmvpe", "rmvpe_onnx", "rmvpe+", "fcpe", "crepe", "crepe-tiny",
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
        ),
        "use_model_warmup": TypedInput(
            default=True,
            description="Prepend warmup audio to improve clone quality on initial segments.",
            type=bool,
            gradio_type="Checkbox",
            render=True,
            group_name="Advanced RVC Options"
        ),
        "warmup_duration": TypedInput(
            default=10.0,
            description="Duration in seconds of warmup audio to prepend (first N seconds of non-silent audio).",
            type=float,
            gradio_type="Slider",
            ge=1.0,
            le=30.0,
            step=0.5,
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
            selected_voice_ui = filtered_kwargs.get("selected_voice", "")
            selected_voice_meta = _resolve_voice_selection(selected_voice_ui)
            selected_voice_path = selected_voice_meta.get("checkpoint_path")
            selected_index_path = selected_voice_meta.get("index_path")
            selected_voice_backend = selected_voice_meta.get("backend")
            if not selected_voice_path or not os.path.exists(selected_voice_path):
                if callback is not None:
                    callback(0, "Selected voice model not found.")
                raise FileNotFoundError(f"Selected voice model not found: {selected_voice_ui}")

            # Resolve to actual checkpoint path (supports zip-extracted models + v3 models).
            selected_voice = selected_voice_path

            v3_pipeline = None
            if selected_voice_backend == "v3":
                try:
                    import torch
                    from modules.rvc_v3.configs.v3_config import RVCV3Config
                    from modules.rvc_v3.inference.pipeline import RVCV3Pipeline
                    from modules.rvc_v3.io.checkpoint_io import load_inference_model

                    _, _, meta = load_inference_model(selected_voice, device="cpu")
                    cfg_dict = meta.get("config", {})
                    if not cfg_dict:
                        raise RuntimeError(
                            f"Missing V3 config metadata for checkpoint: {selected_voice}. "
                            "Re-export this model to .safetensors with _config.json."
                        )
                    v3_config = RVCV3Config.from_dict(cfg_dict)
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    v3_pipeline = RVCV3Pipeline(selected_voice, v3_config, device=device)
                    if selected_index_path and os.path.exists(selected_index_path):
                        try:
                            v3_pipeline.load_retrieval_index(selected_index_path)
                            logger.info(f"Using V3 retrieval index for '{selected_voice_ui}': {selected_index_path}")
                        except Exception as idx_err:
                            logger.warning(f"Could not load V3 retrieval index '{selected_index_path}': {idx_err}")
                    else:
                        logger.info(f"No V3 retrieval index found for '{selected_voice_ui}'. Proceeding without index.")
                except Exception as e:
                    logger.error(f"Failed to initialize V3 pipeline: {e}")
                    if callback is not None:
                        callback(0, f"Failed to initialize V3 model: {e}")
                    raise
            else:
                # Ensure V2 RVC is set up
                config = Config()
                self.vc = VC(config, True)

                # IMPORTANT:
                # VC.vc_multi() calls VC.get_vc() internally and overwrites self.index based on models/trained/*,
                # which breaks zip-extracted indexes living under `.extracted/`.
                # Load the model first, then override index with the resolved zip index path (if present).
                try:
                    if self.vc is not None:
                        self.vc.get_vc(selected_voice)
                        if selected_index_path and os.path.exists(selected_index_path):
                            self.vc.index = selected_index_path
                            logger.info(f"Using resolved index for '{selected_voice}': {selected_index_path}")
                        else:
                            # Keep whatever VC.get_vc discovered (or None)
                            logger.info(f"No index found/resolved for '{selected_voice}'. Proceeding without index.")
                except Exception as e:
                    logger.error(f"Failed to initialize RVC model/index: {e}")
                    if callback is not None:
                        callback(0, f"Failed to initialize RVC model/index: {e}")
                    raise
                
            spk_id = filtered_kwargs.get("speaker_id", 0)
            f0method = filtered_kwargs.get("pitch_extraction_method", "rmvpe+")
            rms_mix_rate = filtered_kwargs.get("volume_mix_rate", 0.9)
            protect = filtered_kwargs.get("accent_strength", 0.2)
            index_rate = filtered_kwargs.get("index_rate", 1)
            filter_radius = filtered_kwargs.get("filter_radius", 3)
            # Default to mono cloning unless explicitly enabled to avoid channel artifacts
            clone_stereo = filtered_kwargs.get("clone_stereo", False)
            pitch_correction = filtered_kwargs.get("pitch_correction", False)
            pitch_correction_humanize = filtered_kwargs.get("pitch_correction_humanize", 0.95)
            merge_type = filtered_kwargs.get("merge_type", "median")
            crepe_hop_length = filtered_kwargs.get("crepe_hop_length", 160)
            f0_autotune = filtered_kwargs.get("f0_autotune", False)
            rmvpe_onnx = filtered_kwargs.get("rmvpe_onnx", False)
            use_model_warmup = filtered_kwargs.get("use_model_warmup", True)
            warmup_duration = filtered_kwargs.get("warmup_duration", 10.0)

            total_steps = len(inputs)
            if clone_bg_vocals:
                total_steps *= 2
            if pitch_correction:
                total_steps *= 2
                
            # For RVC callback
            if self.vc:
                self.vc.total_steps = total_steps
            if selected_voice_backend == "v3":
                logger.info(
                    "Selected V3 model '%s'. Using V3 conversion pipeline (applies pitch_shift/index_rate/speaker_id; "
                    "ignores v2-only knobs such as f0_method/filter_radius/protect/rmvpe options).",
                    selected_voice_ui,
                )

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
                    filtered_inputs = [
                        p for p in last_outputs
                        if "(Vocals)" in p or "(BG_Vocals" in p or "(Vocals_Full)" in p
                    ]
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

                            if selected_voice_backend == "v3":
                                if v3_pipeline is None:
                                    raise RuntimeError("V3 pipeline was not initialized")
                                out_dir = os.path.join(project.project_dir, "cloned")
                                os.makedirs(out_dir, exist_ok=True)
                                base_name, _ = os.path.splitext(os.path.basename(proc_file))
                                model_base = selected_voice_meta.get("model_name") or "v3"
                                model_base = re.sub(r"[^a-zA-Z0-9 _\\-\\(\\)\\.]+", "", model_base).strip() or "v3"
                                output_file = os.path.join(
                                    out_dir,
                                    f"{base_name}(Cloned)({model_base}_rvcv3).wav",
                                )
                                lyrics_for_convert = (custom_text or "").strip() or None
                                if not lyrics_for_convert:
                                    # Prefer track-specific edited lyrics, then project-level fallbacks.
                                    for lyr_path in _candidate_track_lyrics_paths(project.project_dir, proc_file):
                                        lyrics_for_convert = _lyrics_text_from_json(lyr_path)
                                        if lyrics_for_convert:
                                            break
                                if not lyrics_for_convert:
                                    # Reuse existing transcript for this file if present.
                                    existing_transcript = os.path.join(out_dir, f"{base_name}_transcript.json")
                                    if os.path.isfile(existing_transcript):
                                        try:
                                            with open(existing_transcript, "r", encoding="utf-8") as tf:
                                                tdata = json.load(tf)
                                            full = (tdata.get("full_text", "") or "").strip()
                                            if not full:
                                                full = " ".join(
                                                    str(s.get("text", "")).strip()
                                                    for s in tdata.get("segments", [])
                                                    if isinstance(s, dict)
                                                ).strip()
                                            if full:
                                                lyrics_for_convert = "[clean] " + full
                                        except Exception as tf_err:
                                            logger.debug("Could not load existing transcript: %s", tf_err)
                                # Auto-lyrics can destabilize current V3 conditioning for some songs.
                                # Keep it opt-in until text-conditioning quality is consistently better.
                                use_auto_lyrics = os.environ.get("AUDIOCLONE_V3_AUTO_LYRICS", "0").strip() == "1"
                                if use_auto_lyrics and not lyrics_for_convert:
                                    try:
                                        from modules.rvc_v3.data_prep.transcriber import Transcriber
                                        transcriber = Transcriber(output_dir=out_dir, model_size="base")
                                        segs = transcriber.transcribe_file(
                                            proc_file,
                                            os.path.join(out_dir, f"{base_name}_transcript.json"),
                                            language=None,
                                            word_timestamps=True,
                                            overwrite_existing=False,
                                        )
                                        if segs:
                                            lyrics_for_convert = "[clean] " + " ".join(s.get("text", "") for s in segs)
                                    except Exception as tr_err:
                                        logger.debug("V3 auto-transcribe skipped: %s", tr_err)
                                v3_pipeline.convert(
                                    audio_path=proc_file,
                                    lyrics=lyrics_for_convert,
                                    output_path=output_file,
                                    index_rate=float(index_rate),
                                    pitch_shift=int(filtered_kwargs.get("pitch_shift", 0)),
                                    speaker_id=int(spk_id),
                                )
                                clone_outputs.append(output_file)
                                try:
                                    event = {
                                        "backend": "v3",
                                        "model_display": selected_voice_ui,
                                        "checkpoint": selected_voice_path,
                                        "index_path": selected_index_path,
                                        "project_dir": project.project_dir,
                                        "input_file": proc_file,
                                        "output_file": output_file,
                                        "settings": {
                                            "index_rate": float(index_rate),
                                            "pitch_shift": int(filtered_kwargs.get("pitch_shift", 0)),
                                            "speaker_id": int(spk_id),
                                        },
                                        "output_stats": _wav_stats(output_file),
                                        "v3_debug": getattr(v3_pipeline, "last_convert_debug", {}),
                                    }
                                    _append_ab_report(event)
                                except Exception as rep_err:
                                    logger.warning("Could not emit V3 A/B report event: %s", rep_err)
                                if callback is not None:
                                    callback(1.0, f"V3 clone complete: {os.path.basename(output_file)}")
                            else:
                                # Perform the voice conversion with V2 RVC path
                                file_outputs = self.vc.vc_multi(
                                    model=selected_voice,
                                    sid=spk_id,
                                    paths=[proc_file],
                                    f0_up_key=filtered_kwargs.get("pitch_shift", 0),
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
                                    model_display_name=selected_voice_ui,
                                    callback=project_callback,
                                    use_model_warmup=use_model_warmup,
                                    warmup_duration=warmup_duration
                                )
                                clone_outputs.extend(file_outputs)
                                try:
                                    for one_out in file_outputs:
                                        event = {
                                            "backend": "v2",
                                            "model_display": selected_voice_ui,
                                            "checkpoint": selected_voice_path,
                                            "index_path": selected_index_path,
                                            "project_dir": project.project_dir,
                                            "input_file": proc_file,
                                            "output_file": one_out,
                                            "settings": {
                                                "index_rate": float(index_rate),
                                                "pitch_shift": int(filtered_kwargs.get("pitch_shift", 0)),
                                                "speaker_id": int(spk_id),
                                                "f0_method": f0method,
                                                "rms_mix_rate": float(rms_mix_rate),
                                                "protect": float(protect),
                                            },
                                            "output_stats": _wav_stats(one_out),
                                        }
                                        _append_ab_report(event)
                                except Exception as rep_err:
                                    logger.warning("Could not emit V2 A/B report event: %s", rep_err)
                            
                        elif clone_method == "Performer V3":
                            from modules.rvc_v3.performer.service import convert
                            profile_name = filtered_kwargs.get("performer_profile", "huxlxy")
                            if profile_name not in {"tupac", "huxlxy", "shinedown", "chester"}:
                                raise ValueError("Unknown performer profile")
                            clone_outputs.append(convert(proc_file,
                                os.path.join(output_path, "performer_v3_validation", "profiles", profile_name + ".json"),
                                os.path.join(project.project_dir, "cloned"),
                                delivery_mode=filtered_kwargs.get("performer_mode", "preserve"),
                                target_lyrics=filtered_kwargs.get("performer_lyrics") or None,
                                source_transcript=filtered_kwargs.get("performer_source_transcript") or None,
                                phrase_edits=filtered_kwargs.get("performer_edits", []),
                                delivery_strength=filtered_kwargs.get("performer_strength", 1.),
                                seed=filtered_kwargs.get("performer_seed", 20260924)))
                        elif clone_method == "Singing conversion":
                            from modules.svc_backends import convert
                            clone_outputs.append(convert(proc_file,
                                filtered_kwargs.get("svc_reference"), os.path.join(project.project_dir, "cloned"),
                                backend=filtered_kwargs.get("svc_backend", "vevo2"),
                                checkpoint=filtered_kwargs.get("svc_checkpoint") or None,
                                config=filtered_kwargs.get("svc_config") or None,
                                lyrics=filtered_kwargs.get("svc_lyrics") or None,
                                reference_text=filtered_kwargs.get("svc_reference_text") or None,
                                seed=filtered_kwargs.get("svc_seed", 20260924),
                                mode=filtered_kwargs.get("svc_mode", "target_style")))
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
                                strength=voice_strength,
                                temp_dir=project.project_dir
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
            methods.append({"id": "Performer V3", "name": "Performer V3 (experimental)",
                            "description": "Existing V2 preservation or an untrained pronunciation-guide feasibility control"})
            return {"methods": methods}

        return [process_clone_json, list_available_voices, list_clone_methods]
