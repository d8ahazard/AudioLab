# coding: utf-8
import logging
import os
import subprocess
import uuid
import warnings
import json
import tempfile
import sys
import gc
from pathlib import Path
from importlib.metadata import version
from typing import List, Dict, Callable, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
from modules.separator.model_runtime import AudioLabSeparator, BG_MODELS, INSTRUMENT_MODELS, read_outputs, CUSTOM_MODELS, FUSED_DEREVERB
from modules.separator.audio_quality import stereo, fuse, clean_hybrid_vocals, consensus_blend
from modules.separator.stem_manifest import MANIFEST_NAME, PIPELINE_REVISION, file_hash, write_json, model_fingerprint

from modules.separator.instrument_policy import INSTRUMENT_STEMS, DRUM_STEMS, MEGA_EXTRAS, MEGA_MODEL, selected, duplicate_of
from handlers.config import app_path, output_path
from handlers.reverb import extract_reverb
from modules.separator.separation_profiles import (
    SeparationProfile,
    get_profile_models,
    get_profile_defaults, selected_models, SeparationPreset, get_preset_defaults
)

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# Monkey-patch soundfile to add SoundFileRuntimeError if missing.
if not hasattr(sf, "SoundFileRuntimeError"):
    sf.SoundFileRuntimeError = RuntimeError


################################################################################
#                        HELPER UTILITY FUNCTIONS
################################################################################

def ensure_wav(input_path: str, sr: int = 44100) -> str:
    """
    Convert input file to WAV format if necessary.

    Parameters:
        input_path (str): Path to the input audio file.
        sr (int): Target sample rate.

    Returns:
        str: Path to the WAV file.

    Raises:
        FileNotFoundError: If the input file is missing.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Missing file: {input_path}")
    base, ext = os.path.splitext(input_path)
    if ext.lower() == ".wav":
        return input_path
    out_wav = base + "_converted.wav"
    if not os.path.isfile(out_wav):
        # Preserve dynamic range for downstream separation (avoid 16-bit quantization noise).
        # Keep as float WAV so ensemble blending can't accidentally "raise the noise floor".
        cmd = ["ffmpeg", "-y", "-i", input_path, "-acodec", "pcm_f32le", "-ac", "2", "-ar", str(sr), out_wav]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_wav


def write_temp_wav(mix_np: np.ndarray, sr: int, out_dir: str) -> str:
    """
    Writes a numpy audio array to a temporary PCM_16 WAV file.

    Parameters:
        mix_np (np.ndarray): Audio data.
        sr (int): Sample rate.
        out_dir (str): Output directory.

    Returns:
        str: Path to the temporary WAV file.
    """
    if mix_np.ndim == 1:
        mix_np = np.stack([mix_np, mix_np], axis=0)
    wav_data = mix_np.T.astype(np.float32)
    tmp_name = f"tmp_{uuid.uuid4().hex}.wav"
    tmp_path = os.path.join(out_dir, tmp_name)
    # Keep temp audio as float to avoid repeated PCM_16 quantization across multi-pass separation.
    sf.write(tmp_path, wav_data, sr, format="WAV", subtype="FLOAT")
    return tmp_path


################################################################################
#                 MAIN CLASS: ENSEMBLE + ADVANCED SEPARATION
################################################################################

class EnsembleDemucsMDXMusicSeparationModel:
    """
    A multi-model ensemble-based separation approach with additional
    background vocal splitting and transformation chain for audio processing.

    Attributes:
        options (Dict): Configuration options.
        callback (Callable): Optional callback function for progress updates.
        device (torch.device): Computation device.
        separator (Separator): Audio separator instance.
        total_steps (int): Total number of progress steps.
        global_step (int): Current progress step.
    """

    def __init__(self, options: Dict, callback: Callable = None):
        self.callback = callback
        self.options = options
        self.device = torch.device("cuda:0") if torch.cuda.is_available() and not options.get("cpu", False) \
            else torch.device("cpu")
        self.separator = AudioLabSeparator(
            log_level=logging.ERROR,
            model_file_dir=os.path.join(app_path, "models", "audio_separator"),
            invert_using_spec=options.get("separation_profile", "hybrid_cleaned") not in {"v4", "hybrid_cleaned"},
            use_autocast=not options.get("cpu", False), use_soundfile=True,
            normalization_threshold=1.0, amplification_threshold=0.0,
            cpu=options.get("cpu", False), quality=options.get("separation_quality", "balanced"),
            preserve_gain=options.get("separation_profile", "hybrid_cleaned") in {"v4", "hybrid_cleaned"},
        )
        
        # Listener-selected cleaned hybrid is the default; legacy profiles remain.
        profile_str = options.get("separation_profile", "hybrid_cleaned")
        try:
            self.separation_profile = SeparationProfile(profile_str)
        except ValueError:
            logger.warning(f"Invalid separation profile '{profile_str}', defaulting to v2")
            self.separation_profile = SeparationProfile.V2_HIGH_QUALITY
        
        # Get profile defaults
        profile_defaults = get_profile_defaults(self.separation_profile)
        if options.get("separation_preset") and self.separation_profile.value not in {"v4", "hybrid_cleaned"}:
            profile_defaults = get_preset_defaults(SeparationPreset(options["separation_preset"]))
        self.profile_models = selected_models(options)
        if self.separation_profile.value == "v4":
            preset = options.get("separation_preset")
            # Presets select the fusion objective; all V4 routes keep the same three models.
            if preset in {"karaoke", "instrumental"}:
                options.setdefault("instrumental_fusion", "min_magnitude")
            elif preset in {"acappella", "podcast"}:
                options.setdefault("vocal_fusion", "min_magnitude")
        # Old saved Smart Stems flags are deliberately ignored.
        self.smart_stems = "off"
        self.instrument_stems = selected(options, "instrument_stems", INSTRUMENT_STEMS, INSTRUMENT_STEMS)
        self.drum_stems = selected(options, "drum_stems", DRUM_STEMS, DRUM_STEMS)
        self.mega_stems = selected(options, "mega_stems", MEGA_EXTRAS, [])

        # Use profile defaults if values are None or not provided
        ensemble_size_opt = options.get("ensemble_size")
        residual_fill_opt = options.get("residual_fill")
        
        self.ensemble_strength = ensemble_size_opt if ensemble_size_opt is not None else profile_defaults["ensemble_size"]
        self.residual_blend_pct = residual_fill_opt if residual_fill_opt is not None else profile_defaults["residual_fill_pct"]
        self.bleed_guard_multiplier = profile_defaults["bleed_guard_multiplier"]
        
        logger.info(f"Using separation profile: {self.separation_profile.value} "
                   f"(ensemble_size={self.ensemble_strength}, residual_blend={self.residual_blend_pct:.2f})")
        
        # Models are loaded and downloaded only when their stage runs.

        # Flags and options
        self.vocals_only = bool(options.get("vocals_only", False))
        self.separate_drums = bool(options.get("separate_drums", False))
        self.separate_woodwinds = bool(options.get("separate_woodwinds", False))
        self.alt_bass_model = bool(options.get("alt_bass_model", False))
        self.use_vocft = bool(options.get("use_VOCFT", False))

        # Weighted blending for vocals
        self.weight_inst = float(options.get("weight_InstVoc", 8.0))
        self.weight_vocft = float(options.get("weight_VOCFT", 1.0))
        self.weight_rof = float(options.get("weight_VitLarge", 5.0))

        # Overlap values for advanced separation
        self.overlap_large = options.get("overlap_large", 0.6)
        self.overlap_small = options.get("overlap_small", 0.5)

        # Transformation and BG vocal options
        self.reverb_removal = options.get("reverb_removal", "Nothing")
        self.echo_removal = options.get("echo_removal", "Nothing")
        self.crowd_removal = options.get("crowd_removal", "Nothing")
        self.noise_removal = options.get("noise_removal", "Nothing")
        self.delay_removal_model = options.get("delay_removal_model", "dereverb-echo_mel_band_roformer_sdr_13.4843_v2.ckpt")
        self.noise_removal_model = options.get("noise_removal_model", "UVR-DeNoise.pth")
        self.crowd_removal_model = options.get("crowd_removal_model", "UVR-MDX-NET_Crowd_HQ_1.onnx")
        self.separate_bg_vocals = options.get("separate_bg_vocals", True)
        self.bg_vocal_layers = options.get("bg_vocal_layers", 1)
        if not isinstance(self.bg_vocal_layers, int) or not 1 <= self.bg_vocal_layers <= 10:
            raise ValueError("bg_vocal_layers must be an integer from 1 to 10")
        self.store_reverb_ir = options.get("store_reverb_ir", False)

        # Progress tracking
        self.global_step = 0
        self.total_steps = 0
        self.callback = callback or options.get("callback")

    def _advance_progress(self, desc: str, weight: int = 1) -> None:
        """
        Increments progress by a given weight and calls the callback with the current progress.

        Parameters:
            desc (str): Description of the current progress step.
            weight (int): Weight (number of steps) to advance.
        """
        self.global_step += weight
        if self.callback is not None and self.total_steps > 0:
            self.callback(min(0.99, self.global_step / self.total_steps), desc, self.total_steps)
        logger.info(f"[{self.global_step}/{self.total_steps}] {desc}")

    def _residual_subtract(self, base: np.ndarray, component: np.ndarray, sr: int, max_shift_ms: float = 12.0) -> np.ndarray:
        """
        Subtracts component from base with small time alignment and gain matching to reduce hiss.

        - Aligns component to base via cross-correlation within ±max_shift_ms
        - Computes per-channel least-squares gain alpha = (base·comp)/(comp·comp)
        - Clips alpha to a reasonable range to avoid over-subtraction
        - Returns the residual with original base length

        Shapes are (channels, samples).
        """
        if not isinstance(base, np.ndarray) or not isinstance(component, np.ndarray):
            return base
        if base.ndim == 1:
            base = np.stack([base, base], axis=0)
        if component.ndim == 1:
            component = np.stack([component, component], axis=0)
        channels = base.shape[0]
        max_shift = int((max_shift_ms / 1000.0) * float(sr))
        if max_shift < 0:
            max_shift = 0
        # Work on overlap region; keep non-overlap from base
        n = min(base.shape[-1], component.shape[-1])
        residual = np.copy(base)

        def shift_signal(x: np.ndarray, lag: int) -> np.ndarray:
            if lag == 0:
                return x
            if lag > 0:
                # component lags ref → pad front
                pad = np.zeros(lag, dtype=x.dtype)
                y = np.concatenate([pad, x[:-lag]])
            else:
                # component leads ref → pad end
                lag = -lag
                pad = np.zeros(lag, dtype=x.dtype)
                y = np.concatenate([x[lag:], pad])
            return y

        for ch in range(channels):
            ref = base[ch, :n]
            sig = component[ch, :n]
            # Small-lag alignment via cross-correlation
            if max_shift > 0 and ref.size > 0 and sig.size > 0:
                # Limit to a slice to keep computation reasonable on long tracks
                probe_len = min(n, 44100)  # up to ~1s for correlation
                ref_probe = ref[:probe_len]
                sig_probe = sig[:probe_len]
                corr = np.correlate(ref_probe, sig_probe, mode="full")
                center = len(corr) // 2
                window = corr[center - max_shift:center + max_shift + 1]
                best_rel = int(np.argmax(window)) - max_shift
            else:
                best_rel = 0
            sig_aligned = shift_signal(sig, best_rel)
            # Gain match (least-squares alpha)
            denom = float(np.dot(sig_aligned, sig_aligned)) + 1e-8
            alpha = float(np.dot(ref, sig_aligned)) / denom
            # Clip alpha to avoid over-subtraction; allow mild >1 when needed
            alpha = float(np.clip(alpha, 0.0, 1.25))
            res = ref - alpha * sig_aligned
            residual[ch, :n] = res

        # Prevent NaNs/Infs
        if not np.isfinite(residual).all():
            residual = np.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)
        return residual

    def _blend_tracks(self, tracks: List[np.ndarray], weights: List[float]) -> np.ndarray:
        """
        Blends a list of tracks with given weights.

        Parameters:
            tracks (List[np.ndarray]): List of audio stems.
            weights (List[float]): Corresponding weights.

        Returns:
            np.ndarray: Blended audio track.
        """
        max_length = max(t.shape[-1] for t in tracks)
        combined = np.zeros((tracks[0].shape[0], max_length), dtype=np.float32)
        total_weight = max(sum(weights), 1e-6)
        for idx, t in enumerate(tracks):
            weight = weights[idx] if idx < len(weights) else 1.0
            combined[:, :t.shape[-1]] += t * float(weight)
        combined = combined / total_weight
        # IMPORTANT: Never normalize UP here.
        # When multiple model outputs partially phase-cancel, the peak can get small even while
        # broadband noise remains; dividing by peak amplifies hiss (most noticeable during vocal sections).
        peak = float(np.max(np.abs(combined)) if combined.size else 0.0)
        ceiling = 0.99
        if peak > ceiling:
            combined *= (ceiling / peak)
        return combined

    def _separate_as_arrays_current(self, mix_np, sr, desc=None, output_folder=None, task="core"):
        mix_np = stereo(mix_np)
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        # Every stage owns its temporary outputs, even on failure.
        with tempfile.TemporaryDirectory(prefix="tmp_stage_", dir=output_folder) as temp:
            tmp = write_temp_wav(mix_np, sr, temp)
            self.separator.output_dir = temp
            self.separator.model_instance.output_dir = temp
            paths = self.separator.separate(tmp)
            stems = read_outputs(paths, temp, sr, mix_np.shape[-1], task)
            if task == "core" and not {"vocals", "instrumental"} <= stems.keys():
                raise RuntimeError(f"{self.separator.current_model} did not produce both primary stems: {list(stems)}")
            return stems

    def _hybrid_separate_all(self, files_data):
        # Keep the two listened-to recipes independent; downstream stages run once.
        outputs = {}
        for profile in ("v2", "v4"):
            child_options = {**self.options, "separation_profile": profile,
                             "ensemble_size": None, "residual_fill": None,
                             "separation_preset": None, "vocal_fusion": "avg_wave",
                             "instrumental_fusion": "avg_wave"}
            child = EnsembleDemucsMDXMusicSeparationModel(child_options)
            child._advance_progress = self._advance_progress
            try:
                outputs[profile] = child._ensemble_separate_all(files_data)
                self.separator.loaded_models.update(child.separator.loaded_models)
                self.separator.run_records.extend(child.separator.run_records)
            finally:
                child.separator.model_instance = None
                del child
        results = outputs["v2"]
        for key, res in results.items():
            res["instrumental"] = outputs["v4"][key]["instrumental"]
            res["vocals"] = clean_hybrid_vocals(res["vocals"], res["mix_np"], res["instrumental"])
        return results

    def _ensemble_separate_all(self, files_data: List[Dict]) -> Dict[str, Dict]:
        """
        Performs ensemble separation on all files.

        Parameters:
            files_data (List[Dict]): List of file data dictionaries.

        Returns:
            Dict[str, Dict]: Dictionary mapping base names to separation results.
        """
        if self.separation_profile == SeparationProfile.HYBRID_CLEANED:
            return self._hybrid_separate_all(files_data)
        results = {}
        for file in files_data:
            base_name = file.get("key", file["base_name"])
            results[base_name] = {
                "base_name": file["base_name"],
                "mix_np": file["mix_np"],
                "sr": file["sr"],
                "vocals_list": [],
                "instrumental_list": [],
                "v_weights": [],
                "i_weights": [],
                "output_folder": file["output_folder"]
            }
        
        # Get models from the separation profile
        profile_models = self.profile_models
        models_with_weights = [
            (spec.id, spec.vocal_weight, spec.inst_weight)
            for spec in profile_models
        ]
        
        logger.info(f"Using {len(models_with_weights)} models for {self.separation_profile.value} separation")
        
        # Adjust residual blend for small ensembles to avoid vocal leakage
        if self.ensemble_strength <= 2:
            self.options["residual_blend"] = min(float(self.residual_blend_pct), 0.2)
        else:
            self.options["residual_blend"] = float(self.residual_blend_pct)

        for model_name, v_wt, i_wt in models_with_weights:
            self.separator.load_model(model_name, **(next(s.kwargs for s in profile_models if s.id == model_name) or {}))
            for file in files_data:
                base_name = file.get("key", file["base_name"])
                mix_np = file["mix_np"]
                sr = file["sr"]
                self.separator.output_dir = file["output_folder"]
                self.separator.model_instance.output_dir = file["output_folder"]
                desc = f"[Ensemble] {base_name} => {model_name}"
                separated = self._separate_as_arrays_current(mix_np, sr, desc, output_folder=file["output_folder"])
                vstem = separated.get("vocals", np.zeros_like(mix_np))
                istem = separated.get("instrumental", np.zeros_like(mix_np))
                results[base_name]["vocals_list"].append(vstem)
                results[base_name]["instrumental_list"].append(istem)
                results[base_name]["v_weights"].append(v_wt)
                results[base_name]["i_weights"].append(i_wt)
                # User-friendly progress message
                model_idx = models_with_weights.index((model_name, v_wt, i_wt)) + 1
                self._advance_progress(f"Separating with AI model {model_idx} of {len(models_with_weights)}...")
        for base_name, res in results.items():
            # Restore original, strict blending behavior
            res["vocals"] = self._blend_tracks(res["vocals_list"], res["v_weights"])
            res["instrumental"] = self._blend_tracks(res["instrumental_list"], res["i_weights"])
            if self.separation_profile.value == "v4":
                res["vocals"] = fuse(res["vocals_list"], self.options.get("vocal_fusion", "avg_wave"))
                res["instrumental"] = fuse(res["instrumental_list"], self.options.get("instrumental_fusion", "avg_wave"))
                continue
            # Post-blend de-bleed: mix-based residual subtraction blended into instrumental
            try:
                mix_np = res.get("mix_np")
                voc_np = res.get("vocals")
                if isinstance(mix_np, np.ndarray) and isinstance(voc_np, np.ndarray):
                    resid = self._residual_subtract(mix_np, voc_np, res["sr"])  # gain-matched, aligned
                    # Align lengths
                    min_len = min(resid.shape[-1], res["instrumental"].shape[-1])
                    resid = resid[:, :min_len]
                    inst = res["instrumental"][:, :min_len]
                    # Only blend if it reduces correlation with vocals (prevents vocal bleed)
                    def cosine_abs(a: np.ndarray, b: np.ndarray) -> float:
                        a_flat = a.reshape(-1)
                        b_flat = b.reshape(-1)
                        denom = (np.linalg.norm(a_flat) * np.linalg.norm(b_flat)) + 1e-8
                        return float(abs(np.dot(a_flat, b_flat)) / denom)
                    sim_inst = cosine_abs(inst, voc_np[:, :min_len])
                    sim_resid = cosine_abs(resid, voc_np[:, :min_len])
                    
                    # Apply profile-specific bleed guard threshold
                    safe_cos_threshold = 0.12 * self.bleed_guard_multiplier
                    
                    if sim_resid + 1e-6 < sim_inst - 0.01:  # requires a small but real improvement
                        blend = float(self.options.get("residual_blend", 0.4))
                        
                        # Adjust blend factor based on cosine similarity (vocal bleed detection)
                        if sim_resid > safe_cos_threshold:
                            damp = min(1.0, (sim_resid / safe_cos_threshold))
                            blend = blend * (1.0 / (1.0 + 2.0 * (damp - 1.0)))
                        
                        blend = 0.0 if blend < 0 else (1.0 if blend > 1.0 else blend)
                        inst_refined = (1.0 - blend) * inst + blend * resid
                        # Peak safety
                        peak = float(np.max(np.abs(inst_refined)))
                        if peak > 0.99:
                            inst_refined = inst_refined * (0.99 / peak)
                        res["instrumental"] = inst_refined
            except Exception:
                pass
            # Safety: if instrumental is near-silent, derive residual from mix - vocals
            try:
                i_peak = float(np.max(np.abs(res["instrumental"])) if isinstance(res.get("instrumental"), np.ndarray) else 0.0)
            except Exception:
                i_peak = 0.0
            if i_peak < 1e-6:
                mix_np = res.get("mix_np")
                voc_np = res.get("vocals")
                if isinstance(mix_np, np.ndarray) and isinstance(voc_np, np.ndarray):
                    resid = self._residual_subtract(mix_np, voc_np, res["sr"])  # gain-matched, aligned
                    peak = float(np.max(np.abs(resid)))
                    if peak > 1.0:
                        resid = resid / peak
                    res["instrumental"] = resid
        return results

    def _multistem_separation_all(self, results):
        choice = self.options.get("instrument_model", "consensus")
        if choice == "consensus":
            self.separator.preserve_gain = True
        names = ["demucs", "roformer_sw"] if choice == "consensus" else [choice]
        for res in results.values():
            res["_instrument_predictions"] = []
        for name in names:
            self.separator.load_model(INSTRUMENT_MODELS[name])
            for res in results.values():
                audio = res["instrumental"] if choice == "consensus" or self.options.get("instrument_input", "instrumental") == "instrumental" else res["mix_np"]
                stems = self._separate_as_arrays_current(audio, res["sr"], output_folder=res["output_folder"], task="instruments")
                if not set(INSTRUMENT_STEMS) <= stems.keys():
                    raise RuntimeError("Instrument model missing required stems")
                res["_instrument_predictions"].append(stems)
        for res in results.values():
            pool = res["_instrument_predictions"]
            if len(pool) == 1:
                res.update({k: pool[0][k] for k in INSTRUMENT_STEMS})
            else:
                # Include the model vocal/residual bucket as competing evidence,
                # but never replace the approved hybrid lead-vocal estimate.
                roles = [k for k in INSTRUMENT_STEMS + ["vocals"] if k in pool[0] and k in pool[1]]
                avg = {k: (pool[0][k] + pool[1][k]) * .5 for k in roles}
                for role in INSTRUMENT_STEMS:
                    res[role] = consensus_blend(pool[0][role], pool[1][role], sum(v for k, v in avg.items() if k != role))
            self._advance_progress("Separated selected instruments")

    def _alt_bass_separation_all(self, results):
        self.separator.load_model("kuielab_a_bass.onnx")
        for res in results.values():
            stems = self._separate_as_arrays_current(res["instrumental"], res["sr"], output_folder=res["output_folder"], task="instruments")
            if "bass" not in stems:
                raise RuntimeError("Bass model did not produce a bass stem")
            res["bass"] = stems["bass"]
            self._advance_progress("Enhanced bass separation")

    def _advanced_drum_separation_all(self, results):
        self.separator.load_model("MDX23C-DrumSep-aufr33-jarredou.ckpt")
        for res in results.values():
            parents = res.get("_instrument_predictions", [])
            inputs = [p["drums"] for p in parents] if len(parents) == 2 else [res["drums"]]
            pool = [self._separate_as_arrays_current(a, res["sr"], output_folder=res["output_folder"], task="drums") for a in inputs]
            expected = ["drums_" + r for r in DRUM_STEMS]
            if any(not set(expected) <= p.keys() for p in pool):
                raise RuntimeError("Drum model missing configured kit components")
            if len(pool) == 2:
                avg = {k: (pool[0][k] + pool[1][k]) * .5 for k in expected}
                res.update({k: consensus_blend(pool[0][k], pool[1][k], sum(v for r, v in avg.items() if r != k)) for k in expected})
            else:
                res.update(pool[0])
            self._advance_progress("Blended kick, snare, hi-hat, toms, ride and crash")

    def _mega_separation_all(self, results):
        if not self.mega_stems:
            return
        self.separator.model_instance = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for res in results.values():
            with tempfile.TemporaryDirectory(dir=res["output_folder"]) as temp:
                source = Path(temp) / "input.wav"
                sf.write(source, stereo(res["instrumental"]).T, res["sr"], subtype="FLOAT")
                cmd = [sys.executable, "-m", "modules.separator.mega_worker", str(source), temp,
                       self.separator.model_file_dir, "--quality", self.options.get("separation_quality", "balanced")]
                if self.options.get("cpu"):
                    cmd.append("--cpu")
                subprocess.run(cmd, cwd=app_path, check=True)
                info = json.loads((Path(temp) / "mega.json").read_text())
                self.separator.run_records.extend(info["runs"])
                self.separator.loaded_models.add(MEGA_MODEL)
                for role in self.mega_stems:
                    audio, _ = librosa.load(str(Path(temp) / info["outputs"][role]), sr=res["sr"], mono=False)
                    res["mega_" + role.replace("-", "_")] = stereo(audio, res["instrumental"].shape[-1])
            self._advance_progress("Separated selected additional Mega instruments")

    def _woodwinds_separation_all(self, results):
        self.separator.load_model("17_HP-Wind_Inst-UVR.pth")
        for res in results.values():
            stems = self._separate_as_arrays_current(res["other"], res["sr"], output_folder=res["output_folder"], task="instruments")
            if "woodwinds" not in stems:
                raise RuntimeError("Woodwind model did not produce a recognized woodwind stem")
            res["woodwinds"] = stems["woodwinds"]
            res["other"] = res["other"] - res["woodwinds"]
            self._advance_progress("Separated woodwinds")

    def _save_all_stems(self, results):
        output_files = []
        labels = {"vocals": "Vocals", "vocals_full": "Vocals_Full", "bg_vocals": "BG_Vocals",
                  "instrumental": "Instrumental", "drums_hh": "Drums_HH"}
        manifests = {}
        fingerprint = model_fingerprint(self.separator.model_file_dir, self.separator.loaded_models)
        for base_name, res in results.items():
            folder = Path(res["output_folder"])
            manifest = manifests.setdefault(str(folder), {"pipeline": PIPELINE_REVISION,
                "audio_separator": version("audio-separator"), "experimental": self.separation_profile.value == "v4",
                "settings": {k: v for k, v in self.options.items() if k not in {"input_dict", "callback"}},
                "model_hashes": fingerprint, "custom_models": CUSTOM_MODELS,
                "runs": self.separator.run_records, "stems": []})
            candidates = {k: v for k, v in res.items() if k in INSTRUMENT_STEMS and isinstance(v, np.ndarray)}
            for key, arr in res.items():
                if not isinstance(arr, np.ndarray) or key == "mix_np":
                    continue
                parent_key = ("vocals_full" if key.startswith("bg_vocals") or key == "vocals" and "vocals_full" in res
                              else "drums" if key.startswith("drums_") else "instrumental"
                              if key in {"bass", "drums", "guitar", "piano", "other", "woodwinds"} or key.startswith("mega_") else "mix_np")
                if key in INSTRUMENT_STEMS and key not in self.instrument_stems:
                    continue
                if key.startswith("drums_") and key[6:] not in self.drum_stems:
                    continue
                hidden = False
                activity = {"hidden": False, "reason": "user-selected; automatic activity filtering disabled"}
                duplicate = duplicate_of(arr, candidates) if key.startswith("mega_") else None
                if key.startswith("mega_"):
                    candidates[key] = arr
                label = labels.get(key, "BG_Vocals_" + key.rsplit("_", 1)[-1] if key.startswith("bg_vocals_") else key.title())
                if duplicate:
                    label += "_Duplicate_of_" + duplicate.title()
                    logger.warning("%s is a near-identical copy of %s; retained as an alternative", key, duplicate)
                filename = f"{res.get('base_name', base_name)}__({label}).wav"
                relative = str(Path(".hidden_stems") / filename) if hidden else filename
                target = folder / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                sf.write(target, stereo(arr).T, res["sr"], subtype="FLOAT")
                manifest["stems"].append({"role": key, "parent": parent_key, "filename": filename,
                    "path": relative, "hidden": hidden, "activity": activity, "sha256": file_hash(target),
                    "aggregate": key in {"vocals_full", "instrumental", "drums", "bg_vocals"}})
                if duplicate:
                    manifest["stems"][-1]["duplicate_of"] = duplicate
                    manifest["stems"][-1]["note"] = "Near-identical copy retained for user review; excluded from automatic mixing"
                if key == "vocals" and (self.options.get("vocal_reverb") == "Capture reverb" or "vocal_reverb" not in self.options and self.store_reverb_ir and any(
                    self._should_apply_transform(key, flag) for flag in (self.reverb_removal, self.echo_removal)
                )):
                    capture_path = folder / (Path(filename).stem + ".ir")
                    if capture_path.exists():
                        manifest["stems"][-1]["reverb_ir"] = {
                            "path": capture_path.name, "sha256": file_hash(capture_path)}
                if not hidden:
                    output_files.append(str(target))
            self._advance_progress("Saved audio and Smart Stems manifest")
        for folder, manifest in manifests.items():
            write_json(Path(folder) / MANIFEST_NAME, manifest)
        return output_files

    @staticmethod
    def _should_apply_transform(stem_name: str, setting: str) -> bool:
        """
        Determines if a transform should be applied based on stem name and setting.

        Parameters:
            stem_name (str): The stem identifier (e.g., "(vocals)").
            setting (str): The transformation setting.

        Returns:
            bool: True if transform should be applied, False otherwise.
        """
        name = stem_name.lower().strip("()")
        if setting == "All":
            return True
        if setting == "All Vocals":
            return name in {"vocals", "vocals_full"} or name.startswith("bg_vocals")
        if setting == "Main Vocals":
            return name == "vocals"
        return False

    def _apply_bg_vocal_splitting(self, vocals_array, sr, base_name, output_folder):
        choice = self.options.get("backing_vocal_model", "karaoke")
        self.separator.load_model(BG_MODELS[choice])
        stems = self._separate_as_arrays_current(vocals_array, sr, output_folder=output_folder,
                                                 task="karaoke" if choice == "karaoke" else "bve")
        if not {"vocals", "bg_vocals"} <= stems.keys():
            raise RuntimeError("Backing vocal model did not return lead and backing vocals")
        self._advance_progress("Separated lead and backing vocals")
        return stems["vocals"], stems["bg_vocals"]

    def _apply_transform_chain(self, stem_array, sr, base_name, stem_label, output_folder, skip_transforms=None):
        mode = getattr(self, "options", {}).get("vocal_reverb", "Keep wet")
        if mode not in {"Keep wet", "Dry vocals", "Capture reverb"}:
            raise ValueError("Unknown vocal reverb mode")
        use_fused = stem_label == "vocals" and mode != "Keep wet"
        transformations = [
            ("dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt", "dry", self.reverb_removal),
            (self.delay_removal_model, "dry", self.echo_removal),
            (self.crowd_removal_model, "clean", self.crowd_removal),
            (self.noise_removal_model, "clean", self.noise_removal),
        ]
        current = stem_array
        if use_fused:
            self.separator.load_model(FUSED_DEREVERB)
            stems = self._separate_as_arrays_current(stem_array, sr, output_folder=output_folder, task="cleanup")
            if "dry" not in stems:
                raise RuntimeError("Sucial Fused did not produce a dry vocal")
            dry_vocal = stems["dry"]
            if mode == "Dry vocals":
                current = dry_vocal
            else:
                from modules.reverb_ir import save_capture
                ir_path = Path(output_folder) / f"{Path(base_name).name}__(Vocals).ir"
                with tempfile.TemporaryDirectory(dir=output_folder) as temp:
                    dry_path = write_temp_wav(dry_vocal, sr, temp)
                    wet_path = write_temp_wav(stem_array - dry_vocal, sr, temp)
                    extract_reverb(str(dry_path), wet_path, str(ir_path))
                capture = json.loads(ir_path.read_text())
                capture.update({"restore_requested": True, "exported_wet": True, "cloning_input": "wet", "model": FUSED_DEREVERB})
                save_capture(ir_path, capture)
                if not capture.get("valid_for_restore"):
                    logger.warning("Requested reverb capture retained with fit warning: %s", capture.get("rejection_reasons"))
            self._advance_progress("Dried vocals" if mode == "Dry vocals" else "Captured reverb; retaining wet vocals")
        effect_input = None
        effect_dry = None
        for model, role, flag in transformations:
            if stem_label == "vocals" and "vocal_reverb" in getattr(self, "options", {}) and role == "dry":
                continue  # Explicit new mode wins over saved legacy removal toggles.
            if not self._should_apply_transform(stem_label, flag):
                continue
            self.separator.load_model(model)
            stems = self._separate_as_arrays_current(current, sr, output_folder=output_folder, task="cleanup")
            if role not in stems:
                raise RuntimeError(f"Cleanup model {model} missing expected {role} output")
            if role == "dry" and self.store_reverb_ir and stem_label == "vocals":
                if effect_input is None:
                    effect_input = current.copy()
                effect_dry = stems[role]
            current = stems[role]
            self._advance_progress(f"Applied {model} to {stem_label}")
        if effect_input is not None:
            # Capture the combined reverb/echo removal once, before unrelated denoising.
            with tempfile.TemporaryDirectory(dir=output_folder) as temp:
                dry = write_temp_wav(effect_dry, sr, temp)
                wet = write_temp_wav(effect_input - effect_dry, sr, temp)
                ir_path = str(Path(output_folder) / f"{Path(base_name).name}__(Vocals).ir")
                extract_reverb(dry, wet, ir_path)
                capture = json.loads(Path(ir_path).read_text())
                if not capture.get("valid_for_restore"):
                    logger.warning("IR capture not reusable: %s", capture.get("rejection_reasons"))
        return current


################################################################################
#                    TOP-LEVEL PREDICTION + OUTPUT ROUTINE
################################################################################

def predict_with_model(options: Dict, callback: Callable = None) -> List[str]:
    """
    Loads input files, runs ensemble and additional processing, then saves stems.

    Parameters:
        options (Dict): Options for separation and transformation.
        callback (Callable, optional): Callback for progress updates.

    Returns:
        List[str]: List of output file paths.
    """
    input_dict = options["input_dict"]
    files_data = []
    for out_folder, input_files in input_dict.items():
        for ip in input_files:
            if not os.path.isfile(ip):
                continue
            loaded, sr = librosa.load(ip, sr=44100, mono=False)
            loaded = stereo(loaded)
            os.makedirs(out_folder, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(ip))[0]
            files_data.append({"base_name": base_name, "key": str(Path(out_folder).resolve() / base_name), "mix_np": loaded, "sr": sr, "output_folder": out_folder})
    if not files_data:
        return []
    model = EnsembleDemucsMDXMusicSeparationModel(options, callback)

    # Pre-calculate total steps for accurate progress tracking.
    N = len(files_data)
    # Get actual ensemble models from the separation profile
    profile_models = model.profile_models
    ensemble_steps = len(profile_models) * N
    bg_steps = N * int(options.get("bg_vocal_layers", 1)) if model.separate_bg_vocals else 0
    # Compute transformation steps per file based on settings.
    trans_opts = [model.reverb_removal, model.echo_removal, model.crowd_removal, model.noise_removal]
    count_vocals = sum(1 for opt in trans_opts if opt in {"All", "All Vocals", "Main Vocals"})
    count_instrumental = sum(1 for opt in trans_opts if opt == "All")
    transform_steps = (count_vocals + count_instrumental + (options.get("vocal_reverb", "Keep wet") != "Keep wet")) * N
    multi_stem_steps = N if not model.vocals_only else 0
    alt_bass_steps = N if (model.alt_bass_model and not model.vocals_only) else 0
    drum_steps = N if (model.separate_drums and not model.vocals_only) else 0
    ww_steps = N if (model.separate_woodwinds and not model.vocals_only) else 0
    saving_steps = 1 + N
    total_steps = ensemble_steps + bg_steps + transform_steps + multi_stem_steps + alt_bass_steps + drum_steps + ww_steps + saving_steps
    model.total_steps = total_steps

    if model.callback is not None:
        model.callback(0, "Starting ensemble separation...", model.total_steps)

    # Ensemble separation
    results = model._ensemble_separate_all(files_data)

    if model.separate_bg_vocals:
        for base_name, res in results.items():
            res["vocals_full"] = res["vocals"].copy()
            lead, backing = model._apply_bg_vocal_splitting(res["vocals"], res["sr"], base_name, res["output_folder"])
            res["vocals"], res["bg_vocals"] = lead, backing
            remainder = backing
            for layer in range(2, int(options.get("bg_vocal_layers", 1)) + 1):
                part, remainder = model._apply_bg_vocal_splitting(remainder, res["sr"], base_name, res["output_folder"])
                res[f"bg_vocals_{layer - 1}"] = part
                res[f"bg_vocals_{layer}"] = remainder

    # Only run multistem logic when not in vocals-only mode
    if not model.vocals_only and (model.instrument_stems or (model.separate_drums and model.drum_stems) or model.alt_bass_model or model.separate_woodwinds):
        model._multistem_separation_all(results)
        if model.alt_bass_model:
            model._alt_bass_separation_all(results)
        if model.separate_drums and model.drum_stems:
            model._advanced_drum_separation_all(results)
        if model.separate_woodwinds:
            model._woodwinds_separation_all(results)
    model._mega_separation_all(results)
    # Each requested cleanup runs exactly once per final stem, including backing vocals.
    for base_name, res in results.items():
        for key, arr in list(res.items()):
            if isinstance(arr, np.ndarray) and key != "mix_np":
                res[key] = model._apply_transform_chain(arr, res["sr"], res.get("base_name", base_name), key, res["output_folder"])
    output_files = model._save_all_stems(results)
    if model.callback:
        model.callback(1.0, "Separation complete", model.total_steps)
    return output_files


def separate_music(input_dict: Dict[str, List[str]], callback: Callable = None, **kwargs) -> List[str]:
    """
    Wrapper for calling the separation model.

    Example:
        separate_music(
            {"/output/folder": ["/path/to/file.mp3"]},
            callback=your_callback_function,
            cpu=False,
            vocals_only=False,
            separate_drums=True,
            separate_woodwinds=True,
            alt_bass_model=True,
            reverb_removal="Main Vocals",
            crowd_removal="Nothing",
            noise_removal="Nothing"
        )

    Parameters:
        input_dict (Dict[str, List[str]]): Dictionary mapping output folders to lists of input file paths.
        callback (Callable, optional): Progress callback.
        **kwargs: Additional separation and transformation options.

    Returns:
        List[str]: List of output file paths.
    """
    options = {
        "input_dict": input_dict,
        "cpu": kwargs.get("cpu", False),
        "separation_profile": kwargs.get("separation_profile", "hybrid_cleaned"),
        "ensemble_size": kwargs.get("ensemble_size", None),
        "residual_fill": kwargs.get("residual_fill", None),
        "vocals_only": kwargs.get("vocals_only", True),
        "use_VOCFT": kwargs.get("use_VOCFT", False),
        "separate_drums": kwargs.get("separate_drums", False),
        "separate_woodwinds": kwargs.get("separate_woodwinds", False),
        "alt_bass_model": kwargs.get("alt_bass_model", False),
        "weight_InstVoc": kwargs.get("weight_InstVoc", 8.0),
        "weight_VOCFT": kwargs.get("weight_VOCFT", 1.0),
        "weight_VitLarge": kwargs.get("weight_VitLarge", 5.0),
        "reverb_removal": kwargs.get("reverb_removal", "Nothing"),
        "echo_removal": kwargs.get("echo_removal", "Nothing"),
        "delay_removal": kwargs.get("delay_removal", "Nothing"),
        "crowd_removal": kwargs.get("crowd_removal", "Nothing"),
        "noise_removal": kwargs.get("noise_removal", "Nothing"),
        "delay_removal_model": kwargs.get("delay_removal_model", "dereverb-echo_mel_band_roformer_sdr_13.4843_v2.ckpt"),
        "noise_removal_model": kwargs.get("noise_removal_model", "UVR-DeNoise.pth"),
        "crowd_removal_model": kwargs.get("crowd_removal_model", "UVR-MDX-NET_Crowd_HQ_1.onnx"),
        "separate_bg_vocals": kwargs.get("separate_bg_vocals", True),
        "backing_vocal_model": kwargs.get("backing_vocal_model") or "karaoke",
        "bg_vocal_layers": kwargs.get("bg_vocal_layers", 1),
        "store_reverb_ir": kwargs.get("store_reverb_ir", False),
        "callback": callback,
        "ensemble_strength": kwargs.get("ensemble_strength", 2),
        "residual_blend": kwargs.get("residual_blend", 0.4)
    }
    for key in ("separation_quality", "smart_stems", "separation_preset", "backing_vocal_model", "vocal_reverb",
                "instrument_model", "instrument_input", "instrument_stems", "drum_stems", "mega_stems", "vocal_fusion", "instrumental_fusion"):
        if key in kwargs and kwargs[key] is not None:
            options[key] = kwargs[key]
    if options["delay_removal"] != "Nothing" and options["echo_removal"] == "Nothing":
        options["echo_removal"] = options["delay_removal"]
    return predict_with_model(options, callback)
