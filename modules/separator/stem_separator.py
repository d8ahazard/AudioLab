# coding: utf-8
import logging
import os
import subprocess
import uuid
import warnings
from typing import List, Dict, Callable, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
from audio_separator.separator import Separator

from handlers.config import app_path, output_path
from handlers.patch_separate import patch_separator
from handlers.reverb import extract_reverb

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
        cmd = ["ffmpeg", "-y", "-i", input_path, "-acodec", "pcm_s16le", "-ac", "2", "-ar", str(sr), out_wav]
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
    sf.write(tmp_path, wav_data, sr, format="WAV", subtype="PCM_16")
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
        patch_separator()
        self.separator = Separator(
            log_level=logging.ERROR,
            model_file_dir=os.path.join(app_path, "models", "audio_separator"),
            invert_using_spec=True,
            use_autocast=True
        )
        # Download all required models
        self.model_list = [
            "htdemucs_ft.yaml", "htdemucs.yaml", "hdemucs_mmi.yaml", "htdemucs_6s.yaml",
            "MDX23C-8KFFT-InstVoc_HQ.ckpt", "model_bs_roformer_ep_368_sdr_12.9628.ckpt",
            "UVR-MDX-NET-Voc_FT.onnx", "Kim_Vocal_1.onnx", "Kim_Vocal_2.onnx",
            "MDX23C-DrumSep-aufr33-jarredou.ckpt",
            "17_HP-Wind_Inst-UVR.pth",
            "kuielab_a_bass.onnx",
            # Added higher-fidelity candidates
            "vocals_mel_band_roformer.ckpt",
            "melband_roformer_big_beta4.ckpt",
            # Transform models
            "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",
            "dereverb-echo_mel_band_roformer_sdr_13.4843_v2.ckpt"
        ]
        for model in self.model_list:
            self.separator.download_model_files(model)

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
        self.store_reverb_ir = options.get("store_reverb_ir", False)
        self.ensemble_strength = options.get("ensemble_strength", 1)

        # Progress tracking
        self.global_step = 0
        self.total_steps = 0
        self.callback = options.get("callback", None)

    def _advance_progress(self, desc: str, weight: int = 1) -> None:
        """
        Increments progress by a given weight and calls the callback with the current progress.

        Parameters:
            desc (str): Description of the current progress step.
            weight (int): Weight (number of steps) to advance.
        """
        self.global_step += weight
        if self.callback is not None and self.total_steps > 0:
            self.callback(self.global_step / self.total_steps, desc, self.total_steps)
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
        peak = np.max(np.abs(combined))
        if peak > 0:
            combined /= peak
        return combined

    def _separate_as_arrays_current(self, mix_np: np.ndarray, sr: int, desc: str = None, output_folder: str = None) -> \
    Dict[str, np.ndarray]:
        """
        Runs separation for the current model and returns results as arrays.

        Parameters:
            mix_np (np.ndarray): Input mix as a numpy array.
            sr (int): Sample rate.
            desc (str): Optional description.
            output_folder (str): Folder for temporary outputs.

        Returns:
            Dict[str, np.ndarray]: Dictionary with separated stems.
        """
        tmp_wav = write_temp_wav(mix_np, sr, output_folder)
        if desc:
            logger.debug(desc)
        output_partial = self.separator.separate(tmp_wav)
        output_files = [os.path.join(output_folder, f) for f in output_partial]
        stems = {}
        # First pass: classify outputs using robust heuristics
        vocals_idx = None
        inst_idx = None
        lowered = [os.path.basename(f).lower() for f in output_files]
        # Strong instrumental indicators
        inst_tags = ["instrumental", "accompaniment", "no_vocals", "no vocals", "without vocals", "minus vocals", "inst"]
        # Negative qualifiers for vocals
        vocals_neg = ["no_vocals", "no vocals", "without vocals", "bg", "background", "backing"]
        for i, name in enumerate(lowered):
            if any(tag in name for tag in inst_tags):
                inst_idx = i
        for i, name in enumerate(lowered):
            if ("vocals" in name or "(vocals)" in name) and not any(neg in name for neg in vocals_neg):
                vocals_idx = i
                break
        # If still ambiguous and exactly 2 files, choose the non-instrumental as vocals
        if vocals_idx is None and len(output_files) == 2 and inst_idx is not None:
            other = 1 - inst_idx
            vocals_idx = other
        # If we have vocals but no explicit instrumental, pick the other file as instrumental when 2 outputs
        if vocals_idx is not None and inst_idx is None and len(output_files) == 2:
            inst_idx = 1 - vocals_idx
        # If neither detected, try again with simpler rules
        if vocals_idx is None:
            for i, name in enumerate(lowered):
                if "vocals" in name and "no vocals" not in name and "no_vocals" not in name and "bg" not in name:
                    vocals_idx = i
                    break
        if inst_idx is None:
            for i, name in enumerate(lowered):
                if any(tag in name for tag in inst_tags):
                    inst_idx = i
                    break
        # STRICT original mapping: rely only on explicit (Vocals)/(Instrumental) tags
        stems = {}
        vocals_path = None
        for file in output_files:
            arr, _ = librosa.load(file, sr=sr, mono=False)
            if arr.ndim == 1:
                arr = np.stack([arr, arr], axis=0)
            flow = file.lower()
            if "(vocals)" in flow:
                stems["vocals"] = arr
                vocals_path = file
            elif "(instrumental)" in flow:
                stems["instrumental"] = arr
        # Minimal, safe fallback: if instrumental not found by exact tag, accept common aliases
        if "instrumental" not in stems:
            for file in output_files:
                name = os.path.basename(file).lower()
                if any(tag in name for tag in ["accompaniment", "no_vocals", "no vocals", "without vocals", "minus vocals", " inst ", "_inst", "(inst)"]):
                    arr, _ = librosa.load(file, sr=sr, mono=False)
                    if arr.ndim == 1:
                        arr = np.stack([arr, arr], axis=0)
                    stems["instrumental"] = arr
                    break
        # Vocals-only rule: whatever file is NOT the vocals becomes the instrumental
        if self.vocals_only and "vocals" in stems and "instrumental" not in stems:
            for file in output_files:
                if vocals_path is not None and os.path.abspath(file) == os.path.abspath(vocals_path):
                    continue
                # First non-vocal output becomes instrumental
                arr, _ = librosa.load(file, sr=sr, mono=False)
                if arr.ndim == 1:
                    arr = np.stack([arr, arr], axis=0)
                stems["instrumental"] = arr
                break
        # No heuristics, no fallbacks here: strictly keep original behavior
        # Clean up temporary file
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)
        return stems

    def _ensemble_separate_all(self, files_data: List[Dict]) -> Dict[str, Dict]:
        """
        Performs ensemble separation on all files.

        Parameters:
            files_data (List[Dict]): List of file data dictionaries.

        Returns:
            Dict[str, Dict]: Dictionary mapping base names to separation results.
        """
        results = {}
        for file in files_data:
            base_name = file["base_name"]
            results[base_name] = {
                "mix_np": file["mix_np"],
                "sr": file["sr"],
                "vocals_list": [],
                "instrumental_list": [],
                "v_weights": [],
                "i_weights": [],
                "output_folder": file["output_folder"]
            }
        models_with_weights = [
            ("vocals_mel_band_roformer.ckpt", 8.6, 16.0),
            ("model_bs_roformer_ep_368_sdr_12.9628.ckpt", 8.4, 16.0),
            ("melband_roformer_big_beta4.ckpt", 8.5, 16.0),
            ("MDX23C-8KFFT-InstVoc_HQ.ckpt", 7.2, 14.9),
            ("UVR-MDX-NET-Voc_FT.onnx", 6.9, 14.9),
            ("Kim_Vocal_2.onnx", 6.9, 14.9),
            ("Kim_Vocal_1.onnx", 6.8, 14.9),
        ]
        # Avoid over-aggressive blending for very small ensembles which can leak vocals
        if self.ensemble_strength <= 2:
            self.options["residual_blend"] = min(float(self.options.get("residual_blend", 0.4)), 0.2)
        models_with_weights = models_with_weights[:self.ensemble_strength]

        for model_name, v_wt, i_wt in models_with_weights:
            self.separator.load_model(model_name)
            for file in files_data:
                base_name = file["base_name"]
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
                self._advance_progress(f"Ensemble model '{model_name}' processed for {base_name}.")
        for base_name, res in results.items():
            # Restore original, strict blending behavior
            res["vocals"] = self._blend_tracks(res["vocals_list"], res["v_weights"])
            res["instrumental"] = self._blend_tracks(res["instrumental_list"], res["i_weights"])
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
                    if sim_resid + 1e-6 < sim_inst - 0.01:  # requires a small but real improvement
                        blend = float(self.options.get("residual_blend", 0.4))
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

    def _multistem_separation_all(self, results: Dict[str, Dict]) -> None:
        """
        Runs 6-stem separation on the full mix for all files.

        Parameters:
            results (Dict[str, Dict]): Separation results.
        """
        self.separator.load_model("htdemucs_6s.yaml")
        for base_name, res in results.items():
            sr = res["sr"]
            mix_np = res.get("mix_np")
            if mix_np is None or mix_np.size == 0:
                # Fallback to instrumental if mix is unavailable
                mix_np = res.get("instrumental")
            if mix_np is None or mix_np.size == 0:
                mix_np = np.zeros((2, 1), dtype=np.float32)
            self.separator.output_dir = res["output_folder"]
            self.separator.model_instance.output_dir = res["output_folder"]

            tmp_mix_wav = write_temp_wav(mix_np, sr, res["output_folder"])
            demucs_partial = self.separator.separate(tmp_mix_wav)
            demucs_files = [os.path.join(self.separator.output_dir, f) for f in demucs_partial]
            res["drums"] = None
            res["bass"] = None
            res["guitar"] = None
            res["piano"] = None
            res["other"] = None
            for f in demucs_files:
                lowf = os.path.basename(f).lower()
                arr, _ = librosa.load(f, sr=sr, mono=False)
                if arr.ndim == 1:
                    arr = np.stack([arr, arr], axis=0)
                if ("(drums)" in lowf) or ("drums" in lowf) or ("drum" in lowf):
                    res["drums"] = arr
                elif ("(bass)" in lowf) or ("bass" in lowf):
                    res["bass"] = arr
                elif ("(guitar)" in lowf) or ("guitar" in lowf):
                    res["guitar"] = arr
                elif ("(piano)" in lowf) or ("piano" in lowf):
                    res["piano"] = arr
                elif ("(other)" in lowf) or ("other" in lowf) or ("accompaniment" in lowf) or ("rest" in lowf):
                    res["other"] = arr
            if os.path.exists(tmp_mix_wav):
                os.remove(tmp_mix_wav)
            self._advance_progress(f"6-stem separation completed for {base_name}.")

    def _alt_bass_separation_all(self, results: Dict[str, Dict]) -> None:
        """
        Applies an alternate bass separation model on all files.

        Parameters:
            results (Dict[str, Dict]): Separation results.
        """
        self.separator.load_model("kuielab_a_bass.onnx")
        for base_name, res in results.items():
            sr = res["sr"]
            inst = res["instrumental"]
            self.separator.output_dir = res["output_folder"]
            self.separator.model_instance.output_dir = res["output_folder"]
            tmp_instru_wav = write_temp_wav(inst, sr, res["output_folder"])
            alt_bass_out = self.separator.separate(tmp_instru_wav)
            alt_bass_files = [os.path.join(self.separator.output_dir, f) for f in alt_bass_out]
            for bfile in alt_bass_files:
                if not os.path.exists(bfile):
                    continue
                blow = os.path.basename(bfile).lower()
                arrb, _ = librosa.load(bfile, sr=sr, mono=False)
                if arrb.ndim == 1:
                    arrb = np.stack([arrb, arrb], axis=0)
                if "(bass)" in blow:
                    res["bass"] = arrb
            if os.path.exists(tmp_instru_wav):
                os.remove(tmp_instru_wav)
            self._advance_progress(f"Alternate bass separation done for {base_name}.")

    def _advanced_drum_separation_all(self, results: Dict[str, Dict]) -> None:
        """
        Runs advanced drum separation on the drums stem for all files.

        Parameters:
            results (Dict[str, Dict]): Separation results.
        """
        self.separator.load_model("MDX23C-DrumSep-aufr33-jarredou.ckpt")
        for base_name, res in results.items():
            sr = res["sr"]
            drums = res.get("drums", np.zeros_like(res["instrumental"]))
            tmp_drums_wav = write_temp_wav(drums, sr, res["output_folder"])
            output_folder = res["output_folder"]
            self.separator.output_dir = output_folder
            self.separator.model_instance.output_dir = output_folder

            drum_parts = self.separator.separate(tmp_drums_wav)
            drum_part_files = [os.path.join(self.separator.output_dir, f) for f in drum_parts]
            drums_other = np.copy(drums)
            for key in ["drums_kick", "drums_snare", "drums_toms", "drums_hh", "drums_ride", "drums_crash"]:
                res[key] = None
            for dpf in drum_part_files:
                dplow = os.path.basename(dpf).lower()
                arrp, _ = librosa.load(dpf, sr=sr, mono=False)
                if arrp.ndim == 1:
                    arrp = np.stack([arrp, arrp], axis=0)
                if arrp.shape[-1] <= drums_other.shape[-1]:
                    # Gain-matched, time-aligned subtraction to reduce hiss
                    drums_other[:, :arrp.shape[-1]] = self._residual_subtract(drums_other[:, :arrp.shape[-1]], arrp, sr)
                if "(kick)" in dplow:
                    res["drums_kick"] = arrp
                elif "(snare)" in dplow:
                    res["drums_snare"] = arrp
                elif "(toms)" in dplow:
                    res["drums_toms"] = arrp
                elif "(hh)" in dplow:
                    res["drums_hh"] = arrp
                elif "(ride)" in dplow:
                    res["drums_ride"] = arrp
                elif "(crash)" in dplow:
                    res["drums_crash"] = arrp
            # Only set stems that were actually detected; otherwise derive reasonable defaults
            if res.get("drums") is None:
                res["drums"] = np.zeros_like(drums)
            if res.get("bass") is None:
                res["bass"] = np.zeros_like(drums)
            if res.get("guitar") is None:
                res["guitar"] = np.zeros_like(drums)
            if res.get("piano") is None:
                res["piano"] = np.zeros_like(drums)
            if res.get("other") is None:
                res["other"] = np.zeros_like(drums)
            res["drums_other"] = drums_other
            self._advance_progress(f"Advanced drum separation done for {base_name}.")

    def _woodwinds_separation_all(self, results: Dict[str, Dict]) -> None:
        """
        Separates woodwinds from the 'other' stem on all files.

        Parameters:
            results (Dict[str, Dict]): Separation results.
        """
        self.separator.load_model("17_HP-Wind_Inst-UVR.pth")
        for base_name, res in results.items():
            sr = res["sr"]
            other = res.get("other", np.zeros_like(res["instrumental"]))
            tmp_other_wav = write_temp_wav(other, sr, res["output_folder"])
            output_folder = res["output_folder"]
            self.separator.output_dir = output_folder
            self.separator.model_instance.output_dir = output_folder

            ww_parts = self.separator.separate(tmp_other_wav)
            ww_part_files = [os.path.join(self.separator.output_dir, f) for f in ww_parts]
            new_woodwinds = np.zeros_like(other)
            for wfile in ww_part_files:
                if not os.path.exists(wfile):
                    continue
                wflow = os.path.basename(wfile).lower()
                arrw, _ = librosa.load(wfile, sr=sr, mono=False)
                if arrw.ndim == 1:
                    arrw = np.stack([arrw, arrw], axis=0)
                if "(woodwinds)" in wflow:
                    new_woodwinds = arrw
            leftover_other = np.copy(other)
            if new_woodwinds.shape[-1] <= leftover_other.shape[-1]:
                # Gain-matched, time-aligned subtraction to reduce hiss
                leftover_other[:, :new_woodwinds.shape[-1]] = self._residual_subtract(leftover_other[:, :new_woodwinds.shape[-1]], new_woodwinds, sr)
            res["woodwinds"] = new_woodwinds
            res["other"] = leftover_other
            self._advance_progress(f"Woodwinds separated for {base_name}.")

    def _save_all_stems(self, results: Dict[str, Dict]) -> List[str]:
        """
        Saves all final stems to disk and cleans up temporary files.

        Parameters:
            results (Dict[str, Dict]): Separation results.

        Returns:
            List[str]: List of output file paths.
        """
        self._advance_progress("Saving all stems...")
        output_files = []
        stem_names = {
            "vocals": "(Vocals)",
            "bg_vocals": "(BG_Vocals)",
            "instrumental": "(Instrumental)",
            "drums": "(Drums)",
            "bass": "(Bass)",
            "guitar": "(Guitar)",
            "piano": "(Piano)",
            "woodwinds": "(Woodwinds)",
            "other": "(Other)",
            "drums_kick": "(Drums_Kick)",
            "drums_snare": "(Drums_Snare)",
            "drums_toms": "(Drums_Toms)",
            "drums_hh": "(Drums_HH)",
            "drums_ride": "(Drums_Ride)",
            "drums_crash": "(Drums_Crash)",
            "drums_other": "(Drums_Other)"
        }
        for base_name, res in results.items():
            sr = res["sr"]
            output_folder = res["output_folder"]
            for stem_key, label in stem_names.items():
                if stem_key in res and res[stem_key] is not None:
                    # Skip writing stems that are effectively silent (prevents empty files)
                    arr = res[stem_key]
                    if isinstance(arr, np.ndarray) and float(np.max(np.abs(arr)) if arr.size > 0 else 0.0) < 1e-6:
                        continue
                    if stem_key == "bg_vocals" and "bg_vocals_" in base_name:
                        bg_int = int(base_name.split("bg_vocals_")[-1])
                        label = f"(BG_Vocals_{bg_int})"
                    output_name = f"{base_name}__{label}.wav"
                    output_path_file = os.path.join(output_folder, output_name)
                    sf.write(output_path_file, res[stem_key].T, sr, subtype="FLOAT")
                    output_files.append(output_path_file)
            self._advance_progress(f"Stems saved for {base_name}.")
        for base_name, res in results.items():
            output_folder = res["output_folder"]
            for temp_file in os.listdir(output_folder):
                if temp_file.startswith("tmp_"):
                    os.remove(os.path.join(output_folder, temp_file))
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
        if setting == "Nothing":
            return False
        if setting == "All":
            return True
        if setting == "All Vocals":
            return "vocals)" in stem_name.lower()
        if setting == "Main Vocals":
            return "vocals)" in stem_name and "(bg_vocals" not in stem_name.lower()
        return False

    @staticmethod
    def _rename_file(base_in: str, filepath: str) -> str:
        """
        Rebuilds the filename to remove model references.

        Parameters:
            base_in (str): The base input filename.
            filepath (str): The current file path.

        Returns:
            str: The new file path after renaming.
        """
        dirname = os.path.dirname(filepath)
        ext = os.path.splitext(filepath)[1]
        base_only = os.path.splitext(os.path.basename(base_in))[0]
        import re
        all_parens = re.findall(r"\([^)]*\)", os.path.basename(filepath))
        to_strip = [
            "deverb_bs_roformer", "UVR-DeEcho-DeReverb", "UVR-De-Echo-Normal",
            "UVR-DeNoise", "UVR-DeNoise-Lite", "mel_band_roformer", "MDX23C", "UVR-MDX-NET",
            "drumsep", "roformer", "viperx", "crowd", "karaoke", "instrumental", "_InstVoc", "_VOCFT",
            "NoReverb", "NoEcho", "NoDelay", "NoCrowd", "NoNoise", "_mel_band_roformer_karaoke_aufr33_viperx_sdr_10"
        ]
        filtered = []
        for g in all_parens:
            if not any(p.lower() in g.lower() for p in to_strip):
                filtered.append(g)
        final_name = base_only + "_" + "".join(filtered) + ext
        final_name = final_name.replace(") (", ")(").replace("__", "_")
        final_path = os.path.join(dirname, final_name)
        if os.path.exists(filepath):
            if os.path.exists(final_path):
                os.remove(final_path)
            os.rename(filepath, final_path)
        return final_path

    def _apply_bg_vocal_splitting(self, vocals_array: np.ndarray, sr: int, base_name: str, output_folder: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies background vocal splitting to the vocals array.

        Parameters:
            vocals_array (np.ndarray): The vocal stem.
            sr (int): Sample rate.
            base_name (str): Base filename.
            output_folder (str): Folder to store outputs.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (Main vocals, background vocals) if successful;
                                           otherwise, returns (original vocals_array, None).
        """
        tmp_file = write_temp_wav(vocals_array, sr, output_folder)
        self.separator.load_model("UVR-BVE-4B_SN-44100-1.pth")
        self.separator.output_dir = output_folder
        self.separator.model_instance.output_dir = output_folder
        out_files = self.separator.separate(tmp_file)
        self._advance_progress("Background vocal splitting executed.")
        out_files = [os.path.join(output_folder, f) for f in out_files]

        bg = None
        main = None
        for f in out_files:
            arr, _ = librosa.load(f, sr=sr, mono=False)
            if "(Vocals)" in f:
                bg = arr
            elif "(Instrumental)" in f:
                main = arr
        if os.path.exists(tmp_file):
            os.remove(tmp_file)
        if bg is not None and main is not None:
            if np.max(np.abs(bg)) > 0.0:
                return main, bg
            else:
                logger.info("Background vocals are empty after splitting.")
                return vocals_array, None
        return vocals_array, None

    def _apply_transform_chain(self, stem_array: np.ndarray, sr: int, base_name: str, stem_label: str,
                               output_folder: str, skip_transforms: List[str] = None) -> np.ndarray:
        """
        Applies a series of transformations (reverb, crowd, noise removal) to a stem array.

        Parameters:
            stem_array (np.ndarray): Audio stem to process.
            sr (int): Sample rate.
            base_name (str): Base filename.
            stem_label (str): Label for the stem (e.g., "vocals", "instrumental").
            output_folder (str): Output folder for temporary files.
            skip_transforms (List[str], optional): List of transformation labels to skip.

        Returns:
            np.ndarray: Transformed audio stem.
        """
        if skip_transforms is None:
            skip_transforms = []
        transformations = [
            ("dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt", "No Reverb", self.reverb_removal),
            (self.delay_removal_model, "dry", self.echo_removal),
            (self.crowd_removal_model, "No Crowd", self.crowd_removal),
            (self.noise_removal_model, "No Noise", self.noise_removal),
        ]
        current_array = stem_array
        simulated_name = f"({stem_label})"
        for model_file, out_label, transform_flag in transformations:
            if out_label in skip_transforms:
                continue
            if self._should_apply_transform(simulated_name, transform_flag):
                self.separator.load_model(model_file)
                self.separator.output_dir = output_folder
                self.separator.model_instance.output_dir = output_folder
                tmp_file = write_temp_wav(current_array, sr, output_folder)
                out_files = self.separator.separate(tmp_file)
                out_files_full = [os.path.join(output_folder, f) for f in out_files]
                chosen_file = None
                if len(out_files_full) == 2:
                    if out_label.replace(" ", "").lower() in out_files_full[0].replace(" ", "").lower():
                        chosen_file = out_files_full[0]
                        alt_file = out_files_full[1]
                    else:
                        chosen_file = out_files_full[1]
                        alt_file = out_files_full[0]
                    chosen_file = self._rename_file(base_name, chosen_file)
                    if (
                            out_label == "No Echo" or out_label == "No Reverb") and stem_label.lower() == "vocals" and self.store_reverb_ir and alt_file:
                        try:
                            out_ir = os.path.join(output_folder, "impulse_response.ir")
                            logger.info(f"Extracting reverb IR from {os.path.basename(alt_file)}")
                            extract_reverb(chosen_file, alt_file, out_ir)
                        except Exception as e:
                            logger.error(f"Error extracting IR: {e}")
                else:
                    for pf in out_files_full:
                        if out_label.replace(" ", "").lower() in pf.replace(" ", "").lower():
                            chosen_file = self._rename_file(base_name, pf)
                            break
                if chosen_file:
                    current_array, _ = librosa.load(chosen_file, sr=sr, mono=False)
                self._advance_progress(f"TRANSFORM: {out_label} on {stem_label} for {base_name}")
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)
        return current_array


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
            wav_path = ensure_wav(ip)
            loaded, sr = librosa.load(wav_path, sr=44100, mono=False)
            base_name = os.path.splitext(os.path.basename(ip))[0]
            files_data.append({"base_name": base_name, "mix_np": loaded, "sr": sr, "output_folder": out_folder})
    if not files_data:
        return []
    model = EnsembleDemucsMDXMusicSeparationModel(options, callback)

    # Pre-calculate total steps for accurate progress tracking.
    N = len(files_data)
    ensemble_models = [
        ("vocals_mel_band_roformer.ckpt", 8.6, 16.0),
        ("model_bs_roformer_ep_368_sdr_12.9628.ckpt", 8.4, 16.0),
        ("melband_roformer_big_beta4.ckpt", 8.5, 16.0),
        ("MDX23C-8KFFT-InstVoc_HQ.ckpt", 7.2, 14.9),
        ("UVR-MDX-NET-Voc_FT.onnx", 6.9, 14.9),
    ]
    ensemble_models = ensemble_models[:model.ensemble_strength]
    ensemble_steps = len(ensemble_models) * N
    bg_steps = N if model.separate_bg_vocals else 0
    # Compute transformation steps per file based on settings.
    trans_opts = [model.reverb_removal, model.crowd_removal, model.noise_removal]
    count_vocals = sum(1 for opt in trans_opts if opt in {"All", "All Vocals", "Main Vocals"})
    count_instrumental = sum(1 for opt in trans_opts if opt == "All")
    transform_steps = (count_vocals + count_instrumental) * N
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

    # --- NEW ORDER: Apply reverb removal on vocals BEFORE background splitting ---
    if model.reverb_removal != "Nothing" and "vocals" in next(iter(results.values())):
        for base_name, res in results.items():
            if res.get("vocals") is not None:
                # Apply reverb removal transform on vocals only
                res["vocals"] = model._apply_transform_chain(
                    res["vocals"], res["sr"], base_name, "vocals", res["output_folder"]
                )

    # Background vocal splitting comes after reverb removal.
    if model.separate_bg_vocals:
        for base_name, res in results.items():
            if "vocals" in res and res["vocals"] is not None:
                main_vocals, bg_vocals = model._apply_bg_vocal_splitting(
                    res["vocals"], res["sr"], base_name, res["output_folder"]
                )
                res["vocals"] = main_vocals
                if bg_vocals is not None:
                    res["bg_vocals"] = bg_vocals

    # Apply remaining transformation chain:
    # For vocals, skip reverb removal as it was already applied.
    if any(opt != "Nothing" for opt in [model.crowd_removal, model.noise_removal]):
        for base_name, res in results.items():
            if res.get("vocals") is not None:
                res["vocals"] = model._apply_transform_chain(
                    res["vocals"], res["sr"], base_name, "vocals", res["output_folder"], skip_transforms=["No Reverb"]
                )
            if res.get("instrumental") is not None:
                res["instrumental"] = model._apply_transform_chain(
                    res["instrumental"], res["sr"], base_name, "instrumental", res["output_folder"]
                )

    # Only run multistem logic when not in vocals-only mode
    if not model.vocals_only:
        model._multistem_separation_all(results)
        if model.alt_bass_model:
            model._alt_bass_separation_all(results)
        if model.separate_drums:
            model._advanced_drum_separation_all(results)
        if model.separate_woodwinds:
            model._woodwinds_separation_all(results)
    output_files = model._save_all_stems(results)
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
        "bg_vocal_layers": kwargs.get("bg_vocal_layers", 1),
        "store_reverb_ir": kwargs.get("store_reverb_ir", False),
        "callback": callback,
        "ensemble_strength": kwargs.get("ensemble_strength", 2),
        "residual_blend": kwargs.get("residual_blend", 0.4)
    }
    return predict_with_model(options, callback)


def debug_ensemble(tgt_file):
    import time
    model = EnsembleDemucsMDXMusicSeparationModel({}, None)
    base_name = os.path.splitext(os.path.basename(tgt_file))[0]
    loaded, sr = librosa.load(tgt_file, sr=44100, mono=False)

    for i in range(1, 6):
        out_dir = os.path.join(output_path, "ensemble_debug")
        start = time.time()
        model.ensemble_strength = i
        out_dir = os.path.join(out_dir, str(model.ensemble_strength))
        os.makedirs(out_dir, exist_ok=True)
        file_data = {"base_name": base_name, "mix_np": loaded, "sr": sr, "output_folder": out_dir}
        results = model._ensemble_separate_all([file_data])
        model._save_all_stems(results)
        print(f"Time taken for {i} models: {time.time() - start}")


def debug_bg_sep(tgt_file):
    import time
    separator = Separator(
        log_level=logging.ERROR,
        model_file_dir=os.path.join(app_path, "models", "audio_separator"),
        invert_using_spec=True,
        use_autocast=True
    )
    bg_models = [
        "MelBandRoformerSYHFT.ckpt",
        "model_chorus_bs_roformer_ep_267_sdr_24.1275.ckpt",
        "kuielab_a_other.onnx",
        "kuielab_b_other.onnx",
    ]
    out_dir = os.path.join(output_path, "bg_sep_debug")
    os.makedirs(out_dir, exist_ok=True)
    separator.output_dir = out_dir
    for bg_model in bg_models:
        separator.load_model(bg_model)
        start = time.time()
        main, bg_vox = separator.separate(tgt_file)
        print(f"Time taken for {bg_model}: {time.time() - start}")


def debug_reverb(tgt_file):
    reverb_models = [
        ("UVR-De-Echo-Aggressive.pth", "No Echo", "Echo"),
        ("UVR-De-Echo-Normal.pth", "No Echo", "Echo"),
        ("UVR-DeEcho-DeReverb.pth", "No reverb", "Reverb"),
        ("MDX23C-De-Reverb-aufr33-jarredou.ckpt", "dry", "No dry"),
        ("dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt", "noreverb", "reverb"),
        ("dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt", "noreverb", "reverb"),
        ("dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt", "dry", "No dry"),
        ("dereverb-echo_mel_band_roformer_sdr_13.4843_v2.ckpt", "dry", "No dry"),
        ("Reverb_HQ_By_FoxJoy.onnx", "No Reverb", "Reverb"),
    ]
    import time
    from audio_separator.separator import Separator
    separator = Separator(
        log_level=logging.ERROR,
        model_file_dir=os.path.join(app_path, "models", "audio_separator"),
        invert_using_spec=True,
        use_autocast=True
    )
    output_dir = os.path.join(output_path, "reverb_debug")
    os.makedirs(output_dir, exist_ok=True)
    separator.output_dir = output_dir
    # Delete all existing files in output_dir
    for f in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, f))

    for (model_file, dry_string, wet_string) in reverb_models:
        print(f"Running reverb model: {model_file}")
        separator.load_model(model_file)
        start_time = time.time()
        output_files = separator.separate(tgt_file)
        output_files = [os.path.join(output_dir, f) for f in output_files]
        dry_file = None
        wet_file = None
        for f in output_files:
            if dry_string in f:
                dry_file = os.path.join(output_dir, f)
            if wet_string in f:
                wet_file = os.path.join(output_dir, f)
        if not dry_file or not wet_file:
            print(f"Couldn't find files for {dry_string} and {wet_string} in {output_files}")
        print(f"Time taken for {model_file}: {time.time() - start_time}")
