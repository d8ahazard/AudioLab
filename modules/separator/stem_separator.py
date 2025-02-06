# coding: utf-8
import logging
import os
import subprocess
import uuid
import warnings
from typing import List, Dict, Callable

import librosa
import numpy as np
import soundfile as sf
import torch

from handlers.config import app_path

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


################################################################################
#                        HELPER UTILITY FUNCTIONS
################################################################################

def ensure_wav(input_path: str, sr: int = 44100) -> str:
    """ Convert input file to WAV format if necessary. """
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
    """ Writes mix_np to a PCM_16 .wav in out_dir. """
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
    A multi-model ensemble-based separation approach. This version processes all input
    files with each model before moving on to the next stage.
    """
    def __init__(self, options: Dict, callback: Callable = None):
        from audio_separator.separator import Separator

        self.options = options
        self.device = torch.device("cuda:0") if torch.cuda.is_available() and not options.get("cpu", False) \
                                             else torch.device("cpu")

        self.separator = Separator(
            log_level=logging.ERROR,
            model_file_dir=os.path.join(app_path, "models", "audio_separator"),
            output_dir=options["output_folder"]
        )

        # All models we might need:
        self.model_list = [
            "htdemucs_ft.yaml", "htdemucs.yaml", "hdemucs_mmi.yaml", "htdemucs_6s.yaml",
            "MDX23C-8KFFT-InstVoc_HQ.ckpt", "model_bs_roformer_ep_368_sdr_12.9628.ckpt",
            "UVR-MDX-NET-Voc_FT.onnx", "Kim_Vocal_1.onnx", "Kim_Vocal_2.onnx",
            "MDX23C-DrumSep-aufr33-jarredou.ckpt",
            "17_HP-Wind_Inst-UVR.pth",     # For optional woodwind
            "kuielab_a_bass.onnx"          # For optional alternate bass
        ]
        for model in self.model_list:
            self.separator.download_model_files(model)

        # Flags
        self.vocals_only = bool(options.get("vocals_only", False))
        self.separate_drums = bool(options.get("separate_drums", False))
        self.separate_woodwinds = bool(options.get("separate_woodwinds", False))
        self.alt_bass_model = bool(options.get("alt_bass_model", False))
        self.use_vocft = bool(options.get("use_VOCFT", False))

        # Weighted blending for vocals
        self.weight_inst = float(options.get("weight_InstVoc", 8.0))
        self.weight_vocft = float(options.get("weight_VOCFT", 1.0))
        self.weight_rof = float(options.get("weight_VitLarge", 5.0))

        # Overlap values for advanced separation (retained for reference)
        self.overlap_large = options.get("overlap_large", 0.6)
        self.overlap_small = options.get("overlap_small", 0.5)

        # Progress tracking
        self.global_step = 0
        self.total_steps = 0
        self.callback = options.get("callback", None)

    def _advance_progress(self, desc: str):
        """
        Increments self.global_step by 1 and calls the callback if defined.
        Callback signature: callback(current_progress, description, total_steps).
        """
        self.global_step += 1
        if self.callback is not None:
            self.callback(self.global_step / self.total_steps, desc, self.total_steps)
        logger.info(f"[{self.global_step}/{self.total_steps}] {desc}")

    def _blend_tracks(self, tracks, weights):
        """
        Blends a list of tracks given corresponding weights.
        """
        max_length = max(t.shape[-1] for t in tracks)
        combined = np.zeros((tracks[0].shape[0], max_length), dtype=np.float32)
        total_weight = sum(weights)
        for idx, t in enumerate(tracks):
            combined[:, :t.shape[-1]] += t
        combined = combined / total_weight
        peak = np.max(np.abs(combined))
        if peak > 0:
            combined /= peak
        return combined

    def _separate_as_arrays_current(self, mix_np, sr, desc=None):
        """
        Runs separation for the currently loaded model and returns results as arrays.
        (Assumes the model has already been loaded.)
        """
        tmp_wav = write_temp_wav(mix_np, sr, self.options["output_folder"])
        if desc:
            logger.debug(desc)
        output_partial = self.separator.separate(tmp_wav)
        output_files = [os.path.join(self.separator.output_dir, f) for f in output_partial]
        stems = {}
        for file in output_files:
            arr, _ = librosa.load(file, sr=sr, mono=False)
            if arr.ndim == 1:
                arr = np.stack([arr, arr], axis=0)
            flow = file.lower()
            if "(vocals)" in flow:
                stems["vocals"] = arr
            elif "(instrumental)" in flow:
                stems["instrumental"] = arr
            # Other keys are not used in ensemble stage.
        return stems

    def _ensemble_separate_all(self, files_data: List[Dict]) -> Dict[str, Dict]:
        """
        Performs ensemble separation (vocals + instrumental) on all files.
        Returns a dict keyed by base_name with the blended results.
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
                "i_weights": []
            }
        models_with_weights = [
            ("model_bs_roformer_ep_368_sdr_12.9628.ckpt", 8.4, 16.0),
            ("MDX23C-8KFFT-InstVoc_HQ.ckpt", 7.2, 14.9),
            ("UVR-MDX-NET-Voc_FT.onnx", 6.9, 14.9),
            ("Kim_Vocal_2.onnx", 6.9, 14.9),
            ("Kim_Vocal_1.onnx", 6.8, 14.9),
        ]
        for model_name, v_wt, i_wt in models_with_weights:
            self.separator.load_model(model_name)
            for file in files_data:
                base_name = file["base_name"]
                mix_np = file["mix_np"]
                sr = file["sr"]
                desc = f"[Ensemble] {base_name} => {model_name}"
                separated = self._separate_as_arrays_current(mix_np, sr, desc)
                vstem = separated.get("vocals", np.zeros_like(mix_np))
                istem = separated.get("instrumental", np.zeros_like(mix_np))
                results[base_name]["vocals_list"].append(vstem)
                results[base_name]["instrumental_list"].append(istem)
                results[base_name]["v_weights"].append(v_wt)
                results[base_name]["i_weights"].append(i_wt)
                self._advance_progress(f"Ensemble model {model_name} done for {base_name}")
        # Blend the ensemble results for each file.
        for base_name, res in results.items():
            res["vocals"] = self._blend_tracks(res["vocals_list"], res["v_weights"])
            res["instrumental"] = self._blend_tracks(res["instrumental_list"], res["i_weights"])
        return results

    def _multistem_separation_all(self, results: Dict[str, Dict]):
        """
        Runs 6-stem separation (drums, bass, guitar, piano, other) on the instrumental track for all files.
        """
        self.separator.load_model("htdemucs_6s.yaml")
        for base_name, res in results.items():
            sr = res["sr"]
            inst = res["instrumental"]
            tmp_instru_wav = write_temp_wav(inst, sr, self.options["output_folder"])
            demucs_partial = self.separator.separate(tmp_instru_wav)
            demucs_files = [os.path.join(self.separator.output_dir, f) for f in demucs_partial]

            # Initialize multi-stem keys.
            res["drums"] = np.zeros_like(inst)
            res["bass"] = np.zeros_like(inst)
            res["guitar"] = np.zeros_like(inst)
            res["piano"] = np.zeros_like(inst)
            res["other"] = np.zeros_like(inst)

            for f in demucs_files:
                lowf = os.path.basename(f).lower()
                arr, _ = librosa.load(f, sr=sr, mono=False)
                if arr.ndim == 1:
                    arr = np.stack([arr, arr], axis=0)
                if "(drums)" in lowf:
                    res["drums"] = arr
                elif "(bass)" in lowf:
                    res["bass"] = arr
                elif "(guitar)" in lowf:
                    res["guitar"] = arr
                elif "(piano)" in lowf:
                    res["piano"] = arr
                elif "(other)" in lowf:
                    res["other"] = arr
            self._advance_progress(f"6-stem separation done for {base_name}")

    def _alt_bass_separation_all(self, results: Dict[str, Dict]):
        """
        Applies an alternate bass separation model on all files to override the bass stem.
        """
        self.separator.load_model("kuielab_a_bass.onnx")
        for base_name, res in results.items():
            sr = res["sr"]
            inst = res["instrumental"]
            tmp_instru_wav = write_temp_wav(inst, sr, self.options["output_folder"])
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
                    res["bass"] = arrb  # override the previous bass
            self._advance_progress(f"Alt Bass separation done for {base_name}")

    def _advanced_drum_separation_all(self, results: Dict[str, Dict]):
        """
        Runs advanced drum separation on the drums stem for all files.
        """
        self.separator.load_model("MDX23C-DrumSep-aufr33-jarredou.ckpt")
        for base_name, res in results.items():
            sr = res["sr"]
            drums = res.get("drums", np.zeros_like(res["instrumental"]))
            tmp_drums_wav = write_temp_wav(drums, sr, self.options["output_folder"])
            drum_parts = self.separator.separate(tmp_drums_wav)
            drum_part_files = [os.path.join(self.separator.output_dir, f) for f in drum_parts]
            drums_other = np.copy(drums)
            # Initialize extra drum parts
            for key in ["drums_kick", "drums_snare", "drums_toms", "drums_hh", "drums_ride", "drums_crash"]:
                res[key] = None
            for dpf in drum_part_files:
                dplow = os.path.basename(dpf).lower()
                arrp, _ = librosa.load(dpf, sr=sr, mono=False)
                if arrp.ndim == 1:
                    arrp = np.stack([arrp, arrp], axis=0)
                if arrp.shape[-1] <= drums_other.shape[-1]:
                    drums_other[:, :arrp.shape[-1]] -= arrp
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
            res["drums_other"] = drums_other
            self._advance_progress(f"Advanced drum separation done for {base_name}")

    def _woodwinds_separation_all(self, results: Dict[str, Dict]):
        """
        Separates woodwinds from the 'other' stem on all files.
        """
        self.separator.load_model("17_HP-Wind_Inst-UVR.pth")
        for base_name, res in results.items():
            sr = res["sr"]
            other = res.get("other", np.zeros_like(res["instrumental"]))
            tmp_other_wav = write_temp_wav(other, sr, self.options["output_folder"])
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
                leftover_other[:, :new_woodwinds.shape[-1]] -= new_woodwinds
            res["woodwinds"] = new_woodwinds
            res["other"] = leftover_other
            self._advance_progress(f"Woodwinds separation done for {base_name}")

    def _save_all_stems(self, results: Dict[str, Dict]) -> List[str]:
        """
        Saves all final stems to disk for every file and cleans up temporary files.
        """
        self._advance_progress("Saving final stems")
        output_files = []
        stem_names = {
            "vocals": "(Vocals)",
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
        output_folder = self.options["output_folder"]
        for base_name, res in results.items():
            sr = res["sr"]
            for stem_key, label in stem_names.items():
                if stem_key in res and res[stem_key] is not None:
                    output_name = f"{base_name}_{label}.wav"
                    output_path = os.path.join(output_folder, output_name)
                    sf.write(output_path, res[stem_key].T, sr, subtype="FLOAT")
                    output_files.append(output_path)
            self._advance_progress(f"Stems saved for {base_name}")
        # Clean up any temporary files.
        for temp_file in os.listdir(output_folder):
            if temp_file.startswith("tmp_"):
                os.remove(os.path.join(output_folder, temp_file))
        return output_files


################################################################################
#                    TOP-LEVEL PREDICTION + OUTPUT ROUTINE
################################################################################

def separate_music(input_audio: List[str], output_folder: str, callback: Callable = None, **kwargs) -> List[str]:
    """
    Wrapper for calling the separation model.
    Example usage:
        separate_music(
            ["/path/to/somefile.mp3"],
            "/output/folder",
            cpu=False,
            vocals_only=False,
            separate_drums=True,
            separate_woodwinds=True,
            alt_bass_model=True,
            callback=your_callback_function
        )
    """
    os.makedirs(output_folder, exist_ok=True)

    options = {
        "input_audio": input_audio,
        "output_folder": output_folder,
        "cpu": kwargs.get("cpu", False),
        "vocals_only": kwargs.get("vocals_only", False),
        "use_VOCFT": kwargs.get("use_VOCFT", False),
        "separate_drums": kwargs.get("separate_drums", False),
        "separate_woodwinds": kwargs.get("separate_woodwinds", False),
        "alt_bass_model": kwargs.get("alt_bass_model", False),
        "weight_InstVoc": kwargs.get("weight_InstVoc", 8.0),
        "weight_VOCFT": kwargs.get("weight_VOCFT", 1.0),
        "weight_VitLarge": kwargs.get("weight_VitLarge", 5.0)
    }
    return predict_with_model(options, callback)


def predict_with_model(options: Dict, callback: Callable = None) -> List[str]:
    """
    Loads all input files, runs the ensemble separation and then (if requested)
    additional multi-stem, alt bass, advanced drum, and woodwinds separation on
    all files. Finally, writes all final stems to disk.
    """
    input_files = options["input_audio"]
    files_data = []
    # Load all input files first.
    for ip in input_files:
        if not os.path.isfile(ip):
            continue
        wav_path = ensure_wav(ip)
        loaded, sr = librosa.load(wav_path, sr=44100, mono=False)
        base_name = os.path.splitext(os.path.basename(ip))[0]
        files_data.append({"base_name": base_name, "mix_np": loaded, "sr": sr})
    if not files_data:
        return []

    model = EnsembleDemucsMDXMusicSeparationModel(options, callback)
    # Calculate total progress steps.
    ensemble_steps = 5
    multi_stem_steps = 1 if not model.vocals_only else 0
    alt_bass_steps = 1 if (model.alt_bass_model and not model.vocals_only) else 0
    drum_steps = 1 if (model.separate_drums and not model.vocals_only) else 0
    ww_steps = 1 if (model.separate_woodwinds and not model.vocals_only) else 0
    saving_steps = 1
    steps_per_file = ensemble_steps + multi_stem_steps + alt_bass_steps + drum_steps + ww_steps + saving_steps
    model.total_steps = steps_per_file * len(files_data)
    # Stage 1: Ensemble separation across all files.
    results = model._ensemble_separate_all(files_data)
    if not model.vocals_only:
        # Stage 2: Multi-stem separation on the instrumental track.
        model._multistem_separation_all(results)
        # Stage 3: Alt Bass separation.
        if model.alt_bass_model:
            model._alt_bass_separation_all(results)
        # Stage 4: Advanced drum separation.
        if model.separate_drums:
            model._advanced_drum_separation_all(results)
        # Stage 5: Woodwinds separation.
        if model.separate_woodwinds:
            model._woodwinds_separation_all(results)

    # Stage 6: Save all stems.
    output_files = model._save_all_stems(results)
    return output_files
