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
from handlers.reverb import extract_reverb

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
    A multi-model ensemble-based separation approach with additional
    background vocal splitting and transformation chain for audio processing.
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
        # Download all required models
        self.model_list = [
            "htdemucs_ft.yaml", "htdemucs.yaml", "hdemucs_mmi.yaml", "htdemucs_6s.yaml",
            "MDX23C-8KFFT-InstVoc_HQ.ckpt", "model_bs_roformer_ep_368_sdr_12.9628.ckpt",
            "UVR-MDX-NET-Voc_FT.onnx", "Kim_Vocal_1.onnx", "Kim_Vocal_2.onnx",
            "MDX23C-DrumSep-aufr33-jarredou.ckpt",
            "17_HP-Wind_Inst-UVR.pth",
            "kuielab_a_bass.onnx"
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

        # Overlap values for advanced separation
        self.overlap_large = options.get("overlap_large", 0.6)
        self.overlap_small = options.get("overlap_small", 0.5)

        # Transformation and BG vocal options from wrapper
        self.reverb_removal = options.get("reverb_removal", "Nothing")
        self.echo_removal = options.get("echo_removal", "Nothing")
        self.delay_removal = options.get("delay_removal", "Nothing")
        self.crowd_removal = options.get("crowd_removal", "Nothing")
        self.noise_removal = options.get("noise_removal", "Nothing")
        self.delay_removal_model = options.get("delay_removal_model", "UVR-DeEcho-DeReverb.pth")
        self.noise_removal_model = options.get("noise_removal_model", "UVR-DeNoise.pth")
        self.crowd_removal_model = options.get("crowd_removal_model", "UVR-MDX-NET_Crowd_HQ_1.onnx")
        self.separate_bg_vocals = options.get("separate_bg_vocals", True)
        self.store_reverb_ir = options.get("store_reverb_ir", False)

        # Progress tracking
        self.global_step = 0
        self.total_steps = 0
        self.callback = options.get("callback", None)

    def _advance_progress(self, desc: str):
        """ Increments progress and calls callback if defined. """
        self.global_step += 1
        if self.callback is not None:
            self.callback(self.global_step / self.total_steps, desc, self.total_steps)
        logger.info(f"[{self.global_step}/{self.total_steps}] {desc}")

    def _blend_tracks(self, tracks, weights):
        """ Blends a list of tracks with given weights. """
        max_length = max(t.shape[-1] for t in tracks)
        combined = np.zeros((tracks[0].shape[0], max_length), dtype=np.float32)
        total_weight = sum(weights)
        for t in tracks:
            combined[:, :t.shape[-1]] += t
        combined = combined / total_weight
        peak = np.max(np.abs(combined))
        if peak > 0:
            combined /= peak
        return combined

    def _separate_as_arrays_current(self, mix_np, sr, desc=None):
        """ Runs separation for the current model and returns results as arrays. """
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
        return stems

    def _ensemble_separate_all(self, files_data: List[Dict]) -> Dict[str, Dict]:
        """ Performs ensemble separation on all files. """
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
        for base_name, res in results.items():
            res["vocals"] = self._blend_tracks(res["vocals_list"], res["v_weights"])
            res["instrumental"] = self._blend_tracks(res["instrumental_list"], res["i_weights"])
        return results

    def _multistem_separation_all(self, results: Dict[str, Dict]):
        """ Runs 6-stem separation on the instrumental track for all files. """
        self.separator.load_model("htdemucs_6s.yaml")
        for base_name, res in results.items():
            sr = res["sr"]
            inst = res["instrumental"]
            tmp_instru_wav = write_temp_wav(inst, sr, self.options["output_folder"])
            demucs_partial = self.separator.separate(tmp_instru_wav)
            demucs_files = [os.path.join(self.separator.output_dir, f) for f in demucs_partial]
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
        """ Applies an alternate bass separation model on all files. """
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
                    res["bass"] = arrb
            self._advance_progress(f"Alt Bass separation done for {base_name}")

    def _advanced_drum_separation_all(self, results: Dict[str, Dict]):
        """ Runs advanced drum separation on the drums stem for all files. """
        self.separator.load_model("MDX23C-DrumSep-aufr33-jarredou.ckpt")
        for base_name, res in results.items():
            sr = res["sr"]
            drums = res.get("drums", np.zeros_like(res["instrumental"]))
            tmp_drums_wav = write_temp_wav(drums, sr, self.options["output_folder"])
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
        """ Separates woodwinds from the 'other' stem on all files. """
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
        """ Saves all final stems to disk and cleans up temporary files. """
        self._advance_progress("Saving final stems")
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
        for temp_file in os.listdir(output_folder):
            if temp_file.startswith("tmp_"):
                os.remove(os.path.join(output_folder, temp_file))
        return output_files

    @staticmethod
    def _should_apply_transform(stem_name: str, setting: str) -> bool:
        """ Determines if a transform should be applied based on stem name and setting. """
        if setting == "Nothing":
            return False
        if setting == "All":
            return True
        if setting == "All Vocals":
            return "Vocals)" in stem_name
        if setting == "Main Vocals":
            return "Vocals)" in stem_name and "(BG_Vocals)" not in stem_name
        return False

    @staticmethod
    def _rename_bgvocal(filepath: str) -> str:
        """ Renames file for BG vocal separation. """
        dirname = os.path.dirname(filepath)
        oldbase = os.path.basename(filepath)
        newbase = oldbase.replace("(Vocals)_(Instrumental)", "(Vocals)(BG_Vocals)")
        newbase = newbase.replace("(Vocals)_(Vocals)", "(Vocals)(Main Vocals)")
        newbase = newbase.replace("(Vocals)(Instrumental)", "(Vocals)(BG_Vocals)")
        newbase = newbase.replace("(Vocals)(Vocals)", "(Vocals)(Main Vocals)")
        newbase = newbase.replace("_mel_band_roformer_karaoke_aufr33_viperx_sdr_10", "")
        newpath = os.path.join(dirname, newbase)
        if os.path.exists(filepath):
            if os.path.exists(newpath):
                os.remove(newpath)
            os.rename(filepath, newpath)
        return newpath

    @staticmethod
    def _rename_file(base_in: str, filepath: str) -> str:
        """ Rebuilds the filename to remove model references. """
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

    def _apply_bg_vocal_splitting(self, vocals_array, sr, base_name):
        """ Applies background vocal splitting to the vocals array. """
        tmp_file = write_temp_wav(vocals_array, sr, self.options["output_folder"])
        self.separator.load_model("mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt")
        out_files = self.separator.separate(tmp_file)
        out_files_full = [os.path.join(self.options["output_folder"], f) for f in out_files]
        renamed_files = [self._rename_bgvocal(f) for f in out_files_full]
        arrays = []
        for f in renamed_files:
            arr, _ = librosa.load(f, sr=sr, mono=False)
            arrays.append(arr)
        if len(renamed_files) == 2:
            if "(BG_Vocals)" in renamed_files[0]:
                bg = arrays[0]
                main = arrays[1]
            else:
                bg = arrays[1]
                main = arrays[0]
            return main, bg
        return vocals_array, None

    def _apply_transform_chain(self, stem_array, sr, base_name, stem_label, project_dir) -> np.ndarray:
        """ Applies a series of transformations (reverb, echo, etc.) to a stem array. """
        transformations = [
            ("deverb_bs_roformer_8_384dim_10depth.ckpt", "No Reverb", self.reverb_removal),
            (self.delay_removal_model, "No Echo", self.echo_removal),
            (self.delay_removal_model, "No Delay", self.delay_removal),
            (self.crowd_removal_model, "No Crowd", self.crowd_removal),
            (self.noise_removal_model, "No Noise", self.noise_removal),
        ]
        current_array = stem_array
        for model_file, out_label, transform_flag in transformations:
            simulated_name = f"({stem_label})"
            if self._should_apply_transform(simulated_name, transform_flag):
                self.separator.load_model(model_file)
                tmp_file = write_temp_wav(current_array, sr, self.options["output_folder"])
                out_files = self.separator.separate(tmp_file)
                out_files_full = [os.path.join(self.options["output_folder"], f) for f in out_files]
                chosen_file = None
                if len(out_files_full) == 2:
                    if out_label.replace(" ", "").lower() in out_files_full[0].replace(" ", "").lower():
                        chosen_file = out_files_full[0]
                        alt_file = out_files_full[1]
                    else:
                        chosen_file = out_files_full[1]
                        alt_file = out_files_full[0]
                    chosen_file = self._rename_file(base_name, chosen_file)
                    if out_label == "No Reverb" and stem_label.lower() == "vocals" and self.store_reverb_ir and alt_file:
                        try:
                            out_ir = os.path.join(project_dir, "impulse_response.ir")
                            logger.info(f"Extracting reverb IR from {os.path.basename(alt_file)}")
                            extract_reverb(chosen_file, alt_file, out_ir)
                        except Exception as e:
                            print("Error extracting IR:", e)
                else:
                    for pf in out_files_full:
                        if out_label.replace(" ", "").lower() in pf.replace(" ", "").lower():
                            chosen_file = self._rename_file(base_name, pf)
                            break
                if chosen_file:
                    current_array, _ = librosa.load(chosen_file, sr=sr, mono=False)
                self._advance_progress(f"Applied transform {out_label} on {stem_label} for {base_name}")
        return current_array


################################################################################
#                    TOP-LEVEL PREDICTION + OUTPUT ROUTINE
################################################################################

def predict_with_model(options: Dict, callback: Callable = None) -> List[str]:
    """ Loads input files, runs ensemble and additional processing, then saves stems. """
    input_files = options["input_audio"]
    files_data = []
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
    ensemble_steps = 5
    multi_stem_steps = 1 if not model.vocals_only else 0
    alt_bass_steps = 1 if (model.alt_bass_model and not model.vocals_only) else 0
    drum_steps = 1 if (model.separate_drums and not model.vocals_only) else 0
    ww_steps = 1 if (model.separate_woodwinds and not model.vocals_only) else 0
    saving_steps = 1
    steps_per_file = ensemble_steps + multi_stem_steps + alt_bass_steps + drum_steps + ww_steps + saving_steps
    model.total_steps = steps_per_file * len(files_data)
    results = model._ensemble_separate_all(files_data)
    if not model.vocals_only:
        # Apply background vocal splitting if enabled
        if model.separate_bg_vocals:
            for base_name, res in results.items():
                if "vocals" in res:
                    main_vocals, bg_vocals = model._apply_bg_vocal_splitting(res["vocals"], res["sr"], base_name)
                    res["vocals"] = main_vocals
                    res["bg_vocals"] = bg_vocals
        # Apply transformation chain to vocals and instrumental stems if any transform is set
        transform_options = [model.reverb_removal, model.echo_removal, model.delay_removal, model.crowd_removal,
                             model.noise_removal]
        if any(opt != "Nothing" for opt in transform_options):
            for base_name, res in results.items():
                for stem_label in ["vocals", "instrumental"]:
                    if stem_label in res and res[stem_label] is not None:
                        res[stem_label] = model._apply_transform_chain(
                            res[stem_label], res["sr"], base_name, stem_label, options["output_folder"]
                        )
        model._multistem_separation_all(results)
        if model.alt_bass_model:
            model._alt_bass_separation_all(results)
        if model.separate_drums:
            model._advanced_drum_separation_all(results)
        if model.separate_woodwinds:
            model._woodwinds_separation_all(results)
    output_files = model._save_all_stems(results)
    return output_files


def separate_music(input_audio: List[str], output_folder: str, callback: Callable = None, **kwargs) -> List[str]:
    """
    Wrapper for calling the separation model.
    Example:
        separate_music(
            ["/path/to/file.mp3"],
            "/output/folder",
            cpu=False,
            vocals_only=False,
            separate_drums=True,
            separate_woodwinds=True,
            alt_bass_model=True,
            callback=your_callback_function,
            reverb_removal="Main Vocals",
            echo_removal="Nothing",
            delay_removal="Nothing",
            crowd_removal="Nothing",
            noise_removal="Nothing"
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
        "weight_VitLarge": kwargs.get("weight_VitLarge", 5.0),
        "reverb_removal": kwargs.get("reverb_removal", "Nothing"),
        "echo_removal": kwargs.get("echo_removal", "Nothing"),
        "delay_removal": kwargs.get("delay_removal", "Nothing"),
        "crowd_removal": kwargs.get("crowd_removal", "Nothing"),
        "noise_removal": kwargs.get("noise_removal", "Nothing"),
        "delay_removal_model": kwargs.get("delay_removal_model", "UVR-DeEcho-DeReverb.pth"),
        "noise_removal_model": kwargs.get("noise_removal_model", "UVR-DeNoise.pth"),
        "crowd_removal_model": kwargs.get("crowd_removal_model", "UVR-MDX-NET_Crowd_HQ_1.onnx"),
        "separate_bg_vocals": kwargs.get("separate_bg_vocals", True),
        "store_reverb_ir": kwargs.get("store_reverb_ir", False),
        "callback": callback
    }
    return predict_with_model(options, callback)
