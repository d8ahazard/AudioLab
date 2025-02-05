# coding: utf-8
import logging
import os
import subprocess
import uuid
import warnings
from typing import List, Dict

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
    A multi-model ensemble-based separation approach. Leaves all intermediate files.
    If separate_drums=True, also does advanced multi-piece drum separation.
    If separate_woodwinds=True, we try to separate woodwinds from the 'other' track.
    If alt_bass_model=True, we override the normal demucs bass with an alternate model.
    """

    def __init__(self, options: Dict):
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

        # Overlaps for advanced separation (not specifically used below, but kept for reference)
        self.overlap_large = options.get("overlap_large", 0.6)
        self.overlap_small = options.get("overlap_small", 0.5)

        # Progress
        self.global_step = 0
        self.total_steps = 0
        self.callback = options.get("callback", None)

    def _advance_progress(self, desc: str):
        """
        Increments self.global_step by 1 and calls the callback if defined.
        callback signature: callback(current_step, description, total_steps).
        """
        self.global_step += 1
        if self.callback is not None:
            self.callback(self.global_step/self.total_steps, desc, self.total_steps)

    def separate_music_file(self, base_name, mixed_sound_array, sample_rate, callback=None):
        """
        Performs full separation. If vocals_only=True => get (Vocals, Instrumental).
        Otherwise => multi-stem (drums, bass, guitar, piano, other), plus optional expansions.
        """
        stems, sr_dict = self._ensemble_separate(base_name, mixed_sound_array, sample_rate)
        output_files = self._save_stems(base_name, stems, sr_dict, self.options["output_folder"])
        return output_files

    def _ensemble_separate(self, base_name, mix_np, sr):
        """
        1) Ensemble-blend multiple vocal/instrumental models => best Vocals + Instrumental
        2) If user wants >2 stems, run 6-stem Demucs => (drums,bass,guitar,piano,other)
           with optional alt_bass_model, separate_drums, separate_woodwinds.
        3) Return stems dict + sr_dict.
        """
        # Make sure shape => (channels, samples)
        if mix_np.ndim == 1:
            mix_np = np.stack([mix_np, mix_np], axis=0)
        elif (mix_np.shape[0] != 2) and (mix_np.shape[1] == 2):
            mix_np = mix_np.T

        # -------------------------------------------------
        # Step 1: Ensemble for Vocals + Instrumental
        # -------------------------------------------------
        models_with_weights = [
            ("model_bs_roformer_ep_368_sdr_12.9628.ckpt", 8.4, 16.0),
            ("MDX23C-8KFFT-InstVoc_HQ.ckpt", 7.2, 14.9),
            ("UVR-MDX-NET-Voc_FT.onnx", 6.9, 14.9),
            ("Kim_Vocal_2.onnx", 6.9, 14.9),
            ("Kim_Vocal_1.onnx", 6.8, 14.9),
        ]

        all_vocals, all_instrumentals = [], []
        vocal_weights, instrumental_weights = [], []
        for model_name, v_wt, i_wt in models_with_weights:
            desc = f"[Ensemble] {base_name} => {model_name}"
            separated = self._separate_as_arrays(base_name, mix_np, sr, model_name, desc)
            vstem = separated.get("vocals", np.zeros_like(mix_np))
            istem = separated.get("instrumental", np.zeros_like(mix_np))
            all_vocals.append(vstem * v_wt)
            all_instrumentals.append(istem * i_wt)
            vocal_weights.append(v_wt)
            instrumental_weights.append(i_wt)
            self._advance_progress(f"Ensemble model {model_name} done")

        def _blend_tracks(tracks, weights):
            max_length = max(t.shape[-1] for t in tracks)
            combined = np.zeros((tracks[0].shape[0], max_length), dtype=np.float32)
            total_weight = sum(weights)
            for idx, t in enumerate(tracks):
                combined[:, :t.shape[-1]] += t / total_weight
            # normalize
            peak = np.max(np.abs(combined))
            if peak > 0:
                combined /= peak
            return combined

        vocals = _blend_tracks(all_vocals, vocal_weights)
        instruments = _blend_tracks(all_instrumentals, instrumental_weights)
        stems = {"vocals": vocals, "instrumental": instruments}
        sr_dict = {"vocals": sr, "instrumental": sr}

        # If we only want 2-stem, done
        if self.vocals_only:
            return stems, sr_dict

        # -------------------------------------------------
        # Step 2: Multi-stem separation on "instrumental"
        # -------------------------------------------------
        self._advance_progress("Performing 6-stem separation")
        tmp_instru_wav = write_temp_wav(instruments, sr, self.options["output_folder"])
        self.separator.load_model("htdemucs_6s.yaml")
        demucs_partial = self.separator.separate(tmp_instru_wav)
        demucs_files = [os.path.join(self.separator.output_dir, f) for f in demucs_partial]

        stems["drums"] = np.zeros_like(instruments)
        stems["bass"] = np.zeros_like(instruments)
        stems["guitar"] = np.zeros_like(instruments)
        stems["piano"] = np.zeros_like(instruments)
        stems["other"] = np.zeros_like(instruments)
        sr_dict["drums"] = sr
        sr_dict["bass"] = sr
        sr_dict["guitar"] = sr
        sr_dict["piano"] = sr
        sr_dict["other"] = sr

        for f in demucs_files:
            lowf = os.path.basename(f).lower()
            arr, _ = librosa.load(f, sr=sr, mono=False)
            if arr.ndim == 1:
                arr = np.stack([arr, arr], axis=0)
            if "(drums)" in lowf:
                stems["drums"] = arr
            elif "(bass)" in lowf:
                stems["bass"] = arr
            elif "(guitar)" in lowf:
                stems["guitar"] = arr
            elif "(piano)" in lowf:
                stems["piano"] = arr
            elif "(other)" in lowf:
                stems["other"] = arr

        # -------------------------------------------------
        # Step 2.1: Alt Bass Model
        # -------------------------------------------------
        if self.alt_bass_model:
            self._advance_progress("Alt Bass model separation")
            self.separator.load_model("kuielab_a_bass.onnx")
            alt_bass_out = self.separator.separate(tmp_instru_wav)
            alt_bass_files = [os.path.join(self.separator.output_dir, b) for b in alt_bass_out]
            for bfile in alt_bass_files:
                if not os.path.exists(bfile):
                    continue
                blow = os.path.basename(bfile).lower()
                arrb, _ = librosa.load(bfile, sr=sr, mono=False)
                if arrb.ndim == 1:
                    arrb = np.stack([arrb, arrb], axis=0)
                if "(bass)" in blow:
                    stems["bass"] = arrb  # override demucs bass

        # -------------------------------------------------
        # Step 2.2: Advanced drum separation
        # -------------------------------------------------
        if self.separate_drums:
            self._advance_progress("Advanced drum separation")
            tmp_drums_wav = write_temp_wav(stems["drums"], sr, self.options["output_folder"])
            self.separator.load_model("MDX23C-DrumSep-aufr33-jarredou.ckpt")
            drum_parts = self.separator.separate(tmp_drums_wav)
            drum_part_files = [os.path.join(self.separator.output_dir, d) for d in drum_parts]

            drums_other = np.copy(stems["drums"])
            for dpf in drum_part_files:
                dplow = os.path.basename(dpf).lower()
                arrp, _ = librosa.load(dpf, sr=sr, mono=False)
                if arrp.ndim == 1:
                    arrp = np.stack([arrp, arrp], axis=0)
                # Subtract from the "drums_other"
                if arrp.shape[-1] <= drums_other.shape[-1]:
                    drums_other[:, :arrp.shape[-1]] -= arrp

                if "(kick)" in dplow:
                    stems["drums_kick"] = arrp
                    sr_dict["drums_kick"] = sr
                elif "(snare)" in dplow:
                    stems["drums_snare"] = arrp
                    sr_dict["drums_snare"] = sr
                elif "(toms)" in dplow:
                    stems["drums_toms"] = arrp
                    sr_dict["drums_toms"] = sr
                elif "(hh)" in dplow:
                    stems["drums_hh"] = arrp
                    sr_dict["drums_hh"] = sr
                elif "(ride)" in dplow:
                    stems["drums_ride"] = arrp
                    sr_dict["drums_ride"] = sr
                elif "(crash)" in dplow:
                    stems["drums_crash"] = arrp
                    sr_dict["drums_crash"] = sr

            stems["drums_other"] = drums_other
            sr_dict["drums_other"] = sr

        # -------------------------------------------------
        # Step 2.3: Woodwinds separation
        # -------------------------------------------------
        if self.separate_woodwinds:
            self._advance_progress("Woodwinds separation")
            tmp_other_wav = write_temp_wav(stems["other"], sr, self.options["output_folder"])
            self.separator.load_model("17_HP-Wind_Inst-UVR.pth")
            ww_parts = self.separator.separate(tmp_other_wav)
            ww_part_files = [os.path.join(self.separator.output_dir, w) for w in ww_parts]

            new_woodwinds = np.zeros_like(stems["other"])
            for wfile in ww_part_files:
                if not os.path.exists(wfile):
                    continue
                wflow = os.path.basename(wfile).lower()
                arrw, _ = librosa.load(wfile, sr=sr, mono=False)
                if arrw.ndim == 1:
                    arrw = np.stack([arrw, arrw], axis=0)
                if "(woodwinds)" in wflow:
                    new_woodwinds = arrw

            leftover_other = np.copy(stems["other"])
            if new_woodwinds.shape[-1] <= leftover_other.shape[-1]:
                leftover_other[:, :new_woodwinds.shape[-1]] -= new_woodwinds
            stems["woodwinds"] = new_woodwinds
            sr_dict["woodwinds"] = sr
            stems["other"] = leftover_other
            sr_dict["other"] = sr

        return stems, sr_dict

    def _save_stems(self, base_name, stems, sr_dict, output_folder):
        """ Saves final stems and then deletes any temp_ files. """
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

        for stem_name, stem_data in stems.items():
            output_name = f"{base_name}_{stem_names.get(stem_name, stem_name)}.wav"
            output_path = os.path.join(output_folder, output_name)
            sf.write(output_path, stem_data.T, sr_dict[stem_name], subtype="FLOAT")
            output_files.append(output_path)

        # Remove any tmp_ files
        for temp_file in os.listdir(output_folder):
            if temp_file.startswith("tmp_"):
                os.remove(os.path.join(output_folder, temp_file))

        return output_files

    def _separate_as_arrays(self, base_name, mix_np, sr, model_name, desc=None):
        """ Runs separation for a single model and returns results as arrays. """
        tmp_wav = write_temp_wav(mix_np, sr, self.options["output_folder"])
        self.separator.load_model(model_name)

        # Optional description you can log or pass along
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
            elif "(drums)" in flow:
                stems["drums"] = arr
            elif "(bass)" in flow:
                stems["bass"] = arr
            elif "(woodwinds)" in flow:
                stems["woodwinds"] = arr
            elif "(piano)" in flow:
                stems["piano"] = arr
            elif "(guitar)" in flow:
                stems["guitar"] = arr
            elif "(other)" in flow:
                stems["other"] = arr
        return stems


################################################################################
#                    TOP-LEVEL PREDICTION + OUTPUT ROUTINE
################################################################################

def separate_music(input_audio: List[str], output_folder: str, **kwargs) -> List[str]:
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
        "weight_VitLarge": kwargs.get("weight_VitLarge", 5.0),
        "callback": kwargs.get("callback", None),
    }
    return predict_with_model(options)


def predict_with_model(options: Dict) -> List[str]:
    """ Runs the ensemble separation on each file and writes final stems to disk. """
    model = EnsembleDemucsMDXMusicSeparationModel(options)
    input_files = options["input_audio"]

    # ----------------------------------------------------
    # Calculate the total steps for each file
    # ----------------------------------------------------
    # 1) Ensemble vocals: len(models_with_weights) => 5
    ensemble_steps = 5
    # 2) 6-stem separation => 1 if not vocals_only
    multi_stem_steps = 1 if not model.vocals_only else 0
    # 3) alt bass => 1 if alt_bass_model + not vocals_only
    alt_bass_steps = 1 if (model.alt_bass_model and not model.vocals_only) else 0
    # 4) separate drums => 1 if separate_drums + not vocals_only
    drum_steps = 1 if (model.separate_drums and not model.vocals_only) else 0
    # 5) separate woodwinds => 1 if separate_woodwinds + not vocals_only
    ww_steps = 1 if (model.separate_woodwinds and not model.vocals_only) else 0
    # 6) final saving => 1
    saving_steps = 1

    steps_per_file = (
        ensemble_steps + multi_stem_steps
        + alt_bass_steps + drum_steps
        + ww_steps + saving_steps
    )
    total_steps = steps_per_file * len(input_files)
    model.total_steps = total_steps

    # ----------------------------------------------------
    # Process each file
    # ----------------------------------------------------
    output_files = []
    for ip in input_files:
        if not os.path.isfile(ip):
            # Advance some step so callback doesn't freeze?
            if model.callback:
                model.callback(model.global_step, f"Missing file: {ip}", model.total_steps)
            continue

        loaded, sr = librosa.load(ensure_wav(ip), sr=44100, mono=False)
        model.separator.output_dir = options["output_folder"]
        base_name = os.path.splitext(os.path.basename(ip))[0]
        stems = model.separate_music_file(base_name, loaded, sr)
        output_files.extend(stems)

    return output_files
