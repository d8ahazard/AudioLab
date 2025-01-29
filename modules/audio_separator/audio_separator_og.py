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
from audio_separator.separator import Separator

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
    If separate_drums=True, also does advanced multi-piece drum separation for (kick/snare/toms/hh/ride/crash).

    The big fix: We always store an 'instrumental' track (the entire minus vocals),
    plus additional stems from 6-stem separation if the user wants them.
    """

    def __init__(self, options: Dict):
        self.options = options
        self.device = torch.device("cuda:0") if torch.cuda.is_available() and not options.get("cpu",
                                                                                              False) else torch.device(
            "cpu")

        self.separator = Separator(
            log_level=logging.ERROR,
            model_file_dir=os.path.join(app_path, "models", "audio_separator"),
            output_dir=options["output_folder"]
        )

        # Load necessary models
        self.model_list = [
            "htdemucs_ft.yaml", "htdemucs.yaml", "hdemucs_mmi.yaml", "htdemucs_6s.yaml",
            "MDX23C-8KFFT-InstVoc_HQ.ckpt", "model_bs_roformer_ep_368_sdr_12.9628.ckpt",
            "UVR-MDX-NET-Voc_FT.onnx", "Kim_Vocal_1.onnx", "Kim_Vocal_2.onnx",
            "MDX23C-DrumSep-aufr33-jarredou.ckpt", "17_HP-Wind_Inst-UVR.pth"
        ]

        for model in self.model_list:
            logger.info(f"Downloading model: {model}")
            self.separator.download_model_files(model)

        self.vocals_only = bool(options.get("vocals_only", False))
        self.use_vocft = bool(options.get("use_VOCFT", False))
        self.separate_drums = bool(options.get("separate_drums", False))
        self.separate_woodwinds = bool(options.get("separate_woodwinds", False))
        self.alt_bass_model = bool(options.get("alt_bass_model", False))

        # Weighted blending for vocals
        self.weight_inst = float(options.get("weight_InstVoc", 8.0))
        self.weight_vocft = float(options.get("weight_VOCFT", 1.0))
        self.weight_rof = float(options.get("weight_VitLarge", 5.0))

        self.overlap_large = options.get("overlap_large", 0.6)
        self.overlap_small = options.get("overlap_small", 0.5)
        self.global_step = 0
        self.total_steps = 0

    def separate_music_file(self, base_name, mixed_sound_array, sample_rate, callback=None):
        """ Performs full separation, saves output, and cleans intermediate files. """
        stems, sr_dict = self._ensemble_separate(base_name, mixed_sound_array, sample_rate, callback)
        output_files = self._save_stems(base_name, stems, sr_dict, self.options["output_folder"])
        return output_files

    def _ensemble_separate(self, base_name, mix_np, sr, callback=None):
        """ Advanced ensemble separation process. """
        mix_np = np.stack([mix_np, mix_np], axis=0) if mix_np.ndim == 1 else mix_np.T if mix_np.shape[
                                                                                             1] == 2 else mix_np

        """ Uses an ensemble of models to create the best possible vocal track. """
        models_with_weights = [
            ("model_bs_roformer_ep_368_sdr_12.9628.ckpt", 8.4, 16.0),
            ("MDX23C-8KFFT-InstVoc_HQ.ckpt", 7.2, 14.9),
            ("UVR-MDX-NET-Voc_FT.onnx", 6.9, 14.9),
            ("Kim_Vocal_2.onnx", 6.9, 14.9),
            ("Kim_Vocal_1.onnx", 6.8, 14.9)
        ]

        all_vocals, all_instrumentals = [], []
        vocal_weights, instrumental_weights = [], []

        for model_name, vocal_weight, instrumental_weight in models_with_weights:
            separated = self._separate_as_arrays(base_name, mix_np, sr, model_name, callback)
            vocals, instruments = separated.get("vocals", np.zeros_like(mix_np)), separated.get("instrumental",
                                                                                                np.zeros_like(mix_np))

            all_vocals.append(vocals * vocal_weight)
            all_instrumentals.append(instruments * instrumental_weight)
            vocal_weights.append(vocal_weight)
            instrumental_weights.append(instrumental_weight)

        vocals = self._blend_tracks(all_vocals, vocal_weights)
        instruments = self._blend_tracks(all_instrumentals, instrumental_weights)

        stems = {"vocals": vocals, "instrumental": instruments}
        sr_dict = {"vocals": sr, "instrumental": sr}

        return stems, sr_dict

    def _save_stems(self, base_name, stems, sr_dict, output_folder):
        """ Saves final stems and deletes unnecessary intermediate files. """
        output_files = []
        stem_names = {
            "vocals": "(Vocals)",
            "instrumental": "(Instrumental)",
            "drums": "(Drums)",
            "bass": "(Bass)",
            "guitar": "(Guitar)",
            "piano": "(Piano)",
            "woodwinds": "(Woodwinds)",
            "other": "(Other)"
        }

        for stem_name, stem_data in stems.items():
            output_name = f"{base_name}_{stem_names.get(stem_name, stem_name)}.wav"
            output_path = os.path.join(output_folder, output_name)
            sf.write(output_path, stem_data.T, sr_dict[stem_name], subtype="FLOAT")
            output_files.append(output_path)

        # Delete temporary/intermediate files
        for temp_file in os.listdir(output_folder):
            if temp_file.startswith("tmp_"):
                os.remove(os.path.join(output_folder, temp_file))

        return output_files

    def _separate_as_arrays(self, base_name, mix_np, sr, model_name, callback=None):
        """ Runs separation for a single model and returns results as arrays. """
        tmp_wav = write_temp_wav(mix_np, sr, self.options["output_folder"])
        self.separator.load_model(model_name)
        if callback:
            callback(self.global_step / self.total_steps, f"Separating {base_name} with {model_name}...")
        output_files_partial = self.separator.separate(tmp_wav)
        output_files = [os.path.join(self.separator.output_dir, f) for f in output_files_partial]

        stems = {}
        for file in output_files:
            arr, _ = librosa.load(file, sr=sr, mono=False)
            arr = np.stack([arr, arr], axis=0) if arr.ndim == 1 else arr
            if "(vocals)" in file.lower():
                stems["vocals"] = arr
            elif "(instrumental)" in file.lower():
                stems["instrumental"] = arr
            elif "(drums)" in file.lower():
                stems["drums"] = arr
            elif "(woodwinds)" in file.lower():
                stems["woodwinds"] = arr
            elif "(piano)" in file.lower():
                stems["piano"] = arr
        return stems

    def _blend_tracks(self, tracks, weights):
        max_length = max(track.shape[-1] for track in tracks)
        combined = np.zeros((tracks[0].shape[0], max_length), dtype=np.float32)
        for i, track in enumerate(tracks):
            combined[:, :track.shape[-1]] += track / sum(weights)
        return combined / np.max(np.abs(combined))


################################################################################
#                    TOP-LEVEL PREDICTION + OUTPUT ROUTINE
################################################################################

def separate_music(input_audio: List[str], output_folder: str, **kwargs) -> List[str]:
    """ Wrapper for calling the separation model. """
    os.makedirs(output_folder, exist_ok=True)

    options = {
        "input_audio": input_audio,
        "output_folder": output_folder,
        "cpu": kwargs.get("cpu", False),
        "callback": kwargs.get("callback", None)
    }

    return predict_with_model(options)


def predict_with_model(options: Dict) -> List[str]:
    """ Runs the ensemble separation on each file and writes final stems to disk. """
    model = EnsembleDemucsMDXMusicSeparationModel(options)
    # TODO: Multiply this by the total number of separations
    model.total_steps = len(options["input_audio"])
    output_files = []

    for ip in options["input_audio"]:
        loaded, sr = librosa.load(ensure_wav(ip), sr=44100, mono=False)
        model.separator.output_dir = options["output_folder"]
        base_name = os.path.splitext(os.path.basename(ip))[0]
        callback = options.get("callback", None)
        stems = model.separate_music_file(base_name, loaded, sr, callback)
        model.global_step += 1
        output_files.extend(stems)

    return output_files
