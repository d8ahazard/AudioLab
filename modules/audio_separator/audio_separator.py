# coding: utf-8
import errno
import hashlib
import os
import shutil
import subprocess
import uuid
import warnings
from time import time
from typing import List, Optional, Dict
from urllib.request import urlopen, Request

import librosa
import numpy as np
import soundfile as sf
import torch
from audio_separator.separator import Separator
from scipy import signal
from scipy.signal import resample_poly
from tqdm import tqdm

from handlers.config import app_path

warnings.filterwarnings("ignore")


################################################################################
#                        HELPER UTILITY FUNCTIONS
################################################################################

def download_url_to_file(url: str, dst: str, hash_prefix: Optional[str] = None, progress: bool = True) -> None:
    file_size = None
    req = Request(url, headers={"User-Agent": "torch.hub"})
    u = urlopen(req)
    meta = u.info()
    if hasattr(meta, "getheaders"):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length and len(content_length) > 0:
        file_size = int(content_length[0])

    for _ in range(1000):
        tmp_dst = dst + "." + uuid.uuid4().hex + ".partial"
        try:
            f = open(tmp_dst, "w+b")
        except (FileExistsError, FileNotFoundError):
            continue
        break
    else:
        raise FileExistsError(errno.EEXIST, "No usable temporary file name found")

    sha256 = hashlib.sha256() if hash_prefix is not None else None
    try:
        with tqdm(total=file_size, disable=not progress, unit="B", unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = u.read(2 ** 20)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                if sha256 is not None:
                    sha256.update(buffer)
                pbar.update(len(buffer))
        f.close()
        if sha256 is not None:
            digest = sha256.hexdigest()
            if digest[: len(hash_prefix)] != hash_prefix:
                raise RuntimeError(f'invalid hash value (expected "{hash_prefix}", got "{digest}")')
        shutil.move(tmp_dst, dst)
    finally:
        f.close()
        if os.path.exists(tmp_dst):
            os.remove(tmp_dst)


def md5(fname):
    import hashlib
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def match_array_shapes(array_1: np.ndarray, array_2: np.ndarray):
    """
    If one array is shorter, zero-pad or truncate so that both match in length.
    """
    if array_1.shape[1] > array_2.shape[1]:
        array_1 = array_1[:, : array_2.shape[1]]
    elif array_1.shape[1] < array_2.shape[1]:
        padding = array_2.shape[1] - array_1.shape[1]
        array_1 = np.pad(array_1, ((0, 0), (0, padding)), "constant", constant_values=0)
    return array_1


def lr_filter(audio, cutoff, filter_type, order=6, sr=44100):
    """
    Linkwitz-Riley style filter for multi-band blending.
    Expects 'audio' of shape (channels, samples).
    If too short for filtfilt, we skip filtering.
    """
    audio_t = audio.T  # shape => (samples, channels)
    length = audio_t.shape[0]
    min_length = 20
    if length < min_length:
        return audio

    nyquist = 0.5 * sr
    norm_cut = cutoff / nyquist
    b, a = signal.butter(order // 2, norm_cut, btype=filter_type, analog=False, output="ba")
    sos = signal.tf2sos(b, a)
    filtered_t = signal.sosfiltfilt(sos, audio_t, axis=0)
    return filtered_t.T


def change_sr(data, up, down):
    data = data.T
    new_data = resample_poly(data, up, down)
    return new_data.T


def lp_filter(cutoff, data, sample_rate):
    b = signal.firwin(1001, cutoff, fs=sample_rate)
    filtered_data = signal.filtfilt(b, [1.0], data)
    return filtered_data


def ensure_wav(input_path: str, sr: int = 44100) -> str:
    """
    If `input_path` is not a .wav, use ffmpeg to convert it.
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
    Writes `mix_np` to a PCM_16 .wav in `out_dir`.
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
    A multi-model ensemble-based separation approach. Leaves all intermediate files.
    If separate_drums=True, also does advanced multi-piece drum separation for (kick/snare/toms/hh/ride/crash).

    The big fix: We always store an 'instrumental' track (the entire minus vocals),
    plus additional stems from 6-stem separation if the user wants them.
    """

    def __init__(self, options: Dict):
        self.options = options

        # CPU or GPU
        if torch.cuda.is_available() and not options.get("cpu", False):
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        # The main separator
        self.separator = Separator(
            model_file_dir=os.path.join(app_path, "models", "audio_separator"),
            output_dir=options["output_folder"]
        )
        # Download any needed models
        needed = [
            "htdemucs_ft.yaml",
            "hdemucs_mmi.yaml",
            "htdemucs.yaml",
            "htdemucs_6s.yaml",
            "MDX23C-8KFFT-InstVoc_HQ.ckpt",
            "model_bs_roformer_ep_368_sdr_12.9628.ckpt",
            "UVR-MDX-NET-Voc_FT.onnx",
            "MDX23C-DrumSep-aufr33-jarredou.ckpt",
        ]
        for m in needed:
            self.separator.download_model_files(m)

        # Some advanced toggles
        self.vocals_only = bool(options.get("vocals_only", False))
        self.use_vocft = bool(options.get("use_VOCFT", False))
        self.separate_drums = bool(options.get("separate_drums", False))
        self.separate_woodwinds = bool(options.get("separate_woodwinds", False))
        self.alt_bass_model = bool(options.get("alt_bass_model", False))

        # Weighted blending for vocals
        self.weight_inst = float(options.get("weight_InstVoc", 8.0))
        self.weight_vocft = float(options.get("weight_VOCFT", 1.0))
        self.weight_rof = float(options.get("weight_VitLarge", 5.0))

    def separate_music_file(self, mixed_sound_array, sample_rate):
        return self._ensemble_separate(mixed_sound_array, sample_rate)

    def _ensemble_separate(self, mix_np, sr):
        """
        1) Build advanced 'vocals' from an ensemble,
        2) 'instrumental' = mix - vocals, store as stems["instrumental"],
        3) If user wants multi-stem, run 6-stem demucs on 'instrumental' => drums, bass, guitar, piano, other,
           plus optional multi-drum separation.
        4) Return all stems in a dictionary, plus sample_rates dict.
        """

        # Ensure shape => (channels, samples)
        if mix_np.ndim == 1:
            mix_np = np.stack([mix_np, mix_np], axis=0)
        elif (mix_np.shape[0] != 2) and (mix_np.shape[1] == 2):
            mix_np = mix_np.T

        # 1) Vocals
        vocals, instruments = self._build_vocals_ensemble(mix_np, sr)

        stems = {
            "vocals": vocals,
            "instrumental": instruments,
        }
        sr_dict = {
            "vocals": sr,
            "instrumental": sr,
        }

        # If user only wants 2-stem, we skip advanced multi-stem
        if self.vocals_only:
            return stems, sr_dict

        # Otherwise, do multi-stem on 'instrumental' track with 6-stem demucs
        instrument_wav = write_temp_wav(instruments, sr, self.options["output_folder"])
        self.separator.load_model("htdemucs_6s.yaml")
        demucs_files = self.separator.separate(instrument_wav)
        # Join partial filenames
        final_demucs = [os.path.join(self.separator.output_dir, df) for df in demucs_files]

        if self.separate_woodwinds:
            self.separator.load_model("17_HP-Wind_Inst-UVR.pth")

        # 6-stem typically yields (Vocals), (Drums), (Bass), (Guitar), (Piano), (Other).
        # We'll parse them:
        bass_file, drums_file, guitar_file, piano_file, other_file = None, None, None, None, None

        for f in final_demucs:
            lowf = os.path.basename(f).lower()
            if "(drums)" in lowf:
                drums_file = f
            elif "(bass)" in lowf:
                bass_file = f
            elif "(guitar)" in lowf:
                guitar_file = f
            elif "(piano)" in lowf:
                piano_file = f
            elif "(other)" in lowf:
                other_file = f
            # sometimes there's a (vocals) from 6-stem, but we ignore it since we have a better ensemble track

        # Bass
        if not self.alt_bass_model:
            if bass_file and os.path.exists(bass_file):
                bass_data, _ = librosa.load(bass_file, sr=sr, mono=False)
                if bass_data.ndim == 1:
                    bass_data = np.stack([bass_data, bass_data], axis=0)
                stems["bass"] = bass_data
                sr_dict["bass"] = sr
            else:
                stems["bass"] = np.zeros_like(instruments)
                sr_dict["bass"] = sr
        else:
            self.separator.load_model("kuielab_a_bass.onnx")
            bass_outs = self.separator.separate(instrument_wav)
            final_bass = [os.path.join(self.separator.output_dir, b) for b in bass_outs]
            for piece_file in final_bass:
                piece_low = os.path.basename(piece_file).lower()
                arr_piece, _ = librosa.load(piece_file, sr=sr, mono=False)
                if arr_piece.ndim == 1:
                    arr_piece = np.stack([arr_piece, arr_piece], axis=0)
                if "(bass)" in piece_low:
                    stems["bass"] = arr_piece
                    sr_dict["bass"] = sr

        # Drums
        if drums_file and os.path.exists(drums_file):
            drums_data, _ = librosa.load(drums_file, sr=sr, mono=False)
            if drums_data.ndim == 1:
                drums_data = np.stack([drums_data, drums_data], axis=0)
            stems["drums"] = drums_data
            sr_dict["drums"] = sr

            # advanced multi-drum if user wants
            if self.separate_drums:
                self.separator.load_model("MDX23C-DrumSep-aufr33-jarredou.ckpt")
                drum_outs = self.separator.separate(drums_file)
                final_drumsep = [os.path.join(self.separator.output_dir, d) for d in drum_outs]
                drum_other = stems.get("drums", np.zeros_like(instruments))
                drum_other_piece, _ = librosa.load(drum_other, sr=sr, mono=False)
                for piece_file in final_drumsep:
                    piece_low = os.path.basename(piece_file).lower()
                    arr_piece, _ = librosa.load(piece_file, sr=sr, mono=False)
                    # Subtract arr_piece from drum_other_piece
                    drum_other_piece -= arr_piece
                    if arr_piece.ndim == 1:
                        arr_piece = np.stack([arr_piece, arr_piece], axis=0)
                    if "(kick)" in piece_low:
                        stems["drums_kick"] = arr_piece
                        sr_dict["drums_kick"] = sr
                    elif "(snare)" in piece_low:
                        stems["drums_snare"] = arr_piece
                        sr_dict["drums_snare"] = sr
                    elif "(toms)" in piece_low:
                        stems["drums_toms"] = arr_piece
                        sr_dict["drums_toms"] = sr
                    elif "(hh)" in piece_low:
                        stems["drums_hh"] = arr_piece
                        sr_dict["drums_hh"] = sr
                    elif "(ride)" in piece_low:
                        stems["drums_ride"] = arr_piece
                        sr_dict["drums_ride"] = sr
                    elif "(crash)" in piece_low:
                        stems["drums_crash"] = arr_piece
                        sr_dict["drums_crash"] = sr
                stems["drums_other"] = drum_other_piece
                sr_dict["drums_other"] = sr
        else:
            stems["drums"] = np.zeros_like(instruments)
            sr_dict["drums"] = sr

        # Guitar
        if guitar_file and os.path.exists(guitar_file):
            guitar_data, _ = librosa.load(guitar_file, sr=sr, mono=False)
            if guitar_data.ndim == 1:
                guitar_data = np.stack([guitar_data, guitar_data], axis=0)
            stems["guitar"] = guitar_data
            sr_dict["guitar"] = sr
        else:
            stems["guitar"] = np.zeros_like(instruments)
            sr_dict["guitar"] = sr

        # Piano
        if piano_file and os.path.exists(piano_file):
            piano_data, _ = librosa.load(piano_file, sr=sr, mono=False)
            if piano_data.ndim == 1:
                piano_data = np.stack([piano_data, piano_data], axis=0)
            stems["piano"] = piano_data
            sr_dict["piano"] = sr
        else:
            stems["piano"] = np.zeros_like(instruments)
            sr_dict["piano"] = sr

        # Other
        if other_file and os.path.exists(other_file):
            other_data, _ = librosa.load(other_file, sr=sr, mono=False)
            if other_data.ndim == 1:
                other_data = np.stack([other_data, other_data], axis=0)
            stems["other"] = other_data
            sr_dict["other"] = sr
        else:
            stems["other"] = np.zeros_like(instruments)
            sr_dict["other"] = sr

        return stems, sr_dict

    def _build_vocals_ensemble(self, mix_np, sr):
        """
        1) MDX23C InstVoc
        2) Roformer 368
        3) optional VOC-FT
        Weighted multi-band => final vocals & final instrumentals
        """

        # --- 1) Separate using MDX23C and Roformer. Optional VOC-FT. ---
        mdx_files = self._separate_as_arrays(mix_np, sr, "MDX23C-8KFFT-InstVoc_HQ.ckpt")
        mdx_vocals = mdx_files.get("vocals", np.zeros_like(mix_np))
        mdx_instruments = mdx_files.get("instrumental", np.zeros_like(mix_np))

        rof_files = self._separate_as_arrays(mix_np, sr, "model_bs_roformer_ep_368_sdr_12.9628.ckpt")
        rof_vocals = rof_files.get("vocals", np.zeros_like(mix_np))
        rof_instruments = rof_files.get("instrumental", np.zeros_like(mix_np))

        vocft_vocals = None
        vocft_instruments = None
        if self.use_vocft:
            vft_files = self._separate_as_arrays(mix_np, sr, "UVR-MDX-NET-Voc_FT.onnx")
            vocft_vocals = vft_files.get("vocals", np.zeros_like(mix_np))
            vocft_instruments = vft_files.get("instrumental", np.zeros_like(mix_np))

        # --- 2) Match shapes just in case. ---
        mdx_vocals = match_array_shapes(mdx_vocals, mix_np)
        rof_vocals = match_array_shapes(rof_vocals, mix_np)
        mdx_instruments = match_array_shapes(mdx_instruments, mix_np)
        rof_instruments = match_array_shapes(rof_instruments, mix_np)

        if vocft_vocals is not None:
            vocft_vocals = match_array_shapes(vocft_vocals, mix_np)
            vocft_instruments = match_array_shapes(vocft_instruments, mix_np)

        # --- 3) Weighted averaging for vocals (low band). ---
        if vocft_vocals is None:
            wsum_v = self.weight_inst + self.weight_rof
            vocals_low_mix = (self.weight_inst * mdx_vocals + self.weight_rof * rof_vocals) / wsum_v
        else:
            wsum_v = self.weight_vocft + self.weight_inst + self.weight_rof
            vocals_low_mix = (
                                     self.weight_vocft * vocft_vocals +
                                     self.weight_inst * mdx_vocals +
                                     self.weight_rof * rof_vocals
                             ) / wsum_v

        # --- 4) Multi-band final vocals. Use MDX23C as high band reference. ---
        vocals_low = lr_filter(vocals_low_mix, 10000, "lowpass", 6, sr) * 1.01055
        vocals_high = lr_filter(mdx_vocals, 10000, "highpass", 6, sr)
        final_vocals = vocals_low + vocals_high

        # --- 5) Weighted averaging for instruments (low band). ---
        if vocft_instruments is None:
            wsum_i = self.weight_inst + self.weight_rof
            inst_low_mix = (self.weight_inst * mdx_instruments + self.weight_rof * rof_instruments) / wsum_i
        else:
            wsum_i = self.weight_vocft + self.weight_inst + self.weight_rof
            inst_low_mix = (
                                   self.weight_vocft * vocft_instruments +
                                   self.weight_inst * mdx_instruments +
                                   self.weight_rof * rof_instruments
                           ) / wsum_i

        # --- 6) Multi-band final instrumentals. Again, MDX23C as high band reference. ---
        inst_low = lr_filter(inst_low_mix, 10000, "lowpass", 6, sr)
        inst_high = lr_filter(mdx_instruments, 10000, "highpass", 6, sr)
        final_instrumentals = inst_low + inst_high

        # Return two ensemble arrays
        return final_vocals, final_instrumentals

    def _separate_as_arrays(self, mix_np, sr, model_name: str):
        """
        Writes mix_np -> temp wave, calls self.separator, returns dict of { 'vocals': array, 'instrumental': array, etc. }
        """
        stems_dict = {}
        out_dir = self.options["output_folder"]
        tmp_wav = write_temp_wav(mix_np, sr, out_dir)

        self.separator.load_model(model_name)
        out_files = self.separator.separate(tmp_wav)

        final_paths = [os.path.join(self.separator.output_dir, f) for f in out_files]

        for fp in final_paths:
            fn_low = os.path.basename(fp).lower()
            if not os.path.exists(fp):
                continue
            arr, _ = librosa.load(fp, sr=sr, mono=False)
            if arr.ndim == 1:
                arr = np.stack([arr, arr], axis=0)

            if "(vocals)" in fn_low:
                stems_dict["vocals"] = arr
            elif "(instrumental)" in fn_low:
                stems_dict["instrumental"] = arr
            elif "(drums)" in fn_low:
                stems_dict["drums"] = arr
            elif "(bass)" in fn_low:
                stems_dict["bass"] = arr
            elif "(other)" in fn_low:
                stems_dict["other"] = arr
            elif "(guitar)" in fn_low:
                stems_dict["guitar"] = arr
            elif "(piano)" in fn_low:
                stems_dict["piano"] = arr

        return stems_dict


################################################################################
#                    TOP-LEVEL PREDICTION + OUTPUT ROUTINE
################################################################################

def predict_with_model(options: Dict) -> List[str]:
    """
    Runs the ensemble separation on each file, writing final stems to disk.
    Optionally does advanced multi-drum separation if separate_drums=True.
    """
    output_files = []
    model = EnsembleDemucsMDXMusicSeparationModel(options)

    input_audio_list = options.get("input_audio", [])
    output_folder = options.get("output_folder", "./separated")
    output_format = options.get("output_format", "FLOAT")
    callback = options.get("callback", None)

    def safe_callback(step, desc, total):
        if callable(callback):
            callback(step, desc, total)
        print(desc)

    total_steps = len(input_audio_list) * 3
    current_step = 0

    safe_callback(current_step, "Initializing advanced ensemble model", total_steps)

    for i, ipath in enumerate(input_audio_list):
        current_step += 1
        safe_callback(current_step, f"Reading {ipath}", total_steps)
        if not os.path.isfile(ipath):
            print(f"Skipping missing file: {ipath}")
            continue

        # Convert if needed
        ipath_wav = ensure_wav(ipath, sr=44100)
        audio, sr = librosa.load(ipath_wav, sr=44100, mono=False)
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=0)

        # separate
        current_step += 1
        safe_callback(current_step, f"Separating {os.path.basename(ipath)}", total_steps)
        stems, sr_dict = model.separate_music_file(audio, sr)

        # write final
        current_step += 1
        safe_callback(current_step, f"Writing stems for {os.path.basename(ipath)}", total_steps)

        base = os.path.splitext(os.path.basename(ipath))[0]
        for stem_name, stem_data in stems.items():
            # e.g. 'vocals', 'drums_kick', etc.
            # We'll build fancy name e.g. (Vocals) or (Drums_Kick)
            suffix = stem_name.lower()
            suffix = suffix.replace("vocals", "(Vocals)")
            suffix = suffix.replace("drums_kick", "(Drums_Kick)")
            suffix = suffix.replace("drums_snare", "(Drums_Snare)")
            suffix = suffix.replace("drums_toms", "(Drums_Toms)")
            suffix = suffix.replace("drums_hh", "(Drums_HH)")
            suffix = suffix.replace("drums_ride", "(Drums_Ride)")
            suffix = suffix.replace("drums_crash", "(Drums_Crash)")
            suffix = suffix.replace("drums_other", "(Drums_Other)")
            suffix = suffix.replace("drums", "(Drums)")
            suffix = suffix.replace("bass", "(Bass)")
            suffix = suffix.replace("other", "(Other)")
            suffix = suffix.replace("guitar", "(Guitar)")
            suffix = suffix.replace("piano", "(Piano)")
            suffix = suffix.replace("instrumental", "(Instrumental)")
            suffix = suffix.replace("instrum", "(Instrumental)")

            out_wav = f"{base}_{suffix}.wav"
            out_path = os.path.join(output_folder, out_wav)
            sf.write(out_path, stem_data.T, sr_dict[stem_name], subtype=output_format)
            output_files.append(out_path)

    safe_callback(total_steps, "All files processed", total_steps)
    return output_files


def separate_music(
        input_audio: List[str],
        output_folder: str,
        cpu: bool = False,
        overlap_demucs: float = 0.1,
        overlap_VOCFT: float = 0.1,
        overlap_VitLarge: int = 1,
        overlap_InstVoc: int = 1,
        weight_InstVoc: float = 8,
        weight_VOCFT: float = 1,
        weight_VitLarge: float = 5,
        single_onnx: bool = False,
        large_gpu: bool = False,
        BigShifts: int = 7,
        vocals_only: bool = False,
        use_VOCFT: bool = False,
        output_format: str = "FLOAT",
        callback=None,
        separate_drums: bool = False,
        separate_woodwinds: bool = False,
        alt_bass_model: bool = False,
) -> List[str]:
    """
    A convenience wrapper so other code can call this function by name
    (just like your old `separate_music` function).
    """

    start_time = time()
    print("Options: ", {
        "input_audio": input_audio,
        "output_folder": output_folder,
        "cpu": cpu,
        "overlap_demucs": overlap_demucs,
        "overlap_VOCFT": overlap_VOCFT,
        "overlap_VitLarge": overlap_VitLarge,
        "overlap_InstVoc": overlap_InstVoc,
        "weight_InstVoc": weight_InstVoc,
        "weight_VOCFT": weight_VOCFT,
        "weight_VitLarge": weight_VitLarge,
        "single_onnx": single_onnx,
        "large_gpu": large_gpu,
        "BigShifts": BigShifts,
        "vocals_only": vocals_only,
        "use_VOCFT": use_VOCFT,
        "output_format": output_format,
        "separate_drums": separate_drums,
        "separate_woodwinds": separate_woodwinds,
        "alt_bass_model": alt_bass_model,
    })

    os.makedirs(output_folder, exist_ok=True)

    user_opts = {
        "input_audio": input_audio,
        "output_folder": output_folder,
        "cpu": cpu,
        "vocals_only": vocals_only,
        "use_VOCFT": use_VOCFT,
        "separate_drums": separate_drums,
        "weight_InstVoc": weight_InstVoc,
        "weight_VOCFT": weight_VOCFT,
        "weight_VitLarge": weight_VitLarge,
        # ignoring extra overlap/bigshifts for now (or pass them if you want to implement)
    }

    out_files = predict_with_model(user_opts)
    # OPTIONAL: remove tmp_ files if you want
    for fname in os.listdir(output_folder):
        if fname.startswith("tmp_"):
            os.remove(os.path.join(output_folder, fname))

    print("Time: {:.1f} sec".format(time() - start_time))
    return out_files
