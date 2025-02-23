import multiprocessing
import os
import traceback
from pathlib import Path
from typing import Callable

import librosa
import numpy as np
from scipy import signal
from scipy.io import wavfile

from handlers.config import model_path
from modules.rvc.infer.lib.audio import load_audio
from modules.rvc.infer.lib.slicer2 import Slicer

hf_dir = os.path.join(model_path, "hf")
os.makedirs(hf_dir, exist_ok=True)
# Set HF_HUB_CACHE_DIR to the model_path
os.environ["HF_HOME"] = hf_dir
mutex = multiprocessing.Lock()


def println(strr):
    print(strr)


class PreProcess:
    def __init__(self, sr, exp_dir, per=3.0, start_idx=0):
        """
        per is the length of the audio to be sliced (3 seconds by default)
        """
        self.slicer = Slicer(
            sr=sr,
            threshold=-42,
            min_length=1500,
            min_interval=400,
            hop_size=15,
            max_sil_kept=500,
        )
        self.sr = sr
        self.bh, self.ah = signal.butter(N=5, Wn=48, btype="high", fs=self.sr)
        self.per = per
        self.overlap = 0.3
        self.tail = self.per + self.overlap
        self.max = 0.9
        self.alpha = 0.75
        self.exp_dir = exp_dir
        self.gt_wavs_dir = "%s/0_gt_wavs" % exp_dir
        self.wavs16k_dir = "%s/1_16k_wavs" % exp_dir

        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.gt_wavs_dir, exist_ok=True)
        os.makedirs(self.wavs16k_dir, exist_ok=True)

    # def norm_write(self, tmp_audio, idx0, idx1):
    #     # Construct paths using os.path.join
    #     gt_wav_path = os.path.join(self.gt_wavs_dir, f"{idx0}_{idx1}.wav")
    #     wavs16k_path = os.path.join(self.wavs16k_dir, f"{idx0}_{idx1}.wav")
    #     if os.path.exists(gt_wav_path) and os.path.exists(wavs16k_path):
    #         print(f"Skipping {idx0}-{idx1} as files already exist.")
    #         return
    #
    #     try:
    #         tmp_max = np.abs(tmp_audio).max()
    #         if tmp_max > 2.5:
    #             print(f"{idx0}-{idx1}-{tmp_max}-filtered")
    #             return
    #
    #         tmp_audio = (tmp_audio / tmp_max * (self.max * self.alpha)) + (
    #                 1 - self.alpha
    #         ) * tmp_audio
    #
    #         # # Debug: Check if normalization introduced issues
    #         # if not np.isfinite(tmp_audio).all():
    #         #     print(f"Error: Audio buffer contains NaN or infinite values AFTER normalization. Skipping {idx0}-{idx1}.")
    #         #     return
    #         #
    #         # Ensure finite values before writing/resampling
    #         tmp_audio = np.nan_to_num(tmp_audio, nan=0.0, posinf=0.0, neginf=0.0)
    #
    #         wavfile.write(gt_wav_path, self.sr, tmp_audio.astype(np.float32))
    #
    #         # Resample audio safely
    #         tmp_audio = librosa.resample(tmp_audio, orig_sr=self.sr, target_sr=16000)
    #
    #         wavfile.write(wavs16k_path, 16000, tmp_audio.astype(np.float32))
    #     except Exception as e:
    #         # Delete files if resampling fails
    #         if os.path.exists(gt_wav_path):
    #             os.remove(gt_wav_path)
    #         if os.path.exists(wavs16k_path):
    #             os.remove(wavs16k_path)
    #         print(f"Librosa resampling failed for {idx0}-{idx1}, skipping. | Error: {e}")
    #         return
    #
    #     print(f"Successfully processed {idx0}-{idx1}")

    def norm_write(self, tmp_audio, idx0, idx1):
        tmp_audio = (tmp_audio / np.abs(tmp_audio).max() * (self.max * self.alpha)) + (
                1 - self.alpha
        ) * tmp_audio
        wavfile.write(
            "%s/%s_%s.wav" % (self.gt_wavs_dir, idx0, idx1),
            self.sr,
            tmp_audio.astype(np.float32),
        )
        tmp_audio = librosa.resample(tmp_audio, orig_sr=self.sr, target_sr=16000)  # , res_type="soxr_vhq"
        wavfile.write(
            "%s/%s_%s.wav" % (self.wavs16k_dir, idx0, idx1),
            16000,
            tmp_audio.astype(np.float32),
        )

    def pipeline(self, path, idx0, callback: Callable = None):
        try:
            audio, sr = load_audio(path, self.sr)
            # Convert audio to float64 to match filter coefficients from butter()
            audio = audio.astype(np.float64)
            audio = signal.lfilter(self.bh, self.ah, audio)

            slices = list(self.slicer.slice(audio))
            total_steps = sum(
                max(1, int(len(audio) / (self.sr * (self.per - self.overlap))))
                for audio in slices
            )

            idx1 = 0
            processed_steps = 0

            for audio in slices:
                i = 0
                while True:
                    start = int(self.sr * (self.per - self.overlap) * i)
                    i += 1
                    if len(audio[start:]) > self.tail * self.sr:
                        tmp_audio = audio[start: start + int(self.per * self.sr)]
                        if callback is not None:
                            progress = processed_steps / total_steps
                            callback(progress, f"Processing chunk {idx1}", total_steps)
                        self.norm_write(tmp_audio, idx0, idx1)
                        idx1 += 1
                        processed_steps += 1
                    else:
                        tmp_audio = audio[start:]
                        idx1 += 1
                        break

                if callback is not None:
                    progress = processed_steps / total_steps
                    callback(progress, f"Processing chunk {idx1}", total_steps)
                self.norm_write(tmp_audio, idx0, idx1)
                processed_steps += 1

            if callback is not None:
                callback(1.0, "Processing complete", total_steps)

        except Exception:
            print(f"{path}\t-> {traceback.format_exc()}")

    def pipeline_mp(self, infos, callback: Callable = None):
        for path, idx0 in infos:
            self.pipeline(path, idx0, callback)

    def pipeline_mp_inp_dir(self, inp_root: Path, n_p=8, callback: Callable = None):
        noparallel = n_p <= 1
        if not isinstance(inp_root, Path):
            inp_root = Path(inp_root)
        try:
            infos = [
                (os.path.join(inp_root, name), idx)
                for idx, name in enumerate(sorted(list(os.listdir(inp_root))))
            ]
            if noparallel:
                for i in range(n_p):
                    if callback is not None:
                        callback(i / n_p, "Processing %s" % i, n_p)
                    self.pipeline_mp(infos[i::n_p], callback)
            else:

                ps = []
                for i in range(n_p):
                    p = multiprocessing.Process(
                        target=self.pipeline_mp, args=(infos[i::n_p], callback,)
                    )
                    ps.append(p)
                    p.start()
                for i in range(n_p):
                    if callback is not None:
                        callback(i / n_p, "Processed %s" % i, n_p)
                    ps[i].join()
        except:
            println("Fail. %s" % traceback.format_exc())


def preprocess_trainset(inp_root, sr, n_p, exp_dir, per, callback: Callable = None):
    pp = PreProcess(sr, exp_dir, per)
    println("start preprocess")
    pp.pipeline_mp_inp_dir(inp_root, n_p, callback)
    println("end preprocess")
