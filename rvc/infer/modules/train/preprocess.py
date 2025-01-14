import json
import multiprocessing
import os
import sys
import traceback
from pathlib import Path
from typing import Callable

import librosa
import numpy as np
from scipy import signal
from scipy.io import wavfile

from handlers.config import model_path
from rvc.infer.lib.audio import load_audio
from rvc.infer.lib.slicer2 import Slicer

hf_dir = os.path.join(model_path, "hf")
os.makedirs(hf_dir, exist_ok=True)
# Set HF_HUB_CACHE_DIR to the model_path
os.environ["HF_HOME"] = hf_dir

mutex = multiprocessing.Lock()


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
        self.gt_wavs_dir = f"{exp_dir}/0_gt_wavs"
        self.wavs16k_dir = f"{exp_dir}/1_16k_wavs"
        self.start_idx = start_idx

        if Path(self.gt_wavs_dir).exists() and self.start_idx == 0:
            print(
                "gt_wavs_dir exists but start idx is 0. This would cause name collisions. Make sure you are using correct exp dir")

        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.gt_wavs_dir, exist_ok=True)
        os.makedirs(self.wavs16k_dir, exist_ok=True)

    def norm_write(self, tmp_audio, idx0, idx1):
        # Ensure tmp_audio has finite values
        if not np.isfinite(tmp_audio).all():
            print(f"{idx0}-{idx1}: Non-finite values encountered, skipping this audio segment.")
            return

        tmp_max = np.abs(tmp_audio).max()
        if tmp_max > 2.5:
            print(f"{idx0}-{idx1}-{tmp_max}-filtered")
            return

        tmp_audio = (tmp_audio / tmp_max * (self.max * self.alpha)) + (
                1 - self.alpha
        ) * tmp_audio
        wavfile.write(
            "%s/%s_%s.wav" % (self.gt_wavs_dir, idx0, idx1),
            self.sr,
            tmp_audio.astype(np.float32),
        )
        tmp_audio = librosa.resample(
            tmp_audio, orig_sr=self.sr, target_sr=16000
        )
        wavfile.write(
            f"{self.wavs16k_dir}/{idx0}_{idx1}.wav",
            16000,
            tmp_audio.astype(np.float32),
        )

    def pipeline(self, path, idx0):
        try:
            audio = load_audio(path, self.sr)
            audio = signal.lfilter(self.bh, self.ah, audio)

            idx1 = 0
            for audio in self.slicer.slice(audio):
                i = 0
                while True:
                    start = int(self.sr * (self.per - self.overlap) * i)
                    i += 1
                    if len(audio[start:]) > self.tail * self.sr:
                        tmp_audio = audio[start: start + int(self.per * self.sr)]
                        self.norm_write(tmp_audio, idx0, idx1)
                        idx1 += 1
                    else:
                        tmp_audio = audio[start:]
                        idx1 += 1
                        break
                    self.norm_write(tmp_audio, idx0, idx1)
        except Exception as e:
            print(f"Error processing {path}: {e}")
            traceback.print_exc()

    def pipeline_mp(self, infos):
        for path, idx0 in infos:
            self.pipeline(path, idx0)

    def pipeline_mp_inp_dir(self, inp_root: Path, name2id_save_path: Path, n_p=8, callback: Callable = None):
        try:
            files = list(inp_root.glob("**/*.wav")) + list(inp_root.glob("**/*.flac"))
            if not files:
                print(f"No audio files found in {inp_root}")
                return
            print(f"Found {len(files)} files: {files}")

            infos = []
            name2id = {}
            for idx, path in enumerate(sorted(files), self.start_idx):
                name2id[str(path)] = idx
                infos.append((str(path), idx))
            name2id_save_path.write_text(json.dumps(name2id))

            processes = []
            for i in range(n_p):
                if callback:
                    print("Implement callback for pipeline_mp_inp_dir")
                p = multiprocessing.Process(target=self.pipeline_mp, args=(infos[i::n_p],))
                processes.append(p)
                p.start()
            for p in processes:
                p.join()
            print("All processes completed.")
        except Exception as e:
            print(f"Failed in pipeline_mp_inp_dir: {e}")
            traceback.print_exc()


def preprocess_trainset(inp_root, sr, n_p, exp_dir, per, start_idx, name2id_save_path, callback: Callable = None):
    if isinstance(exp_dir, str):
        exp_dir = Path(exp_dir)
    if isinstance(inp_root, str):
        inp_root = Path(inp_root)
    if isinstance(name2id_save_path, str):
        name2id_save_path = Path(name2id_save_path)
    pp = PreProcess(sr, exp_dir, per, start_idx)
    # Find all mp3s in inp_root that don't have a corresponding wav file and convert them to wav
    mp3s = list(inp_root.glob("**/*.mp3"))
    for mp3 in mp3s:
        wav_path = mp3.with_suffix(".wav")
        if not wav_path.exists():
            os.system(f"ffmpeg -i {mp3} {wav_path}")
    print("start preprocess")
    print(sys.argv)
    pp.pipeline_mp_inp_dir(inp_root, n_p=n_p, name2id_save_path=name2id_save_path, callback=callback)
    print("end preprocess")
