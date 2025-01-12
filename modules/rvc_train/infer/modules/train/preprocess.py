import json
import multiprocessing
import os
import sys
import traceback
from pathlib import Path

import librosa
import numpy as np
from scipy import signal
from scipy.io import wavfile

from modules.rvc_train.infer.lib.audio import load_audio
from modules.rvc_train.infer.lib.slicer2 import Slicer

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
        )  # , res_type="soxr_vhq"
        wavfile.write(
            f"{self.wavs16k_dir}/{idx0}_{idx1}.wav",
            16000,
            tmp_audio.astype(np.float32),
        )

    def pipeline(self, path, idx0):
        try:
            audio = load_audio(path, self.sr)
            # zero phased digital filter cause pre-ringing noise...
            # audio = signal.filtfilt(self.bh, self.ah, audio)
            audio = signal.lfilter(self.bh, self.ah, audio)

            idx1 = 0
            for audio in self.slicer.slice(audio):
                i = 0
                while 1:
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
            # print(f"{idx0}-{idx1}")
        except:
            print(f"{path}-{traceback.format_exc()}")

    def pipeline_mp(self, infos):
        for path, idx0 in infos:
            self.pipeline(path, idx0)

    def pipeline_mp_inp_dir(self, inp_root: Path, name2id_save_path: Path, n_p=8):
        try:
            files = list(inp_root.glob("**/*.wav")) + list(inp_root.glob("**/*.flac"))
            infos = []
            name2id = {}
            for idx, path in enumerate(sorted(files), self.start_idx):
                name2id[str(path)] = idx
                infos.append((str(path), idx))
            name2id_save_path.write_text(json.dumps(name2id))

            if False:
                for i in range(n_p):
                    self.pipeline_mp(infos[i::n_p])
            else:
                ps = []
                for i in range(n_p):
                    p = multiprocessing.Process(
                        target=self.pipeline_mp, args=(infos[i::n_p],)
                    )
                    ps.append(p)
                    p.start()
                for i in range(n_p):
                    ps[i].join()
        except:
            print("Fail. %s" % traceback.format_exc())


def preprocess_trainset(inp_root, sr, n_p, exp_dir, per, start_idx, name2id_save_path):
    if isinstance(exp_dir, str):
        exp_dir = Path(exp_dir)
    if isinstance(inp_root, str):
        inp_root = Path(inp_root)
    if isinstance(name2id_save_path, str):
        name2id_save_path = Path(name2id_save_path)
    pp = PreProcess(sr, exp_dir, per, start_idx)
    print("start preprocess")
    print(sys.argv)
    pp.pipeline_mp_inp_dir(inp_root, n_p=n_p, name2id_save_path=name2id_save_path)
    print("end preprocess")


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--inp_root", type=Path, required=True)
#     parser.add_argument("--sr", type=int, default=44100)
#     parser.add_argument("--n_p", type=int, default=1)
#     parser.add_argument("--exp_dir", type=Path, required=True)
#     parser.add_argument("--noparallel", type=bool, default=False)
#     parser.add_argument("--per", type=float, default=3.0)
#     parser.add_argument("--name2id_save_path", type=Path, required=True, help='save mapping name2idx')
#     parser.add_argument("--start_idx", type=int, default=0,
#                         help='needed when adding new datasets into the mixture. idx for new data would start from this value to avoid name collisions')
#     args = parser.parse_args()
#     return args
#
#
# if __name__ == "__main__":
#     now_dir = os.getcwd()
#     sys.path.append(now_dir)
#
#     args = parse_args()
#     inp_root = args.inp_root
#     sr = args.sr
#     n_p = args.n_p
#     exp_dir = args.exp_dir
#     exp_dir.mkdir(exist_ok=True, parents=True)
#     (exp_dir / "preprocess.log").touch()
#     noparallel = args.noparallel
#     per = args.per
#
#     # f = open("%s/preprocess.log" % exp_dir, "a+")
#     f = Path(f"{exp_dir}/preprocess.log").open("a+")
#     preprocess_trainset(inp_root, sr, n_p, exp_dir, per, start_idx=args.start_idx,
#                         name2id_save_path=args.name2id_save_path)
