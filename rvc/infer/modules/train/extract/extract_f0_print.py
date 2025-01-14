import logging
import os
import traceback
from multiprocessing import Process

import numpy as np
import parselmouth
import pyworld

from handlers.config import model_path
from rvc.infer.lib.audio import load_audio

logging.getLogger("numba").setLevel(logging.WARNING)


class FeatureInput(object):
    def __init__(self, samplerate=16000, hop_size=160):
        self.fs = samplerate
        self.hop = hop_size

        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

    def compute_f0(self, path, f0_method):
        x = load_audio(path, self.fs)
        p_len = x.shape[0] // self.hop
        if f0_method == "pm":
            time_step = 160 / 16000 * 1000
            f0_min = 50
            f0_max = 1100
            f0 = (
                parselmouth.Sound(x, self.fs)
                .to_pitch_ac(
                    time_step=time_step / 1000,
                    voicing_threshold=0.6,
                    pitch_floor=f0_min,
                    pitch_ceiling=f0_max,
                )
                .selected_array["frequency"]
            )
            pad_size = (p_len - len(f0) + 1) // 2
            if pad_size > 0 or p_len - len(f0) - pad_size > 0:
                f0 = np.pad(
                    f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant"
                )
        elif f0_method == "harvest":
            f0, t = pyworld.harvest(
                x.astype(np.double),
                fs=self.fs,
                f0_ceil=self.f0_max,
                f0_floor=self.f0_min,
                frame_period=1000 * self.hop / self.fs,
            )
            f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.fs)
        elif f0_method == "dio":
            f0, t = pyworld.dio(
                x.astype(np.double),
                fs=self.fs,
                f0_ceil=self.f0_max,
                f0_floor=self.f0_min,
                frame_period=1000 * self.hop / self.fs,
            )
            f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.fs)
        else:
            if not hasattr(self, "model_rmvpe"):
                from rvc.infer.lib.rmvpe import RMVPE

                print("Loading rmvpe model")
                rmvpe_path = os.path.join(model_path, "rvc", "rmvpe.pt")
                self.model_rmvpe = RMVPE(
                    rmvpe_path, is_half=False, device="cpu"
                )
            f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
        return f0

    def coarse_f0(self, f0):
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * (
                self.f0_bin - 2
        ) / (self.f0_mel_max - self.f0_mel_min) + 1

        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > self.f0_bin - 1] = self.f0_bin - 1
        f0_coarse = np.rint(f0_mel).astype(int)
        assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (
            f0_coarse.max(),
            f0_coarse.min(),
        )
        return f0_coarse

    def go(self, paths, f0_method, f):
        if len(paths) == 0:
            print("no-f0-todo", f)
        else:
            print(f"todo-f0-{len(paths)}", f)
            n = max(len(paths) // 5, 1)
            for idx, (inp_path, opt_path1, opt_path2) in enumerate(paths):
                try:
                    if idx % n == 0:
                        print(f"f0ing,now-{idx},all-{len(paths)},-{inp_path}", f)
                    if (
                            os.path.exists(opt_path1 + ".npy")
                            and os.path.exists(opt_path2 + ".npy")
                    ):
                        continue
                    featur_pit = self.compute_f0(inp_path, f0_method)
                    np.save(opt_path2, featur_pit, allow_pickle=False)
                    coarse_pit = self.coarse_f0(featur_pit)
                    np.save(opt_path1, coarse_pit, allow_pickle=False)
                except Exception:
                    print(f"f0fail-{idx}-{inp_path}-{traceback.format_exc()}", f)


def extract_f0_features(exp_dir, n_p, f0method):
    with open(f"{exp_dir}/extract_f0_feature.log", "a+") as f:
        print("Starting F0 feature extraction", f)
        feature_input = FeatureInput()
        paths = []
        inp_root = f"{exp_dir}/1_16k_wavs"
        opt_root1 = f"{exp_dir}/2a_f0"
        opt_root2 = f"{exp_dir}/2b-f0nsf"

        os.makedirs(opt_root1, exist_ok=True)
        os.makedirs(opt_root2, exist_ok=True)

        for name in sorted(os.listdir(inp_root)):
            inp_path = f"{inp_root}/{name}"
            if "spec" in inp_path:
                continue
            opt_path1 = f"{opt_root1}/{name}"
            opt_path2 = f"{opt_root2}/{name}"
            paths.append([inp_path, opt_path1, opt_path2])

        processes = []
        for i in range(n_p):
            p = Process(target=feature_input.go, args=(paths[i::n_p], f0method, f))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
