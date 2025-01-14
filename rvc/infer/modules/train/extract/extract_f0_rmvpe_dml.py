import logging
import os
import traceback

import numpy as np

from rvc.infer.lib.audio import load_audio

logging.getLogger("numba").setLevel(logging.WARNING)

device = None


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
        # p_len = x.shape[0] // self.hop
        if f0_method == "rmvpe":
            if not hasattr(self, "model_rmvpe"):
                from rvc.infer.lib.rmvpe import RMVPE

                print("Loading rmvpe model")
                self.model_rmvpe = RMVPE(
                    "assets/rmvpe/rmvpe.pt", is_half=False, device=device
                )
            f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
        return f0

    def coarse_f0(self, f0):
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * (
                self.f0_bin - 2
        ) / (self.f0_mel_max - self.f0_mel_min) + 1

        # use 0 or 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > self.f0_bin - 1] = self.f0_bin - 1
        f0_coarse = np.rint(f0_mel).astype(int)
        assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (
            f0_coarse.max(),
            f0_coarse.min(),
        )
        return f0_coarse

    def go(self, paths, f0_method):
        if len(paths) == 0:
            print("no-f0-todo")
        else:
            print("todo-f0-%s" % len(paths))
            n = max(len(paths) // 5, 1)  # 每个进程最多打印5条
            for idx, (inp_path, opt_path1, opt_path2) in enumerate(paths):
                try:
                    if idx % n == 0:
                        print("f0ing,now-%s,all-%s,-%s" % (idx, len(paths), inp_path))
                    if (
                            os.path.exists(opt_path1 + ".npy") == True
                            and os.path.exists(opt_path2 + ".npy") == True
                    ):
                        continue
                    featur_pit = self.compute_f0(inp_path, f0_method)
                    np.save(
                        opt_path2,
                        featur_pit,
                        allow_pickle=False,
                    )  # nsf
                    coarse_pit = self.coarse_f0(featur_pit)
                    np.save(
                        opt_path1,
                        coarse_pit,
                        allow_pickle=False,
                    )  # ori
                except:
                    print("f0fail-%s-%s-%s" % (idx, inp_path, traceback.format_exc()))


def extract_f0_features_rmvpe_dml(exp_dir):
    global device
    f = open("%s/extract_f0_feature.log" % exp_dir, "a+")

    try:
        import torch_directml
        device = torch_directml.device(torch_directml.default_device())

    except ImportError:
        print("Please install torch-directml first.")

    device = torch_directml.device(torch_directml.default_device())
    with open(f"{exp_dir}/extract_f0_feature.log", "a+") as f:
        print("Starting RMVPE F0 feature extraction with DirectML", f)
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

        try:
            feature_input.go(paths, "rmvpe")
        except Exception:
            print(f"f0_all_fail-{traceback.format_exc()}", f)
