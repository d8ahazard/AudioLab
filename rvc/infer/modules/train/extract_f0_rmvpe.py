import argparse
import logging
import os
import traceback
from pathlib import Path
from typing import Callable

import numpy as np

from handlers.config import model_path
from rvc.infer.lib.audio import load_audio
from rvc.infer.lib.rmvpe import RMVPE

hf_dir = os.path.join(model_path, "hf")
os.makedirs(hf_dir, exist_ok=True)
# Set HF_HUB_CACHE_DIR to the model_path
os.environ["HF_HOME"] = hf_dir

logging.getLogger("numba").setLevel(logging.WARNING)
# Todo: Make this not hacky like the OG dev did...

is_half = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default=None)
    parser.add_argument("--n_part", type=int, default=1)
    parser.add_argument("--i_part", type=int, default=0)
    parser.add_argument("--i_gpu", type=int, default=0)
    parser.add_argument("--is_half", action="store_false")
    # is_half by default is True
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.i_gpu)
    # f = open("%s/extract_f0_feature.log" % args.exp_dir, "a+")
    return args


def printt(strr):
    print(strr)
    # f.write("%s\n" % strr)
    # f.flush()


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
                print("Loading rmvpe model")
                rvc_model_file = os.path.join(model_path, "rvc", "rmvpe.pt")
                self.model_rmvpe = RMVPE(
                    rvc_model_file, is_half=is_half, device="cuda"
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

    def go(self, paths, f0_method, callback: Callable = None):
        if len(paths) == 0:
            printt("no-f0-todo")
        else:
            printt("todo-f0-%s" % len(paths))
            n = max(len(paths) // 5, 1)
            for idx, (inp_path, opt_path1, opt_path2) in enumerate(paths):
                try:
                    if idx % n == 0:
                        printt("f0ing,now-%s,all-%s,-%s" % (idx, len(paths), inp_path))
                    if Path(opt_path1).exists() and Path(opt_path2).exists():
                        continue
                    # Ensure folders exist
                    os.makedirs(os.path.dirname(opt_path1), exist_ok=True)
                    os.makedirs(os.path.dirname(opt_path2), exist_ok=True)
                    feature_pitch = self.compute_f0(inp_path, f0_method)
                    np.save(
                        opt_path2,
                        feature_pitch,
                        allow_pickle=False,
                    )  # nsf
                    coarse_pit = self.coarse_f0(feature_pitch)
                    np.save(
                        opt_path1,
                        coarse_pit,
                        allow_pickle=False,
                    )  # ori
                except:
                    printt("f0fail-%s-%s-%s" % (idx, inp_path, traceback.format_exc()))


def extract_f0(exp_dir, n_part=1, i_part=0, i_gpu=0, is_half=True, callback: Callable = None):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(i_gpu)
    feature_input = FeatureInput()
    paths = []
    inp_root = Path(exp_dir) / "1_16k_wavs"
    opt_root1 = Path(exp_dir) / "2a_f0"
    opt_root2 = Path(exp_dir) / "2b-f0nsf"

    os.makedirs(opt_root1, exist_ok=True)
    os.makedirs(opt_root2, exist_ok=True)

    for name in sorted(inp_root.glob("*.wav")):
        paths.append((str(name), str(opt_root1 / name.stem), str(opt_root2 / name.stem)))

    feature_input.go(paths[i_part::n_part], "rmvpe", callback=callback)


# if __name__ == "__main__":
#     args = parse_args()
#     exp_dir = args.exp_dir
#     n_part = args.n_part
#     i_part = args.i_part
#     i_gpu = args.i_gpu
#     is_half = args.is_half
#
#     featureInput = FeatureInput()
#     paths = []
#     inp_root = f"{exp_dir}/1_16k_wavs"
#     opt_root1 = f"{exp_dir}/2a_f0"
#     opt_root2 = f"{exp_dir}/2b-f0nsf"
#
#     os.makedirs(opt_root1, exist_ok=True)
#     os.makedirs(opt_root2, exist_ok=True)
#     for name in sorted(list(os.listdir(inp_root))):
#         inp_path = "%s/%s" % (inp_root, name)
#         if "spec" in inp_path:
#             continue
#         opt_path1 = "%s/%s" % (opt_root1, name)
#         opt_path2 = "%s/%s" % (opt_root2, name)
#         paths.append([inp_path, opt_path1, opt_path2])
#     try:
#         featureInput.go(paths[i_part::n_part], "rmvpe")
#     except:
#         printt("f0_all_fail-%s" % (traceback.format_exc()))
