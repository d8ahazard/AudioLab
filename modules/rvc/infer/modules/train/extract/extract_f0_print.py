import logging
import os
import traceback
from multiprocessing import Process

import numpy as np
import parselmouth
import pyworld

from handlers.config import model_path
from modules.rvc.infer.lib.audio import load_audio

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

    def compute_f0(self, audio_path, f0_method):
        audio = load_audio(audio_path, self.fs)
        p_len = audio.shape[0] // self.hop
        if f0_method == "pm":
            time_step = 160 / self.fs * 1000
            f0_min = 50
            f0_max = 1100
            f0 = (
                parselmouth.Sound(audio, self.fs)
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
                audio.astype(np.double),
                fs=self.fs,
                f0_ceil=self.f0_max,
                f0_floor=self.f0_min,
                frame_period=1000 * self.hop / self.fs,
            )
            f0 = pyworld.stonemask(audio.astype(np.double), f0, t, self.fs)
        elif f0_method == "dio":
            f0, t = pyworld.dio(
                audio.astype(np.double),
                fs=self.fs,
                f0_ceil=self.f0_max,
                f0_floor=self.f0_min,
                frame_period=1000 * self.hop / self.fs,
            )
            f0 = pyworld.stonemask(audio.astype(np.double), f0, t, self.fs)
        else:
            if not hasattr(self, "model_rmvpe"):
                from modules.rvc.infer.lib.rmvpe import RMVPE

                logger = logging.getLogger(__name__)
                logger.info("Loading RMVPE model")
                rmvpe_path = os.path.join(model_path, "rvc", "rmvpe.pt")
                if not os.path.exists(rmvpe_path):
                    logger.error("Error: RMVPE model file %s does not exist.", rmvpe_path)
                    raise FileNotFoundError("RMVPE model file not found: " + rmvpe_path)
                self.model_rmvpe = RMVPE(rmvpe_path, is_half=False, device="cpu")
            f0 = self.model_rmvpe.infer_from_audio(audio, thred=0.03)
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

    def go(self, paths, f0_method):
        logger = logging.getLogger(__name__)
        if len(paths) == 0:
            logger.info("No F0 tasks to process")
        else:
            logger.info("Number of F0 tasks to process: %s", len(paths))
            n = max(len(paths) // 5, 1)
            for idx, (inp_path, opt_path1, opt_path2) in enumerate(paths):
                try:
                    if idx % n == 0:
                        logger.info("Processing F0: %s of %s, input file: %s",
                                    idx, len(paths), inp_path)
                    if os.path.exists(opt_path1 + ".npy") and os.path.exists(opt_path2 + ".npy"):
                        continue
                    f0_values = self.compute_f0(inp_path, f0_method)
                    np.save(opt_path2, f0_values, allow_pickle=False)
                    coarse_values = self.coarse_f0(f0_values)
                    np.save(opt_path1, coarse_values, allow_pickle=False)
                except Exception:
                    logger.error("F0 extraction failed for index %s, file %s: %s",
                                 idx, inp_path, traceback.format_exc())


def extract_f0_features(exp_dir, n_p, f0_method):
    logger = logging.getLogger(__name__)
    logger.info("Starting F0 feature extraction")
    feature_input = FeatureInput()
    paths = []
    inp_root = os.path.join(exp_dir, "1_16k_wavs")
    opt_root1 = os.path.join(exp_dir, "2a_f0")
    opt_root2 = os.path.join(exp_dir, "2b-f0nsf")

    if not os.path.exists(inp_root):
        logger.error("Error: Input directory %s does not exist.", inp_root)
        raise FileNotFoundError("Input directory not found: " + inp_root)

    os.makedirs(opt_root1, exist_ok=True)
    os.makedirs(opt_root2, exist_ok=True)

    for name in sorted(os.listdir(inp_root)):
        inp_path = os.path.join(inp_root, name)
        if "spec" in name:
            continue
        opt_path1 = os.path.join(opt_root1, name)
        opt_path2 = os.path.join(opt_root2, name)
        paths.append([inp_path, opt_path1, opt_path2])

    processes = []
    for i in range(n_p):
        p = Process(target=feature_input.go, args=(paths[i::n_p], f0_method))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
