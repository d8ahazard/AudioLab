import logging
import os
import time
from functools import partial

import faiss
import numpy as np
import pyworld
import torch
import torchcrepe
from scipy import signal

from handlers.config import model_path
from handlers.spectrogram import F0Visualizer
from modules.rvc.infer.lib.audio import pad_audio, autotune_f0, remap_f0
from modules.rvc.infer.lib.rmvpe import RMVPE
from modules.rvc.utils import get_optimal_threads, get_merge_func, gc_collect

logger = logging.getLogger(__name__)

BASE_MODELS_DIR = os.path.join(model_path, "rvc")


class FeatureExtractor:
    def __init__(self, tgt_sr, config, onnx=False):
        self.x_pad, self.x_query, self.x_center, self.x_max, self.is_half = (
            config.x_pad,
            config.x_query,
            config.x_center,
            config.x_max,
            config.is_half,
        )

        self.sr = 16000
        self.window = 160
        self.t_pad = self.sr * self.x_pad
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sr * self.x_query
        self.t_center = self.sr * self.x_center
        self.t_max = self.sr * self.x_max
        self.device = config.device
        self.onnx = onnx
        self.f0_method_dict = {
            "pm": self.get_pm,
            "harvest": self.get_harvest,
            "dio": self.get_dio,
            "rmvpe": self.get_rmvpe,
            "rmvpe_onnx": self.get_rmvpe,
            "rmvpe+": self.get_pitch_dependant_rmvpe,
            "crepe": self.get_f0_official_crepe_computation,
            "crepe-tiny": partial(self.get_f0_official_crepe_computation, model='model'),
            "mangio-crepe": self.get_f0_crepe_computation,
            "mangio-crepe-tiny": partial(self.get_f0_crepe_computation, model='model')
        }
        self.vis = F0Visualizer()

    def __del__(self):
        if hasattr(self, "model_rmvpe") and self.model_rmvpe is not None:
            del self.model_rmvpe
            logger.info("RMVPE model deleted.")
            gc_collect()

    def load_index(self, file_index):
        try:
            if not type(file_index) == str:  # loading file index to save time
                logger.debug("Using preloaded file index.")
                index = file_index
                big_npy = index.reconstruct_n(0, index.ntotal)
            elif file_index == "":
                logger.debug("File index was empty.")
                index = None
                big_npy = None
            else:
                if os.path.isfile(file_index):
                    logger.debug(f"Attempting to load {file_index}....")
                else:
                    logger.debug(f"{file_index} was not found...")
                index = faiss.read_index(file_index)
                logger.debug(f"loaded index: {index}")
                big_npy = index.reconstruct_n(0, index.ntotal)
        except Exception as e:
            logger.debug(f"Could not open Faiss index file for reading. {e}")
            index = None
            big_npy = None
        return index, big_npy

    # Fork Feature: Compute f0 with the crepe method
    def get_f0_crepe_computation(
            self,
            x,
            f0_min,
            f0_max,
            *args,
            **kwargs,
    ):
        x = x.astype(np.float32)
        x /= np.quantile(np.abs(x), 0.999)
        audio = torch.from_numpy(x).to(self.device, copy=True)
        audio = torch.unsqueeze(audio, dim=0)
        if audio.ndim == 2 and audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True).detach()
        audio = audio.detach()
        hop_length = kwargs.get('crepe_hop_length', 160)
        model = kwargs.get('model', 'full')
        logger.debug("Initiating prediction with a crepe_hop_length of: " + str(hop_length))
        pitch: torch.Tensor = torchcrepe.predict(
            audio,
            self.sr,
            hop_length,
            f0_min,
            f0_max,
            model,
            batch_size=hop_length * 2,
            device=self.device,
            pad=True,
        )
        p_len = x.shape[0] // hop_length
        source = np.array(pitch.squeeze(0).cpu().float().numpy())
        source[source < 0.001] = np.nan
        target = np.interp(
            np.arange(0, len(source) * p_len, len(source)) / p_len,
            np.arange(0, len(source)),
            source,
        )
        f0 = np.nan_to_num(target)
        return f0

    def get_f0_official_crepe_computation(
            self,
            x,
            f0_min,
            f0_max,
            *args,
            **kwargs
    ):
        batch_size = 512
        audio = torch.tensor(np.copy(x))[None].float()
        model = kwargs.get('model', 'full')
        f0, pd = torchcrepe.predict(
            audio,
            self.sr,
            self.window,
            f0_min,
            f0_max,
            model,
            batch_size=batch_size,
            device=self.device,
            return_periodicity=True,
        )
        pd = torchcrepe.filter.median(pd, 3)
        f0 = torchcrepe.filter.mean(f0, 3)
        f0[pd < 0.1] = 0
        f0 = f0[0].cpu().numpy()
        return f0

    def get_pm(self, x, *args, **kwargs):
        import parselmouth
        p_len = x.shape[0] // 160 + 1
        f0 = parselmouth.Sound(x, self.sr).to_pitch_ac(
            time_step=0.01,
            voicing_threshold=0.6,
            pitch_floor=kwargs.get('f0_min'),
            pitch_ceiling=kwargs.get('f0_max'),
        ).selected_array["frequency"]

        pad_size = (p_len - len(f0) + 1) // 2
        if pad_size > 0 or p_len - len(f0) - pad_size > 0:
            f0 = np.pad(f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant")
        return f0

    def get_harvest(self, x, *args, **kwargs):
        f0_spectral = pyworld.harvest(
            x.astype(np.double),
            fs=self.sr,
            f0_ceil=kwargs.get('f0_max'),
            f0_floor=kwargs.get('f0_min'),
            frame_period=1000 * kwargs.get('hop_length', 160) / self.sr,
        )
        return pyworld.stonemask(x.astype(np.double), *f0_spectral, self.sr)

    def get_dio(self, x, *args, **kwargs):
        f0_spectral = pyworld.dio(
            x.astype(np.double),
            fs=self.sr,
            f0_ceil=kwargs.get('f0_max'),
            f0_floor=kwargs.get('f0_min'),
            frame_period=1000 * kwargs.get('hop_length', 160) / self.sr,
        )
        return pyworld.stonemask(x.astype(np.double), *f0_spectral, self.sr)

    def get_rmvpe(self, x, *args, **kwargs):
        if not hasattr(self, "model_rmvpe") or self.model_rmvpe is None:
            self.model_rmvpe = RMVPE(os.path.join(BASE_MODELS_DIR, f"rmvpe.{'onnx' if self.onnx else 'pt'}"),
                                     is_half=self.is_half, device=self.device, onnx=self.onnx)
        return self.model_rmvpe.infer_from_audio(x, thred=0.03)

    def get_pitch_dependant_rmvpe(self, x, f0_min=1, f0_max=40000, *args, **kwargs):
        if not hasattr(self, "model_rmvpe") or self.model_rmvpe is None:
            self.model_rmvpe = RMVPE(os.path.join(BASE_MODELS_DIR, f"rmvpe.{'onnx' if self.onnx else 'pt'}"),
                                     is_half=self.is_half, device=self.device, onnx=self.onnx)
        return self.model_rmvpe.infer_from_audio_with_pitch(x, thred=0.03, f0_min=f0_min, f0_max=f0_max)

    # Fork Feature: Acquire median hybrid f0 estimation calculation
    def get_f0_hybrid_computation(
            self,
            methods_list,
            merge_type,
            x,
            f0_min,
            f0_max,
            filter_radius,
            crepe_hop_length,
            time_step,
            threaded=True,
            **kwargs
    ):
        logger.info("Starting hybrid f0 computation.")

        params = {
            'x': x,
            'f0_min': f0_min,
            'f0_max': f0_max,
            'time_step': time_step,
            'filter_radius': filter_radius,
            'crepe_hop_length': crepe_hop_length,
            'model': "full"
        }

        logger.info(f"Calculating f0 pitch estimations for methods: {methods_list}")
        x = x.astype(np.float32)
        x /= np.quantile(np.abs(x), 0.999)

        def _get_f0(method, params):
            #logger.info(f"Computing f0 for method: {method}")
            if method not in self.f0_method_dict:
                raise Exception(f"Method {method} not found.")
            f0 = self.f0_method_dict[method](**params)
            self.vis.add_f0(f0, method)
            # Fix: ensure filter_radius is an odd integer before medfilt (for 'harvest' only)
            filter_radius_local = params.get("filter_radius", 3)
            if filter_radius_local > 2:
                filter_radius_local = int(filter_radius_local)
                if filter_radius_local % 2 == 0:
                    filter_radius_local += 1
                f0 = signal.medfilt(f0, filter_radius_local)
                f0 = f0[1:]  # Get rid of first frame.
                self.vis.add_f0(f0, f"{method}_filtered")

            try:
                f0_min_val = np.min(f0)
            except Exception:
                f0_min_val = "N/A"
            try:
                f0_max_val = np.max(f0)
            except Exception:
                f0_max_val = "N/A"
            if np.isnan(f0_max_val) or np.isnan(f0_min_val):
                logger.warning(f"Method {method} produced NaN values.")
                return None
            return f0

        if threaded:
            start = time.time()
            from multiprocessing.pool import ThreadPool
            with ThreadPool(max(1, len(methods_list))) as pool:
                f0_computation_stack = pool.starmap(_get_f0, [(method, params) for method in methods_list])
            logger.info(f"Completed threaded f0 computations in {time.time() - start:.2f} seconds.")
        else:
            f0_computation_stack = []
            for method in methods_list:
                output = _get_f0(method, params)
                if output is not None:
                    f0_computation_stack.append(output)
                else:
                    logger.warning(f"Method {method} produced NaN values.")

        f0_computation_stack = pad_audio(*f0_computation_stack)  # prevents uneven f0
        logger.info(f"Calculating hybrid median f0 from the stack of: {methods_list} using {merge_type} merge")
        merge_func = get_merge_func(merge_type)
        f0_median_hybrid = merge_func(f0_computation_stack, axis=0)
        self.vis.add_f0(f0_median_hybrid, "merged")
        logger.info("Completed hybrid f0 computation.")
        return f0_median_hybrid

    def get_f0(
            self,
            x,
            f0_up_key,
            f0_method,
            merge_type="median",
            filter_radius=3,
            crepe_hop_length=160,
            f0_autotune=False,
            rmvpe_onnx=False,
            inp_f0=None,
            f0_min=50,
            f0_max=1100,
            **kwargs
    ):
        time_step = self.window / self.sr * 1000
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)
        params = {
            'x': x,
            'f0_up_key': f0_up_key,
            'f0_min': f0_min,
            'f0_max': f0_max,
            'time_step': time_step,
            'filter_radius': filter_radius,
            'crepe_hop_length': crepe_hop_length,
            'model': "full",
            'onnx': rmvpe_onnx
        }

        if hasattr(f0_method, "pop") and len(f0_method) == 1:
            f0_method = f0_method.pop()

        if f0_method == "hybrid":
            methods_list = ["harvest", "rmvpe+", "crepe", "rmvpe"]
            f0 = self.get_f0_hybrid_computation(methods_list, merge_type, **params)
        elif isinstance(f0_method, list):
            f0 = self.get_f0_hybrid_computation(f0_method, merge_type, **params)
        else:
            f0 = self.f0_method_dict[f0_method](**params)

        if f0_autotune:
            f0 = autotune_f0(f0)

        f0 = remap_f0(f0)
        self.vis.add_f0(f0, "remapped")

        f0 *= pow(2, f0_up_key / 12)
        tf0 = self.sr // self.window
        if inp_f0 is not None:
            delta_t = np.round((inp_f0[:, 0].max() - inp_f0[:, 0].min()) * tf0 + 1).astype("int16")
            replace_f0 = np.interp(
                list(range(delta_t)),
                inp_f0[:, 0] * 100,
                inp_f0[:, 1]
            )
            shape = f0[self.x_pad * tf0: self.x_pad * tf0 + len(replace_f0)].shape[0]
            f0[self.x_pad * tf0: self.x_pad * tf0 + len(replace_f0)] = replace_f0[:shape]

        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(int)
        gc_collect()

        return f0_coarse, f0
