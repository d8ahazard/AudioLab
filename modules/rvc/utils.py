import gc
import logging
import multiprocessing
import sys

import numpy as np
import psutil
import torch

MATPLOTLIB_FLAG = False

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging


class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


def gc_collect():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.set_threshold(100, 10, 1)
    gc.collect()


def get_optimal_torch_device(index=0) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(
            f"cuda:{index % torch.cuda.device_count()}"
        )  # Very fast
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_optimal_threads(offset=0):
    cores = multiprocessing.cpu_count() - offset
    return int(max(np.floor(cores * (1 - psutil.cpu_percent())), 1))


def get_merge_func(merge_type: str):
    if merge_type == "min":
        return np.min
    elif merge_type == "max":
        return np.max
    elif merge_type == "median":
        return np.median
    else:
        return np.mean
