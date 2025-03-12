from exllamav2 import (
    ExLlamaV2Cache,
    ExLlamaV2Cache_Q4,
    ExLlamaV2Cache_Q6,
    ExLlamaV2Cache_Q8,
)

import torch
import random
import numpy as np

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_cache_class(cache_mode: str):
    match cache_mode:
        case "Q4":
            return ExLlamaV2Cache_Q4
        case "Q6":
            return ExLlamaV2Cache_Q6
        case "Q8":
            return ExLlamaV2Cache_Q8
        case _:
            return ExLlamaV2Cache
