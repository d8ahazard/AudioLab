import json
import logging
from multiprocessing import cpu_count
from pathlib import Path
import torch

logger = logging.getLogger(__name__)

# Configuration files with their paths
version_config_paths = [
    Path(__file__).parent / "v1" / "32k.json",
    Path(__file__).parent / "v1" / "40k.json",
    Path(__file__).parent / "v1" / "48k.json",
    Path(__file__).parent / "v2" / "48k.json",
    Path(__file__).parent / "v2" / "40k.json",
    Path(__file__).parent / "v2" / "32k.json",
    Path(__file__).parent / "v3" / "40k.json",
]


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


@singleton
class Config:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.is_half = self.device != "cpu"
        self.use_jit = False
        self.n_cpu = cpu_count() if not torch.cuda.is_available() else 0
        self.gpu_name = torch.cuda.get_device_name(int(self.device.split(":")[-1])) if self.device.startswith(
            "cuda") else None
        self.json_config = self.load_config_json()
        self.gpu_mem = None
        self.instead = ""
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def load_config_json(self) -> dict:
        configs = {}
        for config_path in version_config_paths:
            with config_path.open() as f:
                configs[f"{config_path.parent.name}/{config_path.name}"] = json.load(f)
        return configs

    def has_mps(self) -> bool:
        # Check if Metal Performance Shaders are available - for macOS 12.3+.
        return torch.backends.mps.is_available()

    def has_xpu(self) -> bool:
        # Check if XPU is available.
        return hasattr(torch, "xpu") and torch.xpu.is_available()

    def use_fp32_config(self):
        preprocess_path = Path(__file__).parent.parent / "infer/modules/train/preprocess.py"
        for config_path in version_config_paths:
            config = json.loads(config_path.read_text())
            config["train"]["fp16_run"] = False
            config_path.write_text(json.dumps(config))

            if preprocess_path.exists():
                preprocess_content = preprocess_path.read_text().replace("3.7", "3.0")
                preprocess_path.write_text(preprocess_content)
        logger.info("Overwritten preprocess and config.json to use FP32.")

    def device_config(self) -> tuple:
        if self.device.startswith("cuda"):
            self.set_cuda_config()
        elif self.has_mps():
            self.device = "mps"
            self.is_half = False
            self.use_fp32_config()
        else:
            self.device = "cpu"
            self.is_half = False
            self.use_fp32_config()

        # Configuration for 6GB GPU memory
        x_pad, x_query, x_center, x_max = (3, 10, 60, 65) if self.is_half else (1, 6, 38, 41)
        if self.gpu_mem is not None and self.gpu_mem <= 4:
            # Configuration for 5GB GPU memory
            x_pad, x_query, x_center, x_max = (1, 5, 30, 32)

        logger.info(f"is_half:{self.is_half}, device:{self.device}")
        return x_pad, x_query, x_center, x_max

    def set_cuda_config(self):
        i_device = int(self.device.split(":")[-1])
        self.gpu_name = torch.cuda.get_device_name(i_device)
        low_end_gpus = ["16", "P40", "P10", "1060", "1070", "1080"]
        if any(gpu in self.gpu_name for gpu in low_end_gpus) and "V100" not in self.gpu_name.upper():
            logger.info(f"Found GPU {self.gpu_name}, force to FP32")
            self.is_half = False
            self.use_fp32_config()
        else:
            logger.info(f"Found GPU {self.gpu_name}")

        self.gpu_mem = torch.cuda.get_device_properties(i_device).total_memory // (1024 ** 3)


if __name__ == "__main__":
    config = Config()
    # Now you can access the configuration details through the 'config' instance
    print(config.device, config.is_half)
