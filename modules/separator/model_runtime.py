"""AudioLab's instance-local adapter for audio-separator 0.47.0."""
import gc
import copy
import logging
import os
import re
import tempfile
import time
from pathlib import Path

import librosa
import numpy as np
import requests
import soundfile as sf
import torch
from audio_separator.separator import Separator

from modules.separator.audio_quality import stereo
from modules.separator.stem_manifest import file_hash

logger = logging.getLogger(__name__)
FUSED_DEREVERB = "dereverb_echo_mbr_fused_0.5_v2_0.25_big_0.25_super.ckpt"
CUSTOM_MODELS = {
    FUSED_DEREVERB: {
        "repo": "Sucial/Dereverb-Echo_Mel_Band_Roformer", "revision": "9c83bf27196213e6107cf0de5ff2d06d56a46876",
        "checkpoint": FUSED_DEREVERB, "config": "config_dereverb_echo_mbr_v2.yaml", "license": "CC-BY-NC-SA-4.0",
        "checkpoint_sha256": "1596b1063238f487d54a0510a8c92cb28c000c803a271dd618ac49efc99ef3f7",
        "config_sha256": "fb26df515b5acc81aa0a281920decd849588d8d09171e9c20fb8cbd5d2efc120",
    },
    "melband_roformer_big_beta7.ckpt": {
        "repo": "pcunwa/Mel-Band-Roformer-big", "revision": "1508d1ed7c54cb0017b2cbfaabdaf3ca87d2cf74",
        "checkpoint": "big_beta7.ckpt", "config": "big_beta7.yaml", "license": "unspecified",
        "checkpoint_sha256": "9d68b9a8689a3500c45d3418a7811934e557760db07285f483088a0965b0eb88",
        "config_sha256": "a2ad8056dfdb394422d62dd823f7a02b7f17f6be12870da913bd84b962f2e7c9",
    },
    "melband_roformer_becruily_deux.ckpt": {
        "repo": "becruily/mel-band-roformer-deux", "revision": "2da74427d682a3df47a774378fc24d7a1a0cdaad",
        "checkpoint": "becruily_deux.ckpt", "config": "config_deux_becruily.yaml", "license": "CC-BY-NC-4.0",
        "checkpoint_sha256": "10255c02295bf3e3865d4ee50ff752d7b19b124ed5fd93b147babc4333eda3aa",
        "config_sha256": "bb3ea9bce37ca96d63568490d5a92d7e41df3d7726788b6970c10d89eb62d902",
    },
}
BG_MODELS = {
    "bve": "UVR-BVE-4B_SN-44100-1.pth",
    "bve_v2": "UVR-BVE-4B_SN-44100-2.pth",
    "karaoke": "bs_roformer_karaoke_frazer_becruily.ckpt",
}
INSTRUMENT_MODELS = {"demucs": "htdemucs_6s.yaml", "roformer_sw": "BS-Roformer-SW.ckpt"}


def download_verified(url, target, digest):
    target = Path(target)
    if target.exists():
        if file_hash(target) != digest:
            raise ValueError(f"Checksum mismatch: {target}; move the corrupt file before retrying")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=target.parent, suffix=".download")
    try:
        with os.fdopen(fd, "wb") as out, requests.get(url, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            for block in response.iter_content(1024 * 1024):
                out.write(block)
        if file_hash(name) != digest:
            raise ValueError(f"Downloaded checksum mismatch: {target.name}")
        os.replace(name, target)
    finally:
        if os.path.exists(name):
            os.remove(name)


class AudioLabSeparator(Separator):
    def __init__(self, *args, cpu=False, quality="balanced", preserve_gain=True, **kwargs):
        if quality not in {"fast", "balanced", "maximum"}:
            raise ValueError(f"Unknown separation quality: {quality}")
        self.force_cpu = cpu
        self.quality = quality
        self.preserve_gain = preserve_gain
        self.run_records = []
        self.loaded_models = set()
        super().__init__(*args, **kwargs)
        self._base_arch_params = copy.deepcopy(self.arch_specific_params)

    def setup_torch_device(self, system_info):
        if not self.force_cpu:
            return super().setup_torch_device(system_info)
        self.torch_device_cpu = torch.device("cpu")
        self.torch_device = self.torch_device_cpu
        self.onnx_execution_provider = ["CPUExecutionProvider"]

    def configure_cuda(self, ort_providers):
        super().configure_cuda(ort_providers)
        if "CUDAExecutionProvider" in ort_providers:
            # Avoid expensive cuDNN exhaustive searches on every model session.
            self.onnx_execution_provider = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"})]

    def download_model_files(self, model_filename):
        spec = CUSTOM_MODELS.get(model_filename)
        if spec is None:
            return super().download_model_files(model_filename)
        for kind, filename in (("checkpoint", model_filename), ("config", spec["config"])):
            url = f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/{spec[kind]}"
            download_verified(url, Path(self.model_file_dir) / filename, spec[kind + "_sha256"])
        self.model_is_uvr_vip = False
        self.model_friendly_name = model_filename
        return model_filename, "MDXC", model_filename, str(Path(self.model_file_dir) / model_filename), spec["config"]

    def load_model(self, model_filename="model_bs_roformer_ep_317_sdr_12.9755.ckpt", force_reload=False, **settings):
        # Fresh settings each time: no overlap/segment leakage between models.
        self.arch_specific_params = copy.deepcopy(self._base_arch_params)
        for arch, key in (("MDX", "mdx_params"), ("VR", "vr_params"), ("Demucs", "demucs_params")):
            self.arch_specific_params[arch].update(settings.get(key, {}))
        self.arch_specific_params["MDXC"] = {
            "segment_size": 256, "override_model_segment_size": False,
            "batch_size": 1, "overlap": 2 if self.quality == "fast" else None, "pitch_shift": 0,
            **settings.get("mdxc_params", {}),
        }
        if getattr(self, "model_instance", None) is not None:
            self.model_instance = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        super().load_model(model_filename, force_reload=True)
        self.current_model = model_filename
        self.loaded_models.add(model_filename)
        model = self.model_instance
        if type(model).__name__ == "MDXSeparator" and model.invert_using_spec:
            # 0.47.0 passes a time-major primary to channel-major invert_stem.
            # Use its supported waveform subtraction path until upstream fixes it.
            model.invert_using_spec = False
            logger.warning("Using waveform inversion for %s (0.47.0 MDX spectral-inversion shape defect)", model_filename)
        if self.quality == "maximum" and hasattr(model, "override_model_segment_size"):
            model.overlap = max(8, model.overlap)
        if self.preserve_gain:
            # Float WAV supports peaks > 1; avoid independent stem gain changes.
            model.normalization_threshold = float("inf")
            model.amplification_threshold = 0.0
        logger.info("Loaded %s: device=%s precision=%s overlap=%s batch=%s", model_filename,
                    model.torch_device, self.effective_precision, getattr(model, "overlap", None), getattr(model, "batch_size", None))

    def separate(self, *args, **kwargs):
        start = time.perf_counter()
        use_cuda = getattr(self.model_instance.torch_device, "type", "cpu") == "cuda"
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()
        record = {"model": self.current_model, "quality": self.quality, "precision": self.effective_precision,
                  "overlap": getattr(self.model_instance, "overlap", None), "status": "failed"}
        try:
            result = super().separate(*args, **kwargs)
            record["status"] = "ok"
            return result
        except torch.cuda.OutOfMemoryError as exc:
            # Batch is already 1; never silently reduce author chunk size or quality.
            record["error"] = "CUDA out of memory; retry on CPU or choose Fast explicitly"
            raise RuntimeError(record["error"]) from exc
        finally:
            record["seconds"] = time.perf_counter() - start
            record["peak_vram_bytes"] = torch.cuda.max_memory_allocated() if use_cuda else 0
            self.run_records.append(record)


ALIASES = {
    "vocals": "vocals", "vocal": "vocals", "lead vocals": "vocals",
    "instrumental": "instrumental", "accompaniment": "instrumental", "no vocals": "instrumental",
    "drums": "drums", "bass": "bass", "guitar": "guitar", "piano": "piano", "other": "other",
    "kick": "drums_kick", "snare": "drums_snare", "toms": "drums_toms",
    "hh": "drums_hh", "hi hat": "drums_hh", "ride": "drums_ride", "crash": "drums_crash",
    "cymbals": "drums_cymbals", "woodwinds": "woodwinds", "wind inst": "woodwinds",
    "no reverb": "dry", "noreverb": "dry", "dry": "dry", "no echo": "dry", "noecho": "dry",
    "no noise": "clean", "nonoise": "clean", "no crowd": "clean", "nocrowd": "clean",
}


def read_outputs(paths, folder, sr, length, task="core"):
    stems = {}
    for file in paths:
        path = Path(file)
        if not path.is_absolute():
            path = Path(folder) / path
        labels = re.findall(r"\(([^()]*)\)", path.stem)
        role = next((ALIASES.get(label.lower().replace("_", " ").strip()) for label in labels
                     if label.lower().replace("_", " ").strip() in ALIASES), None)
        if role is None:
            continue
        if task == "core" and role == "other":
            role = "instrumental"
        elif task == "drums" and role == "other":
            role = "drums_other"
        elif task == "bve":
            role = {"vocals": "bg_vocals", "instrumental": "vocals"}.get(role, role)
        elif task == "karaoke":
            role = {"instrumental": "bg_vocals"}.get(role, role)
        audio, _ = librosa.load(str(path), sr=sr, mono=False)
        stems[role] = stereo(audio, length)
    return stems
