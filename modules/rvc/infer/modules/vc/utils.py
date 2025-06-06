import os
from functools import lru_cache
import inspect
import ast

import librosa
import pyworld
import torch
from torch.nn import functional as F
from omegaconf import OmegaConf
from fairseq.data import Dictionary
from fairseq import tasks
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.distributed.fully_sharded_data_parallel import FSDP, has_FSDP

from handlers.config import model_path

# Add safe globals for torch.load
torch.serialization.add_safe_globals([Dictionary])

def load_checkpoint_to_cpu(path, arg_overrides=None):
    """Loads a checkpoint to CPU with proper handling of PyTorch 2.6+ changes."""
    with open(path, "rb") as f:
        state = torch.load(f, map_location=torch.device("cpu"), weights_only=False)
    
    if "args" in state and state["args"] is not None and arg_overrides is not None:
        args = state["args"]
        for arg_name, arg_val in arg_overrides.items():
            setattr(args, arg_name, arg_val)
            
    return state

def load_model_ensemble_and_task(
    filenames,
    arg_overrides=None,
    task=None,
    strict=True,
    suffix="",
    num_shards=1,
    state=None,
):
    """Modified version of fairseq's load_model_ensemble_and_task that handles PyTorch 2.6+ changes."""
    assert state is None or len(filenames) == 1
    assert not (strict and num_shards > 1), "Cannot load state dict with strict=True and checkpoint shards > 1"
    
    ensemble = []
    cfg = None
    for filename in filenames:
        orig_filename = filename
        model_shard_state = {"shard_weights": [], "shard_metadata": []}
        assert num_shards > 0
        
        for shard_idx in range(num_shards):
            filename = get_maybe_sharded_checkpoint_filename(
                orig_filename, suffix, shard_idx, num_shards
            )
            
            if not os.path.exists(filename):
                raise IOError(f"Model file not found: {filename}")
                
            if state is None:
                state = load_checkpoint_to_cpu(filename, arg_overrides)
                
            if "args" in state and state["args"] is not None:
                cfg = convert_namespace_to_omegaconf(state["args"])
            elif "cfg" in state and state["cfg"] is not None:
                cfg = state["cfg"]
            else:
                raise RuntimeError(f"Neither args nor cfg exist in state keys = {state.keys()}")

            if task is None:
                task = tasks.setup_task(cfg.task, from_checkpoint=True)

            if "task_state" in state:
                task.load_state_dict(state["task_state"])

            argspec = inspect.getfullargspec(task.build_model)

            if "fsdp_metadata" in state and num_shards > 1:
                model_shard_state["shard_weights"].append(state["model"])
                model_shard_state["shard_metadata"].append(state["fsdp_metadata"])
                # check FSDP import before the code goes too far
                if not has_FSDP:
                    raise ImportError(
                        "Cannot find FullyShardedDataParallel. "
                        "Please install fairscale with: pip install fairscale"
                    )
                if shard_idx == num_shards - 1:
                    consolidated_model_state = FSDP.consolidate_shard_weights(
                        shard_weights=model_shard_state["shard_weights"],
                        shard_metadata=model_shard_state["shard_metadata"],
                    )
                    if "from_checkpoint" in argspec.args:
                        model = task.build_model(cfg.model, from_checkpoint=True)
                    else:
                        model = task.build_model(cfg.model)
                    if (
                        "optimizer_history" in state
                        and len(state["optimizer_history"]) > 0
                        and "num_updates" in state["optimizer_history"][-1]
                    ):
                        model.set_num_updates(
                            state["optimizer_history"][-1]["num_updates"]
                        )
                    model.load_state_dict(
                        consolidated_model_state, strict=strict, model_cfg=cfg.model
                    )
            else:
                # model parallel checkpoint or unsharded checkpoint
                # support old external tasks

                if "from_checkpoint" in argspec.args:
                    model = task.build_model(cfg.model, from_checkpoint=True)
                else:
                    model = task.build_model(cfg.model)
                if (
                    "optimizer_history" in state
                    and len(state["optimizer_history"]) > 0
                    and "num_updates" in state["optimizer_history"][-1]
                ):
                    model.set_num_updates(state["optimizer_history"][-1]["num_updates"])
                model.load_state_dict(
                    state["model"], strict=strict, model_cfg=cfg.model
                )

            # reset state so it gets loaded for the next model in ensemble
            state = None

        # build model for ensemble
        ensemble.append(model)
    return ensemble, cfg, task

def get_maybe_sharded_checkpoint_filename(
    filename: str, suffix: str, shard_idx: int, num_shards: int
) -> str:
    orig_filename = filename
    filename = filename.replace(".pt", suffix + ".pt")
    fsdp_filename = filename[:-3] + f"-shard{shard_idx}.pt"
    model_parallel_filename = orig_filename[:-3] + f"_part{shard_idx}.pt"
    if os.path.exists(fsdp_filename):
        return fsdp_filename
    elif num_shards > 1:
        return model_parallel_filename
    else:
        return filename

def get_index_path_from_model(sid):
    sid_base = os.path.basename(sid)
    sid_base = sid_base.replace(".pth", "")
    if "final" in sid_base:
        sid_base = sid_base.replace("_final", "")
    # Find file that contains sid_base and .index
    for file in os.listdir(os.path.join(model_path, "trained")):
        if sid_base in file and ".index" in file:
            return file
    return None


def load_hubert(config):
    hubert_base = os.path.join(model_path, "rvc", "hubert_base.pt")
    if not os.path.exists(hubert_base):
        raise FileNotFoundError(f"Hubert model not found at {hubert_base}")
    models, _, _ = load_model_ensemble_and_task(
        [hubert_base],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    return hubert_model.eval()


def check_faiss_index_file(file_path):
    """
    Checks whether the given file appears to be a raw FAISS index file or a ZIP archive.

    Parameters:
        file_path (str): Path to the file to check.

    Returns:
        bool: True if the file appears to be a valid FAISS index (non-ZIP), False otherwise.
    """
    import os

    if not os.path.exists(file_path):
        print(f"File does not exist: {file_path}")
        return False

    with open(file_path, "rb") as f:
        header = f.read(4)
    if header == b"PK\x03\x04":
        print("The file appears to be a ZIP archive. Please unzip it to obtain the FAISS index file.")
        return False
    else:
        print("The file appears to be a valid FAISS index file (based on header check).")
        return True


def extract_index_from_zip(zip_path, extract_to):
    """
    Extracts files from a ZIP archive and returns the path to the first extracted file.

    Parameters:
        zip_path (str): Path to the ZIP archive.
        extract_to (str): Directory where the contents should be extracted.

    Returns:
        str: Path to the extracted index file.

    Raises:
        ValueError: If the file is not a valid ZIP archive or if extraction yields no files.
    """
    import os
    import zipfile

    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"{zip_path} is not a valid ZIP file.")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    extracted_files = [f for f in os.listdir(extract_to) if os.path.isfile(os.path.join(extract_to, f))]
    if not extracted_files:
        raise ValueError("No files were extracted from the ZIP archive.")

    # Assumes that the first extracted file is the FAISS index file
    extracted_file_path = os.path.join(extract_to, extracted_files[0])
    print(f"Extracted file: {extracted_file_path}")
    return extracted_file_path


@lru_cache
def cache_harvest_f0(input_audio_path, fs, f0max, f0min, frame_period):
    from modules.rvc.infer.modules.vc.pipeline import input_audio_path2wav

    audio = input_audio_path2wav[input_audio_path]
    f0, t = pyworld.harvest(
        audio,
        fs=fs,
        f0_ceil=f0max,
        f0_floor=f0min,
        frame_period=frame_period,
    )
    f0 = pyworld.stonemask(audio, f0, t, fs)
    return f0


def change_rms(data1, sr1, data2, sr2, rate):
    rms1 = librosa.feature.rms(
        y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2
    )  # 每半秒一个点
    rms2 = librosa.feature.rms(y=data2, frame_length=sr2 // 2 * 2, hop_length=sr2 // 2)
    rms1 = torch.from_numpy(rms1)
    rms1 = F.interpolate(rms1.unsqueeze(0), size=data2.shape[0], mode="linear").squeeze()
    rms2 = torch.from_numpy(rms2)
    rms2 = F.interpolate(rms2.unsqueeze(0), size=data2.shape[0], mode="linear").squeeze()
    rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-6)
    data2 *= (
            torch.pow(rms1, torch.tensor(1 - rate))
            * torch.pow(rms2, torch.tensor(rate - 1))
    ).numpy()
    return data2


