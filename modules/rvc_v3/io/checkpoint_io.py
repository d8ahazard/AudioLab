"""
Safetensors-based load/save for RVC V3 inference and training checkpoints.

Replaces pickle-based .pth/.pt with .safetensors + JSON metadata.
Loaders support both formats for backward compatibility.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file

logger = logging.getLogger(__name__)


def _resolve_path_stem(path: str | Path) -> Path:
    """Resolve path to directory and stem (no extension)."""
    p = Path(path).resolve()
    return p.parent / p.stem


def _flatten_state_dict(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    """Flatten state dict keys with prefix for safetensors."""
    return {f"{prefix}{k}": v for k, v in state_dict.items()}


def _flatten_optimizer_state_dict(
    optim_state: Dict[str, Any], prefix: str
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Extract tensors from optimizer state_dict for safetensors.
    Returns (tensors_dict, param_groups_meta).
    param_groups_meta has params replaced by indices [0,1,2,...].
    """
    tensors = {}
    state = optim_state.get("state", {})
    param_groups = optim_state.get("param_groups", [])

    # Build id -> index mapping from param_groups
    param_ids = []
    for group in param_groups:
        param_ids.extend(group.get("params", []))
    id_to_idx = {pid: i for i, pid in enumerate(param_ids)}

    for param_id, param_state in state.items():
        idx = id_to_idx.get(param_id)
        if idx is None:
            continue
        for key, val in param_state.items():
            if isinstance(val, torch.Tensor):
                tensors[f"{prefix}state.{idx}.{key}"] = val

    # Meta: param_groups with params as indices
    meta_groups = []
    for group in param_groups:
        g = {k: v for k, v in group.items() if k != "params"}
        g["params"] = [id_to_idx.get(pid, i) for i, pid in enumerate(group.get("params", []))]
        meta_groups.append(g)
    param_groups_meta = {"param_groups": meta_groups}
    return tensors, param_groups_meta


def _unflatten_optimizer_state_dict(
    flat: Dict[str, torch.Tensor],
    prefix: str,
    optim: torch.optim.Optimizer,
) -> Dict[str, Any]:
    """Rebuild optimizer state_dict from flat tensors using current optimizer's param ids."""
    # Get current param ids in order
    param_ids = []
    for group in optim.param_groups:
        param_ids.extend(group["params"])
    # Collect tensors by index
    state_by_idx: Dict[int, Dict[str, torch.Tensor]] = {}
    pref = f"{prefix}state."
    for k, v in flat.items():
        if not k.startswith(pref):
            continue
        rest = k[len(pref):]
        parts = rest.split(".", 1)
        if len(parts) != 2:
            continue
        idx = int(parts[0])
        key = parts[1]
        if idx not in state_by_idx:
            state_by_idx[idx] = {}
        state_by_idx[idx][key] = v

    # Build state with current param ids
    state = {}
    for i, param_id in enumerate(param_ids):
        if i in state_by_idx:
            state[param_id] = state_by_idx[i]

    # Build param_groups with current params
    param_groups = []
    for group in optim.param_groups:
        g = {k: v for k, v in group.items()}
        param_groups.append(g)

    return {"state": state, "param_groups": param_groups}


def _unflatten_state_dict(flat: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    """Extract state dict for a given prefix."""
    pref = f"{prefix}"
    return {k[len(pref):]: v for k, v in flat.items() if k.startswith(pref)}


def save_inference_model(
    path_base: str | Path,
    generator: torch.nn.Module,
    text_encoder: torch.nn.Module,
    config: Dict[str, Any],
    version: str = "v3",
    iteration: int = 0,
    lr: float = 0.0001,
) -> Tuple[str, str]:
    """
    Save inference-ready model to .safetensors and _config.json.

    Args:
        path_base: Base path (stem) or full path; .safetensors and _config.json appended.
        generator: Generator module.
        text_encoder: Text encoder module.
        config: Config dict (e.g. from RVCV3Config.to_dict()).
        version: Model version string.
        iteration: Training iteration / global step.
        lr: Learning rate.

    Returns:
        Tuple of (safetensors_path, config_path).
    """
    path_base = Path(path_base)
    if path_base.suffix:
        path_base = path_base.with_suffix("")
    base_dir = path_base.parent
    stem = path_base.name
    base_dir.mkdir(parents=True, exist_ok=True)

    tensors = {}
    tensors.update(_flatten_state_dict(generator.state_dict(), "generator."))
    tensors.update(_flatten_state_dict(text_encoder.state_dict(), "text_encoder."))

    safe_path = base_dir / f"{stem}.safetensors"
    config_path = base_dir / f"{stem}_config.json"

    safe_save_file(tensors, safe_path)
    meta = {
        "config": config,
        "version": version,
        "iteration": iteration,
        "learning_rate": lr,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info("Saved inference model: %s (%.2f MB)", safe_path, safe_path.stat().st_size / (1024 * 1024))
    return str(safe_path), str(config_path)


def save_inference_model_from_state_dicts(
    path_base: str | Path,
    generator_state: Dict[str, torch.Tensor],
    text_encoder_state: Dict[str, torch.Tensor],
    config: Dict[str, Any],
    version: str = "v3",
    iteration: int = 0,
    lr: float = 0.0001,
) -> Tuple[str, str]:
    """Save inference model from state dicts (for expand_weights etc)."""
    path_base = Path(path_base)
    if path_base.suffix:
        path_base = path_base.with_suffix("")
    base_dir = path_base.parent
    stem = path_base.name
    base_dir.mkdir(parents=True, exist_ok=True)

    tensors = {}
    tensors.update(_flatten_state_dict(generator_state, "generator."))
    tensors.update(_flatten_state_dict(text_encoder_state, "text_encoder."))

    safe_path = base_dir / f"{stem}.safetensors"
    config_path = base_dir / f"{stem}_config.json"

    safe_save_file(tensors, safe_path)
    meta = {"config": config, "version": version, "iteration": iteration, "learning_rate": lr}
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info("Saved inference model: %s (%.2f MB)", safe_path, safe_path.stat().st_size / (1024 * 1024))
    return str(safe_path), str(config_path)


def load_inference_model(
    path: str | Path,
    device: str | torch.device = "cpu",
    require_config: bool = True,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Load inference model from .safetensors or .pth.

    Tries .safetensors first, then .pth. Accepts path as stem, .safetensors, or .pth.

    Returns:
        (generator_state_dict, text_encoder_state_dict, meta_dict).
        meta_dict contains config, version, iteration, learning_rate.
    """
    p = Path(path).resolve()
    base_dir = p.parent
    stem = p.stem if p.suffix else p.name
    if not p.suffix:
        stem = p.name

    # Try safetensors first
    safe_path = base_dir / f"{stem}.safetensors"
    config_path = base_dir / f"{stem}_config.json"
    if safe_path.exists():
        tensors = safe_load_file(str(safe_path), device=str(device))
        generator_sd = _unflatten_state_dict(tensors, "generator.")
        text_encoder_sd = _unflatten_state_dict(tensors, "text_encoder.")
        meta = {}
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        else:
            if require_config:
                raise FileNotFoundError(
                    f"Missing config metadata for safetensors checkpoint: {config_path}"
                )
            meta = {"config": {}}
        if require_config and not isinstance(meta.get("config"), dict):
            raise ValueError(f"Invalid config metadata format in {config_path}")
        return generator_sd, text_encoder_sd, meta

    # Fallback to .pth
    pth_path = base_dir / f"{stem}.pth"
    if pth_path.exists():
        ckpt = torch.load(pth_path, map_location=device, weights_only=True)
        gen_sd = ckpt.get("generator", ckpt.get("model", {}))
        text_sd = ckpt.get("text_encoder", {})
        meta = {
            "config": ckpt.get("config", {}),
            "version": ckpt.get("version", "v3"),
            "iteration": ckpt.get("iteration", 0),
            "learning_rate": ckpt.get("learning_rate", 0.0001),
        }
        return gen_sd, text_sd, meta

    raise FileNotFoundError(f"No checkpoint found at {path} (tried {safe_path} and {pth_path})")


def save_training_checkpoint(
    path_base: str | Path,
    epoch: int,
    global_step: int,
    generator: torch.nn.Module,
    text_encoder: torch.nn.Module,
    discriminator: torch.nn.Module,
    optim_g: torch.optim.Optimizer,
    optim_d: torch.optim.Optimizer,
    config: Dict[str, Any],
) -> Tuple[str, str]:
    """
    Save training checkpoint to .safetensors and _meta.json.

    Returns:
        Tuple of (safetensors_path, meta_path).
    """
    path_base = Path(path_base)
    if path_base.suffix in (".safetensors", ".pt", ".pth"):
        path_base = path_base.with_suffix("")
    base_dir = path_base.parent
    stem = path_base.name
    base_dir.mkdir(parents=True, exist_ok=True)

    tensors = {}
    tensors.update(_flatten_state_dict(generator.state_dict(), "generator."))
    tensors.update(_flatten_state_dict(text_encoder.state_dict(), "text_encoder."))
    tensors.update(_flatten_state_dict(discriminator.state_dict(), "discriminator."))

    opt_g_tensors, opt_g_meta = _flatten_optimizer_state_dict(optim_g.state_dict(), "optim_g.")
    opt_d_tensors, opt_d_meta = _flatten_optimizer_state_dict(optim_d.state_dict(), "optim_d.")
    tensors.update(opt_g_tensors)
    tensors.update(opt_d_tensors)

    safe_path = base_dir / f"{stem}.safetensors"
    meta_path = base_dir / f"{stem}_meta.json"
    tmp_safe = base_dir / f"{stem}.safetensors.tmp"
    tmp_meta = base_dir / f"{stem}_meta.json.tmp"

    safe_save_file(tensors, tmp_safe)
    meta = {
        "epoch": epoch,
        "global_step": global_step,
        "config": config,
        "optim_g_param_groups": opt_g_meta.get("param_groups", []),
        "optim_d_param_groups": opt_d_meta.get("param_groups", []),
    }
    with open(tmp_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp_safe, safe_path)
    os.replace(tmp_meta, meta_path)

    logger.debug("Saved training checkpoint: %s", safe_path)
    return str(safe_path), str(meta_path)


def load_training_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
    optim_g: torch.optim.Optimizer | None = None,
    optim_d: torch.optim.Optimizer | None = None,
) -> Dict[str, Any]:
    """
    Load training checkpoint from .safetensors + _meta.json or .pt.

    When loading from safetensors, optim_g and optim_d must be provided to rebuild
    optimizer state dicts (param ids change between runs).

    Returns:
        Dict with epoch, global_step, generator, text_encoder, discriminator, optim_g, optim_d, config.
        State dicts are ready for load_state_dict().
    """
    p = Path(path).resolve()
    base_dir = p.parent
    stem = p.stem if p.suffix else p.name
    if p.suffix in (".safetensors", ".pt"):
        stem = Path(path).stem

    safe_path = base_dir / f"{stem}.safetensors"
    meta_path = base_dir / f"{stem}_meta.json"
    pt_path = base_dir / f"{stem}.pt"

    if safe_path.exists() and meta_path.exists():
        if optim_g is None or optim_d is None:
            raise ValueError("optim_g and optim_d required when loading from safetensors")
        tensors = safe_load_file(str(safe_path), device=str(device))
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        optim_g_sd = _unflatten_optimizer_state_dict(tensors, "optim_g.", optim_g)
        optim_d_sd = _unflatten_optimizer_state_dict(tensors, "optim_d.", optim_d)
        return {
            "epoch": meta["epoch"],
            "global_step": meta["global_step"],
            "generator": _unflatten_state_dict(tensors, "generator."),
            "text_encoder": _unflatten_state_dict(tensors, "text_encoder."),
            "discriminator": _unflatten_state_dict(tensors, "discriminator."),
            "optim_g": optim_g_sd,
            "optim_d": optim_d_sd,
            "config": meta.get("config", {}),
        }

    if pt_path.exists():
        ckpt = torch.load(pt_path, map_location=device, weights_only=True)
        return ckpt

    raise FileNotFoundError(f"No checkpoint found at {path} (tried {safe_path}+{meta_path} and {pt_path})")


def save_discriminator_checkpoint(
    path_base: str | Path,
    discriminator_state: Dict[str, torch.Tensor],
    iteration: int = 0,
    lr: float = 0.0001,
    version: str = "v3",
) -> Tuple[str, str]:
    """Save discriminator checkpoint (for expand_weights)."""
    path_base = Path(path_base)
    if path_base.suffix:
        path_base = path_base.with_suffix("")
    base_dir = path_base.parent
    stem = path_base.name
    base_dir.mkdir(parents=True, exist_ok=True)

    tensors = _flatten_state_dict(discriminator_state, "model.")
    safe_path = base_dir / f"{stem}.safetensors"
    config_path = base_dir / f"{stem}_config.json"

    safe_save_file(tensors, safe_path)
    meta = {"iteration": iteration, "learning_rate": lr, "version": version}
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return str(safe_path), str(config_path)


def load_pretrained_d(path: str | Path, device: str | torch.device = "cpu") -> Dict[str, torch.Tensor]:
    """
    Load discriminator checkpoint from .safetensors or .pth (for pretrained_v3 D files).

    Returns:
        Discriminator state dict.
    """
    p = Path(path).resolve()
    base_dir = p.parent
    stem = p.stem if p.suffix else p.name

    safe_path = base_dir / f"{stem}.safetensors"
    config_path = base_dir / f"{stem}_config.json"
    pth_path = base_dir / f"{stem}.pth"

    if safe_path.exists():
        tensors = safe_load_file(str(safe_path), device=str(device))
        return _unflatten_state_dict(tensors, "model.")

    if pth_path.exists():
        ckpt = torch.load(pth_path, map_location=device, weights_only=True)
        return ckpt.get("model", ckpt)

    raise FileNotFoundError(f"No discriminator checkpoint at {path}")
