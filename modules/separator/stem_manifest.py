"""Recoverable Smart Stems manifests. Hidden files never enter normal outputs."""
import hashlib
import json
import os
import threading
from importlib import resources
from pathlib import Path

PIPELINE_REVISION = "selected-consensus-mega-extras-6"
MANIFEST_NAME = "separation_manifest.json"
_LOCK = threading.RLock()
_HASH_CACHE = {}


def file_hash(path):
    path = Path(path)
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if key not in _HASH_CACHE:
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                h.update(block)
        _HASH_CACHE[key] = h.hexdigest()
    return _HASH_CACHE[key]


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with _LOCK:
        temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(temp, path)


def read_manifest(folder):
    path = Path(folder) / MANIFEST_NAME
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"stems": []}


def safe_path(folder, relative):
    root = Path(folder).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Stem path escapes its project")
    return path


def hidden_stems(folder):
    return [s for s in read_manifest(folder)["stems"] if s.get("hidden")]


def restore_stem(folder, relative):
    with _LOCK:
        manifest = read_manifest(folder)
        item = next((s for s in manifest["stems"] if s["path"] == relative and s.get("hidden")), None)
        if item is None:
            raise ValueError("No matching hidden stem")
        source = safe_path(folder, item["path"])
        target = safe_path(folder, item["filename"])
        if target.exists():
            raise FileExistsError(target)
        if file_hash(source) != item["sha256"]:
            raise ValueError("Hidden stem checksum changed")
        os.replace(source, target)
        item.update(path=item["filename"], hidden=False, restored=True)
        write_json(Path(folder) / MANIFEST_NAME, manifest)
        cache = Path(folder) / "separation_info.json"
        if cache.exists():
            data = json.loads(cache.read_text(encoding="utf-8"))
            data.setdefault("stems", []).append({"path": str(target), "hash": item["sha256"]})
            write_json(cache, data)
        return str(target)


def model_fingerprint(model_dir, model_ids):
    # Include model-specific YAML/JSON dependencies; a changed config invalidates cache.
    root = Path(model_dir)
    files = set(root.glob("*.json"))
    files.update(root / model for model in model_ids)
    registry = json.loads(resources.files("audio_separator").joinpath("models.json").read_text(encoding="utf-8"))
    resolved_configs = set()
    def visit(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in model_ids and isinstance(child, str) and child.endswith(".yaml"):
                    files.add(root / child)
                    resolved_configs.add(key)
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(registry)
    checks = root / "download_checks.json"
    if checks.exists():
        visit(json.loads(checks.read_text(encoding="utf-8")))
    for model, config in {"melband_roformer_big_beta7.ckpt": "big_beta7.yaml",
                          "melband_roformer_becruily_deux.ckpt": "config_deux_becruily.yaml"}.items():
        if model in model_ids:
            files.add(root / config)
            resolved_configs.add(model)
    if any(m.endswith(".ckpt") and m not in resolved_configs for m in model_ids):
        # Older UVR catalogs can express config relationships indirectly.
        # Invalidate broadly rather than reuse an output with untracked YAML.
        files.update(root.glob("*.yaml"))
    # Demucs YAMLs reference weight files rather than being checkpoints themselves.
    if any(m.endswith(".yaml") for m in model_ids):
        files.update(root.glob("*.th"))
    return {p.name: file_hash(p) if p.is_file() else None for p in sorted(files)}


def mixable_stems(paths, folder):
    """Prefer aggregates over their children; prefer lead/backing over full vocals."""
    entries = {s["filename"]: s for s in read_manifest(folder)["stems"]}
    def role(path):
        name = Path(path).name
        if name in entries:
            return entries[name]["role"]
        # Cloning/cleanup appends tags to the original stem's filename.
        return next((s["role"] for filename, s in entries.items()
                     if Path(path).stem.startswith(Path(filename).stem + "(")
                     or Path(path).stem.startswith(Path(filename).stem + "_")), None)
    present = {role(p) for p in paths} - {None}
    excluded = set()
    excluded.update(s['role'] for s in entries.values() if s.get('duplicate_of') in present)
    # Mega extras can overlap the broad "other" or instrumental estimate.
    # Keep them as audition alternatives unless the user removes those parents.
    if 'instrumental' in present or 'other' in present:
        excluded.update(r for r in present if r.startswith('mega_'))
    if "vocals" in present or "bg_vocals" in present:
        excluded.add("vocals_full")
    if "bg_vocals" in present:
        excluded.update(r for r in present if r.startswith("bg_vocals_"))
    if "instrumental" in present:
        excluded.update({"drums", "bass", "guitar", "piano", "other", "woodwinds"})
        excluded.update(r for r in present if r.startswith("drums_"))
    elif "drums" in present:
        excluded.update(r for r in present if r.startswith("drums_"))
    return [p for p in paths if role(p) not in excluded]
