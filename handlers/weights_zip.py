import json
import logging
import os
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from handlers.config import model_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WeightsZipModel:
    display_name: str
    title: str
    author: str
    tags: List[str]
    pth_path: str
    index_path: Optional[str]
    zip_path: str


def _project_root() -> Path:
    # handlers/weights_zip.py -> handlers -> project root
    return Path(__file__).resolve().parent.parent


def _trained_dir() -> Path:
    return Path(model_path) / "trained"


def _extracted_root() -> Path:
    return _project_root() / ".extracted"


def _safe_extract_zip(zf: zipfile.ZipFile, dest: Path) -> None:
    dest = dest.resolve()
    for info in zf.infolist():
        name = info.filename
        # Skip directory entries and common junk
        if not name or name.endswith("/") or name.endswith("\\"):
            continue
        if name.startswith("__MACOSX/") or "/__MACOSX/" in name:
            continue

        target = (dest / name).resolve()
        if dest not in target.parents and target != dest:
            raise ValueError(f"Blocked zip path traversal: {name}")

        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)


def _zip_state(zip_path: Path) -> Dict[str, Any]:
    st = zip_path.stat()
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _read_json_from_zip(zf: zipfile.ZipFile, member: str) -> Optional[Dict[str, Any]]:
    try:
        raw = zf.read(member)
        return json.loads(raw.decode("utf-8", errors="replace"))
    except Exception:
        return None


def _pick_metadata(zf: zipfile.ZipFile) -> Dict[str, Any]:
    names = [n for n in zf.namelist() if n and not n.endswith("/")]
    candidates = []
    for n in names:
        base = os.path.basename(n).lower()
        if base == "metadata.json":
            candidates.insert(0, n)
        elif base.endswith(".json"):
            candidates.append(n)
    for c in candidates:
        meta = _read_json_from_zip(zf, c)
        if isinstance(meta, dict) and meta:
            return meta
    return {}


def _coerce_tags(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        # allow comma-separated tags
        parts = [p.strip() for p in value.split(",")]
        return [p for p in parts if p]
    return [str(value).strip()] if str(value).strip() else []


def _normalize_metadata(meta: Dict[str, Any], fallback_title: str) -> Tuple[str, str, List[str]]:
    title = str(meta.get("title") or meta.get("name") or meta.get("model_name") or fallback_title).strip()
    author = str(meta.get("author") or meta.get("creator") or meta.get("user") or meta.get("username") or "").strip()
    tags = _coerce_tags(meta.get("tags") or meta.get("tag") or meta.get("genres") or meta.get("style"))
    return title, author, tags


def _find_model_files(root: Path) -> Tuple[List[Path], List[Path]]:
    pths: List[Path] = []
    indexes: List[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.name.startswith("._"):
            continue
        low = p.name.lower()
        if low.endswith(".pth"):
            pths.append(p)
        elif ".index" in low and low.endswith(".index"):
            indexes.append(p)
    return pths, indexes


def _pick_index_for_pth(pth: Path, indexes: List[Path]) -> Optional[Path]:
    if not indexes:
        return None
    stem = pth.stem.lower()
    # Prefer index containing the pth stem
    for idx in indexes:
        if stem and stem in idx.stem.lower():
            return idx
    # If only one index exists, use it
    if len(indexes) == 1:
        return indexes[0]
    return None


def _ensure_extracted(zip_path: Path, extracted_dir: Path) -> None:
    extracted_dir.mkdir(parents=True, exist_ok=True)
    state_path = extracted_dir / ".zip_state.json"
    current = _zip_state(zip_path)

    if state_path.exists():
        try:
            existing = json.loads(state_path.read_text(encoding="utf-8"))
            if isinstance(existing, dict) and existing.get("size") == current["size"] and existing.get("mtime_ns") == current["mtime_ns"]:
                return
        except Exception:
            pass

    # Re-extract
    if extracted_dir.exists():
        shutil.rmtree(extracted_dir, ignore_errors=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        _safe_extract_zip(zf, extracted_dir)
    state_path.write_text(json.dumps(current, indent=2), encoding="utf-8")


def prune_missing_extracted() -> None:
    """
    Remove `.extracted/<zip_stem>` folders when the corresponding zip no longer exists.
    Safe to call on every refresh.
    """
    extracted_root = _extracted_root()
    if not extracted_root.exists():
        return
    trained_dir = _trained_dir()
    existing_stems = {p.stem for p in trained_dir.glob("*.zip")}
    for child in extracted_root.iterdir():
        if child.is_dir() and child.name not in existing_stems:
            shutil.rmtree(child, ignore_errors=True)


def list_weights_zip_models() -> List[WeightsZipModel]:
    """
    Discover and (if needed) extract `models/trained/*.zip` into `.extracted/`,
    returning zip-backed RVC model candidates.
    """
    trained_dir = _trained_dir()
    trained_dir.mkdir(parents=True, exist_ok=True)
    extracted_root = _extracted_root()
    extracted_root.mkdir(parents=True, exist_ok=True)

    prune_missing_extracted()

    models: List[WeightsZipModel] = []
    for zip_path in sorted(trained_dir.glob("*.zip"), key=lambda p: p.name.lower()):
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                meta = _pick_metadata(zf)
            extracted_dir = extracted_root / zip_path.stem
            _ensure_extracted(zip_path, extracted_dir)

            pths, indexes = _find_model_files(extracted_dir)
            if not pths:
                continue

            title, author, tags = _normalize_metadata(meta, fallback_title=zip_path.stem)

            # Keep dropdown names unique and human-readable
            multi = len(pths) > 1
            for pth in sorted(pths, key=lambda p: p.name.lower()):
                display_title = title
                if multi:
                    display_title = f"{title} - {pth.stem}"
                display_name = f"{display_title} (zip)"
                idx = _pick_index_for_pth(pth, indexes)
                models.append(
                    WeightsZipModel(
                        display_name=display_name,
                        title=title,
                        author=author,
                        tags=tags,
                        pth_path=str(pth),
                        index_path=str(idx) if idx else None,
                        zip_path=str(zip_path),
                    )
                )
        except zipfile.BadZipFile:
            logger.warning(f"Bad zip: {zip_path}")
        except Exception as e:
            logger.warning(f"Failed to process weights zip {zip_path}: {e}")

    return models

