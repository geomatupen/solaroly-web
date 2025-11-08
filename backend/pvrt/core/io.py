from __future__ import annotations

"""Slim I/O utilities for PVRT.

This module provides safe JSON read/write helpers and small thermal-aware
dataset preparation utilities. Heavy libs (Pillow, numpy, rasterio) are
guarded so the module can be imported in lightweight environments.
"""

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, List
import json
import logging
from shutil import copy2

# Optional heavy imports (guarded)
try:
    from PIL import Image as PILImage
except (ImportError, ModuleNotFoundError):
    PILImage = None

try:
    import numpy as np  # type: ignore
except (ImportError, ModuleNotFoundError):
    np = None

try:
    import rasterio  # type: ignore
    from rasterio.enums import Resampling  # type: ignore
except (ImportError, ModuleNotFoundError):
    rasterio = None
    Resampling = None

LOGGER = logging.getLogger("pvrt")


def read_json_safe(path: Path) -> Dict[str, Any]:
    try:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def write_json_safe(path: Path, obj: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(obj, ensure_ascii=False, indent=2)
    try:
        tmp.write_text(data, encoding="utf-8")
        try:
            tmp.replace(path)
        except OSError:
            path.write_text(data, encoding="utf-8")
    except OSError:
        try:
            path.write_text(data, encoding="utf-8")
        except OSError:
            LOGGER.debug("failed to write json to %s", path)
    try:
        tmp.unlink(missing_ok=True)
    except OSError:
        pass


def load_model_meta(run_or_weights_dir: Path) -> Dict[str, Any]:
    return read_json_safe(Path(run_or_weights_dir) / "model_meta.json")


def save_model_meta(run_dir: Path, meta: Dict[str, Any]) -> None:
    write_json_safe(Path(run_dir) / "model_meta.json", meta)


THERMAL_DIR_CANDIDATES: Iterable[str] = ("thermal", "ir", "t", "temp")


def has_thermal_for_images(images_dir: Path) -> bool:
    d = Path(images_dir)
    td = d / "thermal"
    if td.exists() and td.is_dir():
        for _ in td.iterdir():
            return True
    for name in THERMAL_DIR_CANDIDATES:
        if (d / name).exists():
            return True
    return False


def prepare_dataset_for_run(
    src_train: Path,
    src_valid: Path,
    dest_run: Path,
    selected_bands: Optional[List[str]] = None,
    channel_count: int = 3,
) -> Dict[str, Any]:
    """Prepare a small per-run prepared dataset.

    - 1-channel is coerced to 3-channel (thermal-only) for compatibility.
    - If channel_count==4 and thermal sidecars are present, returns paths
      to the original dirs for backends that can consume RGB+T.
    - Otherwise creates dest_run/prepared/{train,valid} with 3-channel images.
    """
    prepared_root = Path(dest_run) / "prepared"
    train_out = prepared_root / "train"
    valid_out = prepared_root / "valid"
    train_out.mkdir(parents=True, exist_ok=True)
    valid_out.mkdir(parents=True, exist_ok=True)

    if channel_count == 1:
        LOGGER.warning("1-channel datasets are deprecated; coercing to 3-channel")
        channel_count = 3

    if channel_count == 4 and has_thermal_for_images(src_train):
        return {
            "train_dir": str(src_train),
            "valid_dir": str(src_valid),
            "selected_bands": selected_bands or [],
            "channel_count": 4,
        }

    # minimal behavior: copy or convert images to RGB-like outputs
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    def _copy_rgb(src_dir: Path, out_dir: Path) -> None:
        if not src_dir.exists():
            return
        for p in sorted(src_dir.iterdir()):
            if not p.is_file() or p.suffix.lower() not in exts:
                continue
            dst = out_dir / p.name
            try:
                if PILImage is not None:
                    with PILImage.open(p) as im:
                        im = im.convert("RGB")
                        im.save(dst)
                else:
                    copy2(p, dst)
            except (OSError, ValueError):
                LOGGER.debug("failed to process %s", p)

    _copy_rgb(src_train, train_out)
    _copy_rgb(src_valid, valid_out)

    return {
        "train_dir": str(train_out),
        "valid_dir": str(valid_out),
        "selected_bands": selected_bands or [],
        "channel_count": channel_count,
    }


# Public thermal-related constants used by other modules
THERMAL_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


def input_mode_from_meta(meta: Dict[str, Any], default: str = "rgb") -> str:
    """Return normalized input mode string from model metadata.

    Recognizes common synonyms for RGB+thermal models and returns "rgbt"
    for those, otherwise returns "rgb".
    """
    if not isinstance(meta, dict):
        return default
    val = (meta.get("input_mode") or default).strip().lower()
    if val in {"rgbt", "rgb+t", "rgb_thermal", "thermal_rgb", "rgb+thermal", "4ch", "rgbt4"}:
        return "rgbt"
    return "rgb"


def backend_name_from_meta(meta: Dict[str, Any], default: str = "detectron") -> str:
    """Return a normalized backend name from model metadata.

    This is a small compatibility helper used by backend adapters when
    normalizing and saving `model_meta.json`.
    """
    if not isinstance(meta, dict):
        return default
    b = meta.get("backend") or meta.get("engine") or default
    try:
        return str(b).strip().lower()
    except Exception:
        return default

