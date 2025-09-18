# backend/pvrt/core/io.py
"""
I/O helpers used by all backends.

Keeps file-handling and small, re-usable utilities in one place so
individual backends (Detectron, YOLO, ...) stay lean.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Iterable
import json


# ---------- JSON helpers ----------

def read_json_safe(path: Path) -> Dict[str, Any]:
    """
    Read a JSON file if it exists, otherwise return {}.
    Never raises on read/parse errors; returns {} instead.
    """
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def write_json_safe(path: Path, obj: Dict[str, Any]) -> None:
    """
    Write JSON atomically (best-effort): write to a temp file then replace.
    Falls back to direct write if replace fails (e.g., cross-device).
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(obj, ensure_ascii=False, indent=2)
    tmp.write_text(data, encoding="utf-8")
    try:
        tmp.replace(path)
    except Exception:
        # Fall back to direct write
        path.write_text(data, encoding="utf-8")
        try:
            tmp.unlink(missing_ok=True)  # py>=3.8
        except Exception:
            pass


# ---------- Model meta helpers ----------

def load_model_meta(run_or_weights_dir: Path) -> Dict[str, Any]:
    """
    Load `model_meta.json` from a trained run directory.
    Returns {} if not found or invalid.
    """
    meta_path = Path(run_or_weights_dir) / "model_meta.json"
    return read_json_safe(meta_path)


def save_model_meta(run_dir: Path, meta: Dict[str, Any]) -> None:
    """
    Save `model_meta.json` to a trained run directory.
    """
    meta_path = Path(run_dir) / "model_meta.json"
    write_json_safe(meta_path, meta)


def input_mode_from_meta(meta: Dict[str, Any], default: str = "rgb") -> str:
    """
    Normalize the model's input mode from metadata.
    Expected values: "rgb" (3-band), "rgbt" (RGB+Thermal).
    """
    val = (meta.get("input_mode") or default).strip().lower()
    if val in {"rgbt", "rgb+t", "rgb_thermal", "thermal_rgb"}:
        return "rgbt"
    return "rgb"


def backend_name_from_meta(meta: Dict[str, Any], default: str = "detectron") -> str:
    """
    Extract which backend trained the model (e.g., 'detectron', 'yolo').
    """
    name = (meta.get("backend") or default).strip().lower()
    return name or default


# ---------- Thermal availability helpers ----------

THERMAL_DIR_CANDIDATES: Iterable[str] = ("thermal", "ir", "t", "temp")

def has_thermal_for_images(images_dir: Path) -> bool:
    """
    Heuristic to decide if thermal data is available for a set of images.
    Rules (in order):
      1) If a `thermal/pairs.json` exists → True.
      2) If any known thermal subdir contains files → True.
      3) Otherwise → False.
    """
    d = Path(images_dir)

    # 1) Explicit pairing file
    pairs = d / "thermal" / "pairs.json"
    if pairs.exists():
        try:
            j = read_json_safe(pairs)
            if j:  # any content signals presence
                return True
        except Exception:
            # ignore parse errors, fall through to scan
            pass

    # 2) Subdir scan
    for name in THERMAL_DIR_CANDIDATES:
        td = d / name
        if td.exists() and td.is_dir():
            # any file in subdir is considered a positive signal
            for _ in td.iterdir():
                return True

    return False
