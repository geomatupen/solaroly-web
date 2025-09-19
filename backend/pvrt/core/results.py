# backend/pvrt/core/results.py
"""
Result helpers for predictors and trainers.
Keeps output structure consistent across backends and models.
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Sequence, Dict, Any
import json
import cv2

# -------------------------
# Directory layout helpers
# -------------------------

def ensure_results_layout(root: Path | str) -> Dict[str, Path]:
    """
    Create a consistent results folder layout and return handy paths.

    Returns a dict with keys:
      - "root":    <root>
      - "preds":   <root>/preds
      - "overlay": <root>/overlay         # NOTE: singular 'overlay' (your existing folder)
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    preds = root / "preds"
    ovl   = root / "overlay"   # <-- singular, matches your project
    preds.mkdir(exist_ok=True)
    ovl.mkdir(exist_ok=True)
    return {"root": root, "preds": preds, "overlay": ovl}

# -------------------------
# JSON writers
# -------------------------

def write_pred_json(out_dir: Path | str,
                    stem: str,
                    boxes_xyxy: Sequence[Sequence[float]] | None,
                    scores: Sequence[float] | None,
                    classes: Sequence[int] | None,
                    extra: Dict[str, Any] | None = None) -> Path:
    """
    Write a single prediction JSON for one image: <out_dir>/<stem>.json
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data: Dict[str, Any] = {
        "file": f"{stem}",
        "boxes": list(map(list, boxes_xyxy or [])),
        "scores": list(scores or []),
        "classes": list(classes or []),
    }
    if extra:
        for k, v in extra.items():
            if k not in data:
                data[k] = v

    path = out_dir / f"{stem}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

def write_metrics_json(out_root: Path | str, metrics: Dict[str, Any]) -> Path:
    """
    Write a run summary at the root of results.
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    p = out_root / "metrics.json"
    p.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return p

# -------------------------
# Image helpers
# -------------------------

def save_overlay_png(overlays_dir: Path | str, stem: str, bgr_image) -> Path:
    """
    Save ONE PNG overlay image named <stem>.png into the 'overlay' folder.
    Ensures uint8 and overwrites existing file.
    """
    overlays_dir = Path(overlays_dir)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    out = overlays_dir / f"{stem}.png"
    img = bgr_image
    if img.dtype != "uint8":
        import numpy as np
        img = np.clip(img, 0, 255).astype("uint8")
    cv2.imwrite(str(out), img)
    return out
