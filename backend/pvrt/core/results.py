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
from PIL import Image

# -------------------------
# Directory layout helpers
# -------------------------

def ensure_results_layout(root: Path | str) -> Dict[str, Path]:
    """
    Create a consistent results folder layout and return handy paths.

        Returns a dict with keys:
            - "root":    <root>
            - "preds":   <root>/preds
            - "overlays": <root>/overlays
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    preds = root / "preds"
    ovl   = root / "overlays"
    preds.mkdir(exist_ok=True)
    ovl.mkdir(exist_ok=True)
    # Add a dedicated folder for exact normalized thermal previews so
    # inference runs can save the same uint8 images that the model sees.
    thr = root / "thermal"
    thr.mkdir(exist_ok=True)
    return {"root": root, "preds": preds, "overlays": ovl, "thermal": thr}

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
    Save a PNG overlay image named <stem>.png into the 'overlays' folder.
    Ensures the image is uint8 and overwrites any existing file.
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


def save_overlay_jpg(
    overlays_dir: Path | str,
    stem: str,
    bgr_image,
    exif_source: Path | str | None = None,
    quality: int = 92,
) -> Path:
    """
    Save ONE JPEG overlay <stem>.jpg into the 'overlay' folder.
    - Accepts a BGR uint8 (OpenCV) image.
    - If exif_source is provided and has EXIF, copy it (GPS etc.) into the JPEG.
    """
    import numpy as np
    import cv2

    overlays_dir = Path(overlays_dir)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    out = overlays_dir / f"{stem}.jpg"

    img = bgr_image
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    # BGR (cv2) -> RGB (Pillow)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    # Try to copy EXIF (incl. GPS) from the original image
    exif_bytes = None
    if exif_source is not None:
        with Image.open(str(exif_source)) as src:
            exif_bytes = src.info.get("exif")

    save_kwargs = {"quality": quality, "subsampling": 0}
    if exif_bytes:
        save_kwargs["exif"] = exif_bytes

    pil_img.save(out, format="JPEG", **save_kwargs)
    return out