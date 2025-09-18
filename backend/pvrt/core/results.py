# backend/pvrt/core/results.py
"""
Result helpers for predictors and trainers.
Keeps output structure consistent across backends and models.
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterable, Sequence, Dict, Any
import json


# -------------------------
# Directory layout helpers
# -------------------------

def ensure_results_layout(root: Path) -> Dict[str, Path]:
    """
    Create a standard results layout under `root` and return useful paths.

    Layout:
      root/
        preds/         # one JSON per image (boxes, scores, classes, ...)
        artifacts/     # any extra files (debug imgs, timing logs, etc.)
        metrics.json   # optional summary written by caller
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    preds = root / "preds"
    arts  = root / "artifacts"
    preds.mkdir(parents=True, exist_ok=True)
    arts.mkdir(parents=True, exist_ok=True)
    return {"root": root, "preds": preds, "artifacts": arts}


# -------------------------
# JSON writing primitives
# -------------------------

def write_pred_json(
    out_dir: Path,
    stem: str,
    *,
    boxes_xyxy: Sequence[Sequence[float]] | None = None,
    scores: Sequence[float] | None = None,
    classes: Sequence[int] | None = None,
    extra: Dict[str, Any] | None = None,
) -> Path:
    """
    Write a single per-image prediction JSON.

    Required keys (front-end expects these):
      - boxes:   [[x1,y1,x2,y2], ...]
      - scores:  [p1, p2, ...]  in [0,1]
      - classes: [0, 1, ...]    integer class ids

    `extra` can include anything else (e.g., masks file paths, timing, model info).
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
        # don't overwrite required keys
        for k, v in extra.items():
            if k not in data:
                data[k] = v

    path = out_dir / f"{stem}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_metrics_json(out_root: Path, metrics: Dict[str, Any]) -> Path:
    """
    Write an optional metrics summary at the root of results.
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    p = out_root / "metrics.json"
    p.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return p
