"""Inference helper for YOLO using ultralytics.YOLO

Provides `predict_folder` which runs predictions on a folder of images and
saves per-image JSON results and annotated images under out_dir.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import json
import logging

log = logging.getLogger("pvrt")


def _serialize_prediction(r) -> Dict[str, Any]:
    # r is a ultralytics Result-like object
    out = {
        "path": getattr(r, "orig_img_path", getattr(r, "path", None)),
        "boxes": [],
        "scores": [],
        "classes": [],
    }
    try:
        boxes = getattr(r, "boxes", None)
        if boxes is not None:
            for b in boxes:
                try:
                    xyxy = b.xyxy.cpu().numpy().tolist()[0] if hasattr(b, "xyxy") else None
                    conf = float(b.conf.cpu().numpy()[0]) if hasattr(b, "conf") else None
                    cls = int(b.cls.cpu().numpy()[0]) if hasattr(b, "cls") else None
                    out["boxes"].append(xyxy)
                    out["scores"].append(conf)
                    out["classes"].append(cls)
                except Exception:
                    continue
    except Exception:
        pass
    return out


def predict_folder(images_dir: Path, weights_dir: Path, out_dir: Path, score_thresh: float = 0.25, use_thermal: bool = False) -> Path:
    from ultralytics import YOLO

    model_weights = None
    # prefer best.pt then last.pt in weights_dir
    w = Path(weights_dir)
    cand_best = w / "best.pt"
    cand_last = w / "last.pt"
    if cand_best.exists():
        model_weights = str(cand_best)
    elif cand_last.exists():
        model_weights = str(cand_last)
    else:
        # maybe weights_dir is a file path
        if w.is_file():
            model_weights = str(w)

    if model_weights is None:
        raise FileNotFoundError(f"No YOLO weights found in {weights_dir}")

    model = YOLO(model_weights)

    # ensure outputs
    run_dir = Path(out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    results = model.predict(
        source=str(images_dir),
        conf=score_thresh,
        imgsz=1024,
        device=0,
        save=False,
        save_txt=False,
    )

    # results is iterable per-image
    out_json_dir = run_dir / "pred_json"
    out_json_dir.mkdir(parents=True, exist_ok=True)

    for r in results:
        # attempt to resolve path
        p = getattr(r, "orig_img_path", None) or getattr(r, "path", None) or None
        key = Path(p).name if p else f"img_{len(list(out_json_dir.iterdir()))}" 
        js = _serialize_prediction(r)
        out_path = out_json_dir / f"{key}.json"
        out_path.write_text(json.dumps(js, ensure_ascii=False, indent=2), encoding="utf-8")

    return run_dir
