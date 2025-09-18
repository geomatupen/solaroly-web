# backend/pvrt/infer/predict_rgb_only.py
from __future__ import annotations
import json, os, hashlib
from pathlib import Path

import cv2
import numpy as np
import torch

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo

# ---------- helpers ----------

def _pick_device() -> str:
    try:
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

def _load_meta(weights_dir: Path) -> dict:
    p = weights_dir / "model_meta.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}

def _resolve_weights(weights_dir: Path) -> str:
    for name in ("model_final.pth", "model_best.pth", "model.pth"):
        p = weights_dir / name
        if p.exists():
            return str(p)
    return str(weights_dir / "model_final.pth")

def _load_cfg_like_notebook() -> "CfgNode":
    """
    EXACTLY like your notebook:
      - COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml
      - MASK_ON=False (boxes only)
      - BGR inputs via cv2 (do not override INPUT.FORMAT/means/stds)
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.MASK_ON = False
    return cfg

def _find_coco_ann(weights_dir: Path) -> Path | None:
    # optional: derive num_classes if meta missing
    envp = os.environ.get("PVRT_COCO_ANN")
    if envp:
        p = Path(envp)
        if not p.is_absolute():
            p = (weights_dir / p).resolve()
        if p.exists():
            return p
    for base in (weights_dir, weights_dir.parent, weights_dir.parent.parent):
        if base.exists():
            for jf in base.rglob("*_annotations.coco.json"):
                return jf
    return None

def _derive_num_classes_from_coco(ann: Path) -> int:
    try:
        data = json.loads(ann.read_text(encoding="utf-8"))
        cats = data.get("categories") or []
        return len(cats) if isinstance(cats, list) else 0
    except Exception:
        return 0

# ---------- main ----------

def predict_folder(images_dir, out_dir, weights_dir, use_thermal: bool = False) -> Path:
    """
    SAME signature as the thermal version, but ignores use_thermal.
    Writes per-image JSON to out_dir/preds/*.json
    """
    images_dir = Path(images_dir)
    out_dir = Path(out_dir)
    weights_dir = Path(weights_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "preds").mkdir(exist_ok=True)

    meta = _load_meta(weights_dir)

    cfg = _load_cfg_like_notebook()
    cfg.MODEL.WEIGHTS = _resolve_weights(weights_dir)
    cfg.MODEL.DEVICE = _pick_device()

    # Class count: prefer meta, else derive from COCO json, else keep cfg default
    nclasses = int(meta.get("num_classes", 0) or 0)
    if nclasses <= 0:
        ann = _find_coco_ann(weights_dir)
        if ann:
            nclasses = _derive_num_classes_from_coco(ann)
    if nclasses > 0:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = nclasses

    # Threshold: meta override or default 0.6 (your notebook)
    thr = meta.get("score_thresh_test", 0.6)
    try:
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(thr)
    except Exception:
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6

    predictor = DefaultPredictor(cfg)

    # log for auditing
    try:
        wpath = Path(cfg.MODEL.WEIGHTS)
        w_sz = wpath.stat().st_size if wpath.exists() else -1
        w_md5 = hashlib.md5(wpath.read_bytes()).hexdigest()[:8] if wpath.exists() else "missing"
    except Exception:
        w_sz, w_md5 = -1, "n/a"
    (out_dir / "predict_log.txt").write_text(
        "device={}\nmode={}\nweights={}\nweights_bytes={}\nweights_md5={}\n"
        "classes={}\nscore_thresh={}\nmask_on={}\ninput.format={}\n".format(
            cfg.MODEL.DEVICE, "rgb", Path(cfg.MODEL.WEIGHTS).name, w_sz, w_md5,
            cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            getattr(cfg.MODEL, "MASK_ON", False),
            getattr(cfg.INPUT, "FORMAT", "BGR"),
        )
    )

    valid_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    img_paths = [p for p in sorted(images_dir.iterdir()) if p.suffix.lower() in valid_exts]

    for p in img_paths:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            (out_dir / "preds" / f"{p.stem}.json").write_text(
                json.dumps({"file": p.name, "boxes": [], "scores": [], "classes": [], "reason": "read_failed"}, indent=2)
            )
            continue

        outputs = predictor(bgr)
        inst = outputs.get("instances", None)
        inst = inst.to("cpu") if inst is not None else None

        if inst is None or len(inst) == 0:
            boxes, scores, classes = [], [], []
        else:
            boxes   = inst.pred_boxes.tensor.numpy().tolist()
            scores  = inst.scores.numpy().tolist()
            classes = inst.pred_classes.numpy().tolist()

        (out_dir / "preds" / f"{p.stem}.json").write_text(
            json.dumps({"file": p.name, "boxes": boxes, "scores": scores, "classes": classes}, indent=2)
        )

    return out_dir
