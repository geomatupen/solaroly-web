# backend/pvrt/infer/predict_rgb_thermal.py
from __future__ import annotations
import json
import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2 import model_zoo

from ..trainops.model_patch import widen_backbone_conv1_to_4ch


# ------------------------- helpers -------------------------

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

def _load_run_cfg(weights_dir: Path):
    """
    Prefer the exact training config.yaml from the run directory.
    If missing, fall back to a common Faster R-CNN cfg (logged).
    """
    cfg = get_cfg()
    cfg_path = weights_dir / "config.yaml"
    if cfg_path.exists():
        cfg.merge_from_file(str(cfg_path))
        cfg._pvrt_cfg_source = "run_config.yaml"
    else:
        # Fallback to a sane default so we don't crash;
        # weights may not match if the arch differs, but we'll log it.
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg._pvrt_cfg_source = "fallback_frcnn_zoo"
    return cfg

def _find_thermal(rgb_path: Path) -> Path | None:
    for c in (
        rgb_path.with_name(rgb_path.stem + "_thermal.tif"),
        rgb_path.with_name(rgb_path.stem + "_thermal.tiff"),
    ):
        if c.exists():
            return c
    return None

def _normalize_thermal(arr: np.ndarray) -> np.ndarray:
    """Return float32 in [0,1] for thermal band."""
    th = arr.astype(np.float32)
    if np.nanmax(th) > 1.5:
        th = np.clip(th, 0.0, 100.0) / 100.0
    else:
        tmin, tmax = float(np.nanmin(th)), float(np.nanmax(th))
        th = (th - tmin) / (tmax - tmin) if tmax > tmin else th * 0.0
    return th


# ------------------------- builder -------------------------

def _build_thermal_model(weights_dir: Path):
    """
    Build a 4-channel model for rgb+thermal inference.
    Keeps training cfg as-is, only extends PIXEL_MEAN/STD to 4ch and widens the stem.
    Returns: (model, cfg, meta, weights_path)
    """
    device = _pick_device()
    meta = _load_meta(weights_dir)
    cfg = _load_run_cfg(weights_dir)

    weights_path = _resolve_weights(weights_dir)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = device

    # If training cfg didn't record NUM_CLASSES but meta has it, fill it.
    try:
        cfg_nc = int(getattr(cfg.MODEL.ROI_HEADS, "NUM_CLASSES", 0) or 0)
    except Exception:
        cfg_nc = 0
    meta_nc = int(meta.get("num_classes", 0) or 0)
    if cfg_nc <= 0 and meta_nc > 0:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = meta_nc

    # If meta provides a preferred test threshold, take it (else keep cfg's).
    thr = meta.get("score_thresh_test", None)
    if thr is not None:
        try:
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(thr)
        except Exception:
            pass

    # Extend PIXEL_MEAN/STD to 4 channels (preserve original 3ch values).
    means = list(getattr(cfg.MODEL, "PIXEL_MEAN", [103.530, 116.280, 123.675]))
    stds  = list(getattr(cfg.MODEL, "PIXEL_STD",  [1.0,     1.0,     1.0    ]))
    if len(means) == 3:
        means = means + [0.0]
    if len(stds) == 3:
        stds  = stds  + [1.0]
    cfg.MODEL.PIXEL_MEAN = means
    cfg.MODEL.PIXEL_STD  = stds

    # Build & widen to 4ch
    model = build_model(cfg)
    model.eval()
    model.to(device)
    widen_backbone_conv1_to_4ch(model)  # safe if already widened
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    return model, cfg, meta, weights_path


# ------------------------- main entry -------------------------

def predict_folder(images_dir, out_dir, weights_dir, use_thermal: bool = True) -> Path:
    """
    Thermal inference path.
    - Expects RGB images in `images_dir` and a sidecar thermal TIF per image: <stem>_thermal.tif[f]
    - Writes per-image JSONs to out_dir / "preds"
    """
    images_dir = Path(images_dir)
    out_dir = Path(out_dir)
    weights_dir = Path(weights_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "preds").mkdir(exist_ok=True)

    model, cfg, meta, weights_path = _build_thermal_model(weights_dir)
    device = cfg.MODEL.DEVICE

    # Predict log for auditing
    try:
        wpath = Path(weights_path)
        w_sz = wpath.stat().st_size if wpath.exists() else -1
        w_md5 = hashlib.md5(wpath.read_bytes()).hexdigest()[:8] if wpath.exists() else "missing"
    except Exception:
        w_sz, w_md5 = -1, "n/a"

    metaarch = getattr(cfg.MODEL, "META_ARCHITECTURE", "unknown")
    mask_on  = bool(getattr(cfg.MODEL, "MASK_ON", False))
    nclasses = int(getattr(cfg.MODEL.ROI_HEADS, "NUM_CLASSES", 0) or 0)
    fmt      = getattr(cfg.INPUT, "FORMAT", "BGR")
    score_th = getattr(cfg.MODEL.ROI_HEADS, "SCORE_THRESH_TEST", None)
    cfg_src  = getattr(cfg, "_pvrt_cfg_source", "unknown")

    (out_dir / "predict_log.txt").write_text(
        "device={}\nmode={}\nweights={}\nweights_bytes={}\nweights_md5={}\n"
        "cfg_source={}\nmeta_arch={}\nmask_on={}\nclasses={}\n"
        "score_thresh={}\ninput.format={}\npixel_mean={} (len={})\npixel_std={} (len={})\n".format(
            device, "rgbtherm", Path(weights_path).name, w_sz, w_md5,
            cfg_src, metaarch, mask_on, nclasses,
            score_th, fmt, getattr(cfg.MODEL, "PIXEL_MEAN", None), len(getattr(cfg.MODEL, "PIXEL_MEAN", [])),
            getattr(cfg.MODEL, "PIXEL_STD", None), len(getattr(cfg.MODEL, "PIXEL_STD", [])),
        )
    )

    # Let the UI mini-log show which exact weights file is used
    print(f"UI:INFO:test: Using weights (thermal): {weights_path}")

    valid_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    img_paths = [p for p in sorted(images_dir.iterdir()) if p.suffix.lower() in valid_exts]

    for p in img_paths:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            (out_dir / "preds" / f"{p.stem}.json").write_text(
                json.dumps({"file": p.name, "boxes": [], "scores": [], "classes": [], "reason": "read_failed"}, indent=2)
            )
            continue

        th_path = _find_thermal(p)
        if th_path is None:
            (out_dir / "preds" / f"{p.stem}.json").write_text(
                json.dumps({"file": p.name, "boxes": [], "scores": [], "classes": [], "reason": "no_thermal_sidecar"}, indent=2)
            )
            continue

        therm = cv2.imread(str(th_path), cv2.IMREAD_UNCHANGED)
        if therm is None:
            (out_dir / "preds" / f"{p.stem}.json").write_text(
                json.dumps({"file": p.name, "boxes": [], "scores": [], "classes": [], "reason": "thermal_read_failed"}, indent=2)
            )
            continue

        H, W = bgr.shape[:2]
        th = _normalize_thermal(therm)
        if th.shape[:2] != (H, W):
            th = cv2.resize(th, (W, H), interpolation=cv2.INTER_LINEAR)

        # Stack BGR (uint8) + thermal*255.0 → float32 CHW; model normalizer (PIXEL_MEAN/STD) handles scaling
        ch4 = np.dstack([bgr.astype(np.float32), (th * 255.0)]).astype(np.float32)
        tensor = torch.as_tensor(ch4.transpose(2, 0, 1)).to(device)  # (4,H,W)
        inputs = [{"image": tensor, "height": H, "width": W}]

        with torch.no_grad():
            outs = model(inputs)

        inst = outs[0].get("instances", None)
        if inst is not None:
            inst = inst.to("cpu")

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
