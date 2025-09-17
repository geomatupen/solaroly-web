# backend/pvrt/infer/predict_rgb_thermal.py
from __future__ import annotations
import json, os, hashlib
from pathlib import Path

import cv2
import numpy as np
import torch

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

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

def _project_root() -> Path:
    # backend/pvrt/infer/predict_rgb_thermal.py  -> parents[3] == backend
    # project root is backend.parent
    backend_dir = Path(__file__).resolve().parents[3]
    return backend_dir.parent

def _load_cfg_like_notebook(weights_dir: Path):
    """
    Mirror test.ipynb:
      - Faster R-CNN base (boxes only)
      - If weights_dir/config.yaml exists, merge it after the zoo file
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg_yaml = weights_dir / "config.yaml"
    if cfg_yaml.exists():
        cfg.merge_from_file(str(cfg_yaml))
    cfg.MODEL.MASK_ON = False
    return cfg

def _find_coco_ann(weights_dir: Path) -> Path | None:
    """
    Find a *_annotations.coco.json if model_meta lacks num_classes.
    Priority:
      1) PVRT_COCO_ANN env var (absolute or relative to project root)
      2) near the weights_dir (itself / parent / grandparent)
      3) project data folders: <project>/data/train or valid
    """
    # 1) env var
    envp = os.environ.get("PVRT_COCO_ANN")
    if envp:
        p = Path(envp)
        if not p.is_absolute():
            p = (_project_root() / p).resolve()
        if p.exists():
            return p

    # 2) local vicinity of run dir
    for base in (weights_dir, weights_dir.parent, weights_dir.parent.parent):
        if base.exists():
            for jf in base.rglob("*_annotations.coco.json"):
                return jf

    # 3) well-known project data roots
    pr = _project_root()
    for base in (pr / "data" / "train", pr / "data" / "valid"):
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

def _get_num_classes(weights_dir: Path, meta: dict) -> tuple[int, str]:
    """
    Returns (num_classes, source) — 'model_meta.json', 'coco:*.json', or 'default'
    """
    n = meta.get("num_classes")
    if isinstance(n, int) and n > 0:
        return n, "model_meta.json"
    ann = _find_coco_ann(weights_dir)
    if ann:
        n = _derive_num_classes_from_coco(ann)
        if n > 0:
            return n, f"coco:{ann.name}"
    return -1, "default"

def _find_thermal(rgb_path: Path) -> Path | None:
    for c in (
        rgb_path.with_name(rgb_path.stem + "_thermal.tif"),
        rgb_path.with_name(rgb_path.stem + "_thermal.tiff"),
    ):
        if c.exists():
            return c
    return None

def _normalize_thermal(arr: np.ndarray) -> np.ndarray:
    th = arr.astype(np.float32)
    if np.nanmax(th) > 1.5:
        th = np.clip(th, 0.0, 100.0) / 100.0
    else:
        tmin, tmax = float(np.nanmin(th)), float(np.nanmax(th))
        th = (th - tmin) / (tmax - tmin) if tmax > tmin else th * 0.0
    return th


# ------------------------- builder -------------------------

def _build_predictor(weights_dir: Path, use_thermal: bool):
    """
    RGB: DefaultPredictor (BGR, 0–255 stats, boxes-only, SCORE_THRESH from meta or 0.6)
    RGB+Thermal: widen first conv to 4ch and set 4-ch PIXEL_MEAN/STD in 0–255 scale.
    Returns: (mode, predictor_or_model, cfg, meta, weights_path, num_src)
    """
    device = _pick_device()
    meta = _load_meta(weights_dir)

    cfg = _load_cfg_like_notebook(weights_dir)

    weights_path = _resolve_weights(weights_dir)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = device

    print(f"UI:INFO:test: Loading weights file: {weights_path}")


    # NUM_CLASSES
    num_classes, num_src = _get_num_classes(weights_dir, meta)
    if num_classes > 0:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = int(num_classes)
    else:
        num_classes = int(cfg.MODEL.ROI_HEADS.NUM_CLASSES or 1)

    # SCORE THRESH
    thr = 0.6
    if "score_thresh_test" in meta:
        try:
            thr = float(meta["score_thresh_test"])
        except Exception:
            pass
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(thr)

    # input_mode from training (if present)
    input_mode = (meta.get("input_mode") or "rgb").lower()
    rgb_only = (input_mode == "rgb")

    if (not use_thermal) or rgb_only:
        predictor = DefaultPredictor(cfg)
        return "rgb", predictor, cfg, meta, weights_path, num_src

    # RGB+Thermal: 4-ch normalize (0–255 BGR means + 0 for thermal), std=1
    cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675, 0.0]
    cfg.MODEL.PIXEL_STD  = [1.0,     1.0,     1.0,     1.0]
    model = build_model(cfg)
    model.eval()
    model.to(device)
    widen_backbone_conv1_to_4ch(model)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    return "rgbtherm", model, cfg, meta, weights_path, num_src


# ------------------------- main entry -------------------------

def predict_folder(images_dir, out_dir, weights_dir, use_thermal: bool = True) -> Path:
    """
    predict_folder(ds_dir, sess, model_dir, use_thermal)
    Writes per-image JSON into out_dir / "preds".
    Returns out_dir Path.
    """
    images_dir = Path(images_dir)
    out_dir = Path(out_dir)
    weights_dir = Path(weights_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "preds").mkdir(exist_ok=True)

    mode, pred_or_model, cfg, meta, weights_path, num_src = _build_predictor(weights_dir, use_thermal)
    device = cfg.MODEL.DEVICE

    # Detailed log to validate we match the notebook & training
    try:
        wpath = Path(weights_path)
        w_sz = wpath.stat().st_size if wpath.exists() else -1
        w_md5 = hashlib.md5(wpath.read_bytes()).hexdigest()[:8] if wpath.exists() else "missing"
    except Exception:
        w_sz, w_md5 = -1, "n/a"

    (out_dir / "predict_log.txt").write_text(
        "device={}\nmode={}\nweights={}\nweights_bytes={}\nweights_md5={}\n"
        "classes={}\nclasses_src={}\nscore_thresh={}\nmask_on={}\n"
        "input.format={}\nmin_size_test={}\nmax_size_test={}\nrpn_nms_thresh={}\nroi_nms_thresh={}\n".format(
            device,
            mode,
            Path(weights_path).name,
            w_sz,
            w_md5,
            cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            num_src,
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            getattr(cfg.MODEL, "MASK_ON", False),
            getattr(cfg.INPUT, "FORMAT", "BGR"),
            getattr(cfg.INPUT, "MIN_SIZE_TEST", None),
            getattr(cfg.INPUT, "MAX_SIZE_TEST", None),
            getattr(cfg.MODEL.RPN, "NMS_THRESH", None) if hasattr(cfg, "MODEL") else None,
            getattr(cfg.MODEL.ROI_HEADS, "NMS_THRESH_TEST", None),
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

        if mode == "rgb":
            outputs = pred_or_model(bgr)
            inst = outputs.get("instances", None)
            inst = inst.to("cpu") if inst is not None else None
        else:
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

            ch4 = np.dstack([bgr.astype(np.float32), (th * 255.0)]).astype(np.float32)
            tensor = torch.as_tensor(ch4.transpose(2, 0, 1)).to(device)
            inputs = [{"image": tensor, "height": H, "width": W}]
            with torch.no_grad():
                outs = pred_or_model(inputs)
            inst = outs[0].get("instances", None)
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
