# backend/pvrt/infer/predict_rgb_thermal.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Tuple
import json

import cv2
import numpy as np
import torch
import torch.nn as nn

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo

# Shared results helpers
from ....core.results import ensure_results_layout, write_pred_json


# ---------------------------
# Small, focused utilities
# ---------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
              ".JPG", ".JPEG", ".PNG", ".BMP", ".TIF", ".TIFF", ".WEBP"}

THERMAL_DIR_NAMES = ("thermal", "ir", "t", "temp")
THERMAL_EXTS = (".tif", ".tiff", ".png")


def _list_images(d: Path) -> List[Path]:
    return [p for p in sorted(Path(d).iterdir()) if p.suffix in IMAGE_EXTS and p.is_file()]


def _pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _read_meta(weights_dir: Path) -> Dict:
    meta = Path(weights_dir) / "model_meta.json"
    return json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}


def _resolve_num_classes(weights_dir: Path, meta: Dict) -> int:
    # Prefer explicit meta
    if isinstance(meta.get("num_classes"), int) and meta["num_classes"] > 0:
        return int(meta["num_classes"])
    # Fallback: inspect nearby COCO annotations (best-effort)
    for base in (Path(weights_dir), Path(weights_dir).parent, Path(weights_dir).parent.parent):
        for jf in base.glob("*_annotations.coco*.json"):
            try:
                cats = json.loads(jf.read_text(encoding="utf-8")).get("categories", [])
                return max(1, len(cats)) if isinstance(cats, list) else 1
            except Exception:
                continue
    return 1


def _resolve_score_thresh(meta: Dict, default: float = 0.5) -> float:
    try:
        return float(meta.get("score_thresh_test", default))
    except Exception:
        return float(default)


def _resolve_weights_path(weights_dir: Path) -> Path:
    p = Path(weights_dir) / "model_final.pth"
    return p if p.exists() else Path(model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))


def _load_pairs_json(images_dir: Path) -> Dict[str, str]:
    """
    Optional pairing file at `<images_dir>/thermal/pairs.json`:
      { "image_file_name": "relative/or/absolute/path/to/thermal.tif", ... }
    """
    pj = Path(images_dir) / "thermal" / "pairs.json"
    if pj.exists():
        try:
            j = json.loads(pj.read_text(encoding="utf-8"))
            # normalize to strings
            out = {}
            for k, v in (j.items() if isinstance(j, dict) else []):
                if isinstance(k, str) and isinstance(v, str):
                    out[k] = v
            return out
        except Exception:
            return {}
    return {}


def _guess_thermal_sidecar(images_dir: Path, img: Path) -> Optional[Path]:
    """
    Heuristics when no pairs.json is present:
      1) <images_dir>/<thermal-like>/<stem>.tif(f)
      2) <images_dir>/<stem>.tif(f)
      3) None
    """
    # search common thermal subdirs
    for dname in THERMAL_DIR_NAMES:
        tdir = Path(images_dir) / dname
        if tdir.exists() and tdir.is_dir():
            for ext in THERMAL_EXTS:
                cand = tdir / f"{img.stem}{ext}"
                if cand.exists():
                    return cand

    # same directory as the RGB image
    for ext in THERMAL_EXTS:
        cand = Path(images_dir) / f"{img.stem}{ext}"
        if cand.exists():
            return cand

    return None


def _load_thermal_channel(path: Path) -> Optional[np.ndarray]:
    """
    Load a thermal raster (tif/png). Return uint8 2D array scaled 0..255.
    Strategy:
      - If data is float: per-image min/max normalize (robust to outliers with percentiles)
      - If data is uint16/uint8: simple min/max normalize to 0..255
    """
    # Use cv2 to avoid heavy deps; it can read tif in many builds
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        return None

    # Take single channel if multi-band by mistake
    if arr.ndim == 3:
        arr = arr[..., 0]

    arr = arr.astype(np.float32)

    # Robust scaling: clip to [p2, p98] then map to 0..255
    p2, p98 = np.percentile(arr, [2.0, 98.0])
    lo, hi = float(p2), float(p98)
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max() if arr.max() > arr.min() else arr.min() + 1.0)

    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0) * 255.0
    return arr.astype(np.uint8)


def _bgrt_stack(img_bgr: np.ndarray, t_uint8: np.ndarray) -> np.ndarray:
    """
    Make a 4-channel BGRT tensor (last channel is thermal).
    """
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("Expected BGR image with 3 channels")
    if t_uint8.ndim != 2:
        raise ValueError("Expected thermal as single-band uint8")
    if t_uint8.shape[:2] != img_bgr.shape[:2]:
        # resize thermal to match RGB
        t_uint8 = cv2.resize(t_uint8, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.dstack([img_bgr, t_uint8])


# --------- conv1 patching (fallback if your pvrt.model_patch is absent) ---------

def _patch_first_conv_to_4ch(model: nn.Module) -> None:
    """
    Find the first Conv2d expecting 3 channels and replace it with a 4-ch variant.
    The 4th channel weights are initialized as the mean of the first three.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            # Create new conv with same hyperparams but 4 input channels
            new_conv = nn.Conv2d(
                in_channels=4,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=(module.bias is not None),
                padding_mode=module.padding_mode,
            )
            with torch.no_grad():
                w = module.weight  # [out, 3, k, k]
                # init from old conv
                new_conv.weight[:, :3, :, :] = w
                # 4th channel = mean of first three
                new_conv.weight[:, 3:4, :, :] = w.mean(dim=1, keepdim=True)
                if module.bias is not None:
                    new_conv.bias.copy_(module.bias)
            # Replace it in its parent
            parent = model
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], new_conv)
            return  # patched first conv; done


# ---------------------------
# Public API
# ---------------------------

def predict_folder(
    *,
    images_dir: Path,
    weights_dir: Path,
    out_dir: Path,
    score_thresh: Optional[float] = None,
) -> Path:
    """
    RGB + Thermal inference.

    1) For each RGB image, locate a thermal sidecar:
       - If thermal/pairs.json exists, use it (image_name → path)
       - Else try common locations (thermal/<stem>.tif, same folder, etc.)
    2) Form a BGRT 4-channel array (thermal scaled to uint8).
    3) Build a Detectron2 model, patch first conv to 4 channels, set PIXEL_MEAN/STD of length 4.
    4) Write one JSON per image under out_dir/preds/<stem>.json

    Returns: `out_dir` (with the standard results layout).
    """
    images_dir = Path(images_dir)
    weights_dir = Path(weights_dir)
    out_dir = Path(out_dir)

    paths = ensure_results_layout(out_dir)
    preds_dir = paths["preds"]

    # ---- Build config & predictor ----
    meta = _read_meta(weights_dir)
    num_classes = _resolve_num_classes(weights_dir, meta)
    score_thresh_eff = float(score_thresh) if score_thresh is not None else _resolve_score_thresh(meta, 0.5)
    device = _pick_device()

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.DEVICE = device
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = int(num_classes)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(score_thresh_eff)
    cfg.MODEL.WEIGHTS = str(_resolve_weights_path(weights_dir))

    # Make normalization 4-channel aware (BGRT). You can tweak thermal stats if you saved them in meta.
    # Here we keep RGB as Detectron defaults and use mid-gray for thermal as a neutral baseline.
    cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675, 128.0]  # B, G, R, T
    cfg.MODEL.PIXEL_STD  = [57.375, 57.120, 58.395, 58.0]

    # Anchor angles safeguard (axis-aligned)
    if hasattr(cfg.MODEL, "ANCHOR_GENERATOR") and hasattr(cfg.MODEL.ANCHOR_GENERATOR, "ANGLES"):
        cfg.MODEL.ANCHOR_GENERATOR.ANGLES = [[0]]

    predictor = DefaultPredictor(cfg)

    # Patch first conv to accept 4 channels (if not already patched at training)
    _patch_first_conv_to_4ch(predictor.model)

    # ---- pairing: pairs.json or heuristics ----
    pairs = _load_pairs_json(images_dir)

    # ---- Predict every image ----
    for img in _list_images(images_dir):
        # RGB
        bgr = cv2.imread(str(img), cv2.IMREAD_COLOR)
        if bgr is None:
            write_pred_json(preds_dir, img.stem, boxes_xyxy=[], scores=[], classes=[], extra={"file": img.name})
            continue

        # Thermal
        tpath = None
        if img.name in pairs:
            cand = Path(pairs[img.name])
            tpath = cand if cand.is_absolute() else (images_dir / cand)
        else:
            tpath = _guess_thermal_sidecar(images_dir, img)

        if not tpath or not tpath.exists():
            # no thermal; write empty preds to signal mismatch
            write_pred_json(preds_dir, img.stem, boxes_xyxy=[], scores=[], classes=[], extra={
                "file": img.name, "warning": "thermal_sidecar_not_found"
            })
            continue

        t8 = _load_thermal_channel(tpath)
        if t8 is None:
            write_pred_json(preds_dir, img.stem, boxes_xyxy=[], scores=[], classes=[], extra={
                "file": img.name, "warning": "thermal_load_failed"
            })
            continue

        # 4-channel BGRT
        bgrt = _bgrt_stack(bgr, t8)

        # DefaultPredictor expects HxWxC array; it will handle normalization
        outputs = predictor(bgrt)
        inst = outputs.get("instances", None)

        if inst is None or len(inst) == 0:
            boxes, scores, classes = [], [], []
        else:
            inst = inst.to("cpu")
            boxes   = inst.pred_boxes.tensor.numpy().tolist()
            scores  = inst.scores.numpy().tolist()
            classes = inst.pred_classes.numpy().tolist()

        write_pred_json(preds_dir, img.stem, boxes_xyxy=boxes, scores=scores, classes=classes, extra={"file": img.name})

    return out_dir
