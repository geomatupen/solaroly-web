# backend/pvrt/backends/detectron/infer/predict_rgb_only.py
from __future__ import annotations
import json, os, hashlib, logging, time
from pathlib import Path

import cv2
import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo

# helpers
from ....core.results import ensure_results_layout, write_pred_json, write_metrics_json, save_overlay_png, save_overlay_jpg

_LOGGER = "pvrt.test"
def _log() -> logging.Logger:
    lg = logging.getLogger(_LOGGER)
    if not lg.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(message)s"))
        lg.addHandler(h)
        lg.setLevel(logging.INFO)
    lg.propagate = False  # avoid duplicates
    return lg

def _pick_device() -> str:
    try:
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

def _load_meta(d: Path) -> dict:
    p = d / "model_meta.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}

def _resolve_weights(d: Path) -> Path:
    for n in ("model_best.pth", "model_final.pth", "model.pth"):
        p = d / n
        if p.exists(): return p
    return d / "model_final.pth"

def _cfg_like_before():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.MASK_ON = False
    return cfg

def _draw_overlay(bgr, boxes, scores, classes, names):
    from PIL import Image, ImageDraw, ImageFont
    
    # Convert BGR to RGB/RGBA for PIL, preserving alpha if present
    H, W = bgr.shape[:2]
    has_alpha = (bgr.ndim == 3 and bgr.shape[2] == 4)
    
    if has_alpha:
        # BGRA to RGBA
        rgba = cv2.cvtColor(bgr, cv2.COLOR_BGRA2RGBA)
        base = Image.fromarray(rgba, mode='RGBA')
    else:
        # BGR to RGB
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        base = Image.fromarray(rgb, mode='RGB')
    
    draw = ImageDraw.Draw(base)
    pal_rgb = _palette_rgb()
    
    # thickness scales with image size
    thickness = max(1, int(round(min(H, W) * 0.003)))
    try:
        font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    
    pad = 4  # label padding inside the pill

    for bx, sc, cl in zip(boxes, scores, classes):
        if not bx:
            continue
        try:
            x1, y1, x2, y2 = map(int, bx)
        except (TypeError, ValueError) as e:
            logging.getLogger("pvrt").debug("skipping malformed bbox %r: %s", bx, e)
            continue

        # clamp to image bounds
        x1 = max(0, min(W - 1, x1))
        y1 = max(0, min(H - 1, y1))
        x2 = max(0, min(W - 1, x2))
        y2 = max(0, min(H - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        # label text
        name = names[cl] if isinstance(cl, int) and 0 <= cl < len(names) else f"cls_{cl}"
        pct = int(round(float(sc) * 100))
        label = f"{name} {pct}%"

        # vivid color per class (RGB for PIL)
        color_rgb = pal_rgb[int(cl) % len(pal_rgb)] if isinstance(cl, int) else pal_rgb[0]

        # draw the detection box outline
        draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=thickness)

        # compute label box
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th_txt = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th_txt = 40, 10  # fallback
        
        pill_w = tw + 2 * pad
        pill_h = th_txt + 2 * pad

        # default: place above the box; if not enough room, place below
        top = y1 - pill_h if (y1 - pill_h) >= 0 else y1
        left = x1
        
        # draw colored pill background
        draw.rectangle([left, top, left + pill_w, top + pill_h], fill=color_rgb)
        
        # draw text with black shadow
        tx, ty = left + pad, top + pad
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            draw.text((tx + dx, ty + dy), label, fill=(0, 0, 0), font=font)
        draw.text((tx, ty), label, fill=(255, 255, 255), font=font)

    # Convert back to BGR/BGRA, preserving alpha
    if has_alpha:
        out = cv2.cvtColor(np.array(base), cv2.COLOR_RGBA2BGRA)
    else:
        out = cv2.cvtColor(np.array(base), cv2.COLOR_RGB2BGR)
    return out

def _palette_rgb():
    """RGB palette (for PIL)"""
    return [
        (255, 0, 0),     # red
        (0, 170, 255),   # cyan
        (0, 200, 0),     # green
        (255, 0, 200),   # magenta
        (255, 165, 0),   # orange
        (128, 0, 255),   # purple
        (0, 255, 255),   # yellow
        (255, 255, 0),   # yellow-green
    ]



def predict_folder(images_dir, weights_dir, out_dir, score_thresh: float = 0.5) -> Path:
    log = _log()
    log.info("UI:INFO:test: Using model mode RGB only (3ch)")
    t0  = time.time()

    images_dir = Path(images_dir)
    out_dir    = Path(out_dir)
    weights    = Path(weights_dir)

    layout = ensure_results_layout(out_dir)      # {"root","preds","overlays"}
    preds_dir   = layout["preds"]
    overlay_dir = layout["overlays"]

    meta = _load_meta(weights)
    cfg  = _cfg_like_before()
    wpth = _resolve_weights(weights)
    cfg.MODEL.WEIGHTS = str(wpth)
    cfg.MODEL.DEVICE  = _pick_device()

    nclasses = int(meta.get("num_classes", 0) or 0)
    if nclasses > 0:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = nclasses
        names = [str(x) for x in meta.get("class_names", [f"cls_{i}" for i in range(nclasses)])]
    else:
        names = [f"cls_{i}" for i in range(getattr(cfg.MODEL.ROI_HEADS,"NUM_CLASSES",0) or 0)]

    # thr = meta.get("score_thresh_test", 0.6)
    try:
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(score_thresh)
    except (TypeError, ValueError):
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6

    predictor = DefaultPredictor(cfg)

    # header log: compute file size and a short MD5 when possible
    if wpth.exists():
        w_sz = wpth.stat().st_size
        try:
            w_md5 = hashlib.md5(wpth.read_bytes()).hexdigest()[:8]
        except (OSError, IOError):
            w_md5 = "n/a"
    else:
        w_sz, w_md5 = -1, "missing"

    exts = {".jpg",".jpeg",".png",".tif",".tiff",".bmp"}
    imgs = [p for p in sorted(images_dir.iterdir()) if p.suffix.lower() in exts]
    n    = len(imgs)
    log.info("UI:OK:test: Testing started")
    log.info(f"UI:INFO:test: Images={n} | Device={cfg.MODEL.DEVICE} | Thr={getattr(cfg.MODEL.ROI_HEADS,'SCORE_THRESH_TEST',None)} | WeightsMD5={w_md5}")
    log.info(f"UI:INFO:test: Using model: {weights}") 

    total, with_dets = 0, 0
    # ensure variables referenced after the loop exist even if no images processed
    inst = None
    masks = []
    for i, p in enumerate(imgs, 1):
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            write_pred_json(preds_dir, p.stem, [], [], [], extra={"file": p.name, "reason":"read_failed"})
            log.info(f"UI:INFO:test: [{i}/{n}] {p.name}: 0 detections (read_failed)")
            continue

        # For overlay drawing, try to load image with alpha channel (preserves transparency in PNG files)
        overlay_base = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if overlay_base is None or overlay_base.ndim < 3:
            overlay_base = bgr  # Fallback to BGR if alpha read fails
        
        out = predictor(bgr)
        inst = out.get("instances")
        inst = inst.to("cpu") if inst is not None else None

        if inst is None or len(inst) == 0:
            boxes, scores, classes = [], [], []
        else:
            boxes   = inst.pred_boxes.tensor.numpy().tolist()
            scores  = inst.scores.numpy().tolist()
            classes = inst.pred_classes.numpy().tolist()

        k = len(scores); total += k;  with_dets += int(k>0)

        write_pred_json(
            preds_dir, p.stem, boxes, scores, classes,
            extra={"file": p.name}
        )
        overlay = _draw_overlay(overlay_base, boxes, scores, classes, names)
        save_overlay_png(overlay_dir, p.stem, overlay)   # PNG preserves RGBA/alpha channel
        # save_overlay_jpg(overlay_dir, p.stem, overlay, exif_source=images_dir)

        log.info(f"UI:INFO:test: [{i}/{n}] {p.name}: {k} detections")

    elapsed = time.time() - t0
    # training_meta_dir = os.path.join(weights, "model_meta.json")
    # log.info(f"UI:INFO:test: model path={weights.name}")
    metrics = {
        "backend": "detectron",
        "input_mode": "rgb",
        "use_thermal": False,
        "device": cfg.MODEL.DEVICE,
        "score_thresh_test": getattr(cfg.MODEL.ROI_HEADS, "SCORE_THRESH_TEST", None),
        "num_images": n,
        "images_with_detections": with_dets,
        "total_detections": total,
        "avg_detections_per_image": round(total / n, 3) if n else 0.0,
        "elapsed_sec": round(elapsed, 3),
        "img_per_sec": round(n / elapsed, 3) if elapsed > 0 else None,
        "model_name": str(weights.name),
    }
    write_metrics_json(out_dir, metrics)

    # ONE summary line + completion line
    log.info(f"UI:INFO:test: predictions_total={total}")
    log.info("UI:OK:test: Test complete")
    return out_dir
