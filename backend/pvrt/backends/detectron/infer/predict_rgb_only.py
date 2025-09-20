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
from ....core.results import ensure_results_layout, write_pred_json, write_metrics_json, save_overlay_png

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
    try: return "cuda" if torch.cuda.is_available() else "cpu"
    except: return "cpu"

def _load_meta(d: Path) -> dict:
    p = d / "model_meta.json"
    if p.exists():
        try: return json.loads(p.read_text(encoding="utf-8"))
        except: pass
    return {}

def _resolve_weights(d: Path) -> Path:
    for n in ("model_best.pth", "model_final.pth", "model.pth"):
        p = d / n
        if p.exists(): return p
    return d / "model_final.pth"

def _cfg_like_before() -> "CfgNode":
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.MASK_ON = False
    return cfg

def _palette_bgr():
    return [(0,255,255),(255,0,255),(255,255,0),(0,128,255),(0,255,0),(255,0,0),(128,0,255),(0,0,255)]

def _draw_overlay(bgr, boxes, scores, classes, names):
    out = bgr.copy()
    H, W = out.shape[:2]
    pal = _palette_bgr()

    # thickness scales with image size (but never <2)
    thickness = max(2, int(round(min(H, W) * 0.003)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    text_thickness = 1
    pad = 4  # label padding inside the pill

    for bx, sc, cl in zip(boxes, scores, classes):
        if not bx:
            continue
        try:
            x1, y1, x2, y2 = map(int, bx)
        except Exception:
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

        # vivid color per class
        color = pal[int(cl) % len(pal)] if isinstance(cl, int) else pal[0]

        # draw the detection box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)

        # compute label box
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, text_thickness)
        pill_w = tw + 2 * pad
        pill_h = th + 2 * pad

        # default: place above the box; if not enough room, place below
        top = y1 - pill_h
        bottom = y1
        if top < 0:
            top = y1
            bottom = y1 + pill_h

        left = x1
        right = min(W - 1, x1 + pill_w)

        # translucent colored background ("pill")
        overlay = out.copy()
        cv2.rectangle(overlay, (left, top), (right, bottom), color, thickness=-1)
        cv2.addWeighted(overlay, 0.6, out, 0.4, 0.0, out)

        # text position
        tx = left + pad
        ty = bottom - pad if top >= y1 else y1 - pad  # account for above/below placement

        # text with black outline for contrast
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                cv2.putText(out, label, (tx + dx, ty + dy), font, font_scale,
                            (0, 0, 0), text_thickness, lineType=cv2.LINE_AA)
        cv2.putText(out, label, (tx, ty), font, font_scale,
                    (255, 255, 255), text_thickness, lineType=cv2.LINE_AA)

    return out


def predict_folder(images_dir, out_dir, weights_dir, use_thermal: bool=False) -> Path:
    log = _log()
    log.info("UI:INFO:test: Using model mode RGB only (3ch)")
    t0  = time.time()

    images_dir = Path(images_dir)
    out_dir    = Path(out_dir)
    weights    = Path(weights_dir)

    layout = ensure_results_layout(out_dir)      # {"root","preds","overlay"}
    preds_dir   = layout["preds"]
    overlay_dir = layout["overlay"]

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

    thr = meta.get("score_thresh_test", 0.6)
    try:    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(thr)
    except: cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6

    predictor = DefaultPredictor(cfg)

    # header log
    try:
        w_sz  = wpth.stat().st_size if wpth.exists() else -1
        w_md5 = hashlib.md5(wpth.read_bytes()).hexdigest()[:8] if wpth.exists() else "missing"
    except Exception:
        w_sz, w_md5 = -1, "n/a"

    exts = {".jpg",".jpeg",".png",".tif",".tiff",".bmp"}
    imgs = [p for p in sorted(images_dir.iterdir()) if p.suffix.lower() in exts]
    n    = len(imgs)
    log.info("UI:OK:test: Testing started")
    log.info(f"UI:INFO:test: Images={n} | Device={cfg.MODEL.DEVICE} | Thr={getattr(cfg.MODEL.ROI_HEADS,'SCORE_THRESH_TEST',None)} | WeightsMD5={w_md5}")
    log.info(f"UI:INFO:test: Using model: {weights}") 

    total, with_dets = 0, 0
    for i, p in enumerate(imgs, 1):
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            write_pred_json(preds_dir, p.stem, [], [], [], extra={"file": p.name, "reason":"read_failed"})
            log.info(f"UI:INFO:test: [{i}/{n}] {p.name}: 0 detections (read_failed)")
            continue

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

        write_pred_json(preds_dir, p.stem, boxes, scores, classes, extra={"file": p.name})
        overlay = _draw_overlay(bgr, boxes, scores, classes, names)
        save_overlay_png(overlay_dir, p.stem, overlay)   # PNG only, drawn BEFORE save

        log.info(f"UI:INFO:test: [{i}/{n}] {p.name}: {k} detections")

    elapsed = time.time() - t0
    metrics = {
        "backend":"detectron","input_mode":"rgb","use_thermal":False,"device":cfg.MODEL.DEVICE,
        "score_thresh_test": getattr(cfg.MODEL.ROI_HEADS,"SCORE_THRESH_TEST",None),
        "num_images": n, "images_with_detections": with_dets, "total_detections": total,
        "avg_detections_per_image": round(total/n, 3) if n else 0.0,
        "elapsed_sec": round(elapsed, 3),
        "img_per_sec": round(n/elapsed, 3) if elapsed>0 else None
    }
    write_metrics_json(out_dir, metrics)

    # ONE summary line + completion line
    log.info(f"UI:INFO:test: predictions_total={total}")
    # log.info("UI:OK:test: Test complete")
    return out_dir
