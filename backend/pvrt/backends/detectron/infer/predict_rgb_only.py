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
    out = bgr.copy(); pal = _palette_bgr()
    for bx, sc, cl in zip(boxes, scores, classes):
        if not bx: continue
        x1,y1,x2,y2 = map(int, bx)
        name  = names[cl] if 0 <= cl < len(names) else f"cls_{cl}"
        label = f"{name} {int(round(float(sc)*100))}%"
        color = pal[cl % len(pal)]
        cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
        (tw,th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        bx2, by2 = x1+tw+8, y1-th-8
        if by2 < 0:
            cv2.rectangle(out, (x1,y1), (bx2,y1+th+8), color, -1)
            cv2.putText(out, label, (x1+4,y1+th+2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1, cv2.LINE_AA)
        else:
            cv2.rectangle(out, (x1,y1), (bx2,by2),    color, -1)
            cv2.putText(out, label, (x1+4,y1-6),      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1, cv2.LINE_AA)
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
