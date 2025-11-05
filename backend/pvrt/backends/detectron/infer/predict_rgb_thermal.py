# backend/pvrt/infer/predict_rgb_thermal.py
from __future__ import annotations
import json, hashlib, logging, time
from pathlib import Path

import cv2
import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2 import model_zoo


import rasterio
from rasterio.enums import Resampling
from rasterio.crs import CRS as _RIO_CRS
RIO_OK_TH = True

# widen model to 4ch and extend pixel stats
from ..utils.model_patch import make_cfg_4ch, patch_first_conv_to_4ch
from ....core.io import load_model_meta, input_mode_from_meta, THERMAL_EXTS
from .predict_rgb_only import predict_folder as run_rgb

_LOGGER_NAME = "pvrt.test"
def _log() -> logging.Logger:
    lg = logging.getLogger(_LOGGER_NAME)
    if not lg.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(message)s"))
        lg.addHandler(h)
        lg.setLevel(logging.INFO)
    lg.propagate = False
    return lg

def _pick_device() -> str:
    try:
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

def _load_meta(d: Path) -> dict:
    p = d / "model_meta.json"
    if p.exists():
        try: return json.loads(p.read_text(encoding="utf-8"))
        except Exception: pass
    return {}

def _resolve_weights(d: Path) -> Path:
    for n in ("model_best.pth","model_final.pth","model.pth"):
        p = d / n
        if p.exists(): return p
    return d / "model_final.pth"

def _load_cfg(d: Path):
    cfg = get_cfg()
    yml = d / "config.yaml"
    if yml.exists():
        cfg.merge_from_file(str(yml)); cfg._pvrt_cfg_source = "run_config.yaml"
    else:
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg._pvrt_cfg_source = "fallback_frcnn_zoo"
    return cfg

def _find_thermal(rgb: Path) -> Path | None:
    """
    Locate a thermal source for the given RGB `rgb` path.

    Search order (robust against how the decoder writes files):
      1) If `thermal/pairs.json` exists, honor the mapping (preferred).
      2) Check `thermal/{stem}_thermal.*` and `thermal/{stem}.*` (decoder writes `_thermal.tif`).
      3) Check sidecar files next to the RGB image (`{stem}_thermal.*`).
      4) Legacy fallback: `{stem}_thermal.tif`/`.tiff` next to RGB.
    """
    # Use canonical thermal extensions from core.io (includes JPG/PNG and TIFF)
    exts = tuple(sorted(THERMAL_EXTS))
    tdir = rgb.parent / "thermal"

    # 1) pairs.json mapping (decoder writes relative paths there)
    try:
        pjson = tdir / "pairs.json"
        if pjson.exists():
            import json as _json
            try:
                pairs = _json.loads(pjson.read_text(encoding="utf-8"))
                target = pairs.get(rgb.name)
                if target:
                    candidate = (rgb.parent / target).resolve()
                    if candidate.exists():
                        return candidate
            except Exception:
                # ignore malformed pairs.json and fall through
                pass
    except Exception:
        pass

    # 2) thermal/ subfolder: check both {stem}_thermal.* (what decoder writes) and {stem}.*
    for e in exts:
        cand1 = tdir / f"{rgb.stem}_thermal{e}"
        if cand1.exists():
            return cand1
        cand2 = tdir / f"{rgb.stem}{e}"
        if cand2.exists():
            return cand2

    # 3) sidecar with _thermal suffix next to the RGB file
    for e in exts:
        cand = rgb.with_name(f"{rgb.stem}_thermal{e}")
        if cand.exists():
            return cand

    # 4) legacy sidecar fallback: accept TIFFs and common image previews
    for ext in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
        cand = rgb.with_name(f"{rgb.stem}_thermal{ext}")
        if cand.exists():
            return cand

    return None

def _read_rgb_and_thermal_from_path(p: Path):
    """
    Returns (bgr_u8, thermal_raw_or_u8_or_None).

    Priority:
      1) If file is GeoTIFF and has >=4 bands, read RGB from bands 1..3 and thermal from band 4.
      2) Else, read RGB via OpenCV and thermal from sidecar files (_thermal.*) as before.
    """
    # 1) In-band thermal (GeoTIFF)
    if RIO_OK_TH and p.suffix.lower() in (".tif", ".tiff"):
        try:
            with rasterio.open(p) as ds:
                nb = ds.count

                # Build RGB (favor bands 1,2,3 if available; fall back to single-band gray → 3ch)
                rgb_bands = [b for b in (1, 2, 3) if b <= nb]
                if len(rgb_bands) >= 1:
                    arr = ds.read(rgb_bands)                 # C,H,W
                    arr = arr.astype("float32")
                    arr = arr.transpose(1, 2, 0)             # H,W,C

                    # Robust 2–98% stretch per channel → uint8
                    out = np.empty_like(arr, dtype=np.uint8)
                    if out.ndim == 2:
                        arr = arr[..., None]; out = np.empty((*arr.shape[:2], 1), dtype=np.uint8)
                    for c in range(arr.shape[2]):
                        band = arr[..., c]
                        vals = band.reshape(-1)
                        if vals.size == 0:
                            out[..., c] = 0; continue
                        lo = np.percentile(vals, 2.0); hi = np.percentile(vals, 98.0)
                        if hi <= lo: hi = lo + 1.0
                        x = (band - lo) * (255.0/(hi-lo))
                        out[..., c] = np.clip(x, 0, 255).astype(np.uint8)
                    if out.shape[2] == 1:
                        out = np.repeat(out, 3, axis=2)
                    # Detectron code expects BGR
                    bgr = out[..., ::-1].copy()
                else:
                    bgr = None

                # Band-4 thermal if present
                therm = None
                if nb >= 4:
                    therm = ds.read(4)   # raw numeric array (any dtype); normalized later

                if bgr is not None:
                    return bgr, therm
        except Exception:
            # fall through to sidecar
            pass

    # 2) Sidecar thermal (your current behavior)
    bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
    therm = None
    th_path = _find_thermal(p)
    if th_path is not None:
        timg = cv2.imread(str(th_path), cv2.IMREAD_UNCHANGED)
        if timg is not None:
            if timg.ndim == 3:
                timg = cv2.cvtColor(timg, cv2.COLOR_BGR2GRAY)
            therm = timg
    return bgr, therm


def _normalize_thermal(arr: np.ndarray) -> np.ndarray:
    th = arr.astype(np.float32)
    if th.ndim == 3: th = th[...,0]
    vmax = float(np.nanmax(th)) if th.size else 0.0
    if vmax > 1.5:   # looks like °C
        th = np.clip(th, 0.0, 100.0) / 100.0
    else:            # already 0..1, or constant
        tmin, tmax = float(np.nanmin(th)), float(np.nanmax(th))
        th = (th - tmin) / (tmax - tmin) if tmax > tmin else th*0.0
    return th

def _palette_bgr():  # high contrast on false-color
    return [(0,255,255),(255,0,255),(255,255,0),(0,128,255),(0,255,0),(255,0,0),(128,0,255),(0,0,255)]

def _falsecolor(th01: np.ndarray) -> np.ndarray:
    th8 = np.clip(th01*255.0,0,255).astype(np.uint8)
    return cv2.applyColorMap(th8, cv2.COLORMAP_JET)

def _draw_overlay_rgbt(bgr: np.ndarray, th01: np.ndarray, boxes, scores, classes, names) -> np.ndarray:
    base = cv2.addWeighted(bgr, 0.5, _falsecolor(th01), 0.5, 0.0)
    pal  = _palette_bgr()
    for bx, sc, cl in zip(boxes, scores, classes):
        if not bx: continue
        x1,y1,x2,y2 = map(int, bx)
        name  = names[cl] if 0 <= cl < len(names) else f"cls_{cl}"
        label = f"{name} {int(round(float(sc)*100))}%"
        color = pal[cl % len(pal)]
        cv2.rectangle(base, (x1,y1), (x2,y2), color, 2)
        (tw,th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        bx2, by2 = x1+tw+8, y1-th-8
        if by2 < 0:
            cv2.rectangle(base, (x1,y1), (bx2,y1+th+8), color, -1)
            cv2.putText(base, label, (x1+4,y1+th+2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(base, (x1,y1), (bx2,by2),    color, -1)
            cv2.putText(base, label, (x1+4,y1-6),      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    return base

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

# use core helpers for layout / JSON / PNG save
from ....core.results import ensure_results_layout, write_pred_json, write_metrics_json, save_overlay_png

def _build_model_4ch(weights_dir: Path, score_thresh: float):
    device = _pick_device()
    meta   = _load_meta(weights_dir)
    cfg    = _load_cfg(weights_dir)

    wpth = _resolve_weights(weights_dir)
    cfg.MODEL.WEIGHTS = str(wpth)
    cfg.MODEL.DEVICE  = device

    try:
        nc = int(getattr(cfg.MODEL.ROI_HEADS, "NUM_CLASSES", 0) or 0)
    except Exception:
        nc = 0
    m_nc = int(meta.get("num_classes", 0) or 0)
    if nc <= 0 and m_nc > 0:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = m_nc

    thr = meta.get("score_thresh_test")
    if thr is not None:
        try:
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(score_thresh)
        except Exception:
            pass

    # --- IMPORTANT: make cfg 4ch BEFORE build_model ---
    make_cfg_4ch(cfg)  # extend PIXEL_MEAN/STD to 4ch (B,G,R,T)

    # build then widen conv1 to 4ch
    model = build_model(cfg)
    model.eval().to(device)

    # ensure model pixel stats are 4-channel tensors on the right device
    try:
        pm = torch.as_tensor(cfg.MODEL.PIXEL_MEAN, dtype=torch.float32, device=device).view(-1, 1, 1)
        ps = torch.as_tensor(cfg.MODEL.PIXEL_STD,  dtype=torch.float32, device=device).view(-1, 1, 1)
        model.pixel_mean = pm
        model.pixel_std  = ps
    except Exception:
        pass

    patch_first_conv_to_4ch(model)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    names = [str(x) for x in meta.get(
        "class_names",
        [f"cls_{i}" for i in range(int(getattr(cfg.MODEL.ROI_HEADS, "NUM_CLASSES", 0) or 0))]
    )]
    return model, cfg, names, wpth




def predict_folder(images_dir, weights_dir, out_dir, score_thresh: float = 0.5, channel_count: int = 4, **_) -> Path:
    """
    RGB+Thermal inference (sidecar <stem>_thermal.tif[f]).
    PNG overlays only, with class + %; live mini-logs; end summary.
    """
    log = _log()
    t0  = time.time()

    images = Path(images_dir)
    out    = Path(out_dir)
    wdir   = Path(weights_dir)

    meta = load_model_meta(Path(weights_dir))
    model_mode = input_mode_from_meta(meta, default="rgb").lower().strip()

    # If model_mode is NOT rgbt but user forced thermal via bridge, we fallback earlier.
    if model_mode not in {"rgbt", "rgb+thermal", "thermal", "rgb_thermal", "4ch"}:
        log.warning(
            f"UI:WARN:test: thermal path received RGB model (model_mode={model_mode!r}) - FALLBACK to RGB"
        )
        
        return run_rgb(
            images_dir=images_dir,
            out_dir=out_dir,
            weights_dir=weights_dir,
            use_thermal=False,
        )

    # Log which channel configuration will be used for inference
    # Detectron thermal predictor now assumes RGB+Thermal (4-channel) inputs.
    log.info("UI:INFO:test: Running the mode RGB+Thermal (4ch)")

    layout      = ensure_results_layout(out)
    preds_dir   = layout["preds"]
    overlays_dir= layout["overlays"]

    # Build a 4-channel model (RGB + thermal). 1-channel thermal-only runs
    # are no longer supported; all thermal runs use RGB+thermal.
    model, cfg, names, wpth = _build_model_4ch(wdir, score_thresh)

    # one-time header
    try:
        w_sz  = wpth.stat().st_size if wpth.exists() else -1
        w_md5 = hashlib.md5(wpth.read_bytes()).hexdigest()[:8] if wpth.exists() else "missing"
    except Exception:
        w_sz, w_md5 = -1, "n/a"

    exts = {".jpg",".jpeg",".png",".tif",".tiff",".bmp"}
    imgs = [p for p in sorted(images.iterdir()) if p.suffix.lower() in exts]
    n    = len(imgs)

    mode_str = "RGB+Thermal (4ch)"
    log.info(f"UI:OK:test: Testing started ({mode_str})")
    log.info(f"UI:INFO:test: Images={n} | Device={cfg.MODEL.DEVICE} | Thr={getattr(cfg.MODEL.ROI_HEADS,'SCORE_THRESH_TEST',None)} | WeightsMD5={w_md5}")

    total, with_dets = 0, 0
    # ensure variables referenced after the loop exist even if no images processed
    inst = None
    masks = []
    for i, p in enumerate(imgs, 1):
        # unified reader: GeoTIFF band-4 thermal OR sidecar *_thermal.tif
        bgr, therm = _read_rgb_and_thermal_from_path(p)

    # 4-channel model: require both RGB and thermal. Thermal-only runs
    # are no longer supported; we require both BGR and a thermal
        # source for RGB+Thermal inference.
        if bgr is None or therm is None:
            reason = "read_failed" if bgr is None else "no_thermal_source"
            write_pred_json(preds_dir, p.stem, [], [], [], extra={"file": p.name, "reason": reason})
            log.info(f"UI:INFO:test: [{i}/{n}] {p.name}: 0 detections ({reason})")
            continue
        H, W = bgr.shape[:2]
        th = _normalize_thermal(therm)
        if th.shape[:2] != (H, W):
            th = cv2.resize(th, (W, H), interpolation=cv2.INTER_LINEAR)
        ch4 = np.dstack([bgr.astype(np.float32), (th * 255.0)]).astype(np.float32)  # BGRT
        tensor = torch.as_tensor(ch4.transpose(2, 0, 1)).to(cfg.MODEL.DEVICE)

        # Run model
        with torch.no_grad():
            outs = model([{"image": tensor, "height": H, "width": W}])
        inst = outs[0].get("instances", None)
        inst = inst.to("cpu") if inst is not None else None

        if inst is None or len(inst) == 0:
            boxes, scores, classes, masks, polygons = [], [], [], [], []
        else:
            boxes   = inst.pred_boxes.tensor.numpy().tolist()
            scores  = inst.scores.numpy().tolist()
            classes = inst.pred_classes.numpy().tolist()
            # Mask extraction
            if hasattr(inst, "pred_masks"):
                masks = inst.pred_masks.cpu().numpy()
                # Convert masks to polygons (contours)
                polygons = []
                for mask in masks:
                    cnts, _ = cv2.findContours((mask > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    poly = [cnt.squeeze().tolist() for cnt in cnts if cnt.size > 0]
                    polygons.append(poly)
                masks = [mask.astype(np.uint8).tolist() for mask in masks]
            else:
                masks, polygons = [], []

        k = len(scores); total += k
        if k > 0: with_dets += 1

        write_pred_json(
            preds_dir, p.stem, boxes, scores, classes,
            extra={"file": p.name, "masks": masks, "polygons": polygons}
        )
        # For thermal-only runs where BGR may be missing, synthesize a visualization.
        # Use a grayscale background for single-band visualization instead of a
        # false-color map so the image appears as a single-channel gray image
        # (user requested). When RGB is available we still blend falsecolor
        # thermal for enhanced contrast.
        if bgr is None:
            # Prefer using the actual thermal file from the `thermal/` folder
            # (decoder produces grayscale TIFFs) so overlays match the raw
            # thermal images on disk. Fall back to normalized `th` if the
            # file can't be read.
            try:
                tpath = _find_thermal(p)
                gray = None
                if tpath is not None and tpath.exists():
                    # Prefer reading TIFFs with tifffile to preserve float32
                    try:
                        import tifffile
                        timg = tifffile.imread(str(tpath))
                    except Exception:
                        timg = cv2.imread(str(tpath), cv2.IMREAD_UNCHANGED)
                    if timg is not None:
                        # If float or higher-bit-depth, normalize to 0..255
                        if issubclass(timg.dtype.type, np.floating) or timg.dtype.itemsize > 1:
                            try:
                                mn = float(np.nanmin(timg))
                                mx = float(np.nanmax(timg))
                                if mx > mn:
                                    tnorm = (np.clip(timg, mn, mx) - mn) / (mx - mn)
                                else:
                                    tnorm = np.zeros_like(timg, dtype=np.float32)
                                timg8 = (np.clip(tnorm * 255.0, 0, 255)).astype(np.uint8)
                            except Exception:
                                timg8 = np.clip(timg, 0, 255).astype(np.uint8)
                            timg = timg8
                        if timg.ndim == 3:
                            timg = cv2.cvtColor(timg, cv2.COLOR_BGR2GRAY)
                        gray = timg.astype(np.uint8)
                if gray is None:
                    gray = (np.clip(th * 255.0, 0, 255).astype(np.uint8))
                bgr_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            except Exception:
                gray = (np.clip(th * 255.0, 0, 255).astype(np.uint8))
                bgr_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            # draw simple box overlays on the actual thermal grayscale image
            overlay = _draw_overlay(bgr_vis, boxes, scores, classes, names)
        else:
            # color RGB + falsecolor thermal blend for RGB+Thermal runs
            overlay = _draw_overlay_rgbt(bgr, th, boxes, scores, classes, names)
        save_overlay_png(overlays_dir, p.stem, overlay)
        log.info(f"UI:INFO:test: [{i}/{n}] {p.name}: {k} detections")

# ... write metrics + return

    elapsed = time.time() - t0
    metrics = {
        "backend": "detectron",
        "input_mode": "rgbt",
        "channel_count": 4,
        "use_thermal": True,
        "device": cfg.MODEL.DEVICE,
        "score_thresh_test": getattr(cfg.MODEL.ROI_HEADS, "SCORE_THRESH_TEST", None),
        "num_images": n,
        "images_with_detections": with_dets,
        "total_detections": total,
        "avg_detections_per_image": round(total / n, 3) if n else 0.0,
        "elapsed_sec": round(elapsed, 3),
        "img_per_sec": round(n / elapsed, 3) if elapsed > 0 else None,
        "model_name": str(weights_dir.name),
    }
    # Add mask AP if available
    if hasattr(inst, "pred_masks") and hasattr(inst, "scores") and hasattr(inst, "pred_classes"):
        # Placeholder: actual mask AP computation should be added here if available
        metrics["mask_ap"] = None  # TODO: replace with real mask AP if computed
    write_metrics_json(out, metrics)

    log.info(f"UI:INFO:test: predictions_total={total}")
    # log.info("UI:OK:test: Test complete")
    return out

def _draw_overlay_rgbt_with_masks(bgr, th01, boxes, scores, classes, masks, names):
    base = cv2.addWeighted(bgr, 0.5, _falsecolor(th01), 0.5, 0.0)
    pal  = _palette_bgr()
    H, W = base.shape[:2]
    # Draw masks first if present
    if masks:
        for idx, mask in enumerate(masks):
            color = pal[idx % len(pal)]
            mask_arr = np.array(mask, dtype=np.uint8)
            if mask_arr.shape != (H, W):
                mask_arr = cv2.resize(mask_arr, (W, H), interpolation=cv2.INTER_NEAREST)
            colored_mask = np.zeros_like(base)
            for c in range(3):
                colored_mask[..., c] = color[c]
            base = np.where(mask_arr[..., None] > 0, cv2.addWeighted(base, 0.5, colored_mask, 0.5, 0), base)
    # Draw boxes and labels as before
    return _draw_overlay_rgbt(base, th01, boxes, scores, classes, names)
