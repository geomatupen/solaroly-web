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
from ....core.thermal import normalize_thermal
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
        logging.getLogger(_LOGGER_NAME).debug("pick_device probe failed")
        return "cpu"

def _load_meta(d: Path) -> dict:
    p = d / "model_meta.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
            logging.getLogger(_LOGGER_NAME).warning("failed to load model_meta.json: %s", e)
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
    pjson = tdir / "pairs.json"
    if pjson.exists():
        import json as _json
        pairs = _json.loads(pjson.read_text(encoding="utf-8"))
        target = pairs.get(rgb.name)
        if target:
            candidate = (rgb.parent / target).resolve()
            if candidate.exists():
                return candidate

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

    # 2) Sidecar thermal (previous behavior)
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


# Use shared normalization helper from core.thermal
# def _normalize_thermal(arr: np.ndarray) -> np.ndarray:
#     (replaced by normalize_thermal)

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

def _falsecolor(th01: np.ndarray) -> np.ndarray:
    th8 = np.clip(th01*255.0,0,255).astype(np.uint8)
    return cv2.applyColorMap(th8, cv2.COLORMAP_JET)

def _draw_overlay_rgbt(bgr: np.ndarray, th01: np.ndarray, boxes, scores, classes, names) -> np.ndarray:
    """Draw overlays using PIL for better text rendering (matching deleted _draw_overlays)"""
    from PIL import Image, ImageDraw, ImageFont
    
    # Create blended base image
    base_bgr = cv2.addWeighted(bgr, 0.5, _falsecolor(th01), 0.5, 0.0)
    H, W = base_bgr.shape[:2]
    
    # Convert to PIL
    rgb = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB)
    base = Image.fromarray(rgb)
    draw = ImageDraw.Draw(base)
    pal_rgb = _palette_rgb()
    
    thickness = max(1, int(round(min(H, W) * 0.003)))
    try:
        font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    
    pad = 4
    
    for bx, sc, cl in zip(boxes, scores, classes):
        if not bx:
            continue
        try:
            x1, y1, x2, y2 = map(int, bx)
        except (TypeError, ValueError):
            continue
        
        x1 = max(0, min(W - 1, x1))
        y1 = max(0, min(H - 1, y1))
        x2 = max(0, min(W - 1, x2))
        y2 = max(0, min(H - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        
        name = names[cl] if 0 <= cl < len(names) else f"cls_{cl}"
        label = f"{name} {int(round(float(sc)*100))}%"
        color = pal_rgb[cl % len(pal_rgb)]
        
        # Draw box outline
        draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
        
        # Compute label box
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th_txt = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th_txt = 40, 10
        
        pill_w = tw + 2 * pad
        pill_h = th_txt + 2 * pad
        
        top = y1 - pill_h if (y1 - pill_h) >= 0 else y1
        left = x1
        
        # Draw colored pill
        draw.rectangle([left, top, left + pill_w, top + pill_h], fill=color)
        
        # Draw text with shadow
        tx, ty = left + pad, top + pad
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            draw.text((tx + dx, ty + dy), label, fill=(0, 0, 0), font=font)
        draw.text((tx, ty), label, fill=(255, 255, 255), font=font)
    
    # Convert back to BGR
    out = cv2.cvtColor(np.array(base), cv2.COLOR_RGB2BGR)
    return out

def _draw_overlay(bgr, boxes, scores, classes, names):
    """Draw overlays on grayscale thermal image using PIL"""
    from PIL import Image, ImageDraw, ImageFont
    
    H, W = bgr.shape[:2]
    has_alpha = (bgr.ndim == 3 and bgr.shape[2] == 4)
    
    # Convert BGR/BGRA to RGB/RGBA for PIL
    if has_alpha:
        rgba = cv2.cvtColor(bgr, cv2.COLOR_BGRA2RGBA)
        base = Image.fromarray(rgba, mode='RGBA')
    else:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        base = Image.fromarray(rgb, mode='RGB')
    
    draw = ImageDraw.Draw(base)
    pal_rgb = _palette_rgb()
    
    thickness = max(1, int(round(min(H, W) * 0.003)))
    try:
        font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    
    pad = 4
    
    for bx, sc, cl in zip(boxes, scores, classes):
        if not bx:
            continue
        try:
            x1, y1, x2, y2 = map(int, bx)
        except (TypeError, ValueError):
            continue
        
        x1 = max(0, min(W - 1, x1))
        y1 = max(0, min(H - 1, y1))
        x2 = max(0, min(W - 1, x2))
        y2 = max(0, min(H - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        
        name = names[cl] if isinstance(cl, int) and 0 <= cl < len(names) else f"cls_{cl}"
        pct = int(round(float(sc) * 100))
        label = f"{name} {pct}%"
        color = pal_rgb[int(cl) % len(pal_rgb)] if isinstance(cl, int) else pal_rgb[0]
        
        # Draw box outline
        draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
        
        # Compute label box
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th_txt = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th_txt = 40, 10
        
        pill_w = tw + 2 * pad
        pill_h = th_txt + 2 * pad
        
        top = y1 - pill_h if (y1 - pill_h) >= 0 else y1
        left = x1
        
        # Draw colored pill
        draw.rectangle([left, top, left + pill_w, top + pill_h], fill=color)
        
        # Draw text with shadow
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
    except (TypeError, ValueError):
        nc = 0
    m_nc = int(meta.get("num_classes", 0) or 0)
    if nc <= 0 and m_nc > 0:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = m_nc

    thr = meta.get("score_thresh_test")
    if thr is not None:
        try:
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(score_thresh)
        except (TypeError, ValueError):
            logging.getLogger(_LOGGER_NAME).debug("invalid score_thresh provided: %r", score_thresh)

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
    except (TypeError, ValueError, RuntimeError) as e:
        logging.getLogger(_LOGGER_NAME).warning("failed to set model pixel stats: %s", e)

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

    # If model_mode is NOT rgbt but the caller requested thermal via the bridge, fallback earlier.
    if model_mode not in {"rgbt", "rgb+thermal", "thermal", "rgb_thermal", "4ch"}:
        # Model does not support thermal inputs; fallback to RGB predictor.
        log.warning(f"UI:WARN:test: thermal requested but model is RGB-only (model_mode={model_mode}); falling back to RGB")

        return run_rgb(
            images_dir=images_dir,
            out_dir=out_dir,
            weights_dir=weights_dir,
            use_thermal=False,
        )

    # Log which channel configuration will be used for inference
    # Detectron thermal predictor now assumes RGB+Thermal (4-channel) inputs.
    log.info("UI:INFO:test: Running RGB+Thermal (4-channel) inference")

    layout      = ensure_results_layout(out)
    preds_dir   = layout["preds"]
    overlays_dir= layout["overlays"]
    # canonical location for exact uint8 normalized thermal previews
    thermal_dir = layout.get("thermal", out / "thermal")

    # Build a 4-channel model (RGB + thermal). 1-channel thermal-only runs
    # are no longer supported; all thermal runs use RGB+thermal.
    model, cfg, names, wpth = _build_model_4ch(wdir, score_thresh)

    # one-time header: attempt to compute size and short md5; fallback on IO errors
    if wpth.exists():
        w_sz = wpth.stat().st_size
        try:
            w_md5 = hashlib.md5(wpth.read_bytes()).hexdigest()[:8]
        except (OSError, IOError):
            w_md5 = "n/a"
    else:
        w_sz, w_md5 = -1, "missing"

    exts = {".jpg",".jpeg",".png",".tif",".tiff",".bmp"}
    imgs = [p for p in sorted(images.iterdir()) if p.suffix.lower() in exts]
    n    = len(imgs)

    mode_str = "RGB+Thermal (4ch)"
    log.info(f"UI:OK:test: Testing started: {mode_str}")
    log.info(f"UI:INFO:test: Images={n} | Device={cfg.MODEL.DEVICE} | Thr={getattr(cfg.MODEL.ROI_HEADS, 'SCORE_THRESH_TEST', None)} | WeightsMD5={w_md5}")

    total, with_dets = 0, 0
    # ensure variables referenced after the loop exist even if no images processed
    inst = None
    masks = []
    for i, p in enumerate(imgs, 1):
        # unified reader: GeoTIFF band-4 thermal OR sidecar *_thermal.tif
        bgr, therm = _read_rgb_and_thermal_from_path(p)

        # For overlay drawing, try to load image with alpha channel (preserves transparency in PNG files)
        overlay_base = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if overlay_base is None or overlay_base.ndim < 3:
            overlay_base = bgr  # Fallback to BGR if alpha read fails

    # 4-channel model: require both RGB and thermal. Thermal-only runs
    # are no longer supported; we require both BGR and a thermal
        # source for RGB+Thermal inference.
        if bgr is None or therm is None:
            reason = "read_failed" if bgr is None else "no_thermal_source"
            write_pred_json(preds_dir, p.stem, [], [], [], extra={"file": p.name, "reason": reason})
            log.info(f"UI:INFO:test: [{i}/{n}] {p.name}: 0 detections ({reason})")
            continue
        H, W = bgr.shape[:2]
        th = normalize_thermal(therm).astype(np.float32) / 255.0
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
        # For runs where BGR may be missing, synthesize a visualization.
        # Use a grayscale background for single-band visualization instead of a
        # false-color map so the image appears as a single-channel gray image.
        # When RGB is available we blend falsecolor thermal for enhanced contrast.
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
                    except (ImportError, OSError):
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
                            except (TypeError, ValueError):
                                timg8 = np.clip(timg, 0, 255).astype(np.uint8)
                            timg = timg8
                        if timg.ndim == 3:
                            timg = cv2.cvtColor(timg, cv2.COLOR_BGR2GRAY)
                        gray = timg.astype(np.uint8)
                if gray is None:
                    gray = (np.clip(th * 255.0, 0, 255).astype(np.uint8))
                    bgr_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    # also save the exact uint8 normalized thermal preview for
                    # parity with training previews. Use the canonical filename
                    # <stem>.png inside the results thermal folder. Additionally
                    # write a small display-enhanced copy for UI browsing so
                    # thumbnails/overlays are more legible without changing
                    # the canonical training artifact.
                    try:
                        from ....core.thermal import enhance_preview_for_display
                        thermal_dir.mkdir(parents=True, exist_ok=True)
                        t_out = thermal_dir / f"{p.stem}.png"
                        cv2.imwrite(str(t_out), gray)
                        # write enhanced visual copy into a sibling "vis" folder
                        vis_dir = thermal_dir / "vis"
                        vis_dir.mkdir(parents=True, exist_ok=True)
                        vis_img = enhance_preview_for_display(gray)
                        cv2.imwrite(str(vis_dir / f"{p.stem}.png"), vis_img)
                    except Exception:
                        # non-fatal: we still produce overlays; don't crash inference
                        _log().debug("failed to save normalized or enhanced thermal preview for %s", p.name)
            except (FileNotFoundError, OSError, ValueError, TypeError):
                # fallback when reading the preferred thermal file fails
                gray = (np.clip(th * 255.0, 0, 255).astype(np.uint8))
                bgr_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            # draw simple box overlays on the actual thermal grayscale image
            overlay = _draw_overlay(bgr_vis, boxes, scores, classes, names)
        else:
            # color RGB + falsecolor thermal blend for RGB+Thermal runs
            overlay = _draw_overlay_rgbt(overlay_base, th, boxes, scores, classes, names)
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
        # mask AP not computed in this runner
        metrics["mask_ap"] = None
    write_metrics_json(out, metrics)

    log.info("predictions_total=%d", total)
    # log.info("OK:test: Test complete")
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
