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
        # "file" is the filename (used by the UI stitching code); prefer the base
        # name of the original image when available so downstream code can
        # match thermal sidecars and EXIF by name/stem.
        "file": (Path(getattr(r, "orig_img_path", getattr(r, "path", None))).name
                 if (getattr(r, "orig_img_path", getattr(r, "path", None)) is not None) else None),
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


def predict_folder(images_dir: Path, weights_dir: Path, out_dir: Path, score_thresh: float = 0.25, use_thermal: bool = False, channel_count: int = 3) -> Path:
    from ultralytics import YOLO
    from PIL import Image
    import numpy as np
    import time
    import torch
    from ...core.results import write_metrics_json

    def _find_thermal(rgb_path: Path) -> Path | None:
        # Robust thermal discovery that matches the decoder output (pairs.json)
        exts = (".png", ".tif", ".tiff", ".jpg", ".jpeg")
        tdir = rgb_path.parent / "thermal"
        stem = rgb_path.stem

        # 1) pairs.json mapping if present (preferred)
        try:
            pjson = tdir / "pairs.json"
            if pjson.exists():
                import json as _json
                try:
                    pairs = _json.loads(pjson.read_text(encoding="utf-8"))
                    target = pairs.get(rgb_path.name)
                    if target:
                        candidate = (rgb_path.parent / target).resolve()
                        if candidate.exists():
                            return candidate
                except Exception:
                    pass
        except Exception:
            pass

        # 2) thermal/ subfolder: decoder uses {stem}_thermal.tif — check that first
        for e in exts:
            cand1 = tdir / f"{stem}_thermal{e}"
            if cand1.exists():
                return cand1
            cand2 = tdir / f"{stem}{e}"
            if cand2.exists():
                return cand2

        # 3) sidecar next to RGB
        for e in exts:
            cand = rgb_path.with_name(f"{stem}_thermal{e}")
            if cand.exists():
                return cand

        return None

    model_weights = None
    # Accept multiple conventions: Detectron-style model_best.pth/model_final.pth
    # placed directly in the run folder, or Ultralytics-style best.pt/last.pt
    # possibly under a weights/ subfolder. Also accept passing a file path.
    w = Path(weights_dir)
    candidates = [
        w / "model_best.pt",
        w / "model_final.pt",
        w / "weights" / "best.pt",
        w / "weights" / "last.pt",
        w / "best.pt",
        w / "last.pt",
    ]

    for c in candidates:
        if c.exists():
            model_weights = str(c)
            break

    # fallback: maybe weights_dir itself is a file path
    if model_weights is None and w.is_file():
        model_weights = str(w)

    if model_weights is None:
        raise FileNotFoundError(f"No YOLO weights found in {weights_dir}")

    model = YOLO(model_weights)
    # mini-log header for the run
    try:
        tlog = logging.getLogger("pvrt.test")
        try:
            w_sz = Path(model_weights).stat().st_size if Path(model_weights).exists() else -1
            w_md5 = "n/a"
            try:
                import hashlib
                w_md5 = hashlib.md5(Path(model_weights).read_bytes()).hexdigest()[:8] if Path(model_weights).exists() else "missing"
            except Exception:
                w_md5 = "n/a"
        except Exception:
            w_sz, w_md5 = -1, "n/a"
        tlog.info(f"UI:OK:test: YOLO predict start | weights={Path(model_weights).name} md5={w_md5} | source={images_dir} | thr={score_thresh} | channels={channel_count}")
    except Exception:
        pass

    # ensure outputs
    run_dir = Path(out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # If thermal mode with channel variants requested, prepare a temporary folder
    source_dir = Path(images_dir)
    temp_prep = None
    if use_thermal and channel_count in (1, 4):
        temp_prep = run_dir / "predict_merged"
        temp_prep.mkdir(parents=True, exist_ok=True)
        # iterate images and create merged/synth images
        for p in sorted(Path(images_dir).iterdir()):
            if not p.is_file():
                continue
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}:
                continue
            t = _find_thermal(p)
            if t is None:
                # skip images without thermal when thermal required
                continue
            try:
                img = Image.open(p).convert("RGB")
                therm = Image.open(t).convert("L")
                a = np.array(therm).astype(np.float32)
                lo, hi = np.percentile(a, 2), np.percentile(a, 98)
                if hi <= lo: hi = lo + 1.0
                a = np.clip((a - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
                if channel_count == 4:
                    alpha = Image.fromarray(a, mode="L")
                    rgba = Image.merge("RGBA", (*img.split(), alpha))
                    out_path = temp_prep / f"{p.stem}.png"
                    rgba.save(out_path)
                else:
                    # channel_count == 1: create 3-channel grayscale from thermal
                    gray = Image.fromarray(a, mode="L")
                    rgb = Image.merge("RGB", (gray, gray, gray))
                    out_path = temp_prep / f"{p.stem}.png"
                    rgb.save(out_path)
            except Exception:
                continue
        source_dir = temp_prep

    # Safety: if the effective model expects 3 channels (no thermal) but
    # the source tiles/images may contain 4 channels (RGBA or 4-band TIFFs),
    # convert them to 3-channel RGB in a temporary folder so the Ultralytics
    # loader doesn't pass 4-channel arrays to a 3-channel model and raise
    # a channel-mismatch error. This mirrors Detectron's careful band
    # selection behavior.
    if channel_count != 4:
        temp_rgb = run_dir / "predict_rgb"
        # Only create/convert if it doesn't already exist to avoid rework.
        if not temp_rgb.exists():
            from PIL import Image
            temp_rgb.mkdir(parents=True, exist_ok=True)
            for p in sorted(source_dir.iterdir()):
                if not p.is_file():
                    continue
                if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}:
                    continue
                try:
                    # PIL's convert("RGB") will drop alpha or expand single-band
                    # images to RGB; it's a pragmatic choice here to ensure the
                    # YOLO model always sees 3-channel inputs when expected.
                    im = Image.open(p)
                    rgb = im.convert("RGB")
                    outp = temp_rgb / f"{p.stem}.png"
                    rgb.save(outp, format="PNG")
                except Exception:
                    # skip problematic files; YOLO will skip them later
                    continue
        source_dir = temp_rgb

    t0 = time.time()
    results = model.predict(
        source=str(source_dir),
        conf=score_thresh,
        imgsz=1024,
        device=0,
        save=False,
        save_txt=False,
    )

    # results is iterable per-image
    # canonical predictions directory expected by the web UI stitching code
    out_json_dir = run_dir / "preds"
    out_json_dir.mkdir(parents=True, exist_ok=True)

    # results is an iterable; log per-image counts to the mini-log for visibility
    # counters for metrics
    n_images = 0
    total_dets = 0
    images_with_dets = 0

    for r in results:
        # attempt to resolve path
        p = getattr(r, "orig_img_path", None) or getattr(r, "path", None) or None
        n_images += 1
        key = Path(p).name if p else f"img_{len(list(out_json_dir.iterdir()))}"
        js = _serialize_prediction(r)
        dets = len(js.get("boxes", []) or [])
        total_dets += dets
        if dets:
            images_with_dets += 1
        # ensure the JSON uses a simple filename (no folders) so the UI can
        # match by name/stem. Save as {stem}.json to align with other backends.
        out_path = out_json_dir / f"{Path(key).stem}.json"
        out_path.write_text(json.dumps(js, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            logging.getLogger("pvrt.test").info(f"UI:INFO:test: [{out_path.stem}] wrote pred json; boxes={len(js.get('boxes', []))}")
        except Exception:
            pass

        # Create a simple overlay PNG for UI browsing. If we prepared a
        # temp_prep (merged) folder for thermal runs, the prediction's
        # image path will point into that folder and we can extract the
        # thermal channel (alpha) when present. For 1-channel runs we
        # preserve a grayscale background (no falsecolor).
        try:
            from PIL import Image, ImageFont, ImageDraw
            import cv2
            import numpy as np

            # resolve image path
            if p is None:
                continue
            imgp = Path(p)
            if not imgp.exists():
                continue

            # For thermal-only runs prefer the actual thermal image in the
            # dataset (thermal/{stem}_thermal.*). Use that as the overlay
            # background so the UI shows the raw grayscale thermal images.
            im = cv2.imread(str(imgp), cv2.IMREAD_UNCHANGED)
            # default background
            bgr = None
            alpha = None
            has_rgb = False

            if channel_count == 1:
                # attempt to find the original RGB path and its thermal
                try:
                    rgb_candidate = Path(images_dir) / Path(key).name
                    tpath = _find_thermal(rgb_candidate)
                    if tpath is not None and tpath.exists():
                        # Prefer tifffile for float32/16 TIFFs
                        try:
                            import tifffile
                            timg = tifffile.imread(str(tpath))
                        except Exception:
                            timg = cv2.imread(str(tpath), cv2.IMREAD_UNCHANGED)
                        if timg is not None:
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
                            bgr = cv2.cvtColor(timg.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                            has_rgb = False
                except Exception:
                    pass

            if bgr is None:
                # fallback to the predicted image (merged/synth or original)
                if im is None:
                    continue
                if im.ndim == 3 and im.shape[2] == 4:
                    bgr = im[..., :3]
                    alpha = im[..., 3]
                    has_rgb = True
                else:
                    if im.ndim == 2:
                        bgr = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
                    else:
                        bgr = im
                    alpha = None
                    has_rgb = True

            H, W = bgr.shape[:2]

            # draw boxes on a copy
            vis = bgr.copy()
            boxes = js.get("boxes", [])
            scores = js.get("scores", [])
            classes = js.get("classes", [])
            for bi, sc, cl in zip(boxes, scores, classes):
                try:
                    x1, y1, x2, y2 = map(int, bi)
                except Exception:
                    continue
                color = (0, 0, 255)  # red BGR
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                label = f"{int(round(float(sc)*100))}%"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
                cv2.putText(vis, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

            overlays_dir = run_dir / "overlays"
            overlays_dir.mkdir(parents=True, exist_ok=True)
            # UI expects overlays/{stem}.png (no extra suffix)
            out_overlay = overlays_dir / f"{Path(key).stem}.png"
            cv2.imwrite(str(out_overlay), vis)
            try:
                logging.getLogger("pvrt.test").info(f"UI:INFO:test: [{out_overlay.name}] wrote overlay")
            except Exception:
                pass
        except Exception:
            # overlay generation is best-effort; ignore failures
            pass

    # finalize metrics
    elapsed = time.time() - t0
    device = "cpu"
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    metrics = {
        "backend": "yolo",
        "input_mode": "rgbt" if use_thermal else "rgb",
        "use_thermal": bool(use_thermal),
        "device": device,
        "score_thresh_test": float(score_thresh),
        "num_images": int(n_images),
        "images_with_detections": int(images_with_dets),
        "total_detections": int(total_dets),
        "avg_detections_per_image": round((total_dets / n_images), 3) if n_images else 0.0,
        "elapsed_sec": round(elapsed, 3),
        "img_per_sec": round((n_images / elapsed), 3) if elapsed > 0 else None,
        # model_name should point to the run folder (not the raw weight filename)
        # so the frontend can lookup the run's `model_meta.json` via /api/runs/<run>/meta
        "model_name": str(Path(weights_dir).name) if Path(weights_dir).is_dir() else str(Path(weights_dir).parent.name),
        "model_weights_file": str(Path(model_weights).name),
        "channel_count": int(channel_count),
    }

    try:
        write_metrics_json(run_dir, metrics)
    except Exception:
        log.exception("Failed to write metrics.json for YOLO run")

    return run_dir
