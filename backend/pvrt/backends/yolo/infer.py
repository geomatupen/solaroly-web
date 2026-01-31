"""Inference helper for YOLO using ultralytics.YOLO

Provides `predict_folder` which runs predictions on a folder of images and
saves per-image JSON results and annotated images under out_dir.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import json
import logging
from ultralytics import YOLO
from PIL import Image
import numpy as np
import time
import torch
import cv2
from ...core.results import write_metrics_json
from ...core.thermal import normalize_thermal

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
    boxes = getattr(r, "boxes", None)
    # Ultraytics/YOLO's Result.boxes often exposes batch tensors:
    #   r.boxes.xyxy -> (N,4), r.boxes.conf -> (N,), r.boxes.cls -> (N,)
    # or an iterable of Box objects. Handle both shapes robustly.
    try:
        if boxes is not None:
            # preferred path: per-field tensors
            xy = getattr(boxes, "xyxy", None)
            confs = getattr(boxes, "conf", None)
            clss = getattr(boxes, "cls", None)
            if xy is not None and hasattr(xy, "cpu"):
                arr_xy = xy.cpu().numpy()
                arr_conf = confs.cpu().numpy() if confs is not None and hasattr(confs, "cpu") else None
                arr_cls = clss.cpu().numpy() if clss is not None and hasattr(clss, "cpu") else None
                for i in range(arr_xy.shape[0]):
                    out["boxes"].append(arr_xy[i].astype(float).tolist())
                    out["scores"].append(float(arr_conf[i]) if arr_conf is not None else None)
                    out["classes"].append(int(arr_cls[i]) if arr_cls is not None else None)
            else:
                # fallback: iterable of box-like objects
                for b in boxes:
                    try:
                        xy = getattr(b, "xyxy", None)
                        conf = getattr(b, "conf", None)
                        cls = getattr(b, "cls", None)
                        xya = None
                        if xy is not None and hasattr(xy, "cpu"):
                            xya = xy.cpu().numpy().tolist()
                            # some wrappers return shape (1,4) per-box
                            if isinstance(xya, list) and len(xya) == 1 and isinstance(xya[0], list):
                                xya = xya[0]
                        elif hasattr(xy, "tolist"):
                            xya = xy.tolist()
                        if xya is None:
                            xya = None
                        cval = None
                        if conf is not None and hasattr(conf, "cpu"):
                            try:
                                cval = float(conf.cpu().numpy().tolist()[0])
                            except Exception:
                                try:
                                    cval = float(conf.cpu().numpy())
                                except Exception:
                                    cval = None
                        elif conf is not None:
                            try:
                                cval = float(conf)
                            except Exception:
                                cval = None
                        clsval = None
                        if cls is not None and hasattr(cls, "cpu"):
                            try:
                                clsval = int(cls.cpu().numpy().tolist()[0])
                            except Exception:
                                try:
                                    clsval = int(cls.cpu().numpy())
                                except Exception:
                                    clsval = None
                        elif cls is not None:
                            try:
                                clsval = int(cls)
                            except Exception:
                                clsval = None

                        out["boxes"].append(xya)
                        out["scores"].append(cval)
                        out["classes"].append(clsval)
                    except Exception:
                        # skip malformed box entries
                        continue
    except Exception:
        # worst-case fallback: leave boxes empty
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
        # Use canonical thermal extensions from core.io so JPG/PNG previews are accepted.
        # Use canonical thermal extensions but ignore TIFFs: expect decoder-produced previews (JPEG/PNG).
        from ...core.io import THERMAL_EXTS
        exts = tuple(sorted(e for e in THERMAL_EXTS if e not in {'.tif', '.tiff'}))
        if not exts:
            exts = (".png", ".jpg", ".jpeg")
        tdir = rgb_path.parent / "thermal"
        stem = rgb_path.stem

        # 1) pairs.json mapping if present (preferred)
        pjson = tdir / "pairs.json"
        if pjson.exists():
            pairs = json.loads(pjson.read_text(encoding="utf-8"))
            target = pairs.get(rgb_path.name)
            if target:
                candidate = (rgb_path.parent / target).resolve()
                if candidate.exists():
                    return candidate
        # 2) thermal/ subfolder: decoder writes preview files.
        #    Accept common variants produced by different decoders: e.g.
        #    {stem}_thermal.png, {stem}_thermal_preview.png, or {stem}.png
        for e in exts:
            cand_preview = tdir / f"{stem}_thermal_preview{e}"
            if cand_preview.exists():
                return cand_preview
            cand1 = tdir / f"{stem}_thermal{e}"
            if cand1.exists():
                return cand1
            cand2 = tdir / f"{stem}{e}"
            if cand2.exists():
                return cand2

        # 3) sidecar next to RGB: check both _thermal and _thermal_preview variants
        for e in exts:
            cand_preview = rgb_path.with_name(f"{stem}_thermal_preview{e}")
            if cand_preview.exists():
                return cand_preview
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
    tlog = logging.getLogger("pvrt.test")
    w_sz = Path(model_weights).stat().st_size if Path(model_weights).exists() else -1
    w_md5 = "n/a"
    import hashlib
    try:
        w_md5 = hashlib.md5(Path(model_weights).read_bytes()).hexdigest()[:8] if Path(model_weights).exists() else "missing"
    except (OSError, IOError):
        w_md5 = "n/a"
    tlog.info(f"UI:OK:test: YOLO predict start | weights={Path(model_weights).name} md5={w_md5} | source={images_dir} | thr={score_thresh} | channels=3")
    if use_thermal:
        tlog.info("UI:INFO:test: Using thermal grayscale images for testing (thermal)")
    # ensure outputs
    run_dir = Path(out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Use rotated transparent PNGs directly for inference (preserve transparency)
    source_dir = Path(images_dir)

    # Diagnostic logging: report how many images will be fed to the model and
    # a small sample of image sizes. This helps detect cases where the source
    # directory is empty or contains tiny/invalid files (common when thermal
    # previews were missing and placeholders were used).
    def _scan_source(dirp: Path, sample_n: int = 3):
        paths = [p for p in sorted(dirp.iterdir()) if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg'}]
        samples = []
        for p in paths[:sample_n]:
            try:
                with Image.open(p) as im:
                    samples.append({'path': Path(p).name if p else 'unknown', 'size': im.size, 'mode': im.mode})
            except Exception:
                samples.append({'path': Path(p).name if p else 'unknown', 'size': None, 'mode': None})
        return len(paths), samples

    src_count, src_samples = _scan_source(source_dir)
    logging.getLogger('pvrt.test').info(f"YOLO predict: source_dir={source_dir} images={src_count} sample={src_samples}")

    t0 = time.time()
    results = model.predict(
        source=str(source_dir),
        conf=score_thresh,
        imgsz=1024,
        device=0,
        save=False,
        save_txt=False,
    )

    # Debug: write a compact summary of the raw ultralytics Results so we can
    # see whether the model actually produced boxes (even if later steps
    # filtered or failed to serialize them). This helps distinguish "model
    # produced zero detections" from "we dropped/failed to extract boxes".
    try:
        raw_summary = []
        for r in results:
            entry = {}
            pth = getattr(r, "orig_img_path", getattr(r, "path", None))
            entry["file"] = Path(pth).name if pth is not None else None
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                entry["n_boxes"] = 0
                entry["top_conf"] = None
            else:
                # prefer tensor fields
                xy = getattr(boxes, "xyxy", None)
                confs = getattr(boxes, "conf", None)
                try:
                    if xy is not None and hasattr(xy, "cpu"):
                        n = int(xy.cpu().numpy().shape[0])
                    else:
                        # fallback: iterable length
                        n = len(list(boxes))
                except Exception:
                    n = 0
                entry["n_boxes"] = n
                try:
                    if confs is not None and hasattr(confs, "cpu"):
                        arr = confs.cpu().numpy()
                        entry["top_conf"] = float(arr.max()) if arr.size else None
                    else:
                        # try to collect confidences from iterable boxes
                        vals = []
                        for b in boxes:
                            c = getattr(b, "conf", None)
                            if c is None:
                                continue
                            try:
                                if hasattr(c, "cpu"):
                                    vals.append(float(c.cpu().numpy().tolist()[0]))
                                else:
                                    vals.append(float(c))
                            except Exception:
                                continue
                        entry["top_conf"] = max(vals) if vals else None
                except Exception:
                    entry["top_conf"] = None
            raw_summary.append(entry)
        try:
            (run_dir / "raw_results_summary.json").write_text(json.dumps(raw_summary, indent=2), encoding="utf-8")
            logging.getLogger("pvrt.test").info(f"YOLO predict: wrote raw_results_summary.json ({len(raw_summary)} entries)")
        except Exception:
            pass
    except Exception:
        logging.getLogger("pvrt.test").debug("YOLO predict: failed to write raw results summary")

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
        logging.getLogger("pvrt.test").info(f"UI:INFO:test: [{out_path.stem}] wrote pred json; boxes={len(js.get('boxes', []))}")
        # Create a simple overlay PNG for frontend browsing. If a
        # temp_prep (merged) folder was prepared for thermal runs, the
        # prediction's image path may point into that folder and the
        # thermal channel (alpha) can be extracted when present. For
        # single-channel thermal runs a grayscale background is used.
        # generate overlay (best-effort). Failures will propagate to callers.
        from PIL import Image, ImageFont, ImageDraw

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
        # If the input has alpha, preserve it
        if im is None:
            continue
        if im.ndim == 2:
            bgra = cv2.cvtColor(im, cv2.COLOR_GRAY2BGRA)
        elif im.shape[2] == 4:
            bgra = im.copy()
        elif im.shape[2] == 3:
            bgra = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
        else:
            continue

        H, W = bgra.shape[:2]

        # draw boxes on a copy
        vis = bgra.copy()
        boxes = js.get("boxes", [])
        scores = js.get("scores", [])
        classes = js.get("classes", [])
        for bi, sc, cl in zip(boxes, scores, classes):
            try:
                x1, y1, x2, y2 = map(int, bi)
            except (TypeError, ValueError):
                log.debug("yolo.infer: skipping malformed box %r", bi)
                continue
            color = (0, 0, 255, 255)  # red BGRA
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"{int(round(float(sc)*100))}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
            cv2.putText(vis, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255,255), 1, cv2.LINE_AA)

        overlays_dir = run_dir / "overlays"
        overlays_dir.mkdir(parents=True, exist_ok=True)
        out_overlay = overlays_dir / f"{Path(key).stem}.png"
        cv2.imwrite(str(out_overlay), vis)
        logging.getLogger("pvrt.test").info(f"UI:INFO:test: [{out_overlay.name}] wrote overlay (preserved transparency)")

    # finalize metrics
    elapsed = time.time() - t0
    device = "cpu"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    metrics = {
        "backend": "yolo",
        "input_mode": "thermal" if use_thermal else "rgb",
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

    write_metrics_json(run_dir, metrics)

    return run_dir
