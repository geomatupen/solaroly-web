# backend/pvrt/web/app.py
from __future__ import annotations

import asyncio
import io
import json
import logging
import sys
import shutil
import zipfile
import re
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple, Dict

from fastapi import FastAPI, Form, UploadFile, File, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import StreamingResponse

from PIL import Image
import numpy as np

# --- Optional tiler deps (best-effort) ---
try:
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
    import mercantile
    from pyproj import Transformer
    RIO_OK = True
except Exception:
    RIO_OK = False

# --- Backend-agnostic bridge and registry ---
from ..core.registry import register_backend
from .bridge import train_entry, predict_entry
from ..backends.detectron.backend import register as register_detectron

# --- SSE/logging bridge (your existing file) ---
from .sse import LogBroker, SSELogHandler, set_event_loop, sse_response

# --- Reuse your data helpers (RJPEG decode & scanning) ---
from ..dataops.scan_decode_split import (
    ensure_dirp_init, scan_split_decode_thermal, # safe to call only if thermal requested
)

# ---------------- Paths & constants ----------------
ROOT = Path(__file__).resolve().parents[2]        # .../backend/pvrt
PROJECT_ROOT = ROOT.parent                         # repo root
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
VALID_DIR = DATA_DIR / "valid"
TEST_DIR  = DATA_DIR / "test"

OUTPUTS   = PROJECT_ROOT / "outputs" / "runs"      # per-run outputs (weights, meta, logs)
OUTPUTS.mkdir(parents=True, exist_ok=True)

FRONTEND_DIR = PROJECT_ROOT / "frontend"
MEDIA_DIR    = PROJECT_ROOT / "media"              # browsable artifacts (thumbs, overlays, geojson)
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".jpg",".jpeg",".png",".tif",".tiff",".bmp",".webp",".JPG",".JPEG",".PNG",".TIF",".TIFF",".BMP",".WEBP"}

# --------------- FastAPI app & CORS ----------------
app = FastAPI(title="PVRT API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# --------------- Logging / SSE ----------------
broker = LogBroker()
sse_handler = SSELogHandler(broker)
sse_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S"))
sse_handler.setLevel(logging.INFO)

logger = logging.getLogger("pvrt")
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.addHandler(sse_handler)

# --------------- Backend registration ----------------
register_detectron(register_backend)  # Detectron now; add YOLO later by registering it here.

# --------------- Cancel flag (best-effort) ----------------
CANCEL_FLAGS: Dict[str, bool] = {"train": False}

# ================== small, reusable utils ==================

class _StreamToLogger:
    """Redirect print()/stdout/err to our logger (so they reach SSE)."""
    def __init__(self, level=logging.INFO):
        self.level = level
        self._buf = []

    def write(self, msg: str):
        msg = str(msg)
        if not msg:
            return
        # Split on lines so SSE consumers see live flow
        for line in msg.splitlines():
            line = line.strip()
            if line:
                logging.getLogger("pvrt").log(self.level, line)

    def flush(self):
        return

@contextmanager
def redirect_std_to_logger():
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = _StreamToLogger(logging.INFO)
    sys.stderr = _StreamToLogger(logging.INFO)
    try:
        yield
    finally:
        try:
            sys.stdout.flush(); sys.stderr.flush()
        except Exception:
            pass
        sys.stdout, sys.stderr = old_out, old_err

def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def _safe_name(name: str | None) -> str:
    if not name:
        return ""
    name = name.strip()[:128]
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-._")

def _is_image(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS

def _read_model_meta(run_dir: Path) -> dict:
    meta = run_dir / "model_meta.json"
    if meta.exists():
        try:
            return json.loads(meta.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _list_models() -> List[str]:
    models = []
    if OUTPUTS.exists():
        for d in sorted(OUTPUTS.iterdir()):
            if d.is_dir() and (d / "model_meta.json").exists():
                # out.append(p.name)
                meta = _read_model_meta(d)
                models.append({
                    "name": d.name,
                    "mtime": int(d.stat().st_mtime),
                    "input_mode": meta.get("input_mode", "rgb")
                })
    return models

def _unique_dataset_dir(base_name: str) -> Path:
    d = TEST_DIR / base_name
    if not d.exists():
        return d
    i = 1
    while True:
        cand = TEST_DIR / f"{base_name}-{i}"
        if not cand.exists():
            return cand
        i += 1

def _save_zip_as_dataset(zip_file: UploadFile, preferred_base: str | None = None) -> Path:
    base = _safe_name(preferred_base or Path(zip_file.filename or "dataset.zip").stem) or "dataset"
    ds_dir = _unique_dataset_dir(base)
    ds_dir.mkdir(parents=True, exist_ok=True)

    buf = zip_file.file.read()
    with zipfile.ZipFile(io.BytesIO(buf)) as zf:
        zf.extractall(ds_dir)

    # flatten one-level if typical zip has top folder
    kids = list(ds_dir.iterdir())
    if len(kids) == 1 and kids[0].is_dir():
        inner = kids[0]
        for p in inner.iterdir():
            shutil.move(str(p), str(ds_dir))
        inner.rmdir()
    return ds_dir

def _save_images_as_dataset(files: List[UploadFile], preferred_base: str | None = None) -> Path:
    base = _safe_name(preferred_base or "images") or "images"
    ds_dir = _unique_dataset_dir(base)
    ds_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        if not f.filename:
            continue
        ext = Path(f.filename).suffix.lower()
        if ext in IMAGE_EXTS:
            (ds_dir / Path(f.filename).name).write_bytes(f.file.read())
    return ds_dir

# ----------------- overlays & geojson -----------------
# Keep these simple and library-light; they operate on predictor JSON files.

def _coerce_pred_json(p: Path) -> dict:
    try:
        j = json.loads(p.read_text(encoding="utf-8"))
        # normalize keys used by front-end (boxes: [x1,y1,x2,y2], scores, classes)
        j.setdefault("boxes", [])
        j.setdefault("scores", [])
        j.setdefault("classes", [])
        j.setdefault("file", p.stem)
        return j
    except Exception:
        return {"file": p.stem, "boxes": [], "scores": [], "classes": []}

def _draw_overlays(images_dir: Path, preds_dir: Path, out_root: Path, class_names: List[str]) -> Tuple[Path, Path, Path]:
    """
    Produce /overlays (prefer predictor-colored overlays if present) and /thumbs under out_root,
    plus a manifest JSON mapping original file name → generated URLs.
    """
    # local-only imports so you don't have to change module imports
    import json, shutil, logging
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    logger   = logging.getLogger("pvrt.test")
    overlays = out_root / "overlays"; overlays.mkdir(parents=True, exist_ok=True)
    thumbs   = out_root / "thumbs";   thumbs.mkdir(parents=True, exist_ok=True)
    manifest = out_root / "manifest.json"

    # --- 1) Prefer existing colored overlays produced by predictors ---
    #     (predictors save to <session>/overlay/<stem>.png)
    colored_src = out_root / "overlay"  # singular
    use_colored = colored_src.exists() and any(colored_src.glob("*.png"))
    if use_colored:
        logger.info(f"UI:INFO:post: using existing colored overlays from {colored_src}")
    else:
        logger.info("UI:INFO:post: no predictor overlays found → drawing fallback overlays")

    # simple, vivid RGB palette (works on RGB + thermal backgrounds)
    def _palette_rgb():
        return [
            (255, 0, 0), (0, 170, 255), (0, 200, 0), (255, 0, 200),
            (255, 165, 0), (128, 0, 255), (0, 255, 255), (255, 255, 0),
        ]

    mapper: Dict[str, Dict[str, str]] = {}

    for img in sorted(images_dir.iterdir()):
        if not _is_image(img):
            continue

        stem = img.stem
        ov   = overlays / f"{stem}.png"
        th   = thumbs   / f"{stem}.png"

        if use_colored and (colored_src / f"{stem}.png").exists():
            # --- Reuse predictor overlay (copy into /overlays to keep structure consistent)
            src = colored_src / f"{stem}.png"
            try:
                shutil.copyfile(src, ov)
                im_for_thumb = Image.open(ov).convert("RGB")
            except Exception:
                # if copy/open fails, fall back to drawing
                pass
        if not ov.exists():
            # --- 2) Fallback: draw colored boxes + label using JSON preds ---
            pred_json = preds_dir / "preds" / f"{stem}.json"
            jj = _coerce_pred_json(pred_json) if pred_json.exists() else {
                "boxes": [], "scores": [], "classes": [], "file": img.name
            }

            try:
                base = Image.open(img).convert("RGB")
            except Exception:
                base = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))

            draw = ImageDraw.Draw(base)
            W, H = base.size
            pal = _palette_rgb()

            boxes   = jj.get("boxes", []) or []
            scores  = jj.get("scores", []) or []
            classes = jj.get("classes", []) or []

            # scale thickness and text size to image size
            thickness = max(1, int(round(min(W, H) * 0.003)))
            try:
                font = ImageFont.truetype("arial.ttf", size=max(3.5, int(min(W, H) * 0.0015)))
                logger.info(f"using font: {font}")
            except Exception:
                font = ImageFont.load_default()
                logger.info("using default font.")

            for i, b in enumerate(boxes):
                try:
                    x1, y1, x2, y2 = map(int, b)
                except Exception:
                    continue
                if x2 <= x1 or y2 <= y1:
                    continue

                cls_id = classes[i] if i < len(classes) else 0
                name   = class_names[cls_id] if 0 <= cls_id < len(class_names) else f"cls_{cls_id}"
                sc     = float(scores[i]) if i < len(scores) else 0.0
                label  = f"{name} {int(round(sc * 100))}%"

                color  = pal[cls_id % len(pal)]
                # rectangle outline
                draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)

                # label pill (solid color for clarity)
                # measure text
                try:
                    # Pillow >= 8.x
                    bbox = draw.textbbox((0, 0), label, font=font)
                    tw, th_txt = bbox[2] - bbox[0], bbox[3] - bbox[1]
                except Exception:
                    tw, th_txt = draw.textsize(label, font=font)
                pad = 4
                pill_w = tw + 2 * pad
                pill_h = th_txt + 2 * pad

                # place above box if space, else inside/top
                top = y1 - pill_h if (y1 - pill_h) >= 0 else y1
                left = x1
                # background
                draw.rectangle([left, top, left + pill_w, top + pill_h], fill=color)
                # text (white) with a thin black shadow for contrast
                tx, ty = left + pad, top + pad
                for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
                    draw.text((tx + dx, ty + dy), label, fill=(0, 0, 0), font=font)
                draw.text((tx, ty), label, fill=(255, 255, 255), font=font)

            base.save(ov, format="PNG", optimize=True)
            im_for_thumb = base

        # --- Thumb from overlay (colored or fallback) ---
        try:
            w, h = im_for_thumb.size
            tw = max(96, w // 6); thh = max(96, h // 6)
            im_thumb = im_for_thumb.resize((tw, thh))
            im_thumb.save(th, format="PNG", optimize=True)
        except Exception:
            Image.fromarray(np.zeros((96, 96, 3), dtype=np.uint8)).save(th, format="PNG", optimize=True)

        mapper[img.name] = {
            "overlay": f"/media/{ov.relative_to(MEDIA_DIR).as_posix()}" if str(ov).startswith(str(MEDIA_DIR)) else ov.name,
            "thumb":   f"/media/{th.relative_to(MEDIA_DIR).as_posix()}" if str(th).startswith(str(MEDIA_DIR)) else th.name,
        }

    manifest.write_text(json.dumps(mapper, indent=2), encoding="utf-8")
    return overlays, thumbs, manifest


def _preds_to_geojson(images_dir: Path, preds_dir: Path, out_root: Path, class_names: List[str]) -> Tuple[Path, List[Tuple[float,float]]]:
    """
    Build a very simple GeoJSON with image centers as points (best-effort).
    This is intentionally lightweight since many images have no geotags.
    """
    gj = {
        "type": "FeatureCollection",
        "features": []
    }
    centers: List[Tuple[float, float]] = []

    for img in sorted(images_dir.iterdir()):
        if not _is_image(img):
            continue
        # we don't rely on EXIF here; downstream map can still show image footprint later if available
        centers.append((0.0, 0.0))
        gj["features"].append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
            "properties": {"image": img.name}
        })

    out = out_root / "preds.geojson"
    out.write_text(json.dumps(gj, indent=2), encoding="utf-8")
    return out, centers

# ---------- tiny list helpers (datasets/models/sessions) ----------

def _count_top_level_images(d: Path) -> int:
    """Count images directly under d (non-recursive)."""
    if not d.exists() or not d.is_dir():
        return 0
    return sum(1 for p in d.iterdir() if p.is_file() and _is_image(p))

def _list_datasets() -> list[dict]:
    """
    Return detailed dataset info from data/test/.
    Shape: [{"name": "<folder>", "count": <n>, "mtime": <unix-ts>}, ...]
    """
    if not TEST_DIR.exists():
        return []
    items = []
    for p in sorted([x for x in TEST_DIR.iterdir() if x.is_dir()],
                    key=lambda x: x.stat().st_mtime, reverse=True):
        items.append({
            "name": p.name,
            "count": _count_top_level_images(p),
            "mtime": int(p.stat().st_mtime),
        })
    return items


def _list_sessions() -> list[dict]:
    """
    Return detailed sessions from media/sessions/.
    Shape: [{"name": "<session-id>", "mtime": <unix-ts>}, ...]
    """
    base = MEDIA_DIR / "sessions"
    if not base.exists():
        return []
    items = []
    for p in sorted([x for x in base.iterdir() if x.is_dir()],
                    key=lambda x: x.stat().st_mtime, reverse=True):
        items.append({
            "name": p.name,                 # normalized to just the id
            "mtime": int(p.stat().st_mtime)
        })
    return items

# ---------- wire all relevant loggers to SSE ----------
def _wire_logging_to_sse() -> None:
    """
    Stream our logs (pvrt.*) and selected 3rd-party logs to SSE exactly once.
    - 'pvrt' has the SSE handler and does NOT propagate to root.
    - children (pvrt.test, etc.) propagate to 'pvrt' (no handlers of their own).
    - detectron2/fvcore/torch attach SSE handler directly and do NOT propagate.
    - uvicorn* do not propagate to avoid double printing.
    """
    # parent already has sse_handler, ensure one handler only
    parent = logging.getLogger("pvrt")
    parent.setLevel(logging.INFO)
    parent.propagate = False
    parent.handlers = [h for h in parent.handlers if not isinstance(h, SSELogHandler)]
    parent.addHandler(sse_handler)

    # children bubble up to 'pvrt' (no own handlers)
    child = logging.getLogger("pvrt.test")
    child.handlers = []
    child.setLevel(logging.INFO)
    child.propagate = True

    # 3rd-party: attach handler directly, no propagation (no duplicates)
    for name in ("detectron2", "fvcore", "fvcore.common.checkpoint", "torch"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.INFO)
        lg.propagate = False
        lg.handlers = [h for h in lg.handlers if not isinstance(h, SSELogHandler)]
        lg.addHandler(sse_handler)

    # keep uvicorn logs out of the SSE stream (or set to True if you want them)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", ""):  # '' is root
        logging.getLogger(name).propagate = False



def _session_assets(session_dir: Path) -> dict:
    imgs_dir = session_dir / "images"
    overlays = session_dir / "overlays"
    thumbs   = session_dir / "thumbs"
    def _urls(d: Path):
        if not d.exists(): return []
        return [f"/media/{d.relative_to(MEDIA_DIR)}/{p.name}" for p in sorted(d.glob("*")) if p.is_file()]
    tifs = [u for u in _urls(imgs_dir) if u.lower().endswith((".tif", ".tiff"))]
    return {
        "images": _urls(imgs_dir),
        "tifs": tifs,
        "overlays": _urls(overlays),
        "thumbs": _urls(thumbs),
    }

# ================== lifecycle & basic routes ==================

@app.on_event("startup")
async def _on_startup() -> None:
    loop = asyncio.get_running_loop()
    broker.set_loop(loop)              # <-- IMPORTANT for real-time streaming
    _wire_logging_to_sse()  
    logger.info("PVRT API started.")

@app.get("/api/logs")
async def stream_logs():
    q = await broker.subscribe()     # returns a Queue seeded with history
    return sse_response(q)           # no other changes needed


@app.get("/api/health")
async def api_health():
    return {"ok": True, "time": _now_stamp()}

@app.post("/api/cancel")
async def api_cancel(job: str = Form(...)):
    job = job.strip().lower()
    if job == "train":
        CANCEL_FLAGS["train"] = True
        logger.info("UI:INFO:train: Cancel requested (best-effort).")
        return {"ok": True}
    raise HTTPException(status_code=400, detail=f"Unknown job: {job}")

# ================== TRAIN ==================

@app.post("/api/train")
async def api_train(
    use_thermal: bool = Form(False),
    max_iter: int = Form(1000),
    base_lr: float = Form(0.00025),
    ims_per_batch: int = Form(2),
    model_name: str = Form(""),
    backend: str = Form("detectron"),
):
    safe_name = _safe_name(model_name) or _now_stamp()
    run_dir = OUTPUTS / safe_name
    run_dir.mkdir(parents=True, exist_ok=True)

    CANCEL_FLAGS["train"] = False
    logger.info("UI:OK:train: Training started…")
    logger.info(
        f"[train] run={run_dir.name} backend={backend} "
        f"use_thermal={use_thermal} iters={max_iter} lr={base_lr} batch={ims_per_batch}"
    )

    # Prepare thermal pairs if requested
    if use_thermal:
        ensure_dirp_init()
        scan_split_decode_thermal(TRAIN_DIR)
        scan_split_decode_thermal(VALID_DIR)

    # Offload the heavy training to a background thread so SSE can stream
    def _do_train():
        from detectron2.utils.logger import setup_logger
        setup_logger()  # route detectron2/fvcore to std logging (SSE handler will pick it up)
        with redirect_std_to_logger():
            return train_entry(
                backend=backend,
                train_dir=TRAIN_DIR,
                val_dir=VALID_DIR,
                out_dir=run_dir,
                use_thermal_request=use_thermal,
                max_iter=max_iter,
                base_lr=base_lr,
                ims_per_batch=ims_per_batch,
                run_name=run_dir.name,
            )

    try:
        resp = await asyncio.to_thread(_do_train)  # <-- key change
        meta = resp.get("meta", {})
        logger.info(f"[train] complete: run={run_dir.name}")
        logger.info("UI:OK:train: Training completed.")
        return {"ok": True, "run": run_dir.name, "meta": meta}
    except Exception as e:
        logger.exception("Training failed.")
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")


# -------------- List model runs --------------

@app.get("/api/models")
async def api_models():
    return {"ok": True, "models": _list_models()}


# ================== TEST: dataset intake ==================

@app.get("/api/test_datasets")
async def api_test_datasets():
    details = _list_datasets()                      # current shape: [{name, count, mtime}, ...]
    names = [d["name"] for d in details]           # simple shape: ["name", ...]
    return {"ok": True, "datasets": details, "dataset_names": names}

@app.post("/api/test_upload")
async def api_test_upload_underscore(
    files: List[UploadFile] = File(...),
    result_name: str = Form(""),
):
    """
    Alias for upload using underscore path expected by the UI.
    """
    # Reuse the same logic as your existing upload handler.
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    base = _safe_name(result_name) or _safe_name(files[0].filename or "upload")
    created: List[str] = []
    zips = [f for f in files if (f.filename or "").lower().endswith(".zip")]
    imgs = [f for f in files if Path(f.filename or "").suffix.lower() in IMAGE_EXTS]
    if zips:
        ds_dir = _save_zip_as_dataset(zips[0], preferred_base=base); created.append(ds_dir.name)
        for z in zips[1:]:
            dsz = _save_zip_as_dataset(z); created.append(dsz.name)
    elif imgs:
        ds_dir = _save_images_as_dataset(imgs, preferred_base=base); created.append(ds_dir.name)
    if not created:
        return {"ok": False, "created": []}
    logger.info(f"UI:OK:test: Created datasets: {', '.join(created)}")
    return {"ok": True, "created": created}


# ================== TEST: run ==================

@app.post("/api/test_run")
async def api_test_run(
    dataset: str = Form(...),
    model: Optional[str] = Form(default=None),
    use_thermal: bool = Form(default=False),
    result_name: str = Form(default=""),
    forced_backend: Optional[str] = Form(default=None),
):
    ds_dir = TEST_DIR / dataset
    
    if not ds_dir.exists() or not ds_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset}' not found.")

    if model:
        model_dir = OUTPUTS / model
        if not model_dir.exists():
            raise HTTPException(status_code=404, detail=f"Model '{model}' not found.")
    else:
        models = _list_models()
        if not models:
            raise HTTPException(status_code=404, detail="No trained models found.")
        model_dir = OUTPUTS / models[-1]

    session = (_safe_name(result_name) or _now_stamp())
    base = MEDIA_DIR / "sessions"
    ses = base / session

    out_root = MEDIA_DIR / "sessions" / session
    out_root.mkdir(parents=True, exist_ok=True)

    if use_thermal:
        ensure_dirp_init()
        scan_split_decode_thermal(ds_dir)
        logger.info("thermal images decoded")

    def _do_predict():
        with redirect_std_to_logger():
            return predict_entry(
                weights_dir=model_dir,
                images_dir=ds_dir,
                out_dir=out_root,
                use_thermal_request=use_thermal,
                forced_backend=forced_backend,
            )

    try:
        presp = await asyncio.to_thread(_do_predict)  # <-- offload
    except Exception as e:
        logger.exception("Inference failed.")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    preds_dir = Path(presp["results_dir"])
    class_names = (_read_model_meta(model_dir).get("class_names") or [])
    # ov_dir, th_dir, manifest = _draw_overlays(ds_dir, preds_dir, out_root, class_names)
    ov_dir, th_dir, manifest_path = _draw_overlays(ds_dir, preds_dir, out_root, class_names)
    gj, _ = _preds_to_geojson(ds_dir, preds_dir, out_root, class_names)

    if isinstance(manifest_path, (str, Path)):
        mp = Path(manifest_path)
        if mp.suffix.lower() == ".json" and mp.exists():
            try:
                manifest_items = json.loads(mp.read_text(encoding="utf-8"))
            except Exception:
                manifest_items = []
        else:
            manifest_items = []
    elif isinstance(manifest_path, list):
        manifest_items = manifest_path
    else:
        manifest_items = []
    
    assets = _session_assets(ses)

    logger.info(f"UI:OK:test: complete. results={preds_dir}")
    return {
        "ok": True,
        "results_dir": str(preds_dir),
        "overlays": str(ov_dir),
        "thumbs": str(th_dir),
        "manifest": manifest_items,
        "assets": assets,
        "geojson": str(gj),
        "backend": presp.get("used_backend"),
        "model_mode": presp.get("model_mode"),
        "used_thermal": bool(presp.get("used_thermal")),
        "media_root": f"/media/sessions/{out_root.name}",
        "session":session,
    }


# ================== SESSIONS (lightweight) ==================

# -------------- Sessions list --------------
@app.get("/api/sessions")
async def api_sessions():
    """
    Returns both a detailed list and a simple list of session IDs.
    """
    details = _list_sessions()                 # [{"name","mtime"}, ...] where name == session id
    ids = [d["name"] for d in details]
    return {"ok": True, "sessions": details, "session_ids": ids}


@app.get("/api/session_summary")
async def api_session_summary(session: str):
    base = MEDIA_DIR / "sessions"
    ses = base / session
    if not ses.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    gj = ses / "anomalies.geojson"
    manifest_path = ses / "manifest.json"
    manifest = []
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            manifest = []
    return {
        "ok": True,
        "session": session,
        "geojson_url": f"/media/{gj.relative_to(MEDIA_DIR)}" if gj.exists() else None,
        "assets": _session_assets(ses),
        "manifest": manifest,
        "tiler": "ok" if RIO_OK else "unavailable"
    }


# -------------- Simple dynamic tiler for TIFF (XYZ) --------------
_TILER_INDEX: Dict[str, List[Path]] = {}

def _session_tifs(session: str) -> List[Path]:
    ses = MEDIA_DIR / session
    if not ses.exists():
        return []
    return [p for p in (ses / "images").glob("*") if p.suffix.lower() in (".tif", ".tiff")]

@app.get("/api/session_tiles")
async def api_session_tiles(session: str):
    if not RIO_OK:
        return {"ok": False, "reason": "rasterio_not_available", "layers": []}
    tifs = _session_tifs(session)
    _TILER_INDEX[session] = tifs
    layers = []
    for i, p in enumerate(tifs):
        try:
            with rasterio.open(p) as ds:
                # bounds in WGS84
                try:
                    left, bottom, right, top = rasterio.warp.transform_bounds(ds.crs, CRS.from_epsg(4326), *ds.bounds, densify_pts=21)
                except Exception:
                    left, bottom, right, top = -180.0, -85.0, 180.0, 85.0
                layers.append({
                    "name": p.name,
                    "template": f"/tiles/{session}/{i}" + "/{z}/{x}/{y}.png",
                    "bounds": [ [bottom, left], [top, right] ],
                    "minzoom": 0,
                    "maxzoom": 22
                })
        except Exception:
            continue
    return {"ok": True, "layers": layers}

@app.get("/tiles/{session:path}/{idx:int}/{z:int}/{x:int}/{y:int}.png")
async def tile_xyz(session: str, idx: int, z: int, x: int, y: int):
    if not RIO_OK:
        raise HTTPException(status_code=404, detail="Tiler unavailable")
    if session not in _TILER_INDEX:
        _TILER_INDEX[session] = _session_tifs(session)
    tifs = _TILER_INDEX[session]
    if idx < 0 or idx >= len(tifs):
        raise HTTPException(status_code=404, detail="Tile source not found")
    src_path = tifs[idx]
    try:
        with rasterio.open(src_path) as src:
            dst_crs = CRS.from_epsg(3857)  # Web Mercator
            # tile bounds in WebMercator meters
            tb = mercantile.xy_bounds(mercantile.Tile(x=x, y=y, z=z))
            west_m, south_m, east_m, north_m = tb.left, tb.bottom, tb.right, tb.top

            dst_transform = from_bounds(west_m, south_m, east_m, north_m, 256, 256)

            # Prepare destination arrays
            bands = min(3, max(1, src.count))
            dst = np.zeros((bands, 256, 256), dtype=np.float32)

            # Reproject per band directly from dataset
            for b in range(bands):
                reproject(
                    source=rasterio.band(src, b+1),
                    destination=dst[b],
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                    num_threads=1
                )

            # Normalize to 0..255 uint8
            out = np.zeros((256,256,3), dtype=np.uint8)
            if bands == 1:
                a = dst[0]
                finite = np.isfinite(a)
                if np.any(finite):
                    mn, mx = np.percentile(a[finite], [2, 98])
                    if mx <= mn: mx = mn + 1.0
                    norm = np.clip((a - mn) / (mx - mn), 0, 1)
                else:
                    norm = np.zeros_like(a)
                g = (norm * 255).astype(np.uint8)
                out[...,0] = g; out[...,1] = g; out[...,2] = g
            else:
                for i in range(bands):
                    a = dst[i]
                    finite = np.isfinite(a)
                    if np.any(finite):
                        mn, mx = np.percentile(a[finite], [2, 98])
                        if mx <= mn: mx = mn + 1.0
                        norm = np.clip((a - mn) / (mx - mn), 0, 1)
                    else:
                        norm = np.zeros_like(a)
                    out[..., i] = (norm * 255).astype(np.uint8)

            # Encode as PNG
            im = Image.fromarray(out, mode="RGB")
            bio = io.BytesIO()
            im.save(bio, format="PNG", optimize=True)
            return Response(content=bio.getvalue(), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tile error: {e}")

# -------------- Serve media & frontend --------------
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR), html=False), name="media")
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="web")
