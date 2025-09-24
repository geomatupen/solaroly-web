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
from fastapi.responses import JSONResponse

from PIL import Image, ExifTags
from PIL.ExifTags import TAGS, GPSTAGS
import numpy as np
import math

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
    Single-folder version:
      - Only use/create <out_root>/overlays (JPG)
      - Reuse existing JPG overlays if predictor already wrote them there
      - Else draw fallback rectangles from preds JSON and save JPG
      - Also create thumbs/ and a manifest.json
    """
    import json, logging
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    logger   = logging.getLogger("pvrt.test")
    overlays = out_root / "overlays"; overlays.mkdir(parents=True, exist_ok=True)
    thumbs   = out_root / "thumbs";   thumbs.mkdir(parents=True, exist_ok=True)
    manifest = out_root / "manifest.json"

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
        ov   = overlays / f"{stem}.jpg"   # single format
        th   = thumbs   / f"{stem}.jpg"

        # If predictor already wrote an overlay JPG here, reuse it
        if not ov.exists():
            # Fallback: read preds and draw
            pred_json = preds_dir / "preds" / f"{stem}.json"
            jj = _coerce_pred_json(pred_json) if pred_json.exists() else {
                "boxes": [], "scores": [], "classes": [], "file": img.name
            }

            try:
                src = Image.open(img)
                exif = src.info.get("exif", None)
                base = src.convert("RGB")
            except Exception:
                base = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
                exif = None

            draw = ImageDraw.Draw(base)
            W, H = base.size
            pal = _palette_rgb()

            boxes   = jj.get("boxes", []) or []
            scores  = jj.get("scores", []) or []
            classes = jj.get("classes", []) or []

            thickness = max(1, int(round(min(W, H) * 0.003)))
            try:
                font = ImageFont.truetype("arial.ttf", size=max(10, int(min(W, H) * 0.018)))
            except Exception:
                font = ImageFont.load_default()

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
                draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)

                try:
                    bbox = draw.textbbox((0, 0), label, font=font)
                    tw, th_txt = bbox[2] - bbox[0], bbox[3] - bbox[1]
                except Exception:
                    tw, th_txt = draw.textsize(label, font=font)
                pad = 4
                pill_w = tw + 2 * pad
                pill_h = th_txt + 2 * pad

                top = y1 - pill_h if (y1 - pill_h) >= 0 else y1
                left = x1
                draw.rectangle([left, top, left + pill_w, top + pill_h], fill=color)
                tx, ty = left + pad, top + pad
                for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
                    draw.text((tx + dx, ty + dy), label, fill=(0, 0, 0), font=font)
                draw.text((tx, ty), label, fill=(255, 255, 255), font=font)

            # Save fallback as JPG (preserve EXIF if available)
            try:
                if exif:
                    base.save(ov, format="JPEG", quality=90, exif=exif)
                else:
                    base.save(ov, format="JPEG", quality=90)
            except Exception:
                # if JPEG somehow fails, still ensure a file exists
                base.save(ov, format="JPEG", quality=85)

            im_for_thumb = base
        else:
            # open existing overlay for thumbnail
            try:
                im_for_thumb = Image.open(ov).convert("RGB")
            except Exception:
                im_for_thumb = Image.new("RGB", (256, 256), (0, 0, 0))

        # Thumbnail (JPG)
        try:
            w, h = im_for_thumb.size
            tw, thh = max(96, w // 6), max(96, h // 6)
            im_thumb = im_for_thumb.resize((tw, thh))
            im_thumb.save(th, format="JPEG", quality=85)
        except Exception:
            Image.new("RGB", (96, 96), (0, 0, 0)).save(th, format="JPEG", quality=85)

        mapper[img.name] = {
            "overlays": f"/media/{ov.relative_to(MEDIA_DIR).as_posix()}" if str(ov).startswith(str(MEDIA_DIR)) else ov.name,
            "thumb":   f"/media/{th.relative_to(MEDIA_DIR).as_posix()}" if str(th).startswith(str(MEDIA_DIR)) else th.name,
        }

    manifest.write_text(json.dumps(mapper, indent=2), encoding="utf-8")
    return overlays, thumbs, manifest



# ---------- Geo helpers: EXIF GPS + input type detection ----------

# map EXIF tag ids → names once
_EXIF_GPS_TAG = None
try:
    _EXIF_GPS_TAG = {v: k for k, v in ExifTags.TAGS.items()}["GPSInfo"]
except Exception:
    _EXIF_GPS_TAG = 34853  # fallback id

def _to_float_ratio(val):
    # PIL gives (num, den) tuples or IFDRational; normalize to float
    try:
        if hasattr(val, "numerator"):
            return float(val.numerator) / float(val.denominator or 1)
        num, den = val
        return float(num) / float(den or 1)
    except Exception:
        try:
            return float(val)
        except Exception:
            return 0.0

def _dms_to_deg(dms_tuple):
    # (deg, min, sec) -> decimal degrees
    d = _to_float_ratio(dms_tuple[0])
    m = _to_float_ratio(dms_tuple[1])
    s = _to_float_ratio(dms_tuple[2])
    return d + m/60.0 + s/3600.0

def get_image_gps(image_path):
    """
    Return (lat, lon) in WGS84 from EXIF, or (None, None) if missing.
    Works with JPG/TIFF. PNG rarely has EXIF.
    """
    try:
        img = Image.open(image_path)
        exif = getattr(img, "_getexif", lambda: None)()
        if not exif:
            return None, None

        gps = {}
        for k, v in exif.items():
            tag = TAGS.get(k)
            if tag == "GPSInfo":
                for t in v:
                    sub = GPSTAGS.get(t)
                    gps[sub] = v[t]

        def _to_float(x):
            # EXIF rationals can be tuples (num, den)
            return x[0] / x[1] if isinstance(x, tuple) else float(x)

        def _dms_to_dd(dms, ref):
            deg = _to_float(dms[0]); minu = _to_float(dms[1]); sec = _to_float(dms[2])
            dd = deg + minu/60.0 + sec/3600.0
            return -dd if ref in ("S", "W") else dd

        lat = gps.get("GPSLatitude");  lat_ref = gps.get("GPSLatitudeRef")
        lon = gps.get("GPSLongitude"); lon_ref = gps.get("GPSLongitudeRef")
        if lat and lon and lat_ref and lon_ref:
            return _dms_to_dd(lat, lat_ref.strip()), _dms_to_dd(lon, lon_ref.strip())
    except Exception:
        pass
    return None, None


def _detect_image_input_type(images_dir: Path) -> str:
    """
    'tif'  -> exactly one GeoTIFF present (orthophoto case)
    'images' -> otherwise (many JPG/PNG/etc or multiple TIFFs)
    """
    tifs = [p for p in images_dir.glob("*") if p.suffix.lower() in (".tif", ".tiff")]
    return "tif" if len(tifs) == 1 else "images"



# ----------------- overlays & geojson -----------------
def _meters_to_deg(lat_deg: float):
    """
    Return (deg_per_meter_lon, deg_per_meter_lat) at a given latitude.
    Approximate but fine for small images.
    """
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * math.cos(math.radians(lat_deg))
    return (1.0 / meters_per_deg_lon, 1.0 / meters_per_deg_lat)

def _coerce_pred_json(p):
    """Load a preds json that looks like {'boxes':[[x1,y1,x2,y2],...],'scores':[],'classes':[]}."""
    try:
        j = json.loads(Path(p).read_text(encoding="utf-8"))
        return {
            "boxes":  j.get("boxes",  []) or [],
            "scores": j.get("scores", []) or [],
            "classes":j.get("classes",[]) or [],
        }
    except Exception:
        return {"boxes": [], "scores": [], "classes": []}

def _preds_to_geojson(
    images_dir: Path,
    preds_dir: Path,
    out_session: Path,
    class_names,
    score_thresh: float = 0.5,
    meters_per_pixel: float = 0.10,   # << your notebook default (10cm/px). Adjust as needed.
):
    """
    Build polygon GeoJSON for detections using EXIF (lat,lon) as the image center.
    Geometry is Polygon in EPSG:4326. Also writes image centers to images.geojson.
    Returns (anomalies_geojson_path, images_geojson_path).
    """

    anomalies = {"type": "FeatureCollection", "features": []}
    images_fc = {"type": "FeatureCollection", "features": []}

    preds_dir = Path(preds_dir)
    out_session.mkdir(parents=True, exist_ok=True)

    # walk images and match to preds
    for img_path in sorted(Path(images_dir).iterdir()):
        if not img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            continue

        # EXIF WGS84
        lat, lon = get_image_gps(str(img_path))
        # record image center (if present) to images.geojson
        if lat is not None and lon is not None:
            images_fc["features"].append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
                "properties": {"image": img_path.name}
            })

        # load predictions for this image
        stem = img_path.stem
        pred_json = preds_dir / "preds" / f"{stem}.json"
        jj = _coerce_pred_json(pred_json) if pred_json.exists() else {"boxes": [], "scores": [], "classes": []}
        boxes, scores, classes = jj["boxes"], jj["scores"], jj["classes"]

        # image size (for pixel→meter offsets)
        try:
            from PIL import Image
            with Image.open(str(img_path)) as im:
                W, H = im.size
        except Exception:
            W = H = None

        if not boxes or W is None or H is None:
            continue

        # Need EXIF center to place polygons on the map
        if lat is None or lon is None:
            # No EXIF → skip georeferenced polygons for this image
            continue

        # meters → degrees scaling at this latitude
        dlon_per_m, dlat_per_m = _meters_to_deg(lat)

        for i, b in enumerate(boxes):
            try:
                s = float(scores[i]) if i < len(scores) else 0.0
            except Exception:
                s = 0.0
            if s < float(score_thresh):
                continue

            try:
                x1, y1, x2, y2 = map(float, b)
            except Exception:
                continue
            if x2 <= x1 or y2 <= y1:
                continue

            # bbox center in pixels (origin top-left)
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)

            # offsets from image center, px → meters
            dx_px = cx - (W / 2.0)
            dy_px = cy - (H / 2.0)
            dx_m = dx_px * meters_per_pixel
            dy_m = -dy_px * meters_per_pixel   # image y grows down; north is +lat

            # center of bbox in degrees
            lon_c = float(lon) + dx_m * dlon_per_m
            lat_c = float(lat) + dy_m * dlat_per_m

            # half-sizes in meters
            hw_m = 0.5 * (x2 - x1) * meters_per_pixel
            hh_m = 0.5 * (y2 - y1) * meters_per_pixel

            # corner offsets (east, north) in meters → degrees
            corners_m = [
                (+hw_m, +hh_m),
                (-hw_m, +hh_m),
                (-hw_m, -hh_m),
                (+hw_m, -hh_m),
                (+hw_m, +hh_m),  # close ring
            ]
            ring = [[lon_c + ex * dlon_per_m, lat_c + ny * dlat_per_m] for (ex, ny) in corners_m]

            cls_id = int(classes[i]) if i < len(classes) else 0
            cls_name = (
                class_names[cls_id] if isinstance(class_names, (list, tuple))
                and 0 <= cls_id < len(class_names) else f"class_{cls_id}"
            )

            anomalies["features"].append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {
                    "image": img_path.name,
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "score": round(s * 100.0, 2),    # percent like your notebook
                    "box_px": [x1, y1, x2, y2]
                }
            })

    # write files
    anom_gj = out_session / "anomalies.geojson"
    imgs_gj = out_session / "images.geojson"
    anom_gj.write_text(json.dumps(anomalies), encoding="utf-8")
    imgs_gj.write_text(json.dumps(images_fc), encoding="utf-8")

    return anom_gj, imgs_gj




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
    test_threshold: str = Form(default=""),
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
                score_thresh_frontend=test_threshold,
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
    # gj, _ = _preds_to_geojson(ds_dir, preds_dir, out_root, class_names)
    try:
        th_num = float(test_threshold) if str(test_threshold).strip() else 0.0
    except Exception:
        th_num = 0.0

    # anom_gj, imgs_gj = _preds_to_geojson(
    #     ds_dir, preds_dir, out_root, class_names, score_thresh=th_num
    # )
    # after you’ve created the session folder and saved overlays/ + preds/

    session_dir = MEDIA_DIR / "sessions" / session
    
    anom_gj, imgs_gj = _preds_to_geojson(
        images_dir=Path(ds_dir),
        preds_dir=Path(preds_dir),
        out_session=Path(session_dir),
        class_names=class_names,
        score_thresh=float(th_num or 0.0),
        meters_per_pixel=0.05,   #ground sampling distance (GSD)
    )


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

    # --- persist image_input_type in metrics.json ---
    try:
        mpath = out_root / "metrics.json"   # session/<name>/metrics.json
        import json, logging
        metrics = {}
        if mpath.exists():
            metrics = json.loads(mpath.read_text(encoding="utf-8"))
        # images_dir should be the directory containing the test inputs you ran on
        metrics["image_input_type"] = _detect_image_input_type(ds_dir)
        # (optional) persist the numeric threshold you just used if not set by predictor
        metrics.setdefault("score_thresh_test", th_num)
        mpath.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    except Exception as e:
        logging.getLogger("pvrt.test").warning(f"metrics.json patch failed: {e}")


    logger.info(f"UI:OK:test: complete. results={preds_dir}")
    return {
        "ok": True,
        "session": session,
        # keep old key for backward compatibility
        "geojson": str(anom_gj),
        "anomalies_geojson": f"/media/{anom_gj.relative_to(MEDIA_DIR)}",
        "images_geojson":    f"/media/{imgs_gj.relative_to(MEDIA_DIR)}",
        "results_dir": str(preds_dir),
        "overlays": f"/media/{ov_dir.relative_to(MEDIA_DIR)}",
        "thumbs":   f"/media/{th_dir.relative_to(MEDIA_DIR)}",
        "manifest": manifest_items,
        "assets": assets,
        "backend": presp.get("used_backend"),
        "model_mode": presp.get("model_mode"),
        "used_thermal": bool(presp.get("used_thermal")),
        "media_root": f"/media/sessions/{out_root.name}",
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

    gj = ses / "anomalies.geojson"      # existing
    imgs_gj = ses / "images.geojson"    # NEW

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
        # keep old key for backward compatibility (anomalies)
        "geojson_url": f"/media/{gj.relative_to(MEDIA_DIR)}" if gj.exists() else None,
        # NEW: where image footprints live (if you created them)
        "images_geojson_url": f"/media/{imgs_gj.relative_to(MEDIA_DIR)}" if imgs_gj.exists() else None,
        "assets": _session_assets(ses),
        "manifest": manifest,   # still the parsed JSON (not a path)
        "tiler": "ok" if RIO_OK else "unavailable",
    }




@app.get("/api/results/{session}/metrics")
def api_metrics(session: str):
    p = MEDIA_DIR / "sessions" / session / "metrics.json"
    if not p.exists():
        raise HTTPException(404, "metrics.json not found")
    return JSONResponse(json.loads(p.read_text(encoding="utf-8")))

@app.get("/api/runs/{run_name}/meta")
def api_model_meta(run_name: str):
    p = OUTPUTS / run_name / "model_meta.json"
    if not p.exists():
        raise HTTPException(404, "model_meta.json not found")
    return JSONResponse(json.loads(p.read_text(encoding="utf-8")))



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
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS)), name="outputs")
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="web")
