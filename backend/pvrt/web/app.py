# backend/pvrt/web/app.py
from __future__ import annotations

import asyncio
import copy
import io
import json
import logging
import os
import shlex
import sys
import shutil
import zipfile
import re
import subprocess
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from xml.etree import ElementTree as ET

from fastapi import FastAPI, Form, UploadFile, File, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import StreamingResponse
from fastapi.responses import JSONResponse

from PIL import Image, ExifTags
from PIL.ExifTags import TAGS, GPSTAGS
import numpy as np
import math
from io import BytesIO
_TILER_STATS = {}

# --- Optional tiler deps (best-effort) ---
try:
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
    import mercantile
    from pyproj import Transformer
    RIO_OK = True
except (ImportError, ModuleNotFoundError):
    RIO_OK = False

# --- Backend-agnostic bridge and registry ---
from ..core.registry import register_backend
from .bridge import train_entry, predict_entry
from ..backends.detectron.backend import register as register_detectron
from ..backends.yolo.backend import register as register_yolo

# --- SSE/logging bridge ---
from .sse import LogBroker, SSELogHandler, set_event_loop, sse_response

# --- Reuse data helpers (RJPEG decode & scanning) ---
from ..dataops.scan_decode_split import (
    ensure_dirp_init, scan_split_decode_thermal, # safe to call only if thermal requested
)
from ..core.io import has_thermal_for_images

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
COLMAP_BASE  = MEDIA_DIR / "colmap"
COLMAP_BASE.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".jpg",".jpeg",".png",".tif",".tiff",".bmp",".webp",".JPG",".JPEG",".PNG",".TIF",".TIFF",".BMP",".WEBP"}

COLMAP_JOBS: Dict[str, Dict[str, Any]] = {}
COLMAP_MAX_LOGS = 400
COLMAP_PROGRESS_RE = re.compile(r"Processed file \[(\d+)/(\d+)\]")
COLMAP_STAGE_WEIGHTS: List[Tuple[str, float]] = [
    ("feature_extractor", 0.45),
    ("matcher", 0.2),
    ("mapper", 0.3),
    ("model_converter", 0.05),
]
COLMAP_STAGE_WEIGHT_LOOKUP = {label: weight for label, weight in COLMAP_STAGE_WEIGHTS}

# Common non-fatal exceptions we allow to be handled locally across web helpers.
# This intentionally excludes BaseException-derived types such as KeyboardInterrupt
# and SystemExit so they propagate.
COMMON_EXCEPTIONS = (FileNotFoundError, OSError, json.JSONDecodeError, ValueError, ImportError)

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
register_yolo(register_backend)

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
        sys.stdout.flush(); sys.stderr.flush()
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
        return json.loads(meta.read_text(encoding="utf-8"))
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
                    # prefer explicit model_name from meta (may include _4ch suffix),
                    # but keep 'name' as the run folder for lookups
                    "model_name": meta.get("model_name") or None,
                    "input_mode": meta.get("input_mode", "rgb"),
                    "channel_count": meta.get("channel_count"),
                    "backend": meta.get("backend", "detectron")
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


# ---- session helpers ----
from typing import Any

def _as_session_dir(ses) -> Path:
    """
    Accept a session name like 'test_20250927_111404' or an absolute Path,
    and return the media sessions/<name> directory.
    """
    p = Path(ses)
    if p.exists():
        return p
    return MEDIA_DIR / "sessions" / str(ses)

def _load_metrics(ses) -> Dict[str, Any]:
    """
    Read MEDIA_DIR/sessions/<session>/metrics.json safely.
    Returns {} if missing/invalid.
    """
    ses_dir = _as_session_dir(ses)
    mp = ses_dir / "metrics.json"
    return json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}


def _tif_band_count(p: Path) -> int:
    if not RIO_OK:
        return 0
    import rasterio
    with rasterio.open(p) as ds:
        return int(ds.count)

# --- ADD: small GeoTIFF helpers ---
def _iter_geotiffs(dirpath: Path):
    for p in sorted(dirpath.rglob("*")):
        if p.is_file() and p.suffix.lower() in (".tif", ".tiff"):
            yield p



def _render_tif_overlay_preview(
    tif_path,                 # source GeoTIFF (RGB or single-band)
    anomalies_geojson_path,   # EPSG:4326 polygons
    out_png_path,             # where to save the preview PNG (under overlays/)
    max_px=2000,              # max side for downsampled image
    line_thickness=2
):
    """
    Downsample the TIF to <= max_px on the long side, draw anomalies from anomalies_geojson_path
    (WGS84) onto the preview (in source CRS), and save a single PNG for the Results page.
    """
    import json
    from pathlib import Path
    import numpy as np, cv2, rasterio
    from rasterio.enums import Resampling
    from affine import Affine
    from pyproj import Transformer

    tif_path = Path(tif_path); out_png_path = Path(out_png_path)
    out_png_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(tif_path) as src:
        W, H = src.width, src.height
        # target preview size
        scale = max(W, H) / float(max_px) if max(W, H) > max_px else 1.0
        out_w = int(round(W / scale)); out_h = int(round(H / scale))

        # --- Read bands robustly and always end with 3 channels ---
        cnt = src.count
        if cnt >= 3:
            arr = src.read(indexes=[1, 2, 3],
                           out_shape=(3, out_h, out_w),
                           resampling=Resampling.bilinear)
        elif cnt == 2:
            a12 = src.read(indexes=[1, 2],
                           out_shape=(2, out_h, out_w),
                           resampling=Resampling.bilinear)
            a3 = np.zeros((1, out_h, out_w), dtype=a12.dtype)
            arr = np.concatenate([a12, a3], axis=0)
        else:
            a1 = src.read(indexes=1,
                          out_shape=(out_h, out_w),
                          resampling=Resampling.bilinear)
            arr = np.stack([a1, a1, a1], axis=0)  # (3,H,W)

        # robust 2–98% stretch → uint8
        arr = arr.astype("float32")
        for c in range(3):
            a = arr[c]
            lo, hi = np.percentile(a, 2), np.percentile(a, 98)
            if hi <= lo: hi = lo + 1
            arr[c] = np.clip((a - lo) * (255.0/(hi-lo)), 0, 255)

        # (H,W,3) uint8, C-contiguous for OpenCV
        rgb = arr.transpose(1, 2, 0).astype(np.uint8)
        rgb = np.ascontiguousarray(rgb)

        # Transform of the downsampled image (map↔pixel)
        ds_transform = src.transform * Affine.scale(W/out_w, H/out_h)
        inv_ds = ~ds_transform
        tf_wgs_to_src = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)

    # --- Draw polygons ---
    with open(anomalies_geojson_path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    def _draw_ring(ring_lonlat):
        if not ring_lonlat:
            return
        lon = [pt[0] for pt in ring_lonlat]
        lat = [pt[1] for pt in ring_lonlat]
        X, Y = tf_wgs_to_src.transform(lon, lat)  # to source CRS
        pts = []
        for x, y in zip(X, Y):
            col, row = inv_ds * (x, y)
            pts.append([int(round(col)), int(round(row))])
        if len(pts) >= 2:
            pts_np = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
            # draw in-place; rgb is contiguous uint8 (H,W,3)
            cv2.polylines(rgb, [pts_np], isClosed=True, color=(255, 0, 0), thickness=line_thickness)

    feats = gj.get("features", [])
    for feat in feats:
        geom = (feat or {}).get("geometry") or {}
        gtype = geom.get("type")
        if gtype == "Polygon":
            for ring in geom.get("coordinates", []):
                _draw_ring(ring)
        elif gtype == "MultiPolygon":
            for rings in geom.get("coordinates", []):
                for ring in rings:
                    _draw_ring(ring)
        # ignore other geometry types

    cv2.imwrite(str(out_png_path), rgb)
    return out_png_path



# --- ADD: split a GeoTIFF into GeoTIFF tiles (preserve CRS/transform) ---
def _tile_tif_to_dir(tif_path, tiles_dir, tile_size=1024, stride=None):
    """
    Split a GeoTIFF into GeoTIFF tiles preserving CRS/transform; return list of tile paths.
    Logs CRS for source and a few sample tiles for verification.
    """
    import logging, rasterio
    from rasterio.windows import Window
    from rasterio.windows import transform as win_transform
    from pathlib import Path

    def _crs_id(crs):
        if crs is None:
            return "None"
        epsg = crs.to_epsg()
        return f"EPSG:{epsg}" if epsg else crs.to_string()

    logger = logging.getLogger("pvrt")
    tiles_dir = Path(tiles_dir)
    tiles_dir.mkdir(parents=True, exist_ok=True)

    stride = stride or tile_size
    written = []

    with rasterio.open(tif_path) as src:
        logger.info(f"Tiler: source='{Path(tif_path).name}' CRS={_crs_id(src.crs)} size={src.width}x{src.height}")
        W, H = src.width, src.height

        for y0 in range(0, H, stride):
            for x0 in range(0, W, stride):
                w = min(tile_size, W - x0)
                h = min(tile_size, H - y0)
                if w <= 1 or h <= 1:
                    continue

                window = Window(x0, y0, w, h)
                t_tile = win_transform(window, src.transform)

                profile = src.profile.copy()
                profile.update({
                    "height": h,
                    "width":  w,
                    "transform": t_tile,
                    "driver": "GTiff",
                })
                # ensure every tile carries a CRS
                profile["crs"] = src.crs

                outp = tiles_dir / f"{Path(tif_path).stem}_{y0}_{x0}.tif"
                with rasterio.open(outp, "w", **profile) as dst:
                    for b in range(1, src.count + 1):
                        dst.write(src.read(b, window=window), b)
                written.append(outp)

                # Log a few tiles to verify CRS & transform persisted
                if (x0, y0) in [(0, 0), (tile_size, 0), (0, tile_size)]:
                    with rasterio.open(outp) as chk:
                        logger.info(
                            f"Tiler: wrote '{outp.name}' CRS={_crs_id(chk.crs)} transform={chk.transform}"
                        )

    return written


# --- ADD: stitch per-tile JSON predictions into one anomalies.geojson (WGS84) ---
def _build_anomalies_geojson_from_tiles(
    tiles_dir,            # folder that contains the tile GeoTIFFs
    preds_dir,            # session results dir, contains "preds/*.json"
    tif_path,             # original source GeoTIFF
    out_session,          # /media/sessions/<session>
    class_names,          # list of class names
    score_thresh=0.0,
):
    """
    Build ONE merged anomalies.geojson in EPSG:4326 from per-tile predictions.
    Prefers per-instance mask polygons in pixel space (jd['polygons']); falls back to bboxes.
    Clips results to the source TIF footprint. Also writes images.geojson center point.
    Returns (anomalies_geojson_path, images_geojson_path).
    """
    import json, logging
    from pathlib import Path
    import rasterio
    from shapely.geometry import Polygon, mapping, box as shp_box
    from pyproj import Transformer

    def _crs_id(crs):
        try:
            if crs is None:
                return "None"
            epsg = crs.to_epsg()
            return f"EPSG:{epsg}" if epsg else crs.to_string()
        except Exception:
            return str(crs)

    logger = logging.getLogger("pvrt")
    tiles_dir   = Path(tiles_dir)
    preds_dir   = Path(preds_dir)
    tif_path    = Path(tif_path)
    out_session = Path(out_session)
    out_session.mkdir(parents=True, exist_ok=True)

    # -- index tiles (support .tif and .tiff) --
    tile_paths = list(tiles_dir.glob("*.tif")) + list(tiles_dir.glob("*.tiff"))
    tile_index = {p.stem: p for p in tile_paths}

    # -- source CRS + footprint (WGS84) for clipping / fallback --
    src_crs = None
    tif_footprint_wgs84 = None
    with rasterio.open(tif_path) as src_ds:
        src_crs = src_ds.crs
        b = src_ds.bounds
        logger.info(f"Stitcher: source='{tif_path.name}' CRS={_crs_id(src_crs)} bounds={b}")
        tf_src_to_wgs = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
        xs = [b.left, b.right, b.right, b.left]
        ys = [b.bottom, b.bottom, b.top,  b.top]
        lon, lat = tf_src_to_wgs.transform(xs, ys)
        left, right = min(lon), max(lon)
        bottom, top = min(lat), max(lat)
    tif_footprint_wgs84 = shp_box(left, bottom, right, top)

    feats = []
    preds_json_dir = preds_dir / "preds"

    # Read run metrics (if present) so we can make decisions like forcing
    # overlay regeneration for single-channel thermal runs where a raw
    # thermal TIFF exists. Default to None (unknown) so behavior is
    # conservative and preserves reuse when we can't read metrics.
    run_channel_count = None
    mpath = preds_dir / "metrics.json"
    if mpath.exists():
        mm = json.loads(mpath.read_text(encoding="utf-8"))
        run_channel_count = int(mm.get("channel_count") or mm.get("channel", mm.get("input_channels", 0)) or 0)

    def _find_thermal_candidate(p: Path) -> Path | None:
        """Quick check for a thermal source for `p` (same rules as predictors).
        Return a Path if a candidate exists, otherwise None.
        """
        exts = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
        tdir = p.parent / "thermal"
        # pairs.json mapping
        pjson = tdir / "pairs.json"
        if pjson.exists():
            pairs = json.loads(pjson.read_text(encoding="utf-8"))
            rel = pairs.get(p.name)
            if rel:
                cand = (p.parent / rel)
                if cand.exists():
                    return cand
        # decoder naming
        for e in exts:
            cand = tdir / f"{p.stem}_thermal{e}"
            if cand.exists():
                return cand
            cand2 = tdir / f"{p.stem}{e}"
            if cand2.exists():
                return cand2
        # sidecar next to image
        for e in exts:
            cand = p.with_name(f"{p.stem}_thermal{e}")
            if cand.exists():
                return cand
        # legacy: also accept common image preview suffixes in addition to TIFF
        for ext in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
            cand = p.with_name(f"{p.stem}_thermal{ext}")
            if cand.exists():
                return cand
        return None

    # Read run metrics (if present) so we can make decisions like forcing
    # overlay regeneration for single-channel thermal runs where a raw
    # thermal TIFF exists. Default to None (unknown) so behavior is
    # conservative and preserves reuse when we can't read metrics.
    run_channel_count = None
    mpath = preds_dir / "metrics.json"
    if mpath.exists():
        mm = json.loads(mpath.read_text(encoding="utf-8"))
        run_channel_count = int(mm.get("channel_count") or mm.get("channel", mm.get("input_channels", 0)) or 0)

    def _find_thermal_candidate(p: Path) -> Path | None:
        """Quick check for a thermal source for `p` (same rules as predictors).
        Return a Path if a candidate exists, otherwise None.
        """
        try:
            exts = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
            tdir = p.parent / "thermal"
            # pairs.json mapping
            pjson = tdir / "pairs.json"
            if pjson.exists():
                try:
                    pairs = json.loads(pjson.read_text(encoding="utf-8"))
                    rel = pairs.get(p.name)
                    if rel:
                        cand = (p.parent / rel)
                        if cand.exists():
                            return cand
                except Exception as e:
                    logger.debug("ignored web.app error: %s", e)
            # decoder naming
            for e in exts:
                cand = tdir / f"{p.stem}_thermal{e}"
                if cand.exists():
                    return cand
                cand2 = tdir / f"{p.stem}{e}"
                if cand2.exists():
                    return cand2
            # sidecar next to image
            for e in exts:
                cand = p.with_name(f"{p.stem}_thermal{e}")
                if cand.exists():
                    return cand
            # legacy: also accept common image preview suffixes in addition to TIFF
            for ext in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
                cand = p.with_name(f"{p.stem}_thermal{ext}")
                if cand.exists():
                    return cand
        except COMMON_EXCEPTIONS:
            return None
        return None

    for jpath in sorted(preds_json_dir.glob("*.json")):
        try:
            jd = json.loads(jpath.read_text(encoding="utf-8"))
        except COMMON_EXCEPTIONS:
            # Skip malformed prediction JSON files but allow other errors to propagate.
            continue

        polygons_px = jd.get("polygons") or []
        boxes       = jd.get("boxes",   []) or []
        scores      = jd.get("scores",  []) or []
        classes     = jd.get("classes", []) or []

        fname = jd.get("file") or (jpath.stem + ".tif")
        stem  = Path(fname).stem

        tpath = tile_index.get(stem)
        if not tpath or not tpath.exists():
            continue

        with rasterio.open(tpath) as ds:
            tfm = ds.transform
            crs = ds.crs or src_crs    # tiles sometimes lose CRS — fallback to source
            tf_tile_to_wgs = Transformer.from_crs(crs, "EPSG:4326", always_xy=True) if crs else None

            # pixel (col,x; row,y) → map coords using Affine (NOTE: no +0.5 — matches notebook)
            def px_to_map(col, row):
                X, Y = (tfm * (col, row))
                return (X, Y)

            N = max(len(scores), len(classes), len(boxes), len(polygons_px))
            for i in range(N):
                sc = float(scores[i]) if i < len(scores) else 0.0
                if sc < float(score_thresh):
                    continue
                cid = int(classes[i]) if i < len(classes) else 0

                # Prefer precise mask polygon
                if i < len(polygons_px) and polygons_px[i] and len(polygons_px[i]) >= 3:
                    pts = polygons_px[i]  # [[x,y], ...] in pixel space
                    ring_native = [px_to_map(float(x), float(y)) for (x, y) in pts]
                    if ring_native[0] != ring_native[-1]:
                        ring_native.append(ring_native[0])
                else:
                    if i >= len(boxes) or len(boxes[i]) != 4:
                        continue
                    x1, y1, x2, y2 = map(float, boxes[i])
                    ring_native = [
                        px_to_map(x1, y1),
                        px_to_map(x2, y1),
                        px_to_map(x2, y2),
                        px_to_map(x1, y2),
                        px_to_map(x1, y1),
                    ]

                xs, ys = zip(*ring_native)

                # Reproject to EPSG:4326 via pyproj (axis-safe across rasterio versions)
                poly = None
                if tf_tile_to_wgs is not None:
                    try:
                        lon, lat = tf_tile_to_wgs.transform(list(xs), list(ys))
                        if (-180 <= min(lon) <= 180 and -90 <= min(lat) <= 90):
                            poly = Polygon(zip(lon, lat))
                    except COMMON_EXCEPTIONS:
                        poly = None

                if poly is None:
                    # last resort: keep native coords only if they already look like degrees
                    if max(abs(min(xs)), abs(max(xs))) <= 180 and max(abs(min(ys)), abs(max(ys))) <= 90:
                        poly = Polygon(ring_native)
                    else:
                        continue

                # Clip to source footprint
                if tif_footprint_wgs84 is not None and poly.is_valid:
                    poly = poly.intersection(tif_footprint_wgs84)

                if not poly.is_valid or poly.is_empty or poly.area <= 0:
                    continue

                cname = class_names[cid] if isinstance(class_names, (list, tuple)) and 0 <= cid < len(class_names) else f"class_{cid}"
                feats.append({
                    "type": "Feature",
                    "geometry": mapping(poly),
                    "properties": {
                        "score": sc,
                        "class_id": cid,
                        "class_name": cname,
                        "tile": Path(fname).name,
                        "source": tif_path.name,
                    }
                })

    # --- write anomalies.geojson ---
    anom_fc = {"type": "FeatureCollection", "features": feats}
    anom_path = out_session / "anomalies.geojson"
    anom_path.write_text(json.dumps(anom_fc, indent=2), encoding="utf-8")

    # --- write images.geojson (center point in WGS84) ---
    try:
        with rasterio.open(tif_path) as src_ds:
            b = src_ds.bounds
            tf_src_to_wgs = Transformer.from_crs(src_ds.crs, "EPSG:4326", always_xy=True)
            lon, lat = tf_src_to_wgs.transform([b.left, b.right], [b.bottom, b.top])
        cx = (min(lon) + max(lon)) / 2.0
        cy = (min(lat) + max(lat)) / 2.0
    except COMMON_EXCEPTIONS:
        # fallback: union tile bounds
        try:
            xs = []; ys = []
            tf_src_to_wgs = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True) if src_crs else None
            for t in tile_paths:
                with rasterio.open(t) as ds2:
                    bb = ds2.bounds
                    if tf_src_to_wgs:
                        tlon, tlat = tf_src_to_wgs.transform([bb.left, bb.right], [bb.bottom, bb.top])
                        xs += [min(tlon), max(tlon)]
                        ys += [min(tlat), max(tlat)]
            cx = (min(xs) + max(xs)) / 2.0 if xs else 0.0
            cy = (min(ys) + max(ys)) / 2.0 if ys else 0.0
        except COMMON_EXCEPTIONS:
            cx, cy = 0.0, 0.0

    imgs_fc = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [cx, cy]},
            "properties": {"image": Path(tif_path).name}
        }]
    }
    imgs_path = out_session / "images.geojson"
    imgs_path.write_text(json.dumps(imgs_fc, indent=2), encoding="utf-8")
    return anom_path, imgs_path





# --- ADD (helper near others): quick PNG thumb for a GeoTIFF ---
def _save_tif_thumbnail(tif_path: Path, thumbs_dir: Path, max_px: int = 640) -> Path:
    import rasterio
    from PIL import Image
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    out = thumbs_dir / (tif_path.stem + ".png")
    try:
        with rasterio.open(tif_path) as ds:
            bsel = [b for b in (1,2,3) if b <= ds.count] or [1]
            arr = ds.read(bsel)
            arr = arr.astype("float32").transpose(1,2,0)
            # normalize each band 2–98%
            for c in range(arr.shape[2]):
                a = arr[..., c]
                lo, hi = np.percentile(a, 2), np.percentile(a, 98)
                if hi <= lo: hi = lo + 1
                arr[..., c] = np.clip((a - lo) * (255/(hi-lo)), 0, 255)
            if arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=2)
            h, w = arr.shape[:2]
            scale = min(1.0, max_px / max(h, w))
            im = Image.fromarray(arr.astype("uint8"))
            if scale < 1.0:
                im = im.resize((int(w*scale), int(h*scale)), Image.BILINEAR)
            im.save(out, format="PNG", optimize=True)
    except COMMON_EXCEPTIONS:
        from PIL import Image
        Image.new("RGB", (256,256), (40,40,40)).save(out, "PNG", optimize=True)
    return out


# --- Accurate location (COLMAP) helpers ---

def _colmap_dataset_dir(dataset: str) -> Path:
    ds_dir = TEST_DIR / dataset
    if not ds_dir.exists() or not ds_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset}' not found.")
    return ds_dir


def _colmap_meta_path(dataset: str) -> Path:
    return COLMAP_BASE / dataset / "colmap_meta.json"


def _colmap_ready_path(dataset: str) -> Path:
    return COLMAP_BASE / dataset / "ready.json"


def _load_colmap_meta(dataset: str) -> Dict[str, Any]:
    mp = _colmap_meta_path(dataset)
    if not mp.exists():
        return {}
    try:
        return json.loads(mp.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _merge_optical_metadata(
    base_thermal_meta: Dict[str, Any],
    optimization_project: str,
) -> Dict[str, Any]:
    """
    Merge optical project geometry (rotation, gimbal_yaw, heading, lat, lon)
    into thermal metadata by matching image names (strip _T/_V suffixes).
    """
    logger = logging.getLogger("pvrt.test")
    merged = dict(base_thermal_meta)
    
    optical_meta = _load_colmap_meta(optimization_project)
    if not optical_meta:
        logger.warning(f"UI:WARN:test: optimization_project '{optimization_project}' has no colmap_meta.json")
        return merged
    
    def normalize_basename(fname: str) -> str:
        """Remove _T, _V, -T, -V suffixes from filename"""
        base = fname.rsplit(".", 1)[0] if "." in fname else fname
        for suffix in ["_T", "_V", "-T", "-V"]:
            if base.endswith(suffix):
                base = base[:-len(suffix)]
                break
        return base
    
    optical_index = {}
    for fname, entry in optical_meta.items():
        if fname.startswith("__") or not isinstance(entry, dict):
            continue
        optical_index[normalize_basename(fname)] = entry
    
    matched_count = 0
    total_thermal = sum(1 for k in merged.keys() if not k.startswith("__"))
    
    for fname in list(merged.keys()):
        if fname.startswith("__"):
            continue
        norm = normalize_basename(fname)
        if norm in optical_index:
            optical_entry = optical_index[norm]
            thermal_entry = merged[fname]
            for field in ["rotation", "gimbal_yaw", "heading", "lat", "lon", "latitude", "longitude"]:
                if field in optical_entry:
                    thermal_entry[field] = optical_entry[field]
            matched_count += 1
    
    match_pct = (matched_count * 100 // total_thermal) if total_thermal > 0 else 0
    
    logger.info(f"UI:INFO:test: ══════════════════════════════════════════════════════")
    logger.info(f"UI:INFO:test: Optical/Thermal Image Matching Summary:")
    logger.info(f"UI:INFO:test:   - Thermal images: {total_thermal}")
    logger.info(f"UI:INFO:test:   - Optical project: {optimization_project}")
    logger.info(f"UI:INFO:test:   - Matched images: {matched_count}/{total_thermal} ({match_pct}%)")
    
    if match_pct < 80:
        logger.warning(f"UI:WARN:test: Match percentage ({match_pct}%) is below 80% threshold")
        logger.warning(f"UI:WARN:test: Reverting to standard EXIF metadata (non-accurate poses)")
        logger.info(f"UI:INFO:test: ══════════════════════════════════════════════════════")
        raise ValueError(f"Insufficient match rate: only {matched_count}/{total_thermal} images ({match_pct}%) matched. Need ≥80% for optical sync.")
    
    logger.info(f"UI:INFO:test: ✓ Using accurate poses from optical project '{optimization_project}'")
    logger.info(f"UI:INFO:test: ══════════════════════════════════════════════════════")
    
    meta_info = merged.setdefault("__meta__", {})
    if isinstance(meta_info, dict):
        meta_info["source"] = "thermal_exif+optical_geometry"
        meta_info["optimization_project"] = optimization_project
        meta_info["match_percent"] = match_pct
    
    return merged


def _colmap_ready(dataset: str) -> bool:
    rp = _colmap_ready_path(dataset)
    if not rp.exists():
        return False
    try:
        data = json.loads(rp.read_text(encoding="utf-8"))
        return bool(data.get("finalized"))
    except Exception:
        return False


def _set_colmap_ready(dataset: str, job_id: str):
    rp = _colmap_ready_path(dataset)
    rp.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "finalized": True,
        "job_id": job_id,
        "finished_at": _now_stamp(),
    }
    rp.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _clear_colmap_ready(dataset: str):
    rp = _colmap_ready_path(dataset)
    try:
        if rp.exists():
            rp.unlink()
    except Exception:
        pass


def _dataset_image_list(dataset: str) -> List[Path]:
    return [path for path, _ in _colmap_image_sources(dataset)]


def _colmap_image_sources(dataset: str) -> List[Tuple[Path, str]]:
    ds_dir = _colmap_dataset_dir(dataset)
    records: List[Tuple[Path, str]] = []
    for entry in sorted(ds_dir.iterdir()):
        if not entry.is_file() or not _is_image(entry):
            continue
        rel = entry.relative_to(ds_dir).as_posix()
        records.append((entry, rel))
    records.sort(key=lambda pair: pair[1])
    return records


def _gather_colmap_cameras(dataset: str) -> Dict[str, Dict[str, Any]]:
    ds_dir = _colmap_dataset_dir(dataset)
    try:
        camera_meta = _build_camera_meta_from_exif(ds_dir)
    except Exception:
        camera_meta = {}
    sizes_index = _scan_image_sizes(ds_dir)
    meta_lookup = _load_colmap_meta(dataset)
    ready_flag = _colmap_ready(dataset)

    cameras: Dict[str, Dict[str, Any]] = {}
    for img_path in _dataset_image_list(dataset):
        key = img_path.name
        entry = _lookup_camera_meta_entry(camera_meta, key) if camera_meta else None
        lat = float(entry.get("lat")) if entry and entry.get("lat") is not None else None
        lon = float(entry.get("lon")) if entry and entry.get("lon") is not None else None
        alt = float(entry.get("alt")) if entry and entry.get("alt") is not None else None
        w_px = int(entry.get("w_px")) if entry and entry.get("w_px") is not None else None
        h_px = int(entry.get("h_px")) if entry and entry.get("h_px") is not None else None
        mpp = float(entry.get("meters_per_pixel")) if entry and entry.get("meters_per_pixel") is not None else None

        size = sizes_index.get(key)
        if size:
            w_px = w_px or int(size[0])
            h_px = h_px or int(size[1])

        cameras[key] = {
            "file": key,
            "lat": lat,
            "lon": lon,
            "alt": alt,
            "w": w_px,
            "h": h_px,
            "meters_per_pixel": mpp,
            "rotation": entry.get("rotation") if entry else None,
            "rotation_gimbal": entry.get("rotation_gimbal") if entry else None,
            "rotation_aircraft": entry.get("rotation_aircraft") if entry else None,
            "rotation_source": entry.get("rotation_source") if entry else None,
            "status": "pending",
            "calibrated": False,
            "has_gps": lat is not None and lon is not None,
            "optimized": None,
        }

        meta_entry = meta_lookup.get(key) or meta_lookup.get(Path(key).name, None)
        if isinstance(meta_entry, dict):
            cameras[key]["optimized"] = {
                "lat": meta_entry.get("lat"),
                "lon": meta_entry.get("lon"),
                "alt": meta_entry.get("alt"),
                "rotation": meta_entry.get("rotation"),
                "heading": meta_entry.get("rotation"),
            }
            cameras[key]["status"] = "calibrated" if ready_flag else "optimized"
            cameras[key]["calibrated"] = True if ready_flag else False

    return cameras


def _colmap_job_summary(job: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not job:
        return None
    return {
        "id": job.get("id"),
        "status": job.get("status"),
        "dataset": job.get("dataset"),
        "current_step": job.get("current_step"),
        "error": job.get("error"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "params": job.get("params"),
        "progress": job.get("progress"),
        "progress_detail": job.get("progress_detail"),
        "total_images": job.get("total_images"),
    }


def _begin_colmap_stage(job: Dict[str, Any], label: str, offset: float = 0.0):
    weight = COLMAP_STAGE_WEIGHT_LOOKUP.get(label, 0.0)
    job["current_step"] = label
    job["progress_offset"] = offset
    job["progress_stage_weight"] = weight
    job["progress"] = offset
    if label != "feature_extractor":
        job.pop("progress_detail", None)


def _finish_colmap_stage(job: Dict[str, Any]):
    offset = job.get("progress_offset", 0.0)
    weight = job.get("progress_stage_weight", 0.0)
    job["progress"] = max(0.0, min(1.0, offset + weight))
    if job.get("current_step") != "feature_extractor":
        job.pop("progress_detail", None)


def _append_colmap_log(job: Dict[str, Any], line: str):
    line = line.strip()
    if not line:
        return
    logs = job.setdefault("logs", [])
    logs.append(line)
    if len(logs) > COLMAP_MAX_LOGS:
        del logs[: len(logs) - COLMAP_MAX_LOGS]
    logging.getLogger("pvrt.colmap").info(line)

    match = COLMAP_PROGRESS_RE.search(line)
    if match:
        done = int(match.group(1))
        total = int(match.group(2)) or 1
        offset = job.get("progress_offset", 0.0)
        weight = job.get("progress_stage_weight", 0.0) or 0.0
        job["progress"] = max(0.0, min(1.0, offset + weight * (done / total)))
        job["progress_detail"] = {"done": done, "total": total}


def _colmap_state(dataset: str) -> Dict[str, Any]:
    cameras = _gather_colmap_cameras(dataset)
    job = COLMAP_JOBS.get(dataset)
    if job:
        cam_state = job.get("cameras", {})
        for key, state in cam_state.items():
            base = cameras.setdefault(key, {"file": key})
            for field, value in state.items():
                base[field] = value
    ready_flag = _colmap_ready(dataset)
    meta_path = _colmap_meta_path(dataset)
    meta_ref = str(meta_path) if meta_path.exists() else None
    return {
        "dataset": dataset,
        "ready": bool(ready_flag),
        "meta_path": meta_ref,
        "cameras": list(cameras.values()),
        "job": _colmap_job_summary(job),
        "logs": (job or {}).get("logs", [])[-100:],
    }


def _colmap_binary() -> str:
    candidates: List[str] = []
    env_bin = os.environ.get("COLMAP_BIN")
    if env_bin:
        candidates.append(env_bin)
    env_home = os.environ.get("COLMAP_HOME")
    if env_home:
        candidates.extend([
            env_home,
            str(Path(env_home) / "colmap"),
            str(Path(env_home) / "bin" / "colmap"),
        ])
    repo_colmap = PROJECT_ROOT / "colmap"
    candidates.extend([
        shutil.which("colmap"),
        str(repo_colmap),
        str(repo_colmap / "colmap"),
        str(repo_colmap / "bin" / "colmap"),
    ])

    for cand in candidates:
        if not cand:
            continue
        p = Path(cand)
        if p.is_dir():
            p = p / "colmap"
        if p.exists() and os.access(p, os.X_OK):
            return str(p)
    raise HTTPException(status_code=400, detail="COLMAP binary not found. Set COLMAP_BIN or install COLMAP to use accurate locations.")


async def _run_colmap_command(job: Dict[str, Any], label: str, args: List[str], cwd: Optional[Path] = None) -> None:
    job["current_step"] = label
    job.setdefault("logs", [])
    display_cmd = " ".join(shlex.quote(str(a)) for a in args)
    _append_colmap_log(job, f"[{label}] $ {display_cmd}")

    def _runner():
        env = os.environ.copy()
        # Force Qt to run headless so COLMAP binaries do not require an X server.
        env.setdefault("QT_QPA_PLATFORM", "offscreen")
        proc = subprocess.Popen(
            [str(a) for a in args],
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                _append_colmap_log(job, line)
        finally:
            proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"{label} failed with exit code {proc.returncode}")

    await asyncio.to_thread(_runner)


async def _colmap_pipeline(job: Dict[str, Any]) -> None:
    dataset = job.get("dataset")
    if not dataset:
        job["status"] = "failed"
        job["error"] = "Dataset missing"
        return

    job["status"] = "running"
    job.setdefault("logs", [])
    job["started_at"] = job.get("started_at") or _now_stamp()
    job["progress"] = job.get("progress", 0.0)
    job["progress_offset"] = 0.0
    job["progress_stage_weight"] = 0.0

    ds_dir = _colmap_dataset_dir(dataset)
    base_dir = COLMAP_BASE / dataset
    job_dir = base_dir / job["id"]
    job_dir.mkdir(parents=True, exist_ok=True)
    job["work_dir"] = str(job_dir)
    database_path = job_dir / "database.db"
    sparse_dir = job_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    text_dir = job_dir / "model-text"
    text_dir.mkdir(parents=True, exist_ok=True)

    params = job.get("params") or {}

    def _as_int(val: Any) -> Optional[int]:
        try:
            return int(val)
        except (TypeError, ValueError):
            return None

    def _as_float(val: Any) -> Optional[float]:
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    def _as_bool(val: Any) -> Optional[bool]:
        if isinstance(val, str):
            return val.lower() in {"1", "true", "yes", "on"}
        if val is None:
            return None
        try:
            return bool(val)
        except Exception:
            return None

    matcher = str(params.get("matcher", "exhaustive")).lower()
    min_tri = params.get("min_triangulation_angle")
    try:
        min_tri = float(min_tri) if min_tri is not None else None
    except Exception:
        min_tri = None
    seq_overlap = params.get("seq_overlap")
    try:
        seq_overlap = int(seq_overlap) if seq_overlap is not None else None
    except Exception:
        seq_overlap = None
    max_image_size = params.get("max_image_size")
    try:
        max_image_size = int(max_image_size) if max_image_size is not None else None
        if max_image_size is not None and max_image_size <= 0:
            max_image_size = None
    except Exception:
        max_image_size = None
    camera_model = params.get("camera_model")
    if isinstance(camera_model, str):
        camera_model = camera_model.strip() or None
    else:
        camera_model = None
    peak_threshold = _as_float(params.get("peak_threshold"))
    edge_threshold = _as_float(params.get("edge_threshold"))
    max_num_features = _as_int(params.get("max_num_features"))
    num_threads = _as_int(params.get("num_threads"))
    exhaustive_block_size = _as_int(params.get("exhaustive_block_size"))
    ba_refine_focal_length = _as_bool(params.get("ba_refine_focal_length"))
    ba_refine_principal_point = _as_bool(params.get("ba_refine_principal_point"))
    ba_refine_extra_params = _as_bool(params.get("ba_refine_extra_params"))
    min_num_matches = _as_int(params.get("min_num_matches"))
    init_min_inliers = _as_int(params.get("init_min_num_inliers"))
    abs_pose_min_inliers = _as_int(params.get("abs_pose_min_num_inliers"))
    max_model_overlap = _as_int(params.get("max_model_overlap"))
    max_num_models = _as_int(params.get("max_num_models"))
    use_gpu = params.get("use_gpu")
    if isinstance(use_gpu, str):
        use_gpu = use_gpu.lower() in {"1", "true", "yes", "on"}
    elif use_gpu is not None:
        use_gpu = bool(use_gpu)

    image_sources = _colmap_image_sources(dataset)
    if not image_sources:
        raise RuntimeError("Dataset does not contain any supported image files for COLMAP.")
    image_list_path = job_dir / "colmap_images.txt"
    image_list_path.write_text("\n".join(rel for _, rel in image_sources), encoding="utf-8")
    job["total_images"] = len(image_sources)

    _clear_colmap_ready(dataset)

    try:
        colmap_bin = _colmap_binary()
        progress_offset = 0.0

        def _advance_stage(label: str):
            nonlocal progress_offset
            _begin_colmap_stage(job, label, progress_offset)

        def _complete_stage():
            nonlocal progress_offset
            _finish_colmap_stage(job)
            progress_offset = job.get("progress", progress_offset)

        _advance_stage("feature_extractor")
        feature_args = [
            colmap_bin,
            "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(ds_dir),
            "--image_list_path", str(image_list_path),
        ]
        if max_image_size is not None:
            feature_args.extend(["--SiftExtraction.max_image_size", str(max_image_size)])
        if peak_threshold is not None:
            feature_args.extend(["--SiftExtraction.peak_threshold", f"{peak_threshold}"])
        if edge_threshold is not None:
            feature_args.extend(["--SiftExtraction.edge_threshold", f"{edge_threshold}"])
        if max_num_features is not None:
            feature_args.extend(["--SiftExtraction.max_num_features", str(max_num_features)])
        if num_threads is not None and num_threads > 0:
            feature_args.extend(["--SiftExtraction.num_threads", str(num_threads)])
        if use_gpu is not None:
            feature_args.extend(["--FeatureExtraction.use_gpu", "1" if use_gpu else "0"])
        if camera_model:
            feature_args.extend(["--ImageReader.camera_model", camera_model])
        await _run_colmap_command(job, "feature_extractor", feature_args)
        _complete_stage()

        matcher_cmd = "sequential_matcher" if matcher == "sequential" else "exhaustive_matcher"
        matcher_args = [
            colmap_bin,
            matcher_cmd,
            "--database_path", str(database_path),
        ]
        if matcher == "sequential" and seq_overlap is not None:
            matcher_args.extend(["--SequentialMatching.overlap", str(seq_overlap)])
        if matcher == "exhaustive" and exhaustive_block_size is not None:
            matcher_args.extend(["--ExhaustiveMatching.block_size", str(exhaustive_block_size)])

        _advance_stage("matcher")
        await _run_colmap_command(job, matcher_cmd, matcher_args)
        _complete_stage()

        _advance_stage("mapper")
        mapper_args = [
            colmap_bin,
            "mapper",
            "--database_path", str(database_path),
            "--image_path", str(ds_dir),
            "--output_path", str(sparse_dir),
        ]
        if min_tri is not None:
            mapper_args.extend(["--Mapper.filter_min_tri_angle", str(min_tri)])
        min_model_size = params.get("min_model_size")
        if min_model_size is not None:
            mapper_args.extend(["--Mapper.min_model_size", str(min_model_size)])
        if min_num_matches is not None:
            mapper_args.extend(["--Mapper.min_num_matches", str(min_num_matches)])
        if init_min_inliers is not None:
            mapper_args.extend(["--Mapper.init_min_num_inliers", str(init_min_inliers)])
        if abs_pose_min_inliers is not None:
            mapper_args.extend(["--Mapper.abs_pose_min_num_inliers", str(abs_pose_min_inliers)])
        if max_model_overlap is not None:
            mapper_args.extend(["--Mapper.max_model_overlap", str(max_model_overlap)])
        if max_num_models is not None:
            mapper_args.extend(["--Mapper.max_num_models", str(max_num_models)])
        if ba_refine_focal_length is not None:
            mapper_args.extend(["--Mapper.ba_refine_focal_length", "1" if ba_refine_focal_length else "0"])
        if ba_refine_principal_point is not None:
            mapper_args.extend(["--Mapper.ba_refine_principal_point", "1" if ba_refine_principal_point else "0"])
        if ba_refine_extra_params is not None:
            mapper_args.extend(["--Mapper.ba_refine_extra_params", "1" if ba_refine_extra_params else "0"])
        
        # Optional: relaxed filtering for thermal imagery
        max_reproj_error = _as_float(params.get("max_reproj_error"))
        if max_reproj_error is not None:
            mapper_args.extend(["--Mapper.filter_max_reproj_error", str(max_reproj_error)])
        
        ba_global_max_iter = _as_int(params.get("ba_global_max_iterations"))
        if ba_global_max_iter is not None:
            mapper_args.extend(["--Mapper.ba_global_max_num_iterations", str(ba_global_max_iter)])
        
        ba_local_max_iter = _as_int(params.get("ba_local_max_iterations"))
        if ba_local_max_iter is not None:
            mapper_args.extend(["--Mapper.ba_local_max_num_iterations", str(ba_local_max_iter)])

        await _run_colmap_command(job, "mapper", mapper_args)
        _complete_stage()

        model_dirs = [p for p in sparse_dir.iterdir() if p.is_dir()]
        if not model_dirs:
            raise RuntimeError("COLMAP mapper produced no sparse models.")
        
        # Pick the largest model (most registered images)
        def _model_size(model_path: Path) -> int:
            images_bin = model_path / "images.bin"
            if not images_bin.exists():
                return 0
            # Rough estimate: file size / typical bytes per image entry
            return images_bin.stat().st_size
        
        model_dirs.sort(key=_model_size, reverse=True)
        model_dir = model_dirs[0]

        _advance_stage("model_converter")
        converter_args = [
            colmap_bin,
            "model_converter",
            "--input_path", str(model_dir),
            "--output_path", str(text_dir),
            "--output_type", "TXT",
        ]
        await _run_colmap_command(job, "model_converter", converter_args)
        _complete_stage()

        images_txt = text_dir / "images.txt"
        solution = _parse_colmap_images_txt(images_txt)
        if not solution:
            raise RuntimeError("COLMAP images.txt empty; no calibrated cameras recorded.")

        alignment = _align_colmap_solution(dataset, solution)
        meta_path = _write_colmap_meta(dataset, solution, alignment)
        job["meta_path"] = str(meta_path)

        # Read back the written meta to get calibrated rotation values
        colmap_meta = _load_colmap_meta(dataset)
        
        job_cams: Dict[str, Dict[str, Any]] = {}
        for img in _dataset_image_list(dataset):
            name = img.name
            aligned = alignment.get(name)
            meta_entry = colmap_meta.get(name)
            optimized = None
            if aligned and meta_entry:
                optimized = {
                    "lat": meta_entry.get("lat"),
                    "lon": meta_entry.get("lon"),
                    "alt": meta_entry.get("alt"),
                    "rotation": meta_entry.get("rotation"),  # Use calibrated rotation from meta
                    "aligned_center": aligned.get("aligned_center"),
                    "scale": aligned.get("scale"),
                }
            job_cams[name] = {
                "file": name,
                "status": "calibrated" if aligned else "pending",
                "calibrated": bool(aligned),
                "optimized": optimized,
            }
        job["cameras"] = job_cams

        job["status"] = "awaiting_finish"
        job["current_step"] = None
        job["progress"] = 1.0
        job["progress_detail"] = {
            "done": job.get("total_images", 0),
            "total": job.get("total_images", 0),
        }
        job["finished_at"] = _now_stamp()
        _append_colmap_log(job, "COLMAP optimization complete. Click Finish to accept the results.")
    except asyncio.CancelledError:
        job["status"] = "cancelled"
        job["error"] = "cancelled"
        job["finished_at"] = _now_stamp()
        _append_colmap_log(job, "COLMAP optimization cancelled.")
        raise
    except Exception as exc:
        job["status"] = "failed"
        job["error"] = str(exc)
        job["finished_at"] = _now_stamp()
        _append_colmap_log(job, f"ERROR: {exc}")
        return


# ================== Accurate location API ==================


def _parse_colmap_params(raw: str | None) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid params JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="params must be a JSON object")
    return data


def _should_poll(job: Optional[Dict[str, Any]]) -> bool:
    if not job:
        return False
    return job.get("status") in {"queued", "running", "awaiting_finish"}


@app.get("/api/colmap/state")
async def api_colmap_state(dataset: str):
    _colmap_dataset_dir(dataset)
    state = _colmap_state(dataset)
    return {"ok": True, "state": state}


@app.get("/api/colmap/cameras")
async def api_colmap_cameras(dataset: str, page: int = 0, limit: int = 50):
    """Paginated cameras list for dataset."""
    _colmap_dataset_dir(dataset)
    cameras = _gather_colmap_cameras(dataset)
    job = COLMAP_JOBS.get(dataset)
    if job:
        cam_state = job.get("cameras", {})
        for key, state in cam_state.items():
            base = cameras.setdefault(key, {"file": key})
            # Don't overwrite optimized.rotation - use the corrected gimbal+aircraft from metadata
            if "optimized" in state and state["optimized"] is not None:
                # Only merge status/calibrated from job, keep metadata rotation
                if "optimized" not in base:
                    base["optimized"] = {}
                if isinstance(base["optimized"], dict):
                    base["optimized"].update({k: v for k, v in state["optimized"].items() if k != "rotation"})
                base["status"] = state.get("status")
                base["calibrated"] = state.get("calibrated")
            else:
                for field, value in state.items():
                    base[field] = value
    
    all_cameras = list(cameras.values())
    total = len(all_cameras)
    
    # Pagination
    start = max(0, page * limit)
    end = start + limit
    paginated = all_cameras[start:end]
    
    return {
        "ok": True,
        "total": total,
        "page": page,
        "limit": limit,
        "cameras": paginated,
        "has_more": end < total,
    }


@app.post("/api/colmap/start")
async def api_colmap_start(dataset: str = Form(...), params: str = Form(default=""), confirm_reset: bool = Form(default=False)):
    _colmap_dataset_dir(dataset)
    existing = COLMAP_JOBS.get(dataset)
    if existing and _should_poll(existing):
        raise HTTPException(status_code=409, detail="COLMAP optimization is already running for this dataset.")

    # Always clear previous results/cached state on each start; frontend handles user confirmation.
    base_dir = COLMAP_BASE / dataset
    if base_dir.exists():
        try:
            shutil.rmtree(base_dir)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to clear previous COLMAP results: {exc}")
    if dataset in COLMAP_JOBS:
        COLMAP_JOBS.pop(dataset, None)

    job_id = uuid.uuid4().hex[:12]
    job: Dict[str, Any] = {
        "id": job_id,
        "dataset": dataset,
        "status": "queued",
        "params": _parse_colmap_params(params),
        "created_at": _now_stamp(),
        "started_at": None,
        "finished_at": None,
        "logs": [],
        "current_step": None,
        "progress": 0.0,
        "progress_detail": None,
        "progress_offset": 0.0,
        "progress_stage_weight": 0.0,
        "total_images": 0,
    }
    COLMAP_JOBS[dataset] = job

    async def _runner():
        try:
            await _colmap_pipeline(job)
        finally:
            job.pop("task", None)

    task = asyncio.create_task(_runner())
    job["task"] = task

    state = _colmap_state(dataset)
    return {"ok": True, "job": _colmap_job_summary(job), "state": state}


@app.post("/api/colmap/finish")
async def api_colmap_finish(dataset: str = Form(...), job_id: str = Form(...)):
    _colmap_dataset_dir(dataset)
    job = COLMAP_JOBS.get(dataset)
    meta_path = _colmap_meta_path(dataset)

    # If backend was reloaded, the in-memory job is gone; allow finalization if metadata exists
    if not meta_path.exists():
        raise HTTPException(status_code=400, detail="COLMAP metadata missing; rerun optimization.")

    if job:
        if job.get("id") != job_id:
            raise HTTPException(status_code=404, detail="COLMAP job not found for dataset.")
        if job.get("status") != "awaiting_finish":
            raise HTTPException(status_code=400, detail="Job is not ready to finalize.")
    else:
        # Recreate minimal job to log and return state
        job = {
            "id": job_id,
            "dataset": dataset,
            "status": "awaiting_finish",
            "logs": [],
            "meta_path": str(meta_path),
        }
        COLMAP_JOBS[dataset] = job

    _set_colmap_ready(dataset, job_id)
    job["status"] = "finalized"
    job["current_step"] = None
    job["finalized_at"] = _now_stamp()
    _append_colmap_log(job, "Dataset marked ready for accurate locations.")
    state = _colmap_state(dataset)
    return {"ok": True, "state": state}


def _quaternion_to_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    # Normalize quaternion so qw >= 0 (removes 180° ambiguity in COLMAP output).
    if qw < 0:
        qw, qx, qy, qz = -qw, -qx, -qy, -qz
    
    n = qw*qw + qx*qx + qy*qy + qz*qz
    if n < 1e-8:
        return np.eye(3)
    s = 2.0 / n
    x, y, z = qx, qy, qz
    w = qw
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1 - s * (yy + zz),     s * (xy - wz),     s * (xz + wy)],
        [    s * (xy + wz), 1 - s * (xx + zz),     s * (yz - wx)],
        [    s * (xz - wy),     s * (yz + wx), 1 - s * (xx + yy)],
    ], dtype=float)


def _rotation_matrix_to_heading(rot: np.ndarray) -> float:
    try:
        # Extract yaw (heading) from rotation matrix using ZYX Euler angle convention.
        # For a rotation matrix R from ZYX Euler angles:
        # yaw = atan2(R[1,0], R[0,0])
        heading = math.degrees(math.atan2(rot[1, 0], rot[0, 0]))
        while heading > 180.0:
            heading -= 360.0
        while heading < -180.0:
            heading += 360.0
        return heading
    except Exception:
        return 0.0


def _latlon_to_local(lat: float, lon: float, alt: float, origin_lat: float, origin_lon: float, origin_alt: float) -> np.ndarray:
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * math.cos(math.radians(origin_lat))
    east = (lon - origin_lon) * meters_per_deg_lon
    north = (lat - origin_lat) * meters_per_deg_lat
    up = (alt - origin_alt)
    return np.array([east, north, up], dtype=float)


def _local_to_latlon(east: float, north: float, origin_lat: float, origin_lon: float) -> Tuple[float, float]:
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * math.cos(math.radians(origin_lat))
    lat = origin_lat + (north / meters_per_deg_lat)
    lon = origin_lon + (east / meters_per_deg_lon)
    return lat, lon


def _umeyama(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    n = src.shape[0]
    mean_src = src.mean(axis=0)
    mean_dst = dst.mean(axis=0)
    src_demean = src - mean_src
    dst_demean = dst - mean_dst
    cov = (src_demean.T @ dst_demean) / float(n)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    var_src = np.sum(src_demean ** 2) / float(n)
    scale = np.trace(np.diag(D) @ S) / var_src if var_src > 0 else 1.0
    t = mean_dst - scale * R @ mean_src
    return scale, R, t


def _parse_colmap_images_txt(images_txt: Path) -> Dict[str, Dict[str, Any]]:
    data: Dict[str, Dict[str, Any]] = {}
    if not images_txt.exists():
        return data
    with images_txt.open("r", encoding="utf-8", errors="ignore") as fh:
        lines = fh.readlines()
    idx = 0
    ln = len(lines)
    while idx < ln:
        line = lines[idx].strip()
        idx += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        image_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        cam_id = int(parts[8])
        name = parts[9]
        rot = _quaternion_to_matrix(qw, qx, qy, qz)
        center = (-rot.T @ np.array([tx, ty, tz])).tolist()
        data[name] = {
            "id": image_id,
            "camera_id": cam_id,
            "name": name,
            "qvec": (qw, qx, qy, qz),
            "tvec": (tx, ty, tz),
            "rot": rot,
            "center": center,
        }
        # Skip the following line containing 2D points (if present)
        if idx < ln:
            next_line = lines[idx].strip()
            if next_line and not next_line.startswith("#"):
                idx += 1
    return data


def _align_colmap_solution(dataset: str, solution: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    ds_dir = _colmap_dataset_dir(dataset)
    try:
        camera_meta = _build_camera_meta_from_exif(ds_dir)
    except Exception as exc:
        raise RuntimeError(f"Failed to read EXIF metadata for alignment: {exc}")

    pairs = []
    for name, pose in solution.items():
        entry = _lookup_camera_meta_entry(camera_meta, name)
        if not entry:
            continue
        lat = entry.get("lat")
        lon = entry.get("lon")
        if lat is None or lon is None:
            continue
        alt = float(entry.get("alt") or 0.0)
        pairs.append((name, pose, float(lat), float(lon), alt, entry))

    if len(pairs) < 3:
        raise RuntimeError("Need at least 3 GPS-tagged images for COLMAP alignment.")

    origin_lat = sum(p[2] for p in pairs) / len(pairs)
    origin_lon = sum(p[3] for p in pairs) / len(pairs)
    origin_alt = sum(p[4] for p in pairs) / len(pairs)

    gps_points = []
    colmap_points = []
    for _, pose, lat, lon, alt, _ in pairs:
        gps_points.append(_latlon_to_local(lat, lon, alt, origin_lat, origin_lon, origin_alt))
        colmap_points.append(np.array(pose["center"], dtype=float))

    src = np.stack(colmap_points)
    dst = np.stack(gps_points)
    scale, rot_align, trans = _umeyama(src, dst)

    results: Dict[str, Dict[str, Any]] = {}
    for name, pose in solution.items():
        center = np.array(pose["center"], dtype=float)
        aligned = scale * (rot_align @ center) + trans
        lat, lon = _local_to_latlon(aligned[0], aligned[1], origin_lat, origin_lon)
        alt = origin_alt + aligned[2]
        # Compute camera heading (yaw) using the provided quaternion→matrix convention
        # and maintain world→camera orientation while aligning to the GPS/local frame.
        # For alignment: R_wc_local = R_wc @ rot_align.T
        R_wc = np.array(pose["rot"], dtype=float)
        R_wc_local = R_wc @ rot_align.T
        heading = _rotation_matrix_to_heading(R_wc_local)
        results[name] = {
            "lat": lat,
            "lon": lon,
            "alt": alt,
            "rotation": heading,
            "rotation_source": "colmap_aligned",
            "aligned_center": aligned.tolist(),
            "scale": scale,
        }
    return results


def _write_colmap_meta(dataset: str, solution: Dict[str, Dict[str, Any]], alignment: Dict[str, Dict[str, Any]]):
    ds_dir = _colmap_dataset_dir(dataset)
    try:
        camera_meta = _build_camera_meta_from_exif(ds_dir)
    except Exception:
        camera_meta = {}
    sizes_index = _scan_image_sizes(ds_dir)
    # --- Optional yaw calibration: compute dataset-level offset between aligned COLMAP yaw and EXIF yaw ---
    def _norm_deg(a: float) -> float:
        while a > 180.0:
            a -= 360.0
        while a < -180.0:
            a += 360.0
        return a

    diffs_rad = []
    for name, pose in solution.items():
        aligned = alignment.get(name) or {}
        colmap_yaw = aligned.get("rotation")
        base = _lookup_camera_meta_entry(camera_meta, name) or {}
        exif_yaw = base.get("rotation_gimbal")
        if exif_yaw is None:
            exif_yaw = base.get("rotation_aircraft")
        if colmap_yaw is None or exif_yaw is None:
            continue
        diff = _norm_deg(float(colmap_yaw) - float(exif_yaw))
        diffs_rad.append(math.radians(diff))

    yaw_delta: Optional[float] = None
    if len(diffs_rad) >= 5:
        # circular mean of differences
        s = sum(math.sin(d) for d in diffs_rad)
        c = sum(math.cos(d) for d in diffs_rad)
        yaw_delta = _norm_deg(math.degrees(math.atan2(s, c)))

    meta_out: Dict[str, Dict[str, Any]] = {}
    for name, pose in solution.items():
        base = _lookup_camera_meta_entry(camera_meta, name) or {}
        aligned = alignment.get(name)
        if not aligned:
            continue
        lat = aligned.get("lat")
        lon = aligned.get("lon")
        alt = aligned.get("alt")
        # Prefer the aligned COLMAP yaw; optionally calibrate to EXIF frame via dataset-level offset.
        heading = aligned.get("rotation")
        heading_source = aligned.get("rotation_source") or ("colmap_aligned" if heading is not None else None)
        if heading is not None and yaw_delta is not None:
            heading = _norm_deg(float(heading) - yaw_delta)
            heading_source = "colmap_aligned_calibrated"
        
        # Fallback to EXIF gimbal/aircraft yaw if aligned yaw not available
        if heading is None:
            gimbal_yaw = base.get("rotation_gimbal")
            if gimbal_yaw is not None:
                heading = float(gimbal_yaw)
                heading_source = "gimbal_yaw"
            else:
                aircraft_yaw = base.get("rotation_aircraft")
                if aircraft_yaw is not None:
                    heading = float(aircraft_yaw)
                    heading_source = "aircraft_yaw"
        
        entry = {
            "file": name,
            "lat": lat,
            "lon": lon,
            "alt": alt,
            "rotation": heading,
            "rotation_source": heading_source,
            "meters_per_pixel": base.get("meters_per_pixel") or 0.05,
            "w_px": base.get("w_px") or (sizes_index.get(name, (None, None))[0]),
            "h_px": base.get("h_px") or (sizes_index.get(name, (None, None))[1]),
            "source": "colmap",
            "qw": pose["qvec"][0],
            "qx": pose["qvec"][1],
            "qy": pose["qvec"][2],
            "qz": pose["qvec"][3],
            "tx": pose["tvec"][0],
            "ty": pose["tvec"][1],
            "tz": pose["tvec"][2],
        }
        meta_out[name] = entry

    if not meta_out:
        raise RuntimeError("COLMAP alignment produced no entries.")

    # Note: Rotation now prefers the aligned COLMAP yaw (rotation_source='colmap_aligned').
    # If unavailable, it falls back to EXIF gimbal/aircraft yaw.

    meta_path = _colmap_meta_path(dataset)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta_out, indent=2), encoding="utf-8")
    return meta_path


async def _exec_colmap_step(job: Dict[str, Any], step: str, cmd: List[str]):
    job["current_step"] = step
    _append_colmap_log(job, f"[STEP] {step}")
    # Force Qt to run headless so COLMAP binaries do not require an X server.
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )
    try:
        while True:
            if proc.stdout is None:
                break
            line = await proc.stdout.readline()
            if not line:
                break
            try:
                decoded = line.decode("utf-8", errors="ignore").strip()
            except Exception:
                decoded = line.decode(errors="ignore").strip()
            _append_colmap_log(job, decoded)
        code = await proc.wait()
        if code != 0:
            raise RuntimeError(f"COLMAP step '{step}' failed with exit code {code}")
    finally:
        # StreamReader doesn't provide close(); rely on process lifecycle.
        pass


async def _run_colmap_job(job: Dict[str, Any]):
    dataset = job.get("dataset")
    job["status"] = "running"
    job["error"] = None
    job["finished_at"] = None
    try:
        colmap_bin = _colmap_binary()
        ds_dir = _colmap_dataset_dir(dataset)
        work_base = COLMAP_BASE / dataset / job["id"]
        work_base.mkdir(parents=True, exist_ok=True)
        db_path = work_base / "database.db"
        sparse_dir = work_base / "sparse"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        model_text_dir = work_base / "model_text"
        model_text_dir.mkdir(parents=True, exist_ok=True)
        job["workspace"] = str(work_base)

        params = job.get("params", {})
        camera_model = params.get("camera_model", "SIMPLE_RADIAL")
        max_image_size = params.get("max_image_size")
        use_gpu = bool(params.get("use_gpu", True))

        feature_cmd = [
            colmap_bin,
            "feature_extractor",
            "--database_path", str(db_path),
            "--image_path", str(ds_dir),
            "--ImageReader.camera_model", str(camera_model),
            "--ImageReader.single_camera", "1",
        ]
        if max_image_size:
            feature_cmd += ["--SiftExtraction.max_image_size", str(max_image_size)]
        if use_gpu:
            feature_cmd += ["--SiftExtraction.use_gpu", "1"]
        await _exec_colmap_step(job, "feature_extractor", feature_cmd)

        matcher = str(params.get("matcher", "sequential")).lower()
        if matcher not in {"sequential", "exhaustive", "spatial"}:
            matcher = "sequential"
        matcher_cmd = [colmap_bin, f"{matcher}_matcher", "--database_path", str(db_path)]
        if matcher == "sequential":
            matcher_cmd += ["--SequentialMatching.overlap", str(params.get("seq_overlap", 3))]
        if matcher == "spatial":
            matcher_cmd += ["--SpatialMatching.max_num_neighbors", str(params.get("spatial_neighbors", 20))]
        await _exec_colmap_step(job, f"{matcher}_matcher", matcher_cmd)

        mapper_cmd = [
            colmap_bin,
            "mapper",
            "--database_path", str(db_path),
            "--image_path", str(ds_dir),
            "--output_path", str(sparse_dir),
            "--Mapper.min_num_matches", str(params.get("min_matches", 15)),
        ]
        if params.get("ba_iterations"):
            mapper_cmd += ["--Mapper.ba_global_max_num_iterations", str(params.get("ba_iterations"))]
        await _exec_colmap_step(job, "mapper", mapper_cmd)

        recon_dirs = sorted([p for p in sparse_dir.iterdir() if p.is_dir()])
        if not recon_dirs:
            raise RuntimeError("COLMAP mapper produced no reconstructions.")
        convert_cmd = [
            colmap_bin,
            "model_converter",
            "--input_path", str(recon_dirs[0]),
            "--output_path", str(model_text_dir),
            "--output_type", "TXT",
        ]
        await _exec_colmap_step(job, "model_converter", convert_cmd)

        images_txt = model_text_dir / "images.txt"
        solution = _parse_colmap_images_txt(images_txt)
        if not solution:
            raise RuntimeError("COLMAP produced no registered camera poses.")
        alignment = _align_colmap_solution(dataset, solution)
        meta_path = _write_colmap_meta(dataset, solution, alignment)
        job["meta_path"] = str(meta_path)
        
        # Read back the written meta to get calibrated rotation values
        colmap_meta = _load_colmap_meta(dataset)
        
        job.setdefault("cameras", {})
        for name, aligned in alignment.items():
            meta_entry = colmap_meta.get(name)
            cam = job["cameras"].setdefault(name, {"file": name})
            cam["optimized"] = {
                "lat": meta_entry.get("lat") if meta_entry else aligned.get("lat"),
                "lon": meta_entry.get("lon") if meta_entry else aligned.get("lon"),
                "alt": meta_entry.get("alt") if meta_entry else aligned.get("alt"),
                "rotation": meta_entry.get("rotation") if meta_entry else aligned.get("rotation"),  # Use calibrated rotation
            }
            cam["status"] = "optimized"
            cam["calibrated"] = False
        job["status"] = "complete"
        job["finished_at"] = _now_stamp()
        _append_colmap_log(job, f"COLMAP job {job['id']} completed. Metadata written to {meta_path}.")
    except Exception as exc:
        job["status"] = "error"
        job["error"] = str(exc)
        job["finished_at"] = _now_stamp()
        _append_colmap_log(job, f"Error: {exc}")

# Overlays are generated during inference and saved directly to overlays/ folder


# ---------- Geo helpers: EXIF GPS + input type detection ----------

def _palette_rgb():
    return [
        (255, 0, 0), (0, 170, 255), (0, 200, 0), (255, 0, 200),
        (255, 165, 0), (128, 0, 255), (0, 255, 255), (255, 255, 0),
    ]

def _is_image(p: Path) -> bool:
    return p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def _coerce_pred_json(p: Path) -> dict:
    try:
        
        return json.loads(p.read_text(encoding="utf-8"))
    except COMMON_EXCEPTIONS:
        return {"boxes": [], "scores": [], "classes": [], "file": p.stem}


# ---------- Geo helpers: EXIF GPS + input type detection ----------

# Overlays are generated during inference and saved directly to overlays/ folder.
# This eliminates post-test processing overhead and preserves transparency from rotated images.

# map EXIF tag ids → names once
_EXIF_GPS_TAG = None
try:
    _EXIF_GPS_TAG = {v: k for k, v in ExifTags.TAGS.items()}["GPSInfo"]
except Exception:
    _EXIF_GPS_TAG = 34853  # fallback id

_GPS_SUB_TAGS = {v: k for k, v in ExifTags.GPSTAGS.items()}
_EXIF_FOCAL_LENGTH_TAG = 37386
_EXIF_FOCAL_LENGTH_35MM_TAG = 41989
_EXIF_FOCAL_PLANE_X_RES_TAG = 41486
_EXIF_FOCAL_PLANE_RES_UNIT_TAG = 41488
_EXIF_DATETIME_TAG = 306
_EXIF_DATETIME_ORIGINAL_TAG = 36867
_EXIF_DATETIME_DIGITIZED_TAG = 36868


def _parse_exif_timestamp(value) -> Optional[float]:
    if not value:
        return None
    try:
        if isinstance(value, bytes):
            value = value.decode("ascii", errors="ignore")
        value = str(value).strip()
        dt = datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
        return dt.timestamp()
    except Exception:
        return None

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

def _read_exif_latlon(img_path: Path):
    try:
        img = Image.open(img_path)
        exif = img._getexif()
        if not exif:
            return None, None
        # Build GPS dictionary
        gps_tag = next((k for k, v in ExifTags.TAGS.items() if v == "GPSInfo"), None)
        if gps_tag not in exif:
            return None, None
        gps = exif[gps_tag]

        # Map GPS sub-tags
        inv = {v: k for k, v in ExifTags.GPSTAGS.items()}
        lat = gps.get(inv.get("GPSLatitude"));  lat_ref = gps.get(inv.get("GPSLatitudeRef"))
        lon = gps.get(inv.get("GPSLongitude")); lon_ref = gps.get(inv.get("GPSLongitudeRef"))
        if not (lat and lon and lat_ref and lon_ref):
            return None, None

        def _to_float(x):  # rational or float
            return x[0] / x[1] if isinstance(x, tuple) else float(x)

        def dms_to_dd(dms, ref):
            d = _to_float(dms[0]); m = _to_float(dms[1]); s = _to_float(dms[2])
            dd = d + m/60.0 + s/3600.0
            return -dd if ref in ("S", "W") else dd

        return dms_to_dd(lat, lat_ref), dms_to_dd(lon, lon_ref)
    except Exception:
        return None, None

def _scan_exif_latlon(images_dir: Path) -> dict[str, tuple[float, float]]:
    idx = {}
    for p in images_dir.iterdir():
        if not p.is_file():
            continue
        low = p.suffix.lower()
        if low not in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
            continue
        lat, lon = _read_exif_latlon(p)
        if lat is not None and lon is not None:
            idx[p.name] = (lat, lon)
    return idx


def _detect_image_input_type(images_dir: Path) -> str:
    """
    'tif'  -> exactly one GeoTIFF present (orthophoto case)
    'images' -> otherwise (many JPG/PNG/etc or multiple TIFFs)
    """
    tifs = [p for p in images_dir.glob("*") if p.suffix.lower() in (".tif", ".tiff")]
    return "tif" if len(tifs) == 1 else "images"


def _gps_value(gps_dict: dict, name: str):
    if not gps_dict:
        return None
    key = _GPS_SUB_TAGS.get(name)
    if key is None:
        return None
    return gps_dict.get(key)


def _decode_cardinal_ref(ref) -> str:
    if isinstance(ref, bytes):
        try:
            ref = ref.decode("ascii", errors="ignore")
        except Exception:
            ref = str(ref)
    return str(ref).strip().upper()


def _latlon_from_gps(gps_dict: dict) -> tuple[Optional[float], Optional[float]]:
    lat_val = _gps_value(gps_dict, "GPSLatitude")
    lat_ref = _gps_value(gps_dict, "GPSLatitudeRef")
    lon_val = _gps_value(gps_dict, "GPSLongitude")
    lon_ref = _gps_value(gps_dict, "GPSLongitudeRef")
    if not lat_val or not lon_val or not lat_ref or not lon_ref:
        return None, None
    try:
        lat = _dms_to_deg(lat_val)
        lon = _dms_to_deg(lon_val)
        if _decode_cardinal_ref(lat_ref) == "S":
            lat = -lat
        if _decode_cardinal_ref(lon_ref) == "W":
            lon = -lon
        return lat, lon
    except Exception:
        return None, None


def _coerce_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _normalize_heading_deg(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        deg = float(value)
    except Exception:
        return None
    deg = deg % 360.0
    if deg > 180.0:
        deg -= 360.0
    if deg < -180.0:
        deg += 360.0
    return deg


def _camera_heading_to_overlay_rotation(rotation: Optional[float]) -> float:
    """Return the normalized heading for overlay rotation."""
    base = _normalize_heading_deg(rotation)
    return float(base if base is not None else 0.0)


def _camera_meta_session_meta(camera_meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(camera_meta, dict):
        return {}
    meta = camera_meta.get("__meta__")
    return meta if isinstance(meta, dict) else {}


def _camera_heading_from_entry(
    cam_entry: Optional[Dict[str, Any]],
    session_meta: Dict[str, Any],
) -> Optional[float]:
    heading = None
    if cam_entry and isinstance(cam_entry, dict) and cam_entry.get("rotation") is not None:
        heading = _normalize_heading_deg(cam_entry.get("rotation"))
    if heading is None:
        default_rot = _coerce_float(session_meta.get("default_rotation_deg"))
        if default_rot is not None:
            heading = _normalize_heading_deg(default_rot)
    offset = _coerce_float(session_meta.get("rotation_offset_deg"))
    if heading is not None and offset is not None:
        heading = _normalize_heading_deg(heading + offset)
    return heading


def _read_dji_xmp_meta(info: dict) -> Dict[str, float]:
    if not info:
        return {}
    xml_blob = None
    for key in ("XML:com.adobe.xmp", "xmp", "XMP"):
        if key in info:
            xml_blob = info[key]
            break
    if not xml_blob:
        return {}
    try:
        root = ET.fromstring(xml_blob)
    except Exception:
        return {}
    out: Dict[str, float] = {}
    def _store(tag_name: str, value: str):
        if not value:
            return
        try:
            val = float(value)
        except Exception:
            return
        if tag_name == "FlightYawDegree":
            out["flight_yaw"] = val
        elif tag_name == "GimbalYawDegree":
            out["gimbal_yaw"] = val
        elif tag_name == "RelativeAltitude":
            out["relative_altitude"] = val
        elif tag_name == "AbsoluteAltitude":
            out["absolute_altitude"] = val
        elif tag_name == "GimbalRollDegree":
            out["gimbal_roll"] = val

    for elem in root.iter():
        tag = elem.tag.rsplit('}', 1)[-1]
        text = (elem.text or "").strip()
        if text:
            _store(tag, text)
        for attr_key, attr_val in elem.attrib.items():
            attr_tag = attr_key.rsplit('}', 1)[-1]
            _store(attr_tag, (attr_val or "").strip())
    return out


def _compute_meters_per_pixel(
    altitude_m: Optional[float],
    width_px: Optional[int],
    focal_length_mm: Optional[float],
    focal_length_35mm: Optional[float],
    focal_plane_x_res: Optional[float],
    focal_plane_res_unit: Optional[int],
) -> Optional[float]:
    if not altitude_m or altitude_m <= 0 or not width_px or width_px <= 0:
        return None
    if focal_length_35mm and focal_length_35mm > 1e-6:
        try:
            return (float(altitude_m) * 36.0) / (float(focal_length_35mm) * float(width_px))
        except Exception:
            return None
    if focal_length_mm and focal_plane_x_res and focal_plane_x_res > 0 and focal_plane_res_unit:
        unit = int(focal_plane_res_unit)
        per_mm = None
        if unit == 2:      # inches
            per_mm = float(focal_plane_x_res) / 25.4
        elif unit == 3:    # centimeters
            per_mm = float(focal_plane_x_res) / 10.0
        elif unit == 4:    # millimeters
            per_mm = float(focal_plane_x_res)
        if per_mm and per_mm > 0:
            pixel_size_mm = 1.0 / per_mm
            try:
                return (float(altitude_m) * (pixel_size_mm / 1000.0)) / float(focal_length_mm)
            except Exception:
                return None
    return None


def _extract_camera_meta_entry(img_path: Path) -> Optional[Dict[str, Any]]:
    try:
        with Image.open(img_path) as img:
            width, height = img.size
            info = dict(img.info) if img.info else {}
            exif = img._getexif() or {}
    except Exception as exc:
        logger.debug("ignored EXIF parse error for %s: %s", img_path, exc)
        return None

    gps = exif.get(_EXIF_GPS_TAG) if _EXIF_GPS_TAG and exif else None
    lat = lon = None
    alt_from_gps = None
    heading = None
    if gps:
        lat, lon = _latlon_from_gps(gps)
        alt_val = _gps_value(gps, "GPSAltitude")
        if alt_val is not None:
            try:
                alt_from_gps = _to_float_ratio(alt_val)
                alt_ref = _gps_value(gps, "GPSAltitudeRef")
                if alt_ref in (1, b"\x01"):
                    alt_from_gps = -alt_from_gps
            except Exception:
                alt_from_gps = None
        dir_val = _gps_value(gps, "GPSImgDirection")
        if dir_val is not None:
            try:
                heading = _normalize_heading_deg(_to_float_ratio(dir_val))
            except Exception:
                heading = None

    xmp_meta = _read_dji_xmp_meta(info)
    flight_yaw = _normalize_heading_deg(xmp_meta.get("flight_yaw"))
    gimbal_yaw = _normalize_heading_deg(xmp_meta.get("gimbal_yaw"))
    if heading is None:
        # fall back to aircraft/gimbal headings if GPS direction missing
        heading = flight_yaw if flight_yaw is not None else gimbal_yaw

    altitude_m = xmp_meta.get("relative_altitude")
    if altitude_m is None:
        altitude_m = xmp_meta.get("absolute_altitude")
    if altitude_m is None:
        altitude_m = alt_from_gps

    focal_length_mm = None
    focal_length_35mm = None
    focal_plane_x_res = None
    focal_plane_res_unit = None
    capture_ts = None
    if exif:
        if exif.get(_EXIF_FOCAL_LENGTH_TAG) is not None:
            try:
                focal_length_mm = float(_to_float_ratio(exif.get(_EXIF_FOCAL_LENGTH_TAG)))
            except Exception:
                focal_length_mm = None
        if exif.get(_EXIF_FOCAL_LENGTH_35MM_TAG) is not None:
            try:
                focal_length_35mm = float(exif.get(_EXIF_FOCAL_LENGTH_35MM_TAG))
            except Exception:
                focal_length_35mm = None
        if exif.get(_EXIF_FOCAL_PLANE_X_RES_TAG) is not None:
            try:
                focal_plane_x_res = float(_to_float_ratio(exif.get(_EXIF_FOCAL_PLANE_X_RES_TAG)))
            except Exception:
                focal_plane_x_res = None
        if exif.get(_EXIF_FOCAL_PLANE_RES_UNIT_TAG) is not None:
            try:
                focal_plane_res_unit = int(exif.get(_EXIF_FOCAL_PLANE_RES_UNIT_TAG))
            except Exception:
                focal_plane_res_unit = None
        for tag in (_EXIF_DATETIME_ORIGINAL_TAG, _EXIF_DATETIME_TAG, _EXIF_DATETIME_DIGITIZED_TAG):
            if exif.get(tag) is not None:
                capture_ts = _parse_exif_timestamp(exif.get(tag))
                if capture_ts is not None:
                    break

    meters_per_pixel = _compute_meters_per_pixel(
        altitude_m, width, focal_length_mm, focal_length_35mm,
        focal_plane_x_res, focal_plane_res_unit,
    )

    entry: Dict[str, Any] = {
        "file": img_path.name,
        "w_px": int(width),
        "h_px": int(height),
    }
    if lat is not None and lon is not None:
        entry["lat"] = float(lat)
        entry["lon"] = float(lon)
    if altitude_m is not None:
        entry["alt"] = float(altitude_m)
    preferred_rot = None
    if gimbal_yaw is not None:
        entry["rotation_gimbal"] = float(gimbal_yaw)
        preferred_rot = float(gimbal_yaw)
    if flight_yaw is not None:
        entry["rotation_aircraft"] = float(flight_yaw)
        if preferred_rot is None:
            preferred_rot = float(flight_yaw)
    if preferred_rot is None and heading is not None:
        preferred_rot = float(heading)
    if preferred_rot is not None:
        entry["rotation"] = float(preferred_rot)
    if meters_per_pixel is not None and meters_per_pixel > 0:
        entry["meters_per_pixel"] = float(meters_per_pixel)
    if capture_ts is not None:
        entry["timestamp"] = float(capture_ts)
        entry["timestamp_source"] = "exif_datetime"
    if xmp_meta.get("relative_altitude") is not None:
        entry["alt_source"] = "relative_altitude"
    elif alt_from_gps is not None:
        entry["alt_source"] = "gps"
    if meters_per_pixel is not None:
        if xmp_meta.get("relative_altitude") is not None:
            entry["meters_per_pixel_source"] = "relative_altitude"
        elif focal_length_mm is not None or focal_length_35mm is not None:
            entry["meters_per_pixel_source"] = "exif_optics"
    if xmp_meta.get("gimbal_yaw") is not None:
        entry["rotation_source"] = "gimbal_yaw"
    elif xmp_meta.get("flight_yaw") is not None:
        entry["rotation_source"] = "flight_yaw"
    elif heading is not None:
        entry["rotation_source"] = "gps_img_direction"
    return entry


def _build_camera_meta_from_exif(images_dir: Path) -> Dict[str, Dict[str, Any]]:
    meta: Dict[str, Dict[str, Any]] = {}
    if not images_dir or not images_dir.exists():
        return meta
    order_idx: Dict[str, int] = {}
    for order, img_path in enumerate(sorted(images_dir.iterdir())):
        if not img_path.is_file():
            continue
        if img_path.suffix not in IMAGE_EXTS:
            continue
        entry = _extract_camera_meta_entry(img_path)
        if entry:
            meta[img_path.name] = entry
            order_idx[img_path.name] = order
    if meta:
        _augment_camera_rotations_from_track(meta, order_idx)
    return meta



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

def _scan_image_sizes(images_dir: Path) -> dict[str, tuple[int, int]]:
    """Return {'filename': (width, height), ...} without touching EXIF."""
    out: dict[str, tuple[int, int]] = {}
    for p in images_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"):
            continue
        try:
            from PIL import Image
            with Image.open(p) as im:
                w, h = im.size
            out[p.name] = (int(w), int(h))
        except Exception as e:
            logger.debug("ignored web.app error: %s", e)
    return out


def _lookup_camera_meta_entry(camera_meta: Dict[str, Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    if not camera_meta or not name:
        return None
    if name in camera_meta:
        entry = camera_meta.get(name)
        if isinstance(entry, dict) and not str(name).startswith("__"):
            return entry
    try:
        name_only = Path(name).name
        if name_only in camera_meta:
            entry = camera_meta.get(name_only)
            if isinstance(entry, dict) and not str(name_only).startswith("__"):
                return entry
    except Exception:
        pass
    stem = None
    try:
        stem = Path(name).stem
    except Exception:
        stem = None
    if not stem:
        return None
    for key, value in camera_meta.items():
        if not isinstance(value, dict):
            continue
        if str(key).startswith("__"):
            continue
        try:
            if Path(key).stem == stem:
                return value
        except Exception:
            continue
    return None


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> Optional[float]:
    try:
        lat1_rad = math.radians(float(lat1))
        lat2_rad = math.radians(float(lat2))
        dlon = math.radians(float(lon2) - float(lon1))
        x = math.sin(dlon) * math.cos(lat2_rad)
        y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
        if abs(x) < 1e-9 and abs(y) < 1e-9:
            return None
        bearing = math.degrees(math.atan2(x, y))
        if bearing < 0:
            bearing += 360.0
        return bearing
    except Exception:
        return None


def _augment_camera_rotations_from_track(
    camera_meta: Dict[str, Dict[str, Any]],
    order_idx: Dict[str, int],
) -> None:
    if not camera_meta:
        return

    seq = []
    for name, meta in camera_meta.items():
        lat = meta.get("lat")
        lon = meta.get("lon")
        if lat is None or lon is None:
            continue
        ts_val = meta.get("timestamp")
        try:
            ts_float = float(ts_val)
        except Exception:
            ts_float = None
        ord_idx = order_idx.get(name)
        if ord_idx is None:
            try:
                ord_idx = order_idx.get(Path(name).name)
            except Exception:
                ord_idx = None
        seq.append({
            "name": name,
            "lat": float(lat),
            "lon": float(lon),
            "timestamp": ts_float,
            "order": ord_idx if ord_idx is not None else float("inf"),
        })

    if len(seq) < 2:
        return

    seq.sort(key=lambda item: (
        item["timestamp"] if item["timestamp"] is not None else float("inf"),
        item["order"],
        item["name"],
    ))

    for idx, current in enumerate(seq):
        meta = camera_meta.get(current["name"])
        if not meta or meta.get("rotation") is not None:
            continue

        bearings: list[float] = []
        if idx > 0:
            prev = seq[idx - 1]
            b = _bearing_deg(prev["lat"], prev["lon"], current["lat"], current["lon"])
            if b is not None:
                bearings.append(b)
        if idx + 1 < len(seq):
            nxt = seq[idx + 1]
            b = _bearing_deg(current["lat"], current["lon"], nxt["lat"], nxt["lon"])
            if b is not None:
                bearings.append(b)

        if not bearings:
            continue
        sin_sum = sum(math.sin(math.radians(b)) for b in bearings)
        cos_sum = sum(math.cos(math.radians(b)) for b in bearings)
        if abs(sin_sum) < 1e-6 and abs(cos_sum) < 1e-6:
            continue
        avg = math.degrees(math.atan2(sin_sum, cos_sum))
        meta["rotation"] = float(_normalize_heading_deg(avg))
        if not meta.get("rotation_source"):
            meta["rotation_source"] = "track_bearing"


def _preds_to_geojson(
    images_dir: Path,
    preds_dir: Path,
    out_session: Path,
    class_names: List[str],
    score_thresh: float = 0.0,
    meters_per_pixel: float = 0.05,
    exif_index: Optional[Dict[str, Tuple[float, float]]] = None,  # {'file': (lat, lon)}
    camera_meta: Optional[Dict[str, Dict]] = None,
) -> Tuple[Path, Path]:
    """
    Build:
      - images.geojson: points at image GPS with useful props (image name, overlay, thumb if available)
      - anomalies.geojson: bbox polygons converted to WGS84 using center-based pixel→meter→degree math
    """
    from shapely.geometry import Polygon, mapping

    out_session = Path(out_session)
    out_session.mkdir(parents=True, exist_ok=True)

    # 1) GPS (lat,lon) for originals
    gps_index = exif_index or _scan_exif_latlon(images_dir)  # {'file.jpg': (lat,lon)}

    # 2) Image sizes (w,h) for center-based conversion
    sizes_index = _scan_image_sizes(images_dir)              # {'file.jpg': (w,h)}

    # 3) (optional) overlay/thumb URLs from manifest.json (generated during inference)
    manifest_map: Dict[str, Dict[str, str]] = {}
    mpath = out_session / "manifest.json"
    if mpath.exists():
        try:
            manifest_map = json.loads(mpath.read_text(encoding="utf-8"))
        except Exception:
            manifest_map = {}

    # ---------------- images.geojson (points + optional footprint corners) ----------------
    imgs_fc = {"type": "FeatureCollection", "features": []}
    camera_meta = camera_meta or {}
    session_meta = _camera_meta_session_meta(camera_meta)
    camera_meta_keys = {
        key for key, value in camera_meta.items()
        if isinstance(value, dict) and not str(key).startswith("__")
    }
    default_mpp = float(meters_per_pixel or 0.05)

    def _coerce_positive_float(val: Optional[float]) -> Optional[float]:
        try:
            fval = float(val)
        except Exception:
            return None
        return fval if fval > 0 else None

    def _camera_entry_for(name: str) -> Optional[Dict[str, Any]]:
        if not camera_meta:
            return None
        return _lookup_camera_meta_entry(camera_meta, name)

    # Collect candidate filenames from manifest, exif, sizes, and camera_meta
    candidates = set(manifest_map.keys()) | set(gps_index.keys()) | set(sizes_index.keys()) | set(camera_meta_keys)

    for fname in sorted(candidates):
        cam_entry = _camera_entry_for(fname)

        # Determine center lat/lon (priority: camera_meta -> manifest_map -> exif_index)
        latlon = None
        if cam_entry:
            lat_val = cam_entry.get("lat")
            lon_val = cam_entry.get("lon")
            if lat_val is not None and lon_val is not None:
                latlon = (float(lat_val), float(lon_val))
        if latlon is None and isinstance(manifest_map.get(fname), dict):
            e = manifest_map.get(fname)
            if "lat" in e and "lon" in e:
                latlon = (float(e["lat"]), float(e["lon"]))
        if latlon is None:
            latlon = gps_index.get(fname)

        props: Dict[str, object] = {}
        # overlay/thumb from manifest
        if isinstance(manifest_map.get(fname), dict):
            entry = manifest_map[fname]
            if "overlay" in entry:
                props["image"] = Path(entry["overlay"]).name
            if "thumb" in entry:
                props["thumb"] = entry["thumb"]

        # width/height in pixels
        w_px = h_px = None
        if cam_entry:
            if cam_entry.get("w_px"):
                w_px = int(cam_entry.get("w_px"))
            if cam_entry.get("h_px"):
                h_px = int(cam_entry.get("h_px"))
        if w_px is None and isinstance(manifest_map.get(fname), dict):
            ent = manifest_map.get(fname)
            if "w" in ent and "h" in ent:
                w_px = int(ent["w"]); h_px = int(ent["h"])
        if w_px is None and fname in sizes_index:
            w_px, h_px = sizes_index.get(fname)

        entry_mpp = _coerce_positive_float(cam_entry.get("meters_per_pixel")) if cam_entry else None
        image_mpp = entry_mpp or default_mpp
        props["meters_per_pixel"] = float(image_mpp)

        # basic props
        if w_px and h_px:
            props["w"] = int(w_px); props["h"] = int(h_px)

        # If we have a center and pixel dims, compute conservative axis-aligned footprint
        if latlon and w_px and h_px:
            lat_c, lon_c = latlon[0], latlon[1]
            deg_per_m_lon, deg_per_m_lat = _meters_to_deg(lat_c)
            width_m = float(w_px) * float(image_mpp)
            height_m = float(h_px) * float(image_mpp)
            half_w = width_m / 2.0
            half_h = height_m / 2.0
            dx_lon = half_w * deg_per_m_lon
            dy_lat = half_h * deg_per_m_lat

            # corners UL, UR, LR, LL (axis-aligned). If camera rotation is provided, keep rotation in props.
            # compute axis-aligned corner offsets (meters) relative to image center
            props["width_m"] = float(width_m)
            props["height_m"] = float(height_m)
            half_w_m = width_m / 2.0
            half_h_m = height_m / 2.0

            # Rotation comes from camera metadata (or session defaults) and is expressed
            # as camera heading (0°=North, +CW). Compute separate overlay rotation for
            # PNG generation while keeping geo math aligned to heading.
            heading_deg = _camera_heading_from_entry(cam_entry, session_meta)
            rot_overlay = _camera_heading_to_overlay_rotation(heading_deg)
            rot_for_geo = heading_deg if heading_deg is not None else 0.0
            props["rotation"] = float(rot_for_geo)
            if heading_deg is not None:
                props["rotation_heading"] = float(heading_deg)
            props["rotation_overlay"] = float(rot_overlay)

            # Build the four corner coordinates using the same pixel→meter→deg
            # convention used when converting detection boxes to geo-polygons.
            # This ensures image footprints and anomaly reprojections share the
            # same rotation/sign conventions and will align correctly.
            try:
                # need pixel sizes to compute pixel offsets
                if w_px is None or h_px is None:
                    # fallback to computed meters-based axis-aligned corners
                    ul = (lon_c - dx_lon, lat_c + dy_lat)
                    ur = (lon_c + dx_lon, lat_c + dy_lat)
                    lr = (lon_c + dx_lon, lat_c - dy_lat)
                    ll = (lon_c - dx_lon, lat_c - dy_lat)
                    props["corners"] = [list(ul), list(ur), list(lr), list(ll)]
                else:
                    cx = float(w_px) / 2.0
                    cy = float(h_px) / 2.0
                    # pixel corner coordinates (x,y): TL(0,0), TR(w,0), BR(w,h), BL(0,h)
                    pix_corners = [(0.0, 0.0), (float(w_px), 0.0), (float(w_px), float(h_px)), (0.0, float(h_px))]
                    a = math.radians(float(rot_for_geo))
                    ca = math.cos(a); sa = math.sin(a)
                    out_corners = []
                    for (px, py) in pix_corners:
                        dx_m = (px - cx) * float(image_mpp)
                        dy_m = (py - cy) * float(image_mpp)
                        # apply rotation (same as used for boxes)
                        rx = dx_m * ca - dy_m * sa
                        ry = dx_m * sa + dy_m * ca
                        lon_p = lon_c + (rx * deg_per_m_lon)
                        lat_p = lat_c - (ry * deg_per_m_lat)
                        out_corners.append((lon_p, lat_p))
                    props["corners"] = [list(c) for c in out_corners]
            except Exception:
                ul = (lon_c - dx_lon, lat_c + dy_lat)
                ur = (lon_c + dx_lon, lat_c + dy_lat)
                lr = (lon_c + dx_lon, lat_c - dy_lat)
                ll = (lon_c - dx_lon, lat_c - dy_lat)
                props["corners"] = [list(ul), list(ur), list(lr), list(ll)]

        # Only include features that have a geometry (latlon). Otherwise skip.
        if latlon:
            # preserve source filename so downstream reprojection can look up rotation
            props["src"] = fname
            imgs_fc["features"].append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(latlon[1]), float(latlon[0])]},
                "properties": props
            })

    imgs_path = out_session / "images.geojson"
    imgs_path.write_text(json.dumps(imgs_fc, indent=2), encoding="utf-8")

    # Build a quick lookup of per-image rotation from the images.geojson features
    image_rot_map: Dict[str, float] = {}
    for f in imgs_fc.get("features", []):
        try:
            src = f.get("properties", {}).get("src")
            rot = float(f.get("properties", {}).get("rotation") or 0.0)
            if src:
                image_rot_map[src] = rot
                # also map by stem (without extension) for flexible matching
                try:
                    image_rot_map[Path(src).stem] = rot
                except Exception:
                    pass
        except Exception:
            continue

    # ---------------- anomalies.geojson (bbox → polygon using image center) ----------------
    anom_fc = {"type": "FeatureCollection", "features": []}

    preds_json_dir = Path(preds_dir) / "preds"
    for jpath in sorted(preds_json_dir.glob("*.json")):
        try:
            jd = json.loads(jpath.read_text(encoding="utf-8"))
        except Exception:
            continue

        boxes   = jd.get("boxes", []) or []
        scores  = jd.get("scores", []) or []
        classes = jd.get("classes", []) or []
        srcfile = jd.get("file") or (jpath.stem + ".png")

        # Resolve GPS & size for this source image (try exact, then by stem with common extensions)
        latlon = gps_index.get(srcfile)
        wh     = sizes_index.get(srcfile)

        if not latlon or not wh:
            stem = Path(srcfile).stem
            for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
                latlon = latlon or gps_index.get(stem + ext)
                wh     = wh     or sizes_index.get(stem + ext)
                if latlon and wh:
                    break

        if not latlon or not wh:
            # no GPS or no (w,h): we can't georeference these detections
            continue

        lat, lon = latlon
        w, h = wh

        # Degrees per meter at this latitude
        deg_per_m_lon, deg_per_m_lat = _meters_to_deg(lat)

        # Prepare optional per-image camera rotation and GSD if available
        box_mpp = default_mpp
        cam_entry = _camera_entry_for(srcfile)
        heading_deg = _camera_heading_from_entry(cam_entry, session_meta)
        if cam_entry:
            entry_mpp = _coerce_positive_float(cam_entry.get("meters_per_pixel"))
            if entry_mpp is not None:
                box_mpp = entry_mpp

        if heading_deg is None and 'image_rot_map' in locals():
            try:
                if srcfile in image_rot_map:
                    heading_deg = float(image_rot_map.get(srcfile) or 0.0)
                else:
                    stem = Path(srcfile).stem
                    if stem in image_rot_map:
                        heading_deg = float(image_rot_map.get(stem) or 0.0)
            except Exception:
                heading_deg = None

        rotation_deg = float(heading_deg) if heading_deg is not None else 0.0

        # Images are rotated to north-up before inference, so anomalies don't need rotation
        rotation_deg = 0.0

        # Convert each box using CENTER-based deltas (matches reference notebook)
        cx = w / 2.0
        cy = h / 2.0

        for i, b in enumerate(boxes):
            sc = float(scores[i]) if i < len(scores) else 0.0
            if sc < score_thresh:
                continue
            cls_id = int(classes[i]) if i < len(classes) else 0
            cname  = class_names[cls_id] if 0 <= cls_id < len(class_names) else f"cls_{cls_id}"

            x0, y0, x1, y1 = map(float, b)

            # pixel deltas from image center → meters
            dx0_m = (x0 - cx) * box_mpp
            dx1_m = (x1 - cx) * box_mpp
            dy0_m = (y0 - cy) * box_mpp
            dy1_m = (y1 - cy) * box_mpp

            # apply image rotation (if camera metadata provided)
            if rotation_deg and abs(rotation_deg) > 1e-6:
                a = math.radians(rotation_deg)
                ca = math.cos(a); sa = math.sin(a)
                r0x = dx0_m * ca - dy0_m * sa
                r0y = dx0_m * sa + dy0_m * ca
                r1x = dx1_m * ca - dy1_m * sa
                r1y = dx1_m * sa + dy1_m * ca
            else:
                r0x, r0y, r1x, r1y = dx0_m, dy0_m, dx1_m, dy1_m

            # convert the four box corners (x0,y0),(x1,y0),(x1,y1),(x0,y1)
            # to rotated geographic polygon points so the anomaly polygon
            # follows the same image rotation convention as the image
            # footprints. This produces a rotated polygon (not an axis-aligned bbox).
            try:
                corners_px = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
                poly_pts = []
                for (px, py) in corners_px:
                    dx_m = (px - cx) * box_mpp
                    dy_m = (py - cy) * box_mpp
                    if rotation_deg and abs(rotation_deg) > 1e-6:
                        rx = dx_m * ca - dy_m * sa
                        ry = dx_m * sa + dy_m * ca
                    else:
                        rx, ry = dx_m, dy_m
                    lon_p = lon + (rx * deg_per_m_lon)
                    lat_p = lat - (ry * deg_per_m_lat)
                    poly_pts.append((lon_p, lat_p))
                # close polygon
                if poly_pts and poly_pts[0] != poly_pts[-1]:
                    poly = Polygon(poly_pts + [poly_pts[0]])
                else:
                    poly = Polygon(poly_pts)
            except Exception:
                # fallback to axis-aligned bbox if anything fails
                lon0 = lon + (r0x * deg_per_m_lon)
                lon1 = lon + (r1x * deg_per_m_lon)
                lat0 = lat - (r0y * deg_per_m_lat)
                lat1 = lat - (r1y * deg_per_m_lat)
                poly = Polygon([
                    (lon0, lat0), (lon0, lat1),
                    (lon1, lat1), (lon1, lat0), (lon0, lat0)
                ])

            anom_fc["features"].append({
                "type": "Feature",
                "geometry": mapping(poly),
                "properties": {
                    "class": cls_id,
                    "classname": cname,
                    "score": round(sc * 100.0, 2),
                    "image": srcfile,
                }
            })

    anom_path = out_session / "anomalies.geojson"
    anom_path.write_text(json.dumps(anom_fc, indent=2), encoding="utf-8")

    return anom_path, imgs_path






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
            "colmap_ready": _colmap_ready(p.name),
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
    for name in ("detectron2", "fvcore", "fvcore.common.checkpoint", "torch", "ultralytics"):
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
    rotated  = session_dir / "rotated_images"
    def _urls(d: Path):
        if not d.exists(): return []
        return [f"/media/{d.relative_to(MEDIA_DIR)}/{p.name}" for p in sorted(d.glob("*")) if p.is_file()]
    tifs = [u for u in _urls(imgs_dir) if u.lower().endswith((".tif", ".tiff"))]
    return {
        "images": _urls(imgs_dir),
        "tifs": tifs,
        "overlays": _urls(overlays),
        "thumbs": _urls(thumbs),
        "rotated_images": _urls(rotated),
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
    model_name: str = Form("") ,
    backend: str = Form("detectron"),
    model_type: str = Form("fasterrcnn"),
    yolo_family: str = Form("v8"),
    yolo_seg: bool = Form(False),
    yolo_size: str = Form("s"),
    selected_bands: str = Form(None),
    channel_count: int = Form(3),
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
                # Also persist logs to a per-run file so logs can be inspected later.
            # A FileHandler is attached to the root logger for the duration of this run.
            fh = None
            root_logger = logging.getLogger()
            try:
                train_log_path = run_dir / "train.log"
                fh = logging.FileHandler(str(train_log_path), mode="a", encoding="utf-8")
                fh.setLevel(logging.DEBUG)
                fh.setFormatter(logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s", "%m/%d %H:%M:%S"))
                root_logger.addHandler(fh)
                # Also attach the per-run file handler to our 'pvrt' logger
                # so that SSE-forwarded logs and redirected stdout/stderr (which
                # are routed to 'pvrt') are persisted to the per-run file.
                pvrt_logger = logging.getLogger("pvrt")
                pvrt_logger.addHandler(fh)
                root_logger.debug(f"Per-run logging started -> {train_log_path}")
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
    model_type=model_type,
    yolo_family=yolo_family,
    yolo_seg=yolo_seg,
    yolo_size=yolo_size,
    selected_bands=[b.strip() for b in (selected_bands.split(',') if selected_bands else [])] or None,
    channel_count=int(channel_count or 3),

            )
            finally:
                try:
                    if fh:
                        root_logger.debug(f"Per-run logging stopping -> {train_log_path}")
                        try:
                            pvrt_logger.removeHandler(fh)
                        except Exception as e:
                            logger.debug("ignored web.app error: %s", e)
                        try:
                            root_logger.removeHandler(fh)
                        except Exception as e:
                            logger.debug("ignored web.app error: %s", e)
                        try:
                            fh.flush()
                        except Exception as e:
                            logger.debug("ignored web.app error: %s", e)
                        try:
                            fh.close()
                        except Exception as e:
                            logger.debug("ignored web.app error: %s", e)
                except Exception:
                    logging.getLogger("pvrt").exception("Failed to remove/close per-run file handler")

    try:
        resp = await asyncio.to_thread(_do_train)  # <-- key change
        meta = resp.get("meta", {})
        logger.info(f"[train] complete: run={run_dir.name}")
        logger.info("UI:OK:train: Training completed.")
        return {"ok": True, "run": run_dir.name, "meta": meta}
    except Exception as e:
        # Log full traceback and include it in the HTTP error detail to aid debugging.
        import traceback
        tb = traceback.format_exc()
        logger.error("Training failed: %s", e)
        logger.error(tb)
        # Also persist to per-run train log if available
        try:
            # train_log_path may exist in this scope via closure in _do_train; attempt best-effort
            train_log = run_dir / "train.log"
            with open(train_log, "a", encoding="utf-8") as fh:
                fh.write("[TRAIN-ERROR] " + tb + "\n")
        except Exception:
            # non-fatal if we couldn't write the per-run file
            logger.debug("Could not append traceback to per-run train.log")
        # Surface the error back to the caller with a concise message and the last line of the traceback
        last_line = tb.strip().splitlines()[-1] if tb else str(e)
        raise HTTPException(status_code=500, detail=f"Training failed: {last_line}")


# -------------- List model runs --------------

@app.get("/api/models")
async def api_models(backend: Optional[str] = None):
    """List model runs. If `backend` query param is provided, return only models
    whose metadata `backend` matches (case-insensitive). This makes the
    frontend filtering reliable even when client-side heuristics fail.
    """
    models = _list_models()
    if backend:
        try:
            b = str(backend).lower()
            models = [m for m in models if str(m.get("backend", "")).lower() == b]
        except Exception:
            pass
    return {"ok": True, "models": models}


# ================== TEST: dataset intake ==================

@app.get("/api/test_datasets")
async def api_test_datasets():
    details = _list_datasets()                      # current shape: [{name, count, mtime}, ...]
    names = [d["name"] for d in details]           # simple shape: ["name", ...]
    return {"ok": True, "datasets": details, "dataset_names": names}


@app.get("/api/dataset_bands")
async def api_dataset_bands(dataset: str):
    """Return detected bands for a dataset folder under data/test or data/train.
    Query param `dataset` may be a folder name under data/test or the literal 'train'/'valid'.
    """
    # resolve dataset path
    ds = None
    if dataset in ("train", "valid"):
        base = PROJECT_ROOT / "data"
        ds = base / dataset
    else:
        ds = TEST_DIR / dataset

    if not ds or not ds.exists():
        return {"ok": False, "error": "dataset_not_found", "dataset": dataset}

    # simple detection: check for thermal subdir and presence of RGB-like images
    bands = []
    examples = {}
    try:
        if (ds / "thermal").exists():
            bands.append("thermal")
            # collect some examples
            ex = [str(p.relative_to(PROJECT_ROOT)) for p in sorted((ds / "thermal").glob("*"))[:5] if p.is_file()]
            examples["thermal"] = ex

        # rgb if top-level images exist
        imgs = [p for p in ds.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        if imgs:
            bands.insert(0, "rgb")
            examples["rgb"] = [str(p.relative_to(PROJECT_ROOT)) for p in imgs[:5]]

        # if TIFFs available, expose tif as band candidate
        tifs = [p for p in ds.glob("*.tif")]
        if tifs and "rgb" not in bands:
            bands.insert(0, "tif")
            examples["tif"] = [str(p.relative_to(PROJECT_ROOT)) for p in tifs[:5]]

    except Exception as e:
        return {"ok": False, "error": str(e)}

    return {"ok": True, "dataset": str(ds), "bands": bands, "examples": examples}


@app.post("/api/decode_dataset")
async def api_decode_dataset(dataset: str = Form(...)):
    """Trigger thermal decoding for a dataset (images pipeline). Returns updated band list."""
    # resolve dataset path similarly to above
    if dataset in ("train", "valid"):
        ds = PROJECT_ROOT / "data" / dataset
    else:
        ds = TEST_DIR / dataset

    if not ds.exists():
        raise HTTPException(status_code=404, detail="dataset not found")

    try:
        ensure_dirp_init()
        scan_split_decode_thermal(ds)
    except Exception as e:
        logger.exception("decode failed")
        raise HTTPException(status_code=500, detail=f"decode failed: {e}")

    
    return await api_dataset_bands(dataset)

@app.post("/api/test_upload")
async def api_test_upload_underscore(
    files: List[UploadFile] = File(...),
    result_name: str = Form(""),
):
    """
    Upload handler for testing:
      - If uploading non-zip images → single dataset folder (unchanged).
      - If uploading TIF(s) (direct or inside a zip) → ONE DATASET PER TIF (folder name ≈ tif stem).
    """
    logger = logging.getLogger("pvrt.test")

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    base = _safe_name(result_name) or _safe_name(files[0].filename or "upload")
    created: List[str] = []

    # Partition the upload
    zip_files  = [f for f in files if (f.filename or "").lower().endswith(".zip")]
    other_files = [f for f in files if f not in zip_files]

    # ----------- handle zips -----------
    for zf in zip_files:
        # extract to a temporary staging dir under data/test/<temp>
        staging = _unique_dataset_dir(_safe_name(Path(zf.filename or "archive.zip").stem) or "zip")
        staging.mkdir(parents=True, exist_ok=True)
        buf = zf.file.read()
        try:
            with zipfile.ZipFile(io.BytesIO(buf)) as z:
                z.extractall(staging)
        except Exception as e:
            shutil.rmtree(staging, ignore_errors=True)
            raise HTTPException(status_code=400, detail=f"Bad ZIP '{zf.filename}': {e}")

        # flatten one folder if needed
        kids = list(staging.iterdir())
        if len(kids) == 1 and kids[0].is_dir():
            inner = kids[0]
            for p in inner.iterdir():
                shutil.move(str(p), str(staging))
            inner.rmdir()

        # split: every TIF becomes its own dataset; non-TIF images coalesce into one images dataset
        tifs = list(_iter_geotiffs(staging))
        non_tif_imgs = [p for p in staging.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS and p.suffix.lower() not in (".tif",".tiff")]

        # 2a) make one dataset per TIF
        for tif in tifs:
            name = _safe_name(tif.stem) or "tif"
            ds_dir = _unique_dataset_dir(name)
            ds_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(tif), str(ds_dir / tif.name))
            created.append(ds_dir.name)

        # 2b) if there are regular images, keep them together (old behavior)
        if non_tif_imgs:
            name = _safe_name(base) or "images"
            ds_dir = _unique_dataset_dir(name)
            ds_dir.mkdir(parents=True, exist_ok=True)
            for p in non_tif_imgs:
                shutil.move(str(p), str(ds_dir / p.name))
            created.append(ds_dir.name)

        # clean staging
        shutil.rmtree(staging, ignore_errors=True)

    # ----------- handle direct files (non-zip) -----------
    if other_files:
        # separate TIF vs regular images
        direct_tifs = [f for f in other_files if Path(f.filename or "").suffix.lower() in (".tif",".tiff")]
        direct_imgs = [f for f in other_files if Path(f.filename or "").suffix.lower() in IMAGE_EXTS and Path(f.filename or "").suffix.lower() not in (".tif",".tiff")]

        # 3a) one dataset per TIF
        for f in direct_tifs:
            stem = _safe_name(Path(f.filename or "image.tif").stem) or "tif"
            ds_dir = _unique_dataset_dir(stem)
            ds_dir.mkdir(parents=True, exist_ok=True)
            (ds_dir / Path(f.filename).name).write_bytes(f.file.read())
            created.append(ds_dir.name)

        # 3b) regular images together
        if direct_imgs:
            name = _safe_name(base) or "images"
            ds_dir = _unique_dataset_dir(name)
            ds_dir.mkdir(parents=True, exist_ok=True)
            for f in direct_imgs:
                (ds_dir / Path(f.filename).name).write_bytes(f.file.read())
            created.append(ds_dir.name)

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
    backend: Optional[str] = Form(default=None),
    selected_bands: str = Form(None),
    channel_count: int = Form(3),
    accurate_locations: bool = Form(default=False),
    mosaic_enabled: bool = Form(default=False),
    optimization_project: Optional[str] = Form(default=None),
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

    # --- ADD: decide whether this dataset is a single GeoTIFF ---
    input_type = _detect_image_input_type(ds_dir)

    accurate_locations = bool(accurate_locations)
    if accurate_locations and input_type != "images":
        raise HTTPException(status_code=400, detail="Accurate locations are only supported for image datasets.")

    # Default run directory is the original dataset
    run_images_dir = ds_dir
    tiles_dir = None
    tif_src = None
    tif_has_thermal = False  # only meaningful for 'tif'

    if input_type == "images":
        # For image datasets we will *infer* thermal needs from the model metadata.
        # If the selected model was trained for RGB+Thermal (rgbt/4ch) then attempt
        # an idempotent decode pass which will create thermal/ + pairs.json when
        # RJPEG payloads are present. The decode helper is safe to call repeatedly
        # and will early-exit if DIRP isn't available.
        pass
    else:
        # input_type == "tif"
        tifs = [p for p in ds_dir.glob("*") if p.suffix.lower() in (".tif", ".tiff")]
        if not tifs:
            raise HTTPException(status_code=400, detail="No GeoTIFF found in dataset.")
        tif_src = tifs[0]

        # Tile the GeoTIFF; inference runs on *tiles_dir*
        tiles_dir = out_root / "tiles"
        _tile_tif_to_dir(tif_src, tiles_dir, tile_size=1024, stride=1024)
        run_images_dir = tiles_dir

        # small preview (optional)
        thumb_path = _save_tif_thumbnail(tif_src, out_root / "thumbs")

        # Discover if the TIF has band-4 thermal (no DIRP step for mosaics)
        try:
            tif_has_thermal = (_tif_band_count(tif_src) >= 4)
        except Exception:
            tif_has_thermal = False
        logger.info(f"UI:OK:test: Tif has thermal = {tif_has_thermal}")


    logging.getLogger("pvrt").info(
        f"TestRun: image_input_type={input_type} ds='{ds_dir.name}' tiles='{tiles_dir if input_type=='tif' else '-'}'"
    )

    # --- Decide whether to decode / use thermal for inference ---
    meta = _read_model_meta(model_dir)
    model_mode = (meta.get("input_mode") or "rgb").strip().lower()
    model_is_rgbt = model_mode in {"rgbt","rgb+thermal","rgb_thermal","thermal","4ch"}

    # Determine model's declared channel count (1,3,4) with safe default
    try:
        model_chan = int(meta.get("channel_count") or (4 if model_is_rgbt else 3))
    except Exception:
        model_chan = 3

    # If model expects thermal for inference and this is an images dataset, run
    # the idempotent decode pass which will populate images_dir/thermal/pairs.json
    # when RJPEG payloads exist. This lets us infer availability afterwards.
    # If model expects RGB+thermal, attempt the idempotent decode pass which
    # will populate images_dir/thermal/pairs.json. The decoder is safe to call
    # repeatedly and will early-exit if DIRP isn't available.
    if input_type == "images" and model_is_rgbt:
        try:
            ensure_dirp_init()
            scan_split_decode_thermal(ds_dir)
            logger.info("UI:INFO:test: thermal decode attempted for images dataset (test pipeline)")
        except Exception as e:
            # Don't fail the whole test run for DIRP issues; log and continue.
            logger.warning(f"thermal decode attempt failed: {e}")

    # Determine whether the dataset actually has thermal after any decode attempt
    if input_type == "tif":
        data_has_thermal = tif_has_thermal
        data_has_thermal_override = tif_has_thermal
    else:
        # images dir - check for thermal/pairs.json or thermal/ files
        data_has_thermal = has_thermal_for_images(ds_dir)
        data_has_thermal_override = None

    # Infer dataset channel count:
    # - images with both rgb files and thermal/ → 4
    # - images with only thermal/ → 1
    # - tif with 4+ bands flagged earlier → 4
    # - otherwise assume 3 (RGB)
    def _infer_data_channel_count(images_dir: Path, input_type: str, tif_has_thermal_flag: bool) -> int:
        try:
            if input_type == "tif":
                return 4 if tif_has_thermal_flag else 3
            # images directory
            rgb_exists = any(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
            thermal_dir = images_dir / "thermal"
            thermal_exists = thermal_dir.exists() and any(thermal_dir.iterdir())
            if thermal_exists and rgb_exists:
                return 4
            if thermal_exists and not rgb_exists:
                return 1
            return 3
        except Exception:
            return 3

    data_chan = _infer_data_channel_count(ds_dir, input_type, tif_has_thermal)

    # Note: the explicit user-requested path to convert decoded thermal TIFFs into
    # temporary 3-channel JPGs for 3-channel models has been removed. If a model
    # declares RGB+thermal inputs the decoder will still be attempted above; there
    # is no separate "extract_thermal_rgb" behavior anymore.

    # If we materialized a per-run images dir (e.g. thermal_jpg), re-infer
    # the dataset channel count and thermal availability from that run dir
    # so downstream compatibility checks use the actual run inputs.
    try:
        if run_images_dir is not None and run_images_dir.exists() and run_images_dir != ds_dir:
            # recompute data_has_thermal based on the run_images_dir and
            # infer channel count from that location (images vs tif logic
            # handled by the helper).
            data_has_thermal = has_thermal_for_images(run_images_dir)
            data_chan = _infer_data_channel_count(run_images_dir, input_type, tif_has_thermal)
            logging.getLogger("pvrt.test").info(f"UI:INFO:test: re-inferred data_chan={data_chan} after preparing run_images_dir={run_images_dir}")
    except Exception:
        # be conservative and keep previous inferences on failure
        pass

    # Use thermal only if model supports it AND data provides it
    use_thermal_effective = bool(model_is_rgbt and data_has_thermal)

    # Validate model <-> data channel compatibility and provide clear messages
    # Possible cases:
    # - model=3 and data=4 -> we can run RGB-only (ignore thermal) but warn user
    # - model=3 and data=1 -> cannot run (model expects RGB)
    # - model in {1,4} and data=3 -> cannot run (model expects thermal)
    # - model=1 and data=4 -> run using thermal-only (ok)
    # - model=4 and data=1 -> cannot run (expects both RGB+thermal)
    if model_chan == 3 and data_chan == 4:
        # dataset has thermal but model declares 3-channel inputs.
        # New behavior: if the model metadata indicates it was trained
        # to *use* thermal information encoded into 3 channels (thermal-as-RGB),
        # allow running with the thermal-as-RGB path instead of disabling
        # thermal. Otherwise preserve the previous conservative behavior
        # and warn that thermal will be ignored.
        try:
            model_supports_thermal_as_rgb = bool(meta.get("thermal_used") or str(meta.get("input_mode") or "").strip().lower() in {"thermal", "thermal-as-rgb", "thermal_rgb"})
        except Exception:
            model_supports_thermal_as_rgb = False

        if model_supports_thermal_as_rgb:
            # Use thermal grayscale replicated across RGB channels for inference.
            logger.info(f"UI:INFO:test: Model trained for 3-channel RGB but declares thermal_used; using thermal-as-RGB for inference. model={model_dir.name}")
            logging.getLogger("pvrt.test").info("UI:INFO:test: Model trained for 3-channel RGB but declares thermal_used; using thermal-as-RGB for inference.")
            use_thermal_effective = True
        else:
            # dataset has thermal but model is RGB-only; run without thermal but warn
            logger.warning(f"UI:WARN:test: Model trained for 3-channel RGB but dataset contains thermal; running in RGB-only mode (thermal ignored). model={model_dir.name}")
            logging.getLogger("pvrt.test").warning(f"UI:WARN:test: Model trained for 3-channel RGB but dataset contains thermal; running in RGB-only mode (thermal ignored).")
            # ensure we won't request thermal
            use_thermal_effective = False
    elif model_chan == 3 and data_chan == 1:
        # dataset only has thermal images — can't run an RGB model
        msg = "Model expects RGB (3-channel) but dataset contains only thermal images. Convert dataset or use a thermal-capable model."
        logger.error(f"UI:ERR:test: {msg} model={model_dir.name}")
        raise HTTPException(status_code=400, detail=msg)
    elif model_chan in (1, 4) and data_chan == 3:
        # model expects thermal but dataset has no thermal
        msg = "Model expects thermal input but dataset has no thermal images. Enable dataset decoding or choose an RGB model."
        logger.error(f"UI:ERR:test: {msg} model={model_dir.name}")
        raise HTTPException(status_code=400, detail=msg)
    elif model_chan == 1 and data_chan == 4:
        # model expects single-channel thermal; dataset has both → use thermal-only
        logging.getLogger("pvrt.test").info("UI:INFO:test: Model expects single-channel thermal; using thermal band only for inference.")
        use_thermal_effective = True
    elif model_chan == 4 and data_chan == 1:
        # model expects RGB+thermal but dataset only has thermal
        msg = "Model expects 4-channel RGB+thermal input but dataset contains only thermal. Provide RGB or retrain for single-channel thermal."
        logger.error(f"UI:ERR:test: {msg} model={model_dir.name}")
        raise HTTPException(status_code=400, detail=msg)

    logging.getLogger("pvrt.test").info(
        f"UI:INFO:test: will_request={'rgbt' if use_thermal_effective else 'rgb'}; "
        f"override={data_has_thermal_override}"
    )

    session_dir = out_root
    camera_meta: Dict[str, Any] = {}
    if accurate_locations:
        if not _colmap_ready(dataset):
            raise HTTPException(status_code=400, detail="Dataset has not been optimized yet. Run Optimize Locations first.")
        camera_meta = _load_colmap_meta(dataset)
        if not camera_meta:
            raise HTTPException(status_code=400, detail="COLMAP metadata missing. Rerun optimization before enabling accurate locations.")
        
        img_count = sum(1 for k in camera_meta.keys() if not k.startswith("__"))
        logging.getLogger("pvrt.test").info(f"UI:INFO:test: ══════════════════════════════════════════════════════")
        logging.getLogger("pvrt.test").info(f"UI:INFO:test: Using COLMAP Accurate Locations")
        logging.getLogger("pvrt.test").info(f"UI:INFO:test:   - Dataset: {dataset}")
        logging.getLogger("pvrt.test").info(f"UI:INFO:test:   - Images with poses: {img_count}")
        logging.getLogger("pvrt.test").info(f"UI:INFO:test: ══════════════════════════════════════════════════════")
        
        meta_info = camera_meta.setdefault("__meta__", {})
        if isinstance(meta_info, dict):
            meta_info["source"] = "colmap"
            meta_info["accurate_locations"] = True
        try:
            base_meta = _build_camera_meta_from_exif(ds_dir)
            for key, entry in base_meta.items():
                camera_meta.setdefault(key, entry)
        except Exception:
            pass
    elif input_type == "images":
        try:
            camera_meta = _build_camera_meta_from_exif(ds_dir)
        except Exception as e:
            logger.warning(f"Failed to derive camera metadata from EXIF: {e}")
            camera_meta = {}
    
    # Apply optimization_project merge if provided (thermal + optical geometry)
    if optimization_project and input_type == "images" and camera_meta and not accurate_locations:
        try:
            camera_meta = _merge_optical_metadata(camera_meta, optimization_project)
        except ValueError as e:
            # Match rate too low - this is expected, just use EXIF
            logging.getLogger("pvrt.test").warning(f"UI:WARN:test: Optical sync failed: {e}")
            logging.getLogger("pvrt.test").info(f"UI:INFO:test: Falling back to standard EXIF metadata")
        except Exception as e:
            logging.getLogger("pvrt.test").error(f"UI:ERROR:test: Failed to merge optimization_project metadata: {e}")
            logging.getLogger("pvrt.test").info(f"UI:INFO:test: Falling back to standard EXIF metadata")

    if camera_meta:
        try:
            cm_path = session_dir / "camera_meta.json"
            cm_path.write_text(json.dumps(camera_meta, indent=2), encoding="utf-8")
            source_label = "colmap" if accurate_locations else "exif"
            logging.getLogger("pvrt.test").info(
                f"UI:INFO:test: Camera metadata entries={len(camera_meta)} source={source_label}, written to {cm_path}"
            )
        except Exception as e:
            import traceback
            logging.getLogger("pvrt.test").error(f"UI:ERROR:test: Failed to persist camera metadata: {e}")
            logging.getLogger("pvrt.test").error(f"UI:ERROR:test: Traceback: {traceback.format_exc()}")

    # ===== PRE-INFERENCE IMAGE ROTATION & OPTIONAL MOSAIC =====
    # Always rotate images before inference when working with image datasets (non-orthophoto)
    # so predictions run on north-up imagery; mosaic builds from rotated images if enabled.
    logging.getLogger("pvrt.test").info(f"UI:INFO:test: Rotation check - mosaic_enabled={mosaic_enabled}, input_type={input_type}, camera_meta_count={len(camera_meta) if camera_meta else 0}")
    if input_type == "images" and camera_meta:
        try:
            logging.getLogger("pvrt.test").info("UI:INFO:test: ✓ Conditions met: Starting image rotation...")
            logging.getLogger("pvrt.test").info(f"UI:INFO:test: session_dir={session_dir}, out_root={out_root}")
            
            # Verify camera_meta.json was actually written
            cm_path = session_dir / "camera_meta.json"
            if not cm_path.exists():
                raise RuntimeError(f"camera_meta.json not found at {cm_path}")
            cm_size = cm_path.stat().st_size
            
            # Verify it's valid JSON and has entries
            try:
                cm_json = json.loads(cm_path.read_text(encoding='utf-8'))
                cm_count = len([k for k in cm_json.keys() if not k.startswith('__')])
                logging.getLogger("pvrt.test").info(f"UI:INFO:test: ✓ camera_meta.json exists (size={cm_size} bytes, entries={cm_count})")
            except Exception as e:
                raise RuntimeError(f"camera_meta.json invalid: {e}")
            
            # Debug: list what's in session_dir before script
            session_contents_before = sorted([x.name for x in session_dir.glob("*")])
            logging.getLogger("pvrt.test").info(f"UI:INFO:test: session_dir before rotation: {session_contents_before}")
            
            # Call regenerate script to create rotated_images from camera_meta
            # Pass source images directory (ds_dir) so script can access original images
            import subprocess, sys, time
            script = PROJECT_ROOT / "scripts" / "regenerate_geojson_from_preds.py"
            if script.exists():
                logging.getLogger("pvrt.test").info(f"UI:INFO:test: Running regenerate script for rotation (session={session}, src_images={ds_dir})")
                
                # subprocess.run() BLOCKS until rotation completes - no env needed (inherited automatically)
                proc = subprocess.run(
                    [sys.executable, str(script), session, str(ds_dir)],
                    stdout=subprocess.DEVNULL,  # Avoid pipe buffer deadlock with large output
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=str(PROJECT_ROOT),
                    timeout=300
                )
                
                logging.getLogger("pvrt.test").info(f"UI:INFO:test: Rotation completed (exit={proc.returncode})")
                if proc.stderr:
                    for line in proc.stderr.splitlines()[-10:]:  # Last 10 error lines only
                        logging.getLogger("pvrt.test").warning(f"UI:INFO:test: [script] {line}")
                
                # Brief delay for filesystem sync after script completes
                time.sleep(0.3)
            else:
                logging.getLogger("pvrt.test").warning(f"UI:INFO:test: Script not found at {script}")
            
            # Check if rotated_images exist and create mosaic
            rotated_images_dir = session_dir / "rotated_images"
            # Force clear Python's directory listing cache
            import importlib
            from pathlib import Path as PathlibPath
            rotated_images_dir = PathlibPath(str(rotated_images_dir))  # Fresh Path object
            
            rotated_files = list(rotated_images_dir.glob("*")) if rotated_images_dir.exists() else []
            logging.getLogger("pvrt.test").info(f"UI:INFO:test: rotated_images_dir={rotated_images_dir}, exists={rotated_images_dir.exists()}, file_count={len(rotated_files)}")
            
            # Debug: list what's in session_dir after script
            session_contents_after = sorted([x.name for x in session_dir.glob("*")])
            logging.getLogger("pvrt.test").info(f"UI:INFO:test: session_dir after rotation: {session_contents_after}")
            
            if rotated_images_dir.exists() and rotated_files:
                if mosaic_enabled:
                    # MOSAIC PATH: Create mosaic from rotated images, then tile it
                    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
                    from mosaic_from_colmap import create_mosaic_from_rotated_images
                    mosaic_path = out_root / "mosaic.tif"
                    logging.getLogger("pvrt.test").info(f"UI:INFO:test: Creating mosaic from {len(rotated_files)} rotated images...")
                    create_mosaic_from_rotated_images(
                        rotated_images_dir=rotated_images_dir,
                        out_mosaic_path=mosaic_path,
                        plane_z=0.0,
                        resolution=0.1,
                        camera_meta=camera_meta,
                    )
                    logging.getLogger("pvrt.test").info(f"UI:INFO:test: ✓ Mosaic created: {mosaic_path}")
                    tiles_dir = out_root / "tiles"
                    logging.getLogger("pvrt.test").info(f"UI:INFO:test: Tiling mosaic from {mosaic_path} to {tiles_dir}")
                    _tile_tif_to_dir(mosaic_path, tiles_dir, tile_size=1024, stride=1024)
                    run_images_dir = tiles_dir
                    input_type = "tif"
                    tif_src = mosaic_path
                    logging.getLogger("pvrt.test").info(f"UI:INFO:test: ✓ Mosaic tiled; running orthophoto pipeline on {len(list(tiles_dir.glob('*')))} tiles")
                else:
                    # PER-IMAGE PATH: Use rotated images for inference
                    run_images_dir = rotated_images_dir
                    logging.getLogger("pvrt.test").info(f"UI:INFO:test: ✓ Using rotated images for per-image inference: {run_images_dir}")
            else:
                logging.getLogger("pvrt.test").warning(f"✗ Rotated images not found or empty; proceeding with original images (alignment may be incorrect)")
        except Exception as e:
            import traceback
            logging.getLogger("pvrt.test").warning(f"✗ Failed to generate rotation/mosaic: {e}")
            logging.getLogger("pvrt.test").warning(f"Traceback: {traceback.format_exc()}")
            # Continue with original input_type (per-image pipeline on original images)

    def _do_predict():
        with redirect_std_to_logger():
            return predict_entry(
                weights_dir=model_dir,
                images_dir=run_images_dir,                 # tiles dir for tif or rotated images dir
                out_dir=out_root,
                use_thermal_request=use_thermal_effective,   # <-- use effective flag
                forced_backend=(backend or forced_backend),
                score_thresh_frontend=test_threshold,
                data_has_thermal_override=data_has_thermal_override,  # <-- tell bridge TIF has band-4
                selected_bands=[b.strip() for b in (selected_bands.split(',') if selected_bands else [])] or None,
                channel_count=int(channel_count or 3),
            )



    try:
        presp = await asyncio.to_thread(_do_predict)  # <-- offload
    except Exception as e:
        logger.exception("Inference failed.")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    preds_dir = Path(presp["results_dir"])
    manifest_path = out_root / "manifest.json"
    class_names = (_read_model_meta(model_dir).get("class_names") or [])
    # Overlays are generated during inference, no post-processing needed
    logger.info(f"UI:INFO:post: Overlays were generated during inference")
    # gj, _ = _preds_to_geojson(ds_dir, preds_dir, out_root, class_names)
    try:
        th_num = float(test_threshold) if str(test_threshold).strip() else 0.0
    except Exception:
        th_num = 0.0

    # Build EXIF(GPS) + image-size indices once, then merge into manifest.json
    if camera_meta:
        gps_index = {
            fname: (float(entry.get("lat")), float(entry.get("lon")))
            for fname, entry in camera_meta.items()
            if entry.get("lat") is not None and entry.get("lon") is not None
        }
    else:
        gps_index = _scan_exif_latlon(ds_dir)       # {'file.jpg': (lat, lon)}
    sizes_index = _scan_image_sizes(ds_dir)       # {'file.jpg': (w, h)}

    # Load the manifest produced during inference and enrich it
    try:
        manifest_obj = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except Exception:
        manifest_obj = {}

    # If manifest is empty, populate it from overlays directory (generated during inference)
    if not manifest_obj or manifest_obj == {}:
        overlays_dir = out_root / "overlays"
        thumbs_dir = out_root / "thumbs"
        if overlays_dir.exists():
            for overlay_png in sorted(overlays_dir.glob("*.png")):
                # Use the overlay stem with common image extensions to find the original image
                # The overlay filename is based on the rotated image stem
                # Map to original image by checking camera_meta or common extensions
                stem = overlay_png.stem

                # Try to find original image with this stem in camera_meta
                orig_name = None
                if camera_meta:
                    # camera_meta keys are original filenames like "DJI_...jpg"
                    for fname in camera_meta.keys():
                        if Path(fname).stem == stem:
                            orig_name = fname
                            break

                # Fallback: use stem with common image extension
                if not orig_name:
                    for ext in [".jpeg", ".jpg", ".png", ".tif", ".tiff"]:
                        orig_name = f"{stem}{ext}"
                        # We don't actually verify the file exists here, just use the name
                        break

                if orig_name:
                    overlay_url = f"/media/{overlay_png.relative_to(MEDIA_DIR)}"
                    thumb_path = thumbs_dir / f"{stem}.png"
                    thumb_url = (
                        f"/media/{thumb_path.relative_to(MEDIA_DIR)}"
                        if thumb_path.exists()
                        else overlay_url
                    )
                    manifest_obj[orig_name] = {
                        "overlay": overlay_url,
                        "thumb": thumb_url,
                    }

    # manifest_obj is {"orig_filename": {"overlay": "...", "thumb": "..."}, ...}
    for fname, entry in list(manifest_obj.items()):
        if isinstance(entry, dict):
            # add lat/lon if available
            lat_lon = None
            if fname in gps_index:
                lat_lon = gps_index[fname]
            else:
                try:
                    stem = Path(fname).stem
                    for k, v in gps_index.items():
                        try:
                            if Path(k).stem == stem:
                                lat_lon = v
                                break
                        except Exception:
                            continue
                except Exception:
                    lat_lon = None
            if lat_lon:
                lat, lon = lat_lon
                entry["lat"] = float(lat)
                entry["lon"] = float(lon)
            # add w/h if available
            if fname in sizes_index:
                w, h = sizes_index[fname]
                entry["w"] = int(w)
                entry["h"] = int(h)
            
            # add detection count (n) from predictions
            try:
                stem = Path(fname).stem
                pred_file = preds_dir / "preds" / f"{stem}.json"
                if pred_file.exists():
                    pred_data = json.loads(pred_file.read_text(encoding="utf-8"))
                    boxes = pred_data.get("boxes", [])
                    entry["n"] = len(boxes)
                else:
                    entry["n"] = 0
            except Exception:
                entry["n"] = 0

    # Write back as the single source of truth
    Path(manifest_path).write_text(json.dumps(manifest_obj, indent=2), encoding="utf-8")

    # Re-read the enriched manifest and derive exif_index from it
    try:
        manifest_obj = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except Exception:
        manifest_obj = {}

    exif_from_manifest = {}
    for fname, entry in manifest_obj.items():
        if isinstance(entry, dict) and "lat" in entry and "lon" in entry:
            exif_from_manifest[fname] = (float(entry["lat"]), float(entry["lon"]))

    # camera_meta was parsed (if provided) earlier and rotated images were
    # materialized into `run_images_dir` prior to inference.

    try:
        logging.getLogger("pvrt.test").info(f"UI:INFO:test: Post-predict camera_meta keys={list(camera_meta.keys())[:5]} (count={len(camera_meta)}) run_images_dir={run_images_dir}")
    except Exception:
        pass

    # Build GeoJSONs (TIF branch stitches tiles; images branch uses EXIF/GSD)
    session_dir = MEDIA_DIR / "sessions" / session
    if input_type == "tif":
        anom_gj, imgs_gj = _build_anomalies_geojson_from_tiles(
            tiles_dir=tiles_dir,
            preds_dir=preds_dir,
            tif_path=tif_src,
            out_session=session_dir,
            class_names=class_names,
            score_thresh=th_num,
        )

        # Render downsampled overlay PNG for results grid
        ov_dir = out_root / "overlays"; ov_dir.mkdir(parents=True, exist_ok=True)
        overlay_png = ov_dir / f"{tif_src.stem}_overlay.png"
        _render_tif_overlay_preview(
            tif_path=tif_src,
            anomalies_geojson_path=anom_gj,   # <-- use the path we just got
            out_png_path=overlay_png,
            max_px=2000,
            line_thickness=2,
        )

        # Small thumb from overlay
        th_dir = out_root / "thumbs"; th_dir.mkdir(parents=True, exist_ok=True)
        thumb_png = th_dir / f"{tif_src.stem}_overlay_thumb.png"
        try:
            import cv2
            im = cv2.imread(str(overlay_png), cv2.IMREAD_COLOR)
            if im is not None:
                h, w = im.shape[:2]
                scale = 512 / float(max(h, w))
                if scale < 1.0:
                    im = cv2.resize(im, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(thumb_png), im)
        except Exception as e:
            logger.debug("ignored web.app error: %s", e)
        # Update manifest entry for this TIF so results grid shows overlay/thumb
        try:
            m = {}
            mp = Path(manifest_path)
            if mp.exists():
                m = json.loads(mp.read_text(encoding="utf-8"))
            m[tif_src.name] = {
                "overlay": f"/media/{overlay_png.relative_to(MEDIA_DIR)}",
                "thumb":   f"/media/{thumb_png.relative_to(MEDIA_DIR)}",
            }
            mp.write_text(json.dumps(m, indent=2), encoding="utf-8")
        except Exception as e:
            logger.debug("ignored web.app error: %s", e)
    else:
        anom_gj, imgs_gj = _preds_to_geojson(
            images_dir=Path(run_images_dir),
            preds_dir=Path(preds_dir),
            out_session=session_dir,
            class_names=class_names,
            score_thresh=float(th_num or 0.0),
            meters_per_pixel=0.05,
            exif_index=exif_from_manifest,
            camera_meta=camera_meta,
        )
        # For images branch, overlays/thumbs are in session dir
        ov_dir = out_root / "overlays"
        th_dir = out_root / "thumbs"

    # Collect assets for UI
    if isinstance(manifest_path, (str, Path)):
        mp = Path(manifest_path)
        if mp.suffix.lower() == ".json" and mp.exists():
            try:
                manifest_obj = json.loads(mp.read_text(encoding="utf-8"))
                # Convert manifest object to array with file field
                if isinstance(manifest_obj, dict):
                    manifest_items = [{"file": fname, **entry} for fname, entry in manifest_obj.items()]
                else:
                    manifest_items = manifest_obj
            except Exception:
                manifest_items = []
        else:
            manifest_items = []
    elif isinstance(manifest_path, list):
        manifest_items = manifest_path
    else:
        manifest_items = []

    assets = _session_assets(ses)

    # Persist a couple of run metrics
    try:
        mpath = out_root / "metrics.json"
        metrics = {}
        if mpath.exists():
            metrics = json.loads(mpath.read_text(encoding="utf-8"))
        metrics.setdefault("source_tifs", [])  # <— use this exact key
        if str(tif_src) not in metrics["source_tifs"]:
            metrics["source_tifs"].append(str(tif_src))
        mpath.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    except Exception as e:
        logging.getLogger("pvrt").warning(f"metrics.json update failed: {e}")

    logger.info(f"UI:OK:test: complete. results={preds_dir}")
    return {
        "ok": True,
        "session": session,
        "geojson": str(anom_gj),  # backward-compat
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
        "used_channel_count": int(presp.get("used_channel_count") or 0),
        "final_mode": presp.get("final_mode"),
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
            manifest_obj = json.loads(manifest_path.read_text())
            # Convert manifest object to array with file field
            manifest = [{"file": fname, **entry} for fname, entry in manifest_obj.items()]
        except Exception:
            manifest = []
    # collect assets and optional camera_meta.json
    assets = _session_assets(ses)
    camera_meta = None
    try:
        camp = ses / "camera_meta.json"
        if camp.exists():
            try:
                camera_meta = json.loads(camp.read_text(encoding="utf-8"))
            except Exception:
                camera_meta = None
    except Exception:
        camera_meta = None

    rotated_images_available = bool(assets.get("rotated_images")) or (camera_meta is not None and bool(camera_meta))

    return {
        "ok": True,
        "session": session,
        # keep old key for backward compatibility (anomalies)
        "geojson_url": f"/media/{gj.relative_to(MEDIA_DIR)}" if gj.exists() else None,
        # NEW: where image footprints live (if you created them)
        "images_geojson_url": f"/media/{imgs_gj.relative_to(MEDIA_DIR)}" if imgs_gj.exists() else None,
        "assets": assets,
        "manifest": manifest,   # still the parsed JSON (not a path)
        "tiler": "ok" if RIO_OK else "unavailable",
        # Helpful flags for the frontend
        "rotated_images_available": bool(rotated_images_available),
        "camera_meta": camera_meta,
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
_TILER_STATS: Dict[tuple, Dict[str, Any]] = {}   # per (session, idx) cached stretch + meta

def _session_tifs(session: str) -> List[Path]:
    """
    Prefer absolute source paths recorded in metrics.json under 'source_tifs'.
    Fallback: look in media/sessions/<session>/images for *.tif.
    """
    ses_dir = MEDIA_DIR / "sessions" / session
    if not ses_dir.exists():
        return []
    tifs: List[Path] = []
    m = _load_metrics(ses_dir)
    srcs = m.get("source_tifs") if isinstance(m, dict) else None
    if isinstance(srcs, list):
        for s in srcs:
            p = Path(s)
            if p.exists() and p.suffix.lower() in (".tif", ".tiff"):
                tifs.append(p)
    if not tifs:
        tifs = [p for p in (ses_dir / "images").glob("*") if p.suffix.lower() in (".tif", ".tiff")]
    return tifs

# --- small Inferno-like LUT (256x3) for single-band rasters without palette ---
def _inferno_lut_256() -> np.ndarray:
    # coarse stops approximating matplotlib "inferno" ramp:
    stops = [
        (0.00, (0,   0,   3)),
        (0.10, (31,  12,  72)),
        (0.20, (85,  15, 109)),
        (0.40, (167, 27,  79)),
        (0.60, (230, 69,  32)),
        (0.80, (252, 141, 31)),
        (1.00, (252, 255, 164)),
    ]
    xs = [s for s, _ in stops]; cols = [np.array(c, dtype=np.float32) for _, c in stops]
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        # find surrounding stops
        j = max(0, min(len(xs) - 2, next((k for k in range(len(xs)-1) if xs[k] <= t <= xs[k+1]), len(xs)-2)))
        t0, t1 = xs[j], xs[j+1]
        c0, c1 = cols[j], cols[j+1]
        if t1 == t0:
            c = c0
        else:
            a = (t - t0) / (t1 - t0)
            c = (1 - a) * c0 + a * c1
        lut[i] = np.clip(np.round(c), 0, 255).astype(np.uint8)
    return lut


@app.get("/api/session_tiles")
async def api_session_tiles(session: str):
    """
    Return tile layer descriptors for ORIGINAL GeoTIFF(s) of this session.
    The frontend will add them under the Images panel (not the Layers list).
    """
    if not RIO_OK:
        return {"ok": False, "reason": "rasterio_not_available", "layers": []}

    tifs = _session_tifs(session)
    _TILER_INDEX[session] = tifs
    layers = []

    for i, p in enumerate(tifs):
        try:
            with rasterio.open(p) as ds:
                # bounds in WGS84 for fitBounds + attribution data
                try:
                    left, bottom, right, top = rasterio.warp.transform_bounds(
                        ds.crs, CRS.from_epsg(4326), *ds.bounds, densify_pts=21
                    )
                except Exception:
                    # worst-case fallback
                    left, bottom, right, top = (-180.0, -85.0, 180.0, 85.0)
                layers.append({
                    "name": p.name,
                    "template": f"/tiles/{session}/{i}" + "/{z}/{x}/{y}.png",
                    "minzoom": 0,
                    "maxzoom": 22,
                    # [[south, west], [north, east]]
                    "bounds": [[bottom, left], [top, right]],
                })
        except Exception as e:
            logger.warning(f"Tiler: failed to inspect '{p}': {e}")

    return {"ok": True, "session": session, "layers": layers}

# ---------------- XYZ tile endpoint ----------------
# Tile size (Leaflet default)
TILE_SIZE = 256

@app.get("/tiles/{session}/{idx}/{z}/{x}/{y}.png")
def tile_xyz(session: str, idx: int, z: int, x: int, y: int):
    if not RIO_OK:
        raise HTTPException(404, "tiler unavailable")

    tifs = _TILER_INDEX.get(session) or _session_tifs(session)
    _TILER_INDEX[session] = tifs
    if not tifs or idx < 0 or idx >= len(tifs):
        raise HTTPException(404, "tile source not found")
    tif = tifs[idx]

    # XYZ tile bounds → EPSG:3857 meters
    t = mercantile.Tile(x=int(x), y=int(y), z=int(z))
    b = mercantile.xy_bounds(t)
    minx, miny, maxx, maxy = b.left, b.bottom, b.right, b.top

    # tiny Inferno-like LUT (256×3) for single-band
    def lut_inferno():
        stops = [(0.00,(0,0,3)),(0.15,(40,12,80)),(0.35,(152,24,79)),
                 (0.55,(222,65,38)),(0.75,(252,141,31)),(1.00,(252,255,164))]
        xs=[s for s,_ in stops]; cs=[np.array(c,np.float32) for _,c in stops]
        lut=np.zeros((256,3),np.uint8)
        for i in range(256):
            t=i/255.0; j=max(0,min(len(xs)-2,next((k for k in range(len(xs)-1) if xs[k]<=t<=xs[k+1]),len(xs)-2)))
            a=0 if xs[j+1]==xs[j] else (t-xs[j])/(xs[j+1]-xs[j])
            lut[i]=np.clip((1-a)*cs[j]+a*cs[j+1],0,255).astype(np.uint8)
        return lut

    from rasterio.vrt import WarpedVRT
    from rasterio.enums import Resampling, ColorInterp
    from rasterio.windows import from_bounds as win_from_bounds

    with rasterio.open(tif) as src, WarpedVRT(src, crs="EPSG:3857", resampling=Resampling.bilinear) as vrt:
        # overlap clip (prevents boundless)
        vb = vrt.bounds
        oxmin, oymin = max(minx, vb.left),  max(miny, vb.bottom)
        oxmax, oymax = min(maxx, vb.right), min(maxy, vb.top)
        if not (oxmin < oxmax and oymin < oymax):
            rgba = np.zeros((TILE_SIZE, TILE_SIZE, 4), np.uint8)
            im = Image.fromarray(rgba, "RGBA"); buf = BytesIO(); im.save(buf, "PNG"); buf.seek(0)
            return Response(content=buf.getvalue(), media_type="image/png")

        # where does overlap land inside 256×256?
        sx = TILE_SIZE / float(maxx - minx); sy = TILE_SIZE / float(maxy - miny)
        L = max(0, int(np.floor((oxmin - minx) * sx)))
        R = min(TILE_SIZE, int(np.ceil((oxmax - minx) * sx)))
        Tt = max(0, int(np.floor((maxy - oymax) * sy)))  # y inverted
        B = min(TILE_SIZE, int(np.ceil((maxy - oymin) * sy)))
        W, H = max(1, R - L), max(1, B - Tt)
        win = win_from_bounds(oxmin, oymin, oxmax, oymax, transform=vrt.transform)

        # choose visible bands (prefer declared RGB)
        try:
            cis = list(vrt.colorinterp)
            rgb_ids = []
            for want in (ColorInterp.red, ColorInterp.green, ColorInterp.blue):
                j = next((i+1 for i,ci in enumerate(cis) if ci==want), None)
                if j: rgb_ids.append(j)
            vis = rgb_ids if len(rgb_ids)==3 else [next((i+1 for i,ci in enumerate(cis) if ci!=ColorInterp.alpha), 1)]
        except Exception:
            vis = [1]

        # alpha from alpha band or mask
        try:
            a_id = next((i+1 for i,ci in enumerate(vrt.colorinterp) if ci==ColorInterp.alpha), None)
        except Exception:
            a_id = None
        alpha = (vrt.read(a_id, window=win, out_shape=(H, W), resampling=Resampling.nearest,
                          masked=False, out_dtype="uint8") if a_id
                 else vrt.read_masks(vis[0], window=win, out_shape=(H, W)))

        # path A: true RGB uint8 (keep exact colors)
        rgb_uint8 = False
        if len(vis)==3:
            try:
                dts = [src.dtypes[i-1] for i in vis if 1<=i<=src.count]
                rgb_uint8 = len(dts)==3 and all(dt.lower()=="uint8" for dt in dts)
            except Exception as e:
                logger.debug("ignored web.app error: %s", e)
        if len(vis)==3 and rgb_uint8:
            raw = vrt.read(vis, window=win, out_shape=(3,H,W), resampling=Resampling.bilinear,
                           masked=False, out_dtype="uint8")
            canvas = np.zeros((3, TILE_SIZE, TILE_SIZE), np.uint8)
            acan   = np.zeros((TILE_SIZE, TILE_SIZE), np.uint8)
            canvas[:, Tt:B, L:R] = raw; acan[Tt:B, L:R] = alpha
            rgba = np.dstack([canvas[0], canvas[1], canvas[2], acan])
        else:
            # path B: single-band (thermal) → stretch + LUT ; (falls back for non-uint8 RGB)
            band = vrt.read(vis[0], window=win, out_shape=(H,W), resampling=Resampling.bilinear,
                            masked=False, out_dtype="float32")
            # global 2–98% from quick overview (consistent across tiles)
            oh, ow = min(1024, vrt.height), min(1024, vrt.width)
            ov = vrt.read(vis[0], out_shape=(oh, ow), resampling=Resampling.bilinear, masked=True)
            vals = ov.compressed() if hasattr(ov, "compressed") else ov.ravel()
            p2  = float(np.percentile(vals, 2)) if vals.size else 0.0
            p98 = float(np.percentile(vals,98)) if vals.size else 1.0
            g = (np.clip(band, p2, p98) - p2) / max(1e-12, (p98 - p2))
            g8 = (np.nan_to_num(g) * 255 + 0.5).astype(np.uint8)
            rgb = lut_inferno()[g8] if len(vis)==1 else np.dstack([g8,g8,g8])
            canvas = np.zeros((TILE_SIZE, TILE_SIZE, 3), np.uint8)
            acan   = np.zeros((TILE_SIZE, TILE_SIZE), np.uint8)
            canvas[Tt:B, L:R, :] = rgb; acan[Tt:B, L:R] = alpha
            rgba = np.dstack([canvas, acan])

    im = Image.fromarray(rgba, "RGBA")
    buf = BytesIO(); im.save(buf, "PNG"); buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")


# -------------- Serve media & frontend --------------
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR), html=False), name="media")
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS)), name="outputs")
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="web")
