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

IMAGE_EXTS = {".jpg",".jpeg",".png",".tif",".tiff",".bmp",".webp",".JPG",".JPEG",".PNG",".TIF",".TIFF",".BMP",".WEBP"}

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


# ----------------- overlays & geojson -----------------
# Keep these simple and library-light; they operate on predictor JSON files.



def _draw_overlays(images_dir: Path, preds_dir: Path, out_root: Path, class_names: List[str]) -> Tuple[Path, Path, Path]:
    """
    Produce /overlays (prefer pre-rendered overlays if present) and /thumbs under out_root,
    plus a manifest JSON mapping original file name -> generated URLs.

    No EXIF reading/writing here. Overlays are PNG for speed.
    """
    import logging
    import numpy as np
    import cv2
    from PIL import Image, ImageDraw, ImageFont

    logger   = logging.getLogger("pvrt.test")
    overlays = out_root / "overlays"; overlays.mkdir(parents=True, exist_ok=True)
    thumbs   = out_root / "thumbs";   thumbs.mkdir(parents=True, exist_ok=True)
    manifest = out_root / "manifest.json"

    # Standardize on "overlays/" only. If an older "overlay/" folder exists, it is
    # not read to avoid duplicate dirs. To use a fallback copy-from, point
    # colored_src to "overlays".
    colored_src = overlays
    use_colored = colored_src.exists() and any(colored_src.glob("*.png"))
    if use_colored:
        logger.info(f"UI:INFO:post: using existing overlays from {colored_src}")
    else:
        logger.info("UI:INFO:post: no pre-rendered overlays found - drawing from preds JSON")

    # simple, vivid RGB palette
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

    mapper: Dict[str, Dict[str, str]] = {}

    preds_json_dir = preds_dir / "preds"

    # Read run metrics (best-effort). We use channel count to decide whether
    # to force-regenerate overlays when a raw thermal TIFF exists for an image.
    run_channel_count = None
    try:
        mpath = Path(preds_dir) / "metrics.json"
        if mpath.exists():
            mm = json.loads(mpath.read_text(encoding="utf-8"))
            run_channel_count = int(mm.get("channel_count") or mm.get("channel", mm.get("input_channels", 0)) or 0)
    except COMMON_EXCEPTIONS:
        run_channel_count = None

    def _find_thermal_candidate(p: Path) -> Path | None:
        """Local helper mirroring predictor logic: return a Path if a thermal
        candidate exists for `p` (pairs.json, thermal/* naming, sidecars),
        otherwise None.
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

    for img in sorted(images_dir.iterdir()):
        if not _is_image(img):
            continue

        stem = img.stem
        ov   = overlays / f"{stem}.png"
        th   = thumbs   / f"{stem}.png"

        im_for_thumb = None

        if use_colored and (colored_src / f"{stem}.png").exists():
            # Already rendered overlay exists — reuse it by default. However,
            # if this run used a thermal-only source and a raw thermal TIFF
            # exists for this image, force regeneration so we produce a
            # thermal-preview overlay instead of reusing any previously-colored PNG.
            force_regen = False
            try:
                # legacy behavior: treat explicit single-channel runs as thermal-only
                if run_channel_count == 1:
                    if _find_thermal_candidate(img) is not None:
                        force_regen = True
            except Exception:
                force_regen = False

            if force_regen:
                logger.info(f"UI:INFO:post: forcing regeneration for {img.name} (thermal-only source and TIFF available)")
                im_for_thumb = None
            else:
                try:
                    # ensure in-place file exists (we point colored_src==overlays, so this is a no-op)
                    im_for_thumb = Image.open(colored_src / f"{stem}.png").convert("RGB")
                except Exception:
                    im_for_thumb = None

        if im_for_thumb is None:
            # Draw overlay from JSON preds (fast; no EXIF)
            pred_json = preds_json_dir / f"{stem}.json"
            jj = _coerce_pred_json(pred_json) if pred_json.exists() else {
                "boxes": [], "scores": [], "classes": [], "file": img.name
            }

            # Choose base image for overlay depending on run channel count:
            # - if run_channel_count == 3: always use the RGB original (ignore thermal TIFFs)
            # - if run_channel_count == 1: prefer the decoded thermal TIFF (grayscale preview)
            # - if run_channel_count == 4: blend thermal (colormapped) onto RGB as the base
            base = None
            try:
                tdir = img.parent / "thermal"
                tpath = None
                # Only consider thermal sidecars when the run is NOT an RGB-only run.
                if run_channel_count is None or run_channel_count != 3:
                    # 1) pairs.json mapping
                    pjson = tdir / "pairs.json"
                    if pjson.exists():
                        try:
                            pairs = json.loads(pjson.read_text(encoding="utf-8"))
                            rel = pairs.get(img.name)
                            if rel:
                                cand = (img.parent / rel)
                                if cand.exists():
                                    tpath = cand
                        except Exception:
                            tpath = None
                    # 2) decoder naming: {stem}_thermal.*
                    if tpath is None:
                        for ext in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
                            cand = tdir / f"{img.stem}_thermal{ext}"
                            if cand.exists():
                                tpath = cand
                                break

                # If we have a thermal path and the run expects a pure thermal
                # preview, create a grayscale preview and use that as the base. If the run
                # expects 4 channels, blend the thermal colormap onto the RGB base.
                if tpath is not None and tpath.exists():
                    try:
                        tdir = tpath.parent
                        preview = tdir / f"{img.stem}_thermal_preview.png"
                        if run_channel_count == 1:
                            # Single-channel run: prefer cached grayscale preview
                            if preview.exists() and preview.stat().st_mtime >= tpath.stat().st_mtime:
                                try:
                                    base = Image.open(preview).convert("RGB")
                                    logger.info(f"UI:INFO:post: using cached thermal preview {preview} for {img.name}")
                                except COMMON_EXCEPTIONS:
                                    base = None
                            else:
                                try:
                                    import tifffile as _tifffile
                                    arr = _tifffile.imread(str(tpath))
                                except COMMON_EXCEPTIONS:
                                    with Image.open(tpath) as _im:
                                        arr = np.array(_im)

                                if issubclass(getattr(arr, 'dtype').type, np.floating) or getattr(arr, 'dtype').itemsize > 1:
                                    mn = float(np.nanmin(arr)) if arr.size else 0.0
                                    mx = float(np.nanmax(arr)) if arr.size else 1.0
                                    if mx > mn:
                                        norm = (np.clip(arr, mn, mx) - mn) / (mx - mn)
                                    else:
                                        norm = np.zeros_like(arr, dtype=np.float32)
                                    arr8 = (np.clip(norm * 255.0, 0, 255)).astype(np.uint8)
                                else:
                                    arr8 = np.clip(arr, 0, 255).astype(np.uint8)

                                if arr8.ndim == 3:
                                    try:
                                        gray = cv2.cvtColor(arr8, cv2.COLOR_BGR2GRAY)
                                    except COMMON_EXCEPTIONS:
                                        gray = arr8[..., 0]
                                else:
                                    gray = arr8

                                try:
                                    gray_u8 = gray.astype(np.uint8)
                                    cv2.imwrite(str(preview), gray_u8)
                                    base = Image.fromarray(cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2RGB))
                                    logger.info(f"UI:INFO:post: generated thermal preview {preview} for {img.name}")
                                except COMMON_EXCEPTIONS:
                                    try:
                                        base = Image.fromarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
                                    except COMMON_EXCEPTIONS:
                                        base = None

                        elif run_channel_count == 4:
                            # 4-channel run: blend thermal (colormapped) onto RGB
                            try:
                                # load RGB original
                                rgb_base = Image.open(img).convert("RGB")
                                rgb_arr = np.array(rgb_base)
                                # load thermal
                                try:
                                    import tifffile as _tifffile
                                    tarr = _tifffile.imread(str(tpath))
                                except COMMON_EXCEPTIONS:
                                    with Image.open(tpath) as _tim:
                                        tarr = np.array(_tim)

                                if issubclass(getattr(tarr, 'dtype').type, np.floating) or getattr(tarr, 'dtype').itemsize > 1:
                                    mn = float(np.nanmin(tarr)) if tarr.size else 0.0
                                    mx = float(np.nanmax(tarr)) if tarr.size else 1.0
                                    if mx > mn:
                                        tnorm = (np.clip(tarr, mn, mx) - mn) / (mx - mn)
                                    else:
                                        tnorm = np.zeros_like(tarr, dtype=np.float32)
                                    t8 = (np.clip(tnorm * 255.0, 0, 255)).astype(np.uint8)
                                else:
                                    t8 = np.clip(tarr, 0, 255).astype(np.uint8)

                                if t8.ndim == 3:
                                    try:
                                        tgray = cv2.cvtColor(t8, cv2.COLOR_BGR2GRAY)
                                    except COMMON_EXCEPTIONS:
                                        tgray = t8[..., 0]
                                else:
                                    tgray = t8

                                # simple inferno-like colormap (reuse helper if available)
                                lut = _inferno_lut_256()
                                cmap = lut[tgray]
                                # ensure same shape as rgb_arr
                                if cmap.shape[:2] != rgb_arr.shape[:2]:
                                    cmap = cv2.resize(cmap, (rgb_arr.shape[1], rgb_arr.shape[0]), interpolation=cv2.INTER_LINEAR)
                                # blend: 60% rgb + 40% thermal colormap
                                blended = cv2.addWeighted(rgb_arr.astype(np.uint8), 0.6, cmap.astype(np.uint8), 0.4, 0)
                                base = Image.fromarray(blended)
                            except COMMON_EXCEPTIONS:
                                base = None

                    except COMMON_EXCEPTIONS:
                        base = None

            except COMMON_EXCEPTIONS:
                base = None

            if base is None:
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

            # thickness + font based on image size
            thickness = max(1, int(round(min(W, H) * 0.003)))
            try:
                font = ImageFont.load_default()  # no TTF to keep it fast
            except COMMON_EXCEPTIONS:
                font = ImageFont.load_default()

            for i, b in enumerate(boxes):
                try:
                    x1, y1, x2, y2 = map(int, b)
                except COMMON_EXCEPTIONS:
                    continue
                if x2 <= x1 or y2 <= y1:
                    continue

                cls_id = classes[i] if i < len(classes) else 0
                name   = class_names[cls_id] if 0 <= cls_id < len(class_names) else f"cls_{cls_id}"
                sc     = float(scores[i]) if i < len(scores) else 0.0
                label  = f"{name} {int(round(sc * 100))}%"

                color  = pal[cls_id % len(pal)]
                draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)

                # simple label background
                try:
                    bbox = draw.textbbox((0, 0), label, font=font)
                    tw, th_txt = bbox[2] - bbox[0], bbox[3] - bbox[1]
                except COMMON_EXCEPTIONS:
                    tw, th_txt = draw.textsize(label, font=font)
                pad = 4
                pill_w = tw + 2 * pad
                pill_h = th_txt + 2 * pad

                top = y1 - pill_h if (y1 - pill_h) >= 0 else y1
                left = x1
                draw.rectangle([left, top, left + pill_w, top + pill_h], fill=color)
                tx, ty = left + pad, top + pad
                # thin black shadow
                for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
                    draw.text((tx + dx, ty + dy), label, fill=(0, 0, 0), font=font)
                draw.text((tx, ty), label, fill=(255, 255, 255), font=font)

            # Save overlay (fast): prefer OpenCV PNG write for speed, fall back to PIL
            try:
                try:
                    arr_out = np.array(base)  # RGB
                    bgr = cv2.cvtColor(arr_out, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(ov), bgr)
                    logger.info(f"UI:INFO:post: wrote overlay {ov} for {img.name}")
                except COMMON_EXCEPTIONS:
                    # fallback to PIL save
                    base.save(ov, format="PNG", optimize=True)
                    logger.info(f"UI:INFO:post: wrote overlay (PIL fallback) {ov} for {img.name}")
            except COMMON_EXCEPTIONS:
                # final fallback: ensure file exists
                try:
                    Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8)).save(ov, format="PNG", optimize=True)
                except COMMON_EXCEPTIONS as e:
                    logger.debug("ignored web.app error: %s", e)
            im_for_thumb = base

        # Thumb from overlay (fast write via OpenCV)
        try:
            w, h = im_for_thumb.size
            tw = max(96, w // 6); thh = max(96, h // 6)
            im_thumb = im_for_thumb.resize((tw, thh))
            try:
                arr_thumb = np.array(im_thumb)  # RGB
                bgr_thumb = cv2.cvtColor(arr_thumb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(th), bgr_thumb)
                logger.info(f"UI:INFO:post: wrote thumb {th} for {img.name}")
            except COMMON_EXCEPTIONS:
                im_thumb.save(th, format="PNG", optimize=True)
                logger.info(f"UI:INFO:post: wrote thumb (PIL fallback) {th} for {img.name}")
        except COMMON_EXCEPTIONS:
            try:
                Image.fromarray(np.zeros((96, 96, 3), dtype=np.uint8)).save(th, format="PNG", optimize=True)
            except COMMON_EXCEPTIONS as e:
                logger.debug("ignored web.app error: %s", e)
        mapper[img.name] = {
            "overlay": f"/media/{ov.relative_to(MEDIA_DIR).as_posix()}" if str(ov).startswith(str(MEDIA_DIR)) else ov.name,
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


def _preds_to_geojson(
    images_dir: Path,
    preds_dir: Path,
    out_session: Path,
    class_names: List[str],
    score_thresh: float = 0.0,
    meters_per_pixel: float = 0.05,
    exif_index: Optional[Dict[str, Tuple[float, float]]] = None,  # {'file': (lat, lon)}
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

    # 3) (optional) overlay/thumb URLs from manifest.json (written by _draw_overlays)
    manifest_map: Dict[str, Dict[str, str]] = {}
    mpath = out_session / "manifest.json"
    if mpath.exists():
        try:
            manifest_map = json.loads(mpath.read_text(encoding="utf-8"))
        except Exception:
            manifest_map = {}

    # ---------------- images.geojson (points for sidebar/catalog) ----------------
    imgs_fc = {"type": "FeatureCollection", "features": []}
    for fname, (lat, lon) in gps_index.items():
        props = {}
        if isinstance(manifest_map.get(fname), dict):
            entry = manifest_map[fname]
            if "overlay" in entry: props["image"] = Path(entry["overlay"]).name
            if "thumb"   in entry: props["thumb"]   = entry["thumb"]
            if "w" in entry and "h" in entry:
                props["w"] = int(entry["w"]); props["h"] = int(entry["h"])
        elif fname in sizes_index:
            w, h = sizes_index[fname]
            props["w"] = int(w); props["h"] = int(h)

        imgs_fc["features"].append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": props
        })

    imgs_path = out_session / "images.geojson"
    imgs_path.write_text(json.dumps(imgs_fc, indent=2), encoding="utf-8")

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
        # Degrees per pixel in lon/lat
        px_dlon = meters_per_pixel * deg_per_m_lon
        px_dlat = meters_per_pixel * deg_per_m_lat

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

            # pixel deltas from image center → degrees
            dx0 = (x0 - cx) * px_dlon
            dx1 = (x1 - cx) * px_dlon
            dy0 = (y0 - cy) * px_dlat
            dy1 = (y1 - cy) * px_dlat

            # Note: latitude increases northwards (y up). Image y grows down → subtract for lat.
            lon0 = lon + dx0; lon1 = lon + dx1
            lat0 = lat - dy0; lat1 = lat - dy1

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
async def api_models():
    return {"ok": True, "models": _list_models()}


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

    def _do_predict():
        with redirect_std_to_logger():
            return predict_entry(
                weights_dir=model_dir,
                images_dir=run_images_dir,                 # tiles dir for tif
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
    class_names = (_read_model_meta(model_dir).get("class_names") or [])
    # ov_dir, th_dir, manifest = _draw_overlays(ds_dir, preds_dir, out_root, class_names)
    ov_dir, th_dir, manifest_path = _draw_overlays(ds_dir, preds_dir, out_root, class_names)
    # gj, _ = _preds_to_geojson(ds_dir, preds_dir, out_root, class_names)
    try:
        th_num = float(test_threshold) if str(test_threshold).strip() else 0.0
    except Exception:
        th_num = 0.0

    # Build EXIF(GPS) + image-size indices once, then merge into manifest.json
    gps_index   = _scan_exif_latlon(ds_dir)       # {'file.jpg': (lat, lon)}
    sizes_index = _scan_image_sizes(ds_dir)       # {'file.jpg': (w, h)}

    # Load the manifest produced by _draw_overlays and enrich it
    try:
        manifest_obj = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except Exception:
        manifest_obj = {}

    # manifest_obj is {"orig_filename": {"overlay": "...", "thumb": "..."}, ...}
    for fname, entry in list(manifest_obj.items()):
        if isinstance(entry, dict):
            # add lat/lon if available
            if fname in gps_index:
                lat, lon = gps_index[fname]
                entry["lat"] = float(lat)
                entry["lon"] = float(lon)
            # add w/h if available
            if fname in sizes_index:
                w, h = sizes_index[fname]
                entry["w"] = int(w)
                entry["h"] = int(h)

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
            images_dir=Path(ds_dir),
            preds_dir=Path(preds_dir),
            out_session=session_dir,
            class_names=class_names,
            score_thresh=float(th_num or 0.0),
            meters_per_pixel=0.05,
            exif_index=exif_from_manifest,
        )

    # Collect assets for UI
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
