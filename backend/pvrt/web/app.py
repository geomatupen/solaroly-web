# backend/pvrt/web/app.py
from __future__ import annotations

import asyncio
import csv
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
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

from fastapi import FastAPI, Form, UploadFile, File, HTTPException, Response, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import StreamingResponse
from fastapi.responses import JSONResponse, Response as FastAPIResponse
from ..postprocess import analyze_geojson

from PIL import Image, ExifTags
from PIL.ExifTags import TAGS, GPSTAGS
import numpy as np
import math
from io import BytesIO
from ..dataops.camera_geometry import compute_meters_per_pixel as _compute_meters_per_pixel
from ..dataops.row_alignment import RowAlignmentOptions, align_rotated_images_to_rows
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
from .postprocess import create_postprocess_router
from .row_alignment import create_row_alignment_router, resolve_rows_source
from .settings import settings

if settings.enable_detectron:
    from ..backends.detectron.backend import register as register_detectron
else:  # feature disabled or dependency missing
    register_detectron = None  # type: ignore

if settings.enable_yolo:
    from ..backends.yolo.backend import register as register_yolo
else:
    register_yolo = None  # type: ignore

# --- Project management ---
from ..core.projects import ProjectManager, Project

# --- SSE/logging bridge ---
from .sse import LogBroker, SSELogHandler, set_event_loop, sse_response


_ACTIVE_TEST_RUNS: set[str] = set()
_ACTIVE_TEST_RUNS_GUARD = asyncio.Lock()

if settings.enable_colmap:
    from .colmap import (
        router as colmap_router,
        configure_colmap_dependencies,
        _colmap_ready,
        _load_colmap_meta,
        _merge_optical_metadata,
    )
else:
    colmap_router = None
    configure_colmap_dependencies = None

    def _colmap_ready(*_args, **_kwargs) -> bool:  # type: ignore
        return False

    def _load_colmap_meta(*_args, **_kwargs):  # type: ignore
        raise RuntimeError("COLMAP support is disabled on this server.")

    def _merge_optical_metadata(*_args, **_kwargs):  # type: ignore
        raise RuntimeError("COLMAP support is disabled on this server.")
from .mosaic import prepare_rotation_and_mosaic

# --- Reuse data helpers (RJPEG decode & scanning) ---
from ..dataops.scan_decode_split import (
    scan_split_decode_thermal, # safe to call only if thermal requested
)
from ..dataops.thermal_convert import convert_thermal_folder, ensure_dirp_init, scan_conversion_folder
from ..dataops.training_datasets import (
    DatasetUploadError,
    delete_dataset as delete_training_dataset,
    ensure_legacy_dataset,
    get_dataset as get_training_dataset,
    install_dataset as install_training_dataset,
    list_datasets as list_training_datasets,
    rename_dataset as rename_training_dataset,
    resolve_dataset_for_training,
)
from ..core.io import has_thermal_for_images

# -------- Path utilities --------
def convert_windows_path_to_wsl(path_str: str) -> str:
    r"""
    Convert Windows paths to WSL paths if needed.
    E.g., 'E:\Termatics' -> '/mnt/e/Termatics'
    Also handles already-WSL paths and Unix paths.
    """
    if not path_str:
        return path_str
    
    # Check if it looks like a Windows path (e.g., E:\, C:\)
    import re
    match = re.match(r'^([A-Za-z]):[\\\/](.*)$', path_str)
    if match:
        drive_letter = match.group(1).lower()
        path_part = match.group(2).replace('\\', '/')
        return f"/mnt/{drive_letter}/{path_part}"
    
    # Already a Unix/WSL path or a relative path
    return path_str

def looks_like_windows_path(path_str: str) -> bool:
    r"""Check if a path looks like a Windows path (e.g., C:\, D:\Folder\etc)."""
    if not path_str:
        return False
    import re
    return bool(re.match(r'^[A-Za-z]:[\\\/]', path_str))

# ---------------- Paths & constants ----------------
ROOT = Path(__file__).resolve().parents[2]        # .../backend/pvrt
PROJECT_ROOT = ROOT.parent                         # repo root
DEFAULT_PROJECTS_ROOT = PROJECT_ROOT / "backend" / "projects"
DEFAULT_PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)

FRONTEND_DIR = PROJECT_ROOT / "frontend"
# MEDIA_DIR serves project files via /media/* URLs (e.g., /media/backend/projects/.../test/outputs/...)
MEDIA_DIR    = PROJECT_ROOT

IMAGE_EXTS = {".jpg",".jpeg",".png",".tif",".tiff",".bmp",".webp",".JPG",".JPEG",".PNG",".TIF",".TIFF",".BMP",".WEBP"}

# DJI RelativeAltitude is takeoff-relative, while GSD needs the distance to the
# target plane. Rooftop tests can override this default from Advanced options.
DEFAULT_TARGET_SURFACE_HEIGHT_M = 4.0

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
if settings.enable_detectron and register_detectron:
    register_detectron(register_backend)
else:
    logger.info("Detectron backend disabled via settings.")

if settings.enable_yolo and register_yolo:
    register_yolo(register_backend)
elif settings.enable_yolo:
    logger.warning("YOLO backend enabled in settings but import failed.")
else:
    logger.info("YOLO backend disabled via settings.")

# --------------- Project management ----------------
PROJECTS_REGISTRY = DEFAULT_PROJECTS_ROOT / "projects.json"
project_manager = ProjectManager(PROJECTS_REGISTRY)

# Get current active project (first one, or default)
_active_project: Optional[Project] = None
def get_active_project() -> Project:
    """Get the currently active project. If none, return the only/first one."""
    global _active_project
    if _active_project is None:
        projects = project_manager.list_projects()
        if projects:
            _active_project = projects[0]
        else:
            raise ValueError("No projects configured. Please create a project first.")
    return _active_project

def set_active_project(project_id: str) -> Project:
    """Set the active project by ID."""
    global _active_project
    project = project_manager.get_project(project_id)
    if not project:
        raise ValueError(f"Project {project_id} not found")
    _active_project = project
    return project


def _require_thermal_enabled(action: str) -> None:
    """Raise a helpful error when thermal workflows are disabled."""
    if settings.enable_thermal_data_extraction:
        return
    raise HTTPException(
        status_code=400,
        detail=(
            "Thermal data extraction is disabled on this server. "
            f"Set PVRT_ENABLE_THERMAL=1 to {action}."
        ),
    )

# ================== Project-aware path helpers ==================

def get_project_data_dir(project: Optional[Project] = None) -> Path:
    """Get train data directory for project (contains train/ and valid/ subfolders)."""
    if project is None:
        project = get_active_project()
    return project.get_train_data_dir()

def get_project_train_dir(project: Optional[Project] = None) -> Path:
    """Get train/data/train directory for project (training images)."""
    if project is None:
        project = get_active_project()
    return project.get_train_data_dir() / "train"

def get_project_valid_dir(project: Optional[Project] = None) -> Path:
    """Get train/data/valid directory for project (validation images)."""
    if project is None:
        project = get_active_project()
    return project.get_train_data_dir() / "valid"

def get_project_test_dir(project: Optional[Project] = None) -> Path:
    """Get test/data directory for project (test images)."""
    if project is None:
        project = get_active_project()
    return project.get_test_data_dir()

def get_project_output_dir(project: Optional[Project] = None) -> Path:
    """Model runs directory (train/outputs) for project."""
    if project is None:
        project = get_active_project()
    return project.get_train_outputs_dir()

def get_project_sessions_dir(project: Optional[Project] = None) -> Path:
    """Test outputs directory (test/outputs) for project."""
    if project is None:
        project = get_active_project()
    return project.get_test_outputs_dir()


def get_project_overlays_dir(project: Optional[Project] = None) -> Path:
    """Overlays directory for project."""
    if project is None:
        project = get_active_project()
    return project.get_overlays_dir()


def get_project_colmap_dir(project: Optional[Project] = None) -> Path:
    """COLMAP workspace directory for project."""
    if project is None:
        project = get_active_project()
    return project.get_colmap_dir()


def _resolve_thermal_directory(value: str, *, must_exist: bool) -> Path:
    """Resolve an explicit host directory for the standalone converter."""
    raw_value = convert_windows_path_to_wsl(str(value or "").strip())
    if not raw_value:
        raise HTTPException(status_code=400, detail="A folder path is required.")
    candidate = Path(raw_value).expanduser()
    if not candidate.is_absolute():
        raise HTTPException(status_code=400, detail="Enter an absolute input and output folder path.")
    candidate = candidate.resolve()
    if must_exist and (not candidate.exists() or not candidate.is_dir()):
        raise HTTPException(status_code=404, detail=f"Folder not found: {value}")
    return candidate


def _project_training_dataset_destination(project, display_name: str) -> Path:
    """Return the project-owned destination for a newly uploaded dataset."""
    folder_name = _safe_name(display_name)
    if not folder_name:
        raise HTTPException(status_code=400, detail="Training dataset name must contain letters or numbers.")
    destination = (project.get_train_dir() / "datasets" / folder_name).resolve()
    if destination.exists():
        raise HTTPException(
            status_code=409,
            detail="A training dataset folder with this name already exists in the project. Choose another dataset name.",
        )
    return destination


def _media_url(path: Path) -> str:
    """Convert absolute path to /media URL path.
    For project files, use /api/project_file endpoint to support external drives.
    """
    try:
        abs_path = path.resolve()
        # Try to make it relative to PROJECT_ROOT (for files in workspace)
        try:
            rel = abs_path.relative_to(PROJECT_ROOT.resolve())
            return f"/media/{rel.as_posix()}"
        except ValueError:
            # Path is outside PROJECT_ROOT (e.g., external drive project)
            # Use the project file endpoint with URL-encoded absolute path
            from urllib.parse import quote
            return f"/api/project_file/{quote(str(abs_path), safe='')}"
    except Exception:
        return f"/media/{path.name}"


# --------------- Cancel flag (best-effort) ----------------
CANCEL_FLAGS: Dict[str, bool] = {"train": False}
THERMAL_CONVERT_JOBS: Dict[str, Dict[str, Any]] = {}

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


def _write_run_status(run_dir: Path, status: str, **details: Any) -> None:
    path = run_dir / "run_status.json"
    temporary = run_dir / ".run_status.json.tmp"
    payload = {"status": status, "updated_at": datetime.now().isoformat(), **details}
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _model_run_status(run_dir: Path, meta: Optional[dict] = None) -> str:
    """Return completed only for an explicit success or a legacy final artifact."""
    status_path = run_dir / "run_status.json"
    if status_path.is_file():
        try:
            state = json.loads(status_path.read_text(encoding="utf-8"))
            status = str(state.get("status") or "").strip().lower()
            if status:
                return status
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            pass
    has_meta = bool(meta) or (run_dir / "model_meta.json").is_file()
    has_final = any((run_dir / name).is_file() for name in ("model_final.pth", "model_final.pt"))
    return "complete" if has_meta and has_final else "incomplete"

def _model_id(run_dir: Path, meta: Optional[dict] = None) -> str:
    """The existing run-folder name is the immutable model id."""
    return run_dir.name


def _find_model(model_ref: str) -> Tuple[Path, dict]:
    """Resolve either a stable model id or the legacy run-folder name."""
    safe_ref = _safe_name(model_ref)
    output_dir = get_project_output_dir()
    if safe_ref:
        legacy_dir = output_dir / safe_ref
        if legacy_dir.is_dir():
            return legacy_dir, _read_model_meta(legacy_dir)

    for run_dir in output_dir.iterdir() if output_dir.exists() else []:
        if not run_dir.is_dir():
            continue
        meta = _read_model_meta(run_dir)
        if _model_id(run_dir, meta) == model_ref:
            return run_dir, meta
    raise HTTPException(status_code=404, detail="Model not found.")


def _write_model_meta(run_dir: Path, meta: dict) -> None:
    meta_path = run_dir / "model_meta.json"
    temp_path = run_dir / ".model_meta.json.tmp"
    temp_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    temp_path.replace(meta_path)


def _latest_training_metrics(run_dir: Path) -> dict:
    """Read the latest JSON-lines metrics row and latest row containing each loss."""
    path = run_dir / "metrics.json"
    latest: dict = {}
    latest_losses: dict = {}
    if path.is_file():
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                latest = row
                for key, value in row.items():
                    if "loss" in str(key).lower() and isinstance(value, (int, float)):
                        latest_losses[key] = value
        except (OSError, UnicodeDecodeError):
            pass
    summary = {
        key: latest.get(key)
        for key in (
            "iteration", "lr", "eta_seconds",
            "bbox/AP", "bbox/AP50", "bbox/AP75",
            "segm/AP", "segm/AP50", "segm/AP75",
        )
        if latest.get(key) is not None
    }
    summary.update(latest_losses)
    csv_path = run_dir / "results.csv"
    if csv_path.is_file():
        try:
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if rows:
                for raw_key, raw_value in rows[-1].items():
                    key = str(raw_key or "").strip()
                    try:
                        summary[key] = float(raw_value)
                    except (TypeError, ValueError):
                        if raw_value not in (None, ""):
                            summary[key] = raw_value
                if "epoch" in summary and "iteration" not in summary:
                    summary["iteration"] = summary["epoch"]
        except (OSError, UnicodeDecodeError, csv.Error):
            pass
    return summary


def _list_models(include_incomplete: bool = False, project: Optional[Project] = None) -> List[dict]:
    models = []
    output_dir = get_project_output_dir(project)
    if output_dir.exists():
        for d in sorted(output_dir.iterdir()):
            if not d.is_dir() or not any(d.iterdir()):
                continue
            meta = _read_model_meta(d)
            run_status = _model_run_status(d, meta)
            complete = run_status in {"complete", "completed"}
            if not complete and not include_incomplete:
                continue
            models.append({
                "id": _model_id(d, meta),
                "name": d.name,
                "display_name": meta.get("display_name") or meta.get("model_name") or d.name,
                "mtime": int(d.stat().st_mtime),
                "model_name": meta.get("model_name") or None,
                "model_type": meta.get("model_type") or None,
                "task": meta.get("task") or (
                    "segment" if "mask" in str(meta.get("model_type", "")).lower() or bool(meta.get("yolo_seg")) else "detect"
                ),
                "input_mode": meta.get("input_mode"),
                "channel_count": meta.get("channel_count"),
                "backend": meta.get("backend"),
                "thermal_used": bool(meta.get("thermal_used", False)),
                "num_classes": meta.get("num_classes"),
                "complete": complete,
                "status": "complete" if complete else "incomplete",
                "run_status": run_status,
            })
    # Sort by mtime descending (newest first)
    models.sort(key=lambda m: m["mtime"], reverse=True)
    return models

def _unique_dataset_dir(base_name: str) -> Path:
    test_dir = get_project_test_dir()
    d = test_dir / base_name
    if not d.exists():
        return d
    i = 1
    while True:
        cand = test_dir / f"{base_name}-{i}"
        if not cand.exists():
            return cand
        i += 1


# ---- session helpers ----
from typing import Any

def _as_session_dir(ses) -> Path:
    """
    Accept a session name like 'test_20250927_111404' or an absolute Path,
    and return the test outputs/<name> directory.
    """
    p = Path(ses)
    if p.exists():
        return p
    return get_project_sessions_dir() / str(ses)

def _load_metrics(ses) -> Dict[str, Any]:
    """
    Read test_outputs/<session>/metrics.json safely.
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

    logger = logging.getLogger("pvrt.test")
    tiles_dir = Path(tiles_dir)
    tiles_dir.mkdir(parents=True, exist_ok=True)

    stride = stride or tile_size
    written = []

    with rasterio.open(tif_path) as src:
        x_offsets = [x for x in range(0, src.width, stride) if min(tile_size, src.width - x) > 1]
        y_offsets = [y for y in range(0, src.height, stride) if min(tile_size, src.height - y) > 1]
        total_tiles = max(1, len(x_offsets) * len(y_offsets))
        progress_step = max(1, total_tiles // 10)
        logger.info(
            "UI:INFO:test: Splitting orthomosaic %s (%sx%s, %s bands) into %s tiles…",
            Path(tif_path).name, src.width, src.height, src.count, total_tiles,
        )
        logger.info(f"UI:INFO:test: Orthomosaic CRS={_crs_id(src.crs)} tile_size={tile_size}px")
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

                completed = len(written)
                if completed == 1 or completed == total_tiles or completed % progress_step == 0:
                    percent = min(100, int(round(completed * 100 / total_tiles)))
                    logger.info(
                        "UI:INFO:test: Orthomosaic split progress: %s/%s tiles (%s%%)",
                        completed, total_tiles, percent,
                    )

                # Log a few tiles to verify CRS & transform persisted
                if (x0, y0) in [(0, 0), (tile_size, 0), (0, tile_size)]:
                    with rasterio.open(outp) as chk:
                        logger.debug(
                            "Tiler: wrote '%s' CRS=%s transform=%s",
                            outp.name, _crs_id(chk.crs), chk.transform,
                        )

    logger.info("UI:OK:test: Orthomosaic split complete: %s tiles ready.", len(written))
    return written


# --- ADD: stitch per-tile JSON predictions into one predictions.geojson (WGS84) ---
def _build_anomalies_geojson_from_tiles(
    tiles_dir,            # folder that contains the tile GeoTIFFs
    preds_dir,            # session results dir, contains "preds/*.json"
    tif_path,             # original source GeoTIFF
    out_session,          # /test_outputs/<session>
    class_names,          # list of class names
    score_thresh=0.0,
):
    """
    Build ONE merged predictions.geojson in EPSG:4326 from per-tile predictions.
    Prefers per-instance mask polygons in pixel space (jd['polygons']); falls back to bboxes.
    Clips results to the source TIF footprint. Also writes images.geojson center point.
    Returns (predictions_geojson_path, images_geojson_path).
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
                        "prediction_index": i,
                        "source": tif_path.name,
                    }
                })

    # --- write predictions.geojson ---
    anom_fc = {"type": "FeatureCollection", "features": feats}
    anom_path = out_session / "predictions.geojson"
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
        elif tag_name == "GimbalPitchDegree":
            out["gimbal_pitch"] = val
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


def _extract_camera_meta_entry(
    img_path: Path,
    target_surface_height_m: float = 0.0,
) -> Optional[Dict[str, Any]]:
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

    # GSD needs camera-to-surface distance. DJI RelativeAltitude is an AGL-like
    # takeoff-relative approximation; absolute/GPS altitude is a datum height
    # and must not be used directly as ground clearance.
    altitude_m = xmp_meta.get("relative_altitude")
    gsd_altitude_m = (
        max(0.0, float(altitude_m) - float(target_surface_height_m))
        if altitude_m is not None
        else None
    )

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
        gsd_altitude_m, width, focal_length_mm, focal_length_35mm,
        focal_plane_x_res, focal_plane_res_unit, height,
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
        entry["target_surface_height_m"] = float(target_surface_height_m)
        entry["gsd_altitude_m"] = float(gsd_altitude_m)
    if xmp_meta.get("absolute_altitude") is not None:
        entry["absolute_altitude"] = float(xmp_meta["absolute_altitude"])
    elif alt_from_gps is not None:
        entry["absolute_altitude"] = float(alt_from_gps)
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
        entry["meters_per_pixel_status"] = "estimated"
        entry["meters_per_pixel_model"] = "nadir_flat_plane"
        if (
            focal_length_mm
            and focal_plane_x_res
            and focal_plane_res_unit in (2, 3, 4, 5)
        ):
            entry["meters_per_pixel_method"] = "focal_plane_resolution"
        elif focal_length_35mm and height:
            entry["meters_per_pixel_method"] = "35mm_equivalent_diagonal"
    else:
        entry["meters_per_pixel_status"] = "unavailable"
        entry["meters_per_pixel_warning"] = (
            "relative_altitude_missing"
            if altitude_m is None
            else (
                "effective_camera_to_target_distance_nonpositive"
                if not gsd_altitude_m or gsd_altitude_m <= 0
                else "usable_camera_optics_missing"
            )
        )
    if capture_ts is not None:
        entry["timestamp"] = float(capture_ts)
        entry["timestamp_source"] = "exif_datetime"
    if xmp_meta.get("relative_altitude") is not None:
        entry["alt_source"] = "relative_altitude"
    if meters_per_pixel is not None:
        entry["meters_per_pixel_source"] = "relative_altitude_minus_target_surface_height_and_exif_optics"
    if xmp_meta.get("gimbal_yaw") is not None:
        entry["rotation_source"] = "gimbal_yaw"
    elif xmp_meta.get("flight_yaw") is not None:
        entry["rotation_source"] = "flight_yaw"
    elif heading is not None:
        entry["rotation_source"] = "gps_img_direction"
    if xmp_meta.get("gimbal_pitch") is not None:
        entry["gimbal_pitch"] = float(xmp_meta["gimbal_pitch"])
        off_nadir_deg = abs(90.0 - abs(float(xmp_meta["gimbal_pitch"])))
        entry["off_nadir_deg"] = float(off_nadir_deg)
        if meters_per_pixel is not None and off_nadir_deg > 5.0:
            entry["meters_per_pixel_warning"] = "off_nadir_scale_varies_across_image"
    if xmp_meta.get("gimbal_roll") is not None:
        entry["gimbal_roll"] = float(xmp_meta["gimbal_roll"])
    return entry


def _build_camera_meta_from_exif(
    images_dir: Path,
    target_surface_height_m: float = 0.0,
) -> Dict[str, Dict[str, Any]]:
    meta: Dict[str, Dict[str, Any]] = {}
    if not images_dir or not images_dir.exists():
        return meta
    order_idx: Dict[str, int] = {}
    for order, img_path in enumerate(sorted(images_dir.iterdir())):
        if not img_path.is_file():
            continue
        if img_path.suffix not in IMAGE_EXTS:
            continue
        entry = _extract_camera_meta_entry(img_path, target_surface_height_m)
        if entry:
            meta[img_path.name] = entry
            order_idx[img_path.name] = order
    if meta:
        _augment_camera_rotations_from_track(meta, order_idx)
    return meta


def _canonical_image_candidate_names(*name_groups) -> set[str]:
    """Return one preferred filename per source stem.

    Groups are ordered from lowest to highest priority. Generated inference
    files can therefore provide missing images without duplicating an original
    source name supplied by EXIF, camera metadata, or the manifest.
    """
    by_stem: Dict[str, str] = {}
    for names in name_groups:
        for value in sorted((str(name) for name in names if name), key=str.casefold):
            stem = Path(value).stem.casefold()
            if stem:
                by_stem[stem] = value
    return set(by_stem.values())



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

if settings.enable_colmap and configure_colmap_dependencies and colmap_router:
    configure_colmap_dependencies(
        get_project_test_dir=get_project_test_dir,
        get_project_colmap_dir=get_project_colmap_dir,
        now_stamp=_now_stamp,
        is_image=_is_image,
        build_camera_meta_from_exif=_build_camera_meta_from_exif,
        scan_image_sizes=_scan_image_sizes,
        lookup_camera_meta_entry=_lookup_camera_meta_entry,
    )
    app.include_router(colmap_router)
else:
    logger.info("COLMAP integration disabled via settings; skipping router setup.")


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
      - predictions.geojson: bbox polygons converted to WGS84 using center-based pixel→meter→degree math
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

    rotated_by_stem = {
        path.stem: path
        for path in (out_session / "rotated_images").glob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    }
    aligned_centres_by_stem: Dict[str, Tuple[float, float]] = {}
    alignment_path = out_session / "mosaic_alignment.json"
    if alignment_path.is_file():
        try:
            alignment_payload = json.loads(alignment_path.read_text(encoding="utf-8"))
            for image_name, record in (alignment_payload.get("images") or {}).items():
                final_lat_lon = record.get("final_lat_lon") if isinstance(record, dict) else None
                if isinstance(final_lat_lon, list) and len(final_lat_lon) >= 2:
                    aligned_centres_by_stem[Path(image_name).stem] = (
                        float(final_lat_lon[0]), float(final_lat_lon[1])
                    )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            logger.warning("Could not read mosaic image alignment centres from %s", alignment_path)

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

    # A prepared north-up PNG and its original JPEG have the same source stem.
    # Keep one catalog feature per physical image, preferring the manifest's
    # original filename and metadata over the generated PNG size entry.
    candidates = _canonical_image_candidate_names(
        sizes_index.keys(),
        gps_index.keys(),
        camera_meta_keys,
        manifest_map.keys(),
    )

    for fname in sorted(candidates):
        cam_entry = _camera_entry_for(fname)
        source_stem = Path(fname).stem
        prepared_rotated = rotated_by_stem.get(source_stem)

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
        if source_stem in aligned_centres_by_stem:
            latlon = aligned_centres_by_stem[source_stem]

        props: Dict[str, object] = {}
        props["image"] = fname
        if prepared_rotated is not None:
            props["prepared_image"] = _media_url(prepared_rotated)
            props["display_source"] = "prepared_source_image"
        # overlay/thumb from manifest
        if isinstance(manifest_map.get(fname), dict):
            entry = manifest_map[fname]
            if "overlay" in entry:
                props["overlay"] = entry["overlay"]
                props["image"] = fname
            if "thumb" in entry:
                props["thumb"] = entry["thumb"]
            for key in (
                "prepared_image", "display_source", "lens_correction_status", "lens_displacement_px"
            ):
                if key in entry:
                    props[key] = entry[key]

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
        if prepared_rotated is not None:
            try:
                with Image.open(prepared_rotated) as prepared_image:
                    w_px, h_px = prepared_image.size
            except OSError:
                pass

        entry_mpp = _coerce_positive_float(cam_entry.get("meters_per_pixel")) if cam_entry else None
        image_mpp = entry_mpp or default_mpp
        props["meters_per_pixel"] = float(image_mpp)
        if cam_entry and isinstance(cam_entry.get("row_alignment"), dict):
            alignment_record = cam_entry["row_alignment"]
            props["row_alignment_status"] = alignment_record.get("status")
            props["row_alignment_confidence"] = alignment_record.get("confidence")
            props["row_alignment_position_correction_m"] = alignment_record.get("position_correction_m")
            props["row_alignment_rotation_correction_deg"] = alignment_record.get("rotation_correction_deg")

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
            if prepared_rotated is not None:
                heading_deg = _coerce_float((cam_entry or {}).get("row_alignment_rotation_deg")) or 0.0
            else:
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

    # ---------------- predictions.geojson (bbox → polygon using image center) ----------------
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
        polygons = jd.get("polygons", []) or []
        srcfile = jd.get("file") or (jpath.stem + ".png")

        # Resolve the authoritative camera record first. Row alignment updates
        # this per-result metadata while leaving the source image EXIF intact.
        # Predictions and image footprints must therefore share this centre.
        cam_entry = _camera_entry_for(srcfile)
        latlon = None
        if cam_entry:
            lat_value = _coerce_float(cam_entry.get("lat"))
            lon_value = _coerce_float(cam_entry.get("lon"))
            if lat_value is not None and lon_value is not None:
                latlon = (lat_value, lon_value)
        if latlon is None:
            latlon = gps_index.get(srcfile)
        wh     = sizes_index.get(srcfile)

        if not latlon or not wh:
            stem = Path(srcfile).stem
            for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
                if latlon is None:
                    alternate_entry = _camera_entry_for(stem + ext)
                    if alternate_entry:
                        lat_value = _coerce_float(alternate_entry.get("lat"))
                        lon_value = _coerce_float(alternate_entry.get("lon"))
                        if lat_value is not None and lon_value is not None:
                            latlon = (lat_value, lon_value)
                            cam_entry = alternate_entry
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

        # Prepared images are north-up before optional row alignment. Preserve
        # the small residual map rotation without resampling thermal pixels.
        rotation_deg = _coerce_float((cam_entry or {}).get("row_alignment_rotation_deg")) or 0.0

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

            # Prefer the instance-mask outline. Detection models have no mask,
            # so they continue to use the four bounding-box corners.
            try:
                mask_outline = polygons[i] if i < len(polygons) else []
                corners_px = (
                    [(float(point[0]), float(point[1])) for point in mask_outline]
                    if isinstance(mask_outline, list) and len(mask_outline) >= 3
                    else [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
                )
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
                    "prediction_index": i,
                }
            })

    anom_path = out_session / "predictions.geojson"
    anom_path.write_text(json.dumps(anom_fc, indent=2), encoding="utf-8")

    return anom_path, imgs_path


def _append_mosaic_source_images_geojson(
    out_session: Path,
    camera_meta: Dict[str, Dict[str, Any]],
) -> Path:
    """Add corrected/rotated mosaic inputs to the map catalog without predicting them."""
    out_session = Path(out_session)
    images_path = out_session / "images.geojson"
    try:
        feature_collection = json.loads(images_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        feature_collection = {"type": "FeatureCollection", "features": []}
    features = [
        feature
        for feature in feature_collection.get("features", [])
        if feature.get("properties", {}).get("source_role") != "mosaic_input"
    ]

    rotated_by_stem = {
        path.stem: path
        for path in (out_session / "rotated_images").glob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    }
    alignment_images: Dict[str, Dict[str, Any]] = {}
    alignment_resolution: Optional[float] = None
    try:
        alignment_payload = json.loads((out_session / "mosaic_alignment.json").read_text(encoding="utf-8"))
        parsed_resolution = _coerce_float(alignment_payload.get("resolution_m_per_px"))
        alignment_resolution = parsed_resolution if parsed_resolution is not None and parsed_resolution > 0 else None
        alignment_images = {
            Path(name).stem: record
            for name, record in (alignment_payload.get("images") or {}).items()
            if isinstance(record, dict)
        }
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    preprocessing_images: Dict[str, Dict[str, Any]] = {}
    try:
        preprocessing_payload = json.loads((out_session / "preprocessing.json").read_text(encoding="utf-8"))
        preprocessing_images = {
            Path(name).stem: record
            for name, record in (preprocessing_payload.get("images") or {}).items()
            if isinstance(record, dict)
        }
    except (OSError, json.JSONDecodeError, TypeError):
        pass

    for source_name, meta in sorted((camera_meta or {}).items()):
        if str(source_name).startswith("__") or not isinstance(meta, dict):
            continue
        stem = Path(source_name).stem
        prepared_path = rotated_by_stem.get(stem)
        if prepared_path is None:
            continue
        lat = meta.get("lat", meta.get("latitude"))
        lon = meta.get("lon", meta.get("longitude"))
        alignment_record = alignment_images.get(stem, {})
        final_lat_lon = alignment_record.get("final_lat_lon")
        if isinstance(final_lat_lon, list) and len(final_lat_lon) >= 2:
            lat, lon = final_lat_lon[:2]
        if lat is None or lon is None:
            continue
        lat = float(lat)
        lon = float(lon)
        try:
            with Image.open(prepared_path) as prepared_image:
                width, height = prepared_image.size
        except OSError:
            continue
        try:
            # Display the full prepared frame at its metadata-derived ground
            # footprint. The mosaic output resolution may be coarser and causes
            # an in-memory resize during composition, not a source-image crop.
            meters_per_pixel = float(meta.get("meters_per_pixel") or alignment_resolution or 0.05)
        except (TypeError, ValueError):
            meters_per_pixel = 0.05
        if meters_per_pixel <= 0:
            meters_per_pixel = 0.05
        deg_per_m_lon, deg_per_m_lat = _meters_to_deg(lat)
        half_width_m = width * meters_per_pixel / 2.0
        half_height_m = height * meters_per_pixel / 2.0
        left = lon - half_width_m * deg_per_m_lon
        right = lon + half_width_m * deg_per_m_lon
        top = lat + half_height_m * deg_per_m_lat
        bottom = lat - half_height_m * deg_per_m_lat
        correction_record = preprocessing_images.get(stem, {})
        props: Dict[str, Any] = {
            "src": source_name,
            "image": source_name,
            "prepared_image": _media_url(prepared_path),
            "display_source": "prepared_source_image",
            "source_role": "mosaic_input",
            "inference_performed": False,
            "alignment_status": alignment_record.get("status", "gps_only"),
            "w": int(width),
            "h": int(height),
            "meters_per_pixel": meters_per_pixel,
            "width_m": width * meters_per_pixel,
            "height_m": height * meters_per_pixel,
            "rotation": 0.0,
            "rotation_heading": 0.0,
            "rotation_overlay": 0.0,
            "corners": [[left, top], [right, top], [right, bottom], [left, bottom]],
            "lens_correction_status": correction_record.get("status", "not_requested"),
            "lens_displacement_px": correction_record.get("maximum_displacement_px"),
        }
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": props,
        })

    feature_collection = {"type": "FeatureCollection", "features": features}
    images_path.write_text(json.dumps(feature_collection, indent=2), encoding="utf-8")
    return images_path






def _filter_predictions_geojson(predictions_path: Path, overlap_threshold: float = 0.20) -> Path:
    """
    Filter overlapping anomaly features by keeping highest-confidence detections.
    
    Algorithm:
      1. Group features by class (classname property)
      2. For each class, iteratively remove lower-confidence polygons when overlap >threshold% of larger polygon
      3. Iterative: removing one polygon might reveal other overlaps
      4. Write filtered_predictions.geojson in the same session directory
    
    Args:
      predictions_path: Path to predictions.geojson
      overlap_threshold: Remove lower-confidence if overlap > threshold * larger_polygon_area (default 20%)
    
    Returns:
      Path to filtered_predictions.geojson
    """
    from shapely.geometry import shape as geom_shape
    from shapely.geometry import mapping
    
    predictions_path = Path(predictions_path)
    
    # Load predictions.geojson
    try:
        anom_data = json.loads(predictions_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"Failed to load predictions.geojson: {e}, skipping filter")
        return predictions_path
    
    # Group features by classname
    features_by_class: Dict[str, list] = {}
    for feature in anom_data.get("features", []):
        classname = feature.get("properties", {}).get("classname", "unknown")
        if classname not in features_by_class:
            features_by_class[classname] = []
        features_by_class[classname].append(feature)
    
    # Filter features within each class
    filtered_features = []
    
    for classname, features in features_by_class.items():
        # Convert features to (shapely_geom, properties_dict, score, index) tuples
        geoms_with_props = []
        for idx, feature in enumerate(features):
            try:
                geom = geom_shape(feature.get("geometry", {}))
                props = feature.get("properties", {})
                score = float(props.get("score", 0))
                geoms_with_props.append((geom, props, score, idx))
            except Exception:
                # Skip invalid geometries
                pass
        
        # Iteratively remove lower-confidence overlapping polygons
        removed = True
        while removed:
            removed = False
            indices_to_remove = set()
            
            for i in range(len(geoms_with_props)):
                if i in indices_to_remove:
                    continue
                
                geom1, props1, score1, idx1 = geoms_with_props[i]
                
                # Check against all other polygons
                for j in range(i + 1, len(geoms_with_props)):
                    if j in indices_to_remove:
                        continue
                    
                    geom2, props2, score2, idx2 = geoms_with_props[j]
                    
                    # Calculate overlap
                    try:
                        intersection = geom1.intersection(geom2)
                        intersection_area = intersection.area
                        larger_area = max(geom1.area, geom2.area)
                        
                        if larger_area > 0:
                            overlap_ratio = intersection_area / larger_area
                            
                            # If overlap exceeds threshold, remove the one with lower score
                            if overlap_ratio > overlap_threshold:
                                if score1 > score2:
                                    # Remove polygon j (lower confidence)
                                    indices_to_remove.add(j)
                                    removed = True
                                elif score2 > score1:
                                    # Remove polygon i (lower confidence)
                                    indices_to_remove.add(i)
                                    removed = True
                                    break  # i is removed, no need to check further
                                else:
                                    # Equal scores: keep the one with larger area
                                    if geom1.area >= geom2.area:
                                        indices_to_remove.add(j)
                                    else:
                                        indices_to_remove.add(i)
                                        break
                                    removed = True
                    except Exception:
                        # Skip on geometry operations that fail
                        pass
            
            # Remove marked indices
            if indices_to_remove:
                geoms_with_props = [
                    item for idx, item in enumerate(geoms_with_props)
                    if idx not in indices_to_remove
                ]
        
        # Convert back to GeoJSON features
        for geom, props, score, _ in geoms_with_props:
            try:
                filtered_features.append({
                    "type": "Feature",
                    "geometry": mapping(geom),
                    "properties": props
                })
            except Exception:
                pass
    
    # Write filtered_predictions.geojson
    final_fc = {
        "type": "FeatureCollection",
        "features": filtered_features
    }
    
    final_path = predictions_path.parent / "filtered_predictions.geojson"
    final_path.write_text(json.dumps(final_fc, indent=2), encoding="utf-8")
    
    logger.info(f"Generated filtered_predictions.geojson: {len(filtered_features)} filtered features from {len(anom_data.get('features', []))} original")
    return final_path



# ---------- tiny list helpers (datasets/models/sessions) ----------

def _count_top_level_images(d: Path) -> int:
    """Count images directly under d (non-recursive)."""
    if not d.exists() or not d.is_dir():
        return 0
    return sum(1 for p in d.iterdir() if p.is_file() and _is_image(p))


def _asset_display_name(folder: Path, metadata_name: str) -> str:
    path = folder / metadata_name
    if path.is_file():
        try:
            value = json.loads(path.read_text(encoding="utf-8")).get("display_name")
            if isinstance(value, str) and value.strip():
                return value.strip()
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, AttributeError):
            pass
    return folder.name


def _write_asset_display_name(folder: Path, metadata_name: str, display_name: str) -> None:
    temp = folder / f"{metadata_name}.tmp"
    target = folder / metadata_name
    temp.write_text(json.dumps({"display_name": display_name}, indent=2), encoding="utf-8")
    temp.replace(target)


def _write_result_status(result_dir: Path, status: str, **details: Any) -> None:
    path = result_dir / "result_status.json"
    temporary = result_dir / ".result_status.json.tmp"
    payload = {"status": status, "updated_at": datetime.now().isoformat(), **details}
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _result_run_status(result_dir: Path) -> str:
    """Use explicit state for new results and required artifacts for legacy results."""
    status_path = result_dir / "result_status.json"
    if status_path.is_file():
        try:
            state = json.loads(status_path.read_text(encoding="utf-8"))
            status = str(state.get("status") or "").strip().lower()
            if status:
                return status
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, AttributeError):
            pass
    predictions_exist = any((result_dir / name).is_file() for name in ("predictions.geojson", "anomalies.geojson"))
    required_exist = all((result_dir / name).is_file() for name in ("metrics.json", "manifest.json", "images.geojson"))
    return "complete" if predictions_exist and required_exist else "incomplete"


def _project_child(root: Path, child_id: str, label: str) -> Path:
    safe_id = _safe_name(child_id)
    child = root / safe_id
    if not safe_id or not child.is_dir() or child.resolve().parent != root.resolve():
        raise HTTPException(status_code=404, detail=f"{label} not found.")
    return child


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)

def _list_datasets() -> list[dict]:
    """
    Return detailed dataset info from data/test/.
    Shape: [{"name": "<folder>", "count": <n>, "mtime": <unix-ts>}, ...]
    """
    test_dir = get_project_test_dir()
    if not test_dir.exists():
        return []
    items = []
    for p in sorted([x for x in test_dir.iterdir() if x.is_dir()],
                    key=lambda x: x.stat().st_mtime, reverse=True):
        items.append({
            "id": p.name,
            "name": p.name,
            "display_name": _asset_display_name(p, ".dataset_meta.json"),
            "count": _count_top_level_images(p),
            "mtime": int(p.stat().st_mtime),
            "colmap_ready": _colmap_ready(p.name),
            "input_type": _detect_image_input_type(p),
        })
    return items


def _list_sessions() -> list[dict]:
    """
    Return detailed sessions from test_outputs/.
    Shape: [{"name": "<session-id>", "mtime": <unix-ts>}, ...]
    """
    base = get_project_sessions_dir()
    if not base.exists():
        return []
    items = []
    for p in sorted([x for x in base.iterdir() if x.is_dir()],
                    key=lambda x: x.stat().st_mtime, reverse=True):
        if p.name.startswith("."):
            continue
        run_status = _result_run_status(p)
        complete = run_status in {"complete", "completed"}
        items.append({
            "id": p.name,
            "name": p.name,                 # normalized to just the id
            "display_name": _asset_display_name(p, ".result_meta.json"),
            "mtime": int(p.stat().st_mtime),
            "complete": complete,
            "status": "complete" if complete else "incomplete",
            "run_status": run_status,
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
        return [_media_url(p) for p in sorted(d.glob("*")) if p.is_file()]
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


@app.get("/api/features")
async def api_features():
    """Expose enabled feature flags so the UI can toggle controls."""
    return {"ok": True, "features": settings.as_feature_payload()}


def _thermal_job_payload(job: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in job.items() if key != "task"}


@app.post("/api/thermal-convert/scan")
async def api_scan_thermal_conversion_folder(
    input_dir: str = Form(...),
    conversion_type: str = Form("radiometric"),
    include_radiometric: bool = Form(False),
):
    source = _resolve_thermal_directory(input_dir, must_exist=True)
    try:
        scan = await asyncio.to_thread(
            scan_conversion_folder, source, conversion_type=conversion_type,
            include_radiometric=include_radiometric,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "scan": scan}


@app.post("/api/thermal-convert/start")
async def api_start_thermal_conversion(
    input_dir: str = Form(...),
    output_dir: str = Form(...),
    conversion_type: str = Form("radiometric"),
    include_radiometric: bool = Form(False),
    output_format: str = Form("jpg"),
    quality: int = Form(100),
    overwrite: bool = Form(False),
):
    mode = str(conversion_type).strip().lower()
    if mode not in {"radiometric", "standard"}:
        raise HTTPException(status_code=400, detail="Conversion type must be radiometric or standard.")
    if mode == "radiometric":
        _require_thermal_enabled("convert radiometric thermal images")
    source = _resolve_thermal_directory(input_dir, must_exist=True)
    destination = _resolve_thermal_directory(output_dir, must_exist=False)
    if source == destination:
        raise HTTPException(status_code=400, detail="Input and output folders must be different.")
    fmt = str(output_format).lower().lstrip(".")
    if fmt not in {"jpg", "jpeg", "png"}:
        raise HTTPException(status_code=400, detail="Output format must be JPG or PNG.")
    if not 1 <= quality <= 100:
        raise HTTPException(status_code=400, detail="JPEG quality must be between 1 and 100.")
    scan = await asyncio.to_thread(
        scan_conversion_folder,
        source,
        conversion_type=mode,
        include_radiometric=include_radiometric,
    )
    if not scan["supported"]:
        if mode == "standard":
            detail = (
                "No eligible standard JPG, JPEG, or PNG images were found directly in the input folder. "
                "Radiometric JPEGs are skipped unless the visible-pixels option is selected."
            )
        else:
            detail = (
                "No supported radiometric JPG/JPEG images were found directly in the input folder. "
                "Supported payloads are DJI DIRP (including M3T/M3TD) and FLIR FFF "
                "(including DJI Zenmuse XT2)."
            )
        raise HTTPException(
            status_code=400,
            detail=detail,
        )
    for existing in THERMAL_CONVERT_JOBS.values():
        if existing.get("status") in {"queued", "running"} and existing.get("output_dir") == str(destination):
            raise HTTPException(status_code=409, detail="A conversion is already writing to that output folder.")

    job_id = uuid.uuid4().hex[:12]
    job: Dict[str, Any] = {
        "id": job_id,
        "status": "queued",
        "input_dir": str(source),
        "output_dir": str(destination),
        "conversion_type": mode,
        "include_radiometric": bool(include_radiometric) if mode == "standard" else False,
        "output_format": "jpg" if fmt == "jpeg" else fmt,
        "quality": quality,
        "overwrite": overwrite,
        "total": scan["supported"],
        "completed": 0,
        "converted": 0,
        "skipped": 0,
        "failed": 0,
        "unsupported": scan["unsupported"],
        "excluded_radiometric": scan["excluded_radiometric"],
        "ignored_images": scan["ignored_images"],
        "cameras": scan["cameras"],
        "current_file": None,
        "first_error": None,
        "cancel_requested": False,
        "created_at": datetime.now().isoformat(),
        "finished_at": None,
    }
    THERMAL_CONVERT_JOBS[job_id] = job

    async def _runner():
        job["status"] = "running"

        def _progress(state: Dict[str, object]) -> None:
            for key in (
                "total", "completed", "converted", "skipped", "failed", "current_file",
                "first_error", "unsupported", "excluded_radiometric", "ignored_images", "cameras",
            ):
                if key in state:
                    job[key] = state[key]

        try:
            result = await asyncio.to_thread(
                convert_thermal_folder,
                source,
                destination,
                output_format=job["output_format"],
                conversion_type=mode,
                include_radiometric=include_radiometric,
                quality=quality,
                overwrite=overwrite,
                progress=_progress,
                should_cancel=lambda: bool(job.get("cancel_requested")),
            )
            _progress(result)
            job["status"] = "cancelled" if result.get("cancelled") else "completed"
        except Exception as exc:
            job["status"] = "failed"
            job["first_error"] = str(exc)
            logger.exception("Image grayscale conversion job %s failed", job_id)
        finally:
            job["finished_at"] = datetime.now().isoformat()
            job.pop("task", None)

    task = asyncio.create_task(_runner())
    job["task"] = task
    return {"ok": True, "job": _thermal_job_payload(job)}


@app.get("/api/thermal-convert/{job_id}")
async def api_thermal_conversion_status(job_id: str):
    job = THERMAL_CONVERT_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Thermal conversion job not found.")
    return {"ok": True, "job": _thermal_job_payload(job)}


@app.post("/api/thermal-convert/{job_id}/cancel")
async def api_cancel_thermal_conversion(job_id: str):
    job = THERMAL_CONVERT_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Thermal conversion job not found.")
    if job.get("status") in {"queued", "running"}:
        job["cancel_requested"] = True
    return {"ok": True, "job": _thermal_job_payload(job)}

# ================== projects CRUD ==================

@app.get("/api/projects")
async def list_projects():
    """List all projects."""
    projects = project_manager.list_projects()
    return {"projects": [p.model_dump() for p in projects]}

@app.get("/api/projects/default-root")
async def get_default_projects_root():
    """Get default root directory where new projects are created."""
    return {"default_root": str(DEFAULT_PROJECTS_ROOT.resolve())}

@app.get("/api/projects/{project_id}")
async def get_project(project_id: str):
    """Get project by ID."""
    project = project_manager.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project.model_dump()

@app.post("/api/projects")
async def create_project(
    name: str = Form(...),
    description: str = Form(default=""),
    root_path: str = Form(default="")
):
    """Create a new project."""
    import uuid as uuid_lib
    project_id = str(uuid_lib.uuid4())[:8]

    if not name.strip():
        raise HTTPException(status_code=400, detail="Project name is required")

    project_name = name.strip()
    
    if root_path and root_path.strip():
        # User provided a parent directory path
        parent_path = root_path.strip()
        original_path = parent_path  # Keep original for logging
        
        # Auto-detect and convert Windows paths silently
        if looks_like_windows_path(parent_path):
            parent_path = convert_windows_path_to_wsl(parent_path)
            logging.info(f"Converting Windows path: {original_path} -> {parent_path}")
        
        # Check if parent directory exists
        parent_dir = Path(parent_path)
        if not parent_dir.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Parent directory does not exist: '{parent_path}' (converted from: '{original_path}'). Please create the parent folder first or mount the drive if it's external."
            )
        
        # Create project folder inside the parent directory
        resolved_root_path = str((parent_dir / project_name).resolve())
        logging.info(f"Parent directory: {parent_dir}")
        logging.info(f"Project name: {project_name}")
        logging.info(f"Creating project at: {resolved_root_path}")
        
        # Verify path has project name in it
        if resolved_root_path.endswith(project_name.replace(" ", "\\ ")):
            logging.info(f"Path validation OK: ends with project name")
        else:
            logging.warning(f"Path validation: path does NOT end with '{project_name}'")
    else:
        # Use default projects root
        resolved_root_path = str((DEFAULT_PROJECTS_ROOT / project_name).resolve())
        logging.info(f"Using default project path: {resolved_root_path}")

    root = Path(resolved_root_path)
    
    # Create the directory if it doesn't exist
    if not root.exists():
        try:
            root.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created directory: {resolved_root_path}")
            
            # Verify it was actually created
            if not root.exists():
                logging.error(f"Directory creation reported success but folder doesn't exist: {resolved_root_path}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Directory creation failed: Path '{resolved_root_path}' was not created. The parent directory may not be writable."
                )
            
            # Try to create a test file to verify write access
            test_file = root / ".test_write"
            try:
                test_file.write_text("test")
                test_file.unlink()
                logging.info(f"Write access verified for: {resolved_root_path}")
            except Exception as write_err:
                logging.error(f"Write access failed for {resolved_root_path}: {write_err}")
                raise HTTPException(
                    status_code=400,
                    detail=f"No write permissions at '{resolved_root_path}': {str(write_err)}"
                )
                
        except PermissionError as pe:
            logging.error(f"Permission denied creating {resolved_root_path}: {pe}")
            raise HTTPException(
                status_code=400, 
                detail=f"Permission denied: Cannot create directory at '{resolved_root_path}'. Check if the drive/path is accessible and you have write permissions."
            )
        except HTTPException:
            raise
        except Exception as e:
            logging.error(f"Failed to create directory {resolved_root_path}: {e}")
            raise HTTPException(
                status_code=400, 
                detail=f"Failed to create directory at '{resolved_root_path}': {str(e)}"
            )
    else:
        logging.info(f"Directory already exists: {resolved_root_path}")
    
    project = Project(
        id=project_id,
        name=project_name,
        description=description,
        root_path=str(root.resolve())
    )
    
    try:
        created = project_manager.create_project(project)
        # Return the created path with success message
        result = created.model_dump()
        result["created_path"] = str(root.resolve())
        result["success_message"] = f"Project '{project_name}' created successfully at: {result['created_path']}"
        logging.info(f"Project created successfully: {result['created_path']}")
        return result
    except ValueError as e:
        logging.error(f"Failed to create project in registry: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.put("/api/projects/{project_id}")
async def update_project(
    project_id: str,
    name: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    thumbnail_path: Optional[str] = Form(None)
):
    """Update project metadata."""
    updates = {}
    if name is not None:
        updates["name"] = name
    if description is not None:
        updates["description"] = description
    if thumbnail_path is not None:
        updates["thumbnail_path"] = thumbnail_path
    
    project = project_manager.update_project(project_id, updates)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project.model_dump()

@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str):
    """Delete project from registry and remove all associated files and folders."""
    try:
        # Get project before deleting to access root_path
        project = project_manager.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Delete the project directory if it exists
        project_root = Path(project.root_path)
        if project_root.exists():
            try:
                shutil.rmtree(project_root)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to delete project folder: {str(e)}")
        
        # Delete from registry
        if not project_manager.delete_project(project_id):
            raise HTTPException(status_code=404, detail="Project not found")
        
        return {"ok": True}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/api/projects/{project_id}/activate")
async def activate_project(project_id: str):
    """Set the active project."""
    try:
        project = set_active_project(project_id)
        return {"ok": True, "project": project.model_dump()}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/api/active-project")
async def get_active_project_endpoint():
    """Get the currently active project."""
    try:
        project = get_active_project()
        return project.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/cancel")
async def api_cancel(job: str = Form(...)):
    job = job.strip().lower()
    if job == "train":
        CANCEL_FLAGS["train"] = True
        logger.info("UI:INFO:train: Cancel requested (best-effort).")
        return {"ok": True}
    raise HTTPException(status_code=400, detail=f"Unknown job: {job}")

# ================== TRAIN ==================

@app.get("/api/training-datasets")
async def api_training_datasets():
    project = get_active_project()
    await asyncio.to_thread(ensure_legacy_dataset, project.get_train_dir())
    return {
        "ok": True,
        "datasets": list_training_datasets(project.get_train_dir()),
    }


@app.get("/api/training-datasets/{dataset_id}")
async def api_training_dataset_detail(dataset_id: str):
    project = get_active_project()
    dataset = get_training_dataset(project.get_train_dir(), dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Training dataset not found.")
    return {"ok": True, "dataset": dataset}


@app.post("/api/training-datasets/{dataset_id}/rename")
async def api_rename_training_dataset(dataset_id: str, name: str = Form(...)):
    project = get_active_project()
    try:
        dataset = await asyncio.to_thread(
            rename_training_dataset,
            project.get_train_dir(),
            dataset_id,
            name,
        )
    except DatasetUploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    logger.info("UI:OK:train: Renamed training dataset %s to %s", dataset_id, dataset["display_name"])
    return {"ok": True, "dataset": dataset}


@app.delete("/api/training-datasets/{dataset_id}")
async def api_delete_training_dataset(dataset_id: str):
    project = get_active_project()
    try:
        dataset = await asyncio.to_thread(
            delete_training_dataset,
            project.get_train_dir(),
            dataset_id,
        )
    except DatasetUploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        logger.exception("Failed to delete training dataset %s", dataset_id)
        raise HTTPException(status_code=500, detail=f"Could not delete training dataset: {exc}") from exc
    logger.info("UI:OK:train: Deleted training dataset %s", dataset_id)
    return {"ok": True, "id": dataset_id, "display_name": dataset.get("display_name")}


@app.post("/api/training-datasets/upload")
async def api_upload_training_dataset(
    files: List[UploadFile] = File(...),
    display_name: str = Form(...),
):
    if not files:
        raise HTTPException(status_code=400, detail="Choose a ZIP file or a dataset folder.")
    name = str(display_name or "").strip()[:128]
    if not name:
        raise HTTPException(status_code=400, detail="Training dataset name is required.")
    project = get_active_project()
    destination = _project_training_dataset_destination(project, name)
    try:
        dataset = await asyncio.to_thread(
            install_training_dataset,
            files=files,
            destination=destination,
            display_name=name,
            requested_format="auto",
            project_train_dir=project.get_train_dir(),
        )
    except DatasetUploadError as exc:
        detail: Any = {"message": str(exc)}
        if exc.report:
            detail["validation"] = exc.report
        raise HTTPException(status_code=400, detail=detail) from exc
    except OSError as exc:
        logger.exception("Training dataset upload failed")
        raise HTTPException(status_code=500, detail=f"Could not store training dataset: {exc}") from exc
    logger.info("UI:OK:train: Uploaded training dataset %s (%s)", name, dataset["id"])
    return {"ok": True, "dataset": dataset}

@app.post("/api/train")
async def api_train(
    use_thermal: bool = Form(False),
    max_iter: int = Form(1000),
    base_lr: float = Form(0.00025),
    ims_per_batch: int = Form(2),
    model_name: str = Form("") ,
    backend: str = Form("detectron"),
    task: str = Form("detect"),
    model_type: str = Form("fasterrcnn"),
    yolo_family: str = Form("v8"),
    yolo_seg: bool = Form(False),
    yolo_size: str = Form("s"),
    selected_bands: str = Form(None),
    channel_count: int = Form(3),
    clear_existing: bool = Form(False),
    dataset_id: str = Form(""),
):
    if not settings.enabled_backends:
        raise HTTPException(status_code=400, detail="No training backends are enabled on this server.")

    if backend not in settings.enabled_backends:
        raise HTTPException(status_code=400, detail=f"Backend '{backend}' is not available on this server.")

    task = str(task or "detect").strip().lower()
    if task not in {"detect", "segment"}:
        raise HTTPException(status_code=400, detail="Training task must be object detection or instance segmentation.")
    model_type = "maskrcnn" if task == "segment" else "fasterrcnn"
    yolo_seg = task == "segment"
    if backend == "yolo" and yolo_seg and str(yolo_family).lower() != "v8":
        raise HTTPException(status_code=400, detail="YOLO instance segmentation currently uses the YOLOv8 family.")

    if use_thermal:
        _require_thermal_enabled("train models that rely on thermal decoding")

    project = get_active_project()
    await asyncio.to_thread(ensure_legacy_dataset, project.get_train_dir())
    selected_dataset_id = str(dataset_id or "").strip()
    if not selected_dataset_id:
        legacy = next(
            (item for item in list_training_datasets(project.get_train_dir()) if item.get("source") == "legacy_project_data"),
            None,
        )
        if not legacy:
            raise HTTPException(status_code=400, detail="Select a validated training dataset.")
        selected_dataset_id = str(legacy["id"])
    try:
        resolved_dataset = await asyncio.to_thread(
            resolve_dataset_for_training,
            project.get_train_dir(),
            selected_dataset_id,
            backend,
            task,
        )
    except DatasetUploadError as exc:
        detail = str(exc)
        if exc.report and exc.report.get("errors"):
            detail += " " + " ".join(str(value) for value in exc.report["errors"][:5])
        raise HTTPException(status_code=400, detail=detail) from exc
    dataset_entry = resolved_dataset["entry"]
    train_dir = Path(resolved_dataset["train_dir"])
    valid_dir = Path(resolved_dataset["valid_dir"])

    safe_name = _safe_name(model_name) or _now_stamp()
    output_dir = get_project_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / safe_name
    
    # Clear existing directory if requested
    if clear_existing and run_dir.exists():
        import shutil
        try:
            shutil.rmtree(run_dir)
            logger.info(f"[train] Cleared existing run directory: {run_dir.name}")
        except Exception as e:
            logger.warning(f"[train] Failed to clear existing directory: {e}")
    
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_model_meta(run_dir, {
        "model_name": run_dir.name,
        "backend": backend,
        "task": task,
        "model_type": (
            f"yolo{str(yolo_family).lower()}{str(yolo_size).lower()}{'-seg' if yolo_seg else ''}"
            if backend == "yolo" else model_type
        ),
        "yolo_seg": bool(yolo_seg),
        "training_dataset": {
            "id": str(dataset_entry["id"]),
            "name": str(dataset_entry.get("display_name") or ""),
            "path": str(dataset_entry.get("storage_path") or ""),
        },
    })
    _write_run_status(
        run_dir,
        "running",
        backend=backend,
        requested_iterations=max_iter,
        dataset_id=str(dataset_entry["id"]),
        task=task,
    )

    CANCEL_FLAGS["train"] = False
    logger.info("UI:OK:train: Training started…")
    logger.info(
        f"[train] run={run_dir.name} backend={backend} "
        f"dataset={dataset_entry['id']} task={task} use_thermal={use_thermal} "
        f"iters={max_iter} lr={base_lr} batch={ims_per_batch}"
    )

    # Prepare thermal pairs if requested
    if use_thermal:
        ensure_dirp_init()
        scan_split_decode_thermal(train_dir)
        scan_split_decode_thermal(valid_dir)

    # Offload the heavy training to a background thread so SSE can stream
    def _do_train():
        if backend == "detectron":
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
                    train_dir=train_dir,
                    val_dir=valid_dir,
                    out_dir=run_dir,
                    use_thermal_request=use_thermal,
                    max_iter=max_iter,
                    base_lr=base_lr,
                    ims_per_batch=ims_per_batch,
                    run_name=run_dir.name,
                    task=task,
                    model_type=model_type,
                    yolo_family=yolo_family,
                    yolo_seg=yolo_seg,
                    yolo_size=yolo_size,
                    selected_bands=[b.strip() for b in (selected_bands.split(',') if selected_bands else [])] or None,
                    channel_count=int(channel_count or 3),
                    dataset_id=str(dataset_entry["id"]),
                    dataset_name=str(dataset_entry.get("display_name") or ""),
                    dataset_path=str(dataset_entry.get("storage_path") or ""),
                    dataset_format=str(resolved_dataset.get("dataset_format") or ""),
                    dataset_yaml=resolved_dataset.get("dataset_yaml"),
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
        _write_run_status(run_dir, "complete", backend=backend, task=task, requested_iterations=max_iter)
        logger.info(f"[train] complete: run={run_dir.name}")
        logger.info("UI:OK:train: Training completed.")
        return {"ok": True, "run": run_dir.name, "meta": meta}
    except Exception as e:
        # Log full traceback and include it in the HTTP error detail to aid debugging.
        import traceback
        tb = traceback.format_exc()
        _write_run_status(
            run_dir,
            "cancelled" if CANCEL_FLAGS.get("train") else "failed",
            backend=backend,
            task=task,
            requested_iterations=max_iter,
            error=str(e),
        )
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
async def api_models(backend: Optional[str] = None, include_incomplete: bool = False):
    """List model runs. If `backend` query param is provided, return only models
    whose metadata `backend` matches (case-insensitive). This makes the
    frontend filtering reliable even when client-side heuristics fail.
    """
    models = _list_models(include_incomplete=include_incomplete)
    if backend:
        try:
            b = str(backend).lower()
            models = [m for m in models if str(m.get("backend", "")).lower() == b]
        except Exception:
            pass
    return {"ok": True, "models": models}


@app.get("/api/models/{model_id}")
async def api_model_detail(model_id: str):
    run_dir, meta = _find_model(model_id)
    run_status = _model_run_status(run_dir, meta)
    complete = run_status in {"complete", "completed"}
    state = {}
    state_path = run_dir / "run_status.json"
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            state = {}
    checkpoints = [
        name for name in ("model_best.pth", "model_final.pth", "model_best.pt", "model_final.pt")
        if (run_dir / name).is_file()
    ]
    return {
        "ok": True,
        "model": {
            "id": _model_id(run_dir, meta),
            "name": run_dir.name,
            "display_name": meta.get("display_name") or meta.get("model_name") or run_dir.name,
            "status": "complete" if complete else "incomplete",
            "run_status": run_status,
            "mtime": int(run_dir.stat().st_mtime),
            "path": str(run_dir),
            "checkpoints": checkpoints,
        },
        "meta": meta,
        "state": state,
        "latest_metrics": _latest_training_metrics(run_dir),
    }


@app.post("/api/models/{model_id}/rename")
async def api_rename_model(model_id: str, name: str = Form(...)):
    """Rename a model's display label without changing its import path/id."""
    display_name = str(name or "").strip()[:128]
    if not display_name:
        raise HTTPException(status_code=400, detail="Model name cannot be empty.")

    run_dir, meta = _find_model(model_id)
    if _model_run_status(run_dir, meta) not in {"complete", "completed"}:
        raise HTTPException(status_code=400, detail="Incomplete training runs cannot be renamed.")
    meta["display_name"] = display_name
    _write_model_meta(run_dir, meta)
    logger.info("UI:OK:train: Renamed model %s to %s", run_dir.name, display_name)
    return {
        "ok": True,
        "id": run_dir.name,
        "name": run_dir.name,
        "display_name": display_name,
    }


@app.delete("/api/models/{model_id}")
async def api_delete_model(model_id: str):
    """Delete one trained-model output directory; training data is untouched."""
    run_dir, _ = _find_model(model_id)
    output_dir = get_project_output_dir().resolve()
    resolved_run = run_dir.resolve()
    if resolved_run.parent != output_dir:
        raise HTTPException(status_code=400, detail="Invalid model path.")
    try:
        shutil.rmtree(resolved_run)
    except OSError as exc:
        logger.exception("Failed to delete model %s", run_dir.name)
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {exc}") from exc
    logger.info("UI:OK:train: Deleted model %s", run_dir.name)
    return {"ok": True, "id": run_dir.name}


# ================== TEST: dataset intake ==================

@app.get("/api/test_datasets")
async def api_test_datasets():
    details = _list_datasets()                      # current shape: [{name, count, mtime}, ...]
    names = [d["name"] for d in details]           # simple shape: ["name", ...]
    return {"ok": True, "datasets": details, "dataset_names": names}


@app.post("/api/test-datasets/{dataset_id}/rename")
async def api_rename_test_dataset(dataset_id: str, name: str = Form(...)):
    display_name = str(name or "").strip()[:128]
    if not display_name:
        raise HTTPException(status_code=400, detail="Test dataset name cannot be empty.")
    dataset_dir = _project_child(get_project_test_dir(), dataset_id, "Test dataset")
    _write_asset_display_name(dataset_dir, ".dataset_meta.json", display_name)
    logger.info("UI:OK:test: Renamed test dataset %s to %s", dataset_dir.name, display_name)
    return {"ok": True, "id": dataset_dir.name, "name": dataset_dir.name, "display_name": display_name}


@app.delete("/api/test-datasets/{dataset_id}")
async def api_delete_test_dataset(dataset_id: str):
    dataset_dir = _project_child(get_project_test_dir(), dataset_id, "Test dataset")
    try:
        shutil.rmtree(dataset_dir)
    except OSError as exc:
        logger.exception("Failed to delete test dataset %s", dataset_dir.name)
        raise HTTPException(status_code=500, detail=f"Failed to delete test dataset: {exc}") from exc
    logger.info("UI:OK:test: Deleted test dataset %s", dataset_dir.name)
    return {"ok": True, "id": dataset_dir.name}


@app.get("/api/dataset_bands")
async def api_dataset_bands(dataset: str):
    """Return detected bands for a dataset folder under data/test or data/train.
    Query param `dataset` may be a folder name under data/test or the literal 'train'/'valid'.
    """
    # resolve dataset path
    ds = None
    if dataset in ("train", "valid"):
        base = get_project_data_dir()
        ds = base / dataset
    else:
        ds = get_project_test_dir() / dataset

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
    _require_thermal_enabled("decode thermal datasets")
    # resolve dataset path similarly to above
    if dataset in ("train", "valid"):
        ds = get_project_data_dir() / dataset
    else:
        ds = get_project_test_dir() / dataset

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
    project_id: Optional[str] = Form(default=None),
    model: Optional[str] = Form(default=None),
    use_thermal: bool = Form(default=False),
    result_name: str = Form(default=""),
    test_threshold: str = Form(default="0.8"),
    forced_backend: Optional[str] = Form(default=None),
    backend: Optional[str] = Form(default=None),
    selected_bands: str = Form(None),
    channel_count: int = Form(3),
    accurate_locations: bool = Form(default=False),
    mosaic_enabled: bool = Form(default=False),
    create_mosaic: bool = Form(default=False),
    inference_source: Optional[str] = Form(default=None),
    refine_mosaic_alignment: bool = Form(default=False),
    undistort_thermal: bool = Form(default=False),
    export_undistorted_images: bool = Form(default=False),
    target_surface_height_m: float = Form(default=DEFAULT_TARGET_SURFACE_HEIGHT_M),
    align_images_to_rows: bool = Form(default=False),
    row_alignment_job_id: Optional[str] = Form(default=None),
    row_alignment_path: Optional[str] = Form(default=None),
    row_alignment_max_position_m: float = Form(default=8.0),
    row_alignment_max_rotation_deg: float = Form(default=10.0),
    optimization_project: Optional[str] = Form(default=None),
    clear_existing: bool = Form(default=False),
):
    if not settings.enabled_backends:
        raise HTTPException(status_code=400, detail="No inference backends are enabled on this server.")
    if not math.isfinite(target_surface_height_m) or target_surface_height_m < 0:
        raise HTTPException(status_code=400, detail="Target surface height must be zero or a positive number in metres.")
    if not math.isfinite(row_alignment_max_position_m) or not 0.5 <= row_alignment_max_position_m <= 50.0:
        raise HTTPException(status_code=400, detail="Maximum row-alignment position correction must be between 0.5 and 50 metres.")
    if not math.isfinite(row_alignment_max_rotation_deg) or not 0.0 <= row_alignment_max_rotation_deg <= 45.0:
        raise HTTPException(status_code=400, detail="Maximum row-alignment orientation correction must be between 0 and 45 degrees.")

    requested_backend = backend or forced_backend
    if requested_backend and requested_backend not in settings.enabled_backends:
        raise HTTPException(status_code=400, detail=f"Backend '{requested_backend}' is not available on this server.")

    project = project_manager.get_project(project_id) if project_id else get_active_project()
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")

    test_dir = get_project_test_dir(project)
    ds_dir = test_dir / dataset
    
    if not ds_dir.exists() or not ds_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset}' not found.")

    thermal_enabled = settings.enable_thermal_data_extraction

    if model:
        model_dir = get_project_output_dir(project) / model
        if not model_dir.exists():
            raise HTTPException(status_code=404, detail=f"Model '{model}' not found.")
    else:
        models = _list_models(project=project)
        if not models:
            raise HTTPException(status_code=404, detail="No trained models found.")
        model_dir = get_project_output_dir(project) / models[0]["name"]

    legacy_mosaic_request = bool(mosaic_enabled and inference_source is None)
    source_mode = str(inference_source or ("mosaic" if legacy_mosaic_request else "individual")).strip().lower()
    if source_mode not in {"individual", "mosaic"}:
        raise HTTPException(status_code=400, detail="Inference source must be ‘individual’ or ‘mosaic’.")
    mosaic_enabled = bool(create_mosaic or mosaic_enabled or source_mode == "mosaic")
    rows_source_path: Optional[Path] = None
    if align_images_to_rows:
        if mosaic_enabled:
            raise HTTPException(
                status_code=400,
                detail="Rows alignment currently supports individual-image inference only. Disable approximate mosaic creation.",
            )
        if not row_alignment_job_id or not row_alignment_path:
            raise HTTPException(status_code=400, detail="Select a post-processing job and Rows GeoJSON for image alignment.")
        try:
            rows_source_path = resolve_rows_source(
                get_project_sessions_dir(project), row_alignment_job_id, row_alignment_path,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    session = (_safe_name(result_name) or _now_stamp())
    base = get_project_sessions_dir(project)
    ses = base / session

    active_run_key = str(ses.resolve())
    async with _ACTIVE_TEST_RUNS_GUARD:
        if active_run_key in _ACTIVE_TEST_RUNS:
            raise HTTPException(
                status_code=409,
                detail=f"A test named '{session}' is already running. Wait for it to finish or use another result name.",
            )
        if ses.exists() and any(ses.iterdir()) and not clear_existing:
            raise HTTPException(
                status_code=409,
                detail=f"A result named '{session}' already exists. Confirm replacement or use another result name.",
            )
        _ACTIVE_TEST_RUNS.add(active_run_key)
    request_task = asyncio.current_task()
    if request_task is not None:
        request_task.add_done_callback(lambda _task, key=active_run_key: _ACTIVE_TEST_RUNS.discard(key))
    
    # Clear existing session directory if requested
    if clear_existing and ses.exists():
        import shutil
        try:
            shutil.rmtree(ses)
            logger.info(f"[test] Cleared existing session directory: {ses.name}")
        except Exception as e:
            logger.warning(f"[test] Failed to clear existing session: {e}")

    out_root = base / session
    out_root.mkdir(parents=True, exist_ok=True)
    _write_result_status(
        out_root,
        "running",
        dataset=dataset,
        model=model_dir.name,
        lens_correction_requested=bool(undistort_thermal),
        undistort_thermal=bool(undistort_thermal),
        export_undistorted_images_requested=bool(export_undistorted_images and undistort_thermal),
        inference_source=source_mode,
        mosaic_created=mosaic_enabled,
        target_surface_height_m=float(target_surface_height_m),
        align_images_to_rows=bool(align_images_to_rows),
        row_alignment_job_id=row_alignment_job_id if align_images_to_rows else None,
        row_alignment_path=row_alignment_path if align_images_to_rows else None,
    )
    test_logger = logging.getLogger("pvrt.test")
    test_logger.info(
        "UI:INFO:test: Preparing test run '%s' with dataset '%s' and model '%s'…",
        session, dataset, model_dir.name,
    )
    test_logger.info(
        "UI:INFO:test: Target surface height above takeoff: %.2f m%s.",
        target_surface_height_m,
        " (unadjusted relative altitude)" if target_surface_height_m == 0 else "",
    )

    # --- ADD: decide whether this dataset is a single GeoTIFF ---
    input_type = _detect_image_input_type(ds_dir)
    test_logger.info(
        "UI:INFO:test: Input detected: %s.",
        "orthomosaic GeoTIFF" if input_type == "tif" else "image folder",
    )
    if input_type == "tif" and undistort_thermal:
        test_logger.info(
            "UI:OK:test: Lens correction skipped: orthophotos are already geometrically corrected products."
        )
        undistort_thermal = False
    if align_images_to_rows and input_type != "images":
        raise HTTPException(status_code=400, detail="Rows alignment requires a folder of individual images.")

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
        # If the selected model was trained for thermal then attempt
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

        # Tile in a worker thread so the event loop remains free to deliver
        # progress messages to both SSE log panes while disk work is running.
        tiles_dir = out_root / "tiles"
        test_logger.info("UI:INFO:test: Preparing orthomosaic tiles for inference…")
        await asyncio.to_thread(
            _tile_tif_to_dir, tif_src, tiles_dir, 1024, 1024,
        )
        run_images_dir = tiles_dir

        # small preview (optional)
        test_logger.info("UI:INFO:test: Creating orthomosaic preview…")
        thumb_path = await asyncio.to_thread(_save_tif_thumbnail, tif_src, out_root / "thumbs")
        test_logger.info("UI:OK:test: Orthomosaic preview ready.")

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
    model_is_thermal = model_mode == "thermal" or bool(meta.get("thermal_used"))

    if model_is_thermal and not thermal_enabled:
        _require_thermal_enabled("run inference with thermal-trained models")

    # Determine model's declared channel count (always 3 for supported models)
    try:
        model_chan = int(meta.get("channel_count") or 3)
        # Enforce 3-channel constraint
        if model_chan != 3:
            model_chan = 3
    except Exception:
        model_chan = 3

    # If model expects thermal for inference and this is an images dataset, run
    # the idempotent decode pass which will populate images_dir/thermal/pairs.json
    # when RJPEG payloads exist. This lets us infer availability afterwards.
    if input_type == "images" and model_is_thermal:
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
    use_thermal_effective = bool(model_is_thermal and data_has_thermal)

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
        if not settings.enable_colmap:
            raise HTTPException(status_code=400, detail="Accurate locations require COLMAP support, which is disabled on this server.")
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
            base_meta = _build_camera_meta_from_exif(ds_dir, target_surface_height_m)
            for key, entry in base_meta.items():
                existing = _lookup_camera_meta_entry(camera_meta, key)
                if existing is None:
                    camera_meta[key] = entry
                    continue
                # Retain the optimized pose but use the target-plane GSD
                # selected for this thermal test.
                for field in (
                    "alt", "alt_source", "absolute_altitude", "meters_per_pixel",
                    "meters_per_pixel_method", "meters_per_pixel_model",
                    "meters_per_pixel_source", "meters_per_pixel_status",
                    "meters_per_pixel_warning", "target_surface_height_m", "gsd_altitude_m",
                ):
                    if field in entry:
                        existing[field] = entry[field]
                    else:
                        existing.pop(field, None)
        except Exception:
            pass
    elif input_type == "images":
        try:
            camera_meta = _build_camera_meta_from_exif(ds_dir, target_surface_height_m)
        except Exception as e:
            logger.warning(f"Failed to derive camera metadata from EXIF: {e}")
            camera_meta = {}
    
    # Apply optimization_project merge if provided (thermal + optical geometry)
    if optimization_project and input_type == "images" and camera_meta and not accurate_locations:
        if not settings.enable_colmap:
            logging.getLogger("pvrt.test").info("UI:INFO:test: COLMAP disabled; skipping optimization project merge.")
        else:
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

    if input_type == "images":
        test_logger.info(
            "UI:INFO:test: Automatic lens correction is %s.",
            "enabled with fail-closed runtime calibration" if undistort_thermal else "disabled by user",
        )

    # ===== PRE-INFERENCE IMAGE ROTATION & OPTIONAL MOSAIC =====
    test_logger.info("UI:INFO:test: Finalizing prepared inputs…")
    try:
        rotation_result = await asyncio.to_thread(
            prepare_rotation_and_mosaic,
            input_type=input_type,
            session_dir=session_dir,
            out_root=out_root,
            camera_meta=camera_meta,
            mosaic_enabled=mosaic_enabled,
            inference_source=source_mode,
            refine_mosaic_alignment=refine_mosaic_alignment,
            ds_dir=ds_dir,
            model_is_thermal=model_is_thermal,
            undistort_thermal=bool(undistort_thermal),
            export_undistorted_images=bool(export_undistorted_images and undistort_thermal),
            tile_tif_func=_tile_tif_to_dir,
            run_images_dir=run_images_dir,
            tiles_dir=tiles_dir,
            tif_src=tif_src,
        )
    except Exception as exc:
        message = str(exc) or "Image preparation failed."
        _write_result_status(out_root, "failed", dataset=dataset, model=model_dir.name, error=message)
        test_logger.error("UI:ERR:test: Preparation failed: %s", message)
        raise HTTPException(status_code=400, detail=message) from exc
    input_type = rotation_result.input_type
    run_images_dir = rotation_result.run_images_dir
    tiles_dir = rotation_result.tiles_dir
    tif_src = rotation_result.tif_src

    row_alignment_report: Optional[dict[str, Any]] = None
    if align_images_to_rows:
        if input_type != "images" or Path(run_images_dir).name != "rotated_images":
            raise HTTPException(
                status_code=400,
                detail="Rows alignment requires successfully prepared north-up images.",
            )
        if not camera_meta or rows_source_path is None:
            raise HTTPException(
                status_code=400,
                detail="Rows alignment requires readable GPS, orientation, and GSD metadata.",
            )
        test_logger.info("UI:INFO:test: Aligning prepared images to solar rows…")

        def _alignment_progress(done: int, total: int, image_name: str) -> None:
            interval = max(1, total // 10)
            if done == 1 or done == total or done % interval == 0:
                test_logger.info(
                    "UI:INFO:test: Solar-row alignment progress: %s/%s (%s)",
                    done, total, image_name,
                )

        try:
            row_alignment_report = await asyncio.to_thread(
                align_rotated_images_to_rows,
                images_dir=Path(run_images_dir),
                camera_meta=camera_meta,
                rows_geojson=rows_source_path,
                report_path=out_root / "row_alignment.json",
                source={"job_id": row_alignment_job_id, "path": row_alignment_path},
                options=RowAlignmentOptions(
                    maximum_position_correction_m=float(row_alignment_max_position_m),
                    maximum_rotation_correction_deg=float(row_alignment_max_rotation_deg),
                ),
                progress=_alignment_progress,
            )
        except Exception as exc:
            _write_result_status(out_root, "failed", dataset=dataset, model=model_dir.name, error=str(exc))
            test_logger.error("UI:ERR:test: Solar-row alignment failed: %s", exc)
            raise HTTPException(status_code=400, detail=f"Solar-row alignment failed: {exc}") from exc
        (out_root / "camera_meta.json").write_text(json.dumps(camera_meta, indent=2), encoding="utf-8")
        alignment_counts = row_alignment_report.get("counts") or {}
        aligned_count = int(alignment_counts.get("aligned", 0))
        total_alignment_images = int(row_alignment_report.get("image_count", 0))
        test_logger.info(
            "UI:OK:test: Solar-row alignment complete: %s aligned, %s retained original metadata, %s total.",
            aligned_count,
            max(0, total_alignment_images - aligned_count),
            total_alignment_images,
        )
    prepared_count = sum(
        1 for path in run_images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ) if run_images_dir and run_images_dir.is_dir() else 0
    test_logger.info("UI:OK:test: Prepared %s inference images.", prepared_count)

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
        test_logger.info("UI:OK:test: Preparation complete. Starting model inference…")
        presp = await asyncio.to_thread(_do_predict)  # <-- offload
    except Exception as e:
        _write_result_status(out_root, "failed", dataset=dataset, model=model_dir.name, error=str(e))
        test_logger.error("UI:ERR:test: Inference failed: %s", e)
        logger.exception("Inference failed.")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    preds_dir = Path(presp["results_dir"])
    manifest_path = out_root / "manifest.json"
    class_names = (_read_model_meta(model_dir).get("class_names") or [])
    test_logger.info("UI:INFO:test: Inference complete. Preparing prediction manifest…")
    # Overlays are generated during inference, no post-processing needed
    logger.info(f"UI:INFO:post: Overlays were generated during inference")
    # gj, _ = _preds_to_geojson(ds_dir, preds_dir, out_root, class_names)
    try:
        th_num = float(test_threshold) if str(test_threshold).strip() else 0.0
    except Exception:
        th_num = 0.0

    # Build EXIF(GPS) + image-size indices once, then merge into manifest.json
    test_logger.info("UI:INFO:test: Preparing prediction manifest and image metadata…")
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
                    overlay_url = _media_url(overlay_png)
                    thumb_path = thumbs_dir / f"{stem}.png"
                    thumb_url = (
                        _media_url(thumb_path)
                        if thumb_path.exists()
                        else overlay_url
                    )
                    manifest_obj[orig_name] = {
                        "overlay": overlay_url,
                        "thumb": thumb_url,
                    }

    preprocessing_records: Dict[str, Dict[str, Any]] = {}
    preprocessing_path = out_root / "preprocessing.json"
    if preprocessing_path.is_file():
        try:
            preprocessing_payload = json.loads(preprocessing_path.read_text(encoding="utf-8"))
            raw_records = preprocessing_payload.get("images", {})
            if isinstance(raw_records, dict):
                preprocessing_records = {
                    str(name): value for name, value in raw_records.items() if isinstance(value, dict)
                }
        except (OSError, json.JSONDecodeError, TypeError):
            logger.warning("Could not read lens-correction metadata from %s", preprocessing_path)

    prepared_images_by_stem = {
        path.stem: path
        for path in Path(run_images_dir).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    } if Path(run_images_dir).is_dir() else {}

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

            stem = Path(fname).stem
            prepared_image = prepared_images_by_stem.get(stem)
            if prepared_image is not None:
                # Exact post-correction/post-rotation file passed to inference.
                # The Results and Map overlays are rendered on this same image.
                entry["prepared_image"] = _media_url(prepared_image)
                entry["display_source"] = "prepared_inference_image"
            correction_record = preprocessing_records.get(fname)
            if correction_record is None:
                correction_record = next(
                    (value for name, value in preprocessing_records.items() if Path(name).stem == stem),
                    None,
                )
            if correction_record:
                entry["lens_correction_status"] = correction_record.get("status", "unknown")
                entry["lens_displacement_px"] = correction_record.get("maximum_displacement_px")
            
            # add detection count (n) from predictions
            try:
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
    test_logger.info("UI:INFO:test: Generating prediction and image GeoJSON…")
    session_dir = out_root
    
    if input_type == "tif":
        anom_gj, imgs_gj = _build_anomalies_geojson_from_tiles(
            tiles_dir=tiles_dir,
            preds_dir=preds_dir,
            tif_path=tif_src,
            out_session=session_dir,
            class_names=class_names,
            score_thresh=th_num,
        )
        if mosaic_enabled and (out_root / "rotated_images").is_dir():
            imgs_gj = _append_mosaic_source_images_geojson(out_root, camera_meta)
            source_image_count = sum(
                1
                for feature in json.loads(Path(imgs_gj).read_text(encoding="utf-8")).get("features", [])
                if feature.get("properties", {}).get("source_role") == "mosaic_input"
            )
            test_logger.info(
                "UI:OK:test: Added %s prepared source image(s) to the Map catalog; "
                "inference was performed only on mosaic tiles.",
                source_image_count,
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
                "overlay": _media_url(overlay_png),
                "thumb":   _media_url(thumb_png),
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

    test_logger.info("UI:INFO:test: GeoJSON outputs ready. Finalizing result metadata…")

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
        metrics["inference_source"] = source_mode
        metrics["mosaic_created"] = mosaic_enabled
        metrics["target_surface_height_m"] = float(target_surface_height_m)
        metrics["row_alignment_enabled"] = bool(align_images_to_rows)
        if row_alignment_report is not None:
            metrics["row_alignment_counts"] = row_alignment_report.get("counts") or {}
            metrics["row_alignment_source"] = row_alignment_report.get("source") or {}
        if tif_src is not None:
            metrics.setdefault("source_tifs", [])
            if str(tif_src) not in metrics["source_tifs"]:
                metrics["source_tifs"].append(str(tif_src))
        correction_checked = bool((out_root / "preprocessing.json").is_file())
        metrics["lens_correction_checked"] = correction_checked
        metrics["undistort_thermal"] = correction_checked  # backward compatibility
        metrics["undistorted_images_exported"] = bool((out_root / "undistorted_images").is_dir())
        mpath.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    except Exception as e:
        logging.getLogger("pvrt").warning(f"metrics.json update failed: {e}")

    test_logger.info("UI:INFO:test: Finalizing result metadata…")
    preprocessing_path = out_root / "preprocessing.json"
    _write_result_status(
        out_root,
        "complete",
        dataset=dataset,
        model=model_dir.name,
        lens_correction_checked=bool(preprocessing_path.is_file()),
        undistort_thermal=bool(preprocessing_path.is_file()),
        undistorted_images_exported=bool((out_root / "undistorted_images").is_dir()),
        inference_source=source_mode,
        mosaic_created=mosaic_enabled,
        target_surface_height_m=float(target_surface_height_m),
        row_alignment_enabled=bool(align_images_to_rows),
        row_alignment_counts=(row_alignment_report or {}).get("counts") or {},
    )
    logger.info(f"UI:OK:test: Test complete. results={preds_dir}")
    return {
        "ok": True,
        "session": session,
        "geojson": str(anom_gj),  # backward-compat
        "predictions_geojson": _media_url(anom_gj),
        "anomalies_geojson": _media_url(anom_gj),
        "images_geojson":    _media_url(imgs_gj),
        "results_dir": str(preds_dir),
        "overlays": _media_url(ov_dir),
        "thumbs":   _media_url(th_dir),
        "manifest": manifest_items,
        "lens_correction_checked": bool(preprocessing_path.is_file()),
        "undistort_thermal": bool(preprocessing_path.is_file()),
        "undistorted_images_exported": bool((out_root / "undistorted_images").is_dir()),
        "undistorted_images": (
            _media_url(out_root / "undistorted_images")
            if (out_root / "undistorted_images").is_dir()
            else None
        ),
        "preprocessing": _media_url(preprocessing_path) if preprocessing_path.is_file() else None,
        "inference_source": source_mode,
        "mosaic_created": mosaic_enabled,
        "target_surface_height_m": float(target_surface_height_m),
        "row_alignment_enabled": bool(align_images_to_rows),
        "row_alignment": (
            {
                "source": row_alignment_report.get("source") or {},
                "counts": row_alignment_report.get("counts") or {},
                "image_count": row_alignment_report.get("image_count", 0),
                "row_line_count": row_alignment_report.get("row_line_count", 0),
            }
            if row_alignment_report is not None else None
        ),
        "assets": assets,
        "backend": presp.get("used_backend"),
        "model_mode": presp.get("model_mode"),
        "used_thermal": bool(presp.get("used_thermal")),
        "used_channel_count": int(presp.get("used_channel_count") or 0),
        "final_mode": presp.get("final_mode"),
        "media_root": _media_url(out_root),
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


def _postprocess_jobs_dir() -> Path:
    root = get_project_sessions_dir() / ".postprocess_jobs"
    root.mkdir(parents=True, exist_ok=True)
    return root


_POSTPROCESS_WORKSPACE_RE = re.compile(r"^ppjob__(.+)__(segmentation|anomaly)$")


def _postprocess_workspace_id(job_id: str, kind: str) -> str:
    return f"ppjob__{job_id}__{kind}"


def _resolve_postprocess_workspace(result_id: str) -> Path | None:
    match = _POSTPROCESS_WORKSPACE_RE.fullmatch(str(result_id or ""))
    if not match:
        return None
    job_id, kind = match.groups()
    job_dir = _project_child(_postprocess_jobs_dir(), job_id, "Post-processing job")
    workspace = (job_dir / "snapshots" / kind).resolve()
    expected_parent = (job_dir / "snapshots").resolve()
    if workspace.parent != expected_parent or not workspace.is_dir():
        raise HTTPException(status_code=404, detail="Post-processing job snapshot not found.")
    return workspace


def _session_output_dir(result_id: str) -> Path:
    workspace = _resolve_postprocess_workspace(result_id)
    if workspace is not None:
        return workspace
    return _project_child(get_project_sessions_dir(), result_id, "Test result")


def _session_asset_dir(result_id: str) -> Path:
    workspace = _resolve_postprocess_workspace(result_id)
    if workspace is None:
        return _project_child(get_project_sessions_dir(), result_id, "Test result")
    try:
        snapshot = json.loads((workspace / "snapshot.json").read_text(encoding="utf-8"))
        original_result_id = str(snapshot.get("original_result_id") or "")
    except (OSError, json.JSONDecodeError, AttributeError) as exc:
        raise HTTPException(status_code=409, detail="Job snapshot dependency metadata is unavailable.") from exc
    return _project_child(get_project_sessions_dir(), original_result_id, "Referenced test result")


def _copy_postprocess_snapshot(source_result: Path, source_geojson: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp_{uuid.uuid4().hex[:8]}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        shutil.copy2(source_geojson, temporary / "source.geojson")
        _atomic_json(temporary / "snapshot.json", {
            "original_result_id": source_result.name,
            "original_source_path": source_geojson.relative_to(source_result).as_posix(),
        })
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _job_source_metadata(
    job_id: str,
    kind: str,
    original_result_id: str,
    original_path: str,
    fingerprint: dict[str, int],
    summary: dict[str, Any],
) -> dict[str, Any]:
    workspace_id = _postprocess_workspace_id(job_id, kind)
    workspace = _resolve_postprocess_workspace(workspace_id)
    source = workspace / "source.geojson"
    stat = source.stat()
    return {
        "result_id": original_result_id,
        "path": original_path,
        "workspace_result_id": workspace_id,
        "workspace_path": "source.geojson",
        "workspace_url": _media_url(source),
        "workspace_mtime": stat.st_mtime_ns,
        "fingerprint": fingerprint,
        "summary": summary,
    }


def _resolve_postprocess_job_source(result_id: str, relative_path: str) -> tuple[Path, Path, dict[str, int]]:
    result_dir = _project_child(get_project_sessions_dir(), result_id, "Test result")
    source = (result_dir / str(relative_path or "").strip().replace("\\", "/")).resolve()
    try:
        source.relative_to(result_dir.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Source GeoJSON must belong to the selected test result.") from exc
    if source.suffix.lower() != ".geojson" or not source.is_file():
        raise HTTPException(status_code=404, detail="Selected source GeoJSON was not found.")
    stat = source.stat()
    return result_dir, source, {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _read_postprocess_job(job_id: str) -> tuple[Path, dict[str, Any]]:
    directory = _project_child(_postprocess_jobs_dir(), job_id, "Post-processing job")
    try:
        return directory, json.loads((directory / "job.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=404, detail="Post-processing job not found.") from exc


@app.get("/api/postprocess-jobs")
async def api_postprocess_jobs():
    jobs = []
    root = _postprocess_jobs_dir()
    for directory in root.iterdir():
        if not directory.is_dir():
            continue
        metadata_path = directory / "job.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        metadata["id"] = directory.name
        jobs.append(metadata)
    jobs.sort(key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""), reverse=True)
    return {"ok": True, "jobs": jobs}


@app.post("/api/postprocess-jobs")
async def api_create_postprocess_job(request: Request):
    payload = await request.json()
    name = str(payload.get("name") or "Post-processing job").strip()[:128] or "Post-processing job"
    selected_sources: dict[str, Any] = {}
    for kind in ("segmentation", "anomaly"):
        result_id = str(payload.get(f"{kind}_result_id") or "").strip()
        source_path = str(payload.get(f"{kind}_path") or "").strip()
        if not result_id or not source_path:
            raise HTTPException(status_code=400, detail=f"Select the {kind} test result and GeoJSON before creating a job.")
        result_dir, source, fingerprint = _resolve_postprocess_job_source(result_id, source_path)
        try:
            summary = await asyncio.to_thread(analyze_geojson, source, result_dir)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        selected_sources[kind] = {
            "result_id": result_id,
            "path": source_path,
            "result_dir": result_dir,
            "source": source,
            "fingerprint": fingerprint,
            "summary": summary,
        }
    existing_names: set[str] = set()
    for directory in _postprocess_jobs_dir().iterdir():
        try:
            metadata = json.loads((directory / "job.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        existing_name = str(metadata.get("name") or "").strip()
        if existing_name:
            existing_names.add(existing_name.casefold())
    if name.casefold() in existing_names:
        position = 2
        base_name = name
        while f"{base_name} ({position})".casefold() in existing_names:
            position += 1
        name = f"{base_name} ({position})"
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")[:60] or "postprocess"
    job_id = f"{safe_name}_{uuid.uuid4().hex[:8]}"
    now = datetime.now().isoformat()
    directory = _postprocess_jobs_dir() / job_id
    directory.mkdir(parents=True, exist_ok=False)
    try:
        configured_sources: dict[str, Any] = {}
        for kind, selected in selected_sources.items():
            await asyncio.to_thread(
                _copy_postprocess_snapshot,
                selected["result_dir"],
                selected["source"],
                directory / "snapshots" / kind,
            )
            configured_sources[kind] = _job_source_metadata(
                job_id,
                kind,
                selected["result_id"],
                selected["path"],
                selected["fingerprint"],
                selected["summary"],
            )
        metadata = {
            "id": job_id,
            "name": name,
            "created_at": now,
            "updated_at": now,
            "sources": configured_sources,
            "workflows": {},
        }
        _atomic_json(directory / "job.json", metadata)
    except Exception:
        if directory.exists():
            shutil.rmtree(directory)
        raise
    return {"ok": True, "job": metadata}


@app.patch("/api/postprocess-jobs/{job_id}")
async def api_rename_postprocess_job(job_id: str, request: Request):
    directory = _project_child(_postprocess_jobs_dir(), job_id, "Post-processing job")
    metadata_path = directory / "job.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=404, detail="Post-processing job not found.") from exc
    payload = await request.json()
    name = str(payload.get("name") or "").strip()[:128]
    if not name:
        raise HTTPException(status_code=400, detail="Job name cannot be empty.")
    metadata["name"] = name
    metadata["updated_at"] = datetime.now().isoformat()
    _atomic_json(metadata_path, metadata)
    return {"ok": True, "job": {**metadata, "id": directory.name}}


@app.get("/api/postprocess-jobs/{job_id}/config")
async def api_postprocess_job_config(job_id: str):
    directory, metadata = _read_postprocess_job(job_id)
    metadata["id"] = directory.name
    return {"ok": True, "job": metadata}


@app.put("/api/postprocess-jobs/{job_id}/config")
async def api_update_postprocess_job_config(job_id: str, request: Request):
    directory, metadata = _read_postprocess_job(job_id)
    payload = await request.json()
    if not payload.get("confirm_reset"):
        raise HTTPException(status_code=400, detail="Confirm reset before changing a job source.")
    next_sources: dict[str, Any] = {}
    changed: set[str] = set()
    previous_sources = metadata.get("sources") or {}
    for kind in ("segmentation", "anomaly"):
        result_id = str(payload.get(f"{kind}_result_id") or "").strip()
        source_path = str(payload.get(f"{kind}_path") or "").strip()
        if not result_id or not source_path:
            raise HTTPException(status_code=400, detail=f"Select the {kind} test result and GeoJSON before saving the configuration.")
        result_dir, source, fingerprint = _resolve_postprocess_job_source(result_id, source_path)
        try:
            summary = await asyncio.to_thread(analyze_geojson, source, result_dir)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        previous = previous_sources.get(kind) or {}
        if previous.get("result_id") != result_id or previous.get("path") != source_path:
            changed.add(kind)
        if kind in changed or not previous.get("workspace_result_id"):
            await asyncio.to_thread(
                _copy_postprocess_snapshot,
                result_dir,
                source,
                directory / "snapshots" / kind,
            )
            next_sources[kind] = _job_source_metadata(
                job_id, kind, result_id, source_path, fingerprint, summary,
            )
        else:
            next_sources[kind] = previous
            next_sources[kind]["summary"] = summary
            next_sources[kind]["fingerprint"] = fingerprint
    workflows = dict(metadata.get("workflows") or {})
    for kind in changed:
        workflows.pop(kind, None)
    if "segmentation" in changed:
        workflows.pop("anomaly", None)
        anomaly_workflow_root = directory / "snapshots" / "anomaly" / "postprocess"
        if anomaly_workflow_root.is_dir():
            shutil.rmtree(anomaly_workflow_root)
    metadata["sources"] = next_sources
    metadata["workflows"] = workflows
    metadata["updated_at"] = datetime.now().isoformat()
    _atomic_json(directory / "job.json", metadata)
    return {"ok": True, "job": {**metadata, "id": directory.name}, "changed_sources": sorted(changed)}


@app.put("/api/postprocess-jobs/{job_id}/workflow")
async def api_bind_postprocess_job_workflow(job_id: str, request: Request):
    directory, metadata = _read_postprocess_job(job_id)
    payload = await request.json()
    kind = str(payload.get("kind") or "").strip()
    workflow_id = str(payload.get("workflow_id") or "").strip()
    if kind not in {"segmentation", "anomaly"}:
        raise HTTPException(status_code=400, detail="Workflow kind must be segmentation or anomaly.")
    workflows = dict(metadata.get("workflows") or {})
    if not workflow_id:
        workflows.pop(kind, None)
    else:
        source = (metadata.get("sources") or {}).get(kind) or {}
        workspace_id = str(source.get("workspace_result_id") or "")
        workspace = _resolve_postprocess_workspace(workspace_id)
        workflow_dir = (workspace / "postprocess" / re.sub(r"[^A-Za-z0-9._-]+", "_", workflow_id)).resolve()
        if workflow_dir.parent != (workspace / "postprocess").resolve() or not (workflow_dir / "status.json").is_file():
            raise HTTPException(status_code=404, detail="The selected job workflow was not found.")
        workflows[kind] = {"result_id": workspace_id, "workflow_id": workflow_dir.name}
    metadata["workflows"] = workflows
    metadata["updated_at"] = datetime.now().isoformat()
    _atomic_json(directory / "job.json", metadata)
    return {"ok": True, "job": {**metadata, "id": directory.name}}


@app.delete("/api/postprocess-jobs/{job_id}")
async def api_delete_postprocess_job(job_id: str):
    directory = _project_child(_postprocess_jobs_dir(), job_id, "Post-processing job")
    if not directory.is_dir():
        raise HTTPException(status_code=404, detail="Post-processing job not found.")
    shutil.rmtree(directory)
    return {"ok": True, "id": directory.name}


@app.post("/api/results/{session_id}/rename")
async def api_rename_result(session_id: str, name: str = Form(...)):
    display_name = str(name or "").strip()[:128]
    if not display_name:
        raise HTTPException(status_code=400, detail="Result name cannot be empty.")
    session_dir = _project_child(get_project_sessions_dir(), session_id, "Result")
    _write_asset_display_name(session_dir, ".result_meta.json", display_name)
    logger.info("UI:OK:test: Renamed result %s to %s", session_dir.name, display_name)
    return {"ok": True, "id": session_dir.name, "name": session_dir.name, "display_name": display_name}


@app.delete("/api/results/{session_id}")
async def api_delete_result(session_id: str):
    session_dir = _project_child(get_project_sessions_dir(), session_id, "Result")
    dependent_jobs = []
    for job_dir in _postprocess_jobs_dir().iterdir():
        if not job_dir.is_dir():
            continue
        try:
            metadata = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, AttributeError):
            continue
        kinds = [
            kind for kind, source in (metadata.get("sources") or {}).items()
            if str((source or {}).get("result_id") or "") == session_id
        ]
        if kinds:
            dependent_jobs.append(f"{metadata.get('name') or job_dir.name} ({', '.join(kinds)})")
    if dependent_jobs:
        examples = ", ".join(dependent_jobs[:5])
        remainder = len(dependent_jobs) - 5
        if remainder > 0:
            examples += f", and {remainder} more"
        raise HTTPException(
            status_code=409,
            detail=f"This test result is used by post-processing jobs: {examples}. Delete or reconfigure those jobs first.",
        )
    try:
        shutil.rmtree(session_dir)
    except OSError as exc:
        logger.exception("Failed to delete result %s", session_dir.name)
        raise HTTPException(status_code=500, detail=f"Failed to delete result: {exc}") from exc
    logger.info("UI:OK:test: Deleted result %s", session_dir.name)
    return {"ok": True, "id": session_dir.name}


@app.get("/api/session_summary")
async def api_session_summary(session: str):
    ses = _session_asset_dir(session)
    asset_session_id = ses.name

    gj = ses / "predictions.geojson"
    if not gj.exists():
        gj = ses / "anomalies.geojson"  # legacy session compatibility
    imgs_gj = ses / "images.geojson"    # NEW

    manifest_path = ses / "manifest.json"
    manifest = []

    def _normalize_session_asset_url(value: Any) -> Any:
        if not isinstance(value, str) or not value:
            return value
        try:
            parsed = urlparse(value)
            path = parsed.path or value
        except Exception:
            path = value

        marker = f"/{asset_session_id}/"
        if marker in path:
            rel = path.split(marker, 1)[1]
            if rel and any(rel.startswith(prefix) for prefix in ("overlays/", "thumbs/", "images/", "rotated_images/", "predictions.geojson", "anomalies.geojson", "images.geojson")):
                target = ses / rel
                if target.exists():
                    return _media_url(target)
        return value

    if manifest_path.exists():
        try:
            manifest_obj = json.loads(manifest_path.read_text())
            # Convert manifest object to array with file field
            manifest = []
            for fname, entry in manifest_obj.items():
                if isinstance(entry, dict):
                    normalized_entry = dict(entry)
                    if "overlay" in normalized_entry:
                        normalized_entry["overlay"] = _normalize_session_asset_url(normalized_entry.get("overlay"))
                    if "thumb" in normalized_entry:
                        normalized_entry["thumb"] = _normalize_session_asset_url(normalized_entry.get("thumb"))
                    manifest.append({"file": fname, **normalized_entry})
                else:
                    manifest.append({"file": fname})
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
        "geojson_url": _media_url(gj) if gj.exists() else None,
        "predictions_geojson": _media_url(gj) if gj.exists() else None,
        "anomalies_geojson": _media_url(gj) if gj.exists() else None,
        # NEW: where image footprints live (if you created them)
        "images_geojson_url": _media_url(imgs_gj) if imgs_gj.exists() else None,
        "assets": assets,
        "manifest": manifest,   # still the parsed JSON (not a path)
        "tiler": "ok" if RIO_OK else "unavailable",
        # Helpful flags for the frontend
        "rotated_images_available": bool(rotated_images_available),
        "camera_meta": camera_meta,
    }




@app.get("/api/results/{session}/metrics")
def api_metrics(session: str):
    p = get_project_sessions_dir() / session / "metrics.json"
    if not p.exists():
        raise HTTPException(404, "metrics.json not found")
    return JSONResponse(json.loads(p.read_text(encoding="utf-8")))

@app.get("/api/runs/{run_name}/meta")
def api_model_meta(run_name: str):
    p = get_project_output_dir() / run_name / "model_meta.json"
    if not p.exists():
        raise HTTPException(404, "model_meta.json not found")
    return JSONResponse(json.loads(p.read_text(encoding="utf-8")))


# -------------- Simple dynamic tiler for TIFF (XYZ) --------------
_TILER_INDEX: Dict[str, List[Path]] = {}
_TILER_STATS: Dict[tuple, Dict[str, Any]] = {}   # per (session, idx) cached stretch + meta

def _session_tifs(session: str) -> List[Path]:
    """
    Prefer absolute source paths recorded in metrics.json under 'source_tifs'.
    Fall back to the dataset recorded in result_status.json, then to files
    retained inside the result. This supports interrupted legacy runs whose
    inference outputs exist but final metrics bookkeeping did not run.
    """
    try:
        ses_dir = _session_asset_dir(session)
    except HTTPException:
        return []
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
        try:
            status_path = ses_dir / "result_status.json"
            status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
            dataset = str(status.get("dataset") or "").strip()
            if dataset:
                dataset_dir = _project_child(get_project_test_dir(), dataset, "Test dataset")
                tifs = sorted(
                    path for path in dataset_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in (".tif", ".tiff")
                )
        except (OSError, json.JSONDecodeError, HTTPException):
            tifs = []
    if not tifs:
        tifs = [p for p in (ses_dir / "images").glob("*") if p.suffix.lower() in (".tif", ".tiff")]
    if not tifs:
        tifs = [p for p in ses_dir.glob("*") if p.suffix.lower() in (".tif", ".tiff")]
    return tifs

def _stitch_tiles_to_tiff(tifs: List[Path], output_path: Path) -> bool:
    """
    Stitch multiple GeoTIFF tiles into a single GeoTIFF.
    Returns True if successful.
    """
    if not RIO_OK or not tifs:
        return False
    
    try:
        import rasterio
        from rasterio.merge import merge
        from rasterio.io import MemoryFile
        
        # Open all source files
        sources = []
        try:
            for tif_path in tifs:
                if tif_path.exists():
                    sources.append(rasterio.open(tif_path))
            
            if not sources:
                logger.warning("No valid source files found for stitching")
                return False
            
            # Merge all tiles
            merged_data, merged_transform = merge(sources)
            
            # Get CRS from first source
            crs = sources[0].crs
            
            # Write merged result
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=merged_data.shape[1],
                width=merged_data.shape[2],
                count=merged_data.shape[0],
                dtype=merged_data.dtype,
                crs=crs,
                transform=merged_transform,
                compress='lzw'
            ) as dest:
                dest.write(merged_data)
            
            logger.info(f"Successfully stitched {len(tifs)} tiles to {output_path}")
            return True
        finally:
            # Close all source files
            for src in sources:
                if src:
                    src.close()
    except Exception as e:
        logger.error(f"Failed to stitch tiles: {e}", exc_info=True)
        return False


def _build_tile_layer_defs(tile_key: str, tifs: List[Path]) -> List[Dict[str, Any]]:
    """
    Register GeoTIFF sources under tile_key and return Leaflet-ready layer descriptors.
    """
    if not RIO_OK:
        return []
    _TILER_INDEX[tile_key] = tifs
    layers: List[Dict[str, Any]] = []
    for i, p in enumerate(tifs):
        try:
            with rasterio.open(p) as ds:
                # bounds in WGS84 for fitBounds + attribution data
                try:
                    left, bottom, right, top = rasterio.warp.transform_bounds(
                        ds.crs, CRS.from_epsg(4326), *ds.bounds, densify_pts=21
                    )
                    # Validate bounds are finite numbers
                    if not all(math.isfinite(x) for x in [left, bottom, right, top]):
                        raise ValueError("Invalid bounds detected")
                    # Ensure bounds are in valid range
                    left = max(-180.0, min(180.0, left))
                    right = max(-180.0, min(180.0, right))
                    bottom = max(-90.0, min(90.0, bottom))
                    top = max(-90.0, min(90.0, top))
                except Exception as e:
                    logger.warning(f"Failed to transform bounds for {p}: {e}, using defaults")
                    left, bottom, right, top = (-180.0, -85.0, 180.0, 85.0)
                layers.append({
                    "name": p.name,
                    "template": f"/tiles/{tile_key}/{i}" + "/{z}/{x}/{y}.png",
                    "minzoom": 0,
                    "maxzoom": 22,
                    # [[south, west], [north, east]]
                    "bounds": [[bottom, left], [top, right]],
                })
        except Exception as e:
            logger.warning(f"Tiler: failed to inspect '{p}': {e}")
    return layers

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
    layers = _build_tile_layer_defs(session, tifs)

    return {"ok": True, "session": session, "layers": layers}


@app.get("/api/list_overlays")
def api_list_overlays():
    """
    List all saved overlay directories and their metadata.
    Returns GeoJSON and GeoTIFF overlays from project overlays directory.
    """
    overlays_dir = get_project_overlays_dir()
    if not overlays_dir.exists():
        return {"ok": True, "overlays": []}
    
    result = []
    for overlay_dir in overlays_dir.iterdir():
        if not overlay_dir.is_dir():
            continue
        overlay_display_name = None
        overlay_metadata = {}
        try:
            metadata_path = overlay_dir / ".overlay_meta.json"
            if metadata_path.is_file():
                overlay_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                overlay_display_name = str(overlay_metadata.get("display_name") or "").strip() or None
        except (OSError, json.JSONDecodeError, AttributeError):
            overlay_display_name = None
            overlay_metadata = {}

        if overlay_metadata.get("reference_kind") == "postprocess":
            try:
                source_result = str(overlay_metadata.get("source_result") or "").strip()
                workflow_id = str(overlay_metadata.get("workflow_id") or "").strip()
                stage = str(overlay_metadata.get("stage") or "").strip()
                result_dir = _session_output_dir(source_result).resolve()
                workflow_dir = (result_dir / "postprocess" / workflow_id).resolve()
                if workflow_dir.parent != (result_dir / "postprocess").resolve():
                    raise ValueError("Reference escapes the active project.")
                status = json.loads((workflow_dir / "status.json").read_text(encoding="utf-8"))
                output = (status.get("outputs") or {}).get(stage) or {}
                source_path = (result_dir / str(output.get("path") or "")).resolve()
                source_path.relative_to(workflow_dir)
                if source_path.suffix.lower() != ".geojson" or not source_path.is_file():
                    raise FileNotFoundError(source_path)
                stage_label = "Rows" if stage == "solar_rows" else stage.title()
                workflow_name = status.get("display_name") or status.get("parameters", {}).get("output_name") or workflow_id
                reference_name = f"{str(workflow_name).strip()} — {stage_label}"
                result.append({
                    "type": "geojson",
                    "name": reference_name,
                    "overlay_id": overlay_dir.name,
                    "file": source_path.name,
                    "path": _media_url(source_path),
                    "reference": True,
                })
            except (OSError, ValueError, json.JSONDecodeError, AttributeError):
                logger.warning("Skipping broken post-process Map reference: %s", overlay_dir.name)
            continue
        
        # Look for GeoJSON files
        geojson_files = list(overlay_dir.glob("*.geojson")) + list(overlay_dir.glob("*.json"))
        for gj_file in geojson_files:
            result.append({
                "type": "geojson",
                "name": overlay_display_name or gj_file.stem,
                "overlay_id": overlay_dir.name,
                "file": gj_file.name,
                "path": _media_url(gj_file)
            })
        
        # Look for GeoTIFF files
        tif_files = list(overlay_dir.glob("*.tif")) + list(overlay_dir.glob("*.tiff"))
        for tif_file in tif_files:
            result.append({
                "type": "tif",
                "name": tif_file.stem,
                "overlay_id": overlay_dir.name,
                "file": tif_file.name,
            })
    
    return {"ok": True, "overlays": result}


@app.post("/api/upload_geojson_overlay")
async def api_upload_geojson_overlay(
    file: UploadFile = File(...),
    name: str | None = Form(None),
):
    """
    Upload a GeoJSON file and save it under project overlays/<overlay_id>/.
    """
    if not file or not file.filename:
        raise HTTPException(400, "No file uploaded")

    ext = Path(file.filename).suffix.lower()
    if ext not in (".geojson", ".json"):
        raise HTTPException(400, "Only .geojson/.json files are supported")

    safe = _safe_name(name) or _safe_name(Path(file.filename).stem) or "overlay"
    overlay_id = f"overlay-{safe}-{_now_stamp()}-{uuid.uuid4().hex[:6]}"
    overlay_dir = get_project_overlays_dir() / overlay_id
    overlay_dir.mkdir(parents=True, exist_ok=True)

    dest_name = f"{safe}.geojson"
    dest_path = overlay_dir / dest_name

    content = await file.read()
    dest_path.write_bytes(content)

    return {
        "ok": True,
        "overlay_id": overlay_id,
        "path": _media_url(dest_path),
    }


@app.post("/api/delete_overlay")
async def api_delete_overlay(request: Request):
    """
    Delete a saved overlay directory.
    """
    try:
        body = await request.json()
        overlay_id = body.get("overlay_id")
        
        if not overlay_id:
            logger.warning("Delete overlay request missing overlay_id")
            raise HTTPException(400, "overlay_id required")
        
        logger.info("Attempting to delete overlay: %s", overlay_id)
        
        overlay_dir = get_project_overlays_dir() / overlay_id
        if not overlay_dir.exists():
            logger.warning("Overlay directory not found: %s", overlay_dir)
            return {"ok": False, "error": "Overlay not found"}
        
        shutil.rmtree(overlay_dir)
        logger.info("Successfully deleted overlay: %s", overlay_id)
        return {"ok": True, "overlay_id": overlay_id}
    except Exception as e:
        logger.error("Failed to delete overlay: %s", e, exc_info=True)
        raise HTTPException(500, f"Failed to delete overlay: {e}")


@app.get("/api/download_overlay")
async def api_download_overlay(overlay_id: str):
    """
    Download an overlay TIF file.
    """
    try:
        overlay_dir = get_project_overlays_dir() / overlay_id
        if not overlay_dir.exists():
            raise HTTPException(404, "Overlay not found")
        
        # Find the TIF file in the overlay directory
        tif_files = list(overlay_dir.glob("*.tif")) + list(overlay_dir.glob("*.tiff"))
        if not tif_files:
            raise HTTPException(404, "No TIF file found in overlay")
        
        tif_path = tif_files[0]
        
        return Response(
            content=open(tif_path, "rb").read(),
            media_type="image/tiff",
            headers={"Content-Disposition": f"attachment; filename={tif_path.name}"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to download overlay: %s", e, exc_info=True)
        raise HTTPException(500, f"Failed to download overlay: {e}")


@app.get("/api/download_ortho")
async def api_download_ortho(session: str):
    """
    Download session orthophoto as GeoTIFF.
    If multiple tiles exist, stitch them together.
    """
    try:
        if not RIO_OK:
            raise HTTPException(400, "rasterio not available for stitching")
        
        session_dir = get_project_sessions_dir() / session
        if not session_dir.exists():
            raise HTTPException(404, "Session not found")
        
        # Look for GeoTIFF files in session directory
        tif_files = list(session_dir.glob("*.tif")) + list(session_dir.glob("*.tiff"))
        if not tif_files:
            raise HTTPException(404, "No GeoTIFF found in session")
        
        # If single file, serve it directly
        if len(tif_files) == 1:
            tif_path = tif_files[0]
            return Response(
                content=open(tif_path, "rb").read(),
                media_type="image/tiff",
                headers={"Content-Disposition": f"attachment; filename={tif_path.name}"}
            )
        
        # Multiple files: need to stitch them
        # Get tile layer defs to reconstruct bounds
        layers = _build_tile_layer_defs(session, tif_files)
        if not layers:
            raise HTTPException(400, "Could not process tiles for stitching")
        
        # Stitch the tiles into a single GeoTIFF
        output_path = session_dir / f"{session}_orthophoto.tif"
        stitched = _stitch_tiles_to_tiff(tif_files, output_path)
        
        if not stitched or not output_path.exists():
            raise HTTPException(500, "Failed to stitch tiles")
        
        return Response(
            content=open(output_path, "rb").read(),
            media_type="image/tiff",
            headers={"Content-Disposition": f"attachment; filename={output_path.name}"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to download orthophoto: %s", e, exc_info=True)
        raise HTTPException(500, f"Failed to download orthophoto: {e}")


@app.post("/api/upload_tif_overlay")
async def api_upload_tif_overlay(
    file: UploadFile = File(...),
    name: str | None = Form(None),
):
    """
    Upload a GeoTIFF and return tile layer descriptors for map overlays.
    Stores the file under overlays/<overlay_id>/ in active project.
    """
    if not RIO_OK:
        return {"ok": False, "error": "rasterio_not_available"}

    if not file or not file.filename:
        raise HTTPException(400, "No file uploaded")

    ext = Path(file.filename).suffix.lower()
    if ext not in (".tif", ".tiff"):
        raise HTTPException(400, "Only .tif/.tiff files are supported")

    safe = _safe_name(name) or _safe_name(Path(file.filename).stem) or "overlay"
    overlay_id = f"overlay-{safe}-{_now_stamp()}-{uuid.uuid4().hex[:6]}"
    overlay_dir = get_project_overlays_dir() / overlay_id
    overlay_dir.mkdir(parents=True, exist_ok=True)

    dest_name = f"{safe}{ext}"
    dest_path = overlay_dir / dest_name

    with open(dest_path, "wb") as out_f:
        shutil.copyfileobj(file.file, out_f)

    layers = _build_tile_layer_defs(overlay_id, [dest_path])

    return {
        "ok": True,
        "overlay_id": overlay_id,
        "tiles": {"layers": layers},
    }


@app.get("/api/get_overlay_tiles")
def api_get_overlay_tiles(overlay_id: str):
    """
    Get tile layer definitions for a saved GeoTIFF overlay.
    """
    if not RIO_OK:
        return {"ok": False, "error": "rasterio_not_available"}
    
    overlay_dir = get_project_overlays_dir() / overlay_id
    if not overlay_dir.exists():
        raise HTTPException(404, "Overlay not found")
    
    # Find all TIF files in the overlay directory
    tif_files = list(overlay_dir.glob("*.tif")) + list(overlay_dir.glob("*.tiff"))
    if not tif_files:
        raise HTTPException(404, "No GeoTIFF files found in overlay")
    
    layers = _build_tile_layer_defs(overlay_id, tif_files)
    
    return {
        "ok": True,
        "overlay_id": overlay_id,
        "tiles": {"layers": layers},
    }

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
        nodata = src.nodata
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
            # Make nodata transparent (if nodata is set), else treat pure white as transparent
            try:
                if nodata is not None:
                    nd = int(nodata)
                    nd_mask = (canvas[0] == nd) & (canvas[1] == nd) & (canvas[2] == nd)
                    acan[nd_mask] = 0
                else:
                    white_mask = (canvas[0] >= 250) & (canvas[1] >= 250) & (canvas[2] >= 250)
                    acan[white_mask] = 0
            except Exception:
                pass
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
            # Make nodata transparent for single-band
            try:
                if nodata is not None:
                    nd_mask = np.isclose(band, float(nodata))
                    if nd_mask.shape == acan[Tt:B, L:R].shape:
                        acan[Tt:B, L:R][nd_mask] = 0
            except Exception:
                pass
            rgba = np.dstack([canvas, acan])

    im = Image.fromarray(rgba, "RGBA")
    buf = BytesIO(); im.save(buf, "PNG"); buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")


# -------------- Serve media & frontend --------------
# Custom endpoint for serving project files (supports external drives)
@app.get("/api/project_file/{file_path:path}")
async def serve_project_file(file_path: str):
    """Serve files from project directories (including external drives)."""
    from urllib.parse import unquote
    
    try:
        # Decode the URL-encoded path
        decoded_path = unquote(file_path)
        file_path_obj = Path(decoded_path)
        
        # Security: ensure the path exists and is a file
        if not file_path_obj.exists():
            raise HTTPException(status_code=404, detail="File not found")
        if not file_path_obj.is_file():
            raise HTTPException(status_code=400, detail="Not a file")
        
        # Security: ensure the path belongs to a registered project
        is_valid = False
        for project in project_manager.list_projects():
            project_root = Path(project.root_path)
            try:
                file_path_obj.resolve().relative_to(project_root.resolve())
                is_valid = True
                break
            except ValueError:
                continue
        
        if not is_valid:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Determine media type
        import mimetypes
        media_type, _ = mimetypes.guess_type(str(file_path_obj))
        if media_type is None:
            media_type = "application/octet-stream"
        
        # Read and return file
        return FileResponse(
            path=str(file_path_obj),
            media_type=media_type,
            filename=file_path_obj.name
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving project file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

app.include_router(
    create_postprocess_router(get_project_sessions_dir, get_project_overlays_dir, _media_url, logger)
)
app.include_router(create_row_alignment_router(get_project_sessions_dir))

app.mount("/media", StaticFiles(directory=str(MEDIA_DIR), html=False), name="media")
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="web")
