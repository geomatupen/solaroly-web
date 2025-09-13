# backend/pvrt/web/app.py
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

from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response
from starlette.requests import Request

from . import sse as sse_mod
from .sse import LogBroker, SSELogHandler, sse_response

from ..dataops.scan_decode_split import scan_and_decode_split
from ..trainops.trainer_rgb_only import RGBOnlyTrainer
from ..trainops.trainer_rgb_thermal_tolerant import RTolerantTrainer
from ..infer.predict_rgb_thermal import predict_folder

import piexif
import warnings

# Optional image overlay utilities
import cv2
import numpy as np
from PIL import Image

# --- Optional tiler deps ---
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

app = FastAPI(title="PVRT API", version="1.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# ---------------- Logging / SSE ----------------
broker = LogBroker()
sse_handler = SSELogHandler(broker)
sse_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S"))
sse_handler.setLevel(logging.INFO)

logger = logging.getLogger("pvrt")
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.addHandler(sse_handler)

def _attach_external_loggers():
    for name in ("detectron2", "fvcore", "d2", "detectron2.data", "detectron2.utils.events", "detectron2.engine"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.INFO)
        if not any(isinstance(h, SSELogHandler) for h in lg.handlers):
            lg.addHandler(sse_handler)
        lg.propagate = False

@app.on_event("startup")
async def _on_startup():
    loop = asyncio.get_running_loop()
    sse_mod.set_event_loop(loop)
    _attach_external_loggers()
    warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release.*")
    if not RIO_OK:
        logger.info("UI:INFO:test: Raster tiler disabled (missing rasterio/pyproj/mercantile).")
    logger.info("Startup: SSE ready.")

# -------- Redirect stdout/stderr to logging for long jobs --------
class _StreamToLogger(io.TextIOBase):
    def __init__(self, level=logging.INFO):
        super().__init__()
        self.level = level
        self._buf = ""
    def write(self, buf):
        self._buf += str(buf)
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                logging.getLogger("pvrt").log(self.level, line)
    def flush(self):
        if self._buf.strip():
            logging.getLogger("pvrt").log(self.level, self._buf.strip())
            self._buf = ""

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

# ---------------- Paths ----------------
ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
VALID_DIR = DATA_DIR / "valid"
TEST_DIR  = DATA_DIR / "test"
OUTPUTS   = PROJECT_ROOT / "outputs" / "mrcnn"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
MEDIA_DIR = PROJECT_ROOT / "media"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)

def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# ---------------- Common endpoints ----------------
@app.get("/api/logs")
async def stream_logs(request: Request):
    q = await broker.subscribe()
    logger.info("Client subscribed to logs.")
    return sse_response(q)

@app.get("/api/health")
async def api_health():
    return {"ok": True}

def _read_model_meta(out_dir: Path) -> dict:
    p = out_dir / "model_meta.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}

# --------------- Cancel (best-effort) ----------------
CANCEL_FLAGS = {"train": False}

@app.post("/api/cancel")
async def api_cancel(job: str = Form(...)):
    job = job.strip().lower()
    if job == "train":
        CANCEL_FLAGS["train"] = True
        logger.info("UI:INFO:train: Cancel requested (best-effort). Training will stop at the next safe point if supported.")
        return {"ok": True, "job": "train"}
    elif job == "test":
        logger.info("UI:INFO:test: Cancel requested (client-side).")
        return {"ok": True, "job": "test"}
    else:
        raise HTTPException(status_code=400, detail="Unknown job. Use 'train' or 'test'.")

# ---------------- Training helpers ----------------
def _force_axis_aligned(cfg):
    cfg.MODEL.ANCHOR_GENERATOR.ANGLES = [[0]]
    logging.getLogger("pvrt").info("[train] Forcing axis-aligned anchors: MODEL.ANCHOR_GENERATOR.ANGLES=[[0]]")

def _train_rgb_only_with_params(run_dir: Path, max_iter: int, base_lr: float, ims_per_batch: int) -> Path:
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    from detectron2.engine import default_setup
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.utils.events import EventStorage

    from ..trainops.datasets import register_split_coco
    from ..trainops.helpers import get_num_classes

    train_dir, val_dir, out_dir = TRAIN_DIR, VALID_DIR, run_dir
    register_split_coco("pv_train", train_dir)
    register_split_coco("pv_val",   val_dir)
    ann_train = next((p for p in [
        train_dir/"_annotations.coco", train_dir/"_annotations.coco.json",
        train_dir/"train.json", train_dir/"annotations.json"
    ] if p.exists()), None)
    if ann_train is None:
        raise FileNotFoundError(f"COCO JSON not found in {train_dir}")
    num_classes = get_num_classes(ann_train)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("pv_train",)
    cfg.DATASETS.TEST  = ("pv_val",)
    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.SOLVER.IMS_PER_BATCH = int(ims_per_batch)
    cfg.SOLVER.BASE_LR = float(base_lr)
    cfg.SOLVER.MAX_ITER = int(max_iter)
    s1 = int(max_iter * 0.66); s2 = int(max_iter * 0.88)
    cfg.SOLVER.STEPS = (s1, s2)
    cfg.SOLVER.CHECKPOINT_PERIOD = max(1000, int(max_iter/4))

    cfg.INPUT.FORMAT = "RGB"
    cfg.MODEL.PIXEL_MEAN = [0.485, 0.456, 0.406]
    cfg.MODEL.PIXEL_STD  = [0.229, 0.224, 0.225]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = int(num_classes)
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    _force_axis_aligned(cfg)

    cfg.OUTPUT_DIR = str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    default_setup(cfg, {})
    with EventStorage(0):
        trainer = RGBOnlyTrainer(cfg)
        trainer.resume_or_load(resume=False)
        logging.getLogger("pvrt").info(
            f"[train] Using IMS_PER_BATCH={cfg.SOLVER.IMS_PER_BATCH}, BASE_LR={cfg.SOLVER.BASE_LR}, MAX_ITER={cfg.SOLVER.MAX_ITER}"
        )
        trainer.train()

    val_loader = RGBOnlyTrainer.build_test_loader(cfg, "pv_val")
    evaluator = COCOEvaluator("pv_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
    _ = inference_on_dataset(trainer.model, val_loader, evaluator)

    with (out_dir/"model_meta.json").open("w", encoding="utf-8") as f:
        json.dump({"input_mode": "rgb"}, f, indent=2)
    return out_dir / "model_final.pth"

def _train_rgb_thermal_with_params(run_dir: Path, max_iter: int, base_lr: float, ims_per_batch: int) -> Path:
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    from detectron2.engine import default_setup
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.utils.events import EventStorage

    from ..trainops.datasets import register_split_coco
    from ..trainops.helpers import get_num_classes

    train_dir, val_dir, out_dir = TRAIN_DIR, VALID_DIR, run_dir
    register_split_coco("pv_train", train_dir)
    register_split_coco("pv_val",   val_dir)
    ann_train = next((p for p in [
        train_dir/"_annotations.coco", train_dir/"_annotations.coco.json",
        train_dir/"train.json", train_dir/"annotations.json"
    ] if p.exists()), None)
    if ann_train is None:
        raise FileNotFoundError(f"COCO JSON not found in {train_dir}")
    num_classes = get_num_classes(ann_train)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("pv_train",)
    cfg.DATASETS.TEST  = ("pv_val",)
    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.SOLVER.IMS_PER_BATCH = int(ims_per_batch)
    cfg.SOLVER.BASE_LR = float(base_lr)
    cfg.SOLVER.MAX_ITER = int(max_iter)
    s1 = int(max_iter * 0.66); s2 = int(max_iter * 0.88)
    cfg.SOLVER.STEPS = (s1, s2)
    cfg.SOLVER.CHECKPOINT_PERIOD = max(1000, int(max_iter/4))

    cfg.INPUT.FORMAT = "RGB"
    cfg.MODEL.PIXEL_MEAN = [0.5, 0.5, 0.5, 0.5]
    cfg.MODEL.PIXEL_STD  = [0.5, 0.5, 0.5, 0.5]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = int(num_classes)
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    _force_axis_aligned(cfg)

    cfg.OUTPUT_DIR = str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    default_setup(cfg, {})
    with EventStorage(0):
        trainer = RTolerantTrainer(cfg)
        trainer.resume_or_load(resume=False)
        logging.getLogger("pvrt").info(
            f"[train] Using IMS_PER_BATCH={cfg.SOLVER.IMS_PER_BATCH}, BASE_LR={cfg.SOLVER.BASE_LR}, MAX_ITER={cfg.SOLVER.MAX_ITER}"
        )
        trainer.train()

    val_loader = RTolerantTrainer.build_test_loader(cfg, "pv_val")
    evaluator = COCOEvaluator("pv_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
    _ = inference_on_dataset(trainer.model, val_loader, evaluator)

    with (out_dir/"model_meta.json").open("w", encoding="utf-8") as f:
        json.dump({"input_mode": "rgbtherm"}, f, indent=2)
    return out_dir / "model_final.pth"

def _train_job(use_thermal: bool, max_iter: int, base_lr: float, ims_per_batch: int) -> None:
    run_dir = OUTPUTS / _now_stamp()
    run_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S"))
    logger.addHandler(file_handler)

    ext_names = ("detectron2", "fvcore", "detectron2.engine")
    ext_handlers = []
    for name in ext_names:
        lg = logging.getLogger(name)
        fh = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S"))
        lg.addHandler(fh)
        ext_handlers.append((lg, fh))

    CANCEL_FLAGS["train"] = False

    with redirect_std_to_logger():
        try:
            logger.info("UI:OK:train: Training started…")
            logger.info(f"[train] Run dir: {run_dir}")
            logger.info(f"[train] use_thermal={use_thermal} max_iter={max_iter} base_lr={base_lr} batch={ims_per_batch}")
            logger.info(f"UI:INFO:train: iterations={max_iter} batch={ims_per_batch} lr={base_lr}")

            if use_thermal:
                logger.info("[train] Scanning TRAIN/VALID for radiometric data…")
                _, tr_stats = scan_and_decode_split(TRAIN_DIR)
                _, va_stats = scan_and_decode_split(VALID_DIR)
                logger.info(f"[train] TRAIN RJPEG: ok={tr_stats['ok']} fail={tr_stats['fail']} total={tr_stats['total']}")
                logger.info(f"[train] VALID RJPEG: ok={va_stats['ok']} fail={va_stats['fail']} total={va_stats['total']}")
                if CANCEL_FLAGS["train"]:
                    logger.info("UI:INFO:train: Cancel observed before training loop. Aborting.")
                    return
                if tr_stats["ok"] == 0:
                    err = tr_stats.get("first_error") or "No radiometric data or DJI SDK not available."
                    logger.info(f"UI:INFO:train: Could not decode thermal from any TRAIN images. Reason: {err}. Proceeding with RGB-only.")
                    logger.warning("[train] WARNING: No thermal in TRAIN. Falling back to RGB-only.")
                    final = _train_rgb_only_with_params(run_dir, max_iter, base_lr, ims_per_batch)
                else:
                    if tr_stats["fail"] > 0:
                        logger.info(
                            f"UI:INFO:train: Some TRAIN images lack thermal ({tr_stats['fail']}/{tr_stats['total']}). "
                            "Those will use zeros in the thermal channel."
                        )
                    final = _train_rgb_thermal_with_params(run_dir, max_iter, base_lr, ims_per_batch)
            else:
                logger.info("[train] RGB-only selected.")
                final = _train_rgb_only_with_params(run_dir, max_iter, base_lr, ims_per_batch)

            logger.info(f"[train] Done. Weights at: {final}")
            logger.info("UI:OK:train: Training completed.")
        except Exception as e:
            logger.exception(f"[train] FAILED: {e}")
            logger.error(f"UI:ERR:train: Training failed: {e}")
        finally:
            logger.removeHandler(file_handler)
            for lg, fh in ext_handlers:
                lg.removeHandler(fh)

# -------------- Train API --------------
@app.post("/api/train")
async def api_train(
    use_thermal: bool = Form(default=False),
    max_iter: int = Form(default=500),
    base_lr: float = Form(default=0.002),
    ims_per_batch: int = Form(default=4),
):
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, _train_job, use_thermal, max_iter, base_lr, ims_per_batch)
    return {
        "ok": True,
        "use_thermal": use_thermal,
        "max_iter": max_iter,
        "base_lr": base_lr,
        "ims_per_batch": ims_per_batch,
    }

# -------------- List model runs --------------
def _list_models():
    models = []
    for d in sorted([p for p in OUTPUTS.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True):
        if (d / "model_final.pth").exists():
            meta = _read_model_meta(d)
            models.append({
                "name": d.name,
                "mtime": int(d.stat().st_mtime),
                "input_mode": meta.get("input_mode", "rgb")
            })
    return models

@app.get("/api/models")
async def api_models():
    return {"ok": True, "models": _list_models()}

# -------------- Test dataset helpers --------------
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
def _is_image(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS

def _slugify(name: str) -> str:
    base = Path(name).stem
    base = re.sub(r"[^A-Za-z0-9._-]+", "-", base).strip("-._")
    return base or "dataset"

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

def _save_zip_as_dataset(up: UploadFile) -> Path:
    base = _slugify(up.filename or "dataset")
    dest = _unique_dataset_dir(base)
    dest.mkdir(parents=True, exist_ok=True)
    data = up.file.read()
    with io.BytesIO(data) as bio:
        with zipfile.ZipFile(bio) as zf:
            tmp = dest / "__tmp_extract__"
            tmp.mkdir(parents=True, exist_ok=True)
            zf.extractall(tmp)
            for p in tmp.rglob("*"):
                if p.is_file() and _is_image(p):
                    out = dest / p.name
                    k = 1
                    while out.exists():
                        out = dest / f"{p.stem}-{k}{p.suffix}"
                        k += 1
                    shutil.copy2(p, out)
            shutil.rmtree(tmp, ignore_errors=True)
    logger.info(f"[test] Created dataset '{dest.name}' with {len([*dest.glob('*')])} files.")
    return dest

def _save_images_as_dataset(files: List[UploadFile]) -> Path:
    base = _slugify(files[0].filename or "dataset")
    dest = _unique_dataset_dir(base)
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        if not _is_image(Path(f.filename)):
            continue
        out = dest / Path(f.filename).name
        with out.open("wb") as w:
            shutil.copyfileobj(f.file, w)
    logger.info(f"[test] Created dataset '{dest.name}' with {len([*dest.glob('*')])} files.")
    return dest

def _list_datasets():
    datasets = []
    for d in sorted([p for p in TEST_DIR.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True):
        count = sum(1 for p in d.iterdir() if p.is_file() and _is_image(p))
        datasets.append({
            "name": d.name,
            "count": count,
            "mtime": int(d.stat().st_mtime),
        })
    return datasets

# ---- EXIF helpers ----
def _exif_gps_to_deg(gps_ifd) -> Optional[tuple]:
    def _to_deg(val):
        d = val[0][0]/val[0][1]; m = val[1][0]/val[1][1]; s = val[2][0]/val[2][1]
        return d + m/60 + s/3600
    try:
        lat = _to_deg(gps_ifd[piexif.GPSIFD.GPSLatitude])
        lon = _to_deg(gps_ifd[piexif.GPSIFD.GPSLongitude])
        lat_ref = gps_ifd.get(piexif.GPSIFD.GPSLatitudeRef, b'N')
        lon_ref = gps_ifd.get(piexif.GPSIFD.GPSLongitudeRef, b'E')
        if lat_ref in (b'S', 'S'): lat = -lat
        if lon_ref in (b'W', 'W'): lon = -lon
        return (lat, lon)
    except Exception:
        return None

def _read_exif_latlon(img_path: Path) -> Optional[tuple]:
    try:
        exif_dict = piexif.load(str(img_path))
        gps = exif_dict.get("GPS") or {}
        return _exif_gps_to_deg(gps) if gps else None
    except Exception:
        return None

def _to_feature_point(lat, lon, props):
    return {"type":"Feature", "properties": props or {}, "geometry":{"type":"Point","coordinates":[lon,lat]}}

def _to_square_polygon(lat, lon, size_deg=1e-5):
    return {
        "type": "Polygon",
        "coordinates": [[
            [lon - size_deg, lat - size_deg],
            [lon + size_deg, lat - size_deg],
            [lon + size_deg, lat + size_deg],
            [lon - size_deg, lat + size_deg],
            [lon - size_deg, lat - size_deg],
        ]]
    }

def _preds_to_geojson(images_dir: Path, preds_dir: Path, media_session_dir: Path, score_thresh: float = 0.5) -> Tuple[Path, list]:
    features = []
    image_points = []
    for json_file in sorted((preds_dir).glob("*.json")):
        data = json.loads(json_file.read_text())
        name = data.get("file") or json_file.stem
        src_img = images_dir / name
        latlon = _read_exif_latlon(src_img) if src_img.exists() else None
        img_url = f"/media/{media_session_dir.relative_to(MEDIA_DIR)}/images/{src_img.name}" if src_img.exists() else None

        if latlon:
            lat, lon = latlon
            features.append(_to_feature_point(lat, lon, {"type":"image","name":name,"url":img_url}))
            image_points.append({"name": name, "lat": lat, "lon": lon, "url": img_url})

            boxes = data.get("boxes") or []
            scores = data.get("scores") or []
            classes = data.get("classes") or []
            for b, s, c in zip(boxes, scores, classes):
                if s < score_thresh:
                    continue
                props = {"type":"anomaly","image":name,"score":float(s),"class":int(c),"bbox_pixel":b, "polygon_note":"approx_square"}
                poly = _to_square_polygon(lat, lon, 1e-5)
                features.append({"type":"Feature","properties":props,"geometry":poly})
    gj = {"type":"FeatureCollection","features":features, "name":"pvrt_anomalies"}
    out = media_session_dir / "anomalies.geojson"
    out.write_text(json.dumps(gj, indent=2))
    return out, image_points

def _copy_images_for_session(src_dir: Path, ses_dir: Path) -> Path:
    img_dir = ses_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(src_dir.iterdir()):
        if p.is_file() and _is_image(p):
            dst = img_dir / p.name
            if not dst.exists():
                shutil.copy2(p, dst)
    return img_dir

def _ensure_max_filesize_jpg(bgr: "cv2.Mat", max_bytes: int = 40 * 1024 * 1024) -> bytes:
    h, w = bgr.shape[:2]
    scale = 1.0
    quality = 90
    while True:
        resized = bgr if scale >= 0.999 else cv2.resize(bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        if len(buf) <= max_bytes or (scale <= 0.3 and quality <= 60):
            return bytes(buf)
        if scale > 0.5:
            scale *= 0.8
        elif quality > 60:
            quality -= 5
        else:
            scale *= 0.9

def _draw_overlays(images_dir: Path, preds_dir: Path, session_dir: Path) -> Tuple[Path, Path, list]:
    overlays = session_dir / "overlays"
    thumbs   = session_dir / "thumbs"
    overlays.mkdir(exist_ok=True, parents=True)
    thumbs.mkdir(exist_ok=True, parents=True)
    manifest = []

    json_files = sorted(preds_dir.glob("*.json"))
    total = len(json_files)
    done = 0

    for jf in json_files:
        data = json.loads(jf.read_text())
        name = data.get("file") or jf.stem
        src = images_dir / name
        if not src.exists():
            done += 1
            logger.info(f"UI:INFO:test: overlay {done}/{total}: {name} (missing image, skipped)")
            continue

        img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        if img is None:
            try:
                pil = Image.open(src).convert("RGB")
                img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            except Exception:
                done += 1
                logger.info(f"UI:INFO:test: overlay {done}/{total}: {name} (cannot read, skipped)")
                continue

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        boxes  = data.get("boxes") or []
        scores = data.get("scores") or []
        classes= data.get("classes") or []

        for b, s, c in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(int, b[:4])
            cv2.rectangle(img, (x1, y1), (x2, y2), (30, 220, 30), 2)
            label = f"{int(c)}:{s:.2f}"
            cv2.putText(img, label, (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 220, 30), 2, cv2.LINE_AA)

        out_jpg = overlays / f"{Path(name).stem}.jpg"
        jpg = _ensure_max_filesize_jpg(img, 40 * 1024 * 1024)
        out_jpg.write_bytes(jpg)

        # Thumbs smaller (half of previous 160 -> 80px width)
        h, w = img.shape[:2]
        tw = 80
        th = int(h * (tw / w))
        thumb_img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        ok, tbuf = cv2.imencode(".jpg", thumb_img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        (thumbs / out_jpg.name).write_bytes(tbuf)

        manifest.append({
            "file": name,
            "overlay": f"/media/{session_dir.relative_to(MEDIA_DIR)}/overlays/{out_jpg.name}",
            "thumb":   f"/media/{session_dir.relative_to(MEDIA_DIR)}/thumbs/{out_jpg.name}",
        })
        done += 1
        logger.info(f"UI:INFO:test: overlay {done}/{total}: {name}")

    (session_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return overlays, thumbs, manifest

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

def _count_thermal_tifs(ds_dir: Path) -> int:
    tdir = ds_dir / "thermal"
    if not tdir.exists():
        return 0
    return sum(1 for p in tdir.rglob("*") if p.suffix.lower() in (".tif", ".tiff"))

# -------------- List datasets --------------
@app.get("/api/test_datasets")
async def api_test_datasets():
    return {"ok": True, "datasets": _list_datasets()}

# -------------- Upload dataset(s) --------------
@app.post("/api/test_upload")
async def api_test_upload(files: List[UploadFile] = File(...)):
    buffered: List[UploadFile] = []
    for f in files:
        content = await f.read()
        buffered.append(UploadFile(filename=f.filename, file=io.BytesIO(content), headers=f.headers))

    created = []
    with redirect_std_to_logger():
        zips = [f for f in buffered if (f.filename or "").lower().endswith(".zip")]
        imgs = [f for f in buffered if not (f.filename or "").lower().endswith(".zip")]

        for z in zips:
            ds = _save_zip_as_dataset(z)
            created.append(ds.name)

        if imgs:
            ds = _save_images_as_dataset(imgs)
            created.append(ds.name)

        if not created:
            logger.info("UI:WARN:test: No valid files (images/zip) detected.")
            return {"ok": False, "created": []}

        logger.info(f"UI:OK:test: Created datasets: {', '.join(created)}")
        return {"ok": True, "created": created}

# -------------- Run test/inference --------------
@app.post("/api/test_run")
async def api_test_run(
    dataset: str = Form(...),
    model: Optional[str] = Form(default=None),
    use_thermal: bool = Form(default=False),
):
    ds_dir = TEST_DIR / dataset
    if not ds_dir.exists() or not ds_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset}' not found.")

    if model:
        model_dir = OUTPUTS / model
    else:
        models = _list_models()
        if not models:
            raise HTTPException(status_code=404, detail="No trained models found.")
        model_dir = OUTPUTS / models[0]["name"]
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail=f"Model '{model_dir.name}' not found.")

    stamp = _now_stamp()
    sess = MEDIA_DIR / f"sessions/{stamp}"
    sess.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(sess / "test.log", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S"))
    logger.addHandler(file_handler)

    with redirect_std_to_logger():
        try:
            meta = _read_model_meta(model_dir)
            mode = meta.get("input_mode", "rgb")

            logger.info(f"[test] Model: {model_dir.name} (mode={mode})")
            logger.info(f"[test] Dataset: '{dataset}'  use_thermal={use_thermal}")
            logger.info(f"[test] Session: {sess}")

            images_for_map = _copy_images_for_session(ds_dir, sess)

            if use_thermal and mode == "rgbtherm":
                logger.info("[test] Thermal requested. Decoding radiometric RJPEG to .tif…")
                _, stats = scan_and_decode_split(ds_dir)  # writes to ds_dir/thermal/*.tif
                tif_count = _count_thermal_tifs(ds_dir)
                if stats["ok"] == 0:
                    reason = stats.get("first_error") or "No radiometric data or DJI SDK not available."
                    logger.info(
                        "UI:INFO:test: None of the images contain radiometric data. "
                        f"Reason: {reason}. Proceeding with RGB tensors (thermal channel=0)."
                    )
                else:
                    logger.info(f"[test] Thermal decode: ok={stats['ok']} fail={stats['fail']} total={stats['total']}  tif_files={tif_count}")
                    if stats["fail"] > 0:
                        logger.info(
                            f"UI:INFO:test: Some images lack thermal ({stats['fail']}/{stats['total']}). "
                            "Those will use zeros in the thermal channel."
                        )
            elif use_thermal and mode == "rgb":
                logger.info("UI:INFO:test: Model is RGB-only; ignoring 'Use thermal band' during testing.")

            preds_dir = predict_folder(ds_dir, sess, model_dir, use_thermal)
            logger.info(f"[test] Predictions at {preds_dir}")

            ov_dir, th_dir, manifest = _draw_overlays(images_for_map, preds_dir, sess)
            logger.info(f"[test] Overlays: {ov_dir}  Thumbs: {th_dir}")

            gj_path, image_points = _preds_to_geojson(images_for_map, preds_dir, sess)
            logger.info(f"[test] GeoJSON: {gj_path}")

            assets = _session_assets(sess)

            logger.info("UI:OK:test: Test complete. Layers added to the map.")
            return {
                "ok": True,
                "session": str(sess.relative_to(MEDIA_DIR)),
                "geojson_url": f"/media/{gj_path.relative_to(MEDIA_DIR)}",
                "image_points": image_points,
                "manifest": manifest,
                "assets": assets,
                "tiler": "ok" if RIO_OK else "unavailable"
            }
        finally:
            logger.removeHandler(file_handler)

# -------------- Sessions for Results/Map tabs --------------
def _list_sessions():
    base = MEDIA_DIR / "sessions"
    if not base.exists():
        return []
    items = []
    for d in sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True):
        items.append({
            "name": f"sessions/{d.name}",
            "mtime": int(d.stat().st_mtime)
        })
    return items

@app.get("/api/sessions")
async def api_sessions():
    return {"ok": True, "sessions": _list_sessions()}

@app.get("/api/session_summary")
async def api_session_summary(session: str):
    ses = MEDIA_DIR / session
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
