# backend/pvrt/web/app.py
import asyncio
import io
import json
import logging
import sys
import shutil
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request

from . import sse as sse_mod
from .sse import LogBroker, SSELogHandler, sse_response

from ..dataops.scan_decode_split import scan_and_decode_split
from ..trainops.trainer_rgb_only import RGBOnlyTrainer
from ..trainops.trainer_rgb_thermal_tolerant import RTolerantTrainer
from ..infer.predict_rgb_thermal import predict_folder

import piexif
import warnings

app = FastAPI(title="PVRT API", version="0.9.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# ---- Logging / SSE ----
broker = LogBroker()
sse_handler = SSELogHandler(broker)
sse_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S"))
sse_handler.setLevel(logging.INFO)

logger = logging.getLogger("pvrt")
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.addHandler(sse_handler)

def _attach_external_loggers():
    # Attach SSE to Detectron2/FVCore etc., stop propagation to root
    for name in ("detectron2", "fvcore", "d2", "detectron2.data", "detectron2.utils.events", "detectron2.engine"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.INFO)
        if not any(isinstance(h, SSELogHandler) for h in lg.handlers):
            lg.addHandler(sse_handler)
        lg.propagate = False

@app.on_event("startup")
async def _on_startup():
    loop = asyncio.get_running_loop()
    sse_mod.set_event_loop(loop)  # <-- IMPORTANT for thread-safe SSE
    _attach_external_loggers()
    warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release.*")
    logger.info("Startup: SSE ready.")

# ---- Redirect stdout/stderr to logging during long jobs ----
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

# ---- Paths ----
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

# -----------------------------
# Training helpers
# -----------------------------
def _force_axis_aligned(cfg):
    """Keep anchors axis-aligned to match 4-dim boxes with StandardROIHeads."""
    cfg.MODEL.ANCHOR_GENERATOR.ANGLES = [[0]]
    logging.getLogger("pvrt").info("[train] Forcing axis-aligned anchors: MODEL.ANCHOR_GENERATOR.ANGLES=[[0]]")

def _train_rgb_only_with_params(max_iter: int, base_lr: float, ims_per_batch: int):
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    from detectron2.engine import default_setup
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.utils.events import EventStorage

    from ..trainops.datasets import register_split_coco
    from ..trainops.helpers import get_num_classes

    train_dir, val_dir, out_dir = TRAIN_DIR, VALID_DIR, OUTPUTS
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
    default_setup(cfg, {})  # freezes cfg

    # ---- IMPORTANT: wrap trainer lifecycle in EventStorage to satisfy any early logging
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


def _train_rgb_thermal_with_params(max_iter: int, base_lr: float, ims_per_batch: int):
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    from detectron2.engine import default_setup
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.utils.events import EventStorage

    from ..trainops.datasets import register_split_coco
    from ..trainops.helpers import get_num_classes

    train_dir, val_dir, out_dir = TRAIN_DIR, VALID_DIR, OUTPUTS
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
    # 4-channel normalization (RGB + Thermal)
    cfg.MODEL.PIXEL_MEAN = [0.5, 0.5, 0.5, 0.5]
    cfg.MODEL.PIXEL_STD  = [0.5, 0.5, 0.5, 0.5]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = int(num_classes)
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    _force_axis_aligned(cfg)

    cfg.OUTPUT_DIR = str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    default_setup(cfg, {})  # freezes cfg

    # ---- IMPORTANT: wrap trainer lifecycle in EventStorage to satisfy the preflight probe
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
    # Runs in a background thread; stdio is redirected to logger which is SSE-enabled
    with redirect_std_to_logger():
        try:
            logger.info(f"[train] use_thermal={use_thermal} max_iter={max_iter} base_lr={base_lr} batch={ims_per_batch}")
            if use_thermal:
                logger.info("[train] Scanning TRAIN/VALID for radiometric data…")
                _, tr_stats = scan_and_decode_split(TRAIN_DIR)
                _, va_stats = scan_and_decode_split(VALID_DIR)
                logger.info(f"[train] TRAIN RJPEG: ok={tr_stats['ok']} fail={tr_stats['fail']} total={tr_stats['total']}")
                logger.info(f"[train] VALID RJPEG: ok={va_stats['ok']} fail={va_stats['fail']} total={va_stats['total']}")
                if tr_stats["ok"] == 0:
                    err = tr_stats.get("first_error") or "No radiometric data or DJI SDK not available."
                    logger.info(f"UI:INFO:train: Could not decode thermal from any TRAIN images. Reason: {err}. Proceeding with RGB-only.")
                    logger.warning("[train] WARNING: No thermal in TRAIN. Falling back to RGB-only.")
                    final = _train_rgb_only_with_params(max_iter, base_lr, ims_per_batch)
                else:
                    if tr_stats["fail"] > 0:
                        logger.info(
                            f"UI:INFO:train: Some TRAIN images lack thermal ({tr_stats['fail']}/{tr_stats['total']}). "
                            "Those will use zeros in the thermal channel."
                        )
                    final = _train_rgb_thermal_with_params(max_iter, base_lr, ims_per_batch)
            else:
                logger.info("[train] RGB-only selected.")
                final = _train_rgb_only_with_params(max_iter, base_lr, ims_per_batch)

            logger.info(f"[train] Done. Weights at: {final}")
            logger.info("UI:OK:train: Training completed.")
        except Exception as e:
            logger.exception(f"[train] FAILED: {e}")
            logger.error(f"UI:ERR:train: Training failed: {e}")

# -----------------------------
# API: Train (now non-blocking)
# -----------------------------
@app.post("/api/train")
async def api_train(
    use_thermal: bool = Form(default=False),
    max_iter: int = Form(default=9000),
    base_lr: float = Form(default=0.002),
    ims_per_batch: int = Form(default=4),
):
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, _train_job, use_thermal, max_iter, base_lr, ims_per_batch)
    logger.info("UI:OK:train: Training started…")
    return {
        "ok": True,
        "use_thermal": use_thermal,
        "max_iter": max_iter,
        "base_lr": base_lr,
        "ims_per_batch": ims_per_batch,
    }

# -----------------------------
# Test / uploads (kept sync; short)
# -----------------------------
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
    return {
        "type":"Feature",
        "properties": props or {},
        "geometry":{"type":"Point","coordinates":[lon,lat]}
    }

def _preds_to_geojson(images_dir: Path, preds_dir: Path, media_session_dir: Path, score_thresh: float = 0.5) -> tuple[Path, list]:
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
                props = {"type":"anomaly","image":name,"score":float(s),"class":int(c),"bbox_pixel":b}
                features.append(_to_feature_point(lat, lon, props))
    gj = {"type":"FeatureCollection","features":features, "name":"pvrt_anomalies"}
    out = media_session_dir / "anomalies.geojson"
    out.write_text(json.dumps(gj, indent=2))
    return out, image_points

def _save_uploads(files: List[UploadFile], session_dir: Path) -> Path:
    img_dir = session_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        dest = img_dir / f.filename
        with dest.open("wb") as w:
            shutil.copyfileobj(f.file, w)
    return img_dir

@app.post("/api/test_uploads")
async def api_test_uploads(
    use_thermal: bool = Form(default=False),
    files: List[UploadFile] = File(...),
):
    # Buffer files so we can process synchronously
    buffered = []
    for f in files:
        content = await f.read()
        mem = UploadFile(filename=f.filename, file=io.BytesIO(content), headers=f.headers, content_type=f.content_type)
        buffered.append(mem)

    with redirect_std_to_logger():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sess = MEDIA_DIR / f"sessions/{stamp}"
        sess.mkdir(parents=True, exist_ok=True)
        logger.info(f"[test] Session: {sess}")

        images_dir = _save_uploads(buffered, sess)
        logger.info(f"[test] Saved {len(list(images_dir.glob('*')))} files.")

        # Check model mode and radiometric availability for UI messages
        meta = _read_model_meta(OUTPUTS)
        mode = meta.get("input_mode", "rgb")
        if use_thermal and mode == "rgb":
            logger.info("UI:INFO:test: Model is RGB-only; ignoring 'Use thermal band' during testing.")
        elif use_thermal and mode == "rgbtherm":
            _, stats = scan_and_decode_split(images_dir)
            if stats["ok"] == 0:
                reason = stats.get("first_error") or "No radiometric data or DJI SDK not available."
                logger.info(
                    "UI:INFO:test: None of the uploaded images contain radiometric data. "
                    f"Reason: {reason}. Proceeding with RGB tensors (thermal channel=0)."
                )
            elif stats["fail"] > 0:
                logger.info(
                    f"UI:INFO:test: Some uploaded images lack thermal ({stats['fail']}/{stats['total']}). "
                    "Those will use zeros in the thermal channel."
                )

        preds_dir = predict_folder(images_dir, sess, OUTPUTS, use_thermal)
        logger.info(f"[test] Predictions at {preds_dir}")

        gj_path, image_points = _preds_to_geojson(images_dir, preds_dir, sess)
        logger.info(f"[test] GeoJSON: {gj_path}")

        return {
            "ok": True,
            "session": str(sess.relative_to(MEDIA_DIR)),
            "geojson_url": f"/media/{gj_path.relative_to(MEDIA_DIR)}",
            "image_points": image_points,
        }

# ---- Serve media & frontend last
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR), html=False), name="media")
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="web")
