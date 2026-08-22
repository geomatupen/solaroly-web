# backend/pvrt/backends/detectron/backend.py
"""
Detectron backend adapter
-------------------------
Implements the generic Backend interface so the web layer doesn't need to care
whether the engine under the hood is Detectron, YOLO, etc.

Key behaviors preserved:
- Train with RGB (3-channel) or thermal-as-RGB (3-channel grayscale)
- For testing: if user requested thermal AND data has thermal -> test with thermal-as-RGB
- Writes/reads model_meta.json (adds backend + normalized input_mode)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

# Generic interface + configs
from ...core.registry import Backend, TrainConfig, PredictConfig
from ...core.io import (
    load_model_meta,
    save_model_meta,
    input_mode_from_meta,
    backend_name_from_meta,
    has_thermal_for_images,
    THERMAL_EXTS,
)

from .train.datasets import register_split_coco, _find_coco_json
from .train.trainer_rgb_only import RGBOnlyTrainer
from .infer.predict_rgb_only import predict_folder as predict_folder_rgb

# detectron2 bits
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
from detectron2.engine import hooks
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.logger import setup_logger

from shutil import copy2
import math
import json

# Use helpers.py to read class count from COCO (source of truth)
from ...core.helpers import get_num_classes
from ...core.thermal import normalize_thermal

import logging
log = logging.getLogger("pvrt")


# ----------------------------- small helpers ------------------------------ #

def _safe_solver_steps(max_iter: int) -> tuple[int, int]:
    a = max(1, max_iter // 2)
    b = max(2, (3 * max_iter) // 4)
    if b <= a:
        b = a + 1
    return (a, b)


def _normalize_and_save_meta(out_dir: Path, meta: Dict) -> None:
    existing = load_model_meta(out_dir)
    merged = {**existing, **meta}
    merged["input_mode"] = input_mode_from_meta(merged, default=meta.get("input_mode", "rgb"))
    merged["backend"] = backend_name_from_meta(merged, default="detectron")
    save_model_meta(out_dir, merged)


def _coherent_input_resize(cfg) -> None:
    """
    Normalize cfg.INPUT.* so any Detectron2 path that consults it sees a valid combo.
    This prevents 'range' + [800] errors.
    """
    # training policy
    sampling = str(getattr(cfg.INPUT, "MIN_SIZE_TRAIN_SAMPLING", "choice")).lower()
    min_train = getattr(cfg.INPUT, "MIN_SIZE_TRAIN", [800])
    if isinstance(min_train, int):
        min_train = [min_train]
    elif isinstance(min_train, tuple):
        min_train = list(min_train)

    if sampling == "range" and len(min_train) == 1:
        # illegal combo -> coerce to a valid one
        sampling = "choice"

    if sampling == "range":
        lo, hi = int(min(min_train)), int(max(min_train))
        cfg.INPUT.MIN_SIZE_TRAIN = (lo, hi)              # two ints required by 'range'
        cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "range"
    else:
        cfg.INPUT.MIN_SIZE_TRAIN = [int(x) for x in min_train] or [800]
        cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"

    # max caps + test policy
    cfg.INPUT.MAX_SIZE_TRAIN = int(getattr(cfg.INPUT, "MAX_SIZE_TRAIN", 1333))
    min_test = getattr(cfg.INPUT, "MIN_SIZE_TEST", 800)
    if isinstance(min_test, (list, tuple)):
        min_test = int(min_test[0] if min_test else 800)
    cfg.INPUT.MIN_SIZE_TEST = int(min_test)
    cfg.INPUT.MAX_SIZE_TEST = int(getattr(cfg.INPUT, "MAX_SIZE_TEST", 1333))
    if not getattr(cfg.INPUT, "FORMAT", None):
        cfg.INPUT.FORMAT = "BGR"


# ----------------------------- Detectron backend -------------------------- #

class DetectronBackend(Backend):
    """
    Backend adapter for Detectron2.
    """

    def train(self, cfg_in: TrainConfig) -> Path:
        train_dir, val_dir, out_dir = Path(cfg_in.train_dir), Path(cfg_in.val_dir), Path(cfg_in.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Decide if this run should use thermal
        thermal_ok = bool(cfg_in.use_thermal and has_thermal_for_images(train_dir))
        # All models are 3-channel: either 3-channel RGB or 3-channel thermal (decoded)
        try:
            requested_channels = int(getattr(cfg_in, "channel_count", 3))
        except (TypeError, ValueError):
            requested_channels = 3
        # Force to 3 channels
        if requested_channels != 3:
            log.info("UI:INFO:train: channel_count adjusted to 3 (only 3-channel models supported)")
            requested_channels = 3

        # Effective channel count is always 3 for all models
        effective_channels = 3

        log.info("UI:INFO:train: effective_channels=3 (thermal_ok=%s)", thermal_ok)
        # If thermal is available, training will use thermal-as-RGB (grayscale encoded as 3-channel)
        if thermal_ok:
            log.info("UI:INFO:train: Thermal grayscale will be used for training (thermal-as-RGB)")

    # Note: to avoid duplicating image bytes, this backend does not create
    # per-run dataset copies. Backends operate on the existing dataset
    # folders in-place (e.g. `data/train` and `data/valid`). If thermal
    # decoding is needed, the decoding step will write paired files into a
    # `thermal/` subfolder inside those directories; `has_thermal_for_images()` will detect that.

        # Register datasets (idempotent). Provide clearer errors when COCO JSON
        # is missing so callers (web UI) get a helpful message instead of
        # opaque KeyError/registry errors.
        try:
            # quick validation: ensure annotations exist (raises FileNotFoundError)
            _find_coco_json(train_dir)
        except FileNotFoundError as e:
            raise RuntimeError(f"COCO annotations not found in train_dir: {train_dir}") from e
        try:
            _find_coco_json(val_dir)
        except FileNotFoundError:
            # valid may be optional for some workflows; log and continue (will register if present)
            log.warning("No COCO annotations found in val_dir: %s", val_dir)

        register_split_coco("pv_train", train_dir)
        register_split_coco("pv_val",   val_dir)

    # --- num_classes from COCO (helpers.py) ---
        ann_json = _find_coco_json(train_dir)
        num_classes = int(get_num_classes(ann_json))

        # class_names from MetadataCatalog (Detectron populates this at registration)
        # MetadataCatalog.get should provide metadata for the registered dataset.
        # If it does not, surface a clear error.
        try:
            meta_train = MetadataCatalog.get("pv_train")
            class_names = list(meta_train.thing_classes) if getattr(meta_train, "thing_classes", None) else []
        except Exception as e:
            raise RuntimeError(f"Failed to obtain metadata for registered dataset 'pv_train': {e}") from e

        # fallback to COCO if still empty (prevents blank in thermal)
        if not class_names:
            ann_json = _find_coco_json(train_dir)
            data = json.loads(Path(ann_json).read_text(encoding="utf-8"))
            cats = data.get("categories", []) if isinstance(data.get("categories", []), list) else []
            try:
                cats = sorted(cats, key=lambda c: int(c.get("id", 0)))
            except (TypeError, ValueError):
                # keep original order if ids are non-numeric
                log.debug("non-numeric category ids encountered while sorting categories")
            class_names = [str(c.get("name", f"class_{i}")) for i, c in enumerate(cats)]

        # Build cfg (always) - using Faster R-CNN for bounding boxes only
        MODEL_YAML = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        mask_on = False
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(MODEL_YAML))
        cfg.MODEL.MASK_ON = mask_on

        cfg.DATASETS.TRAIN = ("pv_train",)
        cfg.DATASETS.TEST  = ("pv_val",)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL_YAML)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

        cfg.SOLVER.IMS_PER_BATCH = int(cfg_in.ims_per_batch or 2)
        cfg.SOLVER.BASE_LR       = float(cfg_in.base_lr or 0.00025)
        cfg.SOLVER.MAX_ITER      = int(cfg_in.max_iter or 1000)
        # Build solver steps but be defensive: if user set very small MAX_ITER
        # the fvcore MultiStepParamScheduler will raise if the total number
        # of updates is <= number of milestones. Keep only milestones that
        # are strictly less than MAX_ITER and drop the schedule when it's
        # not sensible (very short runs). This prevents failures for tiny
        # smoke runs (e.g., max_iter <= 5) while preserving reasonable
        # multi-step schedules for typical runs.
        raw_steps = list(_safe_solver_steps(cfg.SOLVER.MAX_ITER))
        cfg.SOLVER.STEPS = [int(s) for s in raw_steps if 0 < int(s) < int(cfg.SOLVER.MAX_ITER)]
        # cfg.SOLVER.CHECKPOINT_PERIOD = max(1001, int(cfg.SOLVER.MAX_ITER / 6))
        cfg.SOLVER.CHECKPOINT_PERIOD = 10**9  # disabled time-based checkpoints
        cfg.SOLVER.LOG_PERIOD    = 1

        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
        cfg.TEST.EVAL_PERIOD = 500 # max(50, int(cfg.SOLVER.MAX_ITER // 10))
        cfg.OUTPUT_DIR = str(out_dir)
        cfg.INPUT.FORMAT = "BGR"

        # --- critical: make resize settings coherent to avoid 'range'+[800] ---
        _coherent_input_resize(cfg)

        # Train
        setup_logger()
        trainer = RGBOnlyTrainer(cfg)
        log.info(
            f"[train] run={getattr(cfg_in, 'run_name', out_dir.name)} "
            f"thermal={thermal_ok} classes={num_classes}"
        )

        class _RawLossLogger(hooks.HookBase):
            def after_step(self):
                if self.trainer.iter % cfg.SOLVER.LOG_PERIOD == 0:
                    hb = self.trainer.storage.history("total_loss")
                    s = hb.latest()
                    raw = float(getattr(s, "value", s))
                    logging.getLogger("pvrt.test").info(f"LOG:loss: iter={self.trainer.iter} total_loss(raw)={raw:.4f}")
        # trainer.register_hooks([_RawLossLogger()])

        # capture last seen total_loss from EventStorage
        class _LossTap(hooks.HookBase):
            def __init__(self):
                self.last_raw = None
                self.last_med20 = None  # matches what the console prints (20-iter median)

            def after_step(self):
                hb = self.trainer.storage.history("total_loss")   # HistoryBuffer
                s = hb.latest()                                   # Scalar or float
                self.last_raw = float(getattr(s, "value", s))
                try:
                    self.last_med20 = float(hb.median(20))
                except (TypeError, ValueError):
                    self.last_med20 = self.last_raw


        loss_tap = _LossTap()
        trainer.register_hooks([loss_tap])

        # 2) build val loader/evaluator once
        val_loader = RGBOnlyTrainer.build_test_loader(cfg, "pv_val")
        evaluator  = COCOEvaluator("pv_val", distributed=False, output_dir=str(out_dir))

        _best = {"ap50": float("-inf")} #AP50 - Average Precision at IoU 0.50. ie. area under the entire precision–recall curve
        _latest = {"ap50": None}

        def _ap50_pct_from_results(res) -> float:
            """Return AP50 in PERCENT (0..100) from COCOEvaluator results; NaN if missing."""
            bbox = res.get("bbox")
            if isinstance(bbox, dict):
                try:
                    val = float(bbox.get("AP50"))
                    return val if 0.0 <= val <= 100.0 else float("nan")
                except (TypeError, ValueError):
                    pass
            return float("nan")

        _best   = {"ap50_pct": float("-inf")}
        _latest = {"ap50_pct": None}

        def _eval_and_log():
            # inference_on_dataset may raise a KeyError when the underlying
            # COCO API expects certain top-level keys (like 'info') to be
            # present in the dataset. COCO JSON files may omit these optional
            # fields which can cause pycocotools to fail during loadRes(). To be robust, catch that KeyError, attempt to
            # inject minimal 'info' into the evaluator's COCO dataset, and
            # retry once.
            try:
                res = inference_on_dataset(trainer.model, val_loader, evaluator)
            except KeyError as ke:
                msg = str(ke)
                if "'info'" in msg or 'info' in msg:
                    try:
                        # Best-effort patch: add a minimal 'info' dict to the
                        # internal COCO dataset so pycocotools.loadRes() can
                        # proceed. This avoids forcing users to edit their
                        # COCO files which may be missing optional fields.
                        if hasattr(evaluator, '_coco_api') and getattr(evaluator, '_coco_api') is not None:
                            coco_ds = evaluator._coco_api.dataset
                            if isinstance(coco_ds, dict) and 'info' not in coco_ds:
                                coco_ds['info'] = {'description': 'pvrt dataset (info added)'}
                                evaluator._coco_api.dataset = coco_ds
                                # retry once
                                res = inference_on_dataset(trainer.model, val_loader, evaluator)
                            else:
                                raise
                        else:
                            raise
                    except Exception:
                        # re-raise the original KeyError to be handled upstream
                        raise
                else:
                    raise

            ap50_pct = _ap50_pct_from_results(res)       # percent, not fraction
            if not math.isfinite(ap50_pct):
                ap50_pct = 0.0                           # keep BestCheckpointer deterministic

            # single source of truth for BestCheckpointer
            trainer.storage.put_scalar("val/AP50_pct", ap50_pct, smoothing_hint=False)
            _latest["ap50_pct"] = ap50_pct

            if ap50_pct > _best["ap50_pct"]:
                _best["ap50_pct"] = ap50_pct
                log.info(f"UI:OK:train: new_best AP50={ap50_pct:.3f}% at iter={trainer.iter} → model_best.pth")
                trainer.checkpointer.save("model_best")   # safety write
                _normalize_and_save_meta(out_dir, {
                    "best_model": {
                        "iter": int(trainer.iter),
                        "val_bbox_AP50": round(ap50_pct, 4),    # percent
                        "total_loss_med20": None if loss_tap.last_med20 is None else float(loss_tap.last_med20),
                        "total_loss_raw":   None if loss_tap.last_raw   is None else float(loss_tap.last_raw),
                        "path": str(Path(out_dir.name) / "model_best.pth"),
                    }
                })
            return res


        # evaluate every EVAL_PERIOD; save model_best.pth when metric improves
        trainer.register_hooks([
            # hooks.EvalHook(cfg.TEST.EVAL_PERIOD, lambda: inference_on_dataset(trainer.model, val_loader, evaluator)),
            hooks.EvalHook(cfg.TEST.EVAL_PERIOD, _eval_and_log),
            hooks.BestCheckpointer(
                cfg.TEST.EVAL_PERIOD,
                trainer.checkpointer,
                val_metric="val/AP50_pct",  
                mode="max",
                file_prefix="model_best"
            ),
        ])


        trainer.resume_or_load(resume=False)
        trainer.train()

        

        # Ensure final checkpoint exists and model_final.pth is made a copy of the best (if any)
        best = Path(out_dir) / "model_best.pth"
        if best.exists():
            copy2(best, Path(out_dir) / "model_final.pth")
            log.info("PHASE:save model_final <- model_best")
        else:
            trainer.checkpointer.save("model_final")
            log.info("PHASE:save model_final (no best found)")

        final_ap50_pct = _latest["ap50_pct"]
        if final_ap50_pct is None:
            try:
                s = trainer.storage.history("val/AP50_pct").latest()
                final_ap50_pct = float(getattr(s, "value", s))
            except (TypeError, ValueError, AttributeError):
                final_ap50_pct = None

            _normalize_and_save_meta(out_dir, {
                "final_model": {
                    "iter": int(trainer.iter),
                    "val_bbox_AP50": None if final_ap50_pct is None else round(final_ap50_pct, 4),
                    "total_loss_med20": None if loss_tap.last_med20 is None else float(loss_tap.last_med20),
                    "total_loss_raw":   None if loss_tap.last_raw   is None else float(loss_tap.last_raw),
                    "path": str(Path(out_dir.name) / "model_final.pth"),
                }
            })

        
        MODEL_NAME = Path(MODEL_YAML).stem
        # All models are 3-channel (no channel suffix needed)
        # Prepend the run name (or out_dir.name) so the UI shows runs similarly to YOLO
        run_prefix = getattr(cfg_in, "run_name", "") or out_dir.name
        # Save normalized meta for the run
        _normalize_and_save_meta(out_dir, {
            "backend": "detectron",
            "model_type": "fasterrcnn",
            "input_mode": "thermal" if thermal_ok else "rgb",
            # Record whether this model used thermal data during training so
            # test-time selection can prefer decoded thermal when available.
            "thermal_used": bool(thermal_ok and getattr(cfg_in, "use_thermal", False)),
            "selected_bands": getattr(cfg_in, "selected_bands", None),
            "channel_count": 3,
            "model_name": run_prefix,
            "model_zoo": MODEL_YAML,
            "num_classes": num_classes,
            "class_names": class_names,
            "score_thresh_test": float(cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST),
            "training_dataset": {
                "id": getattr(cfg_in, "dataset_id", ""),
                "name": getattr(cfg_in, "dataset_name", ""),
                "path": getattr(cfg_in, "dataset_path", ""),
                "format": getattr(cfg_in, "dataset_format", "coco"),
            },
            "train_params": {                         # NEW (grouped for clarity)
                "max_iter": int(cfg.SOLVER.MAX_ITER),
                "base_lr": float(cfg.SOLVER.BASE_LR),
                "ims_per_batch": int(cfg.SOLVER.IMS_PER_BATCH),
                "run_name": getattr(cfg_in, "run_name", ""),
            },
        })

        return out_dir

    def predict(self, cfg_in: PredictConfig) -> Path:
        """
        Backend obeys the decision made by the web bridge (cfg_in.use_thermal).
        """
        images_dir = Path(cfg_in.images_dir)
        out_dir    = Path(cfg_in.out_dir)
        weights    = Path(cfg_in.weights_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        meta = load_model_meta(weights)
        model_mode = input_mode_from_meta(meta, default="rgb").lower().strip()

        score_thresh = float(cfg_in.score_thresh) if cfg_in.score_thresh is not None else float(meta.get("score_thresh_test", 0.5))

        # All models are 3-channel. Determine if we use RGB or thermal.
        has_thermal = bool(has_thermal_for_images(images_dir))
        model_trained_with_thermal = bool(meta.get("thermal_used", False))
        use_thermal = bool(cfg_in.use_thermal and (model_mode == "thermal" or model_trained_with_thermal) and has_thermal)

        # model's recorded channel_count (if any)
        try:
            model_chan = int(meta.get("channel_count", 0) or 0)
        except (TypeError, ValueError):
            model_chan = 0

        # Always use 3-channel for inference
        effective_channels_test = 3
        selected_mode = 'thermal' if use_thermal else 'rgb'

        log.info(
            f"UI:INFO:test: backend=detectron | selected={selected_mode} | model_trained={model_chan or 'unknown'} | score_thresh={score_thresh:.3f}"
        )

        # If using thermal grayscale (3-channel), prepare a temporary folder with thermal images
        # SKIP if images_dir is "rotated_images" - they're already thermal-as-RGB from rotation script
        is_rotated_images = images_dir.name == "rotated_images"
        if use_thermal and not is_rotated_images:
            log.info("UI:INFO:test: Thermal grayscale will be used for testing")
            try:
                from shutil import copy2
                import tifffile
            except Exception:
                tifffile = None
            tmp = out_dir / "predict_thermal"
            tmp.mkdir(parents=True, exist_ok=True)
            exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

            # helper to locate thermal preview
            def _locate_thermal_for_rgb(rgb_path: Path):
                tdir = rgb_path.parent / "thermal"
                pjson = tdir / "pairs.json"
                if pjson.exists():
                    try:
                        pairs = json.loads(pjson.read_text(encoding="utf-8"))
                        target = pairs.get(rgb_path.name)
                        if target:
                            candidate = (rgb_path.parent / target).resolve()
                            if candidate.exists():
                                return candidate
                    except Exception:
                        pass
                # check common preview names under thermal/
                for e in sorted(THERMAL_EXTS):
                    cand1 = tdir / f"{rgb_path.stem}_thermal{e}"
                    if cand1.exists():
                        return cand1
                    cand2 = tdir / f"{rgb_path.stem}{e}"
                    if cand2.exists():
                        return cand2
                # sidecar next to RGB
                for e in sorted(THERMAL_EXTS):
                    cand = rgb_path.with_name(f"{rgb_path.stem}_thermal{e}")
                    if cand.exists():
                        return cand
                return None

            from PIL import Image
            import numpy as np
            for p in sorted(images_dir.iterdir()):
                if not p.is_file() or p.suffix.lower() not in exts:
                    continue
                tpath = _locate_thermal_for_rgb(p)
                if tpath is None:
                    continue
                # read + normalize thermal preview
                try:
                    g8 = normalize_thermal(tpath)
                except Exception:
                    # fallback: try reading as uint8 image directly
                    try:
                        from PIL import Image as _Image
                        g8 = np.array(_Image.open(tpath).convert('L')).astype(np.uint8)
                    except Exception:
                        continue
                # write a 3-channel RGB file matching original filename
                try:
                    if g8.ndim == 2:
                        rgb = np.stack([g8, g8, g8], axis=2)
                    else:
                        rgb = g8[..., :3]
                    outp = tmp / p.name
                    Image.fromarray(rgb).save(str(outp))
                except Exception:
                    continue
            # call RGB predictor on the temp folder
            return predict_folder_rgb(images_dir=tmp, weights_dir=weights, out_dir=out_dir, score_thresh=score_thresh)
        elif use_thermal and is_rotated_images:
            # rotated_images are already thermal-as-RGB, use directly
            log.info("UI:INFO:test: Using rotated_images (already thermal-as-RGB)")
            return predict_folder_rgb(images_dir=images_dir, weights_dir=weights, out_dir=out_dir, score_thresh=score_thresh)
        else:
            return predict_folder_rgb(images_dir=images_dir, weights_dir=weights, out_dir=out_dir, score_thresh=score_thresh)


    def read_meta(self, weights_dir: Path) -> dict:
        return load_model_meta(weights_dir)


# -------- registration helper (call this once at app startup) ------------- #

def register(registry_register_backend):
    registry_register_backend("detectron", lambda: DetectronBackend())
