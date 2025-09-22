# backend/pvrt/backends/detectron/backend.py
"""
Detectron backend adapter
-------------------------
Implements the generic Backend interface so the web layer doesn't need to care
whether the engine under the hood is Detectron, YOLO, etc.

Key behaviors preserved:
- Train with RGB+Thermal when both (a) user requested and (b) data has thermal
- Otherwise train RGB only
- For testing: if user requested thermal AND model supports it AND data has thermal
  -> test with thermal; else fallback to RGB
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
)

# your existing modules
from .train.datasets import register_split_coco, _find_coco_json
from .train.trainer_rgb_only import RGBOnlyTrainer
from .train.trainer_rgb_thermal_tolerant import RTolerantTrainer
from .infer.predict_rgb_only import predict_folder as predict_folder_rgb
from .infer.predict_rgb_thermal import predict_folder as predict_folder_rgbt

# detectron2 bits
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
from detectron2.engine import hooks
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.logger import setup_logger

from shutil import copy2
import math

# use your previous helpers.py to read class count from COCO (source of truth)
from ...core.helpers import get_num_classes

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

        # Decide if this run should be 4-channel
        thermal_ok = bool(cfg_in.use_thermal and has_thermal_for_images(train_dir))

        # Register datasets (idempotent)
        register_split_coco("pv_train", train_dir)
        register_split_coco("pv_val",   val_dir)

        # --- num_classes from COCO (your helpers.py) ---
        ann_json = _find_coco_json(train_dir)
        num_classes = int(get_num_classes(ann_json))

        # class_names from MetadataCatalog (Detectron populates this at registration)
        try:
            meta_train = MetadataCatalog.get("pv_train")
            class_names = list(meta_train.thing_classes) if getattr(meta_train, "thing_classes", None) else []
        except Exception:
            class_names = []

        # fallback to COCO if still empty (prevents blank in thermal)
        if not class_names:
            import json
            ann_json = _find_coco_json(train_dir)
            data = json.loads(Path(ann_json).read_text(encoding="utf-8"))
            cats = data.get("categories", []) if isinstance(data.get("categories", []), list) else []
            # preserve label order by id when present
            try:
                cats = sorted(cats, key=lambda c: int(c.get("id", 0)))
            except Exception:
                pass
            class_names = [str(c.get("name", f"class_{i}")) for i, c in enumerate(cats)]

        # Build cfg
        MODEL_YAML = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(MODEL_YAML))
        cfg.MODEL.MASK_ON = False  

        cfg.DATASETS.TRAIN = ("pv_train",)
        cfg.DATASETS.TEST  = ("pv_val",)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL_YAML)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

        cfg.SOLVER.IMS_PER_BATCH = int(cfg_in.ims_per_batch or 2)
        cfg.SOLVER.BASE_LR       = float(cfg_in.base_lr or 0.00025)
        cfg.SOLVER.MAX_ITER      = int(cfg_in.max_iter or 1000)
        cfg.SOLVER.STEPS         = list(_safe_solver_steps(cfg.SOLVER.MAX_ITER))
        # cfg.SOLVER.CHECKPOINT_PERIOD = max(1001, int(cfg.SOLVER.MAX_ITER / 6))
        cfg.SOLVER.CHECKPOINT_PERIOD = 10**9  # disabled time-based checkpoints
        cfg.SOLVER.LOG_PERIOD    = 1

        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
        cfg.TEST.EVAL_PERIOD = 50 # max(50, int(cfg.SOLVER.MAX_ITER // 10))
        cfg.OUTPUT_DIR = str(out_dir)
        cfg.INPUT.FORMAT = "BGR"

        # --- critical: make resize settings coherent to avoid 'range'+[800] ---
        _coherent_input_resize(cfg)

        # Train
        setup_logger()
        trainer = RTolerantTrainer(cfg) if thermal_ok else RGBOnlyTrainer(cfg)
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
                    logging.getLogger("pvrt.test").info(f"UI:LOG:loss: iter={self.trainer.iter} total_loss(raw)={raw:.4f}")
        # trainer.register_hooks([_RawLossLogger()])


        # capture last seen total_loss from EventStorage
        class _LossTap(hooks.HookBase):
            def __init__(self):
                self.last_raw = None
                self.last_med20 = None  # matches what the console prints (20-iter median)

            def after_step(self):
                try:
                    hb = self.trainer.storage.history("total_loss")   # HistoryBuffer
                    s = hb.latest()                                   # Scalar or float
                    self.last_raw = float(getattr(s, "value", s))
                    try:
                        self.last_med20 = float(hb.median(20))
                    except Exception:
                        self.last_med20 = self.last_raw
                except Exception:
                    pass


        loss_tap = _LossTap()
        trainer.register_hooks([loss_tap])

        # 2) build val loader/evaluator once
        val_loader = (RTolerantTrainer if thermal_ok else RGBOnlyTrainer).build_test_loader(cfg, "pv_val")
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
            res = inference_on_dataset(trainer.model, val_loader, evaluator)

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
                        "path": str(Path(out_dir) / "model_best.pth"),
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
        try:
            best = Path(out_dir) / "model_best.pth"
            if best.exists():
                copy2(best, Path(out_dir) / "model_final.pth")
                log.info("PHASE:save model_final <- model_best")
            else:
                trainer.checkpointer.save("model_final")
                log.info("PHASE:save model_final (no best found)")
        except Exception as e:
            log.warning(f"PHASE:save FAILED (non-fatal): {e}")
        finally:
            final_ap50_pct = _latest["ap50_pct"]
            if final_ap50_pct is None:
                try:
                    s = trainer.storage.history("val/AP50_pct").latest()
                    final_ap50_pct = float(getattr(s, "value", s))
                except Exception:
                    final_ap50_pct = None

            _normalize_and_save_meta(out_dir, {
                "final_model": {
                    "iter": int(trainer.iter),
                    "val_bbox_AP50": None if final_ap50_pct is None else round(final_ap50_pct, 4),
                    "total_loss_med20": None if loss_tap.last_med20 is None else float(loss_tap.last_med20),
                    "total_loss_raw":   None if loss_tap.last_raw   is None else float(loss_tap.last_raw),
                    "path": str(Path(out_dir) / "model_final.pth"),
                }
            })

        
        MODEL_NAME = Path(MODEL_YAML).stem
        # Save normalized meta for the run
        _normalize_and_save_meta(out_dir, {
            "backend": "detectron",
            "input_mode": "rgbt" if thermal_ok else "rgb",
            "model_name": MODEL_NAME, 
            "model_zoo": MODEL_YAML,
            "num_classes": num_classes,
            "class_names": class_names,
            "score_thresh_test": float(cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST),
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
        Test with thermal only if:
        - user requested thermal, AND
        - images have thermal sidecars, AND
        - the model was trained with thermal (meta: input_mode == 'rgbt')
        Otherwise, fallback to RGB predictor.
        """
        images_dir = Path(cfg_in.images_dir)
        out_dir    = Path(cfg_in.out_dir)
        weights    = Path(cfg_in.weights_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        meta = load_model_meta(weights)
        model_mode = input_mode_from_meta(meta, default="rgb").lower().strip()

        # final score threshold (cfg_in wins; else meta; else 0.5)
        score_thresh = (
            float(cfg_in.score_thresh)
            if cfg_in.score_thresh is not None
            else float(meta.get("score_thresh_test", 0.5))
        )

        # break the conditions out so we can log a precise reason
        request_thermal   = bool(cfg_in.use_thermal)
        data_has_thermal  = has_thermal_for_images(images_dir)
        model_is_rgbt     = model_mode in {"rgbt", "rgb+thermal", "thermal", "rgb_thermal", "4ch"}

        if request_thermal and data_has_thermal and model_is_rgbt:
            # log.info("UI:INFO:test: decision: use_thermal_request=True, data_has_thermal=True, model_mode=rgbt - rgbt")
            log.info("UI:INFO:test: backend=detectron | selected=rgbt | score_thresh={score_thresh:.3f}")
            return predict_folder_rgbt(
                images_dir=images_dir,
                weights_dir=weights,
                out_dir=out_dir,
                score_thresh=score_thresh,
            )

        # --- fallback to RGB; compute a clear reason for the mini-log ---
        if not request_thermal:
            reason = "request_false"
        elif not data_has_thermal:
            reason = "no_thermal_in_dataset"
        else:
            reason = f"model_mode={model_mode!r}"  # model not trained for thermal

        log.warning(f"UI:WARN:test: decision: FALLBACK to RGB (reason={reason})")
        log.info("UI:INFO:test: backend=detectron | selected=rgb")
        return predict_folder_rgb(
            images_dir=images_dir,
            weights_dir=weights,
            out_dir=out_dir,
            score_thresh=score_thresh, 
        )


    def read_meta(self, weights_dir: Path) -> dict:
        return load_model_meta(weights_dir)


# -------- registration helper (call this once at app startup) ------------- #

def register(registry_register_backend):
    registry_register_backend("detectron", lambda: DetectronBackend())
