"""backend/pvrt/backends/yolo/backend.py

YOLO backend adapter
---------------------
Implements the Backend protocol so the web layer can call YOLO similarly to
Detectron. Uses the ultralytics.YOLO API to train and run predictions.

This adapter follows the same semantics as the Detectron backend:
- respects rgb vs rgbt (thermal) when requested and available
- writes a `model_meta.json` with normalized input_mode and train params
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

from ...core.registry import Backend, TrainConfig, PredictConfig
from ...core.io import (
    load_model_meta,
    save_model_meta,
    input_mode_from_meta,
    backend_name_from_meta,
    has_thermal_for_images,
)

from .train import run_train
from .infer import predict_folder

import json
import logging

log = logging.getLogger("pvrt")


class YOLOBackend(Backend):
    def train(self, cfg_in: TrainConfig) -> Path:
        train_dir = Path(cfg_in.train_dir)
        val_dir = Path(cfg_in.val_dir)
        out_dir = Path(cfg_in.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        thermal_ok = bool(cfg_in.use_thermal and has_thermal_for_images(train_dir))

        # run_train is responsible for using ultralytics.YOLO to train and save artifacts
        # it should return a dict with at least {"best_weights": Path, "final_weights": Path}
        res = run_train(
            train_dir=train_dir,
            val_dir=val_dir,
            out_dir=out_dir,
            use_thermal=thermal_ok,
            max_iter=cfg_in.max_iter,
            base_lr=cfg_in.base_lr,
            ims_per_batch=cfg_in.ims_per_batch,
            run_name=getattr(cfg_in, "run_name", ""),
        )

        # Normalize and write model_meta.json (keep keys compatible with Detectron meta)
        model_name = res.get("model_name", "yolo")
        model_zoo = getattr(cfg_in, "yolo_family", "v8")
        meta = {
            "backend": "yolo",
            "input_mode": "rgbt" if thermal_ok else "rgb",
            "model_name": f"{model_name}-{model_zoo}",
            "model_zoo": model_zoo,
            "num_classes": int(res.get("num_classes", 0)),
            "class_names": res.get("class_names", []),
            "score_thresh_test": float(res.get("score_thresh_test", 0.25)),
            "train_params": {
                "max_iter": int(cfg_in.max_iter or 0),
                "base_lr": float(cfg_in.base_lr or 0.0),
                "ims_per_batch": int(cfg_in.ims_per_batch or 0),
                "run_name": getattr(cfg_in, "run_name", ""),
            },
            "best_model": {
                "path": str(Path(res.get("best_weights", "")).name) if res.get("best_weights") else "",
            },
            "final_model": {
                "path": str(Path(res.get("final_weights", "")).name) if res.get("final_weights") else "",
            },
        }
        save_model_meta(out_dir, meta)
        return out_dir

    def predict(self, cfg_in: PredictConfig) -> Path:
        images_dir = Path(cfg_in.images_dir)
        out_dir = Path(cfg_in.out_dir)
        weights = Path(cfg_in.weights_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        meta = load_model_meta(weights)
        model_mode = input_mode_from_meta(meta, default="rgb").lower().strip()

        score_thresh = float(cfg_in.score_thresh) if cfg_in.score_thresh is not None else float(meta.get("score_thresh_test", 0.25))

        # choose rgbt vs rgb based on user request and model capability
        use_thermal = bool(cfg_in.use_thermal and model_mode == "rgbt" and has_thermal_for_images(images_dir))

        log.info(f"UI:INFO:test: backend=yolo | selected={'rgbt' if use_thermal else 'rgb'} | score_thresh={score_thresh:.3f}")
        return predict_folder(images_dir=images_dir, weights_dir=weights, out_dir=out_dir, score_thresh=score_thresh, use_thermal=use_thermal)

    def read_meta(self, weights_dir: Path) -> dict:
        return load_model_meta(weights_dir)


def register(registry_register_backend):
    registry_register_backend("yolo", lambda: YOLOBackend())
