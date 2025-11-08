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
    # Determine requested channel count (caller intent). If thermal is not
    # available the requested channels are coerced to 3 to avoid confusing
        # log messages and to ensure the training preproc uses RGB-only lists.
        requested_channels = int(getattr(cfg_in, "channel_count", 3))
        # Single-channel models are not supported; coerce any '1' requests to 3.
        if requested_channels == 1:
            log.warning("requested_channels=1 is deprecated; coercing to 3")
            requested_channels = 3
        if not thermal_ok and requested_channels != 3:
            log.warning("requested_channels=%s but thermal_ok=%s; coercing to 3", requested_channels, thermal_ok)
            requested_channels = 3

        # Effective channels for training.
    # If thermal is available and enabled, honor the requested
    # channel_count when it is 3 (thermal provided as 3-channel grayscale).
    # If 4 is requested, use RGB+thermal (4). Single-channel models are
    # not supported; any '1' value was coerced earlier to 3.
        if thermal_ok and getattr(cfg_in, "use_thermal", False):
            if requested_channels == 3:
                effective_channels = 3
            else:
                # default/other requests (including explicit 4) -> RGB+thermal
                effective_channels = 4
        else:
            effective_channels = 3

        log.info("UI:INFO:train: backend=yolo | effective_channels=%s (requested=%s, thermal_ok=%s)", effective_channels, requested_channels, thermal_ok)
        # If thermal is available but the effective channels is 3, thermal
        # will be presented as grayscale encoded into 3 channels for training
        # (thermal-as-RGB).
        if thermal_ok and effective_channels == 3:
            log.info("UI:INFO:train: Thermal grayscale will be used for training (thermal-as-RGB)")

    # Do not create per-run prepared copies. Use the existing
    # `train_dir`/`val_dir` in-place. If thermal decoding is required it
    # should write into `thermal/` subfolders under those directories.

    # run_train is responsible for using ultralytics.YOLO to train and save artifacts
    # and should return a dict with at least {"best_weights": Path, "final_weights": Path}.
    # Ensure the training routine sees the effective channel layout
    # (e.g. 4 == RGB+thermal). Pass the computed effective_channels as
    # requested_channels to run_train to keep metadata consistent.
        res = run_train(
            train_dir=train_dir,
            val_dir=val_dir,
            out_dir=out_dir,
            use_thermal=thermal_ok,
            max_iter=cfg_in.max_iter,
            base_lr=cfg_in.base_lr,
            ims_per_batch=cfg_in.ims_per_batch,
            run_name=getattr(cfg_in, "run_name", ""),
            requested_channels=effective_channels,
        )

        # Normalize and write model_meta.json (keep keys compatible with Detectron meta)
        model_name = res.get("model_name", "yolo")
        model_zoo = getattr(cfg_in, "yolo_family", "v8")
        # append channel suffix to rgbt models
        # Record the actual effective channel count in the saved meta so
        # downstream components (predict) interpret the model correctly.
        ch = int(effective_channels)
    # Only append a _4ch suffix when the model includes thermal as an
    # extra channel. Single-channel model names are not created.
        if ch == 4:
            model_name = f"{model_name}_4ch"
        # prepend the run name (if provided) so the UI shows runs similarly to Detectron
        run_prefix = getattr(cfg_in, "run_name", "") or out_dir.name
        # avoid double underscores when run_prefix is empty
        prefix_part = f"{run_prefix}_" if run_prefix else ""

        # Helper: when ultralytics returns only a weight filename we want to store
        # the path relative to the run folder (e.g. "train_.../model_best.pt") so
        # the frontend can locate `model_meta.json` and other artifacts by run.
        def _make_run_path(candidate_weights, fallback_model_obj):
            # candidate_weights: path-like string (maybe nested) returned by run_train
            # fallback_model_obj: dict from res.get('best_model'|'final_model') that
            # may already contain a .get('path') value; prefer explicit run-relative
            # path if we can compute it from candidate_weights.
            if candidate_weights:
                name = Path(candidate_weights).name
                return f"{run_prefix}/{name}" if run_prefix else name
            # if ultralytics or the extractor already provided a path, keep it
            if isinstance(fallback_model_obj, dict):
                p = fallback_model_obj.get("path")
                if p:
                    return p
            return ""

        meta = {
            "backend": "yolo",
            "input_mode": "rgbt" if ch == 4 else "rgb",
            # Record whether this model was trained with thermal data present
            # and enabled. For some runs thermal may be used but the effective
            # channel count is 3 (thermal-as-RGB); this flag preserves that
            # information for correct test-time selection.
            "thermal_used": bool(thermal_ok and getattr(cfg_in, "use_thermal", False)),
            "selected_bands": getattr(cfg_in, "selected_bands", None),
            "channel_count": ch,
            # model_name includes the training run prefix + base model + zoo
            "model_name": f"{prefix_part}{model_name}-{model_zoo}",
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
            # include full best/final stats (iter, val_bbox_AP50, loss stats, path)
            "best_model": {
                "iter": res.get("best_model", {}).get("iter") if isinstance(res.get("best_model", {}), dict) else None,
                "val_bbox_AP50": res.get("best_model", {}).get("val_bbox_AP50") if isinstance(res.get("best_model", {}), dict) else None,
                "total_loss_med20": res.get("best_model", {}).get("total_loss_med20") if isinstance(res.get("best_model", {}), dict) else None,
                "total_loss_raw": res.get("best_model", {}).get("total_loss_raw") if isinstance(res.get("best_model", {}), dict) else None,
                "path": _make_run_path(res.get("best_weights"), res.get("best_model", {})),
            },
            "final_model": {
                "iter": res.get("final_model", {}).get("iter") if isinstance(res.get("final_model", {}), dict) else None,
                "val_bbox_AP50": res.get("final_model", {}).get("val_bbox_AP50") if isinstance(res.get("final_model", {}), dict) else None,
                "total_loss_med20": res.get("final_model", {}).get("total_loss_med20") if isinstance(res.get("final_model", {}), dict) else None,
                "total_loss_raw": res.get("final_model", {}).get("total_loss_raw") if isinstance(res.get("final_model", {}), dict) else None,
                "path": _make_run_path(res.get("final_weights"), res.get("final_model", {})),
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
        # Preserve whether this model was trained with thermal data even when
        # the recorded input_mode may be 'rgb' (e.g. thermal-as-RGB trained as 3ch).
        model_trained_with_thermal = bool(meta.get("thermal_used", False))

        score_thresh = float(cfg_in.score_thresh) if cfg_in.score_thresh is not None else float(meta.get("score_thresh_test", 0.25))

        # choose rgbt vs rgb based on user request and model capability.
        # Check whether there are decoded thermal files in the images dir.
        has_thermal = bool(has_thermal_for_images(images_dir))
        # If the model was trained with thermal data but saved as 3-channel
        # (thermal-as-RGB), prefer decoded thermal when the user requests
        # thermal inference and the dataset contains thermal files.
        use_thermal = bool(
            cfg_in.use_thermal and (model_mode == "rgbt" or model_trained_with_thermal) and has_thermal
        )

        # If the user explicitly requested thermal but no thermal files are
        # present, warn that we'll fall back to RGB (frontend watches UI:WARN)
        if cfg_in.use_thermal and not has_thermal:
            log.warning(f"UI:WARN:test: requested thermal but no decoded thermal files found in {images_dir}; falling back to rgb")

        try:
            requested = int(getattr(cfg_in, "channel_count", 3))
        except (TypeError, ValueError):
            requested = 3

        # Mirror training logic for test-time: if user requested 3-channel
        # thermal (grayscale-as-RGB), report 3 channels; if requested==1,
        # treat as 3; otherwise use RGB+thermal (4).
        if use_thermal:
            if requested == 1:
                effective_channels_test = 3
            elif requested == 3:
                effective_channels_test = 3
            else:
                effective_channels_test = 4
        else:
            effective_channels_test = 3

        model_chan = int(meta.get("channel_count", 0) or 0)

        # Clarify what "3-channel" means in logs: rgb vs thermal-as-RGB
        if effective_channels_test == 4:
            selected_mode = 'rgbt'
        elif effective_channels_test == 3:
            selected_mode = 'thermal' if use_thermal else 'rgb'
        else:
            selected_mode = 'rgb'

        log.info(
            f"UI:INFO:test: backend=yolo | selected={selected_mode} | model_trained={model_chan or 'unknown'} | requested={requested} | effective_channels={effective_channels_test} | score_thresh={score_thresh:.3f}"
        )
        if effective_channels_test == 3 and use_thermal:
            log.info("UI:INFO:test: Thermal grayscale will be used for testing (thermal)")

        return predict_folder(
            images_dir=images_dir,
            weights_dir=weights,
            out_dir=out_dir,
            score_thresh=score_thresh,
            use_thermal=use_thermal,
            channel_count=effective_channels_test,
        )

    def read_meta(self, weights_dir: Path) -> dict:
        return load_model_meta(weights_dir)


def register(registry_register_backend):
    registry_register_backend("yolo", lambda: YOLOBackend())
