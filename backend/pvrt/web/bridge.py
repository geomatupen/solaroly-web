# backend/pvrt/web/bridge.py
"""
Web bridge: minimal glue between FastAPI routes and ML backends.

- Chooses the backend (defaults to 'detectron' unless specified)
- Applies the thermal-use rules consistently for train/test
- Returns simple paths your API can serialize

Keep this module tiny so app.py remains human-readable.
"""

from __future__ import annotations
from pathlib import Path
from typing import Literal, Optional, Dict, Any

from ..core.registry import get_backend, TrainConfig, PredictConfig
from ..core.io import load_model_meta, input_mode_from_meta, has_thermal_for_images


BackendName = Literal["detectron", "yolo"]  # extend as you add more


def train_entry(
    *,
    backend: BackendName = "detectron",
    train_dir: Path,
    val_dir: Path,
    out_dir: Path,
    use_thermal_request: bool,
    max_iter: int,
    base_lr: float,
    ims_per_batch: int,
    run_name: str = "",
) -> dict:
    """
    Decide RGB vs RGBT (based on data availability + user request), then train.

    Returns a dict with:
      - run_dir: Path
      - meta: dict (model_meta.json contents after training)
    """
    train_dir = Path(train_dir)
    val_dir = Path(val_dir)
    out_dir = Path(out_dir)

    # Only enable thermal if user asked AND thermal exists in training set
    thermal_ok = bool(use_thermal_request and has_thermal_for_images(train_dir))

    backend_impl = get_backend(backend)
    run_dir = backend_impl.train(
        TrainConfig(
            train_dir=train_dir,
            val_dir=val_dir,
            out_dir=out_dir,
            use_thermal=thermal_ok,
            max_iter=int(max_iter),
            base_lr=float(base_lr),
            ims_per_batch=int(ims_per_batch),
            run_name=run_name,
        )
    )
    meta = load_model_meta(run_dir)
    return {"run_dir": run_dir, "meta": meta}


def predict_entry(
    *,
    weights_dir: Path,
    images_dir: Path,
    out_dir: Path,
    use_thermal_request: bool,
    forced_backend: Optional[BackendName] = None,
) -> dict:
    """
    Decide whether to run RGBT or RGB inference:

      Use thermal ONLY if:
        - user requested thermal, AND
        - images have thermal sidecars, AND
        - the model was trained with thermal (meta: input_mode == 'rgbt')

    Returns a dict with:
      - results_dir: Path
      - used_backend: str
      - model_mode: 'rgb' | 'rgbt'
    """
    weights_dir = Path(weights_dir)
    images_dir = Path(images_dir)
    out_dir = Path(out_dir)

    # Read model meta to discover backend + input_mode
    meta = load_model_meta(weights_dir)
    model_mode = input_mode_from_meta(meta, default="rgb")

    # Pick backend: prefer the one that trained the model; allow override
    backend_name = (forced_backend or meta.get("backend") or "detectron").strip().lower()  # type: ignore
    backend_impl = get_backend(backend_name)  # raises KeyError if unknown

    # Thermal decision for prediction
    has_thermal = has_thermal_for_images(images_dir)
    use_thermal = bool(use_thermal_request and has_thermal and model_mode == "rgbt")

    results_dir = backend_impl.predict(
        PredictConfig(
            images_dir=images_dir,
            out_dir=out_dir,
            weights_dir=weights_dir,
            use_thermal=use_thermal,
        )
    )
    return {
        "results_dir": results_dir,
        "used_backend": backend_name,
        "model_mode": model_mode,
        "used_thermal": use_thermal,
    }
