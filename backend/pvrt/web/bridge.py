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
import logging

from ..core.registry import get_backend, TrainConfig, PredictConfig
from ..core.io import load_model_meta, input_mode_from_meta, has_thermal_for_images

BackendName = Literal["detectron", "yolo"]  # extend as you add more

_log_test = logging.getLogger("pvrt.test")   # mini-log (SSE panel)
_log_full = logging.getLogger("pvrt")        # full logs tab


def _select_infer_mode(
    use_thermal_request: bool,
    images_dir: Path,
    model_mode: str,
) -> tuple[bool, str | None]:
    """
    Decide whether to run RGB or RGB+Thermal.

    Returns:
      (use_thermal, reason)
      - use_thermal: True - rgbt, False - rgb
      - reason: non-empty only when falling back to rgb
    """
    data_has_thermal = has_thermal_for_images(images_dir)
    model_is_rgbt = model_mode in {"rgbt", "rgb+thermal", "thermal", "rgb_thermal", "4ch"}

    if use_thermal_request and data_has_thermal and model_is_rgbt:
        # OK to use thermal
        _log_test.info(
            "UI:INFO:test: decision: use_thermal_request=True, data_has_thermal=True, model_mode=rgbt - rgbt"
        )
        return True, None

    # Fallback reasons (kept simple & explicit)
    if not use_thermal_request:
        reason = "request_false"
    elif not data_has_thermal:
        reason = "no_thermal_in_dataset"
    else:
        reason = f"model_mode={model_mode!r}"  # model not trained for thermal

    _log_test.warning(f"UI:WARN:test: decision: FALLBACK to RGB (reason={reason})")
    return False, reason


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
    _log_full.info(
        f"UI:INFO:train: decision: use_thermal_request={use_thermal_request}, "
        f"data_has_thermal={thermal_ok} - {'rgbt' if thermal_ok else 'rgb'}"
    )

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
      - used_thermal: bool
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

    # Thermal decision for prediction (with clear mini-log line)
    use_thermal, _reason = _select_infer_mode(use_thermal_request, images_dir, model_mode)

    # (Optional) backend selection info (helps when override is used)
    if forced_backend and forced_backend != meta.get("backend"):
        _log_full.info(f"UI:INFO:test: backend override: forced={forced_backend} meta={meta.get('backend')} - using {backend_name}")

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
