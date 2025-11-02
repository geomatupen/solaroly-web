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


def _select_infer_mode(use_thermal_request: bool, data_has_thermal: bool, model_mode: str):
    model_is_rgbt = model_mode in {"rgbt","rgb+thermal","thermal","rgb_thermal","4ch"}
    if use_thermal_request and data_has_thermal and model_is_rgbt:
        _log_test.info("UI:INFO:test: decision: use_thermal_request=True, data_has_thermal=True, model_mode=rgbt - rgbt")
        return True, None
    # fallback reason
    if not use_thermal_request: reason = "request_false"
    elif not data_has_thermal:  reason = "no_thermal_in_dataset"
    else:                       reason = f"model_mode={model_mode!r}"
    _log_test.warning(f"UI:INFO:test: decision: FALLBACK to RGB (reason={reason})")
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
    model_type: str = "maskrcnn",
    yolo_family: str = "v8",
    yolo_seg: bool = False,
    yolo_size: str = "s",
    selected_bands: list | None = None,
    channel_count: int = 3,
    augment_options: dict | None = None,
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
        f"data_has_thermal={thermal_ok} - {'rgbt' if thermal_ok else 'rgb'}, learning rate = {base_lr}"
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
            model_type=model_type,
            yolo_family=yolo_family,
            yolo_seg=bool(yolo_seg),
            yolo_size=str(yolo_size),
            selected_bands=selected_bands,
            channel_count=int(channel_count),
            augment_options=augment_options,
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
    score_thresh_frontend: Optional[float] = None,
    data_has_thermal_override: Optional[bool] = None,   # <-- keep this
    selected_bands: list | None = None,
    channel_count: int = 3,
) -> dict:
    meta = load_model_meta(weights_dir)
    model_mode = input_mode_from_meta(meta, default="rgb")
    backend_name = (forced_backend or meta.get("backend") or "detectron").strip().lower()
    backend_impl = get_backend(backend_name)

    if data_has_thermal_override is not None:
        data_has_thermal = bool(data_has_thermal_override)
    else:
        from ..core.io import has_thermal_for_images
        data_has_thermal = has_thermal_for_images(images_dir)

    # If the caller requested thermal but we don't see thermal files yet, attempt
    # an idempotent decode pass here. This mirrors the behavior in app.py but
    # ensures any caller of predict_entry (CLI, API, tests) will attempt to
    # populate images_dir/thermal when RJPEG payloads exist. Always catch
    # exceptions so a missing DIRP library won't fail the whole predict call.
    decode_stats = None
    if use_thermal_request and not data_has_thermal:
        try:
            # local import to avoid heavy DJI SDK import at module-load time
            from ..dataops.scan_decode_split import ensure_dirp_init, scan_split_decode_thermal

            _log_test.info(f"UI:INFO:test: attempting on-demand thermal decode: {images_dir}")
            try:
                ensure_dirp_init()
            except Exception as e:
                _log_full.warning(f"UI:WARN:test: DIRP init failed during on-demand decode: {e}")
            try:
                pairs_path, stats = scan_split_decode_thermal(images_dir)
                decode_stats = stats
                _log_test.info(f"UI:INFO:test: on-demand thermal decode summary -> {stats}")
            except Exception as e:
                _log_full.warning(f"UI:WARN:test: on-demand thermal decode error: {e}")
        except Exception as e:
            # If importing the decoder fails for any reason, log and continue.
            _log_full.warning(f"UI:WARN:test: could not run on-demand decode: {e}")

        # Re-evaluate presence of thermal files after the decode attempt
        try:
            from ..core.io import has_thermal_for_images
            data_has_thermal = has_thermal_for_images(images_dir)
        except Exception:
            data_has_thermal = False

    use_thermal, _ = _select_infer_mode(use_thermal_request, data_has_thermal, model_mode)

    chosen_thresh = float(score_thresh_frontend) if score_thresh_frontend is not None else float(meta.get("score_thresh_test", 0.5))
    _log_test.info(f"UI:INFO:test: Use thermal in bridge.py={use_thermal_request} , data_has_thermal={data_has_thermal}, model_mode={model_mode}")
    _log_test.info(f"UI:INFO:test: threshold={chosen_thresh:.3f} (source={'frontend' if score_thresh_frontend is not None else 'meta'})")

    # Determine the channel_count to request for inference.
    # Priority: explicit caller `channel_count` argument (when non-default), else use model's recorded channel_count, else default to 3.
    try:
        model_chan = int(meta.get("channel_count", 0) or 0)
    except Exception:
        model_chan = 0
    # If caller passed explicit (non-None) channel_count param, prefer it; otherwise prefer model metadata
    final_channel_count = int(channel_count) if channel_count is not None and int(channel_count) != 3 else (model_chan or 3)

    _log_test.info(f"UI:INFO:test: channel selection -> frontend_request={channel_count}, model_trained={model_chan or 'unknown'} -> final_channel_count={final_channel_count}")

    results_dir = backend_impl.predict(
        PredictConfig(
            images_dir=images_dir,
            out_dir=out_dir,
            weights_dir=weights_dir,
            use_thermal=use_thermal,       # backend obeys this boolean
            score_thresh=chosen_thresh,
            selected_bands=selected_bands,
            channel_count=final_channel_count,
        )
    )
    return {
        "results_dir": results_dir,
        "used_backend": backend_name,
        "model_mode": model_mode,
        "used_thermal": use_thermal,
        "used_channel_count": int(final_channel_count),
        "score_thresh": chosen_thresh,
        "decode_stats": decode_stats,
    }


