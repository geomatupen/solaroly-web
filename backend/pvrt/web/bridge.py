# backend/pvrt/web/bridge.py
"""
Web bridge: minimal glue between FastAPI routes and ML backends.

- Chooses the backend (defaults to 'detectron' unless specified)
- Applies the thermal-use rules consistently for train/test
- Returns simple paths the API can serialize

Keep this module tiny so app.py remains human-readable.
"""

from __future__ import annotations
from pathlib import Path
from typing import Literal, Optional, Dict, Any
import logging

from ..core.registry import get_backend, TrainConfig, PredictConfig
from ..core.io import load_model_meta, input_mode_from_meta, has_thermal_for_images

BackendName = Literal["detectron", "yolo"]  # extend when adding more backends

_log_test = logging.getLogger("pvrt.test")   # mini-log (SSE panel)
_log_full = logging.getLogger("pvrt")        # full logs tab


def _select_infer_mode(
    use_thermal_request: bool,
    data_has_thermal: bool,
    model_mode: str,
    model_thermal_used: bool = False,
):
    """
    Decide whether to use thermal at inference time.

    A model may be recorded as 3-channel RGB in `input_mode` but still have
    been trained using decoded thermal images (thermal_as_rgb). Such models
    set `thermal_used` in their metadata. Treat `thermal_used=True` as an
    indicator the model can accept thermal-as-RGB even when `input_mode` is
    'rgb'.
    """
    model_is_thermal = model_mode == "thermal" or bool(model_thermal_used)
    # Allow thermal when either the frontend explicitly requested it OR
    # the model was trained with thermal (`thermal_used=True`) — provided
    # the dataset has decoded thermal previews. This makes models saved as
    # 3-channel but trained with thermal automatically use the thermal
    # previews at test time unless explicitly overridden by the caller.
    if (use_thermal_request or bool(model_thermal_used)) and data_has_thermal and model_is_thermal:
        _log_test.info(
            "decision: using thermal path (model_mode=%s, thermal_used=%s, request=%s)",
            model_mode,
            bool(model_thermal_used),
            bool(use_thermal_request),
        )
        return True, None
    # fallback reason
    if not use_thermal_request:
        reason = "request_false"
    elif not data_has_thermal:
        reason = "no_thermal_in_dataset"
    else:
        reason = f"model_mode={model_mode!r} and thermal_used={bool(model_thermal_used)}"
    _log_test.warning("decision: falling back to RGB (reason=%s)", reason)
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
    task: str = "detect",
    model_type: str = "fasterrcnn",
    yolo_family: str = "v8",
    yolo_seg: bool = False,
    yolo_size: str = "s",
    selected_bands: list | None = None,
    channel_count: int = 3,
    augment_options: dict | None = None,
    dataset_id: str = "",
    dataset_name: str = "",
    dataset_path: str = "",
    dataset_format: str = "",
    dataset_yaml: Path | None = None,
) -> dict:
    """
    Decide RGB vs thermal-as-RGB (based on data availability + user request), then train.

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
        f"INFO:train: decision: use_thermal_request={use_thermal_request}, "
        f"data_has_thermal={thermal_ok} - {'thermal' if thermal_ok else 'rgb'}, learning rate = {base_lr}"
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
            task=task,
            model_type=model_type,
            yolo_family=yolo_family,
            yolo_seg=bool(yolo_seg),
            yolo_size=str(yolo_size),
            selected_bands=selected_bands,
            channel_count=int(channel_count),
            augment_options=augment_options,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            dataset_format=dataset_format,
            dataset_yaml=dataset_yaml,
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
        # Special case: rotated_images from thermal sources are already thermal-as-RGB
        if not data_has_thermal and images_dir.name == "rotated_images":
            camera_meta_path = images_dir.parent / "camera_meta.json"
            if camera_meta_path.exists() and use_thermal_request:
                _log_test.info("UI:INFO:test: rotated_images detected, treating as thermal-ready")
                data_has_thermal = True

    use_thermal, _ = _select_infer_mode(
        use_thermal_request,
        data_has_thermal,
        model_mode,
        model_thermal_used=bool(meta.get("thermal_used", False)),
    )

    chosen_thresh = float(score_thresh_frontend) if score_thresh_frontend is not None else float(meta.get("score_thresh_test", 0.5))
    _log_test.info(f"UI:INFO:test: Use thermal in bridge.py={use_thermal_request} , data_has_thermal={data_has_thermal}, model_mode={model_mode}")
    _log_test.info(f"UI:INFO:test: threshold={chosen_thresh:.3f} (source={'frontend' if score_thresh_frontend is not None else 'meta'})")

    # Determine the channel_count to request for inference.
    # Priority: explicit caller `channel_count` argument (when non-default), else use model's recorded channel_count, else default to 3.
    try:
        model_chan = int(meta.get("channel_count", 0) or 0)
    except (TypeError, ValueError):
        model_chan = 0
    # If caller passed explicit (non-None) channel_count param, prefer it; otherwise prefer model metadata
    final_channel_count = int(channel_count) if channel_count is not None and int(channel_count) != 3 else (model_chan or 3)

    # Determine a human-friendly final mode string for logs
    # Only 3-channel is supported: either RGB or thermal-as-RGB
    if final_channel_count == 3:
        # prefer thermal when use_thermal is True and thermal data/models indicate it
        if use_thermal and (model_mode == 'thermal' or bool(meta.get('thermal_used', False))):
            final_mode = 'thermal'
        else:
            final_mode = 'rgb'
    else:
        final_mode = 'rgb'  # default to rgb if unexpected channel count

    _log_test.info(
        f"UI:INFO:test: channel selection -> frontend_request={channel_count}, model_trained={model_chan or 'unknown'}, data_has_thermal={data_has_thermal}, use_thermal={use_thermal} -> final_channel_count={final_channel_count} ({final_mode})"
    )

    # Emit a concise final-mode message so callers (and SSE clients) can
    # display the chosen mode directly.
    _log_test.info(f"UI:INFO:test: final_mode={final_mode}")

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
        "final_mode": final_mode,
        "score_thresh": chosen_thresh,
    }
