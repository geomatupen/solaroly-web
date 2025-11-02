"""YOLO weights utilities: conversion helpers for channel surgery.

Provides a best-effort function to convert a YOLO/Ultralytics checkpoint so its
first Conv2d accepts a single input channel. The approach:

- Try to instantiate the ultralytics.YOLO object and extract its underlying
  torch.nn.Module state_dict (safe path when ultralytics is available).
- Locate the first Conv2d with in_channels==3, average its weights across the
  input channel dimension to produce a [out,1,k,k] kernel, replace in the
  state_dict, and save a new checkpoint that mirrors the original but with
  the modified "model" entry (or a raw state_dict if the original was such).

This is best-effort: if any step fails we fall back to returning the original
source path and leave training to proceed without conversion.
"""
from __future__ import annotations

from pathlib import Path
import logging
import torch
from typing import Optional

log = logging.getLogger("pvrt")


def _find_and_convert_state_dict(state_dict: dict) -> Optional[dict]:
    """Given a state_dict-like mapping, produce a converted copy where the
    first Conv2d with in_channels==3 has been replaced by a 1-channel kernel
    (averaged across RGB channels). Returns new state_dict or None on failure."""
    try:
        # heuristic: find a weight tensor that looks like conv weight [out,3,k,k]
        for k, v in list(state_dict.items()):
            if isinstance(v, torch.Tensor) and v.ndim == 4 and v.shape[1] == 3:
                w = v
                avg = w.mean(dim=1, keepdim=True)  # [out,1,k,k]
                new_state = dict(state_dict)
                new_state[k] = avg
                log.info(f"YOLO weights: converted conv '{k}' from 3->1 channels")
                return new_state
    except Exception:
        log.exception("Failed to convert state_dict for 1-channel")
    return None


def convert_yolo_checkpoint_to_1ch(src_path: Path, dst_path: Path) -> Path:
    """Convert a YOLO/ultralytics checkpoint at `src_path` into a 1-channel
    first-conv variant saved at `dst_path`. Returns the path to use (dst_path
    on success, else src_path).
    """
    src = Path(src_path)
    dst = Path(dst_path)
    if not src.exists():
        log.warning(f"YOLO weights conversion: source {src} does not exist; skipping conversion")
        return src

    # Try to load using torch first (handles local .pt files)
    try:
        ck = torch.load(str(src), map_location="cpu")
    except Exception:
        log.exception("YOLO weights conversion: torch.load failed; aborting conversion")
        return src

    # Case 1: ck is dict with 'model' key (ultralytics style)
    try:
        if isinstance(ck, dict) and "model" in ck and isinstance(ck["model"], dict):
            new_model = _find_and_convert_state_dict(ck["model"])
            if new_model is None:
                log.warning("YOLO weights conversion: no suitable conv found in 'model' state_dict; skipping")
                return src
            ck2 = dict(ck)
            ck2["model"] = new_model
            dst.parent.mkdir(parents=True, exist_ok=True)
            torch.save(ck2, str(dst))
            log.info(f"YOLO weights conversion: wrote converted checkpoint to {dst}")
            return dst

        # Case 2: ck itself looks like a state_dict
        if isinstance(ck, dict):
            new_model = _find_and_convert_state_dict(ck)
            if new_model is None:
                log.warning("YOLO weights conversion: no suitable conv found in checkpoint state_dict; skipping")
                return src
            dst.parent.mkdir(parents=True, exist_ok=True)
            torch.save(new_model, str(dst))
            log.info(f"YOLO weights conversion: wrote converted state_dict to {dst}")
            return dst

    except Exception:
        log.exception("YOLO weights conversion: unexpected error while converting checkpoint")
        return src

    # Fallback: couldn't interpret checkpoint
    log.warning("YOLO weights conversion: checkpoint format not recognized; skipping conversion")
    return src
