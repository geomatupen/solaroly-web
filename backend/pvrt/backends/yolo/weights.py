"""YOLO weights utilities: conversion helpers for channel surgery.

Provides a best-effort function to convert a YOLO/Ultralytics checkpoint so its
first Conv2d accepts a single input channel. The approach:

- Try to instantiate the ultralytics.YOLO object and extract its underlying
  torch.nn.Module state_dict (safe path when ultralytics is available).
- Locate the first Conv2d with in_channels==3, average its weights across the
  input channel dimension to produce a [out,1,k,k] kernel, replace in the
  state_dict, and save a new checkpoint that mirrors the original but with
  the modified "model" entry (or a raw state_dict if the original was such).

Best-effort: if any step fails the original source path is returned and
training proceeds without conversion.
"""
from __future__ import annotations

from pathlib import Path
import logging
import torch
from typing import Optional

log = logging.getLogger("pvrt")


def convert_yolo_checkpoint_single_removed(*_args, **_kwargs):
  """
  Conversion to a single-channel YOLO checkpoint has been removed.
  This project no longer supports single-channel models; requests to
  convert checkpoints to a single input channel will raise a
  NotImplementedError to make the removal explicit.
  """
  raise NotImplementedError("Single-channel YOLO checkpoint conversion removed. Use 3ch or 4ch workflows.")
