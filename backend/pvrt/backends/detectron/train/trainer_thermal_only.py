from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator

from .aug_utils import build_geometric_augs
from .mapper_thermal_only import ThermalOnlyDatasetMapper
from ..utils.model_patch import make_cfg_1ch, patch_first_conv_to_1ch, ensure_model_pixel_stats_1ch, force_axis_aligned_anchors

import logging
log = logging.getLogger("pvrt")


class ThermalOnlyTrainer(DefaultTrainer):
    """
    Trainer for thermal-only (1-channel) pipelines.

    - Uses `ThermalOnlyDatasetMapper` to produce 1-channel tensors
    - Forces axis-aligned anchors
    - Patches the model's first conv to accept 1 channel
    - Ensures pixel_mean/std are 1-channel
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder: Optional[str] = None):
        return COCOEvaluator(dataset_name, output_dir=output_folder or cfg.OUTPUT_DIR)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = ThermalOnlyDatasetMapper(cfg, is_train=False)
        try:
            return build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        except TypeError:
            return build_detection_test_loader(cfg, dataset_name)

    @classmethod
    def build_train_loader(cls, cfg):
        geom = build_geometric_augs(cfg)
        log.info(
            f"UI:OK:train: AUG:train[thermal-only] = " + ", ".join(type(a).__name__ for a in geom)
        )
        mapper = ThermalOnlyDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_model(cls, cfg):
        # 1) Make cfg consistent for single-channel thermal inputs
        make_cfg_1ch(cfg)
        force_axis_aligned_anchors(cfg)

        # 2) Build base model
        model = super().build_model(cfg)

        # 3) Patch first conv to accept 1 channel and ensure pixel stats
        try:
            patch_first_conv_to_1ch(model)
            ensure_model_pixel_stats_1ch(model)
        except Exception:
            log.exception("Failed to patch model first conv for 1-channel; proceeding with unpatched model")

        return model
