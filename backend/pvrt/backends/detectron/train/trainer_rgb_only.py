# backend/pvrt/trainops/trainer_rgb_only.py
from __future__ import annotations

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader


class RGBOnlyTrainer(DefaultTrainer):
    """
    Minimal trainer for standard 3-channel RGB training.

    - Uses Detectron2's default DatasetMapper (works with COCO-style datasets)
    - Keeps config-driven behavior (SOLVER, DATALOADER, AUGS, etc.) in cfg
    - Provides a COCO evaluator and a public build_test_loader for quick eval

    This class intentionally avoids app-specific logic. Anything about
    thermal, IO, or metadata is handled outside (backend adapter + web layer).
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return COCOEvaluator(dataset_name, output_dir=output_folder or cfg.OUTPUT_DIR)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        # Exposed so callers can run a one-off validation after training
        return build_detection_test_loader(cfg, dataset_name)
