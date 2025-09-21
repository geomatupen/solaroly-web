# backend/pvrt/trainops/trainer_rgb_only.py
from __future__ import annotations

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader
from detectron2.data import DatasetMapper, build_detection_train_loader, build_detection_test_loader
from .aug_utils import build_geometric_augs, build_rgb_photometric_augs

import logging
log = logging.getLogger("pvrt")

class RGBOnlyTrainer(DefaultTrainer):
    """
    Trainer for standard 3-channel RGB training.

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
        augs = build_geometric_augs(cfg)  # geometric only at eval
        try:
            mapper = DatasetMapper(cfg, is_train=False, augmentations=augs)
            return build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        except TypeError:
            return build_detection_test_loader(cfg, dataset_name)

    @classmethod
    def build_train_loader(cls, cfg):
        augs = build_geometric_augs(cfg) + build_rgb_photometric_augs()    
        log.info(
            f"UI:OK:train:AUG:train[rgb] = " + ", ".join(type(a).__name__ for a in augs)
        )
        try:
            mapper = DatasetMapper(cfg, is_train=True, augmentations=augs)   # newer D2
        except TypeError:
            mapper = DatasetMapper(cfg, is_train=True, tfm_gens=augs)        # older D2
        return build_detection_train_loader(cfg, mapper=mapper)

