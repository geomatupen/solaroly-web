# backend/pvrt/backends/detectron/train/aug_utils.py
from __future__ import annotations
import numpy as np
import detectron2.data.transforms as T

class RandHFlip(T.Augmentation):
    def __init__(self, prob=0.5): super().__init__(); self.prob = prob
    def get_transform(self, image):
        return T.NoOpTransform() if np.random.rand() > self.prob else T.HFlipTransform(image.shape[1])

class RandVFlip(T.Augmentation):
    def __init__(self, prob=0.1): super().__init__(); self.prob = prob
    def get_transform(self, image):
        return T.NoOpTransform() if np.random.rand() > self.prob else T.VFlipTransform(image.shape[0])

def build_geometric_augs(cfg):
    """Safe for RGB & RGB-T; never triggers 'both horiz and vert'."""
    return [
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN,
            cfg.INPUT.MAX_SIZE_TRAIN,
            cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
        ),
        RandHFlip(0.5),
        RandVFlip(0.1),
        T.RandomRotation(angle=[-10, 10], sample_style="range", expand=False),
        T.RandomCrop("relative_range", (0.9, 0.9)),
    ]

def build_rgb_photometric_augs():
    """Photometric jitter for RGB only (never touch thermal)."""
    return [
        T.RandomBrightness(0.9, 1.1),
        T.RandomContrast(0.9, 1.1),
        T.RandomLighting(0.05),
    ]
