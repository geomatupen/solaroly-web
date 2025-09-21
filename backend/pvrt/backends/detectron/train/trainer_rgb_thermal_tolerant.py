# backend/pvrt/trainops/trainer_rgb_thermal_tolerant.py
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator
from .aug_utils import build_geometric_augs, build_rgb_photometric_augs

# Reuse your tolerant mapper that guarantees aligned RGB+Thermal - 4 channels
from .mapper_rgb_thermal_tolerant import RGBThermalDatasetMapper
import logging
log = logging.getLogger("pvrt")


def _force_axis_aligned_anchors(cfg) -> None:
    """
    Ensure non-rotated anchors. Prevents 5-dim 'rotated box' crashes in losses.
    """
    if hasattr(cfg.MODEL, "ANCHOR_GENERATOR") and hasattr(cfg.MODEL.ANCHOR_GENERATOR, "ANGLES"):
        cfg.MODEL.ANCHOR_GENERATOR.ANGLES = [[0]]


def _make_cfg_4ch(cfg) -> None:
    """
    Make normalization 4-channel aware. RGB uses Detectron defaults, thermal gets a neutral midpoint.
    """
    cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675, 128.0]  # B, G, R, T
    cfg.MODEL.PIXEL_STD  = [57.375,  57.120,  58.395,  58.0]


def _patch_first_conv_to_4ch(model: nn.Module) -> None:
    """
    Replace the FIRST Conv2d that expects 3 input channels with an equivalent 4-channel layer.
    The 4th channel is initialized as the mean of the first three.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            new_conv = nn.Conv2d(
                in_channels=4,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=(module.bias is not None),
                padding_mode=module.padding_mode,
            )
            with torch.no_grad():
                w = module.weight  # [out, 3, k, k]
                new_conv.weight[:, :3, :, :] = w
                new_conv.weight[:, 3:4, :, :] = w.mean(dim=1, keepdim=True)
                if module.bias is not None:
                    new_conv.bias.copy_(module.bias)

            # graft new conv into the parent
            parent = model
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], new_conv)
            return  # done after first replacement


def _ensure_model_pixel_stats_4ch(model: nn.Module, mean_val: float = 128.0, std_val: float = 58.0) -> None:
    """
    Detectron2 models store pixel stats as buffers (pixel_mean/std). Ensure they are 4-channel.
    """
    pm = getattr(model, "pixel_mean", None)
    ps = getattr(model, "pixel_std", None)
    if pm is None or ps is None:
        return

    # Expect [C,1,1]
    need_resize = (pm.numel() != 4) or (ps.numel() != 4)
    if need_resize:
        device = pm.device
        dtype = pm.dtype
        new_pm = torch.tensor([103.530, 116.280, 123.675, mean_val], dtype=dtype, device=device).view(4, 1, 1)
        new_ps = torch.tensor([57.375,  57.120,  58.395,  std_val], dtype=dtype, device=device).view(4, 1, 1)
        model.register_buffer("pixel_mean", new_pm)
        model.register_buffer("pixel_std", new_ps)


class RTolerantTrainer(DefaultTrainer):
    """
    Trainer for RGB+Thermal (4-channel) pipelines.

    What it does:
    - Uses your RGBThermalDatasetMapper to produce 4-channel training tensors.
    - Forces axis-aligned anchors to avoid rotated-box mismatches.
    - Patches the model's first conv to accept 4 channels.
    - Ensures pixel_mean/STD buffers are 4-channel.
    - Leaves augments, solver, and evaluation to cfg (standard Detectron2).
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder: Optional[str] = None):
        return COCOEvaluator(dataset_name, output_dir=output_folder or cfg.OUTPUT_DIR)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        geom = build_geometric_augs(cfg)
        mapper = RGBThermalDatasetMapper(cfg, is_train=False)
        try:
            return build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        except TypeError:
            return build_detection_test_loader(cfg, dataset_name)


    @classmethod
    def build_train_loader(cls, cfg):
        # log once what the mapper will apply
        geom = build_geometric_augs(cfg)
        log.info(
            f"UI:OK:train: AUG:train[rgbt] = " + ", ".join(type(a).__name__ for a in geom) + " + Photometric(RGB only)"
        )
        mapper = RGBThermalDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_model(cls, cfg):
        # 1) Make the config consistent for 4 channels + axis-aligned anchors
        _make_cfg_4ch(cfg)
        _force_axis_aligned_anchors(cfg)

        # 2) Build the standard Detectron2 model
        model = super().build_model(cfg)

        # 3) Patch first conv to 4-ch and ensure pixel stats reflect 4-ch
        _patch_first_conv_to_4ch(model)
        _ensure_model_pixel_stats_4ch(model)

        return model

