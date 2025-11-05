# backend/pvrt/backends/detectron/utils/model_patch.py
from __future__ import annotations

import torch
import torch.nn as nn


def force_axis_aligned_anchors(cfg) -> None:
    """
    Ensure non-rotated anchors. Prevents rotated-box heads from sneaking in.
    """
    if hasattr(cfg.MODEL, "ANCHOR_GENERATOR") and hasattr(cfg.MODEL.ANCHOR_GENERATOR, "ANGLES"):
        cfg.MODEL.ANCHOR_GENERATOR.ANGLES = [[0]]


def make_cfg_4ch(cfg, thermal_mean: float = 128.0, thermal_std: float = 58.0) -> None:
    """
    Adjust Detectron2 normalization for 4 channels (B, G, R, T).
    Keep RGB as Detectron defaults; set thermal to neutral stats.
    """
    cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675, float(thermal_mean)]
    cfg.MODEL.PIXEL_STD  = [57.375,  57.120,  58.395,  float(thermal_std)]


def patch_first_conv_to_4ch(model: nn.Module) -> None:
    """
    Replace the FIRST Conv2d expecting 3 input channels with an equivalent 4-channel layer.
    The 4th channel is initialized as the mean of the first three.
    No-op if already patched.
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

            # graft the new conv into the parent
            parent = model
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], new_conv)
            return  # patched first conv; done


def ensure_model_pixel_stats_4ch(model: nn.Module, thermal_mean: float = 128.0, thermal_std: float = 58.0) -> None:
    """
    Ensure Detectron2's pixel_mean/std buffers are 4-channel.
    """
    pm = getattr(model, "pixel_mean", None)
    ps = getattr(model, "pixel_std", None)
    if pm is None or ps is None:
        return

    if pm.numel() == 4 and ps.numel() == 4:
        return  # already 4-ch

    device = pm.device
    dtype = pm.dtype
    new_pm = torch.tensor([103.530, 116.280, 123.675, thermal_mean], dtype=dtype, device=device).view(4, 1, 1)
    new_ps = torch.tensor([57.375,  57.120,  58.395,  thermal_std], dtype=dtype, device=device).view(4, 1, 1)
    # register as buffers (overwrite existing)
    model.register_buffer("pixel_mean", new_pm)
    model.register_buffer("pixel_std", new_ps)

