# -*- coding: utf-8 -*-
"""
Tolerant Trainer for RGB + Thermal (4 channels) pipelines.

Key add: BEFORE building the model, if using non-rotated heads (RPN/StandardROIHeads)
but cfg.MODEL.ANCHOR_GENERATOR.ANGLES indicates rotation (e.g. [-90,0,90]),
we force ANGLES to [[0]] so box_dim stays 4 and avoids 4-vs-5 loss crashes.
"""

from typing import Optional
import os
import numpy as np
import torch
import torch.nn as nn

from detectron2.engine import DefaultTrainer
from detectron2.data import (
    build_detection_train_loader,
    build_detection_test_loader,
    get_detection_dataset_dicts,
)

# Prefer loguru if available; otherwise print().
try:
    from loguru import logger as log
except Exception:
    class _P:
        def info(self, *a, **k): print(*a, flush=True)
        def warning(self, *a, **k): print(*a, flush=True)
        def error(self, *a, **k): print(*a, flush=True)
    log = _P()

from .mapper_rgb_thermal_tolerant import RGBThermalDatasetMapper


# ----------------------------- helpers ------------------------------------ #

def _ensure_pixel_stats_4ch_on_model(model, mean_val: float = 0.5, std_val: float = 0.5):
    try:
        if hasattr(model, "pixel_mean") and hasattr(model, "pixel_std"):
            pm = model.pixel_mean
            ps = model.pixel_std
            C = 4
            need = (pm.numel() != C) or (ps.numel() != C)
            if need:
                device = pm.device
                dtype = pm.dtype
                new_pm = torch.tensor([mean_val] * C, dtype=dtype, device=device).view(C, 1, 1)
                new_ps = torch.tensor([std_val] * C, dtype=dtype, device=device).view(C, 1, 1)
                model.register_buffer("pixel_mean", new_pm)
                model.register_buffer("pixel_std", new_ps)
                print(f"[trainer] INFO: Set model.pixel_mean/std to 4-ch ({mean_val}/{std_val}).", flush=True)
    except Exception as e:
        print(f"[trainer] WARN: Could not set pixel stats to 4-ch: {e}", flush=True)


def _try_get_resnet_stem(module: nn.Module) -> Optional[nn.Conv2d]:
    try:
        return module.backbone.bottom_up.stem.conv1  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        return module.backbone.stem.conv1  # type: ignore[attr-defined]
    except Exception:
        pass
    for m in module.modules():
        if isinstance(m, nn.Conv2d) and m.in_channels == 3:
            return m
    return None


def _inflate_conv_in_to_4(conv: nn.Conv2d, method: str = "avg") -> nn.Conv2d:
    assert isinstance(conv, nn.Conv2d)
    W = conv.weight.data
    out_c, in_c, kh, kw = W.shape
    if in_c == 4:
        return conv
    if in_c != 3:
        print(f"[trainer] WARN: Expected 3->4 inflate, found in_ch={in_c}; adapting anyway.", flush=True)

    new_conv = nn.Conv2d(
        in_channels=4,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=(conv.bias is not None),
        padding_mode=conv.padding_mode,
        device=W.device,
        dtype=W.dtype,
    )

    with torch.no_grad():
        if in_c >= 3:
            new_conv.weight[:, :3, :, :] = W[:, :3, :, :]
            if method == "avg":
                new_conv.weight[:, 3:4, :, :] = W[:, :3, :, :].mean(dim=1, keepdim=True)
            elif method == "copy0":
                new_conv.weight[:, 3:4, :, :] = W[:, 0:1, :, :]
            else:
                new_conv.weight[:, 3:4, :, :].zero_()
        else:
            new_conv.weight.zero_()
            new_conv.weight[:, :in_c, :, :] = W
        if conv.bias is not None:
            new_conv.bias.data.copy_(conv.bias.data)

    return new_conv


def _replace_module(parent: nn.Module, child_name: str, new_module: nn.Module) -> bool:
    if not hasattr(parent, child_name):
        return False
    setattr(parent, child_name, new_module)
    return True


def _adapt_backbone_stem_to_4ch(model) -> bool:
    try:
        stem = _try_get_resnet_stem(model)
        if stem is None:
            print("[trainer] WARN: Could not locate a stem Conv2d to adapt to 4-ch.", flush=True)
            return False
        if not isinstance(stem, nn.Conv2d):
            print("[trainer] WARN: Located stem is not Conv2d; skipping 4-ch adaptation.", flush=True)
            return False
        if stem.in_channels == 4:
            print("[trainer] INFO: Stem already accepts 4 channels; nothing to do.", flush=True)
            return True

        new_stem = _inflate_conv_in_to_4(stem, method="avg")

        replaced = False
        try:
            parent = model.backbone.bottom_up.stem  # type: ignore[attr-defined]
            replaced = _replace_module(parent, "conv1", new_stem)
        except Exception:
            pass
        if not replaced:
            try:
                parent = model.backbone.stem  # type: ignore[attr-defined]
                replaced = _replace_module(parent, "conv1", new_stem)
            except Exception:
                pass

        if not replaced:
            for _, module in model.named_modules():
                for name, child in module.named_children():
                    if child is stem:
                        setattr(module, name, new_stem)
                        replaced = True
                        break
                if replaced:
                    break

        if replaced:
            print("[trainer] INFO: Adapted stem conv to in_channels=4 (weights inflated).", flush=True)
        else:
            print("[trainer] WARN: Failed to replace stem conv; model may still expect 3-ch.", flush=True)
        return replaced
    except Exception as e:
        print(f"[trainer] WARN: Exception while adapting stem to 4-ch: {e}", flush=True)
        return False


def _preflight_channel_dump(cfg, mapper, max_samples: int = 120):
    try:
        dataset_names = list(cfg.DATASETS.TRAIN)
        if not dataset_names:
            print("UI:WARN:train: No TRAIN datasets found for preflight.", flush=True)
            return
        print(f"UI:INFO:train: Preflight: inspecting {max_samples} samples from '{dataset_names[0]}'…", flush=True)
        dsets = get_detection_dataset_dicts(dataset_names)

        rng = np.random.RandomState(1234)
        idxs = rng.choice(len(dsets), size=min(max_samples, len(dsets)), replace=False)
        hist = {3: 0, 4: 0, 5: 0, "other": 0}

        for i in idxs:
            out = mapper(dsets[i])
            img = out["image"]  # CxHxW
            C = int(img.shape[0])
            if C in hist:
                hist[C] += 1
            else:
                hist["other"] += 1

        print(f"UI:INFO:train: Preflight channel histogram (after mapper): {hist}", flush=True)
        if hist.get(4, 0) == len(idxs):
            print("UI:OK:train: Preflight: all sampled items are 4-channel.", flush=True)
        else:
            print("UI:WARN:train: Preflight: not all items are 4-channel; mapper should have raised earlier.", flush=True)
    except Exception as e:
        print(f"UI:ERR:train: PEEK failed before training: {type(e).__name__}: {e}", flush=True)


def _angles_imply_rotation(angles_cfg) -> bool:
    # Detectron2 stores angles as list-of-lists per FPN level
    try:
        for sub in angles_cfg:
            for a in sub:
                if float(a) != 0.0:
                    return True
    except Exception:
        pass
    return False


# ----------------------------- Trainer ------------------------------------- #

class RTolerantTrainer(DefaultTrainer):
    """
    DefaultTrainer with:
      - tolerant RGBThermalDatasetMapper for train/eval (C=4 enforced),
      - model input adaptation to 4 channels,
      - pixel stats forced to 4-ch,
      - preflight logging to catch issues early,
      - **rotation/box-dim sanity**: zeroes ANGLES when using non-rotated heads.
    """

    @classmethod
    def build_model(cls, cfg):
        # ---- NEW: sanitize rotation vs heads BEFORE building model ----
        cfg.defrost()
        using_rot_heads = (
            cfg.MODEL.PROPOSAL_GENERATOR.NAME.lower().startswith("rrpn")
            or cfg.MODEL.ROI_HEADS.NAME.lower().startswith("rotated")
            or getattr(cfg.MODEL.ROI_BOX_HEAD, "POOLER_TYPE", "") == "ROIAlignRotated"
        )
        angles_rot = _angles_imply_rotation(cfg.MODEL.ANCHOR_GENERATOR.ANGLES)

        if (not using_rot_heads) and angles_rot:
            print(
                "[trainer] WARN: Detected rotated anchors (ANGLES non-zero) with non-rotated heads "
                f"(RPN/StandardROIHeads). Forcing ANGLES=[[0]] to keep box_dim=4 and avoid 4-vs-5 mismatch.",
                flush=True,
            )
            cfg.MODEL.ANCHOR_GENERATOR.ANGLES = [[0]]

        # optional: echo the final angles choice
        print(f"[trainer] INFO: Final ANGLES={cfg.MODEL.ANCHOR_GENERATOR.ANGLES}", flush=True)
        cfg.freeze()

        # ---- build the model with the sanitized cfg ----
        model = super().build_model(cfg)

        _ensure_pixel_stats_4ch_on_model(model, mean_val=0.5, std_val=0.5)
        _adapt_backbone_stem_to_4ch(model)

        # Try to print the box regression dimensionality if accessible
        try:
            bp = getattr(getattr(model, "roi_heads", None), "box_predictor", None)
            box_dim = None
            if bp is not None and hasattr(bp, "box2box_transform"):
                w = getattr(bp.box2box_transform, "weights", None)
                if w is not None:
                    box_dim = len(w)
            print(f"[dbg][cfg] ROI_HEADS.NAME={cfg.MODEL.ROI_HEADS.NAME}  "
                  f"PROPOSAL_GENERATOR.NAME={cfg.MODEL.PROPOSAL_GENERATOR.NAME}  "
                  f"ANCHOR_GENERATOR.ANGLES={cfg.MODEL.ANCHOR_GENERATOR.ANGLES}  "
                  f"pred_box_dim={box_dim if box_dim is not None else 'NA'}",
                  flush=True)
        except Exception as e:
            print(f"[dbg][cfg] summary failed: {e}", flush=True)

        return model

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = RGBThermalDatasetMapper(cfg, is_train=True, expect_channels=4, log_prefix="UI")
        print("UI:INFO:train: [trainer] Using RGBThermalDatasetMapper (tolerant; enforces 4-ch) for TRAIN.", flush=True)
        _preflight_channel_dump(cfg, mapper, max_samples=120)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = RGBThermalDatasetMapper(cfg, is_train=False, expect_channels=4, log_prefix="UI")
        print(f"UI:INFO:train: [trainer] Using RGBThermalDatasetMapper (tolerant; enforces 4-ch) for EVAL '{dataset_name}'.", flush=True)
        return build_detection_test_loader(cfg, dataset_name=dataset_name, mapper=mapper)

    def train(self):
        # Optional one-batch probe remains (can disable with RGBT_DEBUG_ONEBATCH=0)
        if os.getenv("RGBT_DEBUG_ONEBATCH", "1") == "1":
            try:
                self.model.train()
                mapper = RGBThermalDatasetMapper(self.cfg, is_train=True, expect_channels=4, log_prefix="UI")
                tmp_loader = build_detection_train_loader(self.cfg, mapper=mapper)
                batch = next(iter(tmp_loader))
                print("[dbg] running one-batch probe…", flush=True)
                self.model(batch)
                print("[dbg] one-batch probe ok.", flush=True)
            except Exception as e:
                print("[dbg] one-batch probe crashed:", repr(e), flush=True)
                try:
                    for i, bi in enumerate(batch):
                        shp = tuple(bi["image"].shape) if "image" in bi else None
                        inst = bi.get("instances", None)
                        bshape = tuple(inst.gt_boxes.tensor.shape) if (inst and hasattr(inst, "gt_boxes")) else None
                        print(f"[dbg] sample[{i}] image_shape={shp} gt_boxes_shape={bshape}", flush=True)
                except Exception:
                    pass
                raise
        return super().train()
