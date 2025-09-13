# backend/pvrt/trainops/model_patch.py
import logging
import torch
import torch.nn as nn

log = logging.getLogger("pvrt")

def _get_stem_and_parent(model):
    """
    Return (parent_module, conv1) for the backbone stem, or (None, None) if not found.
    Works with common Detectron2 ResNet backbones.
    """
    try:
        parent = model.backbone.bottom_up.stem
        conv1 = parent.conv1
        return parent, conv1
    except Exception:
        pass
    try:
        parent = model.backbone.stem
        conv1 = parent.conv1
        return parent, conv1
    except Exception:
        pass
    return None, None


def stem_in_channels(model) -> int | None:
    parent, conv1 = _get_stem_and_parent(model)
    if conv1 is None:
        return None
    try:
        return int(conv1.weight.shape[1])
    except Exception:
        return None


def widen_backbone_conv1_to_4ch(model):
    """
    Idempotent widening of stem.conv1 from 3 -> 4 input channels.
    - If already 4, NO-OP (prevents 5-channel bug).
    - If not 3 or 4, log a warning and NO-OP.
    - When widening, the 4th channel is initialized as the mean of the RGB filters.
    """
    parent, conv1 = _get_stem_and_parent(model)
    if conv1 is None:
        log.warning("[patch] Unable to locate stem.conv1; leaving model unchanged.")
        return model

    in_ch = int(conv1.weight.shape[1])
    if in_ch == 4:
        log.info("[patch] stem.conv1 already has 4 input channels; no widening needed.")
        return model
    if in_ch != 3:
        log.warning(f"[patch] stem.conv1 has in_channels={in_ch}; expected 3 (to widen) or 4. Leaving unchanged.")
        return model

    out_ch = conv1.out_channels
    k = conv1.kernel_size
    stride = conv1.stride
    padding = conv1.padding
    bias_flag = conv1.bias is not None

    new_conv = nn.Conv2d(4, out_ch, k, stride=stride, padding=padding, bias=bias_flag)

    with torch.no_grad():
        # Copy RGB weights
        new_conv.weight[:, :3, :, :] = conv1.weight
        # 4th channel initialized as mean of RGB weights
        mean = conv1.weight.mean(dim=1, keepdim=True)  # (out_ch, 1, kH, kW)
        new_conv.weight[:, 3:4, :, :] = mean
        if bias_flag:
            new_conv.bias.copy_(conv1.bias)

    parent.conv1 = new_conv
    log.info("[patch] Widened stem.conv1 from 3 -> 4 input channels.")
    return model
