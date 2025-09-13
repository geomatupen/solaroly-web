import torch
import torch.nn as nn

def widen_backbone_conv1_to_4ch(model):
    """
    Make the ResNet stem accept 4 channels by copying RGB weights and
    seeding the 4th channel with their mean.
    """
    conv1 = model.backbone.bottom_up.stem.conv1
    w = conv1.weight  # [C_out, 3, k, k]
    new_conv = nn.Conv2d(
        4, w.shape[0],
        kernel_size=conv1.kernel_size,
        stride=conv1.stride,
        padding=conv1.padding,
        bias=False
    )
    with torch.no_grad():
        mean_w = w.mean(dim=1, keepdim=True)   # [C_out,1,k,k]
        new_w = torch.cat([w, mean_w], dim=1)  # [C_out,4,k,k]
        new_conv.weight.copy_(new_w)
    model.backbone.bottom_up.stem.conv1 = new_conv
    return model