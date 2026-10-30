# Copyright (c) OpenMMLab. All rights reserved.
"""Modulated deformable convolution (DCNv2) from ``grid_sample`` and a conv.

``torchvision.ops.deform_conv2d`` and mmcv's ``modulated_deform_conv2d`` have
no ONNX / TensorRT export. This version only uses ``F.grid_sample`` (ONNX
``GridSample``, TensorRT ``IGridSampleLayer``) and a grouped 1x1 conv, with
the same offset layout and bilinear sampling (zero outside the image), so a
DCN layer can be swapped to it for export without touching its weights.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def _tap_positions(
    out_size: int, kernel: int, stride: int, pad: int, dilation: int, ref: Tensor
) -> Tensor:
    """Regular sampling positions ``(kernel, out_size)`` along one axis.

    Built from Python ints, so export records a constant (no ``Range`` op).
    """
    positions = [[k * dilation + o * stride - pad for o in range(out_size)] for k in range(kernel)]
    return torch.tensor(positions, dtype=torch.float32, device=ref.device)


def modulated_deform_conv2d_gridsample(
    x: Tensor,
    offset: Tensor,
    mask: Tensor | None,
    weight: Tensor,
    bias: Tensor | None,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    dilation: tuple[int, int] = (1, 1),
    groups: int = 1,
    deform_groups: int = 1,
) -> Tensor:
    """DCNv2 with the semantics of ``torchvision.ops.deform_conv2d``.

    Args:
        x (Tensor): ``(B, C, H, W)`` input.
        offset (Tensor): ``(B, deform_groups * 2 * kh * kw, Ho, Wo)`` offsets
            ``[dy0, dx0, dy1, dx1, ...]`` per deform group, kernel taps in
            row-major order.
        mask (Tensor, optional): ``(B, deform_groups * kh * kw, Ho, Wo)``
            modulation, ``None`` for DCNv1.
        weight (Tensor): ``(O, C // groups, kh, kw)``.
        bias (Tensor, optional): ``(O,)``.

    Returns:
        Tensor: ``(B, O, Ho, Wo)``.
    """
    # Python ints, so the (static) shapes are traced as constants.
    B, C, H, W = (int(n) for n in x.shape)
    out_ch, _, kh, kw = (int(n) for n in weight.shape)
    K = kh * kw
    Ho, Wo = int(offset.shape[-2]), int(offset.shape[-1])
    dg = deform_groups

    offset = offset.reshape(B, dg, K, 2, Ho, Wo)
    # Regular grid per tap: (K, Ho, 1) rows and (K, 1, Wo) columns.
    ys = _tap_positions(Ho, kh, stride[0], padding[0], dilation[0], x)  # (kh, Ho)
    xs = _tap_positions(Wo, kw, stride[1], padding[1], dilation[1], x)  # (kw, Wo)
    base_y = ys[:, None, :, None].expand(kh, kw, Ho, 1).reshape(K, Ho, 1)
    base_x = xs[None, :, None, :].expand(kh, kw, 1, Wo).reshape(K, 1, Wo)
    # Sampling coordinates in fp32: fp16 would round them to 1/8 px at 200.
    py = base_y + offset[:, :, :, 0].float()  # (B, dg, K, Ho, Wo)
    px = base_x + offset[:, :, :, 1].float()
    # Pixel index -> grid_sample coordinate (align_corners=False: centre of
    # pixel i is (2i + 1) / size - 1); zero padding matches torchvision.
    grid = torch.stack(((2 * px + 1) / W - 1, (2 * py + 1) / H - 1), dim=-1)
    grid = grid.reshape(B * dg, K * Ho, Wo, 2)
    cols = F.grid_sample(
        x.reshape(B * dg, C // dg, H, W),
        grid.to(x.dtype),
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False,
    )  # (B * dg, C // dg, K * Ho, Wo)
    cols = cols.reshape(B, dg, C // dg, K, Ho, Wo)
    if mask is not None:
        cols = cols * mask.reshape(B, dg, 1, K, Ho, Wo).to(cols.dtype)
    # (B, C * K, Ho, Wo), channel-major then tap, as the flattened weight.
    cols = cols.reshape(B, C * K, Ho, Wo)
    return F.conv2d(cols, weight.reshape(out_ch, (C // groups) * K, 1, 1), bias, groups=groups)
