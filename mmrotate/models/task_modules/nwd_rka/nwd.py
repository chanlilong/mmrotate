# Copyright (c) OpenMMLab. All rights reserved.
"""Normalized Wasserstein Distance (NWD) between (rotated) boxes.

A box is modelled as a 2D Gaussian ``N(mu, Sigma)`` with ``mu`` its centre and
``Sigma = R diag(w^2/4, h^2/4) R^T``. The squared 2-Wasserstein distance between
two such Gaussians has the closed form (2x2 matrices)::

    W2^2 = |mu1 - mu2|^2 + tr(S1) + tr(S2)
           - 2 * sqrt(tr(S1 S2) + 2 * sqrt(det(S1) det(S2)))

and ``NWD = exp(-sqrt(W2^2) / C)``. For axis-aligned boxes this reduces exactly
to the formula of Xu et al., "Detecting tiny objects in aerial images: A
normalized Wasserstein distance and a new benchmark" (ISPRS 2022)::

    W2^2 = |c1 - c2|^2 + ((w1 - w2)^2 + (h1 - h2)^2) / 4

``C`` is a dataset-dependent constant, the paper uses the average absolute
object size (12.7 on AI-TOD; the reference CUDA NMS defaults to 12.8).
"""

import torch
from torch import Tensor

# Pairwise results are computed in row chunks of at most this many elements so
# the handful of (rows, M) temporaries stays bounded on large prior sets.
_CHUNK_NUMEL = 1 << 24


def box_to_gaussian(boxes: Tensor, eps: float = 1e-6) -> Tensor:
    """Convert boxes to Gaussian parameters ``(x, y, a, b, c, s)``.

    ``[[a, b], [b, c]]`` is the covariance and ``s = sqrt(det) = w * h / 4``.

    Args:
        boxes (Tensor): ``(..., 4)`` horizontal boxes ``(x1, y1, x2, y2)`` or
            ``(..., 5)`` rotated boxes ``(cx, cy, w, h, theta)``.
        eps (float): Minimum width and height.

    Returns:
        Tensor: ``(..., 6)`` float32 Gaussian parameters.
    """
    boxes = boxes.float()
    if boxes.size(-1) == 4:
        xy = (boxes[..., :2] + boxes[..., 2:]) / 2
        wh = boxes[..., 2:] - boxes[..., :2]
        theta = boxes.new_zeros(boxes.shape[:-1])
    elif boxes.size(-1) == 5:
        xy, wh, theta = boxes[..., :2], boxes[..., 2:4], boxes[..., 4]
    else:
        raise ValueError(f'Expected boxes with 4 or 5 values, got {boxes.size(-1)}')
    wh = wh.clamp(min=eps)
    var_w = wh[..., 0].pow(2) / 4
    var_h = wh[..., 1].pow(2) / 4
    cos, sin = torch.cos(theta), torch.sin(theta)
    a = var_w * cos.pow(2) + var_h * sin.pow(2)
    c = var_w * sin.pow(2) + var_h * cos.pow(2)
    b = (var_w - var_h) * sin * cos
    s = wh[..., 0] * wh[..., 1] / 4
    return torch.stack([xy[..., 0], xy[..., 1], a, b, c, s], dim=-1)


def gaussian_nwd(g1: Tensor, g2: Tensor, constant: float, eps: float = 1e-6) -> Tensor:
    """NWD between broadcastable Gaussian parameter tensors ``(..., 6)``."""
    center = (g1[..., 0] - g2[..., 0]).pow(2) + (g1[..., 1] - g2[..., 1]).pow(2)
    trace = g1[..., 2] + g1[..., 4] + g2[..., 2] + g2[..., 4]
    cross = (
        g1[..., 2] * g2[..., 2]
        + 2 * g1[..., 3] * g2[..., 3]
        + g1[..., 4] * g2[..., 4]
        + 2 * g1[..., 5] * g2[..., 5]
    )
    w2 = center + trace - 2 * cross.clamp(min=0).sqrt()
    return torch.exp(-w2.clamp(min=eps).sqrt() / constant)


def rbbox_nwd(
    bboxes1: Tensor,
    bboxes2: Tensor,
    constant: float = 12.8,
    is_aligned: bool = False,
    eps: float = 1e-6,
) -> Tensor:
    """Normalized Wasserstein Distance between two sets of boxes.

    Horizontal ``(x1, y1, x2, y2)`` and rotated ``(cx, cy, w, h, theta)``
    boxes can be mixed; horizontal ones are treated as ``theta = 0``.

    Args:
        bboxes1 (Tensor): ``(m, 4)`` or ``(m, 5)`` boxes.
        bboxes2 (Tensor): ``(n, 4)`` or ``(n, 5)`` boxes.
        constant (float): Normalisation constant ``C``. Defaults to 12.8.
        is_aligned (bool): If True, ``m == n`` and the result is ``(m,)``.
        eps (float): Numerical floor. Defaults to 1e-6.

    Returns:
        Tensor: float32 NWD in ``(0, 1]`` with shape ``(m, n)`` or ``(m,)``.
    """
    rows, cols = bboxes1.size(0), bboxes2.size(0)
    if is_aligned:
        assert rows == cols, 'aligned NWD needs the same number of boxes'
    if rows * cols == 0:
        shape = (rows,) if is_aligned else (rows, cols)
        return bboxes1.new_zeros(shape, dtype=torch.float32)

    g1 = box_to_gaussian(bboxes1, eps)
    g2 = box_to_gaussian(bboxes2, eps)
    if is_aligned:
        return gaussian_nwd(g1, g2, constant, eps)

    out = g1.new_empty((rows, cols))
    chunk = max(1, _CHUNK_NUMEL // cols)
    for start in range(0, rows, chunk):
        out[start : start + chunk] = gaussian_nwd(
            g1[start : start + chunk, None], g2[None], constant, eps
        )
    return out
