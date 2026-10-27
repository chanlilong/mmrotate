# Copyright (c) OpenMMLab. All rights reserved.
"""Generalized Jensen-Shannon divergence (GJSD) between (rotated) boxes.

A box is the 2D Gaussian ``N(mu, Sigma)`` of :func:`box_to_gaussian`. DCFL
(Xu et al., CVPR 2023) ranks priors by ``1 / (1 + GJSD)`` with ``alpha = 0.5``,
for which the GJSD of the reference code
(``mmrotate/core/bbox/iou_calculators/rotate_metric_calculator.py`` of
https://github.com/Chasel-Tsui/mmrotate-dcfl) is the Bhattacharyya distance.
With ``S = Sigma1 + Sigma2`` and ``d = mu1 - mu2`` it has the closed form::

    GJSD = 1/4 * d^T S^-1 d + 1/2 * ln(det(S) / (4 * sqrt(det(Sigma1) det(Sigma2))))

which is evaluated here with 2x2 algebra on the ``(x, y, a, b, c, s)``
parameters. It equals the reference's expression, but avoids its
absolute-coordinate terms (which cancel catastrophically in fp32 for small
boxes far from the origin), its ``(m, n, 2, 2)`` temporaries and its fp16 cast
on CUDA.
"""

import math

import torch
from torch import Tensor

from ..nwd_rka.nwd import box_to_gaussian

# Pairwise results are computed in row chunks of at most this many elements so
# the handful of (rows, M) temporaries stays bounded on large prior sets.
_CHUNK_NUMEL = 1 << 24
# Same width/height floor as the reference ``xy_wh_r_2_xy_sigma``.
_EPS = 1e-7
_TINY = torch.finfo(torch.float32).tiny


def gaussian_gjsd(g1: Tensor, g2: Tensor) -> Tensor:
    """GJSD (``alpha = 0.5``) between broadcastable Gaussian params ``(..., 6)``."""
    dx = g1[..., 0] - g2[..., 0]
    dy = g1[..., 1] - g2[..., 1]
    a = g1[..., 2] + g2[..., 2]
    b = g1[..., 3] + g2[..., 3]
    c = g1[..., 4] + g2[..., 4]
    det = (a * c - b * b).clamp(min=_TINY)
    mahalanobis = (c * dx * dx - 2 * b * dx * dy + a * dy * dy) / det
    log_ratio = det.log() - g1[..., 5].log() - g2[..., 5].log() - math.log(4.0)
    return (0.25 * mahalanobis + 0.5 * log_ratio).clamp(min=0)


def rbbox_gjsd(bboxes1: Tensor, bboxes2: Tensor, is_aligned: bool = False) -> Tensor:
    """GJSD similarity ``1 / (1 + GJSD)`` between two sets of boxes.

    Horizontal ``(x1, y1, x2, y2)`` and rotated ``(cx, cy, w, h, theta)``
    boxes can be mixed; horizontal ones are treated as ``theta = 0``.

    Args:
        bboxes1 (Tensor): ``(m, 4)`` or ``(m, 5)`` boxes.
        bboxes2 (Tensor): ``(n, 4)`` or ``(n, 5)`` boxes.
        is_aligned (bool): If True, ``m == n`` and the result is ``(m,)``.

    Returns:
        Tensor: float32 similarity in ``(0, 1]`` with shape ``(m, n)`` or ``(m,)``.
    """
    rows, cols = bboxes1.size(0), bboxes2.size(0)
    if is_aligned:
        assert rows == cols, 'aligned GJSD needs the same number of boxes'
    if rows * cols == 0:
        shape = (rows,) if is_aligned else (rows, cols)
        return bboxes1.new_zeros(shape, dtype=torch.float32)

    g1 = box_to_gaussian(bboxes1, _EPS)
    g2 = box_to_gaussian(bboxes2, _EPS)
    if is_aligned:
        return 1 / (1 + gaussian_gjsd(g1, g2))

    out = g1.new_empty((rows, cols))
    chunk = max(1, _CHUNK_NUMEL // cols)
    for start in range(0, rows, chunk):
        out[start : start + chunk] = 1 / (
            1 + gaussian_gjsd(g1[start : start + chunk, None], g2[None])
        )
    return out


def gaussian_density(points: Tensor, centers: Tensor, gauss: Tensor) -> Tensor:
    """Unnormalised Gaussian density ``exp(-1/2 * (p - mu)^T Sigma^-1 (p - mu))``.

    Element-wise over broadcastable inputs; ``mu`` is ``centers`` and ``Sigma``
    comes from the Gaussian params ``gauss`` (its own mean is not used).

    Args:
        points (Tensor): ``(..., 2)`` points.
        centers (Tensor): ``(..., 2)`` means.
        gauss (Tensor): ``(..., 6)`` params from :func:`box_to_gaussian`.

    Returns:
        Tensor: density in ``(0, 1]``.
    """
    dx = points[..., 0] - centers[..., 0]
    dy = points[..., 1] - centers[..., 1]
    a, b, c = gauss[..., 2], gauss[..., 3], gauss[..., 4]
    det = (a * c - b * b).clamp(min=_TINY)
    mahalanobis = (c * dx * dx - 2 * b * dx * dy + a * dy * dy) / det
    return torch.exp(-0.5 * mahalanobis)
