# Copyright (c) OpenMMLab. All rights reserved.
"""NMS with the Normalized Wasserstein Distance as overlap measure.

Port of ``wasserstein_nms`` from https://github.com/jwwangchn/mmcv-nwd (the
earlier NWD paper, arXiv:2110.13389), generalised to rotated boxes. On CUDA the
suppression mask is built by a Triton kernel; otherwise by chunked PyTorch.

Importing this module exposes ``nwd_nms`` inside ``mmcv.ops.nms``, where
``mmcv.ops.batched_nms`` resolves ``nms_cfg['type']``, so any head that uses
``batched_nms``/``multiclass_nms`` accepts::

    nms=dict(type='nwd_nms', iou_threshold=0.85, constant=12.8)

Caveat: with a small ``constant`` NWD barely suppresses duplicates of large
objects, which is why the NWD paper only used it for RPN proposals.
"""

import importlib

import numpy as np
import torch
from torch import Tensor

from .nwd import box_to_gaussian, gaussian_nwd

try:
    from .nwd_nms_triton import nwd_nms_mask_triton

    HAS_TRITON = True
except ImportError:  # pragma: no cover - triton ships with CUDA torch builds
    HAS_TRITON = False

_WORD_BITS = 64
_CHUNK_NUMEL = 1 << 24


def nwd_nms_mask_pytorch(gauss: Tensor, threshold: float, constant: float) -> Tensor:
    """Reference for ``nwd_nms_mask_triton``: same ``(N, ceil(N/64))`` bitmask."""
    num_boxes = gauss.size(0)
    num_words = -(-num_boxes // _WORD_BITS)
    padded = num_words * _WORD_BITS
    weights = torch.ones(_WORD_BITS, dtype=torch.int64, device=gauss.device)
    weights = weights << torch.arange(_WORD_BITS, dtype=torch.int64, device=gauss.device)
    cols = torch.arange(num_boxes, device=gauss.device)
    mask = torch.zeros((num_boxes, num_words), dtype=torch.int64, device=gauss.device)
    chunk = max(1, _CHUNK_NUMEL // max(padded, 1))
    for start in range(0, num_boxes, chunk):
        rows = cols[start : start + chunk]
        nwd = gaussian_nwd(gauss[rows, None], gauss[None], constant)
        suppress = (nwd > threshold) & (cols[None] > rows[:, None])
        suppress = torch.nn.functional.pad(suppress, (0, padded - num_boxes))
        # int64 sum of distinct powers of two wraps exactly like a bitwise OR.
        mask[rows] = (suppress.view(-1, num_words, _WORD_BITS).long() * weights).sum(-1)
    return mask


def _greedy_scan(mask: Tensor) -> Tensor:
    """Indices (into the sorted order) kept by greedy NMS over a bitmask."""
    mask_np = mask.cpu().numpy().view(np.uint64)
    removed = np.zeros(mask_np.shape[1], dtype=np.uint64)
    keep = []
    for i in range(mask_np.shape[0]):
        word, bit = divmod(i, _WORD_BITS)
        if not (int(removed[word]) >> bit) & 1:
            keep.append(i)
            removed |= mask_np[i]
    return torch.as_tensor(keep, dtype=torch.long, device=mask.device)


def nwd_nms(
    boxes: Tensor,
    scores: Tensor,
    iou_threshold: float,
    constant: float = 12.8,
    use_triton: bool | None = None,
) -> tuple[Tensor, Tensor]:
    """Greedy NMS that suppresses boxes whose NWD to a kept box exceeds the
    threshold (follows the ``mmcv.ops.nms`` / ``nms_rotated`` contract).

    Args:
        boxes (Tensor): ``(N, 4)`` ``(x1, y1, x2, y2)`` or ``(N, 5)``
            ``(cx, cy, w, h, theta)`` boxes.
        scores (Tensor): ``(N,)`` scores.
        iou_threshold (float): NWD threshold (the NWD paper uses 0.85 for RPN).
        constant (float): NWD normalisation constant. Defaults to 12.8.
        use_triton (bool, optional): Force the Triton (True) or PyTorch
            (False) mask; by default Triton is used for CUDA tensors.

    Returns:
        tuple[Tensor, Tensor]: ``dets`` ``(K, 5|6)`` (boxes with score
        appended, by descending score) and ``keep`` ``(K,)`` indices.
    """
    if boxes.size(0) == 0:
        return boxes.new_zeros((0, boxes.size(-1) + 1)), boxes.new_zeros(0, dtype=torch.long)
    order = scores.argsort(descending=True)
    gauss = box_to_gaussian(boxes[order].detach())
    if use_triton is None:
        use_triton = HAS_TRITON and gauss.is_cuda
    if use_triton:
        mask = nwd_nms_mask_triton(gauss, iou_threshold, constant)
    else:
        mask = nwd_nms_mask_pytorch(gauss, iou_threshold, constant)
    keep = order[_greedy_scan(mask).to(order.device)]
    dets = torch.cat([boxes[keep], scores[keep, None].to(boxes.dtype)], dim=1)
    return dets, keep


def register_nwd_nms() -> None:
    """Expose ``nwd_nms`` to ``mmcv.ops.batched_nms``'s ``type`` lookup.

    ``mmcv.ops.nms`` the attribute is the ``nms`` function, so the module is
    fetched with importlib. Only a new name is added; mmcv itself is unchanged.
    """
    mmcv_nms_module = importlib.import_module('mmcv.ops.nms')
    if getattr(mmcv_nms_module, 'nwd_nms', None) is not nwd_nms:
        mmcv_nms_module.nwd_nms = nwd_nms


register_nwd_nms()
