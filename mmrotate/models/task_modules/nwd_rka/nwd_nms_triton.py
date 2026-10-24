# Copyright (c) OpenMMLab. All rights reserved.
"""Triton kernel for the NWD-NMS suppression bitmask.

Replaces the ``wasserstein_nms`` CUDA op of https://github.com/jwwangchn/mmcv-nwd
(no mmcv compilation needed) and generalises it to rotated boxes. Like mmcv's
CUDA NMS, the GPU builds a ``(N, ceil(N / 64))`` bitmask whose bit ``j`` of row
``i`` says that box ``i`` suppresses the lower-scored box ``j``; the greedy
scan over the mask happens on the host.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor

_BLOCK_ROWS = 32
_WORD_BITS = 64


@triton.jit
def _nwd_nms_mask_kernel(
    gauss_ptr,
    mask_ptr,
    num_boxes,
    num_words,
    threshold,
    inv_constant,
    BLOCK_ROWS: tl.constexpr,
    WORD_BITS: tl.constexpr,
) -> None:
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    word = tl.program_id(1)
    cols = word * WORD_BITS + tl.arange(0, WORD_BITS)
    row_ok = rows < num_boxes
    col_ok = cols < num_boxes

    # Gaussian params per box: (x, y, a, b, c, sqrt(det)), row-major (N, 6).
    x1 = tl.load(gauss_ptr + rows * 6 + 0, mask=row_ok, other=0.0)
    y1 = tl.load(gauss_ptr + rows * 6 + 1, mask=row_ok, other=0.0)
    a1 = tl.load(gauss_ptr + rows * 6 + 2, mask=row_ok, other=0.0)
    b1 = tl.load(gauss_ptr + rows * 6 + 3, mask=row_ok, other=0.0)
    c1 = tl.load(gauss_ptr + rows * 6 + 4, mask=row_ok, other=0.0)
    s1 = tl.load(gauss_ptr + rows * 6 + 5, mask=row_ok, other=0.0)
    x2 = tl.load(gauss_ptr + cols * 6 + 0, mask=col_ok, other=0.0)
    y2 = tl.load(gauss_ptr + cols * 6 + 1, mask=col_ok, other=0.0)
    a2 = tl.load(gauss_ptr + cols * 6 + 2, mask=col_ok, other=0.0)
    b2 = tl.load(gauss_ptr + cols * 6 + 3, mask=col_ok, other=0.0)
    c2 = tl.load(gauss_ptr + cols * 6 + 4, mask=col_ok, other=0.0)
    s2 = tl.load(gauss_ptr + cols * 6 + 5, mask=col_ok, other=0.0)

    dx = x1[:, None] - x2[None, :]
    dy = y1[:, None] - y2[None, :]
    trace = (a1 + c1)[:, None] + (a2 + c2)[None, :]
    cross = (
        a1[:, None] * a2[None, :]
        + 2.0 * b1[:, None] * b2[None, :]
        + c1[:, None] * c2[None, :]
        + 2.0 * s1[:, None] * s2[None, :]
    )
    w2 = dx * dx + dy * dy + trace - 2.0 * tl.sqrt(tl.maximum(cross, 0.0))
    nwd = tl.exp(-tl.sqrt(tl.maximum(w2, 1e-6)) * inv_constant)

    suppress = (nwd > threshold) & (cols[None, :] > rows[:, None]) & col_ok[None, :]
    bit = tl.full([WORD_BITS], 1, tl.int64) << tl.arange(0, WORD_BITS).to(tl.int64)
    # Bits are distinct powers of two, so the sum never carries and equals OR.
    bits = tl.sum(tl.where(suppress, bit[None, :], 0), axis=1)
    tl.store(mask_ptr + rows * num_words + word, bits, mask=row_ok)


def nwd_nms_mask_triton(gauss: Tensor, threshold: float, constant: float) -> Tensor:
    """Suppression bitmask for score-sorted boxes.

    Args:
        gauss (Tensor): ``(N, 6)`` Gaussian params from ``box_to_gaussian``.
        threshold (float): NWD above which the lower-scored box is suppressed.
        constant (float): NWD normalisation constant.

    Returns:
        Tensor: ``(N, ceil(N / 64))`` int64 bitmask.
    """
    gauss = gauss.float().contiguous()
    num_boxes = gauss.size(0)
    num_words = triton.cdiv(num_boxes, _WORD_BITS)
    mask = torch.zeros((num_boxes, num_words), dtype=torch.int64, device=gauss.device)
    if num_boxes == 0:
        return mask
    grid = (triton.cdiv(num_boxes, _BLOCK_ROWS), num_words)
    _nwd_nms_mask_kernel[grid](
        gauss,
        mask,
        num_boxes,
        num_words,
        float(threshold),
        1.0 / float(constant),
        BLOCK_ROWS=_BLOCK_ROWS,
        WORD_BITS=_WORD_BITS,
    )
    return mask
