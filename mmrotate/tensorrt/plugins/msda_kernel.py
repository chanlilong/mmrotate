# Copyright (c) OpenMMLab. All rights reserved.
"""Fused MSDA forward kernel with the argument order TensorRT plugins need.

The same computation as ``_msda_fused_fwd_kernel`` of
``mmrotate/models/backbones/cradiov4/msda_triton_fused.py`` (whose sampling
helper ``_sample_tile`` is reused), but TensorRT launches ahead-of-time plugin
kernels as ``(input tensors..., scalars..., output tensors...)``, so the output
pointer is last here. There is no autotuning either: an engine embeds one
compiled kernel, with ``BLOCK_N`` / ``num_warps`` fixed at export time.
"""

import triton
from triton import language as tl

from mmrotate.models.backbones.cradiov4.msda_triton_fused import _sample_tile


@triton.jit
def msda_fused_fwd_trt_kernel(
    value_ptr,  # [B, I, H, C]
    shapes_ptr,  # [L, 2] (h, w)
    ref_ptr,  # [B or 1, N, LR, 2]
    off_ptr,  # [B, N, H, L, P, 2]
    logit_ptr,  # [B, N, H, L * P]
    N,
    I,  # noqa: E741
    ref_sb,
    out_ptr,  # [B, N, H, C]
    H: tl.constexpr,
    C: tl.constexpr,
    L: tl.constexpr,
    P: tl.constexpr,
    LR: tl.constexpr,
    BLOCK_LP: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    bid = tl.program_id(1)
    hid = tl.program_id(2)
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    a, _, _, dx, dy, _, _, r00, w2, _, m00, m01, m10, m11 = _sample_tile(
        shapes_ptr, ref_ptr, off_ptr, logit_ptr, bid, hid, offs_n, N, I, ref_sb,
        H, L, P, LR, BLOCK_LP,
    )  # fmt: skip
    w00 = (a * (1 - dx) * (1 - dy))[:, :, None]
    w01 = (a * dx * (1 - dy))[:, :, None]
    w10 = (a * (1 - dx) * dy)[:, :, None]
    w11 = (a * dx * dy)[:, :, None]
    r00 = r00[:, :, None]
    r01 = r00 + H
    r10 = r00 + (w2 * H)[:, :, None]
    r11 = r10 + H
    m00 = m00[:, :, None]
    m01 = m01[:, :, None]
    m10 = m10[:, :, None]
    m11 = m11[:, :, None]

    m_n = offs_n < N
    q = (bid * N + offs_n) * H + hid
    offs_c = tl.arange(0, BLOCK_C)
    mc = (offs_c < C)[None, None, :]
    cc = offs_c[None, None, :]
    acc = w00 * tl.load(value_ptr + r00 * C + cc, mask=m00 & mc, other=0.0).to(tl.float32)
    acc += w01 * tl.load(value_ptr + r01 * C + cc, mask=m01 & mc, other=0.0).to(tl.float32)
    acc += w10 * tl.load(value_ptr + r10 * C + cc, mask=m10 & mc, other=0.0).to(tl.float32)
    acc += w11 * tl.load(value_ptr + r11 * C + cc, mask=m11 & mc, other=0.0).to(tl.float32)
    tl.store(
        out_ptr + q[:, None] * C + offs_c[None, :],
        tl.sum(acc, axis=1).to(out_ptr.dtype.element_ty),
        mask=m_n[:, None] & (offs_c < C)[None, :],
    )
