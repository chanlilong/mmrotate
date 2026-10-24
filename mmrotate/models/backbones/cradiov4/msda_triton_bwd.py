# Copyright (c) OpenMMLab. All rights reserved.
"""Atomic-free backward pass of multi-scale deformable attention (Triton).

The vendored backward (``msda_triton_kernels.py``) scatters the value gradient
with one float ``atomic_add`` per (sample, bilinear corner, channel). With
ViT-CoMer's dense CNN-token queries that is ~2G atomics per call, and the
kernel is bound by atomic throughput and contention.

This follows the FlashAttention-2 backward instead: the program that *owns*
an output accumulates it in registers and stores it once.

1. Query pass (one program per (query, batch, head), the "dQ" side): the
   gradients of the sampling locations and attention weights, which are
   per-query anyway, plus one int32 key per sample: the row of its top-left
   bilinear corner on a value grid padded by one pixel at the top/left.
2. Sort the samples by key and mark the segment bounds of every key.
3. Value pass (one program per value row (batch, pixel, head), the "dK/dV"
   side): the samples touching pixel ``(x, y)`` are exactly those with their
   top-left corner at ``(x, y)``, ``(x - 1, y)``, ``(x, y - 1)`` and
   ``(x - 1, y - 1)``. Their bilinear weights are recomputed from the sampling
   locations (recompute instead of storing, as in FlashAttention), their
   ``out_grad`` rows gathered, and the sum stored once.

There are no float atomics, so the value gradient is deterministic. Only
``padding_mode='zeros'`` with ``align_corners=False`` (Deformable-DETR) is
implemented.
"""

import torch
import triton
from torch import Tensor
from triton import language as tl


@triton.autotune(
    configs=[triton.Config({}, num_warps=nw) for nw in (1, 2, 4)],
    key=['C', 'L', 'P'],
)
@triton.jit
def _msda_bwd_query_kernel(
    out_grad_ptr,  # [B, N, H, C]
    value_ptr,  # [B, I, H, C]
    shapes_ptr,  # [L, 2] (h, w)
    starts_ptr,  # [L] level start in I
    pstarts_ptr,  # [L] level start on the padded grid
    loc_ptr,  # [B, N, H, L, P, 2]
    attn_ptr,  # [B, N, H, L, P]
    grad_loc_ptr,
    grad_attn_ptr,
    key_ptr,  # [B, N, H, L, P] int32
    I,  # noqa: E741
    IP,
    C,
    N,
    H,
    L,
    P,
    RP,
    BLOCK_L: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_C: tl.constexpr,
) -> None:
    nid = tl.program_id(0)
    bid = tl.program_id(1)
    hid = tl.program_id(2)
    q = (bid * N + nid) * H + hid

    offs_l = tl.arange(0, BLOCK_L)
    offs_p = tl.arange(0, BLOCK_P)
    offs_c = tl.arange(0, BLOCK_C)
    m_l = offs_l < L
    m_lp = m_l[:, None] & (offs_p[None, :] < P)
    m_c = offs_c < C

    hl = tl.load(shapes_ptr + offs_l * 2, mask=m_l, other=1).to(tl.int32)[:, None]
    wl = tl.load(shapes_ptr + offs_l * 2 + 1, mask=m_l, other=1).to(tl.int32)[:, None]
    sl = tl.load(starts_ptr + offs_l, mask=m_l, other=0).to(tl.int32)[:, None]
    psl = tl.load(pstarts_ptr + offs_l, mask=m_l, other=0).to(tl.int32)[:, None]

    lp = offs_l[:, None] * P + offs_p[None, :]
    base = q * L * P
    x = tl.load(loc_ptr + (base + lp) * 2, mask=m_lp, other=0.0)
    y = tl.load(loc_ptr + (base + lp) * 2 + 1, mask=m_lp, other=0.0)
    a = tl.load(attn_ptr + base + lp, mask=m_lp, other=0.0)
    og = tl.load(out_grad_ptr + q * C + offs_c, mask=m_c, other=0.0)[None, None, :]

    # grid_sample(align_corners=False) coordinates and bilinear stencil
    wf = wl.to(tl.float32)
    hf = hl.to(tl.float32)
    xs = x * wf - 0.5
    ys = y * hf - 0.5
    x0f = tl.floor(xs)
    y0f = tl.floor(ys)
    dx = xs - x0f
    dy = ys - y0f
    # clamp before the int cast so far-away samples cannot overflow
    x0 = tl.minimum(tl.maximum(x0f, -2.0), wf).to(tl.int32)
    y0 = tl.minimum(tl.maximum(y0f, -2.0), hf).to(tl.int32)
    vx0 = (x0 >= 0) & (x0 < wl)
    vx1 = (x0 >= -1) & (x0 + 1 < wl)
    vy0 = (y0 >= 0) & (y0 < hl)
    vy1 = (y0 >= -1) & (y0 + 1 < hl)

    pix00 = bid * I + sl + y0 * wl + x0
    v_off = hid * C + offs_c[None, None, :]
    mc = m_c[None, None, :]
    m00 = (m_lp & vy0 & vx0)[:, :, None] & mc
    m01 = (m_lp & vy0 & vx1)[:, :, None] & mc
    m10 = (m_lp & vy1 & vx0)[:, :, None] & mc
    m11 = (m_lp & vy1 & vx1)[:, :, None] & mc
    v00 = tl.load(value_ptr + pix00[:, :, None] * H * C + v_off, mask=m00, other=0.0)
    v01 = tl.load(value_ptr + (pix00 + 1)[:, :, None] * H * C + v_off, mask=m01, other=0.0)
    v10 = tl.load(value_ptr + (pix00 + wl)[:, :, None] * H * C + v_off, mask=m10, other=0.0)
    v11 = tl.load(value_ptr + (pix00 + wl + 1)[:, :, None] * H * C + v_off, mask=m11, other=0.0)

    dx3 = dx[:, :, None]
    dy3 = dy[:, :, None]
    sample = (v00 * (1 - dx3) + v01 * dx3) * (1 - dy3) + (v10 * (1 - dx3) + v11 * dx3) * dy3
    tl.store(grad_attn_ptr + base + lp, tl.sum(sample * og, axis=2), mask=m_lp)

    ga = og * a[:, :, None]
    gx = tl.sum(ga * ((1 - dy3) * (v01 - v00) + dy3 * (v11 - v10)), axis=2) * wf
    gy = tl.sum(ga * ((1 - dx3) * (v10 - v00) + dx3 * (v11 - v01)), axis=2) * hf
    tl.store(grad_loc_ptr + (base + lp) * 2, gx, mask=m_lp)
    tl.store(grad_loc_ptr + (base + lp) * 2 + 1, gy, mask=m_lp)

    # top-left corner on the grid padded by one at the top/left; samples that
    # touch no pixel get the sentinel key RP (sorted last, never read)
    touches = m_lp & (x0 >= -1) & (x0 < wl) & (y0 >= -1) & (y0 < hl)
    pkey = (bid * IP + psl + (y0 + 1) * (wl + 1) + (x0 + 1)) * H + hid
    tl.store(key_ptr + base + lp, tl.where(touches, pkey, RP), mask=m_lp)


@triton.jit
def _segment_bounds_kernel(keys_ptr, lo_ptr, hi_ptr, S, BLOCK: tl.constexpr) -> None:
    """``[lo[k], hi[k])`` is the run of key ``k`` in the sorted keys."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < S
    k = tl.load(keys_ptr + offs, mask=m, other=-1)
    k_prev = tl.load(keys_ptr + offs - 1, mask=m & (offs > 0), other=-1)
    k_next = tl.load(keys_ptr + offs + 1, mask=m & (offs + 1 < S), other=-1)
    tl.store(lo_ptr + k, offs, mask=m & (k != k_prev))
    tl.store(hi_ptr + k, offs + 1, mask=m & (k != k_next))


@triton.autotune(
    configs=[triton.Config({'BLOCK_S': bs}, num_warps=nw) for bs in (2, 4, 8) for nw in (1, 2, 4)],
    key=['C', 'L'],
)
@triton.jit
def _msda_bwd_value_kernel(
    grad_value_ptr,  # [B, I, H, C]
    out_grad_ptr,  # [B, N, H, C]
    loc_ptr,
    attn_ptr,
    perm_ptr,  # sorted position -> sample index
    lo_ptr,
    hi_ptr,
    shapes_ptr,
    starts_ptr,
    pstarts_ptr,
    I,  # noqa: E741
    IP,
    C,
    H,
    L,
    LP,
    BLOCK_S: tl.constexpr,
    BLOCK_C: tl.constexpr,
) -> None:
    r = tl.program_id(0)  # owned value row (b, pixel, h)
    hid = r % H
    bid = r // H // I
    pix = r // H % I
    lvl = 0
    for lv in range(1, L):
        lvl = tl.where(pix >= tl.load(starts_ptr + lv), lv, lvl)
    w = tl.load(shapes_ptr + lvl * 2 + 1).to(tl.int32)
    h = tl.load(shapes_ptr + lvl * 2).to(tl.int32)
    local = pix - tl.load(starts_ptr + lvl).to(tl.int32)
    py = local // w
    px = local % w
    pbase = bid * IP + tl.load(pstarts_ptr + lvl).to(tl.int32)
    wf = w.to(tl.float32)
    hf = h.to(tl.float32)

    # The 4 roles of this pixel in a stencil: top-left corner of the sample at
    # (x, y), (x - 1, y), (x, y - 1), (x - 1, y - 1). All 4 segments are
    # processed together as one [4, BLOCK_S] tile.
    role = tl.arange(0, 4)
    rx = role % 2  # pixel is a right corner: weight dx
    ry = role // 2  # pixel is a bottom corner: weight dy
    keys = (pbase + (py + 1 - ry) * (w + 1) + (px + 1 - rx)) * H + hid
    lo = tl.load(lo_ptr + keys)
    hi = tl.load(hi_ptr + keys)
    n_max = tl.max(hi - lo)

    offs_c = tl.arange(0, BLOCK_C)
    m_c = offs_c < C
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    for it in range(0, n_max, BLOCK_S):
        offs = lo[:, None] + it + tl.arange(0, BLOCK_S)[None, :]  # [4, S]
        m = offs < hi[:, None]
        s = tl.load(perm_ptr + offs, mask=m, other=0)
        x = tl.load(loc_ptr + s * 2, mask=m, other=0.0)
        y = tl.load(loc_ptr + s * 2 + 1, mask=m, other=0.0)
        a = tl.load(attn_ptr + s, mask=m, other=0.0)
        xs = x * wf - 0.5
        ys = y * hf - 0.5
        dx = xs - tl.floor(xs)
        dy = ys - tl.floor(ys)
        wx = tl.where(rx[:, None] == 1, dx, 1 - dx)
        wy = tl.where(ry[:, None] == 1, dy, 1 - dy)
        g = tl.load(
            out_grad_ptr + (s // LP)[:, :, None] * C + offs_c[None, None, :],
            mask=m[:, :, None] & m_c[None, None, :],
            other=0.0,
        )
        acc += tl.sum(tl.sum((a * wx * wy)[:, :, None] * g, axis=1), axis=0)
    tl.store(grad_value_ptr + r.to(tl.int64) * C + offs_c, acc, mask=m_c)


def msda_bwd_atomic_free(
    out_grad: Tensor,
    value: Tensor,
    spatial_shapes: Tensor,
    sampling_locations: Tensor,
    attention_weights: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Gradients of MSDA (zeros padding, ``align_corners=False``).

    Args:
        out_grad: ``(B, N, H, C)``.
        value: ``(B, I, H, C)``.
        spatial_shapes: ``(L, 2)`` (h, w) per level.
        sampling_locations: ``(B, N, H, L, P, 2)`` normalised (x, y).
        attention_weights: ``(B, N, H, L, P)``.

    Returns:
        Gradients of ``value``, ``sampling_locations`` and
        ``attention_weights``.
    """
    B, I, H, C = value.shape  # noqa: E741
    _, N, _, L, P, _ = sampling_locations.shape
    device = value.device
    out_grad = out_grad.contiguous()
    value = value.contiguous()
    loc = sampling_locations.contiguous()
    attn = attention_weights.contiguous()
    shapes = spatial_shapes.contiguous()
    starts = torch.cat([shapes.new_zeros(1), shapes.prod(1).cumsum(0)[:-1]])
    padded_sizes = (shapes + 1).prod(1)
    pstarts = torch.cat([shapes.new_zeros(1), padded_sizes.cumsum(0)[:-1]])
    IP = int(padded_sizes.sum())
    RP = B * IP * H  # number of padded keys, also the "touches nothing" key
    S = B * N * H * L * P

    grad_loc = torch.empty_like(loc)
    grad_attn = torch.empty_like(attn)
    keys = torch.empty(S, dtype=torch.int32, device=device)
    _msda_bwd_query_kernel[(N, B, H)](
        out_grad, value, shapes, starts, pstarts, loc, attn,
        grad_loc, grad_attn, keys,
        I, IP, C, N, H, L, P, RP,
        BLOCK_L=triton.next_power_of_2(L),
        BLOCK_P=triton.next_power_of_2(P),
        BLOCK_C=triton.next_power_of_2(C),
    )  # fmt: skip

    sorted_keys, perm = torch.sort(keys)
    lo = torch.zeros(RP + 1, dtype=torch.int32, device=device)
    hi = torch.zeros(RP + 1, dtype=torch.int32, device=device)
    _segment_bounds_kernel[(triton.cdiv(S, 1024),)](sorted_keys, lo, hi, S, BLOCK=1024)

    grad_value = torch.empty_like(value)
    _msda_bwd_value_kernel[(B * I * H,)](
        grad_value, out_grad, loc, attn, perm, lo, hi, shapes, starts, pstarts,
        I, IP, C, H, L, L * P,
        BLOCK_C=triton.next_power_of_2(C),
    )  # fmt: skip
    return grad_value, grad_loc, grad_attn
