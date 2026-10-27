# Copyright (c) OpenMMLab. All rights reserved.
"""Fused multi-scale deformable attention (Triton), FlashAttention-2 style.

Compared with the vendored kernels (``msda_triton_kernels.py``) and the
atomic-free backward (``msda_triton_bwd.py``):

* **Nothing is materialised between the projections and the sampling.** The
  kernel takes the raw ``sampling_offsets`` / ``attention_weights`` linear
  outputs and the reference points, and computes the softmax over ``L * P``
  and ``loc = ref + offset / (w, h)`` in registers; their backward runs in the
  per-query backward pass. Only the kernel inputs are saved for backward.
* **Hardware-shaped tiles.** ``C`` and ``H`` are compile-time constants, so the
  gather rows ``(pixel * H + head) * C`` are provably aligned and loaded with
  vector instructions (a runtime ``C = 40`` was ~6x slower than ``C = 32``).
  Each program handles ``BLOCK_N`` queries (FlashAttention's ``BLOCK_M``).
* **Low-precision IO, fp32 math.** ``value`` and the projections are read in
  their own dtype (e.g. bf16 under autocast) and upcast in registers; sums are
  accumulated in fp32.
* **FlashAttention-2 backward split**, as in ``msda_triton_bwd.py``: a
  per-query pass (the dQ side) and a per-value-row pass (the dK/dV side) that
  owns its output, so there are no float atomics. Unlike FlashAttention, the
  value pass does not recompute the attention weights from a stored
  log-sum-exp: that took six dependent scalar gathers per sample, so the query
  pass writes each sample's four corner weights (16 bytes) instead.

The value tensor is L2-resident at ViT-CoMer sizes and sampling locality has
no measurable effect, so FlashAttention's explicit SRAM tiling of K/V does not
apply: the gathers are data-dependent. Only ``padding_mode='zeros'`` with
``align_corners=False`` (Deformable-DETR) is implemented.
"""

import torch
import triton
from torch import Tensor
from torch.amp import custom_bwd, custom_fwd
from torch.autograd.function import Function, FunctionCtx, once_differentiable
from triton import language as tl

from .msda_triton_bwd import _segment_bounds_kernel


@triton.jit
def _sample_tile(  # noqa: ANN202 - Triton helper returning a tuple of tensors
    shapes_ptr,
    ref_ptr,
    off_ptr,
    logit_ptr,
    bid,
    hid,
    offs_n,
    N,
    I,  # noqa: E741
    ref_sb,
    H: tl.constexpr,
    L: tl.constexpr,
    P: tl.constexpr,
    LR: tl.constexpr,
    BLOCK_LP: tl.constexpr,
):
    """Softmax weights and bilinear stencils of a ``[BLOCK_N, BLOCK_LP]`` tile."""
    offs_s = tl.arange(0, BLOCK_LP)
    m_s = offs_s < L * P
    lvl = offs_s // P
    hl = tl.load(shapes_ptr + lvl * 2, mask=m_s, other=1).to(tl.int32)
    wl = tl.load(shapes_ptr + lvl * 2 + 1, mask=m_s, other=1).to(tl.int32)
    sl = tl.zeros([BLOCK_LP], dtype=tl.int32)
    for lv in tl.static_range(L - 1):
        size = (tl.load(shapes_ptr + lv * 2) * tl.load(shapes_ptr + lv * 2 + 1)).to(tl.int32)
        sl += tl.where(lvl > lv, size, 0)

    m_n = offs_n < N
    m = m_n[:, None] & m_s[None, :]
    q = (bid * N + offs_n) * H + hid
    s = q[:, None] * (L * P) + offs_s[None, :]

    logit = tl.load(logit_ptr + s, mask=m, other=0.0).to(tl.float32)
    logit = tl.where(m_s[None, :], logit, float('-inf'))
    mx = tl.max(logit, axis=1)
    e = tl.exp(logit - mx[:, None])
    den = tl.sum(e, axis=1)
    a = e / den[:, None]

    if LR == 1:
        r = ref_ptr + bid * ref_sb + offs_n[:, None] * 2 + offs_s[None, :] * 0
    else:
        r = ref_ptr + bid * ref_sb + (offs_n[:, None] * LR + lvl[None, :]) * 2
    rx = tl.load(r, mask=m, other=0.0).to(tl.float32)
    ry = tl.load(r + 1, mask=m, other=0.0).to(tl.float32)
    ox = tl.load(off_ptr + s * 2, mask=m, other=0.0).to(tl.float32)
    oy = tl.load(off_ptr + s * 2 + 1, mask=m, other=0.0).to(tl.float32)

    # grid_sample(align_corners=False) coordinates and bilinear stencil
    wf = wl.to(tl.float32)[None, :]
    hf = hl.to(tl.float32)[None, :]
    xs = (rx + ox / wf) * wf - 0.5
    ys = (ry + oy / hf) * hf - 0.5
    x0f = tl.floor(xs)
    y0f = tl.floor(ys)
    dx = xs - x0f
    dy = ys - y0f
    # clamp before the int cast so far-away samples cannot overflow
    w2 = wl[None, :]
    h2 = hl[None, :]
    x0 = tl.minimum(tl.maximum(x0f, -2.0), wf).to(tl.int32)
    y0 = tl.minimum(tl.maximum(y0f, -2.0), hf).to(tl.int32)
    vx0 = (x0 >= 0) & (x0 < w2)
    vx1 = (x0 >= -1) & (x0 + 1 < w2)
    vy0 = (y0 >= 0) & (y0 < h2)
    vy1 = (y0 >= -1) & (y0 + 1 < h2)
    # value row (b, pixel, head) of the top-left corner
    r00 = ((bid * I + sl[None, :] + y0 * w2 + x0) * H) + hid
    return (
        a,
        s,
        m,
        dx,
        dy,
        x0,
        y0,
        r00,
        w2,
        h2,
        m & vy0 & vx0,
        m & vy0 & vx1,
        m & vy1 & vx0,
        m & vy1 & vx1,
    )


def _fwd_configs() -> list[triton.Config]:
    return [
        triton.Config({'BLOCK_N': bn}, num_warps=nw)
        for bn in (1, 2, 4, 8, 16)
        for nw in (1, 2, 4, 8)
    ]


@triton.autotune(configs=_fwd_configs(), key=['H', 'C', 'L', 'P', 'LR'])
@triton.jit
def _msda_fused_fwd_kernel(
    out_ptr,  # [B, N, H, C]
    value_ptr,  # [B, I, H, C]
    shapes_ptr,  # [L, 2] (h, w)
    ref_ptr,  # [B or 1, N, LR, 2]
    off_ptr,  # [B, N, H, L, P, 2]
    logit_ptr,  # [B, N, H, L * P]
    N,
    I,  # noqa: E741
    ref_sb,
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


def msda_fused_fwd(
    value: Tensor,
    spatial_shapes: Tensor,
    reference_points: Tensor,
    offsets: Tensor,
    logits: Tensor,
) -> Tensor:
    """Forward of MSDA from the raw projections.

    Args:
        value: ``(B, I, H, C)``, any float dtype.
        spatial_shapes: ``(L, 2)`` (h, w) per level.
        reference_points: ``(B or 1, N, 1 or L, 2)`` normalised (x, y).
        offsets: ``(B, N, H, L, P, 2)`` sampling offsets in pixels of each level.
        logits: ``(B, N, H, L * P)`` attention logits (softmax is applied here).

    Returns:
        ``(B, N, H, C)`` in ``value.dtype``.
    """
    B, I, H, C = value.shape  # noqa: E741
    _, N, _, L, P, _ = offsets.shape
    LR = reference_points.shape[2]
    if LR not in (1, L):
        raise ValueError(f'reference_points must have 1 or {L} levels, got {LR}')
    if max(value.numel(), offsets.numel(), B * N * H * C) >= 2**31:
        raise ValueError('MSDA inputs too large for int32 indexing')
    value = value.contiguous()
    ref = reference_points.contiguous()
    offsets = offsets.contiguous()
    logits = logits.contiguous()
    ref_sb = 0 if ref.shape[0] == 1 else N * LR * 2

    out = value.new_empty(B, N, H, C)

    def grid(meta: dict) -> tuple[int, int, int]:
        return triton.cdiv(N, meta['BLOCK_N']), B, H

    _msda_fused_fwd_kernel[grid](
        out, value, spatial_shapes, ref, offsets, logits, N, I, ref_sb,
        H=H, C=C, L=L, P=P, LR=LR,
        BLOCK_LP=triton.next_power_of_2(L * P),
        BLOCK_C=triton.next_power_of_2(C),
    )  # fmt: skip
    return out


def _bwd_query_configs() -> list[triton.Config]:
    return [triton.Config({'BLOCK_N': bn}, num_warps=nw) for bn in (1, 2, 4, 8) for nw in (1, 2, 4)]


@triton.autotune(configs=_bwd_query_configs(), key=['H', 'C', 'L', 'P', 'LR'])
@triton.jit
def _msda_fused_bwd_query_kernel(
    grad_off_ptr,  # [B, N, H, L, P, 2]
    grad_logit_ptr,  # [B, N, H, L * P]
    key_ptr,  # [B, N, H, L * P] int32
    wts_ptr,  # [B, N, H, L * P, 4] fp32 bilinear corner weights times attention
    out_grad_ptr,  # [B, N, H, C]
    value_ptr,
    shapes_ptr,
    ref_ptr,
    off_ptr,
    logit_ptr,
    N,
    I,  # noqa: E741
    IP,
    RP,
    ref_sb,
    H: tl.constexpr,
    C: tl.constexpr,
    L: tl.constexpr,
    P: tl.constexpr,
    LR: tl.constexpr,
    BLOCK_LP: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    """Per-query gradients (FlashAttention's dQ side) and the value-pass keys."""
    bid = tl.program_id(1)
    hid = tl.program_id(2)
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    a, s, m, dx, dy, x0, y0, r00, w2, h2, m00, m01, m10, m11 = _sample_tile(
        shapes_ptr, ref_ptr, off_ptr, logit_ptr, bid, hid, offs_n, N, I, ref_sb,
        H, L, P, LR, BLOCK_LP,
    )  # fmt: skip
    m_n = offs_n < N
    q = (bid * N + offs_n) * H + hid
    offs_c = tl.arange(0, BLOCK_C)
    mc = (offs_c < C)[None, None, :]
    cc = offs_c[None, None, :]
    og = tl.load(
        out_grad_ptr + q[:, None] * C + offs_c[None, :],
        mask=m_n[:, None] & (offs_c < C)[None, :],
        other=0.0,
    ).to(tl.float32)[:, None, :]
    r00 = r00[:, :, None]
    r10 = r00 + (w2 * H)[:, :, None]
    v00 = tl.load(value_ptr + r00 * C + cc, mask=m00[:, :, None] & mc, other=0.0).to(tl.float32)
    v01 = tl.load(value_ptr + (r00 + H) * C + cc, mask=m01[:, :, None] & mc, other=0.0).to(
        tl.float32
    )
    v10 = tl.load(value_ptr + r10 * C + cc, mask=m10[:, :, None] & mc, other=0.0).to(tl.float32)
    v11 = tl.load(value_ptr + (r10 + H) * C + cc, mask=m11[:, :, None] & mc, other=0.0).to(
        tl.float32
    )

    dx3 = dx[:, :, None]
    dy3 = dy[:, :, None]
    sample = (v00 * (1 - dx3) + v01 * dx3) * (1 - dy3) + (v10 * (1 - dx3) + v11 * dx3) * dy3
    da = tl.sum(sample * og, axis=2)
    gx = tl.sum(og * ((1 - dy3) * (v01 - v00) + dy3 * (v11 - v10)), axis=2)
    gy = tl.sum(og * ((1 - dx3) * (v10 - v00) + dx3 * (v11 - v01)), axis=2)
    # softmax backward in registers; d offset = d loc / (w, h) cancels the scale
    dlogit = a * (da - tl.sum(a * da, axis=1)[:, None])
    tl.store(grad_logit_ptr + s, dlogit.to(grad_logit_ptr.dtype.element_ty), mask=m)
    tl.store(grad_off_ptr + s * 2, (a * gx).to(grad_off_ptr.dtype.element_ty), mask=m)
    tl.store(grad_off_ptr + s * 2 + 1, (a * gy).to(grad_off_ptr.dtype.element_ty), mask=m)

    # top-left corner on the grid padded by one at the top/left; samples that
    # touch no pixel get the sentinel key RP (sorted last, never read)
    lvl = tl.arange(0, BLOCK_LP) // P
    psl = tl.zeros([BLOCK_LP], dtype=tl.int32)
    for lv in tl.static_range(L - 1):
        hh = tl.load(shapes_ptr + lv * 2).to(tl.int32)
        ww = tl.load(shapes_ptr + lv * 2 + 1).to(tl.int32)
        psl += tl.where(lvl > lv, (hh + 1) * (ww + 1), 0)
    touches = m & (x0 >= -1) & (x0 < w2) & (y0 >= -1) & (y0 < h2)
    pkey = (bid * IP + psl[None, :] + (y0 + 1) * (w2 + 1) + (x0 + 1)) * H + hid
    tl.store(key_ptr + s, tl.where(touches, pkey, RP), mask=m)
    tl.store(wts_ptr + s * 4, a * (1 - dx) * (1 - dy), mask=m)
    tl.store(wts_ptr + s * 4 + 1, a * dx * (1 - dy), mask=m)
    tl.store(wts_ptr + s * 4 + 2, a * (1 - dx) * dy, mask=m)
    tl.store(wts_ptr + s * 4 + 3, a * dx * dy, mask=m)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': bs}, num_warps=nw) for bs in (2, 4, 8, 16) for nw in (1, 2, 4)
    ],
    key=['H', 'C', 'L', 'P'],
)
@triton.jit
def _msda_fused_bwd_value_kernel(
    grad_value_ptr,  # [B, I, H, C]
    out_grad_ptr,  # [B, N, H, C]
    wts_ptr,  # [S, 4] corner weights
    perm_ptr,  # sorted position -> sample index
    lo_ptr,
    hi_ptr,
    shapes_ptr,
    I,  # noqa: E741
    IP,
    H: tl.constexpr,
    C: tl.constexpr,
    L: tl.constexpr,
    P: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_C: tl.constexpr,
) -> None:
    """Value gradient (FlashAttention's dK/dV side): one program owns one row."""
    r = tl.program_id(0)  # owned value row (b, pixel, h)
    hid = r % H
    bid = r // H // I
    pix = r // H % I
    start = 0
    pstart = 0
    w = 1
    size_sum = 0
    psize_sum = 0
    for lv in tl.static_range(L):
        hh = tl.load(shapes_ptr + lv * 2).to(tl.int32)
        ww = tl.load(shapes_ptr + lv * 2 + 1).to(tl.int32)
        here = pix >= size_sum
        start = tl.where(here, size_sum, start)
        pstart = tl.where(here, psize_sum, pstart)
        w = tl.where(here, ww, w)
        size_sum += hh * ww
        psize_sum += (hh + 1) * (ww + 1)
    local = pix - start
    py = local // w
    px = local % w

    # The 4 roles of this pixel in a stencil: top-left corner of the sample at
    # (x, y), (x - 1, y), (x, y - 1), (x - 1, y - 1). All 4 segments are
    # processed together as one [4, BLOCK_S] tile.
    role = tl.arange(0, 4)
    rx = role % 2  # pixel is a right corner
    ry = role // 2  # pixel is a bottom corner
    keys = (bid * IP + pstart + (py + 1 - ry) * (w + 1) + (px + 1 - rx)) * H + hid
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
        wt = tl.load(wts_ptr + s * 4 + role[:, None], mask=m, other=0.0)
        g = tl.load(
            out_grad_ptr + (s // (L * P))[:, :, None] * C + offs_c[None, None, :],
            mask=m[:, :, None] & m_c[None, None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(tl.sum(wt[:, :, None] * g, axis=1), axis=0)
    tl.store(grad_value_ptr + r * C + offs_c, acc.to(grad_value_ptr.dtype.element_ty), mask=m_c)


def msda_fused_bwd(
    out_grad: Tensor,
    value: Tensor,
    spatial_shapes: Tensor,
    reference_points: Tensor,
    offsets: Tensor,
    logits: Tensor,
    ref_grad: bool = False,
) -> tuple[Tensor, Tensor | None, Tensor, Tensor]:
    """Gradients of :func:`msda_fused_fwd`, without float atomics.

    Returns:
        Gradients of ``value``, ``reference_points`` (``None`` unless
        ``ref_grad``), ``offsets`` and ``logits``, each in its input's dtype.
    """
    B, I, H, C = value.shape  # noqa: E741
    _, N, _, L, P, _ = offsets.shape
    LR = reference_points.shape[2]
    out_grad = out_grad.contiguous()
    value = value.contiguous()
    ref = reference_points.contiguous()
    offsets = offsets.contiguous()
    logits = logits.contiguous()
    ref_sb = 0 if ref.shape[0] == 1 else N * LR * 2
    IP = int((spatial_shapes + 1).prod(1).sum())
    RP = B * IP * H  # number of padded keys, also the "touches nothing" key
    S = B * N * H * L * P
    consts = dict(H=H, C=C, L=L, P=P, LR=LR, BLOCK_C=triton.next_power_of_2(C))

    # d ref sums d offset over heads and points: keep the terms in fp32
    grad_off = torch.empty_like(offsets, dtype=torch.float32 if ref_grad else None)
    grad_logits = torch.empty_like(logits)
    keys = torch.empty(S, dtype=torch.int32, device=value.device)
    wts = torch.empty(S, 4, dtype=torch.float32, device=value.device)

    def grid(meta: dict) -> tuple[int, int, int]:
        return triton.cdiv(N, meta['BLOCK_N']), B, H

    _msda_fused_bwd_query_kernel[grid](
        grad_off, grad_logits, keys, wts, out_grad, value, spatial_shapes, ref, offsets, logits,
        N, I, IP, RP, ref_sb,
        BLOCK_LP=triton.next_power_of_2(L * P), **consts,
    )  # fmt: skip

    sorted_keys, perm = torch.sort(keys)
    lo = torch.zeros(RP + 1, dtype=torch.int32, device=value.device)
    hi = torch.zeros(RP + 1, dtype=torch.int32, device=value.device)
    _segment_bounds_kernel[(triton.cdiv(S, 1024),)](sorted_keys, lo, hi, S, BLOCK=1024)

    grad_value = torch.empty_like(value)
    _msda_fused_bwd_value_kernel[(B * I * H,)](
        grad_value, out_grad, wts, perm, lo, hi, spatial_shapes, I, IP,
        H=H, C=C, L=L, P=P, BLOCK_C=consts['BLOCK_C'],
    )  # fmt: skip

    grad_ref = None
    if ref_grad:
        # d ref = d loc = d offset * (w, h), summed over heads, points (, levels)
        wh = spatial_shapes.flip(1).to(torch.float32)[:, None, :]
        grad_loc = grad_off * wh
        grad_ref = grad_loc.sum(dim=(2, 3, 4))[:, :, None] if LR == 1 else grad_loc.sum(dim=(2, 4))
        if ref.shape[0] == 1:
            grad_ref = grad_ref.sum(0, keepdim=True)
        grad_ref = grad_ref.to(ref.dtype)
    return grad_value, grad_ref, grad_off.to(offsets.dtype), grad_logits


class FusedMSDAFunction(Function):
    """Autograd wrapper; saves only the kernel inputs."""

    @staticmethod
    @custom_fwd(device_type='cuda')
    def forward(
        ctx: FunctionCtx,
        value: Tensor,
        spatial_shapes: Tensor,
        reference_points: Tensor,
        offsets: Tensor,
        logits: Tensor,
    ) -> Tensor:
        ctx.save_for_backward(value, spatial_shapes, reference_points, offsets, logits)
        return msda_fused_fwd(value, spatial_shapes, reference_points, offsets, logits)

    @staticmethod
    @once_differentiable
    @custom_bwd(device_type='cuda')
    def backward(ctx: FunctionCtx, grad_out: Tensor) -> tuple[Tensor | None, ...]:
        value, spatial_shapes, ref, offsets, logits = ctx.saved_tensors
        grad_value, grad_ref, grad_off, grad_logits = msda_fused_bwd(
            grad_out, value, spatial_shapes, ref, offsets, logits, ctx.needs_input_grad[2]
        )
        return grad_value, None, grad_ref, grad_off, grad_logits
