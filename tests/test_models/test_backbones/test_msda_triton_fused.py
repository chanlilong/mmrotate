# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from torch import Tensor

from mmrotate.models.backbones.cradiov4.ms_deform_attn import (
    HAS_TRITON,
    MSDeformAttn,
    msda_pytorch,
)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAS_TRITON), reason='requires CUDA+triton'
)

CASES = [
    # (level shapes, B, N, H, C, P, reference levels, reference batch, offset spread in px)
    ([(16, 24), (8, 12), (4, 6)], 2, 100, 4, 32, 4, 1, 1, 2.0),
    ([(13, 7)], 3, 57, 3, 20, 3, 1, 3, 1.0),  # non power-of-two everything
    ([(10, 10), (5, 5)], 1, 64, 6, 64, 4, 2, 1, 15.0),  # many samples far outside
    ([(32, 32), (16, 16), (8, 8)], 2, 77, 16, 40, 4, 3, 2, 3.0),  # C-RADIOv4-H CoMer heads
    ([(16, 16)], 2, 336, 6, 64, 4, 1, 1, 3.0),  # DINOv3-S CoMer extractor
]


def _inputs(
    shapes: list[tuple[int, int]],
    B: int,
    N: int,
    H: int,
    C: int,
    P: int,
    LR: int,
    RB: int,
    spread: float,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, ...]:
    g = torch.Generator(device='cuda').manual_seed(0)
    spatial = torch.tensor(shapes, device='cuda')
    num_value = int(spatial.prod(1).sum())
    L = len(shapes)
    value = torch.randn(B, num_value, H, C, device='cuda', generator=g).to(dtype)
    ref = torch.rand(RB, N, LR, 2, device='cuda', generator=g)
    offsets = (torch.randn(B, N, H, L, P, 2, device='cuda', generator=g) * spread).to(dtype)
    logits = torch.randn(B, N, H, L * P, device='cuda', generator=g).to(dtype)
    grad_out = torch.randn(B, N, H, C, device='cuda', generator=g).to(dtype)
    return value, spatial, ref, offsets, logits, grad_out


def _reference(
    value: Tensor, spatial: Tensor, ref: Tensor, offsets: Tensor, logits: Tensor
) -> Tensor:
    """Unfused PyTorch MSDA from the raw projections, as ``MSDeformAttn`` did."""
    B, N, H, L, P, _ = offsets.shape
    attn = logits.softmax(-1).view(B, N, H, L, P)
    normalizer = torch.stack([spatial[..., 1], spatial[..., 0]], -1)
    loc = ref[:, :, None, :, None, :] + offsets / normalizer[None, None, None, :, None, :]
    return msda_pytorch(value, spatial, loc.expand(B, -1, -1, -1, -1, -1), attn)


def _fused(value: Tensor, spatial: Tensor, ref: Tensor, offsets: Tensor, logits: Tensor) -> Tensor:
    from mmrotate.models.backbones.cradiov4.msda_triton_fused import FusedMSDAFunction

    return FusedMSDAFunction.apply(value, spatial, ref, offsets, logits)


@pytest.mark.parametrize('case', CASES)
def test_fused_forward_matches_pytorch(case: tuple) -> None:
    from mmrotate.models.backbones.cradiov4.msda_triton_fused import msda_fused_fwd

    value, spatial, ref, offsets, logits, _ = _inputs(*case)
    out = msda_fused_fwd(value, spatial, ref, offsets, logits)
    torch.testing.assert_close(out, _reference(value, spatial, ref, offsets, logits))


@pytest.mark.parametrize('case', CASES)
def test_fused_gradients_match_pytorch(case: tuple) -> None:
    value, spatial, ref, offsets, logits, grad_out = _inputs(*case)
    inputs = [t.clone().requires_grad_(True) for t in (value, ref, offsets, logits)]
    out = _fused(inputs[0], spatial, *inputs[1:])
    grads = torch.autograd.grad(out, inputs, grad_out)

    ref_in = [t.clone().requires_grad_(True) for t in (value, ref, offsets, logits)]
    ref_out = _reference(ref_in[0], spatial, *ref_in[1:])
    ref_grads = torch.autograd.grad(ref_out, ref_in, grad_out)
    for name, a, b in zip(('value', 'ref', 'offsets', 'logits'), grads, ref_grads, strict=True):
        torch.testing.assert_close(a, b, atol=1e-3, rtol=1e-3, msg=lambda m, n=name: f'{n}: {m}')


@pytest.mark.parametrize('case', CASES[3:])
def test_fused_bf16_io_matches_fp32_math(case: tuple) -> None:
    """bf16 inputs are upcast in registers: same result as fp32 on the same values."""
    value, spatial, ref, offsets, logits, grad_out = _inputs(*case, dtype=torch.bfloat16)
    inputs = [t.clone().requires_grad_(True) for t in (value, ref, offsets, logits)]
    out = _fused(inputs[0], spatial, *inputs[1:])
    assert out.dtype == torch.bfloat16
    grads = torch.autograd.grad(out, inputs, grad_out)

    ref_in = [t.float().requires_grad_(True) for t in (value, ref, offsets, logits)]
    ref_out = _fused(ref_in[0], spatial, *ref_in[1:])
    ref_grads = torch.autograd.grad(ref_out, ref_in, grad_out.float())
    # only the bf16 rounding of the outputs differs
    torch.testing.assert_close(out.float(), ref_out, atol=2e-2, rtol=1e-2)
    for a, b, x in zip(grads, ref_grads, inputs, strict=True):
        assert a.dtype == x.dtype
        torch.testing.assert_close(a.float(), b, atol=2e-2, rtol=1e-2)


def test_fused_backward_is_deterministic() -> None:
    from mmrotate.models.backbones.cradiov4.msda_triton_fused import msda_fused_bwd

    value, spatial, ref, offsets, logits, grad_out = _inputs(*CASES[3])
    first = msda_fused_bwd(grad_out, value, spatial, ref, offsets, logits, True)
    second = msda_fused_bwd(grad_out, value, spatial, ref, offsets, logits, True)
    for a, b in zip(first, second, strict=True):
        assert torch.equal(a, b)


@pytest.mark.parametrize('n_levels', [1, 3])
def test_msdeformattn_fused_matches_unfused(n_levels: int) -> None:
    torch.manual_seed(0)
    fused = MSDeformAttn(d_model=128, n_levels=n_levels, n_heads=4, n_points=4, ratio=0.5)
    unfused = MSDeformAttn(
        d_model=128, n_levels=n_levels, n_heads=4, n_points=4, ratio=0.5, fused=False
    )
    unfused.load_state_dict(fused.state_dict())
    for m in (fused, unfused):
        m.cuda()
        # non-trivial offsets and weights (the init sets the weights to zero)
        torch.manual_seed(1)
        for p in m.parameters():
            p.data += torch.randn_like(p) * 0.02
    shapes = torch.tensor([(16, 16), (8, 8), (4, 4)][:n_levels], device='cuda')
    query = torch.randn(2, 64, 128, device='cuda')
    feat = torch.randn(2, int(shapes.prod(1).sum()), 128, device='cuda')
    ref = torch.rand(1, 64, 1, 2, device='cuda')
    starts = torch.cat([shapes.new_zeros(1), shapes.prod(1).cumsum(0)[:-1]])

    outs, grads = [], []
    for m in (fused, unfused):
        out = m(query, ref, feat, shapes, starts)
        out.square().sum().backward()
        outs.append(out)
        grads.append([p.grad for p in m.parameters()])
    torch.testing.assert_close(outs[0], outs[1], atol=1e-4, rtol=1e-4)
    for a, b in zip(*grads, strict=True):
        torch.testing.assert_close(a, b, atol=1e-3, rtol=1e-3)

    # bf16 autocast: the fused path keeps bf16 IO and must stay close
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out_f = fused(query, ref, feat, shapes, starts)
        out_u = unfused(query, ref, feat, shapes, starts)
    torch.testing.assert_close(out_f.float(), out_u.float(), atol=2e-2, rtol=2e-2)
