# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmrotate.models.backbones.cradiov4.ms_deform_attn import (
    HAS_TRITON,
    multi_scale_deformable_attn,
)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAS_TRITON), reason='requires CUDA+triton'
)


def _inputs(
    shapes: list[tuple[int, int]], B: int, N: int, H: int, C: int, P: int, spread: float
) -> tuple[torch.Tensor, ...]:
    g = torch.Generator(device='cuda').manual_seed(0)
    spatial = torch.tensor(shapes, device='cuda')
    num_value = int(spatial.prod(1).sum())
    L = len(shapes)
    value = torch.randn(B, num_value, H, C, device='cuda', generator=g)
    ref = torch.rand(B, N, 1, 1, 1, 2, device='cuda', generator=g)
    loc = ref + torch.randn(B, N, H, L, P, 2, device='cuda', generator=g) * spread
    attn = torch.rand(B, N, H, L * P, device='cuda', generator=g).softmax(-1).view(B, N, H, L, P)
    grad_out = torch.randn(B, N, H, C, device='cuda', generator=g)
    return value, spatial, loc.contiguous(), attn.contiguous(), grad_out


CASES = [
    # (level shapes, B, N, H, C, P, spread of the sampling offsets)
    ([(16, 24), (8, 12), (4, 6)], 2, 100, 4, 32, 4, 0.05),
    ([(13, 7)], 3, 57, 3, 20, 3, 0.1),  # non power-of-two everything
    ([(10, 10), (5, 5)], 1, 64, 6, 64, 4, 1.5),  # many samples far outside
]


@pytest.mark.parametrize('case', CASES)
def test_atomic_free_matches_vendored_and_pytorch(case: tuple) -> None:
    from mmrotate.models.backbones.cradiov4.msda_triton_bwd import msda_bwd_atomic_free
    from mmrotate.models.backbones.cradiov4.msda_triton_kernels import (
        triton_multi_scale_deformable_attention_bwd,
    )

    value, spatial, loc, attn, grad_out = _inputs(*case)
    ours = msda_bwd_atomic_free(grad_out, value, spatial, loc, attn)
    vendored = triton_multi_scale_deformable_attention_bwd(
        grad_out, value, spatial, loc, attn, 'zeros', False
    )
    for a, b in zip(ours, vendored, strict=True):
        torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)

    # PyTorch grid_sample reference through autograd
    inputs = [t.clone().requires_grad_(True) for t in (value, loc, attn)]
    out = multi_scale_deformable_attn(inputs[0], spatial, inputs[1], inputs[2], backend='pytorch')
    ref = torch.autograd.grad(out, inputs, grad_out)
    for a, b in zip(ours, ref, strict=True):
        torch.testing.assert_close(a, b, atol=1e-3, rtol=1e-3)


def test_atomic_free_is_deterministic() -> None:
    from mmrotate.models.backbones.cradiov4.msda_triton_bwd import msda_bwd_atomic_free

    value, spatial, loc, attn, grad_out = _inputs(*CASES[0])
    first = msda_bwd_atomic_free(grad_out, value, spatial, loc, attn)
    second = msda_bwd_atomic_free(grad_out, value, spatial, loc, attn)
    for a, b in zip(first, second, strict=True):
        assert torch.equal(a, b)


def test_autograd_uses_triton_backward() -> None:
    value, spatial, loc, attn, grad_out = _inputs(*CASES[0])
    inputs = [t.clone().requires_grad_(True) for t in (value, loc, attn)]
    out = multi_scale_deformable_attn(inputs[0], spatial, inputs[1], inputs[2], backend='triton')
    grads = torch.autograd.grad(out, inputs, grad_out)
    ref_in = [t.clone().requires_grad_(True) for t in (value, loc, attn)]
    ref_out = multi_scale_deformable_attn(ref_in[0], spatial, ref_in[1], ref_in[2], 'pytorch')
    ref = torch.autograd.grad(ref_out, ref_in, grad_out)
    for a, b in zip(grads, ref, strict=True):
        torch.testing.assert_close(a, b, atol=1e-3, rtol=1e-3)
