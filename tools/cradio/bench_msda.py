# Copyright (c) OpenMMLab. All rights reserved.
"""Benchmark multi-scale deformable attention at ViT-CoMer shapes.

Compares, forward + backward:

* ``unfused``: PyTorch softmax and sampling locations, vendored Triton forward
  (``msda_triton_kernels.py``) and atomic-free backward (``msda_triton_bwd.py``);
* ``fused``: ``msda_triton_fused.py`` (softmax, locations, bf16 IO in-kernel);

once at the op level (from the raw projections) and once for a full
``MSDeformAttn`` (fp32 and bf16 autocast). The variants are timed interleaved
and the best of ``--rounds`` is reported, so the comparison stays usable on a
shared GPU::

    python tools/cradio/bench_msda.py --batch 2
"""

import argparse
from collections.abc import Callable

import torch
from torch import Tensor
from triton.testing import do_bench

from mmrotate.models.backbones.cradiov4.ms_deform_attn import (
    MSDeformAttn,
    _TritonMSDAFunction,
)
from mmrotate.models.backbones.cradiov4.msda_triton_fused import FusedMSDAFunction

# name: (d_model, heads, value ratio, image size)
MODELS = {
    'C-RADIOv4-H CoMer': (1280, 16, 0.5, 1024),
    'DINOv3-S CoMer': (384, 6, 1.0, 800),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Benchmark MSDA kernels')
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--points', type=int, default=4)
    parser.add_argument('--rounds', type=int, default=5)
    return parser.parse_args()


def best_ms(fns: dict[str, Callable[[], object]], rounds: int) -> dict[str, float]:
    """Best-of-``rounds`` time of each function, measured interleaved."""
    for fn in fns.values():  # compile / autotune
        fn()
    times: dict[str, list[float]] = {k: [] for k in fns}
    for _ in range(rounds):
        for k, fn in fns.items():
            times[k].append(do_bench(fn, warmup=10, rep=50))
    return {k: min(v) for k, v in times.items()}


def unfused_op(
    value: Tensor, shapes: Tensor, ref: Tensor, offsets: Tensor, logits: Tensor
) -> Tensor:
    B, N, H, L, P, _ = offsets.shape
    attn = logits.softmax(-1).view(B, N, H, L, P)
    normalizer = torch.stack([shapes[..., 1], shapes[..., 0]], -1)
    loc = ref[:, :, None, :, None, :] + offsets / normalizer[None, None, None, :, None, :]
    return _TritonMSDAFunction.apply(value, shapes, loc, attn)


def op_inputs(
    B: int, N: int, H: int, C: int, P: int, shapes: list[tuple[int, int]], dtype: torch.dtype
) -> tuple[Tensor, ...]:
    L = len(shapes)
    num_value = sum(h * w for h, w in shapes)
    value = torch.randn(B, num_value, H, C, device='cuda', dtype=dtype, requires_grad=True)
    ref = torch.rand(1, N, 1, 2, device='cuda')
    offsets = torch.randn(B, N, H, L, P, 2, device='cuda') * 2
    offsets = offsets.to(dtype).requires_grad_(True)
    logits = torch.randn(B, N, H, L * P, device='cuda', dtype=dtype, requires_grad=True)
    grad = torch.randn(B, N, H, C, device='cuda', dtype=dtype)
    return value, torch.tensor(shapes, device='cuda'), ref, offsets, logits, grad


def bench_op(args: argparse.Namespace) -> None:
    print('\n## Op level, fwd + bwd from the raw projections (ms)\n')
    print('| model | call | dtype | unfused | fused | speed-up |')
    print('|---|---|---|---|---|---|')
    for model, (dim, H, ratio, size) in MODELS.items():
        C = int(dim * ratio) // H
        pyramid = [(size // 8,) * 2, (size // 16,) * 2, (size // 32,) * 2]
        vit = [(size // 16,) * 2]
        calls = {
            'injector': (vit[0][0] ** 2, pyramid),
            'extractor': (sum(h * w for h, w in pyramid), vit),
        }
        for call, (N, shapes) in calls.items():
            for dtype in (torch.float32, torch.bfloat16):
                value, sh, ref, off, lg, grad = op_inputs(
                    args.batch, N, H, C, args.points, shapes, dtype
                )

                def step(fn: Callable[..., Tensor]) -> Callable[[], None]:
                    def run() -> None:
                        out = fn(value, sh, ref, off, lg)  # noqa: B023
                        out.backward(grad)  # noqa: B023

                    return run

                t = best_ms(
                    {'unfused': step(unfused_op), 'fused': step(FusedMSDAFunction.apply)},
                    args.rounds,
                )
                print(
                    f'| {model} (H={H}, C={C}) | {call} | {str(dtype)[6:]} | '
                    f'{t["unfused"]:.3f} | {t["fused"]:.3f} | {t["unfused"] / t["fused"]:.1f}x |'
                )


def bench_module(args: argparse.Namespace) -> None:
    print('\n## MSDeformAttn fwd + bwd, injector call (ms)\n')
    print('| model | autocast | unfused | fused | speed-up |')
    print('|---|---|---|---|---|')
    for model, (dim, H, ratio, size) in MODELS.items():
        pyramid = [(size // 8,) * 2, (size // 16,) * 2, (size // 32,) * 2]
        shapes = torch.tensor(pyramid, device='cuda')
        starts = torch.cat([shapes.new_zeros(1), shapes.prod(1).cumsum(0)[:-1]])
        N = (size // 16) ** 2
        query = torch.randn(args.batch, N, dim, device='cuda', requires_grad=True)
        feat = torch.randn(args.batch, int(shapes.prod(1).sum()), dim, device='cuda')
        feat.requires_grad_(True)
        ref = torch.rand(1, N, 1, 2, device='cuda')
        mods = {
            k: MSDeformAttn(dim, 3, H, args.points, ratio, fused=k == 'fused').cuda()
            for k in ('unfused', 'fused')
        }
        for amp in (False, True):

            def step(m: MSDeformAttn, amp: bool = amp) -> Callable[[], None]:
                def run() -> None:
                    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                        out = m(query, ref, feat, shapes, starts)  # noqa: B023
                    out.float().sum().backward()

                return run

            t = best_ms({k: step(m) for k, m in mods.items()}, args.rounds)
            print(
                f'| {model} | {"bf16" if amp else "off"} | {t["unfused"]:.3f} | '
                f'{t["fused"]:.3f} | {t["unfused"] / t["fused"]:.1f}x |'
            )


def main() -> None:
    args = parse_args()
    print(f'{torch.cuda.get_device_name()}, batch {args.batch}, {args.points} points')
    bench_op(args)
    bench_module(args)


if __name__ == '__main__':
    main()
