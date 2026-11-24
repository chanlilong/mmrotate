# Copyright (c) OpenMMLab. All rights reserved.
"""Gradient checkpointing for modules that contain BatchNorm.

``torch.utils.checkpoint`` runs the wrapped function twice: once in the forward
pass and once more during backward to rebuild the activations. A BatchNorm layer
in training mode updates its running statistics on every call, so a plain
checkpoint would apply the momentum update twice per step. :func:`checkpoint_bn`
runs the recomputation with a momentum of 0 (the layers still normalise with the
batch statistics, exactly as in the first call, and the running statistics stay
as they are), so the outputs, the gradients and the running statistics are the
same as without checkpointing.
"""

import contextlib
from collections.abc import Callable, Iterator, Sequence

import torch.utils.checkpoint as cp
from torch import Tensor, nn
from torch.nn.modules.batchnorm import _BatchNorm


@contextlib.contextmanager
def frozen_bn_stats(bns: Sequence[_BatchNorm]) -> Iterator[None]:
    """Training-mode BatchNorm that leaves ``running_*`` and ``num_batches_tracked`` alone.

    The running statistics are still passed to the kernel (momentum 0 makes its
    update a no-op), so autograd records the same graph as in a normal call,
    which the checkpoint recomputation relies on.
    """
    states = [(bn.momentum, bn.num_batches_tracked.clone()) for bn in bns]
    for bn in bns:
        bn.momentum = 0.0
    try:
        yield
    finally:
        for bn, (momentum, tracked) in zip(bns, states, strict=True):
            bn.momentum = momentum
            bn.num_batches_tracked.copy_(tracked)


def batch_norms(module: nn.Module) -> list[_BatchNorm]:
    """BatchNorm layers of ``module`` that update running statistics."""
    return [m for m in module.modules() if isinstance(m, _BatchNorm) and m.track_running_stats]


def checkpoint_bn(
    fn: Callable[..., Tensor], *inputs: Tensor, bns: Sequence[_BatchNorm] | None = None
) -> Tensor:
    """``checkpoint(fn, *inputs)`` that updates BatchNorm running statistics once.

    Args:
        fn: Module or function to checkpoint.
        inputs: Its tensor arguments.
        bns: BatchNorm layers that ``fn`` runs. Defaults to the BatchNorm layers
            of ``fn`` when it is a module.
    """
    if bns is None:
        bns = batch_norms(fn) if isinstance(fn, nn.Module) else []
    if not bns:
        return cp.checkpoint(fn, *inputs, use_reentrant=False)
    calls = 0

    def run(*xs: Tensor) -> Tensor:
        # The non-reentrant checkpoint calls ``run`` with grad enabled both
        # times, so the calls are told apart by counting.
        nonlocal calls
        calls += 1
        if calls == 1:
            return fn(*xs)
        with frozen_bn_stats(bns):
            return fn(*xs)

    return cp.checkpoint(run, *inputs, use_reentrant=False)
