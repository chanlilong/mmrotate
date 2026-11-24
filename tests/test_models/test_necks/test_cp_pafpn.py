# Copyright (c) OpenMMLab. All rights reserved.
"""CheckpointedCSPNeXtPAFPN is mmdet's CSPNeXtPAFPN: same keys, outputs, gradients
and BatchNorm statistics."""

import pytest
import torch
import torch.utils.checkpoint as cp
from mmdet.models.necks import CSPNeXtPAFPN
from torch import Tensor, nn

from mmrotate.models.backbones.cradiov4.bn_checkpoint import checkpoint_bn
from mmrotate.models.necks import CheckpointedCSPNeXtPAFPN
from mmrotate.registry import MODELS
from mmrotate.utils import register_all_modules

NECK = dict(
    in_channels=[8, 8, 8, 8],
    out_channels=8,
    num_csp_blocks=2,
    expand_ratio=0.5,
    norm_cfg=dict(type='BN'),
    act_cfg=dict(type='SiLU'),
)


def _inputs() -> list[Tensor]:
    torch.manual_seed(0)
    return [torch.randn(2, 8, 32 // 2**i, 48 // 2**i, requires_grad=True) for i in range(4)]


def _step(neck: nn.Module, inputs: list[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
    outs = neck(inputs)
    loss = sum(o.square().mean() for o in outs)
    grads = torch.autograd.grad(loss, [*inputs, *neck.parameters()])
    return list(outs), list(grads)


@pytest.mark.parametrize('cp_levels', [(0, 1), (0, 1, 2, 3), ()])
def test_matches_cspnext_pafpn(cp_levels: tuple[int, ...]) -> None:
    torch.manual_seed(1)
    ref = CSPNeXtPAFPN(**NECK).train()
    neck = CheckpointedCSPNeXtPAFPN(cp_levels=cp_levels, **NECK).train()
    neck.load_state_dict(ref.state_dict(), strict=True)
    assert list(neck.state_dict()) == list(ref.state_dict())

    for _ in range(2):  # two steps: the running statistics accumulate
        outs_ref, grads_ref = _step(ref, _inputs())
        outs, grads = _step(neck, _inputs())
        for a, b in zip(outs, outs_ref, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=1e-6)
        for a, b in zip(grads, grads_ref, strict=True):
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)
    # running_mean / running_var / num_batches_tracked updated once per step
    for (key, a), b in zip(neck.state_dict().items(), ref.state_dict().values(), strict=True):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7, msg=key)

    neck.eval()
    ref.eval()
    with torch.no_grad():
        for a, b in zip(neck(_inputs()), ref(_inputs()), strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=1e-6)


def test_registry() -> None:
    register_all_modules()
    neck = MODELS.build(dict(type='CheckpointedCSPNeXtPAFPN', cp_levels=[0], **NECK))
    assert isinstance(neck, CheckpointedCSPNeXtPAFPN) and neck.cp_levels == {0}


def test_checkpoint_bn_updates_statistics_once() -> None:
    x = torch.randn(4, 3, 8, 8, requires_grad=True)
    plain, safe, ref = (nn.BatchNorm2d(3).train() for _ in range(3))
    ref(x).sum().backward()
    cp.checkpoint(plain, x, use_reentrant=False).square().sum().backward()
    out = checkpoint_bn(safe, x)
    out.square().sum().backward()
    # A plain checkpoint runs BatchNorm twice ...
    assert plain.num_batches_tracked == 2
    # ... checkpoint_bn counts the batch once and leaves the layer as it was.
    assert safe.num_batches_tracked == ref.num_batches_tracked == 1
    torch.testing.assert_close(safe.running_mean, ref.running_mean)
    torch.testing.assert_close(safe.running_var, ref.running_var)
    assert safe.momentum == ref.momentum
    torch.testing.assert_close(out, ref(x.detach()), atol=1e-6, rtol=0)
