# Copyright (c) OpenMMLab. All rights reserved.
"""The memory / speed switches of CRADIOv4CoMer do not change its results."""

import pytest
import torch
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm

from mmrotate.models.backbones import CRADIOv4CoMer
from mmrotate.models.backbones.cradiov4.comer_modules import MultiscaleExtractor

TINY_VIT = dict(embed_dim=64, depth=4, num_heads=4, max_img_size=256)
TINY_COMER = dict(
    vit_cfg=TINY_VIT,
    interaction_indexes=[[0, 0], [1, 1], [2, 2], [3, 3]],
    conv_inplane=16,
    deform_num_heads=4,
    norm_cfg=dict(type='BN'),
    msda_backend='pytorch',
    out_indices=(0, 1, 2, 3),
    init_values=0.5,  # open the CTI_toV gate so the ViT path carries gradients
)


def _run(model: CRADIOv4CoMer, x: Tensor) -> tuple[list[Tensor], list[Tensor], dict[str, Tensor]]:
    """Outputs, gradients of the trainable parameters and BatchNorm buffers after one step."""
    model.train()
    outs = model(x)
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(sum(o.square().mean() for o in outs), params)
    buffers = {
        f'{name}.{key}': value.clone()
        for name, m in model.named_modules()
        if isinstance(m, _BatchNorm)
        for key, value in m.state_dict().items()
    }
    return list(outs), list(grads), buffers


def _pair(**kwargs: object) -> tuple[CRADIOv4CoMer, CRADIOv4CoMer]:
    torch.manual_seed(0)
    ref = CRADIOv4CoMer(**TINY_COMER)
    model = CRADIOv4CoMer(**{**TINY_COMER, **kwargs})
    model.load_state_dict(ref.state_dict(), strict=True)
    return ref, model


@pytest.mark.parametrize(
    'kwargs',
    [
        dict(with_cp=True),
        dict(with_cp=True, vit_with_cp=True),
    ],
)
def test_same_results(kwargs: dict) -> None:
    ref, model = _pair(**kwargs)
    x = torch.rand(2, 3, 128, 160)
    outs_ref, grads_ref, bn_ref = _run(ref, x)
    outs, grads, bn = _run(model, x)
    for a, b in zip(outs, outs_ref, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
    for a, b in zip(grads, grads_ref, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-6)
    # BatchNorm (SPM, MRFP, output norms) counted the batch once, checkpointed or not.
    assert bn.keys() == bn_ref.keys()
    for key in bn:
        torch.testing.assert_close(bn[key], bn_ref[key], rtol=1e-5, atol=1e-6, msg=key)


def test_single_checkpoint_per_interaction() -> None:
    model = CRADIOv4CoMer(with_cp=True, **TINY_COMER)
    extractors = [m for m in model.modules() if isinstance(m, MultiscaleExtractor)]
    assert extractors and not any(m.with_cp for m in extractors)
    assert all(block.with_cp for block in model.interactions)
