# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import pytest
import torch

from mmrotate.models.backbones import (
    TimmFeatureBackbone,
    TimmViTCoMer,
    TimmViTSimplePyramid,
)
from mmrotate.models.backbones.timm_dinov3 import TimmViT

WEIGHTS = '/mnt/4TB/weights/timm'
VITS = 'vit_small_patch16_dinov3'
CONVNEXT = 'convnext_tiny.dinov3_lvd1689m'


def _strides(outs: tuple[torch.Tensor, ...], size: tuple[int, int]) -> list[int]:
    return [size[0] // o.shape[-2] for o in outs]


def _grads(module: torch.nn.Module) -> tuple[bool, bool]:
    """(any frozen param got a grad, all trainable params got a grad)."""
    frozen = any(p.grad is not None for p in module.parameters() if not p.requires_grad)
    trainable = [p for p in module.parameters() if p.requires_grad]
    return frozen, bool(trainable) and all(p.grad is not None for p in trainable)


def test_timm_vit_matches_timm_forward() -> None:
    vit = TimmViT(VITS).eval()
    x = torch.randn(1, 3, 64, 96)
    feat = vit.forward_features(x, [-1])[0]
    ref = vit.model.forward_features(x)[:, vit.num_skip :]
    assert feat.shape == (1, 384, 4, 6)
    torch.testing.assert_close(feat.flatten(2).transpose(1, 2), ref)


def test_simple_pyramid_shapes_and_freeze() -> None:
    model = TimmViTSimplePyramid(VITS, out_channels=32).train()
    assert not model.vit.training
    outs = model(torch.randn(2, 3, 128, 160))
    assert [o.shape[1] for o in outs] == [32] * 4
    assert _strides(outs, (128, 160)) == [4, 8, 16, 32]
    sum(o.sum() for o in outs).backward()
    frozen_grad, trainable_grad = _grads(model)
    assert not frozen_grad and trainable_grad
    assert not any(p.requires_grad for p in model.vit.parameters())


def test_comer_p2_shapes_and_freeze() -> None:
    model = TimmViTCoMer(
        VITS,
        conv_inplane=16,
        norm_cfg=dict(type='BN'),
        out_indices=(0, 1, 2, 3),
        msda_backend='pytorch',
    ).train()
    assert not model.vit.training
    # non-square, not a multiple of 32: exercises padding and the RoPE grid
    outs = model(torch.randn(1, 3, 112, 144))
    assert [tuple(o.shape[-2:]) for o in outs] == [(28, 36), (14, 18), (7, 9), (4, 5)]
    assert all(o.shape[1] == 384 for o in outs)
    sum(o.sum() for o in outs).backward()
    frozen_grad, trainable_grad = _grads(model)
    assert not frozen_grad and trainable_grad
    assert not any(p.requires_grad for p in model.vit.parameters())


def test_feature_backbone_shapes_and_freeze() -> None:
    model = TimmFeatureBackbone('convnext_tiny').train()
    assert not model.model.training
    outs = model(torch.randn(1, 3, 100, 132))
    assert [o.shape[1] for o in outs] == [96, 192, 384, 768]
    assert [tuple(o.shape[-2:]) for o in outs] == [(25, 33), (13, 17), (7, 9), (4, 5)]
    assert not any(o.requires_grad for o in outs)


@pytest.mark.skipif(not osp.isdir(WEIGHTS), reason='local timm weights not available')
def test_load_local_weights() -> None:
    vit = TimmViT(VITS, checkpoint=osp.join(WEIGHTS, f'{VITS}.lvd1689m'))
    random_vit = TimmViT(VITS)
    w = vit.model.patch_embed.proj.weight
    assert not torch.allclose(w, random_vit.model.patch_embed.proj.weight)
    cnn = TimmFeatureBackbone(CONVNEXT, checkpoint=osp.join(WEIGHTS, CONVNEXT))
    assert cnn.channels == [96, 192, 384, 768]
