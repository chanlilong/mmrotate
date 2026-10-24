# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from mmrotate.models.necks import ViTDetSimpleFPN
from mmrotate.models.necks.vitdet_fpn import LayerNorm2d
from mmrotate.registry import MODELS
from mmrotate.utils import register_all_modules


@pytest.mark.parametrize('out_strides', [(8, 16, 32), (4, 8, 16, 32), (16,)])
@pytest.mark.parametrize('hw', [(8, 12), (7, 9)])
def test_simple_fpn_shapes(out_strides: tuple[int, ...], hw: tuple[int, int]) -> None:
    neck = ViTDetSimpleFPN(in_channels=64, out_channels=32, out_strides=out_strides)
    neck.init_weights()
    h, w = hw
    outs = neck((torch.rand(2, 64, h, w),))
    expected = [(2, 32, -(-h * 16 // s), -(-w * 16 // s)) for s in out_strides]
    assert [o.shape for o in outs] == expected


def test_simple_fpn_vitdet_layout() -> None:
    neck = ViTDetSimpleFPN(in_channels=64, out_channels=32, out_strides=(4, 8, 16, 32))
    s4, s8, s16, s32 = (stage[0] for stage in neck.stages)
    # stride 4: deconv -> LN -> GELU -> deconv, halving the channels each time.
    assert [type(m) for m in s4] == [nn.ConvTranspose2d, LayerNorm2d, nn.GELU, nn.ConvTranspose2d]
    assert (s4[0].out_channels, s4[3].out_channels) == (32, 16)
    assert len(s8) == 1 and s8[0].out_channels == 32
    assert isinstance(s16, nn.Identity)
    assert isinstance(s32, nn.MaxPool2d)
    assert [stage[1].in_channels for stage in neck.stages] == [16, 32, 64, 64]


def test_simple_fpn_backward_and_bn() -> None:
    neck = ViTDetSimpleFPN(
        in_channels=64, out_channels=32, out_strides=(4, 8, 16, 32), norm_cfg=dict(type='BN')
    )
    assert isinstance(neck.stages[0][2], nn.BatchNorm2d)
    # Only the last backbone level is used.
    outs = neck((torch.rand(2, 64, 4, 4), torch.rand(2, 64, 8, 8)))
    assert outs[2].shape[-2:] == (8, 8)
    sum(o.mean() for o in outs).backward()
    unused = [n for n, p in neck.named_parameters() if p.grad is None]
    assert not unused, unused


def test_simple_fpn_invalid_strides() -> None:
    with pytest.raises(ValueError):
        ViTDetSimpleFPN(in_channels=64, out_strides=(32, 16))
    with pytest.raises(ValueError):
        ViTDetSimpleFPN(in_channels=64, out_strides=(12,))
    with pytest.raises(ValueError):
        ViTDetSimpleFPN(in_channels=64, out_strides=(24,))


def test_layer_norm_2d() -> None:
    norm = LayerNorm2d(8)
    nn.init.normal_(norm.weight)
    nn.init.normal_(norm.bias)
    x = torch.randn(2, 8, 5, 6)
    ref = F.layer_norm(x.permute(0, 2, 3, 1), (8,), norm.weight, norm.bias, 1e-6)
    torch.testing.assert_close(norm(x), ref.permute(0, 3, 1, 2))


def test_simple_fpn_registry() -> None:
    register_all_modules()
    neck = MODELS.build(dict(type='ViTDetSimpleFPN', in_channels=64, out_channels=32))
    assert isinstance(neck, ViTDetSimpleFPN)
    assert neck.out_strides == (8, 16, 32)
