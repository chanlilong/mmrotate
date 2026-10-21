# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import pytest
import torch

from mmrotate.models.backbones import CRADIOv4, CRADIOv4CoMer
from mmrotate.models.backbones.cradiov4.checkpoint import load_radio_state_dict
from mmrotate.models.backbones.cradiov4.ms_deform_attn import (
    HAS_TRITON,
    MSDeformAttn,
    multi_scale_deformable_attn,
)
from mmrotate.models.backbones.cradiov4.radio_vit import RADIOVisionTransformer

HF_DIR = '/mnt/zfs/models/nvidia/C-RADIOv4-H'
TINY_VIT = dict(embed_dim=64, depth=4, num_heads=4, max_img_size=256)
TINY_COMER = dict(
    vit_cfg=TINY_VIT,
    interaction_indexes=[[0, 0], [1, 1], [2, 2], [3, 3]],
    conv_inplane=16,
    deform_num_heads=4,
    norm_cfg=dict(type='BN'),
    msda_backend='pytorch',
)


def _msda_inputs(device: str) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    shapes = torch.tensor([[16, 24], [8, 12], [4, 6]], device=device)
    num_value = int(shapes.prod(1).sum())
    value = torch.randn(2, num_value, 4, 40, device=device, requires_grad=True)
    loc = torch.rand(2, 100, 4, 3, 4, 2, device=device) * 1.2 - 0.1
    loc.requires_grad_(True)
    attn = torch.rand(2, 100, 4, 3, 4, device=device).softmax(-1).requires_grad_(True)
    return value, shapes, loc, attn


@pytest.mark.skipif(not (torch.cuda.is_available() and HAS_TRITON), reason='requires CUDA+triton')
def test_msda_triton_matches_pytorch() -> None:
    value, shapes, loc, attn = _msda_inputs('cuda')
    out_t = multi_scale_deformable_attn(value, shapes, loc, attn, backend='triton')
    grads_t = torch.autograd.grad(out_t.square().sum(), (value, loc, attn))
    out_p = multi_scale_deformable_attn(value, shapes, loc, attn, backend='pytorch')
    grads_p = torch.autograd.grad(out_p.square().sum(), (value, loc, attn))
    torch.testing.assert_close(out_t, out_p, atol=1e-4, rtol=1e-4)
    for g_t, g_p in zip(grads_t, grads_p, strict=True):
        torch.testing.assert_close(g_t, g_p, atol=1e-3, rtol=1e-3)


def test_msdeformattn_shapes() -> None:
    attn = MSDeformAttn(d_model=64, n_levels=3, n_heads=4, n_points=4, ratio=0.5, backend='pytorch')
    value, shapes, _, _ = _msda_inputs('cpu')
    feat = torch.randn(2, value.shape[1], 64)
    query = torch.randn(2, 10, 64)
    ref = torch.rand(1, 10, 1, 2)
    starts = torch.cat([shapes.new_zeros(1), shapes.prod(1).cumsum(0)[:-1]])
    assert attn(query, ref, feat, shapes, starts).shape == (2, 10, 64)


def test_radio_vit_shapes() -> None:
    vit = RADIOVisionTransformer(**TINY_VIT)
    summary, feats = vit(torch.rand(2, 3, 64, 96))
    assert summary.shape == (2, 128)
    assert feats.shape == (2, 4 * 6, 64)
    outs = vit.forward_features(torch.rand(1, 3, 64, 96), [1, -1])
    assert [o.shape for o in outs] == [(1, 64, 4, 6)] * 2
    with pytest.raises(ValueError):
        vit(torch.rand(1, 3, 60, 64))


@pytest.mark.parametrize('out_strides', [(8, 16, 32), (4, 8, 16, 32)])
def test_cradiov4_backbone(out_strides: tuple[int, ...]) -> None:
    model = CRADIOv4(
        out_indices=list(range(len(out_strides))), out_strides=out_strides, vit_cfg=TINY_VIT
    )
    model.train()
    assert not model.vit.training
    assert all(not p.requires_grad for p in model.parameters())
    outs = model(torch.rand(1, 3, 128, 192))
    assert [o.shape[-2:] for o in outs] == [(128 // s, 192 // s) for s in out_strides]
    # Sizes that are not a multiple of the strides are padded internally.
    outs = model(torch.rand(1, 3, 125, 130))
    assert [o.shape[-2:] for o in outs] == [(-(-125 // s), -(-130 // s)) for s in out_strides]


@pytest.mark.parametrize('out_indices', [(1, 2, 3), (0, 1, 2, 3)])
def test_cradiov4_comer_backbone(out_indices: tuple[int, ...]) -> None:
    model = CRADIOv4CoMer(out_indices=out_indices, **TINY_COMER)
    model.train()
    assert not model.vit.training
    # 160 x 224 is not a multiple of 32 in height -> padded internally.
    x = torch.rand(2, 3, 160 - 16, 224)
    outs = model(x)
    strides = [4 * 2**i for i in out_indices]
    assert [o.shape for o in outs] == [(2, 64, -(-144 // s), 224 // s) for s in strides]
    sum(o.mean() for o in outs).backward()
    assert all(p.grad is None for p in model.vit.parameters())
    # Every trainable parameter must receive a gradient (DDP without
    # find_unused_parameters relies on it).
    unused = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not unused, unused
    assert (model.spm.fc1 is not None) == (0 in out_indices)
    # gamma=0 at init, but CTI_toV still gets gradients through the gate.
    assert model.interactions[0].cti_tov.gamma.grad is not None


def test_cradiov4_comer_unfrozen_with_cp() -> None:
    model = CRADIOv4CoMer(freeze_vit=False, vit_with_cp=True, with_cp=True, **TINY_COMER)
    model.train()
    outs = model(torch.rand(1, 3, 128, 128))
    sum(o.mean() for o in outs).backward()
    assert model.vit.blocks[0].attn.qkv.weight.grad is not None


@pytest.mark.skipif(not osp.isdir(HF_DIR), reason='C-RADIOv4-H weights not available')
def test_load_hf_weights_strict() -> None:
    vit = RADIOVisionTransformer()
    vit.load_state_dict(load_radio_state_dict(HF_DIR), strict=True)
