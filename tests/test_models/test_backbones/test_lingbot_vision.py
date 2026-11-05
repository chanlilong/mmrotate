# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from pathlib import Path

import pytest
import torch
from torch import nn

from mmrotate.models.backbones import LingBotViT, LingBotViTCoMer
from mmrotate.models.backbones.cradiov4.lora import LoRALinear
from mmrotate.models.backbones.lingbot_vision import load_lingbot_weights
from mmrotate.models.backbones.lingbot_vision.checkpoint import convert_lingbot_state_dict
from mmrotate.models.backbones.lingbot_vision.lingbot_vit import RopePositionEmbedding

WEIGHTS = '/mnt/zfs/models/robbyant'
TINY_VIT = dict(arch='vit_small', embed_dim=32, depth=4, num_heads=2)
TINY_COMER = dict(
    arch='vit_small',
    vit_cfg=dict(embed_dim=32, depth=4, num_heads=2),
    interaction_indexes=[[0, 0], [1, 1], [2, 2], [3, 3]],
    conv_inplane=8,
    deform_num_heads=2,
    norm_cfg=dict(type='BN'),
    out_indices=(0, 1, 2, 3),
    msda_backend='pytorch',
)


def _release_style_state(vit: LingBotViT) -> dict:
    """``vit.model`` weights in the layout of the released ``model.pt``."""
    sd = {k: v.clone() for k, v in vit.model.state_dict().items()}
    for i in range(vit.depth):
        bias = sd[f'blocks.{i}.attn.qkv.bias']
        mask = torch.ones_like(bias)
        mask[bias.numel() // 3 : 2 * bias.numel() // 3] = 0
        sd[f'blocks.{i}.attn.qkv.bias'] = torch.randn_like(bias)
        sd[f'blocks.{i}.attn.qkv.bias_mask'] = mask
    sd['mask_token'] = torch.zeros(1, vit.embed_dim)
    return {'model': sd}


def _grads(module: nn.Module) -> tuple[bool, bool]:
    """(any frozen param got a grad, all trainable params got a grad)."""
    frozen = any(p.grad is not None for p in module.parameters() if not p.requires_grad)
    trainable = [p for p in module.parameters() if p.requires_grad]
    return frozen, bool(trainable) and all(p.grad is not None for p in trainable)


def test_vit_shapes_and_keys() -> None:
    vit = LingBotViT(**TINY_VIT).eval()
    assert vit.num_skip == 5 and vit.patch_size == 16
    feats = vit.forward_features(torch.randn(2, 3, 64, 96), [1, -1])
    assert [f.shape for f in feats] == [(2, 32, 4, 6)] * 2
    keys = set(vit.model.state_dict())
    assert {'cls_token', 'storage_tokens', 'patch_embed.proj.weight', 'rope_embed.periods'} <= keys
    assert 'blocks.0.ls1.gamma' in keys and 'norm.weight' in keys
    assert not any('mask' in k for k in keys)


def test_arch_presets() -> None:
    assert LingBotViT('vit_large', depth=1).embed_dim == 1024
    giant = LingBotViT('vit_giant2', depth=1)
    assert giant.blocks[0].attn.qkv.bias is None
    assert giant.blocks[0].mlp.w1.out_features == 4096  # int(6144 * 2 / 3) aligned to 8
    with pytest.raises(ValueError):
        LingBotViT('vit_tiny')
    with pytest.raises(TypeError):
        LingBotViT('vit_small', foo=1)


def test_rope_tables() -> None:
    rope = RopePositionEmbedding(32, num_heads=2).eval()
    sin, cos = rope(3, 5)
    assert sin.shape == cos.shape == (15, 16) and sin.dtype == torch.float32
    torch.testing.assert_close(sin**2 + cos**2, torch.ones_like(sin))
    # the train-mode coordinate augmentation is off by default
    rope.train()
    torch.testing.assert_close(rope(3, 5)[0], sin)


def test_convert_folds_bias_mask() -> None:
    vit = LingBotViT(**TINY_VIT)
    raw = _release_style_state(vit)
    wrapped = {'model': {f'backbone._orig_mod.{k}': v for k, v in raw['model'].items()}}
    for state in (raw, wrapped):
        sd = convert_lingbot_state_dict(state)
        assert 'mask_token' not in sd and not any(k.endswith('bias_mask') for k in sd)
        bias, mask = (
            raw['model']['blocks.0.attn.qkv.bias'],
            raw['model']['blocks.0.attn.qkv.bias_mask'],
        )
        torch.testing.assert_close(sd['blocks.0.attn.qkv.bias'], bias * mask)
        assert set(sd) == set(vit.model.state_dict())


def test_load_weights_strict(tmp_path: Path) -> None:
    src = LingBotViT(**TINY_VIT)
    torch.save(_release_style_state(src), tmp_path / 'model.pt')
    dst = LingBotViT(**TINY_VIT)
    load_lingbot_weights(dst, str(tmp_path))  # HF snapshot dir
    torch.testing.assert_close(dst.model.cls_token, src.model.cls_token)
    k_bias = dst.blocks[0].attn.qkv.bias[32:64]
    assert torch.count_nonzero(k_bias) == 0
    with pytest.raises(RuntimeError):
        load_lingbot_weights(LingBotViT(**dict(TINY_VIT, depth=3)), str(tmp_path))
    with pytest.raises(FileNotFoundError):
        load_lingbot_weights(dst, str(tmp_path / 'missing'))


def test_comer_p2_shapes_and_freeze() -> None:
    model = LingBotViTCoMer(**TINY_COMER).train()
    assert not model.vit.training
    # non-square, not a multiple of 32: exercises padding and the RoPE grid
    outs = model(torch.randn(1, 3, 112, 144))
    assert [tuple(o.shape[-2:]) for o in outs] == [(28, 36), (14, 18), (7, 9), (4, 5)]
    assert all(o.shape[1] == 32 for o in outs)
    sum(o.sum() for o in outs).backward()
    frozen_grad, trainable_grad = _grads(model)
    assert not frozen_grad and trainable_grad
    assert not any(p.requires_grad for p in model.vit.parameters())


def test_comer_rejects_short_interactions() -> None:
    with pytest.raises(ValueError):
        LingBotViTCoMer(**dict(TINY_COMER, interaction_indexes=[[0, 0], [1, 1], [2, 2]]))


def test_comer_lora_loads_after_injection(tmp_path: Path) -> None:
    src = LingBotViT(**TINY_VIT)
    torch.save(_release_style_state(src), tmp_path / 'model.pt')
    model = LingBotViTCoMer(
        **TINY_COMER, checkpoint=str(tmp_path), vit_with_cp=True, lora_cfg=dict(r=2)
    )
    model.init_weights()
    qkv = model.vit.blocks[0].attn.qkv
    assert isinstance(qkv, LoRALinear)
    torch.testing.assert_close(qkv.weight, src.blocks[0].attn.qkv.weight)
    vit_trainable = {n for n, p in model.vit.named_parameters() if p.requires_grad}
    assert vit_trainable and all('.lora_' in n for n in vit_trainable)
    model.train()
    assert not model.vit.training and qkv.training
    outs = model(torch.randn(1, 3, 64, 64))
    sum(o.sum() for o in outs).backward()
    frozen_grad, trainable_grad = _grads(model)
    assert not frozen_grad and trainable_grad


@pytest.mark.parametrize(('arch', 'name'), [('vit_small', 'small'), ('vit_large', 'large')])
def test_load_released_weights(arch: str, name: str) -> None:
    path = osp.join(WEIGHTS, f'lingbot-vision-vit-{name}')
    if not osp.isfile(osp.join(path, 'model.pt')):
        pytest.skip('LingBot-Vision weights not available')
    vit = LingBotViT(arch).eval()
    load_lingbot_weights(vit, path)
    with torch.no_grad():
        feat = vit.forward_features(torch.randn(1, 3, 64, 80), [-1])[0]
    assert feat.shape == (1, vit.embed_dim, 4, 5) and torch.isfinite(feat).all()
