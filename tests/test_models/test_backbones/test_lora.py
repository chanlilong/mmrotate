# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Callable
from pathlib import Path

import pytest
import timm
import torch
from torch import nn

from mmrotate.models.backbones import (
    CRADIOv4,
    CRADIOv4CoMer,
    TimmViTCoMer,
    TimmViTSimplePyramid,
)
from mmrotate.models.backbones.cradiov4.cradiov4 import load_radio_weights
from mmrotate.models.backbones.cradiov4.lora import LoRALinear, inject_lora
from mmrotate.models.backbones.cradiov4.radio_vit import RADIOVisionTransformer
from mmrotate.models.backbones.timm_dinov3 import TimmViT

VITS = 'vit_small_patch16_dinov3'
TINY_VIT = dict(embed_dim=32, depth=4, num_heads=2, max_img_size=256)
TINY_COMER = dict(
    vit_cfg=TINY_VIT,
    interaction_indexes=[[0, 0], [1, 1], [2, 2], [3, 3]],
    conv_inplane=8,
    deform_num_heads=2,
    norm_cfg=dict(type='BN'),
    out_indices=(0, 1, 2, 3),
    msda_backend='pytorch',
)
LORA = dict(r=4, alpha=8)


def _randomize_lora(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, LoRALinear):
            nn.init.normal_(m.lora_B)


def test_lora_linear() -> None:
    base = nn.Linear(8, 12)
    lora = LoRALinear(base, r=4, alpha=8)
    assert lora.weight is base.weight and lora.bias is base.bias
    assert set(lora.state_dict()) == {'weight', 'bias', 'lora_A', 'lora_B'}
    assert lora.lora_A.shape == (4, 8) and lora.lora_B.shape == (12, 4)
    x = torch.randn(3, 5, 8)
    torch.testing.assert_close(lora(x), base(x))
    _randomize_lora(lora)
    expected = base(x) + x @ lora.lora_A.T @ lora.lora_B.T * 2.0
    torch.testing.assert_close(lora(x), expected)
    with pytest.raises(ValueError):
        LoRALinear(base, r=0, alpha=1)


def test_inject_lora_radio_vit() -> None:
    vit = RADIOVisionTransformer(**TINY_VIT)
    assert inject_lora(vit, r=2) == 4 * 4
    qkv = vit.blocks[0].attn.qkv
    assert isinstance(qkv, LoRALinear) and qkv.lora_A.shape == (2, 32) and qkv.scaling == 1.0

    vit = RADIOVisionTransformer(**TINY_VIT)
    assert inject_lora(vit, r=2, target_modules=['qkv'], block_indices=[-1]) == 1
    assert isinstance(vit.blocks[3].attn.qkv, LoRALinear)
    assert not isinstance(vit.blocks[3].mlp.fc1, LoRALinear)
    assert not isinstance(vit.blocks[0].attn.qkv, LoRALinear)

    with pytest.raises(ValueError):
        inject_lora(RADIOVisionTransformer(**TINY_VIT), target_modules=['w12'])


def test_inject_lora_timm_vit() -> None:
    vit = TimmViT(VITS).eval()
    x = torch.randn(1, 3, 64, 96)
    ref = vit.forward_features(x, [-1])[0]
    assert inject_lora(vit, r=8) == 12 * 4
    num = sum(p.numel() for n, p in vit.named_parameters() if '.lora_' in n)
    assert num == 12 * 8 * ((384 + 1152) + (384 + 384) + 2 * (384 + 1536))
    torch.testing.assert_close(vit.forward_features(x, [-1])[0], ref)
    _randomize_lora(vit)
    assert not torch.allclose(vit.forward_features(x, [-1])[0], ref)


def test_inject_lora_eva_qkv_bias() -> None:
    # Eva with q/k/v biases would bypass ``qkv.forward`` without the guard.
    eva = timm.models.eva.Eva(
        img_size=32, patch_size=16, embed_dim=32, depth=1, num_heads=2, qkv_bias=True
    ).eval()
    x = torch.randn(1, 3, 32, 32)
    ref = eva.forward_features(x)
    inject_lora(eva, r=2, target_modules=['qkv'])
    assert eva.blocks[0].attn.qkv_bias_separate
    torch.testing.assert_close(eva.forward_features(x), ref)
    _randomize_lora(eva)
    assert not torch.allclose(eva.forward_features(x), ref)


BACKBONES: dict[str, Callable[[dict | None], nn.Module]] = {
    'CRADIOv4': lambda lora: CRADIOv4(
        vit_cfg=TINY_VIT, out_indices=(1, 3), out_strides=(8, 16), lora_cfg=lora
    ),
    'CRADIOv4CoMer': lambda lora: CRADIOv4CoMer(lora_cfg=lora, **TINY_COMER),
    'TimmViTCoMer': lambda lora: TimmViTCoMer(
        VITS, conv_inplane=8, norm_cfg=dict(type='BN'), msda_backend='pytorch', lora_cfg=lora
    ),
    'TimmViTSimplePyramid': lambda lora: TimmViTSimplePyramid(VITS, out_channels=16, lora_cfg=lora),
}


@pytest.mark.parametrize('name', list(BACKBONES))
def test_backbone_lora(name: str) -> None:
    torch.manual_seed(0)
    plain = BACKBONES[name](None).eval()
    torch.manual_seed(0)
    model = BACKBONES[name](dict(LORA, dropout=0.1)).eval()
    x = torch.rand(2, 3, 96, 128)
    # Zero-initialised lora_B: same function as the frozen ViT.
    with torch.no_grad():
        for a, b in zip(plain(x), model(x), strict=True):
            torch.testing.assert_close(a, b)

    model.train()
    lora = {n: p for n, p in model.vit.named_parameters() if '.lora_' in n}
    loras = [m for m in model.vit.modules() if isinstance(m, LoRALinear)]
    assert lora and all(p.requires_grad for p in lora.values())
    assert not model.vit.training and all(m.training for m in loras)
    base = {n: p.detach().clone() for n, p in model.vit.named_parameters() if n not in lora}
    assert not any(p.requires_grad for n, p in model.vit.named_parameters() if n in base)

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    for _ in range(2):
        optim.zero_grad()
        sum(o.square().mean() for o in model(x)).backward()
        assert all(p.grad is not None for p in lora.values())
        optim.step()
    params = dict(model.vit.named_parameters())
    assert all(torch.equal(params[n], p) for n, p in base.items())
    assert any(p.abs().sum() > 0 for n, p in lora.items() if n.endswith('lora_B'))

    model.eval()
    assert not any(m.training for m in loras)


def test_load_radio_weights_with_lora(tmp_path: Path) -> None:
    src = RADIOVisionTransformer(**TINY_VIT)
    torch.save(src.state_dict(), tmp_path / 'radio.pth')
    vit = RADIOVisionTransformer(**TINY_VIT)
    inject_lora(vit, r=2)
    load_radio_weights(vit, str(tmp_path / 'radio.pth'))
    torch.testing.assert_close(vit.blocks[1].attn.qkv.weight, src.blocks[1].attn.qkv.weight)

    state = src.state_dict()
    del state['model.blocks.0.attn.qkv.weight']
    torch.save(state, tmp_path / 'broken.pth')
    with pytest.raises(RuntimeError, match='model.blocks.0.attn.qkv.weight'):
        load_radio_weights(vit, str(tmp_path / 'broken.pth'))
