# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from mmengine.optim import OptimWrapper
from torch import nn

from mmrotate.models.backbones import CRADIOv4
from mmrotate.models.backbones.cradiov4.layer_decay import get_layer_id
from mmrotate.models.necks import ViTDetSimpleFPN
from mmrotate.registry import OPTIM_WRAPPER_CONSTRUCTORS
from mmrotate.utils import register_all_modules

DEPTH = 4
RATE = 0.5
BASE_LR = 1e-3
BASE_WD = 0.1
TINY_VIT = dict(embed_dim=32, depth=DEPTH, num_heads=2, max_img_size=128)


class _Detector(nn.Module):
    def __init__(self, frozen: bool = False) -> None:
        super().__init__()
        self.backbone = CRADIOv4(
            out_indices=[-1], out_strides=[16], frozen=frozen, vit_cfg=TINY_VIT
        )
        self.neck = ViTDetSimpleFPN(in_channels=32, out_channels=16)


def _build(model: nn.Module) -> OptimWrapper:
    register_all_modules()
    constructor = OPTIM_WRAPPER_CONSTRUCTORS.build(
        dict(
            type='CRADIOv4LayerDecayOptimWrapperConstructor',
            optim_wrapper_cfg=dict(
                type='OptimWrapper',
                optimizer=dict(type='AdamW', lr=BASE_LR, weight_decay=BASE_WD),
            ),
            paramwise_cfg=dict(num_layers=DEPTH, layer_decay_rate=RATE),
        )
    )
    return constructor(model)


def _group_of(optim_wrapper: OptimWrapper) -> dict[int, dict]:
    return {id(p): group for group in optim_wrapper.optimizer.param_groups for p in group['params']}


@pytest.mark.parametrize(
    ('name', 'layer_id'),
    [
        ('backbone.vit.model.patch_generator.pos_embed', 0),
        ('backbone.vit.model.patch_generator.embedder.weight', 0),
        ('backbone.vit.model.blocks.0.attn.qkv.weight', 1),
        ('backbone.vit.model.blocks.3.mlp.fc2.bias', 4),
        ('backbone.spm.stem.0.weight', 5),
        ('neck.stages.0.1.weight', 5),
        ('bbox_head.rtm_cls.0.weight', 5),
    ],
)
def test_get_layer_id(name: str, layer_id: int) -> None:
    assert get_layer_id(name, DEPTH, 'backbone.vit.') == layer_id


def test_layer_decay_groups() -> None:
    model = _Detector()
    optim_wrapper = _build(model)
    groups = _group_of(optim_wrapper)
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    # Every trainable parameter is in exactly one group.
    assert sum(len(g['params']) for g in optim_wrapper.optimizer.param_groups) == len(trainable)
    assert set(groups) == {id(p) for _, p in trainable}

    named = dict(model.named_parameters())
    lr = {n: groups[id(p)]['lr'] for n, p in named.items()}
    wd = {n: groups[id(p)]['weight_decay'] for n, p in named.items()}
    embed = 'backbone.vit.model.patch_generator.embedder.weight'
    assert lr[embed] == pytest.approx(BASE_LR * RATE ** (DEPTH + 1))
    for i in range(DEPTH):
        name = f'backbone.vit.model.blocks.{i}.attn.qkv.weight'
        assert lr[name] == pytest.approx(BASE_LR * RATE ** (DEPTH - i))
    assert lr['neck.stages.0.1.weight'] == pytest.approx(BASE_LR)

    assert wd[embed] == BASE_WD
    assert wd['neck.stages.0.1.weight'] == BASE_WD
    assert wd['backbone.vit.model.patch_generator.pos_embed'] == 0.0
    assert wd['backbone.vit.model.patch_generator.cls_token.token'] == 0.0
    assert wd['backbone.vit.model.blocks.0.attn.qkv.bias'] == 0.0
    assert wd['backbone.vit.model.blocks.0.norm1.weight'] == 0.0
    assert wd['neck.stages.0.2.weight'] == 0.0  # LayerNorm2d weight (1-D)


def test_layer_decay_skips_frozen() -> None:
    model = _Detector(frozen=True)
    optim_wrapper = _build(model)
    params = [p for g in optim_wrapper.optimizer.param_groups for p in g['params']]
    assert {id(p) for p in params} == {id(p) for p in model.neck.parameters()}
    assert all(g['lr'] == pytest.approx(BASE_LR) for g in optim_wrapper.optimizer.param_groups)


def test_layer_decay_step() -> None:
    model = _Detector()
    optim_wrapper = _build(model)
    outs = model.neck(model.backbone(torch.rand(1, 3, 64, 64)))
    optim_wrapper.update_params(sum(o.mean() for o in outs))
