# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path

import pytest
import torch
from mmengine.config import Config
from torch import nn

from mmrotate.models.backbones import TimmViTCoMer, TimmViTCoMerLite
from mmrotate.registry import MODELS
from mmrotate.utils import register_all_modules

VITS = 'vit_small_patch16_dinov3'
CFG_DIR = Path(__file__).parents[3] / 'configs/comer_lite'
BASELINE = (
    Path(__file__).parents[3] / 'configs/soda_dinov3/vits-dinov3-comer_rtmdet-p2-6e-soda.yaml'
)
CONFIGS = sorted(p.name for p in CFG_DIR.glob('*.yaml'))
SMALL = dict(
    conv_inplane=16,
    norm_cfg=dict(type='BN'),
    out_indices=(0, 1, 2, 3),
    msda_backend='pytorch',
)


def _layer_norms(model: nn.Module) -> int:
    return sum(isinstance(m, nn.LayerNorm) for m in model.interactions.modules())


def test_default_args_match_comer() -> None:
    torch.manual_seed(0)
    ref = TimmViTCoMer(VITS, **SMALL).eval()
    lite = TimmViTCoMerLite(model_name=VITS, **SMALL).eval()
    # same parameters and keys: the reference weights load strictly
    lite.load_state_dict(ref.state_dict())
    x = torch.randn(1, 3, 96, 128)
    with torch.no_grad():
        for a, b in zip(ref(x), lite(x), strict=True):
            torch.testing.assert_close(a, b)


def test_slim_outputs_and_grads() -> None:
    model = TimmViTCoMerLite(
        model_name=VITS, out_channels=[96, 192, 384, 384], dedup_norm=True, extra_num=1, **SMALL
    ).train()
    # non-square, not a multiple of 32: exercises padding and cropping
    outs = model(torch.randn(2, 3, 112, 144))
    assert [o.shape[1] for o in outs] == [96, 192, 384, 384]
    assert [tuple(o.shape[-2:]) for o in outs] == [(28, 36), (14, 18), (7, 9), (4, 5)]
    assert model.up.out_channels == 96 and model.spm.fc1.out_channels == 96
    sum(o.sum() for o in outs).backward()
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert [n for n, p in trainable if p.grad is None] == []
    assert not any(p.grad is not None for p in model.vit.parameters() if not p.requires_grad)


def test_slim_vit_projection_commutes_with_upsampling() -> None:
    model = TimmViTCoMerLite(model_name=VITS, out_channels=[32, 384, 384, 384], **SMALL)
    x3 = torch.randn(1, 384, 5, 7)
    up = nn.functional.interpolate
    a = model.vit_proj0(up(x3, scale_factor=4, mode='bilinear', align_corners=False))
    b = up(model.vit_proj0(x3), scale_factor=4, mode='bilinear', align_corners=False)
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)


def test_stride4_off_ignores_level0_width() -> None:
    model = TimmViTCoMerLite(
        model_name=VITS, out_channels=[96, 192, 384, 384], **dict(SMALL, out_indices=(1, 2, 3))
    )
    assert model.up is None and model.vit_proj0 is None
    outs = model(torch.randn(1, 3, 64, 64))
    assert [o.shape[1] for o in outs] == [192, 384, 384]


def test_dedup_norm_drops_two_norms_per_cnn_interaction() -> None:
    base = TimmViTCoMerLite(model_name=VITS, extra_num=2, **SMALL)
    dedup = TimmViTCoMerLite(model_name=VITS, extra_num=2, dedup_norm=True, **SMALL)
    # 4 CTI_toC + 2 extra CTIs, each loses its outer value norm and inner query norm
    assert _layer_norms(base) - _layer_norms(dedup) == 2 * 6
    keys = dedup.state_dict().keys()
    assert not any(
        k.endswith(('cti_toc.feat_norm.weight', 'cfinter.query_norm.weight')) for k in keys
    )
    assert any(k.endswith('cti_toc.query_norm.weight') for k in keys)


def test_configs_found() -> None:
    assert CONFIGS == [
        'vits-dinov3-comer-cfinter-late_rtmdet-p2-6e-soda.yaml',
        'vits-dinov3-comer-extra0_rtmdet-p2-6e-soda.yaml',
        'vits-dinov3-comer-lite_rtmdet-p2-6e-soda.yaml',
        'vits-dinov3-comer-ratio05_rtmdet-p2-6e-soda.yaml',
        'vits-dinov3-comer-slim_rtmdet-p2-6e-soda.yaml',
    ]


@pytest.mark.parametrize('name', CONFIGS)
def test_config_differs_from_baseline_only_in_adapter(name: str) -> None:
    cfg, base = Config.fromfile(CFG_DIR / name), Config.fromfile(BASELINE)
    for key in ('bbox_head', 'train_cfg', 'test_cfg', 'data_preprocessor'):
        assert cfg.model[key] == base.model[key], key
    for key in ('optim_wrapper', 'param_scheduler', 'train_cfg', 'train_dataloader'):
        assert cfg[key] == base[key], key
    neck, base_neck = dict(cfg.model.neck), dict(base.model.neck)
    assert neck.pop('in_channels') == cfg.model.backbone.get('out_channels', [384] * 4)
    base_neck.pop('in_channels')
    assert neck == base_neck
    assert cfg.model.backbone.freeze_vit is True


@pytest.mark.parametrize('name', CONFIGS)
def test_config_backbone_neck_forward(name: str) -> None:
    register_all_modules()
    cfg = Config.fromfile(CFG_DIR / name)
    cfg.model.backbone.update(checkpoint=None, conv_inplane=16, msda_backend='pytorch')
    backbone = MODELS.build(cfg.model.backbone)
    neck = MODELS.build(cfg.model.neck)
    outs = neck(backbone(torch.randn(1, 3, 96, 128)))
    assert [tuple(o.shape) for o in outs] == [
        (1, 128, 24, 32),
        (1, 128, 12, 16),
        (1, 128, 6, 8),
        (1, 128, 3, 4),
    ]
