# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path

import pytest
import torch
from mmdet.structures import DetDataSample
from mmengine.config import Config
from mmengine.structures import InstanceData

from mmrotate.registry import MODELS
from mmrotate.structures.bbox import RotatedBoxes
from mmrotate.utils import register_all_modules

CFG_DIR = Path(__file__).parents[3] / 'configs/soda_dinov3'
CONFIGS = sorted(p.name for p in CFG_DIR.glob('*.yaml'))
SHARED = ('neck', 'bbox_head', 'test_cfg', 'data_preprocessor')


def _load(name: str) -> Config:
    cfg = Config.fromfile(CFG_DIR / name)
    cfg.model.backbone.checkpoint = None  # random init, no weights needed
    if cfg.model.backbone.type == 'TimmViTCoMer':
        cfg.model.backbone.conv_inplane = 16
    return cfg


def _sample(h: int, w: int) -> DetDataSample:
    sample = DetDataSample(metainfo=dict(img_shape=(h, w), pad_shape=(h, w), scale_factor=(1, 1)))
    boxes = torch.tensor([[40.0, 40, 10, 6, 0.3], [100, 80, 24, 12, -0.5], [60, 120, 5, 5, 0]])
    sample.gt_instances = InstanceData(bboxes=RotatedBoxes(boxes), labels=torch.tensor([2, 0, 5]))
    sample.ignored_instances = InstanceData(
        bboxes=RotatedBoxes(torch.zeros((0, 5))), labels=torch.zeros(0).long()
    )
    return sample


def test_configs_found() -> None:
    assert 'vits-dinov3-comer_rtmdet-p2-6e-soda.yaml' in CONFIGS
    assert 'vits-dinov3-comer-noinj_rtmdet-p2-6e-soda.yaml' in CONFIGS
    assert 'vits-dinov3-comer-lora_rtmdet-p2-6e-soda.yaml' in CONFIGS
    assert 'vits-dinov3-vitdet-lora_rtmdet-p2-6e-soda.yaml' in CONFIGS
    assert len(CONFIGS) == 8


@pytest.mark.parametrize('name', CONFIGS)
def test_config_settings(name: str) -> None:
    cfg = Config.fromfile(CFG_DIR / name)
    base = Config.fromfile(CFG_DIR / 'vits-dinov3-vitdet_rtmdet-p2-6e-soda.yaml')
    # everything but the backbone, neck inputs and assigner is shared
    for key in SHARED:
        a, b = dict(cfg.model[key]), dict(base.model[key])
        a.pop('in_channels', None), b.pop('in_channels', None)
        assert a == b, key
    for key in ('optim_wrapper', 'param_scheduler', 'train_cfg', 'train_dataloader'):
        assert cfg[key] == base[key], key
    assert cfg.optim_wrapper.dtype == 'bfloat16'
    assert cfg.model.bbox_head.anchor_generator.strides == [4, 8, 16, 32]
    backbone = cfg.model.backbone
    assert backbone.get('frozen', backbone.get('freeze_vit')) is True
    if 'nwd' in name:
        assert cfg.model.train_cfg.assigner.type != 'mmdet.DynamicSoftLabelAssigner'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='RotatedIoULoss needs CUDA')
@pytest.mark.parametrize('name', CONFIGS)
def test_config_forward_loss(name: str) -> None:
    register_all_modules()
    cfg = _load(name)
    model = MODELS.build(cfg.model)
    model.init_weights()
    model.cuda().train()
    h, w = 160, 192
    inputs = torch.randint(0, 255, (2, 3, h, w), dtype=torch.uint8)
    data = model.data_preprocessor(
        dict(inputs=list(inputs), data_samples=[_sample(h, w), _sample(h, w)]), True
    )
    losses = model(**data, mode='loss')
    total, _ = model.parse_losses(losses)
    assert torch.isfinite(total)
    total.backward()
    frozen = [p for p in model.backbone.parameters() if not p.requires_grad]
    assert frozen and all(p.grad is None for p in frozen)
    lora = [p for n, p in model.backbone.named_parameters() if '.lora_' in n]
    assert ('lora' in name) == bool(lora) and all(p.grad is not None for p in lora)

    model.eval()
    with torch.no_grad():
        data = model.data_preprocessor(
            dict(inputs=list(inputs), data_samples=[_sample(h, w), _sample(h, w)]), False
        )
        results = model(**data, mode='predict')
    assert len(results) == 2
