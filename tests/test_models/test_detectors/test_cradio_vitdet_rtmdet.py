# Copyright (c) OpenMMLab. All rights reserved.
import time

import pytest
import torch
from mmengine.config import Config, ConfigDict
from mmengine.logging import MessageHub
from mmengine.optim import build_optim_wrapper

from mmrotate.registry import MODELS
from mmrotate.testing import demo_mm_inputs, get_detector_cfg
from mmrotate.utils import register_all_modules

CONFIGS = {
    'cradio/cradiov4-h-vitdet_rtmdet-3x-dota_ms.yaml': [8, 16, 32],
    'cradio/cradiov4-h-vitdet_rtmdet-p2-3x-dota_ms.yaml': [4, 8, 16, 32],
    'cradio/cradiov4-h-vitdet_rtmdet-proxy.yaml': [8, 16, 32],
}
TINY_DIM = 32
TINY_VIT = dict(embed_dim=TINY_DIM, depth=4, num_heads=2, max_img_size=256, with_cp=True)


def _tiny_detector_cfg(cfg_file: str) -> ConfigDict:
    """Shrink the ViT-H config so the full detector runs quickly."""
    model = get_detector_cfg(cfg_file)
    model.backbone.checkpoint = None
    model.backbone.vit_cfg = TINY_VIT
    model.backbone.out_indices = [-1]
    model.neck.in_channels = TINY_DIM
    model.bbox_head.norm_cfg = dict(type='BN')
    return model


@pytest.mark.parametrize('cfg_file', list(CONFIGS))
def test_config_contents(cfg_file: str) -> None:
    cfg = Config.fromfile(f'configs/{cfg_file}')
    model = cfg.model
    assert cfg.default_scope == 'mmrotate'
    assert model.type == 'mmdet.RTMDet'
    assert model.backbone.type == 'CRADIOv4'
    assert model.backbone.checkpoint == '/mnt/zfs/models/nvidia/C-RADIOv4-H'
    assert model.backbone.frozen is False
    assert model.backbone.out_indices == [31]
    assert model.backbone.out_strides == [16]
    assert 'interaction_indexes' not in model.backbone
    assert model.neck.type == 'ViTDetSimpleFPN'
    assert model.neck.in_channels == 1280
    assert model.neck.out_strides == CONFIGS[cfg_file]
    assert model.neck.norm_cfg.type == 'LN2d'
    assert 'num_csp_blocks' not in model.neck
    assert model.bbox_head.type == 'RotatedRTMDetSepBNHead'
    assert model.bbox_head.anchor_generator.strides == model.neck.out_strides
    assert model.bbox_head.in_channels == model.neck.out_channels
    assert '_delete_' not in model
    optim = cfg.optim_wrapper
    assert optim.type == 'AmpOptimWrapper'
    assert optim.dtype == 'bfloat16'
    assert optim.constructor == 'CRADIOv4LayerDecayOptimWrapperConstructor'
    assert optim.optimizer.type == 'AdamW'
    assert dict(optim.paramwise_cfg) == dict(num_layers=32, layer_decay_rate=0.9)


def _build(cfg_file: str, device: str) -> torch.nn.Module:
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('requires CUDA')
    register_all_modules()
    message_hub = MessageHub.get_instance(f'test_cradio_vitdet_rtmdet-{time.time()}')
    message_hub.update_info('iter', 0)
    message_hub.update_info('epoch', 0)
    detector = MODELS.build(_tiny_detector_cfg(cfg_file))
    detector.init_weights()
    detector = detector.to(device)
    assert all(p.requires_grad for p in detector.backbone.vit.parameters())
    return detector


@pytest.mark.parametrize('cfg_file', list(CONFIGS))
def test_forward_loss(cfg_file: str) -> None:
    # mmcv's diff_iou_rotated (RotatedIoULoss) only has a CUDA kernel.
    detector = _build(cfg_file, 'cuda').train()
    packed_inputs = demo_mm_inputs(2, [[3, 128, 128], [3, 125, 130]], use_box_type=True)
    data = detector.data_preprocessor(packed_inputs, True)
    losses = detector.forward(**data, mode='loss')
    assert set(losses) >= {'loss_cls', 'loss_bbox'}
    loss, _ = detector.parse_losses(losses)
    assert torch.isfinite(loss)
    loss.backward()
    # The ViT is fine-tuned end to end.
    assert detector.backbone.vit.blocks[0].attn.qkv.weight.grad is not None
    assert detector.neck.stages[0][1].weight.grad is not None


@pytest.mark.parametrize('cfg_file', list(CONFIGS))
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_forward_predict(cfg_file: str, device: str) -> None:
    detector = _build(cfg_file, device).eval()
    packed_inputs = demo_mm_inputs(2, [[3, 128, 128], [3, 125, 130]], use_box_type=True)
    data = detector.data_preprocessor(packed_inputs, False)
    with torch.no_grad():
        results = detector.forward(**data, mode='predict')
    assert len(results) == 2
    assert results[0].pred_instances.bboxes.shape[-1] == 5


def test_build_optim_wrapper() -> None:
    cfg_file = next(iter(CONFIGS))
    cfg = Config.fromfile(f'configs/{cfg_file}')
    detector = _build(cfg_file, 'cpu')
    optim_cfg = cfg.optim_wrapper
    optim_cfg.paramwise_cfg.num_layers = TINY_VIT['depth']
    # AmpOptimWrapper needs CUDA, the param groups do not.
    optim_cfg.type = 'OptimWrapper'
    optim_cfg.pop('dtype')
    optim_wrapper = build_optim_wrapper(detector, optim_cfg)
    lrs = {g['group_name']: g['lr'] for g in optim_wrapper.optimizer.param_groups}
    base_lr = cfg.optim_wrapper.optimizer.lr
    assert lrs['layer_5_decay'] == pytest.approx(base_lr)
    assert lrs['layer_0_decay'] == pytest.approx(base_lr * 0.9**5)
    num = sum(len(g['params']) for g in optim_wrapper.optimizer.param_groups)
    assert num == sum(1 for p in detector.parameters() if p.requires_grad)
