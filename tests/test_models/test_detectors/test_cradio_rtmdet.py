# Copyright (c) OpenMMLab. All rights reserved.
import time

import pytest
import torch
from mmengine.config import Config, ConfigDict
from mmengine.logging import MessageHub

from mmrotate.registry import MODELS
from mmrotate.testing import demo_mm_inputs, get_detector_cfg
from mmrotate.utils import register_all_modules

CONFIGS = {
    'cradio/cradiov4-h-comer_rtmdet-3x-dota_ms.yaml': ('CRADIOv4CoMer', [8, 16, 32]),
    'cradio/cradiov4-h-comer_rtmdet-p2-3x-dota_ms.yaml': ('CRADIOv4CoMer', [4, 8, 16, 32]),
    'cradio/cradiov4-h_rtmdet-3x-dota_ms.yaml': ('CRADIOv4', [8, 16, 32]),
    'cradio/cradiov4-h-comer_rtmdet-proxy.yaml': ('CRADIOv4CoMer', [8, 16, 32]),
    'cradio/cradiov4-h-comer-lora_rtmdet-p2-3x-dota_ms.yaml': ('CRADIOv4CoMer', [4, 8, 16, 32]),
    'cradio/cradiov4-h-comer-lora_rtmdet-proxy.yaml': ('CRADIOv4CoMer', [4, 8, 16, 32]),
}
TINY_DIM = 32
TINY_VIT = dict(embed_dim=TINY_DIM, depth=4, num_heads=2, max_img_size=256)


def _tiny_detector_cfg(cfg_file: str) -> ConfigDict:
    """Shrink the ViT-H config so the full detector runs quickly on CPU."""
    model = get_detector_cfg(cfg_file)
    backbone = model.backbone
    backbone.checkpoint = None
    backbone.vit_cfg = TINY_VIT
    if backbone.type == 'CRADIOv4CoMer':
        backbone.interaction_indexes = [[0, 0], [1, 1], [2, 2], [3, 3]]
        backbone.deform_num_heads = 2
        backbone.conv_inplane = 8
        backbone.norm_cfg = dict(type='BN')
        backbone.msda_backend = 'pytorch'
    else:
        backbone.out_indices = [1, 2, 3]
    model.neck.in_channels = [TINY_DIM] * len(model.neck.in_channels)
    model.neck.norm_cfg = dict(type='BN')
    model.bbox_head.norm_cfg = dict(type='BN')
    return model


@pytest.mark.parametrize('cfg_file', list(CONFIGS))
def test_config_contents(cfg_file: str) -> None:
    cfg = Config.fromfile(f'configs/{cfg_file}')
    backbone_type, strides = CONFIGS[cfg_file]
    assert cfg.default_scope == 'mmrotate'
    assert cfg.model.type == 'mmdet.RTMDet'
    assert cfg.model.backbone.type == backbone_type
    assert cfg.model.backbone.checkpoint == '/mnt/zfs/models/nvidia/C-RADIOv4-H'
    assert cfg.model.neck.type == 'mmdet.CSPNeXtPAFPN'
    assert cfg.model.bbox_head.type == 'RotatedRTMDetSepBNHead'
    assert cfg.model.bbox_head.anchor_generator.strides == strides
    assert len(cfg.model.neck.in_channels) == len(strides)
    assert '_delete_' not in cfg.model
    assert cfg.optim_wrapper.type == 'AmpOptimWrapper'
    assert cfg.optim_wrapper.dtype == 'bfloat16'
    assert cfg.optim_wrapper.optimizer.type == 'AdamW'


def _build(cfg_file: str, device: str) -> torch.nn.Module:
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('requires CUDA')
    register_all_modules()
    message_hub = MessageHub.get_instance(f'test_cradio_rtmdet-{time.time()}')
    message_hub.update_info('iter', 0)
    message_hub.update_info('epoch', 0)
    detector = MODELS.build(_tiny_detector_cfg(cfg_file))
    detector.init_weights()
    detector = detector.to(device)
    vit_params = dict(detector.backbone.vit.named_parameters())
    assert all(not p.requires_grad for n, p in vit_params.items() if '.lora_' not in n)
    lora = [p for n, p in vit_params.items() if '.lora_' in n]
    assert ('lora' in cfg_file) == bool(lora) and all(p.requires_grad for p in lora)
    return detector


@pytest.mark.parametrize('cfg_file', list(CONFIGS))
def test_forward_loss(cfg_file: str) -> None:
    # mmcv's diff_iou_rotated (RotatedIoULoss) only has a CUDA kernel.
    detector = _build(cfg_file, 'cuda')
    packed_inputs = demo_mm_inputs(2, [[3, 128, 128], [3, 125, 130]], use_box_type=True)
    data = detector.data_preprocessor(packed_inputs, True)
    losses = detector.forward(**data, mode='loss')
    assert set(losses) >= {'loss_cls', 'loss_bbox'}
    loss, _ = detector.parse_losses(losses)
    assert torch.isfinite(loss)
    loss.backward()
    trainable = [p for p in detector.parameters() if p.requires_grad]
    assert trainable and any(p.grad is not None for p in trainable)


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
