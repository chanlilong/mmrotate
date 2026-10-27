# Copyright (c) OpenMMLab. All rights reserved.
import time

import pytest
import torch
from mmengine.config import Config, ConfigDict
from mmengine.logging import MessageHub

from mmrotate.registry import MODELS, TASK_UTILS
from mmrotate.testing import demo_mm_inputs, get_detector_cfg
from mmrotate.utils import register_all_modules

register_all_modules()

# config -> (base config it only changes the head / assigner of, head type)
CONFIGS = {
    'dcfl/rotated-retinanet-dcfl-le135_r50_fpn_3x_dota.yaml': (
        'rotated_retinanet/rotated-retinanet-rbox-le135_r50_fpn_1x_dota.py',
        'RotatedDCFLHead',
    ),
    'dcfl/rotated_rtmdet_l-dcfl-3x-dota_ms.yaml': (
        'rotated_rtmdet/rotated_rtmdet_l-3x-dota_ms.py',
        'RotatedRTMDetSepBNDCFLHead',
    ),
    'dcfl/rotated_rtmdet_tiny-dcfl-proxy.yaml': (
        'rotated_rtmdet/rotated_rtmdet_tiny-3x-dota.py',
        'RotatedRTMDetSepBNDCFLHead',
    ),
    'soda_dinov3/vits-dinov3-comer_rtmdet-p2-dcfl-6e-soda.yaml': (
        'soda_dinov3/vits-dinov3-comer_rtmdet-p2-6e-soda.yaml',
        'RotatedRTMDetSepBNDCFLHead',
    ),
    'cradio/cradiov4-h-comer_rtmdet-dcfl-3x-dota_ms.yaml': (
        'cradio/cradiov4-h-comer_rtmdet-3x-dota_ms.yaml',
        'RotatedRTMDetSepBNDCFLHead',
    ),
}
DCFL_HEAD_KEYS = ('type', 'dilation_rate', 'prior_offset_scale')
# mmcv's RotatedIoULoss (diff_iou_rotated) is CUDA-only; GWD runs on CPU.
CPU_BOX_LOSS = dict(type='GDLoss', loss_type='gwd', loss_weight=5.0)
IMAGE_SHAPES = [[3, 128, 128], [3, 128, 128]]


@pytest.mark.parametrize('cfg_file', list(CONFIGS))
def test_config_contents(cfg_file: str) -> None:
    base_file, head_type = CONFIGS[cfg_file]
    cfg = Config.fromfile(f'configs/{cfg_file}')
    base = Config.fromfile(f'configs/{base_file}')
    head = cfg.model.bbox_head
    assert head.type == head_type
    assigner = cfg.model.train_cfg.assigner
    assert assigner.type == 'C2FAssigner' and '_delete_' not in assigner
    assert type(TASK_UTILS.build(assigner)).__name__ == 'C2FAssigner'
    assert (assigner.get('point_prior_scale') is None) == (head_type == 'RotatedDCFLHead')
    if 'proxy' in cfg_file:  # smoke config: plain BN, no weights, proxy data
        return
    # everything else is the base config's
    for key in ('backbone', 'neck', 'data_preprocessor', 'test_cfg'):
        assert cfg.model[key] == base.model[key], key
    if head_type == 'RotatedRTMDetSepBNDCFLHead':
        a = {k: v for k, v in head.items() if k not in DCFL_HEAD_KEYS}
        b = {k: v for k, v in base.model.bbox_head.items() if k not in DCFL_HEAD_KEYS}
        assert a == b
        assert cfg.train_dataloader == base.train_dataloader
        assert cfg.optim_wrapper == base.optim_wrapper


def _tiny_cradio(model: ConfigDict) -> ConfigDict:
    """Shrink the ViT-H backbone as in test_cradio_rtmdet.py."""
    backbone = model.backbone
    backbone.checkpoint = None
    backbone.vit_cfg = dict(embed_dim=32, depth=4, num_heads=2, max_img_size=256)
    backbone.interaction_indexes = [[0, 0], [1, 1], [2, 2], [3, 3]]
    backbone.deform_num_heads = 2
    backbone.conv_inplane = 8
    backbone.norm_cfg = dict(type='BN')
    backbone.msda_backend = 'pytorch'
    model.neck.in_channels = [32] * len(model.neck.in_channels)
    return model


def _model_cfg(cfg_file: str, device: str) -> ConfigDict:
    model = get_detector_cfg(cfg_file)
    if model.backbone.get('init_cfg'):
        model.backbone.init_cfg = None
    if 'checkpoint' in model.backbone:
        model.backbone.checkpoint = None
    if model.backbone.type == 'CRADIOv4CoMer':
        model = _tiny_cradio(model)
    elif model.backbone.type == 'TimmViTCoMer':
        model.backbone.conv_inplane = 16
    for part in ('backbone', 'neck', 'bbox_head'):
        if model[part].get('norm_cfg', {}).get('type') == 'SyncBN':
            model[part].norm_cfg = dict(type='BN')
    if device == 'cpu':
        model.bbox_head.loss_bbox = CPU_BOX_LOSS
    return model


@pytest.mark.parametrize('cfg_file', list(CONFIGS))
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_loss_and_predict(cfg_file: str, device: str) -> None:
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('requires CUDA')
    if device == 'cpu' and ('cradio' in cfg_file or 'soda' in cfg_file):
        pytest.skip('ViT backbones are covered on CPU by their own tests')
    hub = MessageHub.get_instance(f'test_dcfl_configs-{time.time()}')
    hub.update_info('iter', 0)
    hub.update_info('epoch', 0)
    detector = MODELS.build(_model_cfg(cfg_file, device))
    detector.init_weights()
    detector = detector.to(device)

    packed_inputs = demo_mm_inputs(2, IMAGE_SHAPES, use_box_type=True)
    data = detector.data_preprocessor(packed_inputs, True)
    losses = detector.forward(**data, mode='loss')
    loss, _ = detector.parse_losses(losses)
    assert torch.isfinite(loss)
    assert sum(losses['loss_bbox']) > 0
    loss.backward()

    detector.eval()
    data = detector.data_preprocessor(packed_inputs, False)
    with torch.no_grad():
        results = detector.forward(**data, mode='predict')
    assert len(results) == 2
    assert results[0].pred_instances.bboxes.shape[-1] == 5
