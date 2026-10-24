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

# config -> (path of the assigner in model.train_cfg, assigner type)
CONFIGS = {
    'nwd_rka/rotated-rtmdet-l_nwd-rka-3x-dota.py': ('assigner', 'RankingAssigner'),
    'nwd_rka/rotated-rtmdet-l_nwd-dsl-rka-3x-dota.py': (
        'assigner',
        'RankingDynamicSoftLabelAssigner',
    ),
    'nwd_rka/rotated-rtmdet-tiny_nwd-rka-proxy.py': ('assigner', 'RankingAssigner'),
    'nwd_rka/rotated-retinanet-rbox-le90_r50_fpn_nwd-rka-1x_dota.py': (
        'assigner',
        'RankingAssigner',
    ),
    'nwd_rka/oriented-rcnn-le90_r50_fpn_nwd-rka-1x_dota.py': ('rpn.assigner', 'RankingAssigner'),
    'cradio/cradiov4-h-comer_rtmdet-nwd-rka-3x-dota_ms.yaml': ('assigner', 'RankingAssigner'),
    'cradio/cradiov4-h-comer_rtmdet-nwd-dsl-rka-3x-dota_ms.yaml': (
        'assigner',
        'RankingDynamicSoftLabelAssigner',
    ),
    'cradio/cradiov4-h-vitdet_rtmdet-p2-nwd-rka-3x-dota_ms.yaml': ('assigner', 'RankingAssigner'),
}
# mmcv's RotatedIoULoss (diff_iou_rotated) is CUDA-only; GWD runs on CPU.
CPU_BOX_LOSS = dict(type='GDLoss', loss_type='gwd', loss_weight=5.0)


def _message_hub() -> None:
    hub = MessageHub.get_instance(f'test_nwd_rka-{time.time()}')
    hub.update_info('iter', 0)
    hub.update_info('epoch', 0)


@pytest.mark.parametrize('cfg_file', list(CONFIGS))
def test_config_builds_assigner(cfg_file: str) -> None:
    cfg = Config.fromfile(f'configs/{cfg_file}')
    key, assigner_type = CONFIGS[cfg_file]
    assigner_cfg = cfg.model.train_cfg
    for part in key.split('.'):
        assigner_cfg = assigner_cfg[part]
    assert assigner_cfg.type == assigner_type
    assert '_delete_' not in assigner_cfg
    assert type(TASK_UTILS.build(assigner_cfg)).__name__ == assigner_type


IMAGE_SHAPES = [[3, 128, 128], [3, 125, 130]]


def _loss(model: ConfigDict, image_shapes: list[list[int]] = IMAGE_SHAPES) -> dict:
    _message_hub()
    detector = MODELS.build(model)
    detector.init_weights()
    packed_inputs = demo_mm_inputs(2, image_shapes, use_box_type=True)
    data = detector.data_preprocessor(packed_inputs, True)
    losses = detector.forward(**data, mode='loss')
    assert isinstance(losses, dict)
    for name, value in losses.items():
        values = value if isinstance(value, list) else [value]
        assert all(torch.isfinite(v).all() for v in values), name
    return losses


@pytest.mark.parametrize(
    'assigner_cfg_file',
    [
        'nwd_rka/rotated-rtmdet-l_nwd-rka-3x-dota.py',
        'nwd_rka/rotated-rtmdet-l_nwd-dsl-rka-3x-dota.py',
    ],
)
def test_rtmdet_loss_cpu(assigner_cfg_file: str) -> None:
    model = get_detector_cfg('nwd_rka/rotated-rtmdet-tiny_nwd-rka-proxy.py')
    model.train_cfg.assigner = get_detector_cfg(assigner_cfg_file).train_cfg.assigner
    model.bbox_head.loss_bbox = CPU_BOX_LOSS
    # RTMDet's data preprocessor does not pad to a multiple of the max stride
    losses = _loss(model, [[3, 128, 128], [3, 128, 128]])
    assert sum(losses['loss_bbox']) > 0


def test_retinanet_loss_cpu() -> None:
    model = get_detector_cfg('nwd_rka/rotated-retinanet-rbox-le90_r50_fpn_nwd-rka-1x_dota.py')
    model.backbone.init_cfg = None
    _loss(model)


def test_oriented_rcnn_loss_and_predict_cpu() -> None:
    model = get_detector_cfg('nwd_rka/oriented-rcnn-le90_r50_fpn_nwd-rka-1x_dota.py')
    model.backbone.init_cfg = None
    assert model.train_cfg.rpn_proposal.nms.type == 'nwd_nms'
    assert model.test_cfg.rpn.nms.type == 'nwd_nms'
    _loss(model)

    detector = MODELS.build(model).eval()
    packed_inputs = demo_mm_inputs(2, IMAGE_SHAPES, use_box_type=True)
    data = detector.data_preprocessor(packed_inputs, False)
    with torch.no_grad():
        results = detector.forward(**data, mode='predict')
    assert len(results) == 2
