# Copyright (c) OpenMMLab. All rights reserved.
"""The low-memory C-RADIOv4 CoMer P2 config trains the same model as the plain one."""

import time

import pytest
import torch
from mmengine.config import ConfigDict
from mmengine.logging import MessageHub
from torch import Tensor

from mmrotate.registry import MODELS
from mmrotate.testing import demo_mm_inputs
from mmrotate.testing.frozen_configs import frozen_config, frozen_detector_cfg, real_config
from mmrotate.utils import register_all_modules

PLAIN = 'cradio/cradiov4-h-comer_rtmdet-p2-nwd-dsl-rka-3x-dota_ms.yaml'
LOWMEM = 'cradio/cradiov4-h-comer_rtmdet-p2-nwd-dsl-rka-lowmem-3x-dota_ms.yaml'
PROXY = 'cradio/cradiov4-h-comer_rtmdet-p2-nwd-dsl-rka-lowmem-proxy.yaml'
TINY_DIM = 32


def _tiny(cfg_file: str) -> ConfigDict:
    """Shrink the ViT-H config so the full detector runs quickly."""
    model = frozen_detector_cfg(cfg_file)
    model.backbone.update(
        checkpoint=None,
        vit_cfg=dict(embed_dim=TINY_DIM, depth=4, num_heads=2, max_img_size=256),
        interaction_indexes=[[0, 0], [1, 1], [2, 2], [3, 3]],
        deform_num_heads=2,
        conv_inplane=8,
        norm_cfg=dict(type='BN'),
        msda_backend='pytorch',
        init_values=0.5,
    )
    model.neck.in_channels = [TINY_DIM] * 4
    model.neck.norm_cfg = dict(type='BN')
    model.bbox_head.norm_cfg = dict(type='BN')
    return model


def _build(cfg_file: str, device: str) -> torch.nn.Module:
    register_all_modules()
    hub = MessageHub.get_instance(f'test_cradio_lowmem-{time.time()}')
    hub.update_info('iter', 0)
    hub.update_info('epoch', 0)
    detector = MODELS.build(_tiny(cfg_file))
    detector.init_weights()
    return detector.to(device)


@pytest.mark.parametrize('cfg_file', [LOWMEM, PROXY])
def test_config_contents(cfg_file: str) -> None:
    cfg, plain = real_config(cfg_file), real_config(PLAIN)
    assert cfg.model.neck.pop('type') == 'CheckpointedCSPNeXtPAFPN'
    assert cfg.model.neck.pop('cp_levels') == [0, 1]
    assert cfg.model.bbox_head.pop('type') == 'FastTargetsRotatedRTMDetSepBNHead'
    plain.model.neck.pop('type')
    plain.model.bbox_head.pop('type')
    # Everything else in the model is the plain config's: same architecture.
    assert cfg.model.to_dict() == plain.model.to_dict()
    assert cfg.model.train_cfg.assigner.type == 'RankingDynamicSoftLabelAssigner'
    assert cfg.train_dataloader.batch_size == 4
    ema = [h for h in cfg.custom_hooks if h.type == 'EMAHook']
    assert len(ema) == 1 and ema[0].ema_type == 'FrozenSharedExpMomentumEMA'
    assert cfg.optim_wrapper == plain.optim_wrapper


def test_frozen_snapshot_hooks() -> None:
    cfg = frozen_config(PROXY)
    assert [h.type for h in cfg.custom_hooks] == ['mmdet.NumClassCheckHook', 'EMAHook']


def _loss_and_grads(detector: torch.nn.Module) -> tuple[Tensor, list[Tensor | None]]:
    packed = demo_mm_inputs(2, [[3, 128, 128], [3, 125, 130]], num_items=[6, 9], use_box_type=True)
    detector.train()
    torch.manual_seed(0)  # same DropPath draws in every run
    data = detector.data_preprocessor(packed, True)
    loss, _ = detector.parse_losses(detector.forward(**data, mode='loss'))
    params = [p for p in detector.parameters() if p.requires_grad]
    return loss, list(torch.autograd.grad(loss, params, allow_unused=True))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='RotatedIoULoss needs CUDA')
def test_same_loss_and_gradients_as_plain_config() -> None:
    plain, lowmem = _build(PLAIN, 'cuda'), _build(LOWMEM, 'cuda')
    lowmem.load_state_dict(plain.state_dict(), strict=True)
    assert list(lowmem.state_dict()) == list(plain.state_dict())
    start = {k: v.clone() for k, v in plain.state_dict().items()}

    # The CUDA backward is not bit-reproducible, so the plain detector is run
    # twice: its own run-to-run difference is the yardstick for the gradients.
    loss_a, grads_a = _loss_and_grads(plain)
    after_step = {k: v.clone() for k, v in plain.state_dict().items()}
    plain.load_state_dict(start)
    _, grads_a2 = _loss_and_grads(plain)
    loss_b, grads_b = _loss_and_grads(lowmem)

    assert torch.isfinite(loss_a)
    torch.testing.assert_close(loss_b, loss_a, rtol=1e-6, atol=0)
    scale = max(g.abs().max() for g in grads_a if g is not None)
    for a, a2, b in zip(grads_a, grads_a2, grads_b, strict=True):
        assert (a is None) == (b is None)
        if a is not None:
            noise = (a - a2).abs().max()
            assert (b - a).abs().max() <= 10 * noise + 1e-5 * scale
    # BatchNorm statistics moved the same way in both (checkpointing counted once).
    for key, value in lowmem.state_dict().items():
        torch.testing.assert_close(
            value.float(), after_step[key].float(), rtol=1e-4, atol=1e-5, msg=key
        )


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_forward_predict(device: str) -> None:
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('requires CUDA')
    detector = _build(LOWMEM, device).eval()
    data = detector.data_preprocessor(demo_mm_inputs(1, [[3, 128, 128]], use_box_type=True), False)
    with torch.no_grad():
        results = detector.forward(**data, mode='predict')
    assert results[0].pred_instances.bboxes.shape[-1] == 5
