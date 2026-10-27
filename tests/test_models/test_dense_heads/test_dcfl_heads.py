# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
import torch.nn.functional as F
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData

from mmrotate.models.dense_heads import RotatedDCFLHead, RotatedRTMDetSepBNDCFLHead
from mmrotate.models.dense_heads.dcfl import (
    DCFLModulatedDeformConv,
    DCFLSharedOffsetDeformConv,
    dynamic_prior_shift,
)
from mmrotate.structures import RotatedBoxes
from mmrotate.utils import register_all_modules

register_all_modules()

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
C2F = dict(type='C2FAssigner', topk=9, topq=4)
# mmcv's RotatedIoULoss (diff_iou_rotated) is CUDA-only; GWD runs on CPU.
CPU_BOX_LOSS = dict(type='GDLoss', loss_type='gwd', loss_weight=5.0)
CUDA_BOX_LOSS = dict(type='RotatedIoULoss', mode='linear', loss_weight=2.0)
TEST_CFG = ConfigDict(
    nms_pre=100,
    min_bbox_size=0,
    score_thr=0.0,
    nms=dict(type='nms_rotated', iou_threshold=0.1),
    max_per_img=20,
)


def _gt(device: str) -> InstanceData:
    boxes = torch.tensor([[40.0, 36, 12, 6, 0.4], [90, 70, 30, 16, -0.3], [20, 100, 6, 6, 0.0]])
    return InstanceData(
        bboxes=RotatedBoxes(boxes.to(device)), labels=torch.tensor([0, 2, 1], device=device)
    )


def _empty_gt(device: str) -> InstanceData:
    return InstanceData(
        bboxes=RotatedBoxes(torch.zeros((0, 5), device=device)),
        labels=torch.zeros(0, dtype=torch.long, device=device),
    )


@requires_cuda
def test_dilated_conv_is_reference_fixed_offset_dcn() -> None:
    """The reference's fixed-offset ``DeformConv2d`` is a dilated conv."""
    from mmcv.ops import deform_conv2d

    d = 3
    x = torch.randn(2, 8, 13, 11, device='cuda')
    weight = torch.randn(8, 8, 3, 3, device='cuda')
    item = torch.full((2, 1, 13, 11), d - 1.0, device='cuda')
    zeros = torch.zeros_like(item)
    # sampling_loc of mmrotate-dcfl's RDCFLHead.forward_single
    offset = torch.cat(
        (-item, -item, -item, zeros, -item, item, zeros, -item, zeros, zeros, zeros, item)
        + (item, -item, item, zeros, item, item),
        dim=1,
    )
    ref = deform_conv2d(x, offset, weight, 1, 1)
    torch.testing.assert_close(
        F.conv2d(x, weight, padding=d, dilation=d), ref, rtol=1e-4, atol=1e-4
    )


@requires_cuda
def test_matches_mmcv_dcnv2_pack() -> None:
    """Same parameters and output as ``ModulatedDeformConv2dPack`` (i.e. the
    MMCV-DCFL ``ModulatedDeformConvG``), whose checkpoints therefore load."""
    from mmcv.ops import ModulatedDeformConv2dPack

    ref = ModulatedDeformConv2dPack(8, 6, 3, padding=1, bias=False).cuda()
    torch.nn.init.normal_(ref.conv_offset.weight, std=0.1)
    torch.nn.init.normal_(ref.conv_offset.bias, std=0.5)
    layer = DCFLModulatedDeformConv(8, 6, 3, padding=1, bias=False).cuda()
    layer.load_state_dict(ref.state_dict())
    x = torch.randn(2, 8, 9, 10, device='cuda')
    out, offset, mask = layer(x)
    torch.testing.assert_close(out, ref(x))
    assert offset.shape == (2, 18, 9, 10) and mask.shape == (2, 9, 9, 10)
    assert ((mask > 0) & (mask < 1)).all()


@requires_cuda
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_backends_match(dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    layers = {}
    for backend in ('mmcv', 'torchvision'):
        torch.manual_seed(0)
        layers[backend] = DCFLModulatedDeformConv(8, 8, 3, padding=1, backend=backend).cuda()
        torch.nn.init.normal_(layers[backend].conv_offset.weight, std=0.1)
    x = torch.randn(2, 8, 12, 9, device='cuda')
    outs, grads = {}, {}
    for backend, layer in layers.items():
        xb = x.clone().requires_grad_()
        with torch.autocast('cuda', dtype=dtype, enabled=dtype != torch.float32):
            out, offset, mask = layer(xb)
        assert out.dtype == dtype
        out.float().square().sum().backward()
        outs[backend] = out.float()
        grads[backend] = (xb.grad, layer.weight.grad, layer.conv_offset.weight.grad)
    tol = dict(rtol=1e-4, atol=1e-4) if dtype == torch.float32 else dict(rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(outs['mmcv'], outs['torchvision'], **tol)
    for a, b in zip(grads['mmcv'], grads['torchvision'], strict=True):
        torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-3 if dtype == torch.float32 else 5e-1)


def test_shared_offset_conv_cpu() -> None:
    layer = DCFLModulatedDeformConv(4, 4, 3, padding=1)
    shared = DCFLSharedOffsetDeformConv(4, 4, 3, padding=1)
    shared.load_state_dict(layer.state_dict(), strict=False)
    x = torch.randn(1, 4, 7, 7)
    out, offset, mask = layer(x)
    # zero-initialised offsets: plain conv with weights * 0.5
    torch.testing.assert_close(
        out, F.conv2d(x, layer.weight * 0.5, padding=1), rtol=1e-4, atol=1e-5
    )
    torch.testing.assert_close(shared(x, offset, mask), out)


def test_dynamic_prior_shift() -> None:
    offset = torch.zeros(2, 18, 3, 4)
    offset[0, 0::2] = 2.0  # dy of all 9 points
    offset[0, 1, 1, 2] = 9.0  # dx of point 0 at (y=1, x=2)
    shift = dynamic_prior_shift(offset, (8, 16), 0.5)
    assert shift.shape == (2, 12, 2)
    torch.testing.assert_close(shift[0, :, 1], torch.full((12,), 2.0 * 16 * 0.5))
    assert shift[0, 1 * 4 + 2, 0] == 9.0 / 9 * 8 * 0.5
    assert shift[0, 0, 0] == 0 and (shift[1] == 0).all()


def _rtmdet_head(device: str, **kwargs: object) -> RotatedRTMDetSepBNDCFLHead:
    head = RotatedRTMDetSepBNDCFLHead(
        num_classes=3,
        in_channels=8,
        feat_channels=8,
        stacked_convs=1,
        angle_version='le90',
        anchor_generator=dict(type='mmdet.MlvlPointGenerator', offset=0, strides=[8, 16, 32]),
        bbox_coder=dict(type='DistanceAnglePointCoder', angle_version='le90'),
        loss_cls=dict(type='mmdet.QualityFocalLoss', use_sigmoid=True, beta=2.0, loss_weight=1.0),
        loss_bbox=CUDA_BOX_LOSS if device == 'cuda' else CPU_BOX_LOSS,
        norm_cfg=dict(type='BN'),
        train_cfg=dict(
            assigner=dict(
                **C2F, point_prior_scale=4.0, quality_calculator=dict(type='RBboxOverlaps2D')
            ),
            allowed_border=-1,
            pos_weight=-1,
            debug=False,
        ),
        test_cfg=TEST_CFG,
        **kwargs,
    )
    head.init_weights()
    return head.to(device)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=requires_cuda)])
def test_rtmdet_dcfl_head(device: str) -> None:
    head = _rtmdet_head(device)
    assert head.reg_dcns[0] is head.reg_dcns[2]  # shared across levels
    assert head.reg_dcn_norms[0] is not head.reg_dcn_norms[2]
    assert (head.reg_dcns[0].conv_offset.weight == 0).all()
    s = 128
    metas = [dict(img_shape=(s, s), pad_shape=(s, s), scale_factor=(1.0, 1.0))]
    feats = tuple(torch.rand(1, 8, s // st, s // st, device=device) for st in (8, 16, 32))
    outs = head(feats)
    assert len(outs) == 4 and outs[3][0].shape == (1, 18, 16, 16)

    losses = head.loss_by_feat(*outs, [_gt(device)], metas)
    assert losses['loss_cls'][0] > 0 and sum(losses['loss_bbox']) > 0
    sum(losses['loss_cls']).add(sum(losses['loss_bbox'])).backward()
    # the cls DCN's offsets come from the reg DCN, so both branches train it
    assert head.reg_dcns[0].conv_offset.weight.grad.abs().sum() > 0
    assert head._dcfl_offsets is None

    empty = head.loss_by_feat(*head(feats), [_empty_gt(device)], metas)
    assert empty['loss_cls'][0] > 0 and sum(empty['loss_bbox']) == 0

    head.eval()
    with torch.no_grad():
        results = head.predict_by_feat(*head(feats), batch_img_metas=metas, rescale=False)
    assert results[0].bboxes.shape[-1] == 5 and len(results[0]) <= 20


def test_rtmdet_dynamic_priors_reach_assigner() -> None:
    head = _rtmdet_head('cpu')
    seen = {}

    def assign(pred_instances: InstanceData, *args: object) -> object:
        seen.setdefault('priors', pred_instances.priors.clone())
        return original(pred_instances, *args)

    original = head.assigner.assign
    head.assigner.assign = assign
    torch.nn.init.constant_(head.reg_dcns[0].conv_offset.bias, 0.0)
    head.reg_dcns[0].conv_offset.bias.data[:18:2] = 1.0  # dy = 1 feature pixel everywhere
    s = 64
    metas = [dict(img_shape=(s, s), pad_shape=(s, s), scale_factor=(1.0, 1.0))]
    feats = tuple(torch.rand(1, 8, s // st, s // st) for st in (8, 16, 32))
    head.loss_by_feat(*head(feats), [_gt('cpu')], metas)
    points = torch.cat(
        head.prior_generator.grid_priors([(8, 8), (4, 4), (2, 2)], device='cpu', with_stride=True)
    )
    expected = points.clone()
    expected[:, 1] += points[:, 3] * 0.5
    torch.testing.assert_close(seen['priors'], expected)


def _retina_head(device: str) -> RotatedDCFLHead:
    head = RotatedDCFLHead(
        num_classes=3,
        in_channels=8,
        feat_channels=8,
        stacked_convs=4,
        dilation_rate=3,
        anchor_generator=dict(
            type='FakeRotatedAnchorGenerator',
            angle_version='le135',
            octave_base_scale=4,
            scales_per_octave=1,
            ratios=[1.0],
            strides=[8, 16, 32],
        ),
        bbox_coder=dict(
            type='DeltaXYWHTRBBoxCoder',
            angle_version='le135',
            norm_factor=1,
            edge_swap=False,
            proj_xy=True,
            target_means=(0.0,) * 5,
            target_stds=(1.0,) * 5,
        ),
        loss_cls=dict(
            type='mmdet.FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0
        ),
        reg_decoded_bbox=True,
        loss_bbox=dict(type='RotatedIoULoss', loss_weight=1.0)
        if device == 'cuda'
        else CPU_BOX_LOSS,
        train_cfg=dict(
            assigner=C2F,
            sampler=dict(type='mmdet.PseudoSampler'),
            allowed_border=-1,
            pos_weight=-1,
            debug=False,
        ),
        test_cfg=TEST_CFG,
    )
    head.init_weights()
    return head.to(device)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=requires_cuda)])
def test_retina_dcfl_head(device: str) -> None:
    head = _retina_head(device)
    assert [type(m).__name__ for m in head.reg_convs] == [
        'ConvModule',
        'ConvModule',
        'Conv2d',
        'DCFLModulatedDeformConv',
    ]
    assert head.reg_convs[2].dilation == (3, 3) and head.reg_convs[2].bias is None
    assert (head.reg_convs[-1].conv_offset.weight == 0).all()
    s = 128
    metas = [
        dict(img_shape=(s, s), pad_shape=(s, s), scale_factor=(1.0, 1.0)),
        dict(img_shape=(s - 8, s), pad_shape=(s, s), scale_factor=(1.0, 1.0)),
    ]
    feats = [torch.rand(2, 8, s // st, s // st, device=device) for st in (8, 16, 32)]
    cls_scores, bbox_preds, offsets = head(feats)
    assert offsets[1].shape == (2, 18, 8, 8)
    losses = head.loss_by_feat(
        cls_scores, bbox_preds, offsets, [_gt(device), _empty_gt(device)], metas
    )
    assert sum(losses['loss_cls']) > 0 and sum(losses['loss_bbox']) > 0
    (sum(losses['loss_cls']) + sum(losses['loss_bbox'])).backward()
    assert head.reg_convs[-1].conv_offset.weight.grad.abs().sum() > 0
    assert head._dcfl_preds is None

    head.eval()
    with torch.no_grad():
        results = head.predict_by_feat(*head(feats), batch_img_metas=metas, rescale=False)
    assert len(results) == 2 and results[0].bboxes.shape[-1] == 5
