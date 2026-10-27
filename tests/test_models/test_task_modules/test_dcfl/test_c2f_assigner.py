# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from mmengine.structures import InstanceData
from torch import Tensor

from mmrotate.models.task_modules.dcfl import C2FAssigner, RBboxGJSD2D
from mmrotate.registry import TASK_UTILS
from mmrotate.structures.bbox import RotatedBoxes, rbbox_overlaps
from mmrotate.utils import register_all_modules

register_all_modules()


def _ref_xy_sigma(xywhr: Tensor) -> tuple[Tensor, Tensor]:
    xy = xywhr[..., :2]
    wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7)
    cos_r, sin_r = torch.cos(xywhr[..., 4]), torch.sin(xywhr[..., 4])
    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    S = 0.5 * torch.diag_embed(wh)
    return xy, R.bmm(S.square()).bmm(R.permute(0, 2, 1))


def reference_c2f(
    overlaps: Tensor,
    priors: Tensor,
    pred_bboxes: Tensor,
    scores: Tensor,
    gt_bboxes: Tensor,
    topk: int,
    topq: int,
    gauss_thr: float,
) -> Tensor:
    """``C2FAssigner.assign`` of mmrotate-dcfl (coarse2fine_assigner.py), loops kept."""
    num_gts, num_bboxes = overlaps.shape
    assigned = overlaps.new_full((num_bboxes,), -1, dtype=torch.long)
    max_overlaps, _ = overlaps.max(dim=0)
    gt_max_overlaps, _ = overlaps.topk(topk, dim=1, largest=True, sorted=True)
    assigned[(max_overlaps >= 0) & (max_overlaps < 0.8)] = 0
    for i in range(num_gts):
        for j in range(topk):
            assigned[overlaps[i, :] == gt_max_overlaps[i, j]] = i + 1

    can_positive_inds = torch.nonzero(assigned > 0)
    poscan = assigned[can_positive_inds].squeeze(-1)
    can_other_mask = assigned <= 0
    can_pos_scores = scores[can_positive_inds.squeeze(-1)]
    can_bbox_pred = pred_bboxes[can_positive_inds.squeeze(-1)]
    num_cand = poscan.size(0)
    can_pos_iou = rbbox_overlaps(gt_bboxes, can_bbox_pred)[poscan - 1, range(num_cand)]
    can_pos_cls, _ = torch.max(can_pos_scores, 1)
    quality = (can_pos_iou + can_pos_cls.sigmoid()).unsqueeze(0).repeat(num_gts, 1)
    gt_poscan = torch.zeros_like(quality) - 100
    gt_poscan[poscan - 1, range(num_cand)] = quality[poscan - 1, range(num_cand)]
    q = min(topq, num_cand)
    _, gt_argmax_quality = gt_poscan.topk(q, dim=1, largest=True, sorted=True)
    pre = assigned
    assigned = pre * can_other_mask
    assigned_pos_prior = torch.zeros((num_gts, q, 5))
    for i in range(num_gts):
        for j in range(q):
            remap = can_positive_inds[gt_argmax_quality[i, j], 0]
            assigned[remap] = pre[remap]
            assigned_pos_prior[i, j, :] = priors[remap, :]

    xy_gt, sigma_t = _ref_xy_sigma(gt_bboxes)
    xy_pt = assigned_pos_prior[..., :2].mean(dim=-2)
    xy_gt = xy_gt[None, :, :2].unsqueeze(-1)
    xy_pt = xy_pt[None, :, :2].unsqueeze(-1)
    xy_a = priors[:, None, :2].unsqueeze(-1)
    inv_sigma_t = (
        torch.stack(
            (sigma_t[..., 1, 1], -sigma_t[..., 0, 1], -sigma_t[..., 1, 0], sigma_t[..., 0, 0]),
            dim=-1,
        ).reshape(-1, 2, 2)
        / sigma_t.det()[:, None, None]
    )
    gaussian_gt = (
        torch.exp(
            -0.5 * (xy_a - xy_gt).permute(0, 1, 3, 2).matmul(inv_sigma_t).matmul(xy_a - xy_gt)
        )
        .squeeze(-1)
        .squeeze(-1)
    )
    gaussian_pt = (
        torch.exp(
            -0.5 * (xy_a - xy_pt).permute(0, 1, 3, 2).matmul(inv_sigma_t).matmul(xy_a - xy_pt)
        )
        .squeeze(-1)
        .squeeze(-1)
    )
    gaussian = 0.7 * gaussian_gt + 0.3 * gaussian_pt
    inside_flag = gaussian >= torch.exp(torch.tensor([-gauss_thr]))
    inside_mask = inside_flag[range(num_bboxes), (assigned - 1).clamp(min=0)]
    return assigned * inside_mask


def grid_anchors(size: float, stride: float, num: int) -> Tensor:
    xs = torch.arange(num, dtype=torch.float32) * stride + stride / 2
    yy, xx = torch.meshgrid(xs, xs, indexing='ij')
    n = xx.numel()
    return torch.stack(
        [
            xx.flatten(),
            yy.flatten(),
            torch.full((n,), size),
            torch.full((n,), size),
            torch.zeros(n),
        ],
        dim=1,
    )


def _inputs(seed: int = 0, num_classes: int = 3) -> tuple[Tensor, Tensor, Tensor, InstanceData]:
    g = torch.Generator().manual_seed(seed)
    anchors = grid_anchors(32, 8, 16)
    # well separated GTs with off-grid centres, so there are no candidate
    # conflicts and every GT has topk >= topq candidates
    gts = InstanceData(
        bboxes=RotatedBoxes(
            torch.tensor(
                [
                    [21.3, 18.9, 14, 6, 0.3],
                    [97.7, 30.2, 30, 22, -0.6],
                    [40.1, 101.4, 9, 9, 0.0],
                    [104.6, 99.2, 40, 12, 1.1],
                ]
            )
        ),
        labels=torch.tensor([0, 2, 1, 2]),
    )
    jitter = torch.randn(anchors.shape, generator=g) * torch.tensor([3.0, 3.0, 4.0, 4.0, 0.2])
    pred_bboxes = anchors + jitter
    scores = torch.randn(anchors.size(0), num_classes, generator=g)
    return anchors, pred_bboxes, scores, gts


@pytest.mark.parametrize('gauss_thr', [0.6, 2.0, 100.0])
def test_matches_reference(gauss_thr: float) -> None:
    anchors, pred_bboxes, scores, gts = _inputs()
    assigner = C2FAssigner(topk=9, topq=4, gauss_thr=gauss_thr)
    result = assigner.assign(InstanceData(priors=anchors, bboxes=pred_bboxes, scores=scores), gts)
    gt_bboxes = gts.bboxes.tensor
    overlaps = RBboxGJSD2D()(gt_bboxes, anchors)
    expected = reference_c2f(overlaps, anchors, pred_bboxes, scores, gt_bboxes, 9, 4, gauss_thr)
    # positives and their GTs agree; the reference also turned some ignored
    # priors into negatives (a constraint evaluated against GT 0)
    assert torch.equal(result.gt_inds.clamp(min=0), expected.clamp(min=0))
    pos = result.gt_inds > 0
    assert torch.equal(result.labels[pos], gts.labels[result.gt_inds[pos] - 1])
    assert (result.labels[~pos] == -1).all()
    num_pos = [(result.gt_inds == i + 1).sum().item() for i in range(4)]
    assert max(num_pos) <= 4
    if gauss_thr == 100.0:
        assert num_pos == [4, 4, 4, 4]


def test_candidates_not_kept_are_negative() -> None:
    anchors, pred_bboxes, scores, gts = _inputs(1)
    pred = InstanceData(priors=anchors, bboxes=pred_bboxes, scores=scores)
    coarse = C2FAssigner(topk=9, topq=9, constraint=None).assign(pred, gts)
    fine = C2FAssigner(topk=9, topq=3, constraint=None).assign(pred, gts)
    assert ((coarse.gt_inds > 0).sum(), (fine.gt_inds > 0).sum()) == (36, 12)
    dropped = (coarse.gt_inds > 0) & (fine.gt_inds <= 0)
    assert (fine.gt_inds[dropped] == 0).all()
    # fine positives keep their coarse GT and are the best by quality
    kept = fine.gt_inds > 0
    assert torch.equal(fine.gt_inds[kept], coarse.gt_inds[kept])
    # the coarse metric is returned as max_overlaps
    overlaps = RBboxGJSD2D()(gts.bboxes, anchors)
    torch.testing.assert_close(fine.max_overlaps, overlaps.max(0).values)
    neg = fine.max_overlaps < 0.8
    assert (fine.gt_inds[neg & ~kept] == 0).all()


def test_gt_with_few_candidates() -> None:
    """Fewer than ``topq`` candidates: no other GT's candidates are taken."""
    anchors, pred_bboxes, scores, gts = _inputs(2)
    result = C2FAssigner(topk=2, topq=6, constraint=None).assign(
        InstanceData(priors=anchors, bboxes=pred_bboxes, scores=scores), gts
    )
    for i in range(4):
        assert (result.gt_inds == i + 1).sum() == 2


def test_no_gt_and_ignore() -> None:
    anchors, pred_bboxes, scores, _ = _inputs()
    pred = InstanceData(priors=anchors, bboxes=pred_bboxes, scores=scores)
    empty = InstanceData(bboxes=RotatedBoxes(torch.zeros((0, 5))), labels=torch.zeros(0).long())
    result = C2FAssigner().assign(pred, empty)
    assert (result.gt_inds == 0).all() and (result.labels == -1).all()
    with pytest.raises(AssertionError, match='decoded predictions'):
        C2FAssigner().assign(InstanceData(priors=anchors), empty)


def test_point_priors_and_quality() -> None:
    anchors, pred_bboxes, scores, gts = _inputs(3)
    strides = torch.full((anchors.size(0), 2), 8.0)
    points = torch.cat([anchors[:, :2], strides], dim=1)
    assigner = C2FAssigner(
        topk=9, topq=4, point_prior_scale=4.0, quality_calculator=dict(type='RBboxOverlaps2D')
    )
    result = assigner.assign(
        InstanceData(priors=points, bboxes=RotatedBoxes(pred_bboxes), scores=scores), gts
    )
    # 4x-stride point boxes are the 32 px anchors
    box_result = C2FAssigner(topk=9, topq=4).assign(
        InstanceData(priors=anchors, bboxes=pred_bboxes, scores=scores), gts
    )
    assert torch.equal(result.gt_inds, box_result.gt_inds)
    pos = result.gt_inds > 0
    iou = rbbox_overlaps(
        pred_bboxes[pos], gts.bboxes.tensor[result.gt_inds[pos] - 1], is_aligned=True
    )
    torch.testing.assert_close(result.max_overlaps[pos], iou)


def test_build_and_cpu_assign_matches_gpu() -> None:
    assigner = TASK_UTILS.build(dict(type='C2FAssigner', topk=9, topq=4))
    assert 'C2FAssigner' in repr(assigner)
    if not torch.cuda.is_available():
        return
    anchors, pred_bboxes, scores, gts = _inputs(4)
    pred = InstanceData(priors=anchors, bboxes=pred_bboxes, scores=scores)
    cpu = assigner.assign(pred, gts)
    gpu = assigner.assign(pred.to('cuda'), gts.to('cuda'))
    on_cpu = C2FAssigner(topk=9, topq=4, gpu_assign_thr=2).assign(pred.to('cuda'), gts.to('cuda'))
    assert torch.equal(gpu.gt_inds.cpu(), cpu.gt_inds)
    assert torch.equal(on_cpu.gt_inds.cpu(), cpu.gt_inds)
    assert on_cpu.gt_inds.is_cuda
