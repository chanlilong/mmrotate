# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.structures import InstanceData

from mmrotate.models.task_modules.nwd_rka import RankingAssigner, RankingDynamicSoftLabelAssigner
from mmrotate.registry import TASK_UTILS
from mmrotate.structures.bbox import RotatedBoxes
from mmrotate.utils import register_all_modules

register_all_modules()


def grid_anchors(size: float, stride: float, num: int) -> torch.Tensor:
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


def gt_instances(boxes: list[list[float]], labels: list[int]) -> InstanceData:
    return InstanceData(
        bboxes=RotatedBoxes(torch.tensor(boxes, dtype=torch.float32)),
        labels=torch.tensor(labels, dtype=torch.long),
    )


def test_every_gt_gets_topk_positives() -> None:
    anchors = grid_anchors(16, 8, 10)
    gts = gt_instances([[20, 20, 4, 3, 0.2], [60, 44, 30, 12, 0.5]], [1, 4])
    assigner = RankingAssigner(topk=3)
    result = assigner.assign(InstanceData(priors=RotatedBoxes(anchors)), gts)
    for gt_ind, label in ((1, 1), (2, 4)):
        pos = result.gt_inds == gt_ind
        assert pos.sum() == 3
        assert (result.labels[pos] == label).all()
    # Priors with best NWD < neg_thr are negatives, nothing positive is negative.
    nwd = assigner.iou_calculator(gts.bboxes, anchors).max(0).values
    assert (result.gt_inds[(nwd < 0.3) & (result.gt_inds <= 0)] == 0).all()
    assert (result.gt_inds[(nwd >= 0.3) & (result.gt_inds <= 0)] == -1).all()


def test_topk_matches_reference_loop() -> None:
    """Same positives as the mmdet-aitod loop when there are no ties/conflicts."""
    anchors = grid_anchors(12, 8, 12)
    # off-grid centres so that no two anchors tie for a GT
    gts = gt_instances(
        [[15.3, 13.1, 6, 4, 0.0], [70.6, 30.2, 10, 10, 0.3], [41.1, 78.7, 20, 8, -0.4]], [0, 1, 2]
    )
    assigner = RankingAssigner(topk=2)
    result = assigner.assign(InstanceData(priors=RotatedBoxes(anchors)), gts)
    overlaps = assigner.iou_calculator(gts.bboxes, anchors)
    expected = torch.full((anchors.size(0),), -1, dtype=torch.long)
    max_overlaps = overlaps.max(0).values
    expected[(max_overlaps >= 0) & (max_overlaps < 0.3)] = 0
    top = overlaps.topk(2, dim=1).values
    for i in range(3):
        for j in range(2):
            expected[overlaps[i] == top[i, j]] = i + 1
    assert torch.equal(result.gt_inds, expected)


def test_conflict_goes_to_best_gt() -> None:
    anchors = torch.tensor(
        [[10, 10, 8, 8, 0], [30, 10, 8, 8, 0], [90, 90, 8, 8, 0]], dtype=torch.float32
    )
    # both GTs rank anchor 0 first; the second GT is much closer to it
    gts = gt_instances([[16, 10, 8, 8, 0], [10.5, 10, 8, 8, 0]], [0, 1])
    result = RankingAssigner(topk=1).assign(InstanceData(priors=RotatedBoxes(anchors)), gts)
    assert result.gt_inds[0] == 2


def test_empty_inputs() -> None:
    anchors = grid_anchors(16, 8, 4)
    assigner = RankingAssigner(topk=2)
    no_gts = InstanceData(
        bboxes=RotatedBoxes(torch.zeros(0, 5)), labels=torch.zeros(0, dtype=torch.long)
    )
    result = assigner.assign(InstanceData(priors=anchors), no_gts)
    assert (result.gt_inds == 0).all()
    result = assigner.assign(InstanceData(priors=anchors[:0]), gt_instances([[5, 5, 4, 4, 0]], [0]))
    assert result.gt_inds.numel() == 0


def test_ignore_and_cpu_path() -> None:
    anchors = grid_anchors(16, 8, 8)
    gts = gt_instances([[20, 20, 6, 6, 0], [44, 44, 6, 6, 0]], [0, 1])
    ignore = InstanceData(bboxes=RotatedBoxes(torch.tensor([[60, 60, 20, 20, 0.0]])))
    assigner = RankingAssigner(topk=2, ignore_iof_thr=0.5, gpu_assign_thr=1)
    result = assigner.assign(InstanceData(priors=RotatedBoxes(anchors)), gts, ignore)
    inside_ignore = (anchors[:, :2] - 60).abs().max(1).values < 6
    assert (result.gt_inds[inside_ignore] == -1).all()
    assert (result.gt_inds > 0).sum() == 4


def test_point_priors_and_quality() -> None:
    xs = torch.arange(8, dtype=torch.float32) * 8
    yy, xx = torch.meshgrid(xs, xs, indexing='ij')
    priors = torch.stack(
        [xx.flatten(), yy.flatten(), torch.full((64,), 8.0), torch.full((64,), 8.0)], 1
    )
    gts = gt_instances([[21, 19, 3, 3, 0.1]], [2])
    pred = torch.cat([priors[:, :2], torch.full((64, 2), 3.0), torch.zeros(64, 1)], 1)
    assigner = TASK_UTILS.build(
        dict(
            type='RankingAssigner',
            topk=4,
            point_prior_scale=2.0,
            quality_calculator=dict(type='RBboxOverlaps2D'),
        )
    )
    result = assigner.assign(InstanceData(priors=priors, bboxes=pred), gts)
    pos = result.gt_inds > 0
    assert pos.sum() == 4
    # the 4 nearest grid points to (21, 19) are (16|24, 16|24)
    assert set(map(tuple, priors[pos, :2].tolist())) == {(16, 16), (24, 16), (16, 24), (24, 24)}
    expected = TASK_UTILS.build(dict(type='RBboxOverlaps2D'))(
        pred[pos], gts.bboxes.tensor.expand(4, 5), is_aligned=True
    )
    torch.testing.assert_close(result.max_overlaps[pos], expected)


def test_ranking_dsl_adds_candidates_for_tiny_gt() -> None:
    stride = 16.0
    xs = torch.arange(8, dtype=torch.float32) * stride
    yy, xx = torch.meshgrid(xs, xs, indexing='ij')
    n = xx.numel()
    priors = torch.stack(
        [xx.flatten(), yy.flatten(), torch.full((n,), stride), torch.full((n,), stride)], 1
    )
    # tiny GT between prior centres: no prior centre lies inside it
    gts = gt_instances([[40, 40, 4, 4, 0.3]], [0])
    pred = InstanceData(
        priors=priors,
        bboxes=torch.cat([priors[:, :2], torch.full((n, 2), 6.0), torch.zeros(n, 1)], 1),
        scores=torch.zeros(n, 3),
    )
    stock = TASK_UTILS.build(
        dict(type='mmdet.DynamicSoftLabelAssigner', iou_calculator=dict(type='RBboxNWD2D'))
    ).assign(pred, gts)
    assert (stock.gt_inds > 0).sum() == 0

    hybrid = RankingDynamicSoftLabelAssigner(
        rka_topk=4, rka_prior_scale=1.0, iou_calculator=dict(type='RBboxNWD2D')
    )
    result = hybrid.assign(pred, gts)
    pos = result.gt_inds > 0
    assert pos.sum() >= 1
    assert (result.labels[pos] == 0).all()
    assert (result.max_overlaps[pos] > 0).all()
