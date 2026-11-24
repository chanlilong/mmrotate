# Copyright (c) OpenMMLab. All rights reserved.
"""ChunkedDynamicSoftLabelAssigner / RankingDynamicSoftLabelAssigner give the same
assignment as the unchunked references, for any GT slicing."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from mmdet.models.task_modules.assigners import AssignResult, DynamicSoftLabelAssigner
from mmdet.models.task_modules.assigners.dynamic_soft_label_assigner import (
    EPS,
    INF,
    center_of_mass,
)
from mmdet.models.task_modules.prior_generators import MlvlPointGenerator
from mmdet.structures.bbox import BaseBoxes
from mmengine.structures import InstanceData

from mmrotate.models.task_modules.chunked_dsl import (
    ChunkedDynamicSoftLabelAssigner,
    gt_slices,
)
from mmrotate.models.task_modules.nwd_rka import RankingDynamicSoftLabelAssigner
from mmrotate.models.task_modules.nwd_rka.nwd_calculator import to_box_tensor
from mmrotate.models.task_modules.nwd_rka.ranking_assigner import point_priors_to_boxes
from mmrotate.registry import TASK_UTILS
from mmrotate.structures.bbox import RotatedBoxes
from mmrotate.utils import register_all_modules

register_all_modules()

DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
# Small budgets force slices of a single GT up to a few dozen GTs.
BUDGETS = [1e-6, 0.5, 8.0]
DSL_KW = dict(topk=13, iou_calculator=dict(type='RBboxOverlaps2D'))
RKA_KW = dict(
    topk=13,
    rka_topk=3,
    rka_prior_scale=4.0,
    rka_calculator=dict(type='RBboxNWD2D', constant=16.6),
    iou_calculator=dict(type='RBboxOverlaps2D'),
)


class ReferenceRankingDSL(DynamicSoftLabelAssigner):
    """``RankingDynamicSoftLabelAssigner`` as it was before GT slicing (verbatim
    ``assign``), the reference for the equality tests."""

    def __init__(
        self,
        rka_topk: int = 3,
        rka_prior_scale: float = 4.0,
        rka_calculator: dict | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.rka_topk = rka_topk
        self.rka_prior_scale = rka_prior_scale
        self.rka_calculator = TASK_UTILS.build(rka_calculator or dict(type='RBboxNWD2D'))

    def assign(
        self,
        pred_instances: InstanceData,
        gt_instances: InstanceData,
        gt_instances_ignore: InstanceData | None = None,
        **kwargs: object,
    ) -> AssignResult:
        """Assign GTs to point priors (see ``DynamicSoftLabelAssigner``)."""
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        num_gt = gt_bboxes.size(0)

        decoded_bboxes = pred_instances.bboxes
        pred_scores = pred_instances.scores
        priors = pred_instances.priors
        num_bboxes = decoded_bboxes.size(0)

        # assign 0 by default
        assigned_gt_inds = decoded_bboxes.new_full((num_bboxes,), 0, dtype=torch.long)
        if num_gt == 0 or num_bboxes == 0:
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes,))
            assigned_labels = decoded_bboxes.new_full((num_bboxes,), -1, dtype=torch.long)
            return AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        prior_center = priors[:, :2]
        if isinstance(gt_bboxes, BaseBoxes):
            is_in_gts = gt_bboxes.find_inside_points(prior_center)
        else:
            # Tensor boxes will be treated as horizontal boxes by defaults
            lt_ = prior_center[:, None] - gt_bboxes[:, :2]
            rb_ = gt_bboxes[:, 2:] - prior_center[:, None]
            deltas = torch.cat([lt_, rb_], dim=-1)
            is_in_gts = deltas.min(dim=-1).values > 0

        # RKA: every GT also gets its top-k priors by NWD as candidates.
        rka_overlaps = self.rka_calculator(
            point_priors_to_boxes(priors, self.rka_prior_scale), to_box_tensor(gt_bboxes)
        )
        rka_inds = rka_overlaps.topk(min(self.rka_topk, num_bboxes), dim=0).indices
        is_rka = torch.zeros_like(is_in_gts, dtype=torch.bool).scatter_(0, rka_inds, True)
        valid_mask = (is_in_gts.bool() | is_rka).any(dim=1)

        valid_decoded_bbox = decoded_bboxes[valid_mask]
        valid_pred_scores = pred_scores[valid_mask]
        num_valid = valid_decoded_bbox.size(0)

        if hasattr(gt_instances, 'masks'):
            gt_center = center_of_mass(gt_instances.masks, eps=EPS)
        elif isinstance(gt_bboxes, BaseBoxes):
            gt_center = gt_bboxes.centers
        else:
            # Tensor boxes will be treated as horizontal boxes by defaults
            gt_center = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2.0
        valid_prior = priors[valid_mask]
        strides = valid_prior[:, 2]
        distance = (valid_prior[:, None, :2] - gt_center[None, :, :]).pow(2).sum(
            -1
        ).sqrt() / strides[:, None]
        soft_center_prior = torch.pow(10, distance - self.soft_center_radius)

        pairwise_ious = self.iou_calculator(valid_decoded_bbox, gt_bboxes)
        iou_cost = -torch.log(pairwise_ious + EPS) * self.iou_weight

        gt_onehot_label = (
            F.one_hot(gt_labels.to(torch.int64), pred_scores.shape[-1])
            .float()
            .unsqueeze(0)
            .repeat(num_valid, 1, 1)
        )
        valid_pred_scores = valid_pred_scores.unsqueeze(1).repeat(1, num_gt, 1)

        soft_label = gt_onehot_label * pairwise_ious[..., None]
        scale_factor = soft_label - valid_pred_scores.sigmoid()
        soft_cls_cost = F.binary_cross_entropy_with_logits(
            valid_pred_scores, soft_label, reduction='none'
        ) * scale_factor.abs().pow(2.0)
        soft_cls_cost = soft_cls_cost.sum(dim=-1)

        cost_matrix = soft_cls_cost + iou_cost + soft_center_prior

        matched_pred_ious, matched_gt_inds = self.dynamic_k_matching(
            cost_matrix, pairwise_ious, num_gt, valid_mask
        )

        # convert to AssignResult format
        assigned_gt_inds[valid_mask] = matched_gt_inds + 1
        assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
        assigned_labels[valid_mask] = gt_labels[matched_gt_inds].long()
        max_overlaps = assigned_gt_inds.new_full((num_bboxes,), -INF, dtype=torch.float32)
        max_overlaps[valid_mask] = matched_pred_ious
        return AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)


def dense_scene(
    num_gt: int,
    size: int = 256,
    box_range: tuple[float, float] = (4.0, 40.0),
    num_classes: int = 9,
    seed: int = 0,
    device: str = 'cpu',
    score_dtype: torch.dtype = torch.float32,
) -> tuple[InstanceData, InstanceData]:
    """P2-P5 point priors of a ``size`` image, predictions near them and ``num_gt``
    random rotated GTs (some sharing centres / sizes to exercise ties)."""
    g = torch.Generator().manual_seed(seed)
    gen = MlvlPointGenerator([4, 8, 16, 32], offset=0)
    feats = [(math.ceil(size / s), math.ceil(size / s)) for s in (4, 8, 16, 32)]
    priors = torch.cat(gen.grid_priors(feats, device='cpu', with_stride=True))
    n = priors.size(0)
    lo, hi = box_range
    pred_wh = lo + (hi - lo) * torch.rand(n, 2, generator=g)
    pred_xy = priors[:, :2] + (torch.rand(n, 2, generator=g) - 0.5) * priors[:, 2:4]
    pred_a = (torch.rand(n, 1, generator=g) - 0.5) * math.pi
    pred = InstanceData(
        priors=priors.to(device),
        bboxes=torch.cat([pred_xy, pred_wh, pred_a], 1).to(device),
        scores=(2 * torch.randn(n, num_classes, generator=g)).to(device, score_dtype),
    )
    wh = lo + (hi - lo) * torch.rand(num_gt, 2, generator=g)
    xy = wh.max(1, keepdim=True).values / 2 + (size - wh.max(1, keepdim=True).values) * torch.rand(
        num_gt, 2, generator=g
    )
    a = (torch.rand(num_gt, 1, generator=g) - 0.5) * math.pi
    boxes = torch.cat([xy, wh, a], 1)
    if num_gt >= 4:  # duplicates: identical GTs and GTs on a prior centre
        boxes[1] = boxes[0]
        boxes[3, :2] = priors[n // 3, :2]
    gts = InstanceData(
        bboxes=RotatedBoxes(boxes.to(device)),
        labels=torch.randint(0, num_classes, (num_gt,), generator=g).to(device),
    )
    return pred, gts


def assert_same(result: AssignResult, ref: AssignResult) -> None:
    assert result.num_gts == ref.num_gts
    assert torch.equal(result.gt_inds, ref.gt_inds)
    assert torch.equal(result.labels, ref.labels)
    assert torch.equal(result.max_overlaps, ref.max_overlaps)


def test_gt_slices() -> None:
    assert gt_slices(10, 100, 8, None) == [slice(0, 10)]
    assert gt_slices(0, 100, 8, 1.0) == [slice(0, 0)]
    # 1 MB / (1000 rows * 100 B) = 10 GTs per slice
    sl = gt_slices(25, 1000, 100, 1.0)
    assert [(s.start, s.stop) for s in sl] == [(0, 10), (10, 20), (20, 25)]
    assert gt_slices(5, 10**9, 100, 1.0) == [slice(i, i + 1) for i in range(5)]


def test_registry_and_validation() -> None:
    a = TASK_UTILS.build(dict(type='ChunkedDynamicSoftLabelAssigner', chunk_budget_mb=64, **DSL_KW))
    assert isinstance(a, ChunkedDynamicSoftLabelAssigner) and a.chunk_budget_mb == 64
    r = TASK_UTILS.build(dict(type='RankingDynamicSoftLabelAssigner', **RKA_KW))
    assert isinstance(r, ChunkedDynamicSoftLabelAssigner) and r.chunk_budget_mb == 1024.0
    with pytest.raises(ValueError):
        ChunkedDynamicSoftLabelAssigner(chunk_budget_mb=0, **DSL_KW)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('num_gt', [1, 7, 64, 300])
@pytest.mark.parametrize('budget', BUDGETS + [None])
def test_chunked_dsl_matches_mmdet(device: str, num_gt: int, budget: float | None) -> None:
    ref = DynamicSoftLabelAssigner(**DSL_KW)
    chunked = ChunkedDynamicSoftLabelAssigner(chunk_budget_mb=budget, **DSL_KW)
    for seed in range(3):
        pred, gts = dense_scene(num_gt, seed=seed, device=device)
        assert_same(chunked.assign(pred, gts), ref.assign(pred, gts))


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('num_gt', [1, 7, 64, 300])
@pytest.mark.parametrize('budget', BUDGETS + [None])
def test_ranking_dsl_matches_reference(device: str, num_gt: int, budget: float | None) -> None:
    ref = ReferenceRankingDSL(**RKA_KW)
    chunked = RankingDynamicSoftLabelAssigner(chunk_budget_mb=budget, **RKA_KW)
    for seed in range(3):
        # tiny GTs too: they only get candidates through RKA
        pred, gts = dense_scene(num_gt, box_range=(1.0, 24.0), seed=seed, device=device)
        assert_same(chunked.assign(pred, gts), ref.assign(pred, gts))


@pytest.mark.parametrize('device', DEVICES)
def test_nwd_iou_calculator_with_grad(device: str) -> None:
    """A differentiable iou_calculator (NWD) with inputs that require grad (the
    training case): same assignment, and max_overlaps tracks grad like the reference.
    (Backpropagating through it fails in mmdet's dynamic_k_matching already, which
    edits ``valid_mask`` in place, so only the autograd flag is compared.)"""
    kw = dict(RKA_KW, iou_calculator=dict(type='RBboxNWD2D', constant=16.6))
    pred, gts = dense_scene(40, seed=3, device=device)
    results = []
    for assigner in (
        ReferenceRankingDSL(**kw),
        RankingDynamicSoftLabelAssigner(chunk_budget_mb=0.01, **kw),
    ):
        p = InstanceData(
            priors=pred.priors,
            bboxes=pred.bboxes.clone().requires_grad_(True),
            scores=pred.scores.clone().requires_grad_(True),
        )
        results.append(assigner.assign(p, gts))
    assert_same(results[1], results[0])
    assert results[1].max_overlaps.requires_grad == results[0].max_overlaps.requires_grad


@pytest.mark.parametrize('device', DEVICES)
def test_edge_cases(device: str) -> None:
    ref = DynamicSoftLabelAssigner(**DSL_KW)
    chunked = ChunkedDynamicSoftLabelAssigner(chunk_budget_mb=1e-6, **DSL_KW)
    pred, _ = dense_scene(1, device=device)
    # no GT
    empty = InstanceData(
        bboxes=RotatedBoxes(torch.zeros(0, 5, device=device)),
        labels=torch.zeros(0, dtype=torch.long, device=device),
    )
    assert_same(chunked.assign(pred, empty), ref.assign(pred, empty))
    # GT outside the image: no prior inside, num_valid == 0
    far = InstanceData(
        bboxes=RotatedBoxes(torch.tensor([[5000.0, 5000.0, 10.0, 10.0, 0.0]], device=device)),
        labels=torch.zeros(1, dtype=torch.long, device=device),
    )
    assert_same(chunked.assign(pred, far), ref.assign(pred, far))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_bf16_autocast_and_memory_cuda() -> None:
    """Training conditions (bf16 scores under autocast, grad mode on, 2000 GTs on a
    1024 image): same assignment, and a much lower assigner peak."""
    pred, gts = dense_scene(
        2000, size=1024, box_range=(8.0, 32.0), seed=7, device='cuda', score_dtype=torch.bfloat16
    )
    ref = DynamicSoftLabelAssigner(**DSL_KW)
    chunked = ChunkedDynamicSoftLabelAssigner(chunk_budget_mb=256, **DSL_KW)
    peaks = []
    results = []
    for assigner in (ref, chunked):
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            results.append(assigner.assign(pred, gts))
        torch.cuda.synchronize()
        peaks.append(torch.cuda.max_memory_allocated() - base)
    assert_same(results[1], results[0])
    assert peaks[1] * 4 < peaks[0], peaks


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('shape', [(1, 1), (5, 3), (200, 30), (40, 300)])
def test_dynamic_k_matching_matches_mmdet_with_ties(device: str, shape: tuple[int, int]) -> None:
    """The single-topk dynamic-k matching against mmdet's loop over GTs, on cost
    matrices full of equal and infinite entries (ties at the cut)."""
    ref = DynamicSoftLabelAssigner(**DSL_KW)
    chunked = ChunkedDynamicSoftLabelAssigner(**DSL_KW)
    num_valid, num_gt = shape
    for seed in range(8):
        g = torch.Generator().manual_seed(seed)
        levels = (2, 5, 1000)[seed % 3]  # few distinct costs -> many ties
        cost = torch.randint(0, levels, shape, generator=g).float()
        cost[torch.rand(shape, generator=g) < 0.3] = float('inf')
        ious = torch.rand(shape, generator=g) * (seed % 4 + 1) / 4
        cost, ious = cost.to(device), ious.to(device)
        outs = []
        for assigner in (ref, chunked):
            valid = torch.ones(num_valid + 3, dtype=torch.bool, device=device)
            valid[:3] = False
            matched = assigner.dynamic_k_matching(cost.clone(), ious.clone(), num_gt, valid)
            outs.append((*matched, valid))
        for a, b in zip(outs[1], outs[0], strict=True):
            assert torch.equal(a, b)
