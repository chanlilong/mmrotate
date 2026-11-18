# Copyright (c) OpenMMLab. All rights reserved.
"""Memory-bounded building blocks of dynamic soft-label (DSL) assignment.

mmdet's ``DynamicSoftLabelAssigner`` builds every (prior, GT) quantity at once:
``num_priors x num_gt x 4`` tensors for the inside-GT test and several
``num_valid x num_gt x num_classes`` tensors for the classification cost. With
stride-4 heads on large images and thousands of GTs that is tens of GB for one
image. Every column of these matrices depends on one GT only, so the helpers
below compute them over slices of GTs that fit a byte budget, with the same
operations per element, and only keep the 2D results (candidate mask, cost and
IoU matrices) that dynamic-k matching needs.

The cost formulas are adapted from mmdet 3.3.0
``mmdet/models/task_modules/assigners/dynamic_soft_label_assigner.py``
(Apache-2.0, OpenMMLab).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from mmdet.models.task_modules.assigners.dynamic_soft_label_assigner import EPS
from mmdet.structures.bbox import BaseBoxes
from torch import Tensor

MB = 1024**2

# Rough bytes of temporaries per (row, GT) pair, used to size the GT slices.
# Inside test for rotated boxes: offsets, the broadcast 2x2 rotation matrices,
# the matmul result and the comparisons.
INSIDE_BYTES_PER_PAIR = 64
# Ranking metric between priors and GTs (one float plus top-k scratch).
RANK_BYTES_PER_PAIR = 16


def cost_bytes_per_pair(num_classes: int) -> int:
    """Classification cost temporaries (one-hot, repeated scores, soft label,
    sigmoid, scale, BCE, ...) are ``num_classes`` wide; plus the 2D terms."""
    return 32 * num_classes + 48


def gt_slices(num_gt: int, rows: int, bytes_per_pair: int, budget_mb: float | None) -> list[slice]:
    """Split ``range(num_gt)`` so that ``rows x len(slice) x bytes_per_pair`` fits
    ``budget_mb``. ``budget_mb=None`` gives a single slice (no chunking)."""
    if budget_mb is None or num_gt == 0:
        return [slice(0, num_gt)]
    per_gt = max(1, rows * bytes_per_pair)
    size = int(max(1, min(num_gt, budget_mb * MB // per_gt)))
    return [slice(start, min(start + size, num_gt)) for start in range(0, num_gt, size)]


def _inside(prior_center: Tensor, gt_bboxes: BaseBoxes | Tensor) -> Tensor:
    """``(num_priors, num_gt)`` bool, as in mmdet's DSL."""
    if isinstance(gt_bboxes, BaseBoxes):
        return gt_bboxes.find_inside_points(prior_center)
    # Tensor boxes are treated as horizontal boxes by default
    lt_ = prior_center[:, None] - gt_bboxes[:, :2]
    rb_ = gt_bboxes[:, 2:] - prior_center[:, None]
    deltas = torch.cat([lt_, rb_], dim=-1)
    return deltas.min(dim=-1).values > 0


@torch.no_grad()
def inside_any_gt(
    prior_center: Tensor, gt_bboxes: BaseBoxes | Tensor, budget_mb: float | None
) -> Tensor:
    """``(num_priors,)`` bool: the prior centre lies inside at least one GT."""
    num_gt = gt_bboxes.size(0)
    mask = prior_center.new_zeros(prior_center.size(0), dtype=torch.bool)
    for sl in gt_slices(num_gt, prior_center.size(0), INSIDE_BYTES_PER_PAIR, budget_mb):
        mask |= _inside(prior_center, gt_bboxes[sl]).any(dim=1)
    return mask


@torch.no_grad()
def topk_per_gt_mask(
    prior_boxes: Tensor,
    gt_boxes: Tensor,
    metric: Callable[[Tensor, Tensor], Tensor],
    k: int,
    budget_mb: float | None,
) -> Tensor:
    """``(num_priors,)`` bool: the prior is among the ``k`` best of some GT by ``metric``
    (``metric(prior_boxes, gt_boxes)`` is ``(num_priors, num_gt)``, higher is better)."""
    num_priors = prior_boxes.size(0)
    mask = prior_boxes.new_zeros(num_priors, dtype=torch.bool)
    k = min(k, num_priors)
    if k <= 0:
        return mask
    for sl in gt_slices(gt_boxes.size(0), num_priors, RANK_BYTES_PER_PAIR, budget_mb):
        inds = metric(prior_boxes, gt_boxes[sl]).topk(k, dim=0).indices
        mask[inds.flatten()] = True
    return mask


def _cls_center_cost(
    valid_prior: Tensor,
    valid_pred_scores: Tensor,
    gt_labels: Tensor,
    gt_center: Tensor,
    pairwise_ious: Tensor,
    soft_center_radius: float,
    iou_weight: float,
) -> Tensor:
    """mmdet DSL cost for one GT slice (same operations, same order)."""
    num_valid, num_gt = pairwise_ious.shape
    strides = valid_prior[:, 2]
    offset = valid_prior[:, None, :2] - gt_center[None, :, :]
    distance = offset.pow(2).sum(-1).sqrt() / strides[:, None]
    soft_center_prior = torch.pow(10, distance - soft_center_radius)
    iou_cost = -torch.log(pairwise_ious + EPS) * iou_weight
    gt_onehot_label = (
        F.one_hot(gt_labels.to(torch.int64), valid_pred_scores.shape[-1])
        .float()
        .unsqueeze(0)
        .repeat(num_valid, 1, 1)
    )
    scores = valid_pred_scores.unsqueeze(1).repeat(1, num_gt, 1)
    soft_label = gt_onehot_label * pairwise_ious[..., None]
    scale_factor = soft_label - scores.sigmoid()
    soft_cls_cost = F.binary_cross_entropy_with_logits(
        scores, soft_label, reduction='none'
    ) * scale_factor.abs().pow(2.0)
    soft_cls_cost = soft_cls_cost.sum(dim=-1)
    return soft_cls_cost + iou_cost + soft_center_prior


def dsl_cost_and_ious(
    valid_prior: Tensor,
    valid_decoded_bbox: Tensor,
    valid_pred_scores: Tensor,
    gt_bboxes: BaseBoxes | Tensor,
    gt_labels: Tensor,
    gt_center: Tensor,
    iou_calculator: Callable[..., Tensor],
    soft_center_radius: float,
    iou_weight: float,
    budget_mb: float | None,
) -> tuple[Tensor, Tensor]:
    """``(cost_matrix, pairwise_ious)``, both ``(num_valid, num_gt)``.

    ``pairwise_ious`` is computed with the caller's grad mode, as in mmdet (the
    matched IoUs become the soft targets of the loss). The cost only selects
    indices, so it is built under ``no_grad``: otherwise autograd would keep
    every slice's temporaries alive until the assigner returns.
    """
    num_valid, num_gt = valid_decoded_bbox.size(0), gt_bboxes.size(0)
    slices = gt_slices(
        num_gt, num_valid, cost_bytes_per_pair(valid_pred_scores.size(-1)), budget_mb
    )
    cost: Tensor | None = None
    ious: list[Tensor] = []
    for sl in slices:
        chunk_ious = iou_calculator(valid_decoded_bbox, gt_bboxes[sl])
        with torch.no_grad():
            chunk_cost = _cls_center_cost(
                valid_prior,
                valid_pred_scores,
                gt_labels[sl],
                gt_center[sl],
                chunk_ious.detach(),
                soft_center_radius,
                iou_weight,
            )
        if len(slices) == 1:
            return chunk_cost, chunk_ious
        if cost is None:
            cost = chunk_cost.new_empty((num_valid, num_gt))
        cost[:, sl] = chunk_cost
        ious.append(chunk_ious)
    assert cost is not None
    # cat keeps the IoU graph when the calculator is differentiable (e.g. NWD)
    return cost, torch.cat(ious, dim=1)
