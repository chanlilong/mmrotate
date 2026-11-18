# Copyright (c) OpenMMLab. All rights reserved.
"""``DynamicSoftLabelAssigner`` with memory bounded by GT slicing.

``assign`` is adapted from mmdet 3.3.0
``mmdet/models/task_modules/assigners/dynamic_soft_label_assigner.py``
(Apache-2.0, OpenMMLab). The (prior, GT) matrices are built over slices of GTs
(see ``chunked_cost.py``); the candidate mask, cost, IoUs and dynamic-k matching
are the same, so the assignment is the same as mmdet's.
"""

from __future__ import annotations

import torch
from mmdet.models.task_modules.assigners import AssignResult, DynamicSoftLabelAssigner
from mmdet.models.task_modules.assigners.dynamic_soft_label_assigner import (
    EPS,
    INF,
    center_of_mass,
)
from mmdet.structures.bbox import BaseBoxes
from mmengine.structures import InstanceData
from torch import Tensor

from mmrotate.registry import TASK_UTILS

from .chunked_cost import dsl_cost_and_ious, inside_any_gt


@TASK_UTILS.register_module()
class ChunkedDynamicSoftLabelAssigner(DynamicSoftLabelAssigner):
    """RTMDet's dynamic soft-label assigner with bounded memory.

    Gives the same assignment as ``mmdet.DynamicSoftLabelAssigner``, but the
    per-(prior, GT) temporaries are computed over slices of GTs so their size is
    bounded by ``chunk_budget_mb`` instead of growing with
    ``num_priors x num_gt x num_classes``. Only the 2D cost and IoU matrices of
    dynamic-k matching are kept whole.

    Args:
        chunk_budget_mb (float | None): Approximate memory for the temporaries
            of one GT slice. ``None`` computes all GTs at once (like mmdet).
            Defaults to 1024.
        **kwargs: ``DynamicSoftLabelAssigner`` arguments (``soft_center_radius``,
            ``topk``, ``iou_weight``, ``iou_calculator``).
    """

    def __init__(self, chunk_budget_mb: float | None = 1024.0, **kwargs: object) -> None:
        super().__init__(**kwargs)
        if chunk_budget_mb is not None and chunk_budget_mb <= 0:
            raise ValueError(f'chunk_budget_mb must be positive or None, got {chunk_budget_mb}')
        self.chunk_budget_mb = chunk_budget_mb

    def candidate_mask(
        self, priors: Tensor, gt_bboxes: BaseBoxes | Tensor, num_bboxes: int
    ) -> Tensor:
        """``(num_priors,)`` bool candidates: priors whose centre is inside a GT."""
        return inside_any_gt(priors[:, :2], gt_bboxes, self.chunk_budget_mb)

    def assign(
        self,
        pred_instances: InstanceData,
        gt_instances: InstanceData,
        gt_instances_ignore: InstanceData | None = None,
        **kwargs: object,
    ) -> AssignResult:
        """Assign GTs to priors (see ``DynamicSoftLabelAssigner.assign``)."""
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
            # No ground truth or boxes, return empty assignment
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes,))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            assigned_labels = decoded_bboxes.new_full((num_bboxes,), -1, dtype=torch.long)
            return AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        valid_mask = self.candidate_mask(priors, gt_bboxes, num_bboxes)

        valid_decoded_bbox = decoded_bboxes[valid_mask]
        valid_pred_scores = pred_scores[valid_mask]
        num_valid = valid_decoded_bbox.size(0)

        if num_valid == 0:
            # No valid bboxes, return empty assignment
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes,))
            assigned_labels = decoded_bboxes.new_full((num_bboxes,), -1, dtype=torch.long)
            return AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
        if hasattr(gt_instances, 'masks'):
            gt_center = center_of_mass(gt_instances.masks, eps=EPS)
        elif isinstance(gt_bboxes, BaseBoxes):
            gt_center = gt_bboxes.centers
        else:
            # Tensor boxes will be treated as horizontal boxes by defaults
            gt_center = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2.0

        cost_matrix, pairwise_ious = dsl_cost_and_ious(
            priors[valid_mask],
            valid_decoded_bbox,
            valid_pred_scores,
            gt_bboxes,
            gt_labels,
            gt_center,
            self.iou_calculator,
            self.soft_center_radius,
            self.iou_weight,
            self.chunk_budget_mb,
        )

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
