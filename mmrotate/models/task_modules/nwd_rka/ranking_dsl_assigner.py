# Copyright (c) OpenMMLab. All rights reserved.
"""Dynamic soft-label assignment with RKA candidates (hybrid DSL + RKA).

``assign`` is adapted from mmdet 3.3.0
``mmdet/models/task_modules/assigners/dynamic_soft_label_assigner.py``
(Apache-2.0, OpenMMLab); the only functional change is the candidate mask.
"""

import torch
import torch.nn.functional as F
from mmdet.models.task_modules.assigners import AssignResult, DynamicSoftLabelAssigner
from mmdet.models.task_modules.assigners.dynamic_soft_label_assigner import (
    EPS,
    INF,
    center_of_mass,
)
from mmdet.structures.bbox import BaseBoxes
from mmengine.structures import InstanceData

from mmrotate.registry import TASK_UTILS

from .nwd_calculator import to_box_tensor
from .ranking_assigner import point_priors_to_boxes


@TASK_UTILS.register_module()
class RankingDynamicSoftLabelAssigner(DynamicSoftLabelAssigner):
    """``DynamicSoftLabelAssigner`` whose candidates also include each GT's
    top-k priors by NWD.

    Stock DSL only considers priors whose centre lies inside some GT, so a GT
    smaller than the prior stride can end up with no candidate and therefore
    no positive. Here the candidate set is the union of those priors and each
    GT's ``rka_topk`` priors ranked by ``rka_calculator`` between GTs and
    ``rka_prior_scale * stride`` prior boxes (the RKA step of NWD-RKA). The
    cost, dynamic-k matching and soft labels are unchanged and use
    ``iou_calculator``, which can itself be ``RBboxNWD2D``.

    Args:
        rka_topk (int): Extra candidates per GT. Defaults to 3.
        rka_prior_scale (float): Prior box size in strides. Defaults to 4.0.
        rka_calculator (dict): Metric used to rank priors per GT.
            Defaults to ``dict(type='RBboxNWD2D')``.
        **kwargs: ``DynamicSoftLabelAssigner`` arguments (``soft_center_radius``,
            ``topk``, ``iou_weight``, ``iou_calculator``).
    """

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
