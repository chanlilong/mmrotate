# Copyright (c) OpenMMLab. All rights reserved.
"""Ranking-based assignment (RKA) from NWD-RKA (Xu et al., ISPRS 2022).

Ported from ``mmdet/core/bbox/assigners/ranking_assigner.py`` of
https://github.com/Chasel-Tsui/mmdet-aitod (Apache-2.0) to the mmdet 3.x
assigner interface, vectorised, and extended to rotated boxes and point priors.
"""

import torch
from mmdet.models.task_modules.assigners import AssignResult, BaseAssigner
from mmdet.structures.bbox import BaseBoxes
from mmengine.structures import InstanceData
from torch import Tensor

from mmrotate.registry import TASK_UTILS

from .nwd_calculator import to_box_tensor


def point_priors_to_boxes(priors: Tensor, scale: float) -> Tensor:
    """Turn point priors ``(x, y, stride_w, stride_h)`` into rotated boxes
    ``(x, y, scale * stride_w, scale * stride_h, 0)``."""
    assert priors.size(-1) >= 4, 'point priors need strides, use MlvlPointGenerator(with_stride)'
    return torch.cat(
        [priors[:, :2], priors[:, 2:4] * scale, priors.new_zeros((priors.size(0), 1))], dim=-1
    )


def prior_boxes(priors: BaseBoxes | Tensor, point_prior_scale: float | None) -> Tensor:
    """Box tensor for anchor priors, or boxes built from point priors."""
    if point_prior_scale is None:
        return to_box_tensor(priors)
    return point_priors_to_boxes(priors, point_prior_scale)


@TASK_UTILS.register_module()
class RankingAssigner(BaseAssigner):
    """Ranking-based assigner (RKA).

    Each GT takes its ``topk`` priors with the highest metric (NWD by
    default) as positives, regardless of the absolute value, so tiny GTs
    always get positives. Priors whose best metric over all GTs is in
    ``[0, neg_thr)`` are negatives; the remaining priors are ignored (-1).

    Differences from the reference implementation: positives are taken from
    the top-k indices (the reference marks every prior whose value *equals* a
    top-k value, so ties all become positive), and a prior in the top-k of
    several GTs goes to the GT with the highest metric (the reference keeps the
    last GT in index order).

    Works with anchor priors (horizontal or rotated boxes, e.g. RetinaNet or
    the RPN) and, with ``point_prior_scale``, with point priors
    ``(x, y, stride_w, stride_h)`` as used by RTMDet/FCOS-style heads.

    Args:
        topk (int): Positives per GT. The paper uses 2 for the RPN and 3 for
            RetinaNet. Defaults to 1.
        neg_thr (float): Priors whose best metric is below this are negatives.
            Defaults to 0.3.
        iou_calculator (dict): Metric between GTs and priors.
            Defaults to ``dict(type='RBboxNWD2D')``.
        ignore_iof_thr (float): IoF threshold for ignoring priors that overlap
            ``gt_instances_ignore``. Negative disables. Defaults to -1.
        ignore_wrt_candidates (bool): Compute IoF w.r.t. the priors (True) or
            the ignore boxes (False). Defaults to True.
        gpu_assign_thr (int): Assign on CPU when there are more GTs than this.
            Negative disables. Defaults to -1.
        point_prior_scale (float, optional): If set, priors are points and are
            turned into boxes of ``scale * stride`` for ranking.
        quality_calculator (dict, optional): If set and ``pred_instances`` has
            decoded ``bboxes``, ``max_overlaps`` of positives is replaced by
            the aligned quality of each predicted box w.r.t. its GT (detached).
            Heads such as RTMDet use ``max_overlaps`` as the soft
            classification target and box-loss weight.
    """

    def __init__(
        self,
        topk: int = 1,
        neg_thr: float = 0.3,
        iou_calculator: dict | None = None,
        ignore_iof_thr: float = -1,
        ignore_wrt_candidates: bool = True,
        gpu_assign_thr: int = -1,
        point_prior_scale: float | None = None,
        quality_calculator: dict | None = None,
    ) -> None:
        self.topk = topk
        self.neg_thr = neg_thr
        self.iou_calculator = TASK_UTILS.build(iou_calculator or dict(type='RBboxNWD2D'))
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.gpu_assign_thr = gpu_assign_thr
        self.point_prior_scale = point_prior_scale
        self.quality_calculator = (
            TASK_UTILS.build(quality_calculator) if quality_calculator is not None else None
        )

    def assign(
        self,
        pred_instances: InstanceData,
        gt_instances: InstanceData,
        gt_instances_ignore: InstanceData | None = None,
        **kwargs: object,
    ) -> AssignResult:
        """Assign GTs to priors.

        Args:
            pred_instances (InstanceData): Must have ``priors``; may have
                decoded ``bboxes`` (used by ``quality_calculator``).
            gt_instances (InstanceData): GT ``bboxes`` and ``labels``.
            gt_instances_ignore (InstanceData, optional): Boxes to ignore.

        Returns:
            AssignResult: ``gt_inds`` is -1 (ignore), 0 (negative) or the
            1-based GT index.
        """
        gt_bboxes = to_box_tensor(gt_instances.bboxes)
        gt_labels = gt_instances.labels
        priors = prior_boxes(pred_instances.priors, self.point_prior_scale)
        ignore_bboxes = None
        if gt_instances_ignore is not None and 'bboxes' in gt_instances_ignore:
            ignore_bboxes = to_box_tensor(gt_instances_ignore.bboxes)

        device = priors.device
        assign_on_cpu = 0 < self.gpu_assign_thr < gt_bboxes.size(0)
        if assign_on_cpu:
            priors, gt_labels = priors.cpu(), gt_labels.cpu()
            if ignore_bboxes is not None:
                ignore_bboxes = ignore_bboxes.cpu()

        overlaps = self.iou_calculator(gt_bboxes.to(priors.device), priors)

        if (
            self.ignore_iof_thr > 0
            and ignore_bboxes is not None
            and ignore_bboxes.numel() > 0
            and priors.numel() > 0
        ):
            if self.ignore_wrt_candidates:
                ignore_overlaps = self.iou_calculator(priors, ignore_bboxes, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            else:
                ignore_overlaps = self.iou_calculator(ignore_bboxes, priors, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
            overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1

        assign_result = self.assign_wrt_ranking(overlaps, gt_labels)

        if assign_on_cpu:
            assign_result.gt_inds = assign_result.gt_inds.to(device)
            assign_result.max_overlaps = assign_result.max_overlaps.to(device)
            assign_result.labels = assign_result.labels.to(device)

        if self.quality_calculator is not None and 'bboxes' in pred_instances:
            pos = assign_result.gt_inds > 0
            if pos.any():
                pred_bboxes = to_box_tensor(pred_instances.bboxes)[pos].detach()
                pos_gt_bboxes = gt_bboxes[assign_result.gt_inds[pos] - 1]
                quality = self.quality_calculator(pred_bboxes, pos_gt_bboxes, is_aligned=True)
                assign_result.max_overlaps[pos] = quality.to(assign_result.max_overlaps.dtype)
        return assign_result

    def assign_wrt_ranking(self, overlaps: Tensor, gt_labels: Tensor) -> AssignResult:
        """Assign from a ``(num_gts, num_priors)`` metric matrix."""
        num_gts, num_priors = overlaps.shape
        assigned_gt_inds = overlaps.new_full((num_priors,), -1, dtype=torch.long)
        assigned_labels = overlaps.new_full((num_priors,), -1, dtype=torch.long)

        if num_gts == 0 or num_priors == 0:
            max_overlaps = overlaps.new_zeros((num_priors,))
            if num_gts == 0:
                assigned_gt_inds[:] = 0
            return AssignResult(num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        max_overlaps, _ = overlaps.max(dim=0)
        assigned_gt_inds[(max_overlaps >= 0) & (max_overlaps < self.neg_thr)] = 0

        # Every GT's top-k are candidates (ignored priors, metric -1, excluded);
        # a prior ranked by several GTs goes to the one with the highest metric.
        topk = min(self.topk, num_priors)
        topk_overlaps, topk_inds = overlaps.topk(topk, dim=1)
        cand_overlaps, cand_priors = topk_overlaps.flatten(), topk_inds.flatten()
        cand_gts = torch.arange(num_gts, device=overlaps.device).repeat_interleave(topk)
        valid = cand_overlaps >= 0
        cand_overlaps, cand_priors, cand_gts = (
            cand_overlaps[valid],
            cand_priors[valid],
            cand_gts[valid],
        )
        best = overlaps.new_full((num_priors,), float('-inf'))
        best.scatter_reduce_(0, cand_priors, cand_overlaps, reduce='amax')
        win = cand_overlaps == best[cand_priors]
        assigned_gt_inds[cand_priors[win]] = cand_gts[win] + 1
        pos = assigned_gt_inds > 0
        assigned_labels[pos] = gt_labels[assigned_gt_inds[pos] - 1].long()
        return AssignResult(num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)

    def __repr__(self) -> str:
        """str: a string describing the module."""
        return (
            f'{self.__class__.__name__}(topk={self.topk}, neg_thr={self.neg_thr}, '
            f'iou_calculator={self.iou_calculator}, '
            f'point_prior_scale={self.point_prior_scale}, '
            f'quality_calculator={self.quality_calculator})'
        )
