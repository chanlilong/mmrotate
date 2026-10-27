# Copyright (c) OpenMMLab. All rights reserved.
"""Coarse-to-fine assignment (C2F) from DCFL (Xu et al., CVPR 2023).

Ported from ``mmrotate/core/bbox/assigners/coarse2fine_assigner.py`` of
https://github.com/Chasel-Tsui/mmrotate-dcfl (Apache-2.0) to the mmdet 3.x
assigner interface, vectorised, and extended to point priors.
"""

import math

import torch
from mmdet.models.task_modules.assigners import AssignResult, BaseAssigner
from mmengine.structures import InstanceData
from torch import Tensor

from mmrotate.registry import TASK_UTILS

from ..nwd_rka.nwd import box_to_gaussian
from ..nwd_rka.nwd_calculator import to_box_tensor, to_rbox_tensor
from ..nwd_rka.ranking_assigner import prior_boxes
from .gjsd import gaussian_density


@TASK_UTILS.register_module()
class C2FAssigner(BaseAssigner):
    """Coarse-to-fine assigner of DCFL.

    1. **Coarse:** every GT takes its ``topk`` priors with the highest
       ``iou_calculator`` metric (GJSD similarity by default) as candidates;
       a prior that is a candidate of several GTs goes to the GT with the
       highest index (as in the reference). Priors whose best metric is in
       ``[0, neg_thr)`` are negatives, the other non-candidates are ignored.
    2. **Fine:** every GT keeps its ``topq`` candidates with the highest
       quality ``IoU(pred, gt) + max_c sigmoid(score_c)`` as positives; the
       other candidates become negatives.
    3. **Constraint** (``constraint='dgmm'``): a positive is kept only if
       ``w * N(p; gt) + (1 - w) * N(p; mean of the GT's positive priors)``
       is at least ``exp(-gauss_thr)``, where ``N`` is the unnormalised
       Gaussian of the GT's covariance and ``p`` the prior centre; rejected
       positives become negatives.

    With the dynamic priors of the DCFL heads, ``pred_instances.priors`` are
    the priors shifted by the mean DCN offset.

    Differences from the reference implementation:

    - The fine step ranks each GT's own candidates. The reference ranks a
      ``(num_gts, num_candidates)`` matrix padded with -100, so a GT with fewer
      than ``topq`` candidates also picked up other GTs' candidates.
    - The constraint is only applied to positives. The reference also
      evaluated it for ignored priors against GT 0 and made the ones outside
      negatives.
    - The GJSD is evaluated in closed form (see :mod:`.gjsd`) in fp32 instead
      of fp16 on CUDA, and the quality uses the aligned IoU.

    Args:
        topk (int): Coarse candidates per GT. Defaults to 16.
        topq (int): Fine positives per GT. Defaults to 12.
        neg_thr (float): Priors whose best metric is below this are negatives.
            Defaults to 0.8.
        constraint (str, optional): ``'dgmm'`` or None. Defaults to ``'dgmm'``.
        gauss_thr (float): Constraint threshold, positives need a density of
            at least ``exp(-gauss_thr)``. Defaults to 0.6.
        dgmm_weight (float): Weight ``w`` of the GT Gaussian in the
            constraint. Defaults to 0.7.
        iou_calculator (dict): Coarse metric between GTs and priors.
            Defaults to ``dict(type='RBboxGJSD2D')``.
        pos_iou_calculator (dict): IoU between predictions and GTs in the
            quality. Defaults to ``dict(type='RBboxOverlaps2D')``.
        point_prior_scale (float, optional): If set, priors are points
            ``(x, y, stride_w, stride_h)`` and are turned into boxes of
            ``scale * stride`` for the coarse step (RTMDet/FCOS-style heads).
        quality_calculator (dict, optional): If set, ``max_overlaps`` of
            positives is replaced by the aligned quality of each predicted box
            w.r.t. its GT (detached). Heads such as RTMDet use
            ``max_overlaps`` as the soft classification target and box-loss
            weight.
        ignore_iof_thr (float): IoF threshold for ignoring priors that overlap
            ``gt_instances_ignore``. Negative disables. Defaults to -1.
        ignore_wrt_candidates (bool): Compute IoF w.r.t. the priors (True) or
            the ignore boxes (False). Defaults to True.
        gpu_assign_thr (int): Assign on CPU when there are more GTs than this.
            Negative disables. Defaults to -1.
    """

    def __init__(
        self,
        topk: int = 16,
        topq: int = 12,
        neg_thr: float = 0.8,
        constraint: str | None = 'dgmm',
        gauss_thr: float = 0.6,
        dgmm_weight: float = 0.7,
        iou_calculator: dict | None = None,
        pos_iou_calculator: dict | None = None,
        point_prior_scale: float | None = None,
        quality_calculator: dict | None = None,
        ignore_iof_thr: float = -1,
        ignore_wrt_candidates: bool = True,
        gpu_assign_thr: int = -1,
    ) -> None:
        assert constraint in (None, 'dgmm'), f'unknown constraint {constraint}'
        self.topk = topk
        self.topq = topq
        self.neg_thr = neg_thr
        self.constraint = constraint
        self.gauss_thr = gauss_thr
        self.dgmm_weight = dgmm_weight
        self.iou_calculator = TASK_UTILS.build(iou_calculator or dict(type='RBboxGJSD2D'))
        self.pos_iou_calculator = TASK_UTILS.build(
            pos_iou_calculator or dict(type='RBboxOverlaps2D')
        )
        self.point_prior_scale = point_prior_scale
        self.quality_calculator = (
            TASK_UTILS.build(quality_calculator) if quality_calculator is not None else None
        )
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.gpu_assign_thr = gpu_assign_thr

    def assign(
        self,
        pred_instances: InstanceData,
        gt_instances: InstanceData,
        gt_instances_ignore: InstanceData | None = None,
        **kwargs: object,
    ) -> AssignResult:
        """Assign GTs to priors.

        Args:
            pred_instances (InstanceData): ``priors`` (rotated anchors, or
                points if ``point_prior_scale`` is set), decoded ``bboxes`` and
                classification ``scores`` (logits).
            gt_instances (InstanceData): GT ``bboxes`` and ``labels``.
            gt_instances_ignore (InstanceData, optional): Boxes to ignore.

        Returns:
            AssignResult: ``gt_inds`` is -1 (ignore), 0 (negative) or the
            1-based GT index.
        """
        assert 'bboxes' in pred_instances and 'scores' in pred_instances, (
            'C2FAssigner needs the decoded predictions (bboxes) and scores'
        )
        gt_bboxes = to_rbox_tensor(to_box_tensor(gt_instances.bboxes))
        gt_labels = gt_instances.labels
        priors = prior_boxes(pred_instances.priors, self.point_prior_scale)
        pred_bboxes = to_rbox_tensor(to_box_tensor(pred_instances.bboxes)).detach()
        pred_scores = pred_instances.scores.detach()
        ignore_bboxes = None
        if gt_instances_ignore is not None and 'bboxes' in gt_instances_ignore:
            ignore_bboxes = to_box_tensor(gt_instances_ignore.bboxes)

        device = priors.device
        assign_on_cpu = 0 < self.gpu_assign_thr < gt_bboxes.size(0)
        if assign_on_cpu:
            priors, pred_bboxes, pred_scores = priors.cpu(), pred_bboxes.cpu(), pred_scores.cpu()
            gt_bboxes, gt_labels = gt_bboxes.cpu(), gt_labels.cpu()
            if ignore_bboxes is not None:
                ignore_bboxes = ignore_bboxes.cpu()

        overlaps = self.iou_calculator(gt_bboxes, priors)

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

        assign_result = self.assign_wrt_c2f(
            overlaps, priors, pred_bboxes, pred_scores, gt_bboxes, gt_labels
        )

        if assign_on_cpu:
            assign_result.gt_inds = assign_result.gt_inds.to(device)
            assign_result.max_overlaps = assign_result.max_overlaps.to(device)
            assign_result.labels = assign_result.labels.to(device)
            pred_bboxes, gt_bboxes = pred_bboxes.to(device), gt_bboxes.to(device)

        if self.quality_calculator is not None:
            pos = assign_result.gt_inds > 0
            if pos.any():
                pos_gt_bboxes = gt_bboxes[assign_result.gt_inds[pos] - 1]
                quality = self.quality_calculator(pred_bboxes[pos], pos_gt_bboxes, is_aligned=True)
                assign_result.max_overlaps[pos] = quality.to(assign_result.max_overlaps.dtype)
        return assign_result

    def assign_wrt_c2f(
        self,
        overlaps: Tensor,
        priors: Tensor,
        pred_bboxes: Tensor,
        pred_scores: Tensor,
        gt_bboxes: Tensor,
        gt_labels: Tensor,
    ) -> AssignResult:
        """Assign from a ``(num_gts, num_priors)`` coarse metric matrix.

        Args:
            overlaps (Tensor): ``(G, N)`` coarse metric between GTs and priors.
            priors (Tensor): ``(N, 4|5)`` prior boxes (centres are used by the
                constraint).
            pred_bboxes (Tensor): ``(N, 5)`` decoded predicted rotated boxes.
            pred_scores (Tensor): ``(N, C)`` classification logits.
            gt_bboxes (Tensor): ``(G, 5)`` rotated GT boxes.
            gt_labels (Tensor): ``(G,)`` GT labels.

        Returns:
            AssignResult: the assignment; ``max_overlaps`` is the best coarse
            metric of every prior.
        """
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

        # Coarse: every prior whose metric reaches a GT's k-th best value is a
        # candidate of that GT (ties included, as in the reference loop); a
        # prior claimed by several GTs goes to the last one.
        topk = min(self.topk, num_priors)
        kth = overlaps.topk(topk, dim=1).values[:, -1:]
        is_cand = (overlaps >= kth) & (overlaps >= 0)
        cand_inds = is_cand.any(dim=0).nonzero().squeeze(1)
        last_gt = num_gts - 1 - is_cand[:, cand_inds].flip(0).to(torch.uint8).argmax(dim=0)
        # candidates the fine step does not keep become negatives
        assigned_gt_inds[cand_inds] = 0

        # Fine: per GT, the top-q candidates by IoU(pred, gt) + best score.
        cand_iou = self.pos_iou_calculator(
            pred_bboxes[cand_inds], gt_bboxes[last_gt], is_aligned=True
        )
        cand_cls = pred_scores[cand_inds].float().sigmoid().max(dim=1).values
        quality = cand_iou.float() + cand_cls
        order = quality.argsort(descending=True)
        order = order[last_gt[order].argsort(stable=True)]
        sorted_gt = last_gt[order]
        counts = torch.bincount(sorted_gt, minlength=num_gts)
        starts = counts.cumsum(0) - counts
        rank = torch.arange(order.numel(), device=order.device) - starts[sorted_gt]
        keep = order[rank < self.topq]
        pos_inds, pos_gt = cand_inds[keep], last_gt[keep]
        assigned_gt_inds[pos_inds] = pos_gt + 1

        if self.constraint == 'dgmm' and pos_inds.numel() > 0:
            centers = priors[:, :2].float()
            pos_centers = centers[pos_inds]
            num_pos = torch.bincount(pos_gt, minlength=num_gts).clamp(min=1)
            pos_mean = centers.new_zeros((num_gts, 2)).index_add_(0, pos_gt, pos_centers)
            pos_mean = pos_mean / num_pos[:, None]
            gauss = box_to_gaussian(gt_bboxes, 1e-7)[pos_gt]
            density = self.dgmm_weight * gaussian_density(pos_centers, gauss[:, :2], gauss) + (
                1 - self.dgmm_weight
            ) * gaussian_density(pos_centers, pos_mean[pos_gt], gauss)
            assigned_gt_inds[pos_inds[density < math.exp(-self.gauss_thr)]] = 0

        pos = assigned_gt_inds > 0
        assigned_labels[pos] = gt_labels[assigned_gt_inds[pos] - 1].long()
        return AssignResult(num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)

    def __repr__(self) -> str:
        """str: a string describing the module."""
        return (
            f'{self.__class__.__name__}(topk={self.topk}, topq={self.topq}, '
            f'neg_thr={self.neg_thr}, constraint={self.constraint}, '
            f'gauss_thr={self.gauss_thr}, iou_calculator={self.iou_calculator}, '
            f'point_prior_scale={self.point_prior_scale}, '
            f'quality_calculator={self.quality_calculator})'
        )
