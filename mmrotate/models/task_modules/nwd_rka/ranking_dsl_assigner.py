# Copyright (c) OpenMMLab. All rights reserved.
"""Dynamic soft-label assignment with RKA candidates (hybrid DSL + RKA).

Built on :class:`ChunkedDynamicSoftLabelAssigner` (mmdet's DSL with GT-sliced,
memory-bounded cost computation); the only functional change from DSL is the
candidate mask.
"""

from mmdet.structures.bbox import BaseBoxes
from torch import Tensor

from mmrotate.registry import TASK_UTILS

from ..chunked_dsl import ChunkedDynamicSoftLabelAssigner, topk_per_gt_mask
from .nwd_calculator import to_box_tensor
from .ranking_assigner import point_priors_to_boxes


@TASK_UTILS.register_module()
class RankingDynamicSoftLabelAssigner(ChunkedDynamicSoftLabelAssigner):
    """``DynamicSoftLabelAssigner`` whose candidates also include each GT's
    top-k priors by NWD.

    Stock DSL only considers priors whose centre lies inside some GT, so a GT
    smaller than the prior stride can end up with no candidate and therefore
    no positive. Here the candidate set is the union of those priors and each
    GT's ``rka_topk`` priors ranked by ``rka_calculator`` between GTs and
    ``rka_prior_scale * stride`` prior boxes (the RKA step of NWD-RKA). The
    cost, dynamic-k matching and soft labels are unchanged and use
    ``iou_calculator``, which can itself be ``RBboxNWD2D``.

    All (prior, GT) matrices are computed over slices of GTs bounded by
    ``chunk_budget_mb`` (see :class:`ChunkedDynamicSoftLabelAssigner`); the
    assignment does not depend on it.

    Args:
        rka_topk (int): Extra candidates per GT. Defaults to 3.
        rka_prior_scale (float): Prior box size in strides. Defaults to 4.0.
        rka_calculator (dict): Metric used to rank priors per GT.
            Defaults to ``dict(type='RBboxNWD2D')``.
        chunk_budget_mb (float | None): Memory for one GT slice's temporaries;
            ``None`` computes all GTs at once. Defaults to 1024.
        **kwargs: ``DynamicSoftLabelAssigner`` arguments (``soft_center_radius``,
            ``topk``, ``iou_weight``, ``iou_calculator``).
    """

    def __init__(
        self,
        rka_topk: int = 3,
        rka_prior_scale: float = 4.0,
        rka_calculator: dict | None = None,
        chunk_budget_mb: float | None = 1024.0,
        **kwargs: object,
    ) -> None:
        super().__init__(chunk_budget_mb=chunk_budget_mb, **kwargs)
        self.rka_topk = rka_topk
        self.rka_prior_scale = rka_prior_scale
        self.rka_calculator = TASK_UTILS.build(rka_calculator or dict(type='RBboxNWD2D'))

    def candidate_mask(
        self, priors: Tensor, gt_bboxes: BaseBoxes | Tensor, num_bboxes: int
    ) -> Tensor:
        """Priors inside a GT, plus every GT's ``rka_topk`` priors by NWD (RKA)."""
        is_in_gts = super().candidate_mask(priors, gt_bboxes, num_bboxes)
        is_rka = topk_per_gt_mask(
            point_priors_to_boxes(priors, self.rka_prior_scale),
            to_box_tensor(gt_bboxes),
            self.rka_calculator,
            min(self.rka_topk, num_bboxes),
            self.chunk_budget_mb,
        )
        return is_in_gts | is_rka
