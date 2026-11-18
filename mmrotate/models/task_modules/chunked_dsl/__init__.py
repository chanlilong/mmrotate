# Copyright (c) OpenMMLab. All rights reserved.
"""Memory-bounded dynamic soft-label assignment.

``ChunkedDynamicSoftLabelAssigner`` (registered in ``TASK_UTILS``) gives the same
assignment as RTMDet's ``mmdet.DynamicSoftLabelAssigner`` with the
(prior, GT) temporaries computed over slices of GTs, so dense images with
thousands of GTs no longer need tens of GB in the assigner.
"""

from .chunked_cost import dsl_cost_and_ious, gt_slices, inside_any_gt, topk_per_gt_mask
from .chunked_dsl_assigner import ChunkedDynamicSoftLabelAssigner

__all__ = [
    'ChunkedDynamicSoftLabelAssigner',
    'dsl_cost_and_ious',
    'gt_slices',
    'inside_any_gt',
    'topk_per_gt_mask',
]
