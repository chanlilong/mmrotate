# Copyright (c) OpenMMLab. All rights reserved.
"""NWD-RKA: Normalized Wasserstein Distance + Ranking-based Assignment.

Xu et al., "Detecting tiny objects in aerial images: A normalized Wasserstein
distance and a new benchmark", ISPRS J. 2022, reference code
https://github.com/Chasel-Tsui/mmdet-aitod. Generalised here to rotated boxes
and point priors; NWD-NMS from https://github.com/jwwangchn/mmcv-nwd is
reimplemented with Triton so no custom mmcv build is required.

Registered in ``TASK_UTILS``: ``RBboxNWD2D`` (IoU-calculator interface),
``RankingAssigner`` and ``RankingDynamicSoftLabelAssigner``. ``nwd_nms`` is
usable as ``nms=dict(type='nwd_nms', ...)`` in any ``batched_nms`` config.
"""

from .nwd import box_to_gaussian, rbbox_nwd
from .nwd_calculator import RBboxNWD2D
from .nwd_nms import nwd_nms
from .ranking_assigner import RankingAssigner
from .ranking_dsl_assigner import RankingDynamicSoftLabelAssigner

__all__ = [
    'RBboxNWD2D',
    'RankingAssigner',
    'RankingDynamicSoftLabelAssigner',
    'box_to_gaussian',
    'nwd_nms',
    'rbbox_nwd',
]
