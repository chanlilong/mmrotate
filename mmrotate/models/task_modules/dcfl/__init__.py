# Copyright (c) OpenMMLab. All rights reserved.
"""DCFL: Dynamic Coarse-to-Fine Learning for oriented tiny objects.

Xu et al., "Dynamic Coarse-to-Fine Learning for Oriented Tiny Object
Detection", CVPR 2023, reference code https://github.com/Chasel-Tsui/mmrotate-dcfl.
This package holds the label-assignment side: the GJSD prior metric
(``RBboxGJSD2D``) and the coarse-to-fine assigner (``C2FAssigner``), both
registered in ``TASK_UTILS``. The dynamic-prior heads are in
``mmrotate/models/dense_heads/dcfl``.
"""

from .c2f_assigner import C2FAssigner
from .gjsd import gaussian_density, gaussian_gjsd, rbbox_gjsd
from .gjsd_calculator import RBboxGJSD2D

__all__ = [
    'C2FAssigner',
    'RBboxGJSD2D',
    'gaussian_density',
    'gaussian_gjsd',
    'rbbox_gjsd',
]
