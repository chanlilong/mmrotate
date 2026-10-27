# Copyright (c) OpenMMLab. All rights reserved.
"""DCFL heads: Dynamic Coarse-to-Fine Learning for oriented tiny objects.

Xu et al., "Dynamic Coarse-to-Fine Learning for Oriented Tiny Object
Detection", CVPR 2023, reference code https://github.com/Chasel-Tsui/mmrotate-dcfl.
Registered in ``MODELS``: ``RotatedDCFLHead`` (the paper's rotated RetinaNet
head) and ``RotatedRTMDetSepBNDCFLHead`` (the same dynamic priors on the
rotated RTMDet head). Both feed DCN-shifted priors to the assigner, normally
``C2FAssigner`` from ``mmrotate/models/task_modules/dcfl``. The DCNv2 of
MMCV-DCFL runs on official mmcv ops (or torchvision), no custom build needed.
"""

from .layers import (
    DCFLModulatedDeformConv,
    DCFLSharedOffsetDeformConv,
    dynamic_prior_shift,
    modulated_deform_conv,
)
from .rotated_dcfl_head import RotatedDCFLHead
from .rotated_rtmdet_dcfl_head import RotatedRTMDetSepBNDCFLHead

__all__ = [
    'DCFLModulatedDeformConv',
    'DCFLSharedOffsetDeformConv',
    'RotatedDCFLHead',
    'RotatedRTMDetSepBNDCFLHead',
    'dynamic_prior_shift',
    'modulated_deform_conv',
]
