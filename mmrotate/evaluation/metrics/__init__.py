# Copyright (c) OpenMMLab. All rights reserved.
from .dota_metric import DOTAMetric
from .rotated_coco_metric import RotatedCocoMetric
from .soda import *  # noqa: F401,F403

__all__ = ['DOTAMetric', 'RotatedCocoMetric']
