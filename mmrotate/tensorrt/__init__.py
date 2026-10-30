# Copyright (c) OpenMMLab. All rights reserved.
"""TensorRT export and inference for mmrotate single-stage detectors.

See ``TENSORRT.md`` in this folder. ``tensorrt`` and ``modelopt`` are only
imported when an engine is built, run or quantized.
"""

from .deploy import init_trt_detector, make_deploy_cfg
from .detector import TRTSingleStageDetector
from .export import build_detector, export_onnx, flatten_outputs, unflatten_outputs
from .export_patches import export_mode

__all__ = [
    'TRTSingleStageDetector',
    'build_detector',
    'export_mode',
    'export_onnx',
    'flatten_outputs',
    'init_trt_detector',
    'make_deploy_cfg',
    'unflatten_outputs',
]
