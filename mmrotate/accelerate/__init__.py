# Copyright (c) OpenMMLab. All rights reserved.
"""HF accelerate integration: DDP / DeepSpeed ZeRO 0-3 / bf16 / fp16 / fp8 training.

Import (or ``custom_imports: {imports: [mmrotate.accelerate]}``) registers:

* ``AccelerateRunner`` (``runner_type: mmrotate.AccelerateRunner``)
* ``AccelerateOptimWrapper``, ``AccelerateModelWrapper``
* ``AccelerateValLoop`` / ``AccelerateTestLoop`` (parquet result bridge)
* ``ConsistentOrderDataset``, ``ZeRO3EMAHook``, ``AccelerateDistSamplerSeedHook``

See docs/en/user_guides/accelerate_training.md.
"""

from .accelerate_loop import AccelerateTestLoop, AccelerateValLoop, BridgedEvaluator
from .accelerate_optim_wrapper import AccelerateOptimWrapper
from .accelerate_runner import AccelerateRunner
from .accelerator_builder import (
    AccelerateSettings,
    build_accelerator,
    build_deepspeed_config,
    infer_mixed_precision,
    resolve_settings,
)
from .data_order import ConsistentOrderDataset
from .hooks import AccelerateDistSamplerSeedHook, ZeRO3EMAHook
from .model_wrapper import AccelerateModelWrapper
from .result_bridge import ParquetResultBridge

__all__ = [
    'AccelerateDistSamplerSeedHook',
    'AccelerateModelWrapper',
    'AccelerateOptimWrapper',
    'AccelerateRunner',
    'AccelerateSettings',
    'AccelerateTestLoop',
    'AccelerateValLoop',
    'BridgedEvaluator',
    'ConsistentOrderDataset',
    'ParquetResultBridge',
    'ZeRO3EMAHook',
    'build_accelerator',
    'build_deepspeed_config',
    'infer_mixed_precision',
    'resolve_settings',
]
