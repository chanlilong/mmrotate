# Copyright (c) OpenMMLab. All rights reserved.
"""INT8 refinement beyond range calibration: AWQ, GPTQ and partial quantization.

Applied by :func:`mmrotate.tensorrt.precision.quantize` to a model ModelOpt
has quantized and calibrated; see :func:`refine` for the order.
"""

import logging
from collections.abc import Callable, Sequence

from torch import Tensor, nn

from .awq import awq_clip, awq_scale
from .common import quant_layers
from .gptq import gptq
from .sensitivity import auto_exclude, layer_sensitivity, select_layers

logger = logging.getLogger(__name__)

QUANT_ALGOS = ('awq-scale', 'awq-clip', 'gptq')

__all__ = [
    'QUANT_ALGOS',
    'auto_exclude',
    'awq_clip',
    'awq_scale',
    'gptq',
    'layer_sensitivity',
    'quant_layers',
    'refine',
    'select_layers',
]


def refine(
    model: nn.Module,
    batches: Sequence[Tensor],
    algos: Sequence[str] = (),
    exclude_budget: float | None = None,
    recalibrate: Callable[[nn.Module], None] | None = None,
) -> dict:
    """Refine an 8-bit quantized, calibrated model in place.

    Order (each step sees the result of the previous ones):

    1. ``awq-scale``: per-channel input scales, then ``recalibrate`` (the
       activation ranges change).
    2. ``awq-clip``: weight clip ratios.
    3. ``exclude_budget``: leave the layers that are still too sensitive in
       fp16, with their original weights.
    4. ``gptq``: error-compensating rounding of the remaining layers.

    Args:
        model: The detector with ModelOpt quantizers.
        batches: Preprocessed calibration batches (a few are enough: every
            step runs one or two forward passes per layer).
        algos: Subset of :data:`QUANT_ALGOS`.
        exclude_budget: Output noise budget of the automatic exclusion
            (``None``: keep every layer quantized).
        recalibrate: Re-runs the activation calibration of ``model``.

    Returns:
        A JSON-able report of what each step did.
    """
    unknown = sorted(set(algos) - set(QUANT_ALGOS))
    if unknown:
        raise ValueError(f'Unknown quantization algorithms {unknown}; choose from {QUANT_ALGOS}')
    originals = {layer.name: layer.module.weight.detach().clone() for layer in quant_layers(model)}
    report: dict = dict(layers=len(originals))
    if 'awq-scale' in algos:
        scaled = [r for r in awq_scale(model, batches) if r['alpha'] is not None]
        report['awq_scale'] = dict(scaled=len(scaled), layers=[r['layer'] for r in scaled])
        if recalibrate is not None:
            recalibrate(model)
    if 'awq-clip' in algos:
        clipped = awq_clip(model, batches)
        before = sum(r['err_before'] for r in clipped)
        after = sum(r['err_after'] for r in clipped)
        report['awq_clip'] = dict(layers=len(clipped), error_ratio=after / max(before, 1e-30))
    if exclude_budget is not None:
        result = auto_exclude(model, batches, exclude_budget, originals)
        report['auto_exclude'] = dict(
            budget=exclude_budget, excluded=result['excluded'], noise=result['noise']
        )
    if 'gptq' in algos:
        rounded = gptq(model, batches)
        nearest = sum(r['err_nearest'] for r in rounded)
        best = sum(r['err_gptq'] for r in rounded)
        report['gptq'] = dict(layers=len(rounded), error_ratio=best / max(nearest, 1e-30))
    return report
