# Copyright (c) OpenMMLab. All rights reserved.
"""Per-layer INT8 sensitivity and automatic mixed precision.

Some layers do not survive 8 bits whatever the rounding (inputs with a heavy
tailed distribution, layers whose error is amplified downstream). TensorRT
runs such layers in fp16 if they simply have no Q/DQ nodes, at a small cost in
speed, so the usual remedy is partial quantization:

1. :func:`layer_sensitivity`: quantize one layer at a time and measure the
   error it causes at the network outputs, as noise power relative to the
   output power.
2. :func:`select_layers`: keep the layers quantized from the least sensitive
   on, while the summed noise stays within a budget (the errors of different
   layers are roughly independent, so their powers add).
"""

import logging
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from .common import QuantLayer, clear_pre_scale, quant_layers, quantizers_disabled

logger = logging.getLogger(__name__)


def _leaves(outputs: object) -> list[Tensor]:
    if isinstance(outputs, Tensor):
        return [outputs]
    return [leaf for item in outputs for leaf in _leaves(item)]


@torch.no_grad()
def _outputs(model: nn.Module, batches: Sequence[Tensor]) -> list[list[Tensor]]:
    device = next(model.parameters()).device
    return [[leaf.float() for leaf in _leaves(model._forward(x.to(device)))] for x in batches]


def output_noise(outputs: list[list[Tensor]], reference: list[list[Tensor]]) -> float:
    """Mean over the output tensors of ``||out - ref||^2 / ||ref - mean(ref)||^2``."""
    num_leaves = len(reference[0])
    ratios = []
    for i in range(num_leaves):
        err = sum(
            float((o[i] - r[i]).square().sum()) for o, r in zip(outputs, reference, strict=True)
        )
        mean = sum(float(r[i].sum()) for r in reference) / sum(r[i].numel() for r in reference)
        power = sum(float((r[i] - mean).square().sum()) for r in reference)
        ratios.append(err / max(power, 1e-30))
    return sum(ratios) / num_leaves


@torch.no_grad()
def layer_sensitivity(model: nn.Module, batches: Sequence[Tensor]) -> dict[str, float]:
    """Output noise caused by quantizing each layer alone, by layer name."""
    with quantizers_disabled(model):
        reference = _outputs(model, batches)
    result = {}
    for layer in quant_layers(model):
        with quantizers_disabled(model, keep=(layer.input_quantizer, layer.weight_quantizer)):
            result[layer.name] = output_noise(_outputs(model, batches), reference)
    return result


def select_layers(sensitivity: dict[str, float], budget: float) -> list[str]:
    """Names of the layers to leave unquantized so the summed noise fits ``budget``."""
    total = 0.0
    excluded = []
    for name, noise in sorted(sensitivity.items(), key=lambda kv: kv[1]):
        if total + noise <= budget:
            total += noise
        else:
            excluded.append(name)
    return excluded


@torch.no_grad()
def auto_exclude(
    model: nn.Module,
    batches: Sequence[Tensor],
    budget: float,
    originals: dict[str, Tensor] | None = None,
) -> dict:
    """Disable the quantizers of the layers that do not fit the noise budget.

    Args:
        model: Quantized, calibrated model.
        batches: Calibration batches to measure on.
        budget: Allowed sum of single-layer output noise (noise power relative
            to output power; 0.01 is 20 dB).
        originals: Float weights by layer name; excluded layers get them back
            (and lose their input scale), so they run exactly as the float
            layer.

    Returns:
        ``sensitivity`` per layer, the ``excluded`` names and the measured
        ``noise`` of the resulting mixed-precision model.
    """
    sensitivity = layer_sensitivity(model, batches)
    excluded = set(select_layers(sensitivity, budget))
    for layer in quant_layers(model):
        if layer.name in excluded:
            restore_float(layer, (originals or {}).get(layer.name))
    with quantizers_disabled(model):
        reference = _outputs(model, batches)
    noise = output_noise(_outputs(model, batches), reference)
    logger.info(
        'Sensitivity: %d of %d layers kept in fp16 (budget %.3g), output noise %.3g',
        len(excluded), len(sensitivity), budget, noise,
    )  # fmt: skip
    for name in sorted(excluded, key=lambda n: -sensitivity[n])[:10]:
        logger.info('  %s: noise %.3g', name, sensitivity[name])
    return dict(sensitivity=sensitivity, excluded=sorted(excluded), noise=noise)


def restore_float(layer: QuantLayer, weight: Tensor | None) -> None:
    """Turn ``layer`` back into its float form."""
    layer.input_quantizer.disable()
    layer.weight_quantizer.disable()
    if weight is not None:
        layer.module.weight.data.copy_(weight.to(layer.module.weight))
        clear_pre_scale(layer)
