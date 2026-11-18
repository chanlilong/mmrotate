# Copyright (c) OpenMMLab. All rights reserved.
"""AWQ (activation-aware weight quantization) for W8A8 conv / linear layers.

AWQ (Lin et al., 2023) has two parts, both searched per layer against the
layer's float output on calibration data:

**Scale search** (:func:`awq_scale`). Each input channel ``c`` is divided by
``s_c`` before quantization and its weights multiplied by ``s_c``, which leaves
the float layer unchanged. With ``s_c = act_range_c^a / w_range_c^(1 - a)`` the
search over ``a`` moves dynamic range between the activations and the weights.
For W8A8 this is what repairs the per-tensor activation quantization TensorRT
is limited to: one outlier channel (range 157 against a typical 1 at the neck
input of the DINOv3 models) otherwise sets the step size of all channels. The
division is not folded into the previous layer (the tensor usually feeds
several layers and passes through non-linearities): it stays as an explicit
per-channel multiply in front of the Q/DQ pair (``pre_quant_scale`` of the
ModelOpt quantizer), i.e. per-channel activation quantization.

**Clip search** (:func:`awq_clip`). Each output channel's weight range is
shrunk by the ratio that minimises that channel's output error, trading
clipping of a few large weights for a finer step.
"""

import logging
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from .common import (
    QuantLayer,
    capture,
    channel_shape,
    fake_quant,
    layer_forward,
    pre_scale,
    quant_layers,
    quantizers_disabled,
    weight_amax,
)

logger = logging.getLogger(__name__)

ALPHAS = tuple(i / 10 for i in range(11))
CLIP_RATIOS = tuple(1.0 - 0.05 * i for i in range(11))


def input_channel_range(layer: QuantLayer, x: Tensor) -> Tensor:
    """``max |x|`` per input channel."""
    dims = (0, 2, 3) if layer.is_conv else tuple(range(x.dim() - 1))
    return x.abs().amax(dims)


def weight_channel_range(layer: QuantLayer, weight: Tensor) -> Tensor:
    """``max |w|`` over the weights that multiply each input channel."""
    if not layer.is_conv:
        return weight.abs().amax(0)
    g = layer.groups
    out, in_g, kh, kw = weight.shape
    return weight.abs().reshape(g, out // g, in_g, kh, kw).amax((1, 3, 4)).reshape(-1)


def scale_weight(layer: QuantLayer, weight: Tensor, s: Tensor) -> Tensor:
    """``weight`` with the weights of input channel ``c`` multiplied by ``s[c]``."""
    if not layer.is_conv:
        return weight * s[None, :]
    g = layer.groups
    out, in_g, kh, kw = weight.shape
    scaled = weight.reshape(g, out // g, in_g, kh, kw) * s.reshape(g, 1, in_g, 1, 1)
    return scaled.reshape(out, in_g, kh, kw)


def _quantized_error(
    layer: QuantLayer, xs: Sequence[Tensor], refs: Sequence[Tensor], weight: Tensor, s: Tensor
) -> float:
    """Squared output error of the W8A8 layer with input scale ``1 / s``."""
    shape = channel_shape(layer)
    w = scale_weight(layer, weight, s)
    wq = fake_quant(w, weight_amax(w))
    inv = (1.0 / s).reshape(shape)
    amax = max(float((x * inv).abs().max()) for x in xs)
    amax = torch.tensor(amax, device=weight.device)
    err = 0.0
    for x, ref in zip(xs, refs, strict=True):
        out = layer_forward(layer, fake_quant(x * inv, amax), wq)
        err += float((out - ref).square().sum())
    return err


@torch.no_grad()
def awq_scale_layer(
    layer: QuantLayer,
    xs: Sequence[Tensor],
    alphas: Sequence[float] = ALPHAS,
    min_gain: float = 0.1,
) -> dict:
    """Search and apply the per-input-channel scale of one layer.

    Args:
        layer: The layer; its weight and input quantizer are updated in place
            if a scale lowers the quantized output error by ``min_gain``.
        xs: Float inputs of the layer (before any input scale).
        alphas: Exponents to try.
        min_gain: Minimum relative error reduction to apply a scale.

    Returns:
        ``alpha`` (``None`` if unscaled), and the errors without / with it.
    """
    weight = layer.module.weight.detach().float()
    refs = [layer_forward(layer, x, weight) for x in xs]
    ones = torch.ones(weight.shape[1] * layer.groups, device=weight.device)
    base = _quantized_error(layer, xs, refs, weight, ones)
    act = torch.stack([input_channel_range(layer, x) for x in xs]).amax(0)
    w_range = weight_channel_range(layer, weight)
    # Channels that are always zero, or unused by the weights, keep scale 1.
    live = (act > 0) & (w_range > 0)
    best = (base, None, ones)
    for alpha in alphas:
        s = torch.ones_like(act)
        s[live] = act[live].pow(alpha) / w_range[live].pow(1 - alpha)
        # Scale-free: a common factor moves between the two quantizer ranges.
        s = s / s[live].log().mean().exp() if bool(live.any()) else s
        s = s.clamp(1e-4, 1e4)
        err = _quantized_error(layer, xs, refs, weight, s)
        if err < best[0]:
            best = (err, alpha, s)
    err, alpha, s = best
    if alpha is None or err > (1 - min_gain) * base:
        return dict(alpha=None, err_before=base, err_after=base)
    module = layer.module
    module.weight.data.copy_(scale_weight(layer, weight, s).to(module.weight.dtype))
    inv = (1.0 / s).reshape(channel_shape(layer))
    layer.input_quantizer.pre_quant_scale = inv
    layer.input_quantizer.amax = torch.stack([(x * inv).abs().amax() for x in xs]).amax()
    layer.weight_quantizer.amax = weight_amax(module.weight.float())
    return dict(alpha=alpha, err_before=base, err_after=err)


@torch.no_grad()
def awq_clip_layer(
    layer: QuantLayer, xs: Sequence[Tensor], ratios: Sequence[float] = CLIP_RATIOS
) -> dict:
    """Search and apply the per-output-channel weight clip ratio of one layer.

    Args:
        layer: The layer; ``weight_quantizer.amax`` is updated in place.
        xs: Float inputs of the layer (before any input scale).
        ratios: Fractions of ``max |w|`` to try (must include 1).

    Returns:
        The mean chosen ratio and the errors without / with clipping.
    """
    weight = layer.module.weight.detach().float()
    scale = pre_scale(layer)
    if scale is not None:
        xs = [x * scale.float() for x in xs]
    act_amax = layer.input_quantizer.amax.detach().float().reshape(())
    xqs = [fake_quant(x, act_amax) for x in xs]
    refs = [layer_forward(layer, x, weight) for x in xs]
    base_amax = weight_amax(weight)
    out_dims = (0, 2, 3) if layer.is_conv else None
    errors = []
    for ratio in ratios:
        wq = fake_quant(weight.clamp(-base_amax * ratio, base_amax * ratio), base_amax * ratio)
        err = torch.zeros(weight.shape[0], device=weight.device)
        for xq, ref in zip(xqs, refs, strict=True):
            diff = (layer_forward(layer, xq, wq) - ref).square()
            err += diff.sum(out_dims) if out_dims else diff.flatten(0, -2).sum(0)
        errors.append(err)
    errors = torch.stack(errors)  # (ratios, out channels)
    best = errors.argmin(0)
    chosen = torch.tensor(ratios, device=weight.device)[best]
    layer.weight_quantizer.amax = base_amax * chosen.reshape(base_amax.shape)
    first = ratios.index(1.0)
    return dict(
        mean_ratio=float(chosen.mean()),
        err_before=float(errors[first].sum()),
        err_after=float(errors.gather(0, best[None]).sum()),
    )


def _layer_inputs(model: nn.Module, layer: QuantLayer, batches: Sequence[Tensor]) -> list[Tensor]:
    """Float inputs of ``layer``, before its input scale (model run without quantization)."""
    with quantizers_disabled(model):
        return capture(model, layer.module, batches)


@torch.no_grad()
def awq_scale(model: nn.Module, batches: Sequence[Tensor], min_gain: float = 0.1) -> list[dict]:
    """Apply :func:`awq_scale_layer` to every quantized layer of ``model``.

    Call the activation calibration again afterwards (the ranges set here come
    from ``batches`` only).
    """
    report = []
    for layer in quant_layers(model):
        if pre_scale(layer) is not None:
            continue  # already scaled
        xs = _layer_inputs(model, layer, batches)
        if not xs:
            continue
        result = awq_scale_layer(layer, xs, min_gain=min_gain)
        report.append(dict(layer=layer.name, **result))
    scaled = [r for r in report if r['alpha'] is not None]
    logger.info('AWQ scale: %d of %d layers scaled', len(scaled), len(report))
    for r in sorted(scaled, key=lambda r: r['err_after'] / max(r['err_before'], 1e-30))[:10]:
        logger.info(
            '  %s: alpha %.1f, layer error x%.3f',
            r['layer'], r['alpha'], r['err_after'] / max(r['err_before'], 1e-30),
        )  # fmt: skip
    return report


@torch.no_grad()
def awq_clip(model: nn.Module, batches: Sequence[Tensor]) -> list[dict]:
    """Apply :func:`awq_clip_layer` to every quantized layer of ``model``."""
    report = []
    for layer in quant_layers(model):
        xs = _layer_inputs(model, layer, batches)
        if not xs:
            continue
        report.append(dict(layer=layer.name, **awq_clip_layer(layer, xs)))
    if report:
        before = sum(r['err_before'] for r in report)
        after = sum(r['err_after'] for r in report)
        logger.info(
            'AWQ clip: %d layers, summed layer error x%.3f', len(report), after / max(before, 1e-30)
        )
    return report
