# Copyright (c) OpenMMLab. All rights reserved.
"""GPTQ weight rounding for W8A8 conv / linear layers.

GPTQ (Frantar et al., 2022) quantizes the weights of a layer one input column
at a time and, after each column, updates the not-yet-quantized columns to
compensate the error just made, using the inverse Hessian ``H^-1`` of the
layer's reconstruction loss ``||W X - Q X||^2`` (``H = X X^T`` over the layer
inputs). Layers are processed in forward order and ``X`` is taken from the
quantized model so far (quantized activations, already rounded earlier
layers), so each layer also absorbs part of the error made before it.

Weights are rounded onto the grid of the layer's weight quantizer (per output
channel, ``amax / 127``; set by calibration or the AWQ clip search), so the
exported Q/DQ nodes reproduce them exactly. Grouped / depthwise convolutions
keep round-to-nearest (their per-group Hessians are tiny and rounding to
nearest is already near optimal for them).
"""

import logging
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .common import QuantLayer, capture, fake_quant, quant_layers

logger = logging.getLogger(__name__)


def input_hessian(layer: QuantLayer, xs: Sequence[Tensor]) -> Tensor:
    """``X X^T / n`` (float64) over the columns the weights multiply.

    For a conv the columns are the unfolded input patches, in the order of the
    flattened weight ``(in, kh, kw)``.
    """
    m = layer.module
    hessian = None
    count = 0
    for batch in xs:
        for x in batch.split(1) if layer.is_conv else (batch,):
            if layer.is_conv:
                cols = F.unfold(x, m.kernel_size, m.dilation, m.padding, m.stride)
                cols = cols.permute(1, 0, 2).flatten(1)  # (d, positions)
            else:
                cols = x.flatten(0, -2).t()  # (d, tokens)
            cols = cols.double()
            hessian = cols @ cols.t() if hessian is None else hessian.addmm_(cols, cols.t())
            count += cols.shape[1]
    if hessian is None:
        raise ValueError('No inputs to build the Hessian from')
    return hessian / count


@torch.no_grad()
def gptq_round(
    weight: Tensor,
    hessian: Tensor,
    amax: Tensor,
    block_size: int = 128,
    percdamp: float = 0.01,
) -> Tensor:
    """GPTQ rounding of a 2-d weight.

    Args:
        weight: ``(out, d)`` float weights.
        hessian: ``(d, d)`` input Hessian.
        amax: ``(out,)`` or ``(out, 1)`` quantizer range per output channel.
        block_size: Columns per lazy-update block.
        percdamp: Dampening added to the Hessian diagonal, as a fraction of
            its mean.

    Returns:
        The quantized weights (on the grid ``amax / 127 * [-128, 127]``).
    """
    w = weight.double().clone()
    h = hessian.double().clone()
    amax = amax.double().reshape(-1, 1)
    d = w.shape[1]
    dead = h.diagonal() == 0
    h[dead, dead] = 1.0
    w[:, dead] = 0.0
    h.diagonal().add_(percdamp * h.diagonal().mean())
    # Upper Cholesky factor of H^-1: row i holds the update weights of column i.
    hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)
    quantized = torch.zeros_like(w)
    for start in range(0, d, block_size):
        end = min(start + block_size, d)
        block = w[:, start:end].clone()
        errors = torch.zeros_like(block)
        hinv_block = hinv[start:end, start:end]
        for i in range(end - start):
            column = block[:, i : i + 1]
            q = fake_quant(column, amax)
            quantized[:, start + i : start + i + 1] = q
            err = (column - q) / hinv_block[i, i]
            block[:, i:] -= err @ hinv_block[i : i + 1, i:]
            errors[:, i : i + 1] = err
        w[:, end:] -= errors @ hinv[start:end, end:]
    return quantized.to(weight.dtype)


def reconstruction_error(weight: Tensor, quantized: Tensor, hessian: Tensor) -> float:
    """``tr((W - Q) H (W - Q)^T)``: the layer output error on the Hessian's inputs."""
    diff = (weight - quantized).double()
    return float(((diff @ hessian.double()) * diff).sum())


@torch.no_grad()
def gptq_layer(layer: QuantLayer, xs: Sequence[Tensor], percdamp: float = 0.01) -> dict:
    """Round the weights of one layer with GPTQ, in place.

    Args:
        layer: A linear layer or a convolution with ``groups == 1``.
        xs: Inputs of the layer's op, i.e. after its input quantizer.
        percdamp: See :func:`gptq_round`.

    Returns:
        The reconstruction errors of round-to-nearest and of GPTQ.
    """
    module = layer.module
    weight = module.weight.detach().float()
    flat = weight.flatten(1)
    amax = layer.weight_quantizer.amax.detach().float().reshape(-1, 1)
    hessian = input_hessian(layer, xs)
    nearest = fake_quant(flat.clamp(-amax, amax), amax)
    rounded = gptq_round(flat.clamp(-amax, amax), hessian, amax, percdamp=percdamp)
    err_nearest = reconstruction_error(flat, nearest, hessian)
    err_gptq = reconstruction_error(flat, rounded, hessian)
    if err_gptq < err_nearest:  # dampening can make GPTQ lose on tiny, ill-conditioned layers
        module.weight.data.copy_(rounded.reshape(weight.shape).to(module.weight.dtype))
    else:
        err_gptq = err_nearest
    return dict(err_nearest=err_nearest, err_gptq=err_gptq)


@torch.no_grad()
def gptq(model: nn.Module, batches: Sequence[Tensor], percdamp: float = 0.01) -> list[dict]:
    """GPTQ-round every quantized ``Linear`` / ungrouped ``Conv2d`` of ``model``, in forward order.

    Run this last: it bakes the current weight quantizer ranges into the
    weights.
    """
    report = []
    for layer in quant_layers(model):  # named_modules order follows the forward order closely
        if layer.groups != 1:
            continue
        # Output of the input quantizer = input of the op, from the model quantized so far.
        xs = capture(model, layer.input_quantizer, batches, output=True)
        if not xs:
            continue
        report.append(dict(layer=layer.name, **gptq_layer(layer, xs, percdamp)))
    if report:
        nearest = sum(r['err_nearest'] for r in report)
        best = sum(r['err_gptq'] for r in report)
        logger.info(
            'GPTQ: %d layers, summed reconstruction error x%.3f of round-to-nearest',
            len(report), best / max(nearest, 1e-30),
        )  # fmt: skip
    return report
