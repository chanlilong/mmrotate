# Copyright (c) OpenMMLab. All rights reserved.
"""Shared pieces of the INT8 refinement algorithms.

All algorithms work on a model that ModelOpt has already quantized and
calibrated (``mmrotate.tensorrt.precision.quantize``): every ``Conv2d`` /
``Linear`` is a ModelOpt quant module with an ``input_quantizer`` (per-tensor
activation range ``amax``) and a ``weight_quantizer`` (per-output-channel
``amax``). The fake quantization below is the one the exported Q/DQ nodes
perform: symmetric, ``scale = amax / 127``, integers in ``[-128, 127]``.
"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

QMAX = 127.0


@dataclass
class QuantLayer:
    """A quantized conv / linear layer and its two quantizers."""

    name: str
    module: nn.Module

    @property
    def input_quantizer(self) -> nn.Module:
        return self.module.input_quantizer

    @property
    def weight_quantizer(self) -> nn.Module:
        return self.module.weight_quantizer

    @property
    def is_conv(self) -> bool:
        return isinstance(self.module, nn.Conv2d)

    @property
    def groups(self) -> int:
        return self.module.groups if self.is_conv else 1


def quant_layers(model: nn.Module) -> list[QuantLayer]:
    """The ``Conv2d`` / ``Linear`` layers whose input and weights are quantized to 8 bits."""
    layers = []
    for name, module in model.named_modules():
        iq = getattr(module, 'input_quantizer', None)
        wq = getattr(module, 'weight_quantizer', None)
        if (
            isinstance(module, nn.Conv2d | nn.Linear)
            and getattr(iq, 'is_enabled', False)
            and getattr(wq, 'is_enabled', False)
            and iq.num_bits == 8
            and wq.num_bits == 8
        ):
            layers.append(QuantLayer(name, module))
    return layers


def fake_quant(x: Tensor, amax: Tensor) -> Tensor:
    """Symmetric 8-bit fake quantization; ``amax`` broadcasts against ``x``."""
    scale = (amax / QMAX).clamp_min(1e-12)
    return torch.clamp(torch.round(x / scale), -QMAX - 1, QMAX) * scale


def weight_amax(weight: Tensor) -> Tensor:
    """Per-output-channel ``max |w|``, shaped to broadcast against ``weight``."""
    amax = weight.detach().abs().flatten(1).amax(1)
    return amax.reshape(-1, *([1] * (weight.dim() - 1)))


def layer_forward(layer: QuantLayer, x: Tensor, weight: Tensor) -> Tensor:
    """The layer's float op on ``x`` with ``weight`` (no quantizers)."""
    m = layer.module
    bias = None if m.bias is None else m.bias.float()
    if layer.is_conv:
        if m.padding_mode != 'zeros':
            raise NotImplementedError(f'{layer.name}: padding_mode {m.padding_mode}')
        return F.conv2d(x, weight, bias, m.stride, m.padding, m.dilation, m.groups)
    return F.linear(x, weight, bias)


def channel_shape(layer: QuantLayer) -> tuple[int, ...]:
    """Shape that broadcasts a per-input-channel vector against the layer input."""
    return (1, -1, 1, 1) if layer.is_conv else (-1,)


def pre_scale(layer: QuantLayer) -> Tensor | None:
    """The per-channel multiplier applied to the input before quantization."""
    return getattr(layer.input_quantizer, 'pre_quant_scale', None)


def clear_pre_scale(layer: QuantLayer) -> None:
    """Remove the per-channel input scale of ``layer`` (ModelOpt has no setter for that)."""
    if hasattr(layer.input_quantizer, '_pre_quant_scale'):
        del layer.input_quantizer._pre_quant_scale


class _StopForward(Exception):  # noqa: N818 - control flow, not an error
    """Ends a forward pass once the wanted tensor was captured."""


@torch.no_grad()
def capture(
    model: nn.Module,
    target: nn.Module,
    batches: Sequence[Tensor],
    output: bool = False,
) -> list[Tensor]:
    """The first input (or output) of ``target`` in each forward pass over ``batches``.

    The forward pass is cut short once the tensor is captured. Returns fp32
    tensors on the model's device; an empty list if ``target`` is never called
    (e.g. a layer whose weight is used functionally).
    """
    device = next(model.parameters()).device
    captured: list[Tensor] = []

    def pre_hook(module: nn.Module, args: tuple) -> None:
        captured.append(args[0].detach().float())
        raise _StopForward

    def post_hook(module: nn.Module, args: tuple, out: Tensor) -> None:
        captured.append(out.detach().float())
        raise _StopForward

    handle = (
        target.register_forward_hook(post_hook)
        if output
        else target.register_forward_pre_hook(pre_hook)
    )
    try:
        for x in batches:
            try:
                model._forward(x.to(device))
            except _StopForward:
                pass
    finally:
        handle.remove()
    return captured


def all_quantizers(model: nn.Module) -> list[nn.Module]:
    """Every enabled ModelOpt quantizer of ``model``."""
    from modelopt.torch.quantization.nn import TensorQuantizer

    return [m for m in model.modules() if isinstance(m, TensorQuantizer) and m.is_enabled]


@contextmanager
def quantizers_disabled(model: nn.Module, keep: Sequence[nn.Module] = ()) -> Iterator[None]:
    """Run ``model`` in float inside the context, except the quantizers in ``keep``.

    Per-channel input scales stay active, so the float model is unchanged by
    the AWQ scaling.
    """
    keep_ids = {id(q) for q in keep}
    off = [q for q in all_quantizers(model) if id(q) not in keep_ids]
    for q in off:
        q.disable()
    try:
        yield
    finally:
        for q in off:
            q.enable()
