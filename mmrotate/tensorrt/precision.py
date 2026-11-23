# Copyright (c) OpenMMLab. All rights reserved.
"""Precision of the exported graph: autocast for fp16/bf16, ModelOpt PTQ for int8/fp8.

TensorRT builds strongly typed networks, so the precision is fixed in the
ONNX graph:

- ``fp16`` / ``bf16``: the graph is traced under ``torch.autocast``, i.e. the
  same mixed precision as torch AMP inference (convs / matmuls in low
  precision, normalisation and softmax in fp32).
- ``int8`` / ``fp8``: ModelOpt post-training quantization of the torch model
  (Q/DQ on conv / linear weights and inputs) calibrated on dataset images,
  then traced under fp16 autocast for the remaining layers. INT8 can be refined
  with the algorithms of :mod:`mmrotate.tensorrt.quant` (AWQ scale / clip
  search, GPTQ rounding, automatic exclusion of sensitive layers). FP8 Q/DQ use
  ModelOpt's ONNX export; INT8 Q/DQ use :class:`_Int8QDQ`, as ModelOpt's INT8
  fake-quant op crashes the TorchScript exporter of torch 2.14.
"""

import copy
import logging
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager

import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)

PRECISIONS = ('fp32', 'fp16', 'bf16', 'int8', 'fp8')
CALIB_METHODS = ('max', 'mse', 'entropy', 'percentile')
# Kept in high precision by default: the final prediction convs of the RTMDet
# heads, and the deformable-attention sampling offsets / attention logits
# (ViT-CoMer), which become sub-pixel positions and softmax inputs. With these
# quantized to fp8, TensorRT also returned NaNs on a CoMer test model.
DEFAULT_QUANT_EXCLUDE = ('*bbox_head.rtm_*', '*.sampling_offsets*', '*.attention_weights*')
# Also kept in high precision when attention is exported in fused form: the
# projection that takes the attention output (timm / RADIO / LingBot ViTs). A
# Q node directly on a fused attention makes TensorRT build an int8 attention,
# which has no kernel beyond 512 tokens (the engine fails to build).
FUSED_ATTENTION_EXCLUDE = ('*.attn.proj',)


def autocast_dtype(precision: str) -> torch.dtype | None:
    """Low-precision dtype the graph is traced with (``None``: fp32)."""
    return {'fp32': None, 'bf16': torch.bfloat16}.get(precision, torch.float16)


def fp32_module_types(precision: str) -> tuple[type[nn.Module], ...]:
    """Module types kept in fp32 because TensorRT lacks low-precision kernels."""
    return (nn.ConvTranspose2d,) if precision == 'bf16' else ()


def _quant_cfg(precision: str, exclude: Sequence[str]) -> dict:
    import modelopt.torch.quantization as mtq

    base = {'int8': mtq.INT8_DEFAULT_CFG, 'fp8': mtq.FP8_DEFAULT_CFG}[precision]
    cfg = copy.deepcopy(base)
    for pattern in exclude:
        pattern = pattern if pattern.endswith('*') else pattern + '*'
        cfg['quant_cfg'].append({'quantizer_name': pattern, 'enable': False})
    return cfg


def _input_quantizers(model: nn.Module) -> Iterable[tuple[str, nn.Module]]:
    from modelopt.torch.quantization.nn import TensorQuantizer

    for name, module in model.named_modules():
        if (
            isinstance(module, TensorQuantizer)
            and name.endswith('input_quantizer')
            and module.is_enabled
        ):
            yield name, module


@torch.no_grad()
def _histogram_recalibrate(
    model: nn.Module,
    forward_loop: Callable[[nn.Module], None],
    method: str,
    percentile: float,
) -> None:
    """Re-calibrate the activation quantizers' amax from histograms."""
    from modelopt.torch.quantization.calib import HistogramCalibrator

    quantizers = list(_input_quantizers(model))
    for _, q in quantizers:
        q._calibrator = HistogramCalibrator(num_bits=q.num_bits, axis=q.axis, unsigned=q.unsigned)
        q.disable_quant()
        q.enable_calib()
    forward_loop(model)
    for name, q in quantizers:
        kwargs = dict(percentile=percentile) if method == 'percentile' else {}
        try:
            q.load_calib_amax(method, **kwargs)
        except RuntimeError:  # never reached by the calibration data
            logger.warning('No calibration data reached %s, keeping its max amax', name)
        q.enable_quant()
        q.disable_calib()


@torch.no_grad()
def _recalibrate_inputs(model: nn.Module, forward_loop: Callable[[nn.Module], None]) -> None:
    """Max-calibrate the activation quantizers again (after their inputs were rescaled)."""
    from modelopt.torch.quantization.calib import MaxCalibrator

    quantizers = list(_input_quantizers(model))
    for _, q in quantizers:
        q._calibrator = MaxCalibrator(num_bits=q.num_bits, axis=q.axis, unsigned=q.unsigned)
        q.disable_quant()
        q.enable_calib()
    forward_loop(model)
    for name, q in quantizers:
        try:
            q.load_calib_amax()
        except RuntimeError:  # never reached by the calibration data
            logger.warning('No calibration data reached %s, keeping its amax', name)
        q.enable_quant()
        q.disable_calib()


# Largest quantized magnitude (scale = amax / bound) per precision.
_QUANT_BOUND = {'int8': 127.0, 'fp8': 448.0}
_FP16_MIN_NORMAL = 2.0**-14


def _fix_degenerate_amax(model: nn.Module, precision: str) -> None:
    """Clamp amax so the fp16 Q/DQ scale (``amax / bound``) is a normal fp16.

    Quantizers that only saw zeros (amax 0 / NaN) or tiny values would get a
    zero scale in the fp16 graph, which TensorRT rejects. The clamp never
    clips: the true values are below the new amax.
    """
    from modelopt.torch.quantization.nn import TensorQuantizer

    min_amax = _QUANT_BOUND[precision] * _FP16_MIN_NORMAL
    for name, module in model.named_modules():
        if not isinstance(module, TensorQuantizer) or not module.is_enabled:
            continue
        amax = module.amax
        if amax is None:
            continue
        bad = ~torch.isfinite(amax) | (amax < min_amax)
        if bad.any():
            logger.info('%s: %d amax value(s) raised to %.3g', name, int(bad.sum()), min_amax)
            module.amax = torch.where(bad, torch.full_like(amax, min_amax), amax)


def _fused_conv(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """A new conv equal to ``bn(conv(x))`` (eval mode)."""
    fused = copy.deepcopy(conv)
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    fused.weight.data = conv.weight.data * scale.reshape(-1, 1, 1, 1)
    bias = conv.bias.data if conv.bias is not None else torch.zeros_like(bn.running_mean)
    fused.bias = nn.Parameter((bias - bn.running_mean) * scale + bn.bias)
    return fused


@torch.no_grad()
def fold_conv_bn(model: nn.Module) -> int:
    """Fold eval-mode BatchNorm2d into the preceding conv, in place.

    Weights must be quantized after folding, as TensorRT runs them: quantizing
    the conv and scaling by ``gamma / sigma`` afterwards multiplies the
    quantization error of channels with a small running variance. Handles
    mmcv ``ConvModule`` (conv before norm) and Conv2d -> BatchNorm2d pairs in
    ``nn.Sequential``. Every fused conv is a new module, so convs shared by
    several ConvModules with separate BNs (RTMDet ``share_conv``) are un-shared
    correctly.

    Returns:
        The number of folded BatchNorms.
    """
    num = 0
    for module in list(model.modules()):
        norm_name = getattr(module, 'norm_name', None)
        conv = getattr(module, 'conv', None)
        order = getattr(module, 'order', ('conv', 'norm', 'act'))
        if (
            norm_name
            and isinstance(conv, nn.Conv2d)
            and isinstance(getattr(module, norm_name, None), nn.BatchNorm2d)
            and order.index('conv') < order.index('norm')
        ):
            module.conv = _fused_conv(conv, getattr(module, norm_name))
            setattr(module, norm_name, nn.Identity())
            num += 1
        elif isinstance(module, nn.Sequential):
            names = list(module._modules)
            for a, b in zip(names[:-1], names[1:], strict=True):
                conv, bn = module._modules[a], module._modules[b]
                if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                    module._modules[a] = _fused_conv(conv, bn)
                    module._modules[b] = nn.Identity()
                    num += 1
    return num


def quantize(
    model: nn.Module,
    precision: str,
    batches: Sequence[Tensor],
    method: str = 'max',
    exclude: Sequence[str] = DEFAULT_QUANT_EXCLUDE,
    percentile: float = 99.99,
    algos: Sequence[str] = (),
    exclude_budget: float | None = None,
    refine_batches: int = 2,
    report: dict | None = None,
) -> nn.Module:
    """Insert ModelOpt quantizers into ``model`` and calibrate them in place.

    BatchNorms are folded into their convs first (:func:`fold_conv_bn`). With
    ``algos`` / ``exclude_budget`` the calibrated INT8 model is then refined by
    :func:`mmrotate.tensorrt.quant.refine`.

    Args:
        model: Detector in eval mode (inside ``export_mode``).
        precision: ``'int8'`` or ``'fp8'``.
        batches: Preprocessed input batches, run through ``model._forward``.
        method: ``'max'`` / ``'mse'`` (ModelOpt algorithms) or ``'entropy'`` /
            ``'percentile'`` (histograms of the activations, int8 only).
        exclude: Module name patterns (fnmatch) that are not quantized.
        percentile: Percentile for ``method='percentile'``.
        algos: INT8 refinement algorithms, a subset of
            :data:`mmrotate.tensorrt.quant.QUANT_ALGOS`.
        exclude_budget: Output noise budget of the automatic exclusion of
            sensitive layers (INT8; ``None``: off).
        refine_batches: Calibration batches the refinement works on.
        report: If given, filled with what the refinement did.
    """
    import modelopt.torch.quantization as mtq

    refining = bool(algos) or exclude_budget is not None
    if refining and precision != 'int8':
        raise ValueError('AWQ / GPTQ / automatic exclusion are implemented for int8 only')

    if precision not in ('int8', 'fp8'):
        raise ValueError(f'Cannot quantize to {precision}')
    if method not in CALIB_METHODS:
        raise ValueError(f'Unknown calibration method {method}')
    if method in ('entropy', 'percentile') and precision == 'fp8':
        raise ValueError(f'{method} calibration is for int8; use max or mse for fp8')
    device = next(model.parameters()).device

    def forward_loop(m: nn.Module) -> None:
        for x in batches:
            m._forward(x.to(device))

    logger.info('Folded %d BatchNorms into convs', fold_conv_bn(model))
    cfg = _quant_cfg(precision, exclude)
    if method == 'mse':
        cfg['algorithm'] = 'mse'
    model = mtq.quantize(model, cfg, forward_loop)

    def recalibrate(m: nn.Module) -> None:
        if method in ('entropy', 'percentile'):
            _histogram_recalibrate(m, forward_loop, method, percentile)
        else:
            _recalibrate_inputs(m, forward_loop)

    if method in ('entropy', 'percentile'):
        _histogram_recalibrate(model, forward_loop, method, percentile)
    if refining:
        from .quant import refine

        result = refine(model, batches[:refine_batches], algos, exclude_budget, recalibrate)
        if report is not None:
            report.update(result)
    _fix_degenerate_amax(model, precision)
    num = sum(1 for _ in _input_quantizers(model))
    logger.info('Quantized %d layer inputs to %s (%s calibration)', num, precision, method)
    return model


class _Int8QDQ(torch.autograd.Function):
    """INT8 fake quantization exported as ONNX ``QuantizeLinear`` + ``DequantizeLinear``."""

    @staticmethod
    def forward(ctx: object, x: Tensor, scale: Tensor, axis: int | None) -> Tensor:
        if axis is not None:
            shape = [1] * x.dim()
            shape[axis] = -1
            scale = scale.reshape(shape)
        return torch.clamp(torch.round(x / scale), -128, 127) * scale

    @staticmethod
    def symbolic(g: object, x: object, scale: object, axis: int | None) -> object:
        zero_point = g.op('Constant', value_t=torch.zeros((), dtype=torch.int8))
        if axis is not None:
            channels = scale.type().sizes()[0]
            zero_point = g.op('Constant', value_t=torch.zeros(channels, dtype=torch.int8))
        kwargs = {} if axis is None else {'axis_i': axis}
        q = g.op('QuantizeLinear', x, scale, zero_point, **kwargs)
        return g.op('DequantizeLinear', q, scale, zero_point, **kwargs)


def _int8_qdq_forward(quantizer: nn.Module) -> Callable[[Tensor], Tensor]:
    amax = quantizer.amax.detach().float()
    axis = None
    if amax.numel() > 1:
        axis = next(i for i, n in enumerate(amax.shape) if n == amax.numel())
    scale = (amax.flatten() if axis is not None else amax.reshape(())) / 127.0
    scale = scale.masked_fill(scale == 0, 1.0)

    pre_scale = getattr(quantizer, 'pre_quant_scale', None)

    def forward(x: Tensor) -> Tensor:
        # Q/DQ in the dtype the consumer runs in, so TensorRT fuses them.
        if torch.is_autocast_enabled(x.device.type):
            x = x.to(torch.get_autocast_dtype(x.device.type))
        if pre_scale is not None:  # AWQ per-channel input scale
            x = x * pre_scale.to(x.device, x.dtype)
        return _Int8QDQ.apply(x, scale.to(x.device, x.dtype), axis)

    return forward


def _fp32_layer_forward(forward: Callable[[Tensor], Tensor]) -> Callable[[Tensor], Tensor]:
    def wrapped(x: Tensor) -> Tensor:
        with torch.autocast(x.device.type, enabled=False):
            return forward(x.float())

    return wrapped


@contextmanager
def int8_export_mode(model: nn.Module, fp32_layers: bool = False) -> Iterator[None]:
    """Export the enabled INT8 quantizers of ``model`` as ONNX Q/DQ pairs.

    Args:
        model: Quantized model.
        fp32_layers: Type the quantized layers and their Q/DQ nodes fp32
            instead of the autocast dtype. They run as int8 kernels either
            way; the fp32 typing puts a cast between an unquantized fp16
            layer and the Q node of the next layer, where TensorRT otherwise
            fuses the two into an fp16-input / int8-output kernel that is
            missing or wrong for some layers. Needed for partially quantized
            models (automatic exclusion); typing only the convolutions fp32
            is not enough (tested on ConvNeXt-T).
    """
    from modelopt.torch.quantization.nn import TensorQuantizer

    patched = []
    if fp32_layers:
        for module in model.modules():
            quantizer = getattr(module, 'input_quantizer', None)
            if (
                isinstance(quantizer, TensorQuantizer)
                and quantizer.is_enabled
                and quantizer.num_bits == 8
            ):
                module.forward = _fp32_layer_forward(module.forward)
                patched.append(module)
    for module in model.modules():
        if isinstance(module, TensorQuantizer) and module.is_enabled and module.num_bits == 8:
            module.forward = _int8_qdq_forward(module)
            patched.append(module)
    try:
        yield
    finally:
        for module in patched:
            del module.forward
