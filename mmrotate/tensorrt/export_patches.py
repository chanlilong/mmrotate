# Copyright (c) OpenMMLab. All rights reserved.
"""Swap modules that cannot be exported to ONNX / TensorRT for equivalents.

Applied (and undone) around calibration and export, on the module instances
only, so no model file changes and the weights stay the same:

- ViT-CoMer ``MSDeformAttn``: the Triton kernels become the ``grid_sample``
  PyTorch path (``backend='pytorch'``, ``fused=False``).
- DCFL DCNv2 layers: torchvision / mmcv ``deform_conv2d`` becomes
  :func:`modulated_deform_conv2d_gridsample`.
- ViTDet ``LayerNorm2d``: ``F.layer_norm`` with the channel count as a
  constant ``normalized_shape`` (``x.shape[-1:]`` traces as a tensor, which
  the ONNX exporter rejects).
- Optionally, modules of the given types run in fp32 with autocast off
  (e.g. ``ConvTranspose2d``, which TensorRT has no bf16 kernel for).
- ``F.scaled_dot_product_attention`` becomes ``softmax((q * scale) @ k^T) @ v``
  with the logits and softmax in fp32 (default on). Torch's own ONNX
  decomposition runs them in the input dtype, so under fp16 autocast the
  logits are fp16, while torch's fused kernels accumulate in fp32. DINOv3 ViTs
  have q / k activations of ~2000: their logits overflow fp16 (NaN engines),
  and even at fp32 TensorRT reproduced the default decomposition poorly
  (output errors of ~25 at batch 8 on ViT-S CoMer, vs 4e-3 with this form).
"""

import types
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .deform_conv import modulated_deform_conv2d_gridsample


def _is_msda(module: nn.Module) -> bool:
    from mmrotate.models.backbones.cradiov4.ms_deform_attn import MSDeformAttn

    return isinstance(module, MSDeformAttn)


def _is_dcfl_dcn(module: nn.Module) -> bool:
    try:
        from mmrotate.models.dense_heads.dcfl.layers import _ModulatedDeformConvBase
    except ImportError:  # pragma: no cover - torchvision without deform_conv2d
        return False
    return isinstance(module, _ModulatedDeformConvBase)


def _is_layer_norm_2d(module: nn.Module) -> bool:
    from mmrotate.models.necks.vitdet_fpn.simple_fpn import LayerNorm2d

    return isinstance(module, LayerNorm2d)


def _static_layer_norm_2d(self: nn.Module, x: Tensor) -> Tensor:
    """``LayerNorm2d.forward`` with a constant ``normalized_shape``."""
    x = F.layer_norm(
        x.permute(0, 2, 3, 1), (self.weight.shape[0],), self.weight, self.bias, self.eps
    )
    return x.permute(0, 3, 1, 2)


def _gridsample_deform_conv(self: nn.Module, x: Tensor, offset: Tensor, mask: Tensor) -> Tensor:
    """Drop-in for ``_ModulatedDeformConvBase.deform_conv``."""
    return modulated_deform_conv2d_gridsample(
        x,
        offset,
        mask,
        self.weight,
        self.bias,
        self.stride,
        self.padding,
        self.dilation,
        self.groups,
        self.deform_groups,
    )


def _sdpa_fp32_logits(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> Tensor:
    """``F.scaled_dot_product_attention`` (inference) with fp32 logits / softmax."""
    if is_causal or dropout_p or enable_gqa:
        raise NotImplementedError('fp32-logit attention covers plain inference attention only')
    scale = query.shape[-1] ** -0.5 if scale is None else scale
    with torch.autocast(query.device.type, enabled=False):
        logits = (query.float() * scale) @ key.float().transpose(-2, -1)
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                logits = logits.masked_fill(~attn_mask, float('-inf'))
            else:
                logits = logits + attn_mask.float()
        attn = logits.softmax(dim=-1)
    return attn.to(value.dtype) @ value


def _fp32_forward(forward: Callable[[Tensor], Tensor]) -> Callable[[Tensor], Tensor]:
    """``forward`` in fp32 with autocast off; the output stays fp32."""

    def wrapped(x: Tensor) -> Tensor:
        with torch.autocast(x.device.type, enabled=False):
            return forward(x.float())

    return wrapped


@contextmanager
def export_mode(
    model: nn.Module,
    fp32_types: Sequence[type[nn.Module]] = (),
    fp32_attention: bool = True,
) -> Iterator[list[str]]:
    """Make ``model`` exportable in place; yields the names of patched modules.

    Args:
        model: The detector.
        fp32_types: Module types to run in fp32 under autocast.
        fp32_attention: Replace ``F.scaled_dot_product_attention`` with the
            explicit fp32-logit form (patched globally while the context is
            active).
    """
    undo: list[Callable[[], None]] = []
    patched: list[str] = []
    if fp32_attention:
        sdpa = F.scaled_dot_product_attention
        F.scaled_dot_product_attention = _sdpa_fp32_logits

        def _restore_sdpa() -> None:
            F.scaled_dot_product_attention = sdpa

        undo.append(_restore_sdpa)
    for name, module in model.named_modules():
        if fp32_types and isinstance(module, tuple(fp32_types)):
            module.forward = _fp32_forward(module.forward)

            def _restore_fp32(m: nn.Module = module) -> None:
                del m.forward

            undo.append(_restore_fp32)
            patched.append(name)
        elif _is_msda(module):
            old = (module.backend, module.fused)
            module.backend, module.fused = 'pytorch', False

            def _restore_msda(m: nn.Module = module, old: tuple = old) -> None:
                m.backend, m.fused = old

            undo.append(_restore_msda)
            patched.append(name)
        elif _is_dcfl_dcn(module):
            module.deform_conv = types.MethodType(_gridsample_deform_conv, module)

            def _restore_dcn(m: nn.Module = module) -> None:
                del m.deform_conv

            undo.append(_restore_dcn)
            patched.append(name)
        elif _is_layer_norm_2d(module):
            module.forward = types.MethodType(_static_layer_norm_2d, module)

            def _restore_ln(m: nn.Module = module) -> None:
                del m.forward

            undo.append(_restore_ln)
            patched.append(name)
    try:
        yield patched
    finally:
        for fn in reversed(undo):
            fn()
