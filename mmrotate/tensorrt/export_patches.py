# Copyright (c) OpenMMLab. All rights reserved.
"""Swap modules that cannot be exported to ONNX / TensorRT for equivalents.

Applied (and undone) around calibration and export, on the module instances
only, so no model file changes and the weights stay the same:

- ViT-CoMer ``MSDeformAttn``: with ``msda='plugin'`` the fused Triton kernel is
  exported as one ``mmrotate::MSDAFused`` node, which the TensorRT plugin of
  :mod:`~mmrotate.tensorrt.plugins.msda` runs with the same kernel; with
  ``msda='gridsample'`` it becomes the ``grid_sample`` PyTorch path
  (``backend='pytorch'``, ``fused=False``; standard ONNX ops only).
- DCFL DCNv2 layers: torchvision / mmcv ``deform_conv2d`` becomes
  :func:`modulated_deform_conv2d_gridsample`.
- ViTDet ``LayerNorm2d``: ``F.layer_norm`` with the channel count as a
  constant ``normalized_shape`` (``x.shape[-1:]`` traces as a tensor, which
  the ONNX exporter rejects).
- Optionally, modules of the given types run in fp32 with autocast off
  (e.g. ``ConvTranspose2d``, which TensorRT has no bf16 kernel for).
- ``F.scaled_dot_product_attention`` becomes one of the explicit forms of
  :mod:`~mmrotate.tensorrt.attention`, chosen per call: forms TensorRT fuses
  into a single attention kernel (``fused``, ``fused-fp32acc``) or the
  unfused fp32-logit form (``legacy``). Torch's own ONNX decomposition runs
  under autocast with an fp32 softmax between low-precision matmuls, which
  TensorRT does not fuse, and its fp16 logits overflow on DINOv3 ViTs (q / k
  activations of ~2000).
"""

import types
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .attention import AttentionExporter, patch_sdpa
from .deform_conv import modulated_deform_conv2d_gridsample

MSDA_EXPORTS = ('plugin', 'gridsample')


def _is_msda(module: nn.Module) -> bool:
    from mmrotate.models.backbones.cradiov4.ms_deform_attn import MSDeformAttn

    return isinstance(module, MSDeformAttn)


def has_msda(model: nn.Module) -> bool:
    """Whether ``model`` contains ViT-CoMer deformable attention."""
    return any(_is_msda(module) for module in model.modules())


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
    attention: str | Sequence[str] | None = 'legacy',
    msda: str = 'gridsample',
) -> Iterator[list[str]]:
    """Make ``model`` exportable in place; yields the names of patched modules.

    Args:
        model: The detector.
        fp32_types: Module types to run in fp32 under autocast.
        attention: Export form of ``F.scaled_dot_product_attention`` (patched
            globally while the context is active): one form of
            :data:`~mmrotate.tensorrt.attention.ATTENTION_FORMS` for every
            call, or one per call in call order (see
            :func:`~mmrotate.tensorrt.attention.resolve_attention`). ``None``
            leaves torch's own decomposition.
        msda: ``'plugin'`` exports ``MSDeformAttn`` as the ``MSDAFused`` custom
            op (needs CUDA and Triton), ``'gridsample'`` as ``grid_sample``.
    """
    if msda not in MSDA_EXPORTS:
        raise ValueError(f'Unknown MSDA export {msda!r}; choose from {MSDA_EXPORTS}')
    undo: list[Callable[[], None]] = []
    patched: list[str] = []
    with ExitStack() as stack:
        if attention is not None:
            stack.enter_context(patch_sdpa(AttentionExporter(attention)))
        has_msda = False
        for name, module in model.named_modules():
            if fp32_types and isinstance(module, tuple(fp32_types)):
                module.forward = _fp32_forward(module.forward)

                def _restore_fp32(m: nn.Module = module) -> None:
                    del m.forward

                undo.append(_restore_fp32)
                patched.append(name)
            elif _is_msda(module):
                has_msda = True
                old = (module.backend, module.fused)
                module.backend, module.fused = (
                    ('triton', True) if msda == 'plugin' else ('pytorch', False)
                )

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
        if has_msda and msda == 'plugin':
            from .plugins.msda import export_msda_as_plugin

            stack.enter_context(export_msda_as_plugin())
        try:
            yield patched
        finally:
            for fn in reversed(undo):
                fn()
