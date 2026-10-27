# Copyright (c) OpenMMLab. All rights reserved.
"""DCNv2 layers of the DCFL heads and the dynamic-prior shift.

MMCV-DCFL (https://github.com/Chasel-Tsui/MMCV-DCFL) only adds Python
wrappers to mmcv: ``ModulatedDeformConvG`` is ``ModulatedDeformConv2dPack``
returning ``(out, offset, mask)``. The kernels are the stock
``modulated_deform_conv2d`` of official mmcv, so no custom build is needed;
``torchvision.ops.deform_conv2d`` (same maths and offset layout) is the
fallback when the mmcv op is unavailable or the input is on the CPU.
"""

import math

import torch
from torch import Tensor, nn
from torch.nn.modules.utils import _pair

try:
    from mmcv.ops import modulated_deform_conv2d as _mmcv_modulated_deform_conv2d

    HAS_MMCV_DCN = True
except ImportError:  # pragma: no cover - mmcv without compiled ops
    HAS_MMCV_DCN = False

from torchvision.ops import deform_conv2d as _tv_deform_conv2d

BACKENDS = ('auto', 'mmcv', 'torchvision')


def modulated_deform_conv(
    x: Tensor,
    offset: Tensor,
    mask: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    groups: int = 1,
    deform_groups: int = 1,
    backend: str = 'auto',
) -> Tensor:
    """Modulated deformable convolution (DCNv2) with a selectable backend.

    Offsets are ``(B, deform_groups * 2 * kh * kw, H, W)`` as
    ``[dy0, dx0, dy1, dx1, ...]`` (mmcv and torchvision agree), masks are
    ``(B, deform_groups * kh * kw, H, W)``. Runs in fp32 with autocast off,
    since the mmcv kernel has no bf16 path; the output has the autocast dtype
    (like a regular conv) or else ``x``'s dtype.

    Args:
        backend (str): ``'mmcv'``, ``'torchvision'`` or ``'auto'`` (mmcv for
            CUDA inputs when available, torchvision otherwise).
    """
    assert backend in BACKENDS, f'unknown DCN backend {backend}'
    if backend == 'auto':
        backend = 'mmcv' if HAS_MMCV_DCN and x.is_cuda else 'torchvision'
    device_type = x.device.type
    if torch.is_autocast_enabled(device_type):
        dtype = torch.get_autocast_dtype(device_type)
    else:
        dtype = x.dtype
    with torch.autocast(device_type=device_type, enabled=False):
        x, offset, mask, weight = x.float(), offset.float(), mask.float(), weight.float()
        bias = bias.float() if bias is not None else None
        if backend == 'mmcv':
            assert HAS_MMCV_DCN, 'mmcv.ops.modulated_deform_conv2d is not available'
            out = _mmcv_modulated_deform_conv2d(
                x, offset, mask, weight, bias, stride, padding, dilation, groups, deform_groups
            )
        else:
            out = _tv_deform_conv2d(
                x,
                offset,
                weight,
                bias,
                stride=stride,
                padding=padding,
                dilation=dilation,
                mask=mask,
            )
    return out.to(dtype)


class _ModulatedDeformConvBase(nn.Module):
    """Weights of a DCNv2 layer, named as mmcv's ``ModulatedDeformConv2d``."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 1,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        deform_groups: int = 1,
        bias: bool = False,
        backend: str = 'auto',
    ) -> None:
        super().__init__()
        assert backend in BACKENDS, f'unknown DCN backend {backend}'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deform_groups = deform_groups
        self.backend = backend
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """mmcv's ``ModulatedDeformConv2d`` initialisation."""
        stdv = 1.0 / math.sqrt(self.in_channels * math.prod(self.kernel_size))
        nn.init.uniform_(self.weight, -stdv, stdv)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def deform_conv(self, x: Tensor, offset: Tensor, mask: Tensor) -> Tensor:
        """DCNv2 of ``x`` with the given offsets and masks."""
        return modulated_deform_conv(
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
            self.backend,
        )

    def extra_repr(self) -> str:
        """str: layer settings."""
        return (
            f'{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, '
            f'stride={self.stride}, padding={self.padding}, dilation={self.dilation}, '
            f'groups={self.groups}, deform_groups={self.deform_groups}, '
            f'bias={self.bias is not None}, backend={self.backend!r}'
        )


class DCFLModulatedDeformConv(_ModulatedDeformConvBase):
    """DCNv2 that predicts its own offsets and masks and returns them.

    Equivalent of MMCV-DCFL's ``ModulatedDeformConvG`` (same parameter names,
    so its checkpoints load): ``forward(x) -> (out, offset, mask)``. The
    offsets drive the DCFL dynamic priors and the classification branch's
    :class:`DCFLSharedOffsetDeformConv`.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deform_groups * 3 * math.prod(self.kernel_size),
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True,
        )
        self.init_offset()

    def init_offset(self) -> None:
        """Zero the offset/mask branch (regular sampling, mask 0.5)."""
        nn.init.zeros_(self.conv_offset.weight)
        nn.init.zeros_(self.conv_offset.bias)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return the output, offsets ``[dy0, dx0, ...]`` and masks."""
        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return self.deform_conv(x, offset, mask), offset, mask


class DCFLSharedOffsetDeformConv(_ModulatedDeformConvBase):
    """DCNv2 that samples with offsets and masks given by another layer.

    Same as mmcv's ``ModulatedDeformConv2d``: ``forward(x, offset, mask)``.
    """

    def forward(self, x: Tensor, offset: Tensor, mask: Tensor) -> Tensor:
        """DCNv2 of ``x`` with external offsets and masks."""
        return self.deform_conv(x, offset, mask)


def dynamic_prior_shift(offset: Tensor, stride: tuple[float, float], scale: float) -> Tensor:
    """Shift of each location's priors: the mean DCN offset, in image pixels.

    DCFL moves every prior by the mean of the ``K`` sampling offsets of the
    regression branch's DCN before label assignment. The reference scales the
    feature-pixel offsets by ``img_h / 256 * 2 ** level`` (level 0 = stride 8),
    which is ``stride / 2`` at its 1024 input, i.e. ``scale = 0.5``.

    Args:
        offset (Tensor): ``(B, 2 * K, H, W)`` offsets ``[dy0, dx0, dy1, ...]``.
        stride (tuple[float, float]): Level stride ``(stride_w, stride_h)``.
        scale (float): Multiplier on ``offset * stride``.

    Returns:
        Tensor: ``(B, H * W, 2)`` detached fp32 ``(dx, dy)`` shifts.
    """
    num_imgs, channels, height, width = offset.shape
    mean = offset.detach().float().view(num_imgs, channels // 2, 2, height, width).mean(dim=1)
    shift = mean.flip(1).flatten(2).transpose(1, 2)
    return shift * shift.new_tensor([stride[0], stride[1]]) * scale
