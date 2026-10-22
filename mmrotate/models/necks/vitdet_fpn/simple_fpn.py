# Copyright (c) OpenMMLab. All rights reserved.
"""Simple feature pyramid of ViTDet (Li et al., ECCV 2022).

Builds a multi-scale pyramid from the single stride-16 map of a plain ViT,
following ``SimpleFeaturePyramid`` in detectron2 and the ``fpn1..fpn4`` stages
of https://github.com/ViTAE-Transformer/ViTDet: deconvolutions to upsample,
identity at the input stride and max pooling to downsample, each followed by a
1x1 and a 3x3 conv with LayerNorm. There is no top-down fusion.
"""

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmengine.model import BaseModule
from torch import Tensor, nn

from mmrotate.registry import MODELS


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dim of an NCHW tensor."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, x.shape[-1:], self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2).contiguous()


def build_norm(norm_cfg: dict, num_channels: int) -> nn.Module:
    """``type='LN2d'`` gives :class:`LayerNorm2d`, anything else goes to mmcv."""
    cfg = dict(norm_cfg)
    if cfg.pop('type') == 'LN2d':
        return LayerNorm2d(num_channels, **cfg)
    return build_norm_layer(norm_cfg, num_channels)[1]


@MODELS.register_module()
class ViTDetSimpleFPN(BaseModule):
    """ViTDet simple feature pyramid on the last map of a plain ViT.

    For each output stride ``s`` the input (at ``in_stride``) is resampled:
    ``log2(in_stride / s)`` stride-2 deconvs (halving the channels, with
    norm + GELU in between) when ``s < in_stride``, identity when equal and
    max pooling when ``s > in_stride``. Each level then gets a 1x1 and a 3x3
    conv, both followed by the norm, to ``out_channels``.

    Args:
        in_channels: Channels of the ViT feature map.
        out_channels: Channels of every output level.
        in_stride: Stride of the ViT feature map (its patch size).
        out_strides: Output strides, in increasing order.
        norm_cfg: Norm after the convs; ``LN2d`` as in ViTDet.
        init_cfg: Standard mmengine init config.
    """

    def __init__(
        self,
        in_channels: int = 1280,
        out_channels: int = 256,
        in_stride: int = 16,
        out_strides: Sequence[int] = (8, 16, 32),
        norm_cfg: dict | None = None,
        init_cfg: dict | list[dict] | None = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        norm_cfg = norm_cfg or dict(type='LN2d')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_stride = in_stride
        self.out_strides = tuple(out_strides)
        if list(self.out_strides) != sorted(set(self.out_strides)):
            raise ValueError('out_strides must be strictly increasing')

        self.stages = nn.ModuleList()
        for stride in self.out_strides:
            scale, dim = self._build_scale(in_channels, in_stride, stride, norm_cfg)
            self.stages.append(
                nn.Sequential(
                    scale,
                    nn.Conv2d(dim, out_channels, 1, bias=False),
                    build_norm(norm_cfg, out_channels),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                    build_norm(norm_cfg, out_channels),
                )
            )

    @staticmethod
    def _build_scale(
        dim: int, in_stride: int, stride: int, norm_cfg: dict
    ) -> tuple[nn.Module, int]:
        """Resampling from ``in_stride`` to ``stride`` and its output dim."""
        if stride == in_stride:
            return nn.Identity(), dim
        if stride > in_stride:
            if stride % in_stride:
                raise ValueError(f'stride {stride} is not a multiple of {in_stride}')
            k = stride // in_stride
            return nn.MaxPool2d(kernel_size=k, stride=k, ceil_mode=True), dim
        num_up = math.log2(in_stride / stride)
        if not num_up.is_integer():
            raise ValueError(f'in_stride / stride must be a power of 2, got {in_stride}/{stride}')
        layers: list[nn.Module] = []
        for i in range(int(num_up)):
            if i > 0:
                layers += [build_norm(norm_cfg, dim), nn.GELU()]
            layers.append(nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2))
            dim //= 2
        return nn.Sequential(*layers), dim

    def init_weights(self) -> None:
        super().init_weights()
        if self.init_cfg is not None:
            return
        for m in self.modules():
            if isinstance(m, nn.Conv2d | nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, inputs: Sequence[Tensor]) -> tuple[Tensor, ...]:
        x = inputs[-1]
        return tuple(stage(x) for stage in self.stages)
