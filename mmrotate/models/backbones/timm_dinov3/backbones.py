# Copyright (c) OpenMMLab. All rights reserved.
"""Frozen timm (DINOv3) backbones for multi-scale detection.

* :class:`TimmViTSimplePyramid` - plain ViT + ViTDet simple feature pyramid.
* :class:`TimmViTCoMer` - plain ViT + ViT-CoMer adapter.
* :class:`TimmFeatureBackbone` - any timm ``features_only`` CNN (ConvNeXt).
"""

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from mmengine.model import BaseModule
from torch import Tensor, nn

from mmrotate.registry import MODELS

from ...necks.vitdet_fpn import ViTDetSimpleFPN
from ..cradiov4.cradiov4 import freeze_module
from ..cradiov4.vit_comer import CRADIOv4CoMer
from .timm_vit import TimmViT, create_timm_model


def _pad_to(x: Tensor, divisor: int) -> tuple[Tensor, int, int]:
    in_h, in_w = x.shape[-2:]
    pad_h, pad_w = (-in_h) % divisor, (-in_w) % divisor
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    return x, in_h, in_w


@MODELS.register_module()
class TimmViTSimplePyramid(BaseModule):
    """Frozen timm ViT + trainable ViTDet simple feature pyramid.

    The last block's tokens (after the final norm) at stride 16 are turned
    into ``out_strides`` maps by :class:`ViTDetSimpleFPN` (deconvs / identity /
    max pooling, no top-down fusion), as in ViTDet.

    Args:
        model_name: timm ViT name.
        checkpoint: Local timm weights (HF snapshot dir or file).
        frozen: Freeze the ViT; it then runs without autograd.
        out_index: Block whose output feeds the pyramid (``-1``: last).
        out_channels: Channels of every pyramid level.
        out_strides: Pyramid strides, e.g. ``(4, 8, 16, 32)`` for P2-P5.
        norm_cfg: Pyramid norm, ``LN2d`` as in ViTDet.
        vit_cfg: Extra kwargs for :class:`TimmViT`.
        init_cfg: Standard mmengine init config.
    """

    def __init__(
        self,
        model_name: str = 'vit_small_patch16_dinov3',
        checkpoint: str | None = None,
        frozen: bool = True,
        out_index: int = -1,
        out_channels: int = 256,
        out_strides: Sequence[int] = (4, 8, 16, 32),
        norm_cfg: dict | None = None,
        vit_cfg: dict | None = None,
        init_cfg: dict | list[dict] | None = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.frozen = frozen
        self.out_index = out_index
        self.out_strides = tuple(out_strides)
        self.vit = TimmViT(model_name, checkpoint, **(vit_cfg or {}))
        self.fpn = ViTDetSimpleFPN(
            in_channels=self.vit.embed_dim,
            out_channels=out_channels,
            in_stride=self.vit.patch_size,
            out_strides=out_strides,
            norm_cfg=norm_cfg or dict(type='LN2d'),
        )
        if frozen:
            freeze_module(self.vit)

    def train(self, mode: bool = True) -> 'TimmViTSimplePyramid':
        super().train(mode)
        if self.frozen:
            self.vit.eval()
        return self

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        x, in_h, in_w = _pad_to(x, max(self.vit.patch_size, *self.out_strides))
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen):
            feat = self.vit.forward_features(x, [self.out_index])[0]
        outs = self.fpn([feat])
        return tuple(
            out[..., : math.ceil(in_h / s), : math.ceil(in_w / s)]
            for out, s in zip(outs, self.out_strides, strict=True)
        )


@MODELS.register_module()
class TimmViTCoMer(CRADIOv4CoMer):
    """ViT-CoMer adapter on a (frozen) timm ViT, e.g. DINOv3 ViT-S/16.

    Same adapter as :class:`CRADIOv4CoMer` (SPM CNN branch, CTI blocks with
    multi-scale deformable attention); only the plain ViT differs. Defaults
    follow ViT-CoMer-S (12 blocks, 4 stages of 3, 6 deformable heads).

    Args:
        model_name: timm ViT name.
        checkpoint: Local timm weights (HF snapshot dir or file).
        vit_cfg: Extra kwargs for :class:`TimmViT`.
        interaction_indexes: Inclusive ``[start, end]`` block ranges.
        deform_num_heads: Deformable attention heads.
        deform_ratio: Value projection ratio of the deformable attention.
        **kwargs: Remaining :class:`CRADIOv4CoMer` arguments.
    """

    def __init__(
        self,
        model_name: str = 'vit_small_patch16_dinov3',
        checkpoint: str | None = None,
        vit_cfg: dict | None = None,
        interaction_indexes: Sequence[Sequence[int]] = ((0, 2), (3, 5), (6, 8), (9, 11)),
        deform_num_heads: int = 6,
        deform_ratio: float = 1.0,
        **kwargs: object,
    ) -> None:
        super().__init__(
            checkpoint=None,
            vit_cfg=dict(vit_cfg or {}, model_name=model_name, checkpoint=checkpoint),
            interaction_indexes=interaction_indexes,
            deform_num_heads=deform_num_heads,
            deform_ratio=deform_ratio,
            **kwargs,
        )

    def _build_vit(self, vit_cfg: dict | None, with_cp: bool) -> nn.Module:
        return TimmViT(**dict(vit_cfg or {}, with_cp=with_cp))


@MODELS.register_module()
class TimmFeatureBackbone(BaseModule):
    """A (frozen) timm ``features_only`` backbone, e.g. DINOv3 ConvNeXt-T.

    Args:
        model_name: timm model name, e.g. ``convnext_tiny.dinov3_lvd1689m``.
        checkpoint: Local timm weights (HF snapshot dir or file).
        out_indices: Feature stages to return (ConvNeXt: 0..3 = strides 4..32).
        frozen: Freeze the model; it then runs without autograd.
        model_cfg: Extra ``timm.create_model`` kwargs.
        init_cfg: Standard mmengine init config.
    """

    def __init__(
        self,
        model_name: str = 'convnext_tiny',
        checkpoint: str | None = None,
        out_indices: Sequence[int] = (0, 1, 2, 3),
        frozen: bool = True,
        model_cfg: dict | None = None,
        init_cfg: dict | list[dict] | None = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.frozen = frozen
        self.model = create_timm_model(
            model_name,
            checkpoint,
            features_only=True,
            out_indices=tuple(out_indices),
            **(model_cfg or {}),
        )
        self.channels: list[int] = self.model.feature_info.channels()
        self.strides: list[int] = self.model.feature_info.reduction()
        if frozen:
            freeze_module(self.model)

    def train(self, mode: bool = True) -> 'TimmFeatureBackbone':
        super().train(mode)
        if self.frozen:
            self.model.eval()
        return self

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        x, in_h, in_w = _pad_to(x, max(self.strides))
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen):
            feats = self.model(x)
        return tuple(
            f[..., : math.ceil(in_h / s), : math.ceil(in_w / s)]
            for f, s in zip(feats, self.strides, strict=True)
        )
