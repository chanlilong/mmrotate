# Copyright (c) OpenMMLab. All rights reserved.
import math
from collections.abc import Sequence

import torch.nn.functional as F
from mmengine.logging import MMLogger
from mmengine.model import BaseModule
from torch import Tensor

from mmrotate.registry import MODELS

from .checkpoint import load_radio_state_dict
from .radio_vit import RADIOVisionTransformer


def load_radio_weights(vit: RADIOVisionTransformer, checkpoint: str) -> None:
    """Load HF C-RADIO weights into ``vit`` with strict key matching."""
    state_dict = load_radio_state_dict(checkpoint)
    vit.load_state_dict(state_dict, strict=True)
    MMLogger.get_current_instance().info(f'Loaded C-RADIO weights from {checkpoint}')


def freeze_module(module: BaseModule | RADIOVisionTransformer) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad = False


def resize_to_stride(feat: Tensor, src_stride: int, dst_stride: int) -> Tensor:
    """Parameter-free resampling of a feature map between strides."""
    if dst_stride == src_stride:
        return feat
    if dst_stride < src_stride:
        return F.interpolate(
            feat, scale_factor=src_stride / dst_stride, mode='bilinear', align_corners=False
        )
    k = dst_stride // src_stride
    return F.max_pool2d(feat, kernel_size=k, stride=k)


@MODELS.register_module()
class CRADIOv4(BaseModule):
    """NVIDIA C-RADIOv4 ViT as a plain multi-scale backbone.

    Features of the blocks in ``out_indices`` (all at the patch stride) are
    resampled without parameters to ``out_strides`` (ViTDet-style simple
    pyramid), so the backbone stays weight-identical to the HF release.

    Args:
        checkpoint: HF model directory (e.g. ``/models/nvidia/C-RADIOv4-H``)
            or a ``.safetensors`` / ``.pth`` file. Loaded in ``init_weights``.
        out_indices: Block indices whose outputs are returned.
        out_strides: Output stride of each returned level.
        frozen: Freeze all ViT weights and keep it in eval mode.
        vit_cfg: Extra kwargs for :class:`RADIOVisionTransformer`.
        init_cfg: Standard mmengine init config.
    """

    def __init__(
        self,
        checkpoint: str | None = None,
        out_indices: Sequence[int] = (15, 23, 31),
        out_strides: Sequence[int] = (8, 16, 32),
        frozen: bool = True,
        vit_cfg: dict | None = None,
        init_cfg: dict | list[dict] | None = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        if len(out_indices) != len(out_strides):
            raise ValueError('out_indices and out_strides must have the same length')
        self.checkpoint = checkpoint
        self.out_indices = tuple(out_indices)
        self.out_strides = tuple(out_strides)
        self.frozen = frozen
        self.vit = RADIOVisionTransformer(**(vit_cfg or {}))
        if frozen:
            freeze_module(self.vit)

    def init_weights(self) -> None:
        super().init_weights()
        if self.checkpoint is not None:
            load_radio_weights(self.vit, self.checkpoint)

    def train(self, mode: bool = True) -> 'CRADIOv4':
        super().train(mode)
        if self.frozen:
            self.vit.eval()
        return self

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        # Pad bottom/right so every output stride divides the input evenly.
        in_h, in_w = x.shape[-2:]
        divisor = max(self.vit.patch_size, *self.out_strides)
        pad_h, pad_w = (-in_h) % divisor, (-in_w) % divisor
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        feats = self.vit.forward_features(x, self.out_indices)
        patch = self.vit.patch_size
        outs = []
        for feat, stride in zip(feats, self.out_strides, strict=True):
            out = resize_to_stride(feat, patch, stride)
            outs.append(out[..., : math.ceil(in_h / stride), : math.ceil(in_w / stride)])
        return tuple(outs)
