# Copyright (c) OpenMMLab. All rights reserved.
"""ViT-CoMer adapter on top of a (frozen) C-RADIOv4 ViT.

Follows ``ViTCoMer`` from https://github.com/Traffic-X/ViT-CoMer
(``detection/mmdet_custom/models/backbones/vit_comer.py``) with the RADIO
ViT as the plain transformer. RADIO's prefix (cls + register) tokens are
carried through the ViT blocks, while the CoMer modules only see the spatial
tokens.
"""

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmengine.model import BaseModule
from torch import Tensor, nn

from mmrotate.registry import MODELS

from .comer_modules import CNN, CTIBlock, deform_inputs
from .cradiov4 import freeze_module, load_radio_weights
from .ms_deform_attn import MSDeformAttn
from .radio_vit import RADIOVisionTransformer

BoolOrList = bool | Sequence[bool]


def _pick(value: BoolOrList, i: int) -> bool:
    return value if isinstance(value, bool) else bool(value[i])


@MODELS.register_module()
class CRADIOv4CoMer(BaseModule):
    """C-RADIOv4 + ViT-CoMer multi-scale backbone.

    Outputs ``embed_dim``-channel maps at strides 4/8/16/32, selected by
    ``out_indices`` (``(1, 2, 3)`` -> strides 8/16/32).

    Args:
        checkpoint: HF C-RADIO directory or weights file, loaded with strict
            key matching in ``init_weights``.
        freeze_vit: Freeze the RADIO ViT (kept in eval mode). Gradients still
            flow through it to the CoMer modules.
        vit_cfg: Kwargs for :class:`RADIOVisionTransformer`.
        vit_with_cp: Gradient checkpointing for the ViT blocks.
        interaction_indexes: Inclusive ``[start, end]`` ViT block ranges, one
            per interaction stage.
        conv_inplane: Width of the CNN branch (SPM).
        n_points: Deformable sampling points.
        deform_num_heads: Deformable attention heads.
        init_values: Initial value of the CTI_toV gate ``gamma``.
        with_cffn: Use the ConvFFN in the extractors.
        cffn_ratio: ConvFFN hidden ratio.
        deform_ratio: Value projection ratio of the deformable attention.
        add_vit_feature: Add the final ViT tokens to every output level.
        use_extra_CTI: Add ``extra_num`` extra CTI blocks after the last stage.
        use_CTI_toV: Enable CNN -> ViT interaction (bool or per stage).
        use_CTI_toC: Enable ViT -> CNN interaction (bool or per stage).
        cnn_feature_interaction: Multi-scale self deformable attention in the
            CNN branch (bool or per stage).
        extra_num: Number of extra CTI blocks.
        drop_path_rate: Stochastic depth of the adapter modules.
        with_cp: Gradient checkpointing for the adapter modules.
        norm_cfg: Norm for the CNN branch and the output norms.
        out_indices: Output levels (0: s4, 1: s8, 2: s16, 3: s32).
        msda_backend: ``'auto'`` | ``'triton'`` | ``'pytorch'``.
        init_cfg: Standard mmengine init config.
    """

    def __init__(
        self,
        checkpoint: str | None = None,
        freeze_vit: bool = True,
        vit_cfg: dict | None = None,
        vit_with_cp: bool = False,
        interaction_indexes: Sequence[Sequence[int]] = (
            (0, 7),
            (8, 15),
            (16, 23),
            (24, 31),
        ),
        conv_inplane: int = 64,
        n_points: int = 4,
        deform_num_heads: int = 16,
        init_values: float = 0.0,
        with_cffn: bool = True,
        cffn_ratio: float = 0.25,
        deform_ratio: float = 0.5,
        add_vit_feature: bool = True,
        use_extra_CTI: bool = True,
        use_CTI_toV: BoolOrList = True,
        use_CTI_toC: BoolOrList = True,
        cnn_feature_interaction: BoolOrList = True,
        extra_num: int = 4,
        drop_path_rate: float = 0.0,
        with_cp: bool = False,
        norm_cfg: dict | None = None,
        out_indices: Sequence[int] = (1, 2, 3),
        msda_backend: str = 'auto',
        init_cfg: dict | list[dict] | None = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        norm_cfg = norm_cfg or dict(type='SyncBN', requires_grad=True)
        self.checkpoint = checkpoint
        self.freeze_vit = freeze_vit
        self.interaction_indexes = [tuple(idx) for idx in interaction_indexes]
        self.add_vit_feature = add_vit_feature
        self.out_indices = tuple(out_indices)

        self.vit = RADIOVisionTransformer(**dict(vit_cfg or {}, with_cp=vit_with_cp))
        if self.vit.patch_size != 16:
            raise ValueError('ViT-CoMer assumes a patch size of 16')
        embed_dim = self.vit.embed_dim
        self.embed_dim = embed_dim

        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        # The stride 4 branch (spm.fc1, up, norm1) is only built when used.
        self.with_stride4 = 0 in self.out_indices
        self.spm = CNN(
            inplanes=conv_inplane,
            embed_dim=embed_dim,
            norm_cfg=norm_cfg,
            with_stride4=self.with_stride4,
        )
        n_stages = len(self.interaction_indexes)
        self.interactions = nn.Sequential(
            *[
                CTIBlock(
                    dim=embed_dim,
                    num_heads=deform_num_heads,
                    n_points=n_points,
                    init_values=init_values,
                    drop_path=drop_path_rate,
                    with_cffn=with_cffn,
                    cffn_ratio=cffn_ratio,
                    deform_ratio=deform_ratio,
                    use_CTI_toV=_pick(use_CTI_toV, i),
                    use_CTI_toC=_pick(use_CTI_toC, i),
                    cnn_feature_interaction=_pick(cnn_feature_interaction, i),
                    extra_CTI=(i == n_stages - 1) and use_extra_CTI,
                    extra_num=extra_num,
                    with_cp=with_cp,
                    msda_backend=msda_backend,
                )
                for i in range(n_stages)
            ]
        )
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2) if self.with_stride4 else None
        self.norm1 = build_norm_layer(norm_cfg, embed_dim)[1] if self.with_stride4 else None
        self.norm2 = build_norm_layer(norm_cfg, embed_dim)[1]
        self.norm3 = build_norm_layer(norm_cfg, embed_dim)[1]
        self.norm4 = build_norm_layer(norm_cfg, embed_dim)[1]

        if self.up is not None:
            self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        nn.init.normal_(self.level_embed)

        if freeze_vit:
            freeze_module(self.vit)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm | nn.BatchNorm2d | nn.SyncBatchNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d | nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @staticmethod
    def _init_deform_weights(m: nn.Module) -> None:
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def init_weights(self) -> None:
        super().init_weights()
        if self.checkpoint is not None:
            load_radio_weights(self.vit, self.checkpoint)

    def train(self, mode: bool = True) -> 'CRADIOv4CoMer':
        super().train(mode)
        if self.freeze_vit:
            self.vit.eval()
        return self

    def _add_level_embed(self, c2: Tensor, c3: Tensor, c4: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return c2 + self.level_embed[0], c3 + self.level_embed[1], c4 + self.level_embed[2]

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        in_h, in_w = x.shape[-2:]
        pad_h, pad_w = (-in_h) % 32, (-in_w) % 32
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        # CNN branch (SPM)
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # ViT tokens: prefix (cls + registers) kept aside, spatial tokens to CoMer.
        tokens, H, W = self.vit.patch_embed(x)
        num_skip = self.vit.num_skip
        prefix = [tokens[:, :num_skip]]
        x = tokens[:, num_skip:]
        bs, _, dim = x.shape

        for layer, (start, end) in zip(self.interactions, self.interaction_indexes, strict=True):

            def run_blocks(feat: Tensor, start: int = start, end: int = end) -> Tensor:
                out = self.vit.run_blocks(torch.cat([prefix[0], feat], dim=1), start, end)
                prefix[0] = out[:, :num_skip]
                return out[:, num_skip:]

            x, c = layer(x, c, run_blocks, deform_inputs1, deform_inputs2, H, W)

        # Split & reshape
        n2, n3 = c2.size(1), c3.size(1)
        c2 = c[:, :n2].transpose(1, 2).reshape(bs, dim, H * 2, W * 2)
        c3 = c[:, n2 : n2 + n3].transpose(1, 2).reshape(bs, dim, H, W)
        c4 = c[:, n2 + n3 :].transpose(1, 2).reshape(bs, dim, H // 2, W // 2)
        if self.up is not None and c1 is not None:
            c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x3 = x.transpose(1, 2).reshape(bs, dim, H, W)
            x2 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x3, scale_factor=0.5, mode='bilinear', align_corners=False)
            c2, c3, c4 = c2 + x2, c3 + x3, c4 + x4
            if c1 is not None:
                c1 = c1 + F.interpolate(x3, scale_factor=4, mode='bilinear', align_corners=False)

        norms = (self.norm1, self.norm2, self.norm3, self.norm4)
        feats = (c1, c2, c3, c4)
        outs = []
        for i in self.out_indices:
            out = norms[i](feats[i])
            if pad_h or pad_w:
                stride = 4 * 2**i
                out = out[..., : math.ceil(in_h / stride), : math.ceil(in_w / stride)]
            outs.append(out)
        return tuple(outs)
