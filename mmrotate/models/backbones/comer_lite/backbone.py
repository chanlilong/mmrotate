# Copyright (c) OpenMMLab. All rights reserved.
"""Cheaper ViT-CoMer variants on a timm ViT (see ``COMER_LITE.md``).

:class:`TimmViTCoMerLite` keeps the :class:`TimmViTCoMer` adapter and adds two
options that the upstream module has no switch for:

* ``out_channels``: per-level output widths. CoMer emits ``embed_dim`` channels
  at every stride, so the neck's stride-4 layers run at 384 channels. Here the
  stride-4 path (``up``, ``spm.fc1`` and the projected ViT map) is built at
  ``out_channels[0]`` directly, and the stride 8/16/32 outputs get a 1x1 conv +
  norm when their width differs from ``embed_dim``.
* ``dedup_norm``: ``CTI_toC._interact`` LayerNorms the CNN tokens and
  ``MultiscaleExtractor`` normalises the same tensors again. The inner query
  norm and the outer value norm are replaced by identities, so each attention
  input goes through one LayerNorm. The residual stream still starts from the
  outer ``query_norm``, as upstream.

With the default arguments the module is identical to :class:`TimmViTCoMer`
(same parameters, same state-dict keys, same outputs).
"""

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from torch import Tensor, nn

from mmrotate.registry import MODELS

from ..cradiov4.comer_modules import CTI_toC, deform_inputs
from ..timm_dinov3 import TimmViTCoMer


@MODELS.register_module()
class TimmViTCoMerLite(TimmViTCoMer):
    """:class:`TimmViTCoMer` with slim outputs and de-duplicated norms.

    Args:
        out_channels: Output width per level (strides 4/8/16/32). ``None``
            keeps ``embed_dim`` everywhere.
        dedup_norm: Apply one LayerNorm per attention input in the CNN-side
            interactions (``CTI_toC`` and the extra CTIs) instead of two.
        **kwargs: :class:`TimmViTCoMer` arguments.
    """

    def __init__(
        self,
        out_channels: Sequence[int] | None = None,
        dedup_norm: bool = False,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        dim = self.embed_dim
        self.out_channels = list(out_channels) if out_channels is not None else [dim] * 4
        if len(self.out_channels) != 4:
            raise ValueError(f'out_channels needs 4 entries, got {self.out_channels}')
        norm_cfg = kwargs.get('norm_cfg') or dict(type='SyncBN', requires_grad=True)

        # Stride 4 at its final width: no 384-channel map at 4x the resolution.
        c0 = self.out_channels[0]
        self.vit_proj0: nn.Module | None = None
        if self.with_stride4 and c0 != dim:
            self.up = nn.ConvTranspose2d(dim, c0, 2, 2)
            self.spm.fc1 = nn.Conv2d(self.spm.fc1.in_channels, c0, kernel_size=1, bias=True)
            self.norm1 = build_norm_layer(norm_cfg, c0)[1]
            if self.add_vit_feature:
                # 1x1 conv and bilinear upsampling commute, so projecting at
                # stride 16 equals projecting the upsampled map, at 1/16 the cost.
                self.vit_proj0 = nn.Conv2d(dim, c0, kernel_size=1, bias=False)
            for m in (self.up, self.spm.fc1, self.norm1, self.vit_proj0):
                if m is not None:
                    self._init_weights(m)

        self.out_projs = nn.ModuleList([nn.Identity()])
        for c in self.out_channels[1:]:
            proj = nn.Identity()
            if c != dim:
                proj = nn.Sequential(
                    nn.Conv2d(dim, c, kernel_size=1, bias=False),
                    build_norm_layer(norm_cfg, c)[1],
                )
                proj.apply(self._init_weights)
            self.out_projs.append(proj)

        if dedup_norm:
            for block in self.interactions:
                ctis = [getattr(block, 'cti_toc', None), *(block.extra_CTIs or [])]
                for cti in ctis:
                    if isinstance(cti, CTI_toC) and cti.cnn_feature_interaction:
                        cti.feat_norm = nn.Identity()
                        cti.cfinter.query_norm = nn.Identity()

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        in_h, in_w = x.shape[-2:]
        pad_h, pad_w = (-in_h) % 32, (-in_w) % 32
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

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
                x1 = self.vit_proj0(x3) if self.vit_proj0 is not None else x3
                c1 = c1 + F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)

        norms = (self.norm1, self.norm2, self.norm3, self.norm4)
        feats = (c1, c2, c3, c4)
        outs = []
        for i in self.out_indices:
            out = self.out_projs[i](norms[i](feats[i]))
            if pad_h or pad_w:
                stride = 4 * 2**i
                out = out[..., : math.ceil(in_h / stride), : math.ceil(in_w / stride)]
            outs.append(out)
        return tuple(outs)
