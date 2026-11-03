# Copyright (c) OpenMMLab. All rights reserved.
"""ViT-CoMer adapter on a (frozen) LingBot-Vision ViT."""

from collections.abc import Sequence

from torch import nn

from mmrotate.registry import MODELS

from ..cradiov4.vit_comer import CRADIOv4CoMer
from .checkpoint import load_lingbot_weights
from .lingbot_vit import LingBotViT


@MODELS.register_module()
class LingBotViTCoMer(CRADIOv4CoMer):
    """ViT-CoMer adapter on a (frozen) LingBot-Vision ViT, e.g. ViT-S/16 or ViT-L/16.

    Same adapter as :class:`CRADIOv4CoMer` (SPM CNN branch, CTI blocks with
    multi-scale deformable attention); only the plain ViT differs. Defaults
    follow ViT-CoMer-S (12 blocks, 4 stages of 3, 6 deformable heads).

    The ViT weights are loaded in ``init_weights`` (after the adapter init and
    any LoRA injection) with strict key matching, as for C-RADIO.

    Args:
        arch: LingBot architecture, see ``LINGBOT_ARCHS``.
        checkpoint: HF LingBot-Vision directory (``model.pt``) or weights file.
            ``None`` keeps the random init (tests).
        vit_cfg: Extra kwargs for :class:`LingBotViT`.
        interaction_indexes: Inclusive ``[start, end]`` block ranges.
        deform_num_heads: Deformable attention heads.
        deform_ratio: Value projection ratio of the deformable attention.
        **kwargs: Remaining :class:`CRADIOv4CoMer` arguments (e.g. ``lora_cfg``).
    """

    def __init__(
        self,
        arch: str = 'vit_small',
        checkpoint: str | None = None,
        vit_cfg: dict | None = None,
        interaction_indexes: Sequence[Sequence[int]] = ((0, 2), (3, 5), (6, 8), (9, 11)),
        deform_num_heads: int = 6,
        deform_ratio: float = 1.0,
        **kwargs: object,
    ) -> None:
        super().__init__(
            checkpoint=None,
            vit_cfg=dict(vit_cfg or {}, arch=arch),
            interaction_indexes=interaction_indexes,
            deform_num_heads=deform_num_heads,
            deform_ratio=deform_ratio,
            **kwargs,
        )
        self.lingbot_checkpoint = checkpoint
        last = self.interaction_indexes[-1][1]
        if last != self.vit.depth - 1:
            raise ValueError(
                f'interaction_indexes end at block {last}, but {arch} has {self.vit.depth} blocks'
            )

    def _build_vit(self, vit_cfg: dict | None, with_cp: bool) -> nn.Module:
        return LingBotViT(**dict(vit_cfg or {}, with_cp=with_cp))

    def init_weights(self) -> None:
        super().init_weights()
        if self.lingbot_checkpoint is not None:
            load_lingbot_weights(self.vit, self.lingbot_checkpoint)
