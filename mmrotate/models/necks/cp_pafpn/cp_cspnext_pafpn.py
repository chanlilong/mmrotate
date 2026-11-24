# Copyright (c) OpenMMLab. All rights reserved.
"""``mmdet.CSPNeXtPAFPN`` with gradient checkpointing on its high-resolution layers.

With a stride 4 (P2) input, RTMDet's PAFPN runs its CSP layers on 256x256 and
128x128 maps at the backbone's full width (1280 channels for C-RADIOv4-H CoMer,
2560 after the concatenation), and every convolution keeps three such maps for
backward. Checkpointing the layers of those levels keeps only the input of each
convolution block instead. The module tree, the parameters and the results are
those of ``CSPNeXtPAFPN``; ``forward`` follows mmdet 3.3.0
``mmdet/models/necks/cspnext_pafpn.py`` (Apache-2.0, OpenMMLab).
"""

from collections.abc import Sequence

import torch
from mmdet.models.layers import CSPLayer
from mmdet.models.necks import CSPNeXtPAFPN
from torch import Tensor, nn

from mmrotate.models.backbones.cradiov4.bn_checkpoint import checkpoint_bn
from mmrotate.registry import MODELS


@MODELS.register_module()
class CheckpointedCSPNeXtPAFPN(CSPNeXtPAFPN):
    """``CSPNeXtPAFPN`` that checkpoints the layers of the levels in ``cp_levels``.

    Same state-dict keys and outputs as ``mmdet.CSPNeXtPAFPN``. BatchNorm running
    statistics are updated once per step (see ``checkpoint_bn``).

    Args:
        cp_levels: Pyramid levels (0 is the highest resolution input) whose
            layers are checkpointed: the top-down and bottom-up CSP layers
            working at that resolution, and the reduce, downsample and output
            convolutions producing a map of that resolution. In a CSP layer
            every convolution block is its own checkpoint, so backward rebuilds
            one block at a time. Defaults to the two highest resolutions.
        **kwargs: ``CSPNeXtPAFPN`` arguments.
    """

    def __init__(self, cp_levels: Sequence[int] = (0, 1), **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.cp_levels = frozenset(cp_levels)

    def _use_cp(self, x: Tensor, level: int) -> bool:
        return level in self.cp_levels and x.requires_grad

    def _conv(self, conv: nn.Module, x: Tensor, level: int) -> Tensor:
        return checkpoint_bn(conv, x) if self._use_cp(x, level) else conv(x)

    def _csp(self, layer: CSPLayer, x: Tensor, level: int) -> Tensor:
        """``CSPLayer.forward`` with one checkpoint per convolution block."""
        if not self._use_cp(x, level):
            return layer(x)
        x_short = checkpoint_bn(layer.short_conv, x)
        x_main = checkpoint_bn(layer.main_conv, x)
        for block in layer.blocks:
            x_main = checkpoint_bn(block, x_main)
        x_final = torch.cat((x_main, x_short), dim=1)
        if layer.channel_attention:
            x_final = layer.attention(x_final)
        return checkpoint_bn(layer.final_conv, x_final)

    def forward(self, inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        num = len(self.in_channels)
        assert len(inputs) == num

        # top-down path
        inner_outs = [inputs[-1]]
        for idx in range(num - 1, 0, -1):
            feat_high = self._conv(self.reduce_layers[num - 1 - idx], inner_outs[0], idx)
            inner_outs[0] = feat_high
            upsample_feat = self.upsample(feat_high)
            inner_out = self._csp(
                self.top_down_blocks[num - 1 - idx],
                torch.cat([upsample_feat, inputs[idx - 1]], 1),
                idx - 1,
            )
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(num - 1):
            downsample_feat = self._conv(self.downsamples[idx], outs[-1], idx + 1)
            out = self._csp(
                self.bottom_up_blocks[idx],
                torch.cat([downsample_feat, inner_outs[idx + 1]], 1),
                idx + 1,
            )
            outs.append(out)

        # out convs
        for idx, conv in enumerate(self.out_convs):
            outs[idx] = self._conv(conv, outs[idx], idx)
        return tuple(outs)
