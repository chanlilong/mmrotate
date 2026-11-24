# Copyright (c) OpenMMLab. All rights reserved.
"""EMA model that does not copy or average frozen parameters.

``EMAHook`` keeps a full copy of the model and averages every parameter on
every iteration. With a frozen backbone (e.g. the 632 M parameter C-RADIOv4-H
ViT) the copy of the frozen part costs its size in GPU memory (2.5 GB) and two
passes over it per step, to average weights that never change.
"""

from collections.abc import Mapping

from mmdet.models.layers import ExpMomentumEMA
from torch import Tensor, nn

from mmrotate.registry import MODELS


@MODELS.register_module()
class FrozenSharedExpMomentumEMA(ExpMomentumEMA):
    """``mmdet.ExpMomentumEMA`` whose frozen parameters alias the source model's.

    Parameters with ``requires_grad=False`` when the EMA model is built share
    their storage with the source model and are left out of the averaging. The
    EMA of a constant is the constant, so the averaged model is the same; the
    only difference is that the frozen EMA weights are exactly the source
    weights instead of drifting by float rounding at every update. State-dict
    keys are unchanged, so checkpoints stay compatible both ways.

    Parameters that are unfrozen after the EMA model is built are not averaged.

    Args:
        model: The model to be averaged.
        **kwargs: ``ExpMomentumEMA`` arguments.
    """

    def __init__(self, model: nn.Module, **kwargs: object) -> None:
        super().__init__(model, **kwargs)
        source = dict(model.named_parameters())
        averaged = dict(self.module.named_parameters())
        shared = [name for name, p in source.items() if not p.requires_grad]
        for name in shared:
            if averaged[name].device != source[name].device:
                raise ValueError('frozen parameters can only be shared on the same device')
            averaged[name].data = source[name].data
        self.shared_parameters = frozenset(shared)
        self.avg_parameters = {
            k: v for k, v in self.avg_parameters.items() if k not in self.shared_parameters
        }

    def load_state_dict(self, state_dict: Mapping[str, Tensor], strict: bool = True) -> object:
        """Load the averaged parameters; the shared ones stay the source model's."""
        skip = {f'module.{name}' for name in self.shared_parameters}
        kept = {k: v for k, v in state_dict.items() if k not in skip}
        return super().load_state_dict(kept, strict=strict and len(kept) == len(state_dict))
