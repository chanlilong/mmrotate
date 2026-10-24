# Copyright (c) OpenMMLab. All rights reserved.
"""Layer-wise learning rate decay for fine-tuning the C-RADIOv4 ViT.

mmengine port of ``LayerDecayOptimizerConstructor`` used by ViTDet
(https://github.com/ViTAE-Transformer/ViTDet, ``mmcv_custom/``), adapted to
the parameter names of :class:`RADIOVisionTransformer`.
"""

from mmengine.logging import MMLogger
from mmengine.optim import DefaultOptimWrapperConstructor
from torch import nn

from mmrotate.registry import OPTIM_WRAPPER_CONSTRUCTORS

NO_DECAY_KEYS = ('pos_embed', 'cls_token')


def get_layer_id(name: str, num_layers: int, prefix: str) -> int:
    """Layer id of a parameter: 0 for the embedding, ``i + 1`` for block ``i``
    and ``num_layers + 1`` for everything outside the ViT (adapter, neck,
    head)."""
    if not name.startswith(prefix):
        return num_layers + 1
    name = name[len(prefix) :]
    if name.startswith('model.blocks.'):
        return int(name.split('.')[2]) + 1
    if name.startswith(('model.patch_generator.', 'input_conditioner.')):
        return 0
    return num_layers + 1


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class CRADIOv4LayerDecayOptimWrapperConstructor(DefaultOptimWrapperConstructor):
    """Optimizer constructor with ViTDet layer-wise lr decay for C-RADIOv4.

    The lr of layer ``id`` is ``base_lr * layer_decay_rate ** (num_layers + 1
    - id)``: the patch embedding gets the smallest lr, the last ViT block
    ``layer_decay_rate * base_lr`` and the rest of the detector ``base_lr``.
    1-D parameters, biases, ``pos_embed`` and ``cls_token`` get no weight
    decay. Frozen parameters are skipped.

    ``paramwise_cfg`` keys:
        num_layers: Depth of the ViT (32 for C-RADIOv4-H).
        layer_decay_rate: Per-layer lr decay (ViTDet: 0.7 B, 0.8 L, 0.9 H).
        backbone_prefix: Name prefix of the RADIO ViT parameters.
    """

    def add_params(
        self,
        params: list[dict],
        module: nn.Module,
        prefix: str = '',
        is_dcn_module: int | float | None = None,
    ) -> None:
        num_layers = self.paramwise_cfg['num_layers']
        decay_rate = self.paramwise_cfg['layer_decay_rate']
        vit_prefix = self.paramwise_cfg.get('backbone_prefix', 'backbone.vit.')
        weight_decay = self.base_wd or 0.0

        groups: dict[str, dict] = {}
        for name, param in module.named_parameters(prefix=prefix):
            if not param.requires_grad:
                continue
            if param.ndim == 1 or name.endswith('.bias') or any(k in name for k in NO_DECAY_KEYS):
                decay_name, this_wd = 'no_decay', 0.0
            else:
                decay_name, this_wd = 'decay', weight_decay
            layer_id = get_layer_id(name, num_layers, vit_prefix)
            group_name = f'layer_{layer_id}_{decay_name}'
            if group_name not in groups:
                scale = decay_rate ** (num_layers + 1 - layer_id)
                groups[group_name] = dict(
                    params=[],
                    lr=self.base_lr * scale,
                    lr_scale=scale,
                    weight_decay=this_wd,
                    group_name=group_name,
                    num_params=0,
                )
            groups[group_name]['params'].append(param)
            groups[group_name]['num_params'] += param.numel()

        logger = MMLogger.get_current_instance()
        for group_name, group in sorted(groups.items(), key=lambda kv: kv[1]['lr_scale']):
            logger.info(
                f'layer decay group {group_name}: lr_scale={group["lr_scale"]:.4g} '
                f'weight_decay={group["weight_decay"]} params={group.pop("num_params")}'
            )
        params.extend(groups.values())
