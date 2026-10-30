# Copyright (c) OpenMMLab. All rights reserved.
"""Tiny versions of the C-RADIOv4 RTMDet configs for the TensorRT tests.

Together they cover every export patch: plain ViT (``radio``), ViT-CoMer's
deformable attention (``comer``), the DCFL head's DCNv2 (``dcfl``) and the
ViTDet pyramid's ``LayerNorm2d`` (``vitdet``).
"""

import torch
from mmengine.config import ConfigDict

from mmrotate.registry import MODELS
from mmrotate.testing import get_detector_cfg
from mmrotate.utils import register_all_modules

TINY_DIM = 32
TINY_VIT = dict(embed_dim=TINY_DIM, depth=4, num_heads=2, max_img_size=256)
CONFIGS = {
    'radio': 'cradio/cradiov4-h_rtmdet-3x-dota_ms.yaml',
    'comer': 'cradio/cradiov4-h-comer_rtmdet-3x-dota_ms.yaml',
    'dcfl': 'cradio/cradiov4-h-comer_rtmdet-dcfl-3x-dota_ms.yaml',
    'vitdet': 'cradio/cradiov4-h-vitdet_rtmdet-3x-dota_ms.yaml',
}


def tiny_model_cfg(name: str) -> ConfigDict:
    """``model`` config of :data:`CONFIGS[name]` with a 4-block, 32-dim ViT."""
    model = get_detector_cfg(CONFIGS[name])
    backbone = model.backbone
    backbone.checkpoint = None
    backbone.vit_cfg = TINY_VIT
    if backbone.type == 'CRADIOv4CoMer':
        backbone.interaction_indexes = [[0, 0], [1, 1], [2, 2], [3, 3]]
        backbone.deform_num_heads = 2
        backbone.conv_inplane = 8
        backbone.norm_cfg = dict(type='BN')
    elif model.neck.type == 'ViTDetSimpleFPN':
        backbone.out_indices = [3]
    else:
        backbone.out_indices = [1, 2, 3]
    if model.neck.type == 'ViTDetSimpleFPN':
        model.neck.in_channels = TINY_DIM
    else:
        model.neck.in_channels = [TINY_DIM] * len(model.neck.in_channels)
        model.neck.norm_cfg = dict(type='BN')
    model.bbox_head.norm_cfg = dict(type='BN')
    return model


def build_tiny(name: str, device: str = 'cpu', seed: int = 0) -> torch.nn.Module:
    """A tiny detector in eval mode with non-trivial head / DCN weights."""
    register_all_modules()
    torch.manual_seed(seed)
    model = MODELS.build(tiny_model_cfg(name))
    model.init_weights()
    for module_name, module in model.named_modules():
        # Zero-initialised offsets would hide errors in the deformable convs.
        if module_name.endswith('conv_offset'):
            torch.nn.init.normal_(module.weight, std=0.05)
    for param_name, param in model.bbox_head.named_parameters():
        if param.dim() > 1:
            torch.nn.init.normal_(param, std=0.05)
        elif 'rtm_cls' in param_name:  # scores around 0.5 instead of the 0.01 prior
            torch.nn.init.zeros_(param)
    return model.to(device).eval()
