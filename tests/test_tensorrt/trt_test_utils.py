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


def convert_tiny(
    model: torch.nn.Module,
    name: str,
    out_dir: object,
    input_shape: tuple[int, int, int, int],
    precision: str = 'fp32',
    attention: str = 'legacy',
    plugins: str | None = None,
) -> dict:
    """``tools/tensorrt/convert.py`` for a tiny model: export, build, write the deploy dir.

    Args:
        model: Detector from :func:`build_tiny` on a CUDA device.
        name: Its key in :data:`CONFIGS`.
        out_dir: Deploy directory (a ``pathlib.Path``).
        input_shape: Static engine input ``(B, 3, H, W)``.
        precision: ``'fp32'`` / ``'fp16'`` / ``'bf16'``.
        attention: Attention mode of ``convert.py --attention``.
        plugins: ``'aot'`` / ``'jit'`` to export deformable attention as the
            plugin node, ``None`` for the ``grid_sample`` export.

    Returns:
        The meta dict written to ``trt_meta.json`` (with the ``fusion`` counts).
    """
    from mmengine.config import Config

    from mmrotate.tensorrt import engine_report, make_deploy_cfg
    from mmrotate.tensorrt import precision as prec
    from mmrotate.tensorrt.attention import resolve_attention
    from mmrotate.tensorrt.builder import build_engine
    from mmrotate.tensorrt.deploy import DEPLOY_CFG, ENGINE_FILE, write_meta
    from mmrotate.tensorrt.export import export_onnx
    from mmrotate.tensorrt.export_patches import export_mode
    from mmrotate.tensorrt.runtime import TRTEngine

    low_dtype = prec.autocast_dtype(precision)
    probe = [torch.randn(1, *input_shape[1:], device='cuda')]
    plan, stats = resolve_attention(attention, model, probe, low_dtype)
    with export_mode(
        model,
        prec.fp32_module_types(precision),
        attention=plan or 'legacy',
        msda='gridsample' if plugins is None else 'plugin',
    ):
        spec, names = export_onnx(model, out_dir / 'm.onnx', input_shape, autocast_dtype=low_dtype)
    build_engine(
        out_dir / 'm.onnx', out_dir / ENGINE_FILE, tf32=precision != 'fp32', plugins=plugins
    )
    engine = TRTEngine(out_dir / ENGINE_FILE, 'cuda')
    fusion = engine_report.fusion_summary(engine_report.layer_info(engine))
    del engine
    meta = dict(
        input_shape=list(input_shape),
        output_names=names,
        output_spec=spec,
        precision=precision,
        attention=stats,
        plugins=plugins,
        fusion=fusion,
    )
    write_meta(out_dir, meta)
    cfg = Config(dict(model=tiny_model_cfg(name), default_scope='mmrotate'))
    make_deploy_cfg(cfg).dump(str(out_dir / DEPLOY_CFG))
    return meta
