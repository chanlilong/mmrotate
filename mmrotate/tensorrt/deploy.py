# Copyright (c) OpenMMLab. All rights reserved.
"""Deploy directory written by ``tools/tensorrt/convert.py``::

    <out_dir>/
        end2end.engine    serialized TensorRT engine
        trt_meta.json     I/O names, output nesting, precision, build environment
        deploy_cfg.py     the source config with ``model`` replaced by a
                          ``TRTSingleStageDetector``; usable with Runner.test()
        end2end.onnx      (with --keep-onnx) the graph the engine was built from

Paths inside ``deploy_cfg.py`` are relative to its folder (``{{ fileDirname }}``),
so the directory can be moved or copied to another machine with the same GPU
and TensorRT version.
"""

import copy
import json
from pathlib import Path

from mmengine.config import Config, ConfigDict
from mmengine.registry import init_default_scope

from mmrotate.registry import MODELS
from mmrotate.utils import register_all_modules

from .detector import META_FILE, TRTSingleStageDetector

ENGINE_FILE = 'end2end.engine'
ONNX_FILE = 'end2end.onnx'
DEPLOY_CFG = 'deploy_cfg.py'
# Training-only hooks that look for modules / weights the engine replaces.
_DROP_HOOKS = ('EMAHook', 'NumClassCheckHook')


def make_deploy_cfg(cfg: Config) -> Config:
    """``cfg`` with ``model`` replaced by the TensorRT detector."""
    deploy = copy.deepcopy(cfg)
    model = cfg.model
    deploy.model = ConfigDict(
        type='TRTSingleStageDetector',
        engine_file='{{ fileDirname }}/' + ENGINE_FILE,
        meta_file='{{ fileDirname }}/' + META_FILE,
        bbox_head=copy.deepcopy(model.bbox_head),
        data_preprocessor=copy.deepcopy(model.get('data_preprocessor')),
        train_cfg=copy.deepcopy(model.get('train_cfg')),
        test_cfg=copy.deepcopy(model.get('test_cfg')),
    )
    deploy.custom_hooks = [
        hook
        for hook in cfg.get('custom_hooks', [])
        if not str(hook.get('type', '')).endswith(_DROP_HOOKS)
    ]
    deploy.load_from = None
    deploy.resume = False
    imports = list(cfg.get('custom_imports', {}).get('imports', []))
    deploy.custom_imports = dict(
        imports=[*imports, 'mmrotate.tensorrt'], allow_failed_imports=False
    )
    return deploy


def write_meta(out_dir: str | Path, meta: dict) -> None:
    """Write ``trt_meta.json``."""
    Path(out_dir, META_FILE).write_text(json.dumps(meta, indent=2))


def init_trt_detector(
    deploy: str | Path | Config, device: str = 'cuda:0'
) -> TRTSingleStageDetector:
    """Build the TensorRT detector from a deploy dir or its ``deploy_cfg.py``.

    The returned model has ``cfg`` and ``dataset_meta`` set, so it works with
    ``mmdet.apis.inference_detector`` and the mmrotate visualizers.
    """
    if isinstance(deploy, str | Path):
        path = Path(deploy)
        cfg = Config.fromfile(path / DEPLOY_CFG if path.is_dir() else path)
    else:
        cfg = deploy
    register_all_modules(init_default_scope=False)
    init_default_scope(cfg.get('default_scope', 'mmrotate'))
    model = MODELS.build(cfg.model)
    if not isinstance(model, TRTSingleStageDetector):
        raise TypeError(f'{type(model).__name__} is not a TRTSingleStageDetector')
    model.cfg = cfg
    return model.to(device).eval()
