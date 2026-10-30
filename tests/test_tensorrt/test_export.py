# Copyright (c) OpenMMLab. All rights reserved.
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
from mmengine.config import Config, ConfigDict

from mmrotate.tensorrt import make_deploy_cfg
from mmrotate.tensorrt.export import (
    check_single_stage,
    export_onnx,
    flatten_outputs,
    infer_input_shape,
    output_names,
    unflatten_outputs,
)
from mmrotate.tensorrt.export_patches import export_mode

REPO = Path(__file__).parents[2]
CFG_ROOT = REPO / 'configs'


def test_flatten_roundtrip() -> None:
    a, b, c = torch.zeros(1), torch.ones(2), torch.full((3,), 2.0)
    nested = ((a, b), (c,), a)
    leaves, spec = flatten_outputs(nested)
    assert spec == [[None, None], [None], None]
    assert len(leaves) == 4
    assert output_names(spec) == ['out_0_0', 'out_0_1', 'out_1_0', 'out_2']
    rebuilt = unflatten_outputs(leaves, spec)
    assert isinstance(rebuilt, tuple) and isinstance(rebuilt[0], tuple)
    assert rebuilt[1][0] is c and rebuilt[2] is a
    with pytest.raises(ValueError):
        unflatten_outputs([*leaves, a], spec)


def test_rejects_other_detectors() -> None:
    with pytest.raises(TypeError, match='SingleStageDetector'):
        check_single_stage(torch.nn.Conv2d(1, 1, 1))


@pytest.mark.parametrize('name', ['comer', 'dcfl', 'vitdet'])
def test_export_mode_patches_and_restores(
    name: str, tiny_detector: Callable[..., torch.nn.Module]
) -> None:
    model = tiny_detector(name)
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        ref, _ = flatten_outputs(model._forward(x))
        before = {n: dict(vars(m)) for n, m in model.named_modules()}
        with export_mode(model, fp32_types=(torch.nn.ConvTranspose2d,)) as patched:
            assert patched
            out, _ = flatten_outputs(model._forward(x))
        after = {n: dict(vars(m)) for n, m in model.named_modules()}
    for r, o in zip(ref, out, strict=True):
        torch.testing.assert_close(o, r, rtol=1e-4, atol=1e-4)
    assert before.keys() == after.keys()
    for key in before:
        assert before[key].keys() == after[key].keys(), key
        for attr in ('backend', 'fused'):
            assert before[key].get(attr) == after[key].get(attr)


@pytest.mark.parametrize('name', ['radio', 'comer', 'dcfl', 'vitdet'])
def test_onnx_matches_torch(
    name: str, tiny_detector: Callable[..., torch.nn.Module], tmp_path: Path
) -> None:
    ort = pytest.importorskip('onnxruntime')
    model = tiny_detector(name)
    x = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        ref, ref_spec = flatten_outputs(model._forward(x))
    with export_mode(model):
        spec, names = export_onnx(model, tmp_path / 'm.onnx', tuple(x.shape))
    assert spec == ref_spec and len(names) == len(ref)
    session = ort.InferenceSession(str(tmp_path / 'm.onnx'), providers=['CPUExecutionProvider'])
    outs = session.run(names, {'inputs': x.numpy()})
    for r, o in zip(ref, outs, strict=True):
        np.testing.assert_allclose(o, r.numpy(), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    'cfg_file,shape',
    [
        ('soda_dinov3/convnext-t-dinov3_rtmdet-p2-6e-soda.yaml', (800, 800)),
        ('cradio/cradiov4-h-comer_rtmdet-3x-dota_ms.yaml', (1024, 1024)),
    ],
)
def test_infer_input_shape(cfg_file: str, shape: tuple[int, int]) -> None:
    assert infer_input_shape(Config.fromfile(CFG_ROOT / cfg_file)) == shape


def test_make_deploy_cfg(tmp_path: Path) -> None:
    cfg = Config.fromfile(CFG_ROOT / 'cradio/cradiov4-h-comer_rtmdet-3x-dota_ms.yaml')
    deploy = make_deploy_cfg(cfg)
    assert deploy.model.type == 'TRTSingleStageDetector'
    assert deploy.model.bbox_head == cfg.model.bbox_head
    assert deploy.model.test_cfg == cfg.model.test_cfg
    assert 'backbone' not in deploy.model
    hooks = [h['type'] for h in deploy.custom_hooks]
    assert not any(h.endswith(('EMAHook', 'NumClassCheckHook')) for h in hooks)
    assert 'mmrotate.tensorrt' in deploy.custom_imports['imports']
    assert deploy.load_from is None
    # The dumped config resolves the engine path relative to its own folder.
    deploy.dump(str(tmp_path / 'deploy_cfg.py'))
    loaded = Config.fromfile(tmp_path / 'deploy_cfg.py')
    assert loaded.model.engine_file == f'{tmp_path}/end2end.engine'


def test_calibration_loader(tmp_path: Path) -> None:
    from mmrotate.tensorrt.calibration import build_input_loader, iter_batches
    from mmrotate.utils import register_all_modules

    register_all_modules()
    data_root = tmp_path / 'proxy'
    subprocess.run(
        [
            sys.executable,
            str(REPO / 'tools/data/dota/make_proxy_dota.py'),
            '--out',
            str(data_root),
            '--num-train',
            '3',
            '--num-test',
            '1',
            '--size',
            '256',
        ],
        check=True,
    )
    cfg = Config.fromfile(REPO / 'configs/cradio/cradiov4-h-comer_rtmdet-proxy.yaml')
    for split in ('train', 'val', 'test'):
        cfg[f'{split}_dataloader'].dataset.data_root = f'{data_root}/'
    cfg.test_dataloader.dataset.pipeline = [
        dict(type='mmdet.LoadImageFromFile'),
        dict(type='mmdet.Resize', scale=(256, 256), keep_ratio=True),
        dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
        dict(type='mmdet.PackDetInputs'),
    ]
    from mmrotate.registry import MODELS

    preprocessor = MODELS.build(ConfigDict(cfg.model.data_preprocessor))
    loader = build_input_loader(cfg, 'train', num_workers=0)
    batches = list(iter_batches(loader, preprocessor, (2, 3, 320, 320), num_images=3))
    assert [b.shape for b, _ in batches] == [(2, 3, 320, 320), (1, 3, 320, 320)]
    assert sum(len(s) for _, s in batches) == 3
