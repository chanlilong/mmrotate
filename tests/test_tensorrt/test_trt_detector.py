# Copyright (c) OpenMMLab. All rights reserved.
"""End-to-end TensorRT tests: export, build, save, reload, predict."""

import copy
import json
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch
from mmengine.config import Config
from trt_test_utils import CONFIGS, tiny_model_cfg

from mmrotate.tensorrt import init_trt_detector, make_deploy_cfg
from mmrotate.tensorrt.deploy import DEPLOY_CFG, ENGINE_FILE, write_meta
from mmrotate.tensorrt.detector import TRTSingleStageDetector
from mmrotate.tensorrt.export import export_onnx, flatten_outputs
from mmrotate.tensorrt.export_patches import export_mode
from mmrotate.tensorrt.verify import detection_agreement, instances_to_dict, raw_output_errors
from mmrotate.testing import demo_mm_inputs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
trt = pytest.importorskip('tensorrt')
HW = (128, 128)


def _convert(
    model: torch.nn.Module,
    name: str,
    out_dir: Path,
    batch_size: int,
    precision: str = 'fp32',
    calib: list[torch.Tensor] | None = None,
) -> None:
    """What tools/tensorrt/convert.py does, for a model built in the test."""
    from mmrotate.tensorrt import precision as prec
    from mmrotate.tensorrt.builder import build_engine

    shape = (batch_size, 3, *HW)
    with export_mode(model, prec.fp32_module_types(precision)):
        if precision in ('int8', 'fp8'):
            model = prec.quantize(model, precision, calib)
        qdq = prec.int8_export_mode(model) if precision == 'int8' else nullcontext()
        with qdq:
            spec, names = export_onnx(
                model,
                out_dir / 'm.onnx',
                shape,
                autocast_dtype=prec.autocast_dtype(precision),
            )
    build_engine(out_dir / 'm.onnx', out_dir / ENGINE_FILE, tf32=precision != 'fp32')
    write_meta(
        out_dir,
        dict(input_shape=list(shape), output_names=names, output_spec=spec, precision=precision),
    )
    cfg = Config(dict(model=tiny_model_cfg(name), default_scope='mmrotate'))
    make_deploy_cfg(cfg).dump(str(out_dir / DEPLOY_CFG))


def _inputs(model: torch.nn.Module, num: int) -> dict:
    packed = demo_mm_inputs(num, [[3, *HW]] * num, use_box_type=True)
    return model.data_preprocessor(packed, False)


@pytest.fixture(autouse=True)
def _no_tf32() -> None:
    torch.backends.cudnn.allow_tf32 = False


@pytest.mark.parametrize('name', list(CONFIGS))
def test_fp32_engine_matches_torch(
    name: str, tiny_detector: Callable[..., torch.nn.Module], tmp_path: Path
) -> None:
    model = tiny_detector(name, 'cuda')
    _convert(model, name, tmp_path, batch_size=2)
    trt_model = init_trt_detector(tmp_path, 'cuda')
    assert isinstance(trt_model, TRTSingleStageDetector)
    data = _inputs(model, 3)  # 3 images through a batch-2 engine: chunk + pad
    with torch.no_grad():
        errors = raw_output_errors(
            model._forward(data['inputs']), trt_model._forward(data['inputs'])
        )
        ref = model.predict(data['inputs'], copy.deepcopy(data['data_samples']))
        out = trt_model.predict(data['inputs'], copy.deepcopy(data['data_samples']))
    assert errors['max_rel_err'] < 1e-3, errors
    assert len(out) == 3
    for r, o in zip(ref, out, strict=True):
        assert o.pred_instances.bboxes.shape[-1] == 5
        assert set(o.pred_instances.keys()) == set(r.pred_instances.keys())
    agreement = detection_agreement(
        [instances_to_dict(r.pred_instances) for r in ref],
        [instances_to_dict(o.pred_instances) for o in out],
        score_thr=0.05,
    )
    # Rotated NMS can flip on 1e-6 perturbations, so not every box must match.
    assert agreement['torch_dets'] > 0, agreement
    assert agreement['recall'] > 0.9 and agreement['mean_score_diff'] < 1e-3, agreement


def test_padding_and_errors(tiny_detector: Callable[..., torch.nn.Module], tmp_path: Path) -> None:
    model = tiny_detector('radio', 'cuda')
    _convert(model, 'radio', tmp_path, batch_size=1)
    trt_model = init_trt_detector(tmp_path, 'cuda')
    small = torch.randn(1, 3, 96, 128, device='cuda')
    with pytest.warns(UserWarning, match='Padding inputs'):
        outs = trt_model._forward(small)
    leaves, _ = flatten_outputs(outs)
    assert leaves[0].shape[-2:] == (HW[0] // 8, HW[1] // 8)
    with pytest.raises(ValueError, match='does not fit'):
        trt_model._forward(torch.randn(1, 3, 160, 128, device='cuda'))
    with pytest.raises(NotImplementedError):
        trt_model.loss(small, [])


@pytest.mark.parametrize('precision', ['fp16', 'bf16', 'int8', 'fp8'])
def test_reduced_precision(
    precision: str, tiny_detector: Callable[..., torch.nn.Module], tmp_path: Path
) -> None:
    if precision == 'fp8' and torch.cuda.get_device_capability() < (8, 9):
        pytest.skip('FP8 needs compute capability 8.9+')
    if precision in ('int8', 'fp8'):
        pytest.importorskip('modelopt.torch.quantization')
    model = tiny_detector('comer', 'cuda')
    data = _inputs(model, 2)
    with torch.no_grad():
        ref = model._forward(data['inputs'])
    calib = [_inputs(model, 2)['inputs'] for _ in range(2)]
    _convert(model, 'comer', tmp_path, batch_size=2, precision=precision, calib=calib)
    meta = json.loads((tmp_path / 'trt_meta.json').read_text())
    assert meta['precision'] == precision
    trt_model = init_trt_detector(tmp_path, 'cuda')
    with torch.no_grad():
        out = trt_model._forward(data['inputs'])
    leaves, _ = flatten_outputs(out)
    assert all(torch.isfinite(x).all() for x in leaves)
    errors = raw_output_errors(ref, out)
    # Relative to the largest output: per-output errors blow up on near-zero maps.
    assert errors['max_abs_err'] < 0.05 * errors['max_abs_ref'], errors


def test_inference_detector(tiny_detector: Callable[..., torch.nn.Module], tmp_path: Path) -> None:
    import numpy as np
    from mmdet.apis import inference_detector

    model = tiny_detector('radio', 'cuda')
    _convert(model, 'radio', tmp_path, batch_size=1)
    cfg = Config.fromfile(tmp_path / DEPLOY_CFG)
    cfg.test_dataloader = dict(
        dataset=dict(
            pipeline=[
                dict(type='mmdet.LoadImageFromFile'),
                dict(type='mmdet.Resize', scale=HW, keep_ratio=True),
                dict(
                    type='mmdet.PackDetInputs',
                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'),
                ),
            ]
        )
    )
    trt_model = init_trt_detector(cfg, 'cuda')
    image = np.random.default_rng(0).integers(0, 255, (100, 120, 3), dtype=np.uint8)
    result = inference_detector(trt_model, image)
    assert result.pred_instances.bboxes.shape[-1] == 5
    assert result.ori_shape == (100, 120)
