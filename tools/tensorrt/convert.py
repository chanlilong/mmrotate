# Copyright (c) OpenMMLab. All rights reserved.
"""Convert an mmrotate single-stage detector to a TensorRT engine.

Writes a deploy directory (engine, ``trt_meta.json``, ``deploy_cfg.py``)
that ``mmrotate.tensorrt.init_trt_detector`` / ``tools/tensorrt/test_trt.py``
load as a ``TRTSingleStageDetector``::

    python tools/tensorrt/convert.py CONFIG CHECKPOINT --out-dir DIR \\
        [--precision fp32|fp16|bf16|int8|fp8] [--calib-split train --calib-num 64] \\
        [--verify 16] [--cfg-options test_dataloader.dataset.data_root=...]

See ``mmrotate/tensorrt/TENSORRT.md``.
"""

import argparse
import copy
import hashlib
import json
import logging
import shutil
import time
import warnings
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import torch
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope

from mmrotate.tensorrt import precision as prec
from mmrotate.tensorrt.builder import build_engine
from mmrotate.tensorrt.calibration import build_input_loader, iter_batches
from mmrotate.tensorrt.deploy import (
    DEPLOY_CFG,
    ENGINE_FILE,
    ONNX_FILE,
    init_trt_detector,
    make_deploy_cfg,
    write_meta,
)
from mmrotate.tensorrt.export import INPUT_NAME, build_detector, export_onnx, infer_input_shape
from mmrotate.tensorrt.export_patches import export_mode
from mmrotate.tensorrt.verify import (
    detection_agreement,
    instances_to_dict,
    raw_output_errors,
    to_cpu,
)

logger = logging.getLogger('mmrotate.tensorrt')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Convert a detector to TensorRT')
    parser.add_argument('config', help='model config')
    parser.add_argument('checkpoint', help='detector checkpoint')
    parser.add_argument('--out-dir', required=True, help='deploy directory to write')
    parser.add_argument('--precision', default='fp16', choices=prec.PRECISIONS)
    parser.add_argument(
        '--input-shape',
        type=int,
        nargs=2,
        metavar=('H', 'W'),
        help='static network input size (default: from the test pipeline)',
    )
    parser.add_argument('--batch-size', type=int, default=1, help='engine batch size')
    parser.add_argument(
        '--calib-split', default='train', choices=('train', 'val', 'test'), help='int8/fp8 data'
    )
    parser.add_argument('--calib-num', type=int, default=64, help='calibration images')
    parser.add_argument('--calib-method', default='max', choices=prec.CALIB_METHODS)
    parser.add_argument('--calib-percentile', type=float, default=99.99)
    parser.add_argument(
        '--quant-exclude',
        nargs='*',
        default=list(prec.DEFAULT_QUANT_EXCLUDE),
        help='module name patterns (fnmatch) kept in high precision',
    )
    parser.add_argument('--opset', type=int, default=19)
    parser.add_argument('--opt-level', type=int, default=3, help='TensorRT optimization level')
    parser.add_argument('--workspace-gb', type=float, help='TensorRT workspace limit')
    parser.add_argument('--keep-onnx', action='store_true', help='keep the ONNX graph')
    parser.add_argument(
        '--verify', type=int, default=8, metavar='N', help='compare with torch on N images'
    )
    parser.add_argument('--verify-split', default='val', choices=('train', 'val', 'test'))
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    return parser.parse_args()


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _versions() -> dict:
    import tensorrt as trt

    try:
        import modelopt

        modelopt_version = modelopt.__version__
    except ImportError:
        modelopt_version = None
    return dict(tensorrt=trt.__version__, torch=torch.__version__, modelopt=modelopt_version)


def _collect(
    cfg: Config,
    model: torch.nn.Module,
    split: str,
    num: int,
    shape: tuple[int, int, int, int],
    args: argparse.Namespace,
    shuffle: bool,
) -> list[tuple[torch.Tensor, list]]:
    """Preprocessed ``(inputs, data_samples)`` batches, inputs on the CPU."""
    loader = build_input_loader(cfg, split, shuffle=shuffle, num_workers=args.num_workers)
    batches = [
        (inputs.cpu(), samples)
        for inputs, samples in iter_batches(loader, model.data_preprocessor, shape, num)
    ]
    logger.info('Loaded %d %s images', sum(len(s) for _, s in batches), split)
    return batches


@torch.no_grad()
def _torch_reference(model: torch.nn.Module, batches: list, device: str) -> tuple[list, list]:
    raw, dets = [], []
    for inputs, samples in batches:
        inputs = inputs.to(device)
        raw.append(to_cpu(model._forward(inputs)))
        results = model.predict(inputs, copy.deepcopy(samples))
        dets += [instances_to_dict(r.pred_instances) for r in results]
    return raw, dets


@torch.no_grad()
def _verify(out_dir: Path, batches: list, ref_raw: list, ref_dets: list, device: str) -> dict:
    trt_model = init_trt_detector(out_dir, device)
    errors, dets = [], []
    for (inputs, samples), raw in zip(batches, ref_raw, strict=True):
        inputs = inputs.to(device)
        errors.append(raw_output_errors(raw, trt_model._forward(inputs)))
        results = trt_model.predict(inputs, copy.deepcopy(samples))
        dets += [instances_to_dict(r.pred_instances) for r in results]
    report = dict(
        images=len(ref_dets),
        finite=all(e['finite'] for e in errors),
        max_abs_err=max(e['max_abs_err'] for e in errors),
        max_rel_err=max(e['max_rel_err'] for e in errors),
    )
    report['detections'] = [
        detection_agreement(ref_dets, dets, score_thr=thr) for thr in (0.05, 0.3)
    ]
    return report


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s: %(message)s')
    warnings.filterwarnings('ignore', message='.*building extension.*')
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    init_default_scope(cfg.get('default_scope', 'mmrotate'))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    # fp32 reference without TF32 convs / matmuls (the fp32 engine has none either).
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    hw = tuple(args.input_shape) if args.input_shape else infer_input_shape(cfg)
    shape = (args.batch_size, 3, *hw)
    logger.info('Engine input %s, precision %s', shape, args.precision)
    model = build_detector(cfg, args.checkpoint, device)
    dataset_meta = getattr(model, 'dataset_meta', None) or cfg.test_dataloader.dataset.get(
        'metainfo'
    )

    verify_batches, ref_raw, ref_dets = [], [], []
    if args.verify > 0:
        verify_batches = _collect(cfg, model, args.verify_split, args.verify, shape, args, False)
        ref_raw, ref_dets = _torch_reference(model, verify_batches, device)

    calib = None
    quantized = args.precision in ('int8', 'fp8')
    if quantized:
        calib_batches = _collect(
            cfg, model, args.calib_split, args.calib_num, shape, args, shuffle=True
        )
        calib = dict(
            split=args.calib_split,
            images=sum(len(s) for _, s in calib_batches),
            method=args.calib_method,
            exclude=args.quant_exclude,
        )

    onnx_dir = out_dir / 'onnx'
    onnx_file = onnx_dir / ONNX_FILE
    start = time.perf_counter()
    with export_mode(model, prec.fp32_module_types(args.precision)) as patched:
        if patched:
            logger.info('Export patches applied to %d modules', len(patched))
        if quantized:
            model = prec.quantize(
                model,
                args.precision,
                [x for x, _ in calib_batches],
                method=args.calib_method,
                exclude=args.quant_exclude,
                percentile=args.calib_percentile,
            )
        qdq = prec.int8_export_mode(model) if args.precision == 'int8' else nullcontext()
        with qdq:
            spec, names = export_onnx(
                model,
                onnx_file,
                shape,
                opset=args.opset,
                autocast_dtype=prec.autocast_dtype(args.precision),
            )
    logger.info('ONNX export took %.1fs', time.perf_counter() - start)
    del model
    torch.cuda.empty_cache()

    start = time.perf_counter()
    build_engine(
        onnx_file,
        out_dir / ENGINE_FILE,
        workspace_gb=args.workspace_gb,
        opt_level=args.opt_level,
        timing_cache=out_dir / 'timing.cache',
        tf32=args.precision != 'fp32',
    )
    logger.info('TensorRT build took %.1fs', time.perf_counter() - start)
    if args.keep_onnx:
        for f in onnx_dir.iterdir():
            shutil.move(str(f), out_dir / f.name)
    shutil.rmtree(onnx_dir)

    props = torch.cuda.get_device_properties(device)
    meta = dict(
        format_version=1,
        input_name=INPUT_NAME,
        input_shape=list(shape),
        output_names=names,
        output_spec=spec,
        precision=args.precision,
        calibration=calib,
        dataset_meta=dataset_meta,
        source=dict(
            config=str(Path(args.config).resolve()),
            checkpoint=str(Path(args.checkpoint).resolve()),
            checkpoint_sha256=_sha256(args.checkpoint),
        ),
        build=dict(
            **_versions(),
            gpu=props.name,
            compute_capability=f'{props.major}.{props.minor}',
            date=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        ),
    )
    write_meta(out_dir, meta)
    make_deploy_cfg(cfg).dump(str(out_dir / DEPLOY_CFG))

    if verify_batches:
        meta['verify'] = _verify(out_dir, verify_batches, ref_raw, ref_dets, device)
        write_meta(out_dir, meta)
        logger.info('Verification against torch fp32:\n%s', json.dumps(meta['verify'], indent=2))
        if not meta['verify']['finite']:
            logger.error(
                'The engine returns NaN / inf: the model overflows %s; try bf16 or fp32',
                args.precision,
            )
    logger.info('Wrote %s', out_dir)


if __name__ == '__main__':
    main()
