# Copyright (c) OpenMMLab. All rights reserved.
"""Convert an mmrotate single-stage detector to a TensorRT engine.

Writes a deploy directory (engine, ``trt_meta.json``, ``deploy_cfg.py``)
that ``mmrotate.tensorrt.init_trt_detector`` / ``tools/tensorrt/test_trt.py``
load as a ``TRTSingleStageDetector``::

    python tools/tensorrt/convert.py CONFIG CHECKPOINT --out-dir DIR \\
        [--precision fp32|fp16|bf16|int8|fp8] [--calib-split train --calib-num 64] \\
        [--attention auto|fused|fused-fp32acc|legacy] [--msda auto|aot|jit|gridsample] \\
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

from mmrotate.tensorrt import engine_report
from mmrotate.tensorrt import precision as prec
from mmrotate.tensorrt.attention import ATTENTION_MODES, resolve_attention
from mmrotate.tensorrt.builder import HARDWARE_COMPAT, build_engine
from mmrotate.tensorrt.calibration import build_input_loader, iter_batches
from mmrotate.tensorrt.deploy import (
    DEPLOY_CFG,
    ENGINE_FILE,
    ONNX_FILE,
    init_trt_detector,
    make_deploy_cfg,
    write_meta,
)
from mmrotate.tensorrt.export import (
    INPUT_NAME,
    build_detector,
    export_onnx,
    flatten_outputs,
    infer_input_shape,
)
from mmrotate.tensorrt.export_patches import export_mode, has_msda
from mmrotate.tensorrt.quant.sensitivity import output_noise
from mmrotate.tensorrt.runtime import TRTEngine
from mmrotate.tensorrt.verify import (
    detection_agreement,
    instances_to_dict,
    raw_output_errors,
    to_cpu,
)

logger = logging.getLogger('mmrotate.tensorrt')
# A quantized engine may be this much noisier than the fake-quantized torch model (fp16
# layers, kernel differences) before the conversion is reported as failed.
QUANT_NOISE_FACTOR = 4.0
# Largest noise of a quantized engine measured against the fake-quantized torch model itself.
# Engines that reproduce it measured <= 0.13 (INT8 / FP8 engines of the DINOv3 models on an
# RTX PRO 6000 Blackwell); engines TensorRT compiled wrongly >= 0.5 (e.g. INT8 ConvNeXt-T built
# on Blackwell without --quant-fp32-layers: 5 mAP instead of 37) while only 2-3x noisier than
# the quantized model's own noise, which QUANT_NOISE_FACTOR lets through.
QUANT_ENGINE_NOISE_MAX = 0.3


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
    parser.add_argument(
        '--quant-algo',
        default='',
        help='comma-separated INT8 refinement after calibration: awq-scale (per-channel input '
        'scale search), awq-clip (weight clip search), gptq (error-compensating rounding)',
    )
    parser.add_argument(
        '--quant-auto-exclude',
        type=float,
        metavar='BUDGET',
        help='INT8: keep the most sensitive layers in fp16 so that the summed single-layer '
        'output noise (noise / signal power) stays below BUDGET, e.g. 0.01',
    )
    parser.add_argument(
        '--quant-fp32-layers',
        action='store_true',
        help='INT8: type the quantized layers fp32 instead of fp16 (see int8_export_mode)',
    )
    parser.add_argument(
        '--quant-refine-batches', type=int, default=2, help='calibration batches for the above'
    )
    parser.add_argument(
        '--attention',
        default='auto',
        choices=ATTENTION_MODES,
        help='export form of ViT attention: auto picks, per attention call, a form TensorRT '
        'fuses (fp32 accumulation where fp16 logits would overflow); legacy is the unfused '
        'fp32-logit form (always used for fp32 engines)',
    )
    parser.add_argument(
        '--require-fused-attention',
        action='store_true',
        help='fail if TensorRT did not fuse every attention call into an mha layer',
    )
    parser.add_argument(
        '--msda',
        default='auto',
        choices=('auto', 'aot', 'jit', 'gridsample'),
        help='ViT-CoMer deformable attention: the Triton kernel as a TensorRT plugin embedded '
        'in the engine (aot) or called back in Python (jit), or grid_sample ONNX ops; auto '
        'tries aot, then jit',
    )
    parser.add_argument(
        '--hardware-compat',
        default='none',
        choices=tuple(HARDWARE_COMPAT),
        help="GPUs the engine runs on: none (the build GPU's compute capability), ampere+ "
        '(every Ampere / Ada / Hopper / Blackwell GPU, at some speed cost) or same-cc (GPUs '
        "with the build GPU's compute capability)",
    )
    parser.add_argument('--opset', type=int, default=19)
    parser.add_argument(
        '--opt-level',
        type=int,
        help='TensorRT optimization level (default: 1 for fp32, which reproduces torch where '
        'level 3 tactics did not, 3 otherwise)',
    )
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


def _leaves(raw: list) -> list[list[torch.Tensor]]:
    return [[leaf.float().cpu() for leaf in flatten_outputs(outs)[0]] for outs in raw]


@torch.no_grad()
def _verify(
    out_dir: Path,
    batches: list,
    ref_raw: list,
    ref_dets: list,
    device: str,
    quant_raw: list | None = None,
) -> dict:
    """Compare the engine with the torch model on ``batches``.

    ``noise`` is the error power of the engine outputs relative to the output
    power of the torch fp32 model. For quantized engines,
    ``noise_torch_quantized`` is the same for the fake-quantized torch model
    (what quantization is expected to cost) and ``noise_vs_torch_quantized``
    measures the engine against that model (what TensorRT adds on top).
    """
    trt_model = init_trt_detector(out_dir, device)
    errors, dets, raw = [], [], []
    for (inputs, samples), ref in zip(batches, ref_raw, strict=True):
        inputs = inputs.to(device)
        outs = trt_model._forward(inputs)
        raw.append(to_cpu(outs))
        errors.append(raw_output_errors(ref, outs))
        results = trt_model.predict(inputs, copy.deepcopy(samples))
        dets += [instances_to_dict(r.pred_instances) for r in results]
    report = dict(
        images=len(ref_dets),
        finite=all(e['finite'] for e in errors),
        max_abs_err=max(e['max_abs_err'] for e in errors),
        max_rel_err=max(e['max_rel_err'] for e in errors),
        noise=output_noise(_leaves(raw), _leaves(ref_raw)),
    )
    if quant_raw is not None:
        report['noise_torch_quantized'] = output_noise(_leaves(quant_raw), _leaves(ref_raw))
        report['noise_vs_torch_quantized'] = output_noise(_leaves(raw), _leaves(quant_raw))
    report['detections'] = [
        detection_agreement(ref_dets, dets, score_thr=thr) for thr in (0.05, 0.3)
    ]
    return report


def _fusion_report(engine_file: Path, attention_plan: list[str], device: str) -> dict:
    """Fused attention / plugin layer counts of the engine vs the export plan."""
    engine = TRTEngine(engine_file, device)
    report = engine_report.fusion_summary(engine_report.layer_info(engine))
    del engine
    fusable = sum(form != 'legacy' for form in attention_plan)
    report['attention_calls'] = len(attention_plan)
    report['attention_fusable'] = fusable
    report['attention_unfused'] = max(fusable - report['mha'], 0)
    logger.info(
        'Engine: %d layers, %d fused attention (mha) for %d attention calls, %d plugin layers',
        report['layers'], report['mha'], len(attention_plan), report['plugins'],
    )  # fmt: skip
    if report['attention_unfused']:
        logger.warning(
            'TensorRT left %d attention call(s) unfused; they run as separate matmuls',
            report['attention_unfused'],
        )
    return report


def main() -> None:
    args = parse_args()
    if args.opt_level is None:
        args.opt_level = 1 if args.precision == 'fp32' else 3
    if args.precision == 'fp8' and args.hardware_compat == 'ampere+':
        raise SystemExit('FP8 needs compute capability 8.9+: an fp8 engine cannot run on Ampere')
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
    calib_batches = []
    quant_raw = None
    quantized = args.precision in ('int8', 'fp8')
    quant_algos = [a for a in args.quant_algo.split(',') if a]
    if quantized:
        calib_batches = _collect(
            cfg, model, args.calib_split, args.calib_num, shape, args, shuffle=True
        )
        calib = dict(
            split=args.calib_split,
            images=sum(len(s) for _, s in calib_batches),
            method=args.calib_method,
            exclude=args.quant_exclude,
            algos=quant_algos,
            auto_exclude=args.quant_auto_exclude,
            refine={},
        )

    low_dtype = prec.autocast_dtype(args.precision)
    attention_mode = 'legacy' if low_dtype is None else args.attention
    probe_batches = [x for x, _ in (verify_batches or calib_batches)]
    if not probe_batches:
        probe = _collect(cfg, model, args.verify_split, 2, shape, args, False)
        probe_batches = [x for x, _ in probe]
    attention_plan, attention_stats = resolve_attention(
        attention_mode, model, [b[:2] for b in probe_batches[:1]], low_dtype
    )
    if quantized and any(form != 'legacy' for form in attention_plan):
        if args.quant_auto_exclude is not None or args.quant_fp32_layers:
            raise SystemExit(
                '--quant-auto-exclude / --quant-fp32-layers do not work with fused attention: '
                'the fp32-typed quantized layers leave TensorRT without an attention kernel and '
                'the engine fails to build. Use --quant-algo awq-scale alone for ViT models.'
            )
        args.quant_exclude = [*args.quant_exclude, *prec.FUSED_ATTENTION_EXCLUDE]
        calib['exclude'] = args.quant_exclude
        logger.info('Fused attention: %s kept unquantized', prec.FUSED_ATTENTION_EXCLUDE)
    msda = args.msda
    if not has_msda(model):
        msda = 'gridsample'
    elif msda == 'auto':
        msda = 'aot'
    plugins = None if msda == 'gridsample' else msda

    onnx_dir = out_dir / 'onnx'
    onnx_file = onnx_dir / ONNX_FILE
    start = time.perf_counter()
    with export_mode(
        model,
        prec.fp32_module_types(args.precision),
        attention=attention_plan or 'legacy',
        msda='gridsample' if plugins is None else 'plugin',
    ) as patched:
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
                algos=quant_algos,
                exclude_budget=args.quant_auto_exclude,
                refine_batches=args.quant_refine_batches,
                report=calib['refine'],
            )
            with torch.no_grad():
                quant_raw = [to_cpu(model._forward(x.to(device))) for x, _ in verify_batches]
        # Partially quantized graphs need fp32-typed quantized layers, see int8_export_mode.
        fp32_layers = args.quant_fp32_layers or args.quant_auto_exclude is not None
        qdq = (
            prec.int8_export_mode(model, fp32_layers) if args.precision == 'int8' else nullcontext()
        )
        with qdq:
            spec, names = export_onnx(
                model,
                onnx_file,
                shape,
                opset=args.opset,
                autocast_dtype=low_dtype,
            )
    logger.info('ONNX export took %.1fs', time.perf_counter() - start)
    del model
    torch.cuda.empty_cache()

    start = time.perf_counter()
    while True:
        try:
            build_engine(
                onnx_file,
                out_dir / ENGINE_FILE,
                workspace_gb=args.workspace_gb,
                opt_level=args.opt_level,
                timing_cache=out_dir / 'timing.cache',
                tf32=args.precision != 'fp32',
                plugins=plugins,
                hardware_compat=args.hardware_compat,
            )
            break
        except Exception:
            if not (args.msda == 'auto' and plugins == 'aot'):
                raise
            logger.exception('Building with the embedded (aot) MSDA plugin failed, trying jit')
            plugins = 'jit'
    logger.info('TensorRT build took %.1fs', time.perf_counter() - start)
    fusion = _fusion_report(out_dir / ENGINE_FILE, attention_plan, device)
    if fusion['attention_unfused'] and args.require_fused_attention:
        raise RuntimeError(
            f'TensorRT fused {fusion["mha"]} of {fusion["attention_fusable"]} attention calls'
        )
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
        attention=attention_stats,
        plugins=plugins,
        fusion=fusion,
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
            hardware_compat=args.hardware_compat,
            opt_level=args.opt_level,
            date=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        ),
    )
    write_meta(out_dir, meta)
    make_deploy_cfg(cfg).dump(str(out_dir / DEPLOY_CFG))

    if verify_batches:
        verify = _verify(out_dir, verify_batches, ref_raw, ref_dets, device, quant_raw)
        meta['verify'] = verify
        write_meta(out_dir, meta)
        logger.info('Verification against torch fp32:\n%s', json.dumps(verify, indent=2))
        if quant_raw and (
            verify['noise'] > QUANT_NOISE_FACTOR * verify['noise_torch_quantized']
            or verify['noise_vs_torch_quantized'] > QUANT_ENGINE_NOISE_MAX
        ):
            raise RuntimeError(
                f'The engine does not reproduce the quantized torch model: output noise '
                f'{verify["noise"]:.3g} against {verify["noise_torch_quantized"]:.3g} expected '
                f'from quantization, {verify["noise_vs_torch_quantized"]:.3g} against the '
                f'quantized model itself. TensorRT built a different graph (seen when an '
                f'unquantized fp16 convolution feeds a quantized layer); try '
                f'--quant-algo awq-scale with --quant-fp32-layers, or other --quant-exclude '
                f'patterns. The deploy directory is kept for inspection.'
            )
        if not meta['verify']['finite']:
            logger.error(
                'The engine returns NaN / inf: the model overflows %s; try bf16 or fp32',
                args.precision,
            )
    logger.info('Wrote %s', out_dir)


if __name__ == '__main__':
    main()
