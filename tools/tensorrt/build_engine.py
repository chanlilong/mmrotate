# Copyright (c) OpenMMLab. All rights reserved.
"""Rebuild the engine of a deploy directory for the GPU this runs on.

An engine runs only on the GPU architecture it was built for (unless built
with ``--hardware-compat``). The ONNX graph does not depend on the GPU: it
already holds the precision, the INT8 / FP8 Q/DQ nodes and the plugin nodes.
So convert once with ``--keep-onnx`` and build an engine per target GPU, on
that GPU, without the checkpoint, the dataset or ModelOpt::

    python tools/tensorrt/convert.py CONFIG CKPT --out-dir work_dirs/trt/m_fp16 --keep-onnx
    # on the target machine
    python tools/tensorrt/build_engine.py work_dirs/trt/m_fp16 --out-dir work_dirs/trt/m_fp16_ada

Needs TensorRT, and Triton plus this package when the graph has ViT-CoMer
deformable-attention plugin nodes. See ``mmrotate/tensorrt/TENSORRT.md``.
"""

import argparse
import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from mmrotate.tensorrt import engine_report
from mmrotate.tensorrt.builder import HARDWARE_COMPAT, build_engine
from mmrotate.tensorrt.deploy import ENGINE_FILE, ONNX_FILE, init_trt_detector, write_meta
from mmrotate.tensorrt.detector import META_FILE
from mmrotate.tensorrt.runtime import TRTEngine

logger = logging.getLogger('mmrotate.tensorrt')
# Files of a deploy directory that are not copied: rebuilt, specific to the build GPU, or
# results measured on the source engine (logs, profiles, hidden marker files of scripts).
_SKIP = (ENGINE_FILE, ENGINE_FILE + '.bak', 'timing.cache', 'profile.json')
_SKIP_SUFFIXES = ('.log',)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Rebuild a deploy directory for this GPU')
    parser.add_argument('deploy_dir', help='deploy directory converted with --keep-onnx')
    parser.add_argument(
        '--out-dir', help='new deploy directory (default: rebuild in place, keeping a backup)'
    )
    parser.add_argument('--hardware-compat', default='none', choices=tuple(HARDWARE_COMPAT))
    parser.add_argument('--opt-level', type=int, help='default: the one of the source build')
    parser.add_argument('--workspace-gb', type=float, help='TensorRT workspace limit')
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def _copy_deploy_dir(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        skip = f.name in _SKIP or f.name.startswith('.') or f.name.endswith(_SKIP_SUFFIXES)
        if f.is_file() and not skip:
            shutil.copy2(f, dst / f.name)


def _check_outputs(out_dir: Path, device: str) -> bool:
    """Run the engine once on a random batch; True when every output is finite."""
    model = init_trt_detector(out_dir, device)
    shape = model.meta['input_shape']
    with torch.no_grad():
        outs = model.engine(torch.randn(shape, device=device))
    return all(bool(torch.isfinite(t.float()).all()) for t in outs)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s: %(message)s')
    src = Path(args.deploy_dir)
    if not (src / ONNX_FILE).exists():
        raise SystemExit(f'{src} has no {ONNX_FILE}: convert with --keep-onnx')
    meta = json.loads((src / META_FILE).read_text())
    if meta.get('precision') == 'fp8' and args.hardware_compat == 'ampere+':
        raise SystemExit('FP8 needs compute capability 8.9+: an fp8 engine cannot run on Ampere')
    props = torch.cuda.get_device_properties(args.device)
    cc = f'{props.major}.{props.minor}'
    if meta.get('precision') == 'fp8' and (props.major, props.minor) < (8, 9):
        raise SystemExit(f'FP8 needs compute capability 8.9+, this GPU is {cc}')

    out_dir = Path(args.out_dir) if args.out_dir else src
    if out_dir.resolve() == src.resolve():
        backup = src / (ENGINE_FILE + '.bak')
        if (src / ENGINE_FILE).exists():
            shutil.copy2(src / ENGINE_FILE, backup)
            logger.info('Previous engine kept as %s', backup)
    else:
        _copy_deploy_dir(src, out_dir)

    build = meta.get('build', {})
    opt_level = args.opt_level if args.opt_level is not None else build.get('opt_level', 3)
    start = time.perf_counter()
    torch.cuda.set_device(args.device)
    build_engine(
        out_dir / ONNX_FILE,
        out_dir / ENGINE_FILE,
        workspace_gb=args.workspace_gb,
        opt_level=opt_level,
        timing_cache=out_dir / 'timing.cache',
        tf32=meta.get('precision') != 'fp32',
        plugins=meta.get('plugins'),
        hardware_compat=args.hardware_compat,
    )
    logger.info('TensorRT build took %.1fs', time.perf_counter() - start)

    import tensorrt as trt

    engine = TRTEngine(out_dir / ENGINE_FILE, args.device)
    fusion = engine_report.fusion_summary(engine_report.layer_info(engine))
    del engine
    previous = meta.get('fusion') or {}
    for key in ('attention_calls', 'attention_fusable'):
        if key in previous:
            fusion[key] = previous[key]
    if 'attention_fusable' in fusion:
        fusion['attention_unfused'] = max(fusion['attention_fusable'] - fusion['mha'], 0)
        if fusion['attention_unfused']:
            logger.warning(
                'TensorRT left %d attention call(s) unfused', fusion['attention_unfused']
            )
    meta['fusion'] = fusion
    meta['build'] = dict(
        build,
        tensorrt=trt.__version__,
        gpu=props.name,
        compute_capability=cc,
        hardware_compat=args.hardware_compat,
        opt_level=opt_level,
        date=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        rebuilt_from=dict(
            deploy_dir=str(src.resolve()),
            gpu=build.get('gpu'),
            compute_capability=build.get('compute_capability'),
        ),
    )
    # The source verification was measured on another engine.
    if 'verify' in meta:
        meta['verify_source_engine'] = meta.pop('verify')
    write_meta(out_dir, meta)
    finite = _check_outputs(out_dir, args.device)
    logger.info(
        'Engine: %d layers, %d mha, %d plugin layers; outputs on a random batch %s',
        fusion['layers'], fusion['mha'], fusion['plugins'], 'finite' if finite else 'NOT FINITE',
    )  # fmt: skip
    if not finite:
        raise RuntimeError(f'The rebuilt engine returns NaN / inf ({out_dir})')
    logger.info('Wrote %s', out_dir)


if __name__ == '__main__':
    main()
