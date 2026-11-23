# Copyright (c) OpenMMLab. All rights reserved.
"""Latency of a torch detector vs its TensorRT deploy directories on dataset images.

Times the network (``_forward``: backbone, neck, head forward) and, with
``--e2e``, the whole ``predict`` (network + box decoding + rotated NMS, which is
the same torch code for both) on preprocessed images of the config's dataset::

    python tools/tensorrt/bench.py CONFIG CHECKPOINT DEPLOY_DIR [DEPLOY_DIR ...] [--e2e]

Real images matter: deformable attention samples at data-dependent locations
and NMS time depends on the number of detections.
"""

import argparse
import copy
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import torch
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope

from mmrotate.tensorrt import init_trt_detector
from mmrotate.tensorrt.calibration import build_input_loader, iter_batches
from mmrotate.tensorrt.export import build_detector

Batch = tuple[torch.Tensor, list]
TORCH_MODES = (('fp32', None), ('fp16 amp', torch.float16), ('bf16 amp', torch.bfloat16))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Benchmark torch vs TensorRT')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('deploy_dirs', nargs='*')
    parser.add_argument('--images', type=int, default=64, help='dataset images to cycle over')
    parser.add_argument('--split', default='val', choices=('train', 'val', 'test'))
    parser.add_argument('--iters', type=int, default=50, help='timed batches per model')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--e2e', action='store_true', help='also time predict (with NMS)')
    parser.add_argument('--no-torch', action='store_true', help='skip the torch baselines')
    parser.add_argument(
        '--batch-size', type=int, help='torch batch size when no deploy dir is given'
    )
    parser.add_argument('--input-shape', type=int, nargs=2, metavar=('H', 'W'))
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out', help='optional JSON output')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    return parser.parse_args()


@torch.no_grad()
def _time(fn: Callable[[Batch], object], batches: list[Batch], warmup: int, iters: int) -> dict:
    """Per-batch latency statistics (ms) of ``fn`` cycling over ``batches``."""
    for i in range(warmup):
        fn(batches[i % len(batches)])
    torch.cuda.synchronize()
    times = []
    for i in range(iters):
        batch = batches[i % len(batches)]
        start = time.perf_counter()
        fn(batch)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    times.sort()
    return dict(
        median=statistics.median(times),
        mean=statistics.fmean(times),
        p90=times[min(len(times) - 1, int(0.9 * len(times)))],
    )


def _load_batches(
    cfg: Config, model: torch.nn.Module, shape: tuple[int, ...], args: argparse.Namespace
) -> list[Batch]:
    """Full batches of preprocessed images on the device."""
    loader = build_input_loader(cfg, args.split, shuffle=False, num_workers=args.num_workers)
    batches = [
        (inputs.to(args.device), samples)
        for inputs, samples in iter_batches(loader, model.data_preprocessor, shape, args.images)
        if inputs.shape[0] == shape[0]
    ]
    if not batches:
        raise ValueError(f'--images {args.images} gives no full batch of {shape[0]}')
    return batches


def _bench(
    name: str,
    forward: Callable[[Batch], object],
    predict: Callable[[Batch], object] | None,
    batches: list[Batch],
    args: argparse.Namespace,
) -> dict:
    row = dict(model=name, shape=list(batches[0][0].shape))
    row['forward'] = _time(forward, batches, args.warmup, args.iters)
    if predict is not None:
        row['e2e'] = _time(predict, batches, args.warmup, args.iters)
    return row


def main() -> None:
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    init_default_scope(cfg.get('default_scope', 'mmrotate'))
    torch_model = build_detector(cfg, args.checkpoint, args.device)

    metas = {d: json.loads((Path(d) / 'trt_meta.json').read_text()) for d in args.deploy_dirs}
    shapes = sorted({tuple(m['input_shape']) for m in metas.values()})
    if not shapes:
        if not (args.batch_size and args.input_shape):
            raise SystemExit('Give deploy dirs, or --batch-size and --input-shape')
        shapes = [(args.batch_size, 3, *args.input_shape)]
    batches = {shape: _load_batches(cfg, torch_model, shape, args) for shape in shapes}

    rows = []
    for deploy_dir, meta in metas.items():
        model = init_trt_detector(deploy_dir, args.device)

        def predict(batch: Batch, model: torch.nn.Module = model) -> object:
            return model.predict(batch[0], copy.deepcopy(batch[1]))

        row = _bench(
            f'TensorRT {meta["precision"]}',
            lambda batch, model=model: model._forward(batch[0]),
            predict if args.e2e else None,
            batches[tuple(meta['input_shape'])],
            args,
        )
        row['dir'] = deploy_dir
        rows.append(row)
        del model
        torch.cuda.empty_cache()

    for shape in [] if args.no_torch else shapes:
        for name, dtype in TORCH_MODES:

            def autocast(dtype: torch.dtype | None = dtype) -> torch.autocast:
                return torch.autocast('cuda', dtype=dtype or torch.float32, enabled=bool(dtype))

            def forward(batch: Batch, autocast: Callable = autocast) -> object:
                with autocast():
                    return torch_model._forward(batch[0])

            def predict(batch: Batch, autocast: Callable = autocast) -> object:
                with autocast():
                    return torch_model.predict(batch[0], copy.deepcopy(batch[1]))

            rows.append(
                _bench(
                    f'torch {name}', forward, predict if args.e2e else None, batches[shape], args
                )
            )

    header = '| model | input | forward ms (median / p90) | forward img/s |'
    rule = '|---|---|---|---|'
    if args.e2e:
        header += ' e2e ms (median / p90) | e2e img/s |'
        rule += '---|---|'
    print('\n' + header + '\n' + rule)
    for r in rows:
        bs = r['shape'][0]
        f = r['forward']
        label = r['model'] + (f' `{Path(r["dir"]).name}`' if r.get('dir') else '')
        line = (
            f'| {label} | {"x".join(map(str, r["shape"]))} | {f["median"]:.1f} / {f["p90"]:.1f} '
            f'| {bs / f["median"] * 1000:.1f} |'
        )
        if args.e2e:
            e = r['e2e']
            line += f' {e["median"]:.1f} / {e["p90"]:.1f} | {bs / e["median"] * 1000:.1f} |'
        print(line)
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
