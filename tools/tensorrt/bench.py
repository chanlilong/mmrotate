# Copyright (c) OpenMMLab. All rights reserved.
"""Network latency of a torch detector vs its TensorRT deploy directories.

Times ``_forward`` (backbone, neck, head forward; no post-processing, which
is the same torch code for both) at the engine's input shape::

    python tools/tensorrt/bench.py CONFIG CHECKPOINT DEPLOY_DIR [DEPLOY_DIR ...]
"""

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

import torch
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope

from mmrotate.tensorrt import init_trt_detector
from mmrotate.tensorrt.export import build_detector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Benchmark torch vs TensorRT')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('deploy_dirs', nargs='+')
    parser.add_argument('--iters', type=int, default=50)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out', help='optional JSON output')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    return parser.parse_args()


@torch.no_grad()
def _time(fn: Callable[[], object], warmup: int, iters: int) -> float:
    """Mean milliseconds per call of ``fn``."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000


def main() -> None:
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    init_default_scope(cfg.get('default_scope', 'mmrotate'))
    rows = []
    shapes = {}
    for deploy_dir in args.deploy_dirs:
        meta = json.loads((Path(deploy_dir) / 'trt_meta.json').read_text())
        shapes[deploy_dir] = tuple(meta['input_shape'])
        model = init_trt_detector(deploy_dir, args.device)
        x = torch.randn(shapes[deploy_dir], device=args.device)
        ms = _time(lambda m=model, x=x: m._forward(x), args.warmup, args.iters)
        rows.append(
            dict(model=f'TensorRT {meta["precision"]}', dir=deploy_dir, shape=x.shape, ms=ms)
        )
        del model
        torch.cuda.empty_cache()

    torch_model = build_detector(cfg, args.checkpoint, args.device)
    for shape in sorted(set(shapes.values())):
        x = torch.randn(shape, device=args.device)
        for name, dtype in (
            ('fp32', None),
            ('fp16 amp', torch.float16),
            ('bf16 amp', torch.bfloat16),
        ):

            def run(x: torch.Tensor = x, dtype: torch.dtype | None = dtype) -> object:
                with torch.autocast(
                    'cuda', dtype=dtype or torch.float32, enabled=dtype is not None
                ):
                    return torch_model._forward(x)

            ms = _time(run, args.warmup, args.iters)
            rows.append(dict(model=f'torch {name}', dir='', shape=x.shape, ms=ms))

    print('\n| model | input | ms / batch | images / s |')
    print('|---|---|---|---|')
    for r in rows:
        shape = 'x'.join(map(str, r['shape']))
        print(f'| {r["model"]} | {shape} | {r["ms"]:.1f} | {r["shape"][0] / r["ms"] * 1000:.1f} |')
    if args.out:
        for r in rows:
            r['shape'] = list(r['shape'])
        Path(args.out).write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
