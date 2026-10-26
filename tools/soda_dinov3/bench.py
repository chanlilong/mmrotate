# Copyright (c) OpenMMLab. All rights reserved.
"""Measure training/inference cost of detector configs on their real data.

For every config: parameter counts (total / trainable / backbone), training
throughput (images/s, real dataloader, optimiser step, bf16 as configured),
peak GPU memory and inference throughput. Prints a markdown table and,
optionally, writes JSON::

    python tools/soda_dinov3/bench.py configs/soda_dinov3/*.yaml --out bench.json
"""

import argparse
import json
import os.path as osp
import tempfile
import time
from collections.abc import Iterator

import torch
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmrotate.utils import register_all_modules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Benchmark detector configs')
    parser.add_argument('configs', nargs='+')
    parser.add_argument('--train-iters', type=int, default=60)
    parser.add_argument('--test-iters', type=int, default=30)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--out', help='optional JSON output')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    return parser.parse_args()


def _cycle(loader: torch.utils.data.DataLoader) -> Iterator[dict]:
    while True:
        yield from loader


def _count(module: torch.nn.Module, trainable: bool = False) -> float:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable) / 1e6


def bench_config(path: str, args: argparse.Namespace) -> dict:
    cfg = Config.fromfile(path)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    cfg.work_dir = tempfile.mkdtemp(prefix='bench_')
    cfg.log_level = 'WARNING'
    cfg.default_hooks.pop('visualization', None)
    runner = Runner.from_cfg(cfg)
    runner._init_model_weights()
    model = runner.model.cuda()
    optim_wrapper = runner.build_optim_wrapper(cfg.optim_wrapper)
    batch_size = cfg.train_dataloader.batch_size
    result = dict(
        config=osp.basename(path),
        params_total_M=round(_count(model), 2),
        params_trainable_M=round(_count(model, trainable=True), 2),
        backbone_total_M=round(_count(model.backbone), 2),
        backbone_trainable_M=round(_count(model.backbone, trainable=True), 2),
        batch_size=batch_size,
    )

    model.train()
    data = _cycle(runner.build_dataloader(cfg.train_dataloader))
    torch.cuda.reset_peak_memory_stats()
    for i in range(args.warmup + args.train_iters):
        if i == args.warmup:
            torch.cuda.synchronize()
            start = time.perf_counter()
        model.train_step(next(data), optim_wrapper)
    torch.cuda.synchronize()
    train_time = (time.perf_counter() - start) / args.train_iters
    result['train_img_per_s'] = round(batch_size / train_time, 2)
    result['train_peak_mem_GB'] = round(torch.cuda.max_memory_allocated() / 1e9, 2)

    model.eval()
    test_data = _cycle(runner.build_dataloader(cfg.test_dataloader))
    test_bs = cfg.test_dataloader.batch_size
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for i in range(args.warmup + args.test_iters):
            if i == args.warmup:
                torch.cuda.synchronize()
                start = time.perf_counter()
            model.test_step(next(test_data))
    torch.cuda.synchronize()
    result['test_img_per_s'] = round(test_bs * args.test_iters / (time.perf_counter() - start), 2)
    n_train = len(runner.build_dataloader(cfg.train_dataloader).dataset)
    epochs = cfg.train_cfg.max_epochs
    result['est_train_hours'] = round(n_train * epochs / result['train_img_per_s'] / 3600, 2)
    del model, runner, optim_wrapper
    torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    register_all_modules()
    results = []
    for path in args.configs:
        res = bench_config(path, args)
        print(json.dumps(res), flush=True)
        results.append(res)
    keys = list(results[0])
    print('\n| ' + ' | '.join(keys) + ' |')
    print('|' + '---|' * len(keys))
    for res in results:
        print('| ' + ' | '.join(str(res[k]) for k in keys) + ' |')
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
