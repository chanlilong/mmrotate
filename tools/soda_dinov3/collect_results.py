# Copyright (c) OpenMMLab. All rights reserved.
"""Collect SODA-A DINOv3 experiment results into markdown tables.

Reads every run directory under ``work_root`` (as produced by
``run_experiments.sh``): test-split metrics (``<run>/test/<ts>/<ts>.json``),
``val_mini`` curves and training time (``<run>/<ts>/vis_data/scalars.json``)
and, if present, ``bench.json``. Writes ``results.md`` and ``results.json``::

    python tools/soda_dinov3/collect_results.py work_dirs/soda_dinov3
"""

import argparse
import glob
import json
import os.path as osp
import re
from collections import defaultdict

import numpy as np

MAIN = ('mAP', 'AP50', 'AP75', 'AP_eS', 'AP_rS', 'AP_gS', 'AP_N')
CLASSES = (
    'airplane',
    'helicopter',
    'small-vehicle',
    'large-vehicle',
    'ship',
    'container',
    'storage-tank',
    'swimming-pool',
    'windmill',
)
SEED_RE = re.compile(r'_seed(\d+)$')


def _load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def read_run(run_dir: str) -> dict | None:
    test_jsons = sorted(glob.glob(osp.join(run_dir, 'test', '*', '*.json')))
    test = {}
    for path in test_jsons:
        with open(path) as f:
            data = json.load(f)
        if 'soda/mAP' in data:
            test = {k.removeprefix('soda/'): v for k, v in data.items() if k.startswith('soda/')}
    curve: dict[int, float] = {}
    iter_times: list[float] = []
    for path in sorted(glob.glob(osp.join(run_dir, '*', 'vis_data', 'scalars.json'))):
        for row in _load_jsonl(path):
            if 'soda/mAP' in row:
                curve[int(row['step'])] = row['soda/mAP']
            elif 'time' in row and 'loss' in row:
                iter_times.append(row['time'])
    if not test and not curve:
        return None
    name = osp.basename(run_dir.rstrip('/'))
    match = SEED_RE.search(name)
    return dict(
        name=name,
        config=SEED_RE.sub('', name),
        seed=int(match.group(1)) if match else 0,
        test=test,
        val_curve=dict(sorted(curve.items())),
        mean_iter_time=float(np.mean(iter_times)) if iter_times else None,
    )


def _fmt(v: float | None, scale: float = 100.0) -> str:
    return '-' if v is None else f'{v * scale:.1f}'


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    lines += ['| ' + ' | '.join(row) + ' |' for row in rows]
    return '\n'.join(lines)


def build_markdown(runs: list[dict], bench: list[dict]) -> str:
    out = []
    finished = [r for r in runs if r['test']]
    out.append('### Test split (AP x 100)\n')
    out.append(
        _table(
            ['run', *MAIN],
            [[r['name'], *(_fmt(r['test'].get(k)) for k in MAIN)] for r in finished],
        )
    )

    by_cfg: dict[str, list[dict]] = defaultdict(list)
    for r in finished:
        by_cfg[r['config']].append(r)
    multi = {c: rs for c, rs in by_cfg.items() if len(rs) > 1}
    if multi:
        out.append('\n### Seed variation (mean, +/- half range, AP x 100)\n')
        rows = []
        for cfg, rs in multi.items():
            row = [f'{cfg} (n={len(rs)})']
            for k in MAIN:
                vals = np.array([r['test'][k] for r in rs]) * 100
                row.append(f'{vals.mean():.1f} +/- {(vals.max() - vals.min()) / 2:.1f}')
            rows.append(row)
        out.append(_table(['config', *MAIN], rows))

    out.append('\n### Per-class test mAP@[.5:.95] (x 100)\n')
    out.append(
        _table(
            ['run', *CLASSES],
            [[r['name'], *(_fmt(r['test'].get(f'AP_{c}')) for c in CLASSES)] for r in finished],
        )
    )

    steps = sorted({s for r in runs for s in r['val_curve']})
    if steps:
        out.append('\n### val_mini mAP@[.5:.95] during training (x 100, by epoch)\n')
        out.append(
            _table(
                ['run', *(f'ep {s}' for s in steps)],
                [[r['name'], *(_fmt(r['val_curve'].get(s)) for s in steps)] for r in runs],
            )
        )

    if bench:
        out.append('\n### Cost (batch 8, 800x800, bf16, RTX 3090)\n')
        keys = [
            'params_total_M',
            'params_trainable_M',
            'backbone_total_M',
            'backbone_trainable_M',
            'train_img_per_s',
            'train_peak_mem_GB',
            'test_img_per_s',
        ]
        out.append(
            _table(['config', *keys], [[b['config'], *(str(b[k]) for k in keys)] for b in bench])
        )
    return '\n'.join(out) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description='Collect SODA-A DINOv3 results')
    parser.add_argument('work_root')
    args = parser.parse_args()
    runs = []
    for run_dir in sorted(glob.glob(osp.join(args.work_root, '*', ''))):
        run = read_run(run_dir)
        if run is not None:
            runs.append(run)
    bench_path = osp.join(args.work_root, 'bench.json')
    bench = json.load(open(bench_path)) if osp.isfile(bench_path) else []
    markdown = build_markdown(runs, bench)
    with open(osp.join(args.work_root, 'results.md'), 'w') as f:
        f.write(markdown)
    with open(osp.join(args.work_root, 'results.json'), 'w') as f:
        json.dump(dict(runs=runs, bench=bench), f, indent=2)
    print(markdown)


if __name__ == '__main__':
    main()
