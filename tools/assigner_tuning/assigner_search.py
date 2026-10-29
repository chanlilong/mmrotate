# Copyright (c) OpenMMLab. All rights reserved.
"""Successive-halving search over label-assigner candidates with short trainings.

The second stage after ``assigner_probe.py``: takes the probe's best
candidates (or a ``--spec`` / ``--sweep`` like the probe), trains each on a
small random subset of the training set with the config's full schedule,
evaluates on the config's val set, keeps the best ``1/eta`` and doubles their
data, and so on::

    python tools/assigner_tuning/assigner_search.py CONFIG \\
        --probe work_dirs/assigner_probe/NAME/probe.json --top 6 --max-starved '<8:0.02' \\
        --budgets 0.125 0.25 0.5 --eta 2 --metric soda/mAP \\
        --cmd 'sbatch --wait -p sep --gres=gpu:1 tools/assigner_tuning/slurm_train.sbatch \\
               {config} {work_dir}' --max-parallel 4

Each trial is a standalone config (``<work-dir>/configs/<trial>.py``: the
config with the candidate assigner, ``train_dataloader.dataset.indices`` set to
the subset and validation only after the last epoch) trained by ``--cmd``
(placeholders ``{config}``, ``{work_dir}``, ``{name}``; default
``python tools/train.py {config} --work-dir {work_dir}``). The config's own
assigner is kept in every rung as the reference. All trials share the seed and
the subset, so differences are not data or seed luck, but read anything within
about +-0.3 AP at a small budget as a tie. Finished trials (metric present) are
not re-run, so an interrupted search can be restarted with the same command.
Writes ``<work-dir>/search.md`` and ``search.json``.
"""

import argparse
import copy
import glob
import json
import math
import os
import os.path as osp
import shlex
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import assigner_probe as probe  # noqa: E402

from mmrotate.registry import DATASETS  # noqa: E402

DEFAULT_CMD = 'python tools/train.py {config} --work-dir {work_dir}'


@dataclass
class Trial:
    """One candidate at one budget."""

    name: str
    candidate: probe.Candidate
    budget: float
    rung: int
    work_dir: str
    metric: float | None = None


def slug(text: str) -> str:
    """File-system safe trial name."""
    keep = [c if c.isalnum() or c in '.-' else '_' for c in text.replace(': ', '__')]
    return ''.join(keep).strip('_')[:120]


def candidates_from_probe(
    path: str, top: int, names: Sequence[str] | None, max_starved: Sequence[str] | None = None
) -> list:
    """The ``top`` best (or the named) candidates of a ``probe.json``.

    ``max_starved`` (``BIN:SHARE`` as for the probe) drops the candidates that
    starve more GTs than allowed before taking the ``top``.
    """
    with open(path) as f:
        results = json.load(f)['results']
    if names:
        by_name = {r['name']: r for r in results}
        missing = [n for n in names if n not in by_name]
        if missing:
            raise ValueError(f'not in {path}: {missing}')
        results = [by_name[n] for n in names]
    else:
        if max_starved and results:
            caps = probe.parse_caps(max_starved, list(results[0]['summary']['by_size']))
            kept = [r for r in results if not probe.violations(r['summary'], caps)]
            print(f'--max-starved keeps {len(kept)} of {len(results)} probe candidates')
            results = kept
        results = results[:top]
    return [probe.Candidate(r['name'], r['family'], r['cfg'], r['changed']) for r in results]


def subset_indices(cfg: Config, fraction: float, seed: int) -> list[int] | None:
    """A fixed random subset of the (filtered) training set, None for the whole set."""
    if fraction >= 1:
        return None
    ds_cfg = copy.deepcopy(cfg.train_dataloader.dataset)
    ds_cfg['lazy_init'] = True
    dataset = DATASETS.build(ds_cfg)
    dataset.full_init()
    n = len(dataset)
    rng = np.random.default_rng(seed)
    # Nested subsets: the first k of one permutation, so a larger budget adds data.
    return sorted(rng.permutation(n)[: max(1, round(fraction * n))].tolist())


def trial_config(
    cfg: Config, cand: probe.Candidate, indices: list[int] | None, assigner_key: str
) -> Config:
    """The config with the candidate assigner, the subset and a single final validation."""
    out = copy.deepcopy(cfg)
    node = out
    keys = assigner_key.split('.')
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = copy.deepcopy(cand.cfg)
    if indices is not None:
        out.train_dataloader.dataset.indices = indices
    if out.train_cfg.get('type', 'EpochBasedTrainLoop') == 'EpochBasedTrainLoop':
        out.train_cfg.val_interval = out.train_cfg.max_epochs
    return out


def read_metric(work_dir: str, key: str) -> float | None:
    """Last logged value of ``key`` in ``<work_dir>/*/vis_data/scalars.json``."""
    value = None
    for path in sorted(glob.glob(osp.join(work_dir, '*', 'vis_data', 'scalars.json'))):
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                if key in rec:
                    value = float(rec[key])
    return value


def run_trials(
    trials: list[Trial], cmd: str, max_parallel: int, metric: str, dry_run: bool
) -> None:
    """Run the trials without a metric, ``max_parallel`` at a time."""
    todo = []
    for t in trials:
        t.metric = read_metric(t.work_dir, metric)
        if t.metric is None:
            todo.append(t)
        else:
            print(f'  [done] {t.name}: {t.metric:.4f}')
    running: list[tuple[Trial, subprocess.Popen]] = []
    while todo or running:
        while todo and len(running) < max_parallel:
            t = todo.pop(0)
            line = cmd.format(
                config=shlex.quote(osp.join(t.work_dir, 'config.py')),
                work_dir=shlex.quote(t.work_dir),
                name=shlex.quote(t.name),
            )
            print(f'  [run] {t.name}: {line}', flush=True)
            if dry_run:
                continue
            log = open(osp.join(t.work_dir, 'launch.log'), 'a')  # noqa: SIM115
            running.append((t, subprocess.Popen(line, shell=True, stdout=log, stderr=log)))
        for t, proc in list(running):
            if proc.poll() is None:
                continue
            running.remove((t, proc))
            t.metric = read_metric(t.work_dir, metric)
            status = f'{t.metric:.4f}' if t.metric is not None else 'NO METRIC'
            print(f'  [end] {t.name}: exit {proc.returncode}, {metric} {status}', flush=True)
        if running:
            time.sleep(30)


def write_report(path: str, args: argparse.Namespace, rungs: list[list[Trial]]) -> None:
    """Markdown table: one column per budget, rows sorted by the last budget reached."""
    names: list[str] = []
    for rung in rungs:
        names += [t.candidate.name for t in rung if t.candidate.name not in names]
    cell = {(t.candidate.name, t.rung): t.metric for rung in rungs for t in rung}
    last = {n: max(r for (m, r) in cell if m == n) for n in names}
    names.sort(key=lambda n: (-last[n], -(cell[(n, last[n])] or -math.inf)))
    lines = [
        f'# Assigner search: {osp.basename(args.config)}',
        '',
        f'`{args.metric}` (x100) after the full schedule on a random share of the training set;',
        f'the best 1/{args.eta} of each rung go on. `*`: the config assigner (reference).',
        '',
        '| candidate | ' + ' | '.join(f'{100 * b:g}% data' for b in args.budgets) + ' |',
        '|---|' + '---|' * len(args.budgets),
    ]
    for n in names:
        mark = ' *' if n == rungs[0][0].candidate.name else ''
        vals = []
        for r in range(len(args.budgets)):
            v = cell.get((n, r))
            vals.append('' if (n, r) not in cell else ('-' if v is None else f'{100 * v:.2f}'))
        lines.append(f'| {n}{mark} | ' + ' | '.join(vals) + ' |')
    lines.append('')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command line."""
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('config')
    p.add_argument('--probe', help='probe.json of assigner_probe.py')
    p.add_argument('--top', type=int, default=6, help='best probe candidates to train')
    p.add_argument('--names', nargs='+', help='probe candidates by name instead of --top')
    p.add_argument(
        '--max-starved',
        action='append',
        help='BIN:SHARE cap on starved GTs in probe.json, e.g. "<8:0.02" (repeatable)',
    )
    p.add_argument('--spec', help='YAML families, as for the probe')
    p.add_argument('--sweep', action='append', default=[], help='key=v1,v2 on the config assigner')
    p.add_argument('--mode', choices=('oat', 'grid'), default='oat')
    p.add_argument('--budgets', type=float, nargs='+', default=[0.125, 0.25, 0.5])
    p.add_argument('--eta', type=float, default=2.0)
    p.add_argument('--metric', default='soda/mAP', help='val key in scalars.json')
    p.add_argument('--assigner-key', default='model.train_cfg.assigner')
    p.add_argument('--cmd', default=DEFAULT_CMD, help='{config} {work_dir} {name}')
    p.add_argument('--max-parallel', type=int, default=1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--work-dir', help='default work_dirs/assigner_search/<config name>')
    p.add_argument('--dry-run', action='store_true', help='write configs, print commands')
    p.add_argument('--cfg-options', nargs='+', action=DictAction)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the search."""
    args = parse_args(argv)
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    init_default_scope(cfg.get('default_scope', 'mmrotate'))
    cfg.randomness = dict(cfg.get('randomness', {}), seed=args.seed)
    work = args.work_dir or osp.join(
        'work_dirs', 'assigner_search', osp.splitext(osp.basename(args.config))[0]
    )

    ref_cfg = probe.get_path(cfg.to_dict(), args.assigner_key)
    ref = probe.Candidate('config', 'config', ref_cfg)
    cands = []
    if args.probe:
        cands += candidates_from_probe(args.probe, args.top, args.names, args.max_starved)
    if args.spec or args.sweep:
        args.no_config_assigner = not args.sweep
        cands += probe.gather_candidates(cfg, args)
    seen = {probe.canonical(ref.cfg)}
    pool = [ref]
    for c in cands:  # drop candidates that build the same assigner as an earlier one
        if probe.canonical(c.cfg) not in seen:
            seen.add(probe.canonical(c.cfg))
            pool.append(c)
    print(f'{len(pool)} candidates (incl. the config assigner), budgets {args.budgets}')

    rungs: list[list[Trial]] = []
    for r, budget in enumerate(args.budgets):
        indices = subset_indices(cfg, budget, args.seed)
        trials = []
        for c in pool:
            name = f'b{budget:g}_{slug(c.name)}'
            wd = osp.join(work, name)
            os.makedirs(wd, exist_ok=True)
            trial_config(cfg, c, indices, args.assigner_key).dump(osp.join(wd, 'config.py'))
            trials.append(Trial(name, c, budget, r, wd))
        print(f'rung {r}: {len(trials)} trials on {100 * budget:g}% of the training set')
        run_trials(trials, args.cmd, args.max_parallel, args.metric, args.dry_run)
        rungs.append(trials)
        write_report(osp.join(work, 'search.md'), args, rungs)
        if args.dry_run:
            pool = pool[: max(1, math.ceil(len(pool) / args.eta))]
            continue
        scored = sorted(
            (t for t in trials if t.metric is not None and t.candidate is not ref),
            key=lambda t: -t.metric,
        )
        keep = max(1, math.ceil((len(pool) - 1) / args.eta))
        pool = [ref] + [t.candidate for t in scored[:keep]]
    with open(osp.join(work, 'search.json'), 'w') as f:
        json.dump(
            [
                dict(name=t.candidate.name, cfg=t.candidate.cfg, budget=t.budget, metric=t.metric)
                for rung in rungs
                for t in rung
            ],
            f,
            indent=1,
            default=str,
        )
    print(f'wrote {work}/search.md, search.json')


if __name__ == '__main__':
    main()
