# Copyright (c) OpenMMLab. All rights reserved.
"""Summarise ``AssignmentLogHook`` logs: is a starved object starved once or always?

::

    python tools/assigner_tuning/assignment_log_report.py \\
        DSL=work_dirs/assign_log/dsl DCFL=work_dirs/assign_log/dcfl \\
        [--size-edges 8 12 20 32] [--focus-size '<8'] [--out work_dirs/assign_log/report.md]

Each argument is ``NAME=PATH``, PATH a work dir holding ``assignment_log*.npz``
(ranks are merged) or one ``.npz``. Per run and object size bin it reports:

- ``starved/epoch``: share of (object, epoch) pairs without a positive, what a
  probe of one checkpoint measures;
- ``never``: objects without a positive in any epoch, i.e. trained only as
  background for the whole run;
- ``late never``: objects without a positive in every epoch of the second half;
- ``always``: objects with a positive in every epoch;
- ``pos/visit`` and ``best IoU`` (over the epochs in which the object was matched).

Only objects whose image was visited in every logged epoch are counted. A
second table follows the focus bin epoch by epoch: the share of objects not
matched in any epoch so far, which falls towards ``never`` if starvation moves
between objects and stays flat if it hits the same objects every time.
"""

import argparse
import glob
import itertools
import json
import os.path as osp
from collections.abc import Sequence

import numpy as np

DEFAULT_SIZE_EDGES = (8.0, 12.0, 20.0, 32.0)


def bin_names(edges: Sequence[float]) -> list[str]:
    """``(8, 12)`` -> ``['<8', '8-12', '>=12']`` (as in ``assigner_probe.py``)."""
    names = [f'<{edges[0]:g}']
    names += [f'{a:g}-{b:g}' for a, b in itertools.pairwise(edges)]
    return names + [f'>={edges[-1]:g}']


def load_log(path: str) -> dict[str, np.ndarray]:
    """One run's log; the rank files of a work dir are merged by image."""
    files = [path] if path.endswith('.npz') else sorted(glob.glob(osp.join(path, '*.npz')))
    files = [f for f in files if osp.basename(f).startswith('assignment_log') or f == path]
    if not files:
        raise FileNotFoundError(f'no assignment_log*.npz in {path}')
    logs = [dict(np.load(f)) for f in files]
    if len(logs) == 1:
        return logs[0]
    # an image can be visited by different ranks in different epochs: merge by path
    per_image: dict[str, dict[str, np.ndarray]] = {}
    for log in logs:
        for i, img in enumerate(log['images']):
            m = log['obj_image'] == i
            cur = dict(
                seen=log['seen'][:, i].copy(),
                size=log['obj_size'][m],
                label=log['obj_label'][m],
                npos=log['npos'][:, m].copy(),
                best=log['best_iou'][:, m].astype(np.float32),
            )
            old = per_image.get(str(img))
            if old is None:
                per_image[str(img)] = cur
            else:
                old['seen'] += cur['seen']
                old['npos'] += cur['npos']
                old['best'] = np.maximum(old['best'], cur['best'])
    entries = list(per_image.values())
    sizes = [e['size'].size for e in entries]
    return dict(
        images=np.array(list(per_image)),
        seen=np.stack([e['seen'] for e in entries], 1),
        obj_image=np.repeat(np.arange(len(entries)), sizes),
        obj_size=np.concatenate([e['size'] for e in entries]),
        obj_label=np.concatenate([e['label'] for e in entries]),
        obj_density=np.repeat(sizes, sizes),
        npos=np.concatenate([e['npos'] for e in entries], 1),
        best_iou=np.concatenate([e['best'] for e in entries], 1),
    )


def summarize(log: dict[str, np.ndarray], size_edges: Sequence[float]) -> dict:
    """Per size bin starvation over the run, plus the running ``never`` per epoch."""
    epochs = np.flatnonzero(log['seen'].sum(1) > 0)  # logged epochs (a run may be partial)
    visits = log['seen'][epochs][:, log['obj_image']]  # (E, N)
    seen = visits > 0
    npos = log['npos'][epochs]
    best = log['best_iou'][epochs].astype(np.float32)
    complete = seen.all(0)
    matched = npos > 0
    late = epochs.size // 2  # first epoch of the second half
    size_bin = np.digitize(log['obj_size'], size_edges)

    def block(m: np.ndarray) -> dict:
        m = m & complete
        if not m.any():
            return dict(objects=0)
        hit = matched[:, m]
        n_hit = max(1, int(hit.sum()))
        return dict(
            objects=int(m.sum()),
            starved_per_epoch=float(1 - hit.mean()),
            starved_last=float(1 - hit[-1].mean()),
            never=float((~hit.any(0)).mean()),
            late_never=float((~hit[late:].any(0)).mean()),
            always=float(hit.all(0).mean()),
            pos_per_visit=float(npos[:, m].sum() / visits[:, m].sum()),
            best_iou=float(best[:, m][hit].sum() / n_hit),
            never_so_far=[float((~hit[: e + 1].any(0)).mean()) for e in range(hit.shape[0])],
        )

    out = dict(epochs=[int(e) + 1 for e in epochs], all=block(np.ones_like(complete)))
    out['by_size'] = {n: block(size_bin == i) for i, n in enumerate(bin_names(size_edges))}
    return out


def _pct(v: float | None) -> str:
    return '-' if v is None else f'{100 * v:.1f}'


def write_markdown(path: str, runs: dict[str, dict], focus: str) -> None:
    """Size-bin table per run, then the running ``never`` of the focus bin."""
    lines = [
        '# Assignment log: starved once or always?',
        '',
        'Per object (exact duplicates merged), over the epochs of each training run. '
        '`starved/epoch` is what a one-checkpoint probe measures, `never` is the share '
        'of objects trained only as background for the whole run.',
        '',
        '| run | size (px) | objects | starved/epoch % | starved last epoch % | never % '
        '| late never % | always % | pos/visit | best IoU |',
        '|---|---|---|---|---|---|---|---|---|---|',
    ]
    for name, s in runs.items():
        for bin_name, b in [('all', s['all']), *s['by_size'].items()]:
            if not b.get('objects'):
                continue
            lines.append(
                f'| {name} | {bin_name} | {b["objects"]} | {_pct(b["starved_per_epoch"])} '
                f'| {_pct(b["starved_last"])} | {_pct(b["never"])} | {_pct(b["late_never"])} '
                f'| {_pct(b["always"])} | {b["pos_per_visit"]:.1f} | {b["best_iou"]:.3f} |'
            )
    n_ep = max(len(s['epochs']) for s in runs.values())
    lines += [
        '',
        f'## Objects {focus} px not matched in any epoch so far (%)',
        '',
        '| run | ' + ' | '.join(f'≤ ep {e + 1}' for e in range(n_ep)) + ' |',
        '|---|' + '---|' * n_ep,
    ]
    for name, s in runs.items():
        so_far = s['by_size'].get(focus, {}).get('never_so_far', [])
        cells = [_pct(v) for v in so_far] + [''] * (n_ep - len(so_far))
        lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    lines.append('')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command line."""
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('runs', nargs='+', help='NAME=PATH (work dir or .npz)')
    p.add_argument('--size-edges', type=float, nargs='+', default=list(DEFAULT_SIZE_EDGES))
    p.add_argument('--focus-size', default=None, help='bin of the per-epoch table (first bin)')
    p.add_argument('--out', default='assignment_log_report.md', help='markdown; .json next to it')
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Load every run, write the markdown and JSON reports."""
    args = parse_args(argv)
    names = bin_names(args.size_edges)
    focus = args.focus_size or names[0]
    if focus not in names:
        raise ValueError(f'--focus-size must be one of {names}')
    runs = {}
    for spec in args.runs:
        name, sep, path = spec.partition('=')
        if not sep:
            name, path = osp.basename(osp.normpath(spec)), spec
        runs[name] = summarize(load_log(path), args.size_edges)
        b = runs[name]['by_size'].get(focus, {})
        print(
            f'{name}: epochs {runs[name]["epochs"]}, {focus} px: starved/epoch '
            f'{_pct(b.get("starved_per_epoch"))}%, never {_pct(b.get("never"))}%'
        )
    write_markdown(args.out, runs, focus)
    with open(osp.splitext(args.out)[0] + '.json', 'w') as f:
        json.dump(runs, f, indent=1)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
