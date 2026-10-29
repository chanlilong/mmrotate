# Copyright (c) OpenMMLab. All rights reserved.
"""Training-free screen of label-assigner hyperparameters.

Runs the detector's loss path once on a sample of training images, caches what
the head hands its assigner (priors, decoded boxes, class scores, GTs), then
re-runs the assignment for every candidate assigner on that cache and reports
what each candidate would train on::

    python tools/assigner_tuning/assigner_probe.py CONFIG [--checkpoint CKPT] \\
        --sweep topk=9,13,17 --sweep soft_center_radius=2,3,5 \\
        [--spec tools/assigner_tuning/specs/soda_rtmdet.yaml] [--num-images 120] \\
        [--out-dir work_dirs/assigner_probe/NAME] [--cfg-options ...]

A forward pass per image plus one assignment per image and candidate: a sweep
of 30 candidates on 120 images takes minutes, against hours for one training.
The probe only shows *what the assigner does* (how many GTs go untrained, how
good the positives are, which scales they come from); it ranks candidates as a
screen before short training runs (``assigner_search.py``), see README.md.

Candidates:

- ``--sweep key=v1,v2`` varies a key of the config's own assigner (dotted
  keys reach nested dicts, e.g. ``iou_calculator.constant=12,16.6``; ``none``
  is None). ``--mode oat`` (default) varies one key at a time around the
  config value, ``--mode grid`` takes the full product.
- ``--spec FILE`` (YAML) names assigner families, each a ``base`` assigner
  dict, an optional ``sweep`` and ``mode``, so different assigners (DSL,
  RKA, C2F, ...) are compared on the same predictions.

Prediction-aware assigners (DSL, the fine stage of C2F) depend on the model's
predictions, so give a ``--checkpoint``; a short run of the baseline (about one
epoch) is representative of most of training, a finished run of its end.
Without one the model is at initialisation, where only the prior geometry
(centre prior, ranking metric) matters.

Metrics, per GT (sizes are ``sqrt(w * h)`` of the GT in pixels):

- ``starved``: share of GTs with no positive prior, which are never trained;
- ``pos/GT``: positives per GT;
- ``pos IoU``: mean rotated IoU between the predicted boxes of the positives
  and their GT (low: the assigner trains priors whose predictions do not fit
  the object, i.e. noisy targets);
- ``coverage``: mean over GTs of the best positive's IoU (0 when starved), the
  recall the training targets allow;
- ``score``: ``coverage * pos IoU``, the default ranking: high when every
  object gets a well-fitting positive *and* few poorly fitting ones;
- ``offset``: mean distance between a positive's prior centre and its GT
  centre, in GT sizes (positives off the object learn background context);
- ``ignored``: share of priors the assigner neither trains as positive nor
  as negative.

A starved GT is worse than a missing one: its priors are trained as
background. ``--max-starved BIN:SHARE`` (e.g. ``'<8:0.02'``, ``all:0.01``;
repeatable) is a hard cap: candidates above it are ranked after every
candidate within it, whatever their score, and marked in ``probe.md``.
"""

import argparse
import copy
import inspect
import itertools
import json
import math
import os
import os.path as osp
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
import yaml
from mmcv.ops import box_iou_rotated
from mmengine.config import Config, DictAction
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmengine.structures import InstanceData

from mmrotate.registry import DATASETS, MODELS, TASK_UTILS

DEFAULT_SIZE_EDGES = (8.0, 12.0, 20.0, 32.0)
DENSITY_EDGES = (50, 200, 500)
RANK_KEYS = ('score', 'coverage', 'starved', 'pos_iou')


# ----------------------------------------------------------------------------- candidates


@dataclass
class Candidate:
    """One assigner configuration to probe."""

    name: str
    family: str
    cfg: dict
    changed: dict = field(default_factory=dict)


def parse_value(text: str) -> object:
    """YAML scalar (``none`` -> None, numbers, booleans, strings)."""
    if text.strip().lower() in ('none', 'null'):
        return None
    return yaml.safe_load(text)


def parse_sweep(items: Sequence[str]) -> dict[str, list]:
    """``['topk=9,13', 'iou_calculator.constant=12,16.6']`` -> dict of value lists."""
    sweep: dict[str, list] = {}
    for item in items:
        key, sep, values = item.partition('=')
        if not sep or not values:
            raise ValueError(f'--sweep wants key=v1,v2,..., got {item!r}')
        sweep[key.strip()] = [parse_value(v) for v in values.split(',')]
    return sweep


def get_path(cfg: dict, path: str) -> object:
    """Value at a dotted path, None when missing."""
    node: object = cfg
    for key in path.split('.'):
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def set_path(cfg: dict, path: str, value: object) -> None:
    """Set a dotted path, creating intermediate dicts."""
    keys = path.split('.')
    node = cfg
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def _fmt(value: object) -> str:
    return 'none' if value is None else str(value)


def canonical(cfg: dict) -> str:
    """Signature of an assigner config with arguments at their default value dropped.

    Two configs that build the same assigner (e.g. one spelling out a default)
    get the same signature.
    """
    out = copy.deepcopy(cfg)
    try:
        cls = TASK_UTILS.get(out.get('type'))
    except (KeyError, TypeError):
        cls = None
    defaults = {}
    for klass in reversed(getattr(cls, '__mro__', ())):
        init = klass.__dict__.get('__init__')
        if init is None:
            continue
        for name, param in inspect.signature(init).parameters.items():
            if param.default is not inspect.Parameter.empty:
                defaults[name] = param.default
    for key in [k for k in out if k in defaults and out[k] == defaults[k]]:
        del out[key]
    if cls is not None:
        out['type'] = f'{cls.__module__}.{cls.__name__}'
    return json.dumps(out, sort_keys=True, default=str)


def expand_family(family: str, base: dict, sweep: dict[str, list], mode: str) -> list[Candidate]:
    """The base assigner plus its sweep (``oat``: one key at a time, ``grid``: product)."""
    base = copy.deepcopy(base)
    out = [Candidate(family, family, base)]
    seen = {canonical(base)}

    def add(changes: dict) -> None:
        cfg = copy.deepcopy(base)
        for key, value in changes.items():
            set_path(cfg, key, value)
        sig = canonical(cfg)
        if sig in seen:
            return
        seen.add(sig)
        label = ' '.join(f'{k}={_fmt(v)}' for k, v in changes.items())
        out.append(Candidate(f'{family}: {label}', family, cfg, dict(changes)))

    if mode == 'oat':
        for key, values in sweep.items():
            for value in values:
                add({key: value})
    elif mode == 'grid':
        keys = list(sweep)
        for values in itertools.product(*(sweep[k] for k in keys)):
            add(dict(zip(keys, values, strict=True)))
    else:
        raise ValueError(f'unknown sweep mode {mode!r}')
    return out


def load_spec(path: str, default_mode: str) -> list[Candidate]:
    """Candidates of a YAML spec ``{family: {base: {...}, sweep: {...}, mode: oat}}``."""
    with open(path) as f:
        spec = yaml.safe_load(f)
    out = []
    for family, entry in spec.items():
        if 'base' not in entry:
            raise ValueError(f'{path}: family {family!r} has no base assigner')
        sweep = {
            k: [parse_value(x) if isinstance(x, str) else x for x in v]
            for k, v in (entry.get('sweep') or {}).items()
        }
        out += expand_family(family, entry['base'], sweep, entry.get('mode', default_mode))
    return out


# ----------------------------------------------------------------------------- capture


@dataclass
class Sample:
    """What the head passed to its assigner for one training image."""

    pred: InstanceData
    gt: InstanceData
    ignore: InstanceData | None


def _detach(data: InstanceData | None) -> InstanceData | None:
    if data is None:
        return None
    out = InstanceData(metainfo=data.metainfo)
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            value = value.detach()
            value = value.float() if value.is_floating_point() else value
        elif hasattr(value, 'tensor'):  # BaseBoxes
            value = type(value)(value.tensor.detach().float())
        out[key] = value.to('cpu') if hasattr(value, 'to') else value
    return out


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_images(dataset: object, num: int, seed: int) -> list[int]:
    """Indices stratified by GT count (``DENSITY_EDGES``), random without annotations."""
    rng = np.random.default_rng(seed)
    n = len(dataset)
    try:
        counts = np.array(
            [
                sum(1 for i in dataset.get_data_info(k)['instances'] if not i.get('ignore_flag'))
                for k in range(n)
            ]
        )
    except (AttributeError, KeyError, TypeError):
        return sorted(rng.choice(n, min(num, n), replace=False).tolist())
    edges = (1, *DENSITY_EDGES, math.inf)
    buckets = [
        np.nonzero((counts >= lo) & (counts < hi))[0] for lo, hi in itertools.pairwise(edges)
    ]
    buckets = sorted((b for b in buckets if len(b)), key=len)
    picked = []
    for i, b in enumerate(buckets):
        # equal shares; what a small bucket cannot fill goes to the larger ones
        per = math.ceil((num - len(picked)) / (len(buckets) - i))
        picked += rng.choice(b, min(per, len(b)), replace=False).tolist()
    return sorted(picked)


def collect_samples(
    cfg: Config,
    checkpoint: str | None,
    indices: Sequence[int],
    dataset: object,
    device: str,
    head_name: str,
    seed: int,
) -> list[Sample]:
    """One loss forward per image (stopped at the assignment); the cached assigner inputs.

    Images without GTs are skipped by most heads before assignment, and dropped here.
    """
    model = MODELS.build(cfg.model)
    if checkpoint:
        load_checkpoint(model, checkpoint, map_location='cpu', logger='silent')
    model = model.to(device).eval()
    head = getattr(model, head_name)
    assigner = head.assigner
    samples: list[Sample] = []
    orig = assigner.assign

    class Captured(Exception):  # noqa: N818
        """Raised by the capture hook: nothing past the assignment is needed."""

    def capture(
        pred_instances: InstanceData,
        gt_instances: InstanceData,
        gt_instances_ignore: InstanceData | None = None,
        **kwargs: object,
    ) -> object:
        samples.append(
            Sample(_detach(pred_instances), _detach(gt_instances), _detach(gt_instances_ignore))
        )
        raise Captured

    assigner.assign = capture
    use_amp = device.startswith('cuda')
    for n, k in enumerate(indices):
        _seed(seed + int(k))  # same augmentation for the same image across probe runs
        data = model.data_preprocessor(pseudo_collate([dataset[k]]), True)
        with (
            torch.no_grad(),
            torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp),
        ):
            try:
                model.loss(data['inputs'], data['data_samples'])
            except Captured:
                pass
        if (n + 1) % 20 == 0:
            print(f'  forward {n + 1}/{len(indices)}', flush=True)
    assigner.assign = orig
    if not samples:
        raise RuntimeError(f'the loss never called {head_name}.assigner (see --head)')
    return [s for s in samples if len(s.gt) > 0]


# ----------------------------------------------------------------------------- metrics


def _boxes(x: object) -> torch.Tensor:
    t = x.tensor if hasattr(x, 'tensor') else x
    return t[..., :5].float()


def duplicate_groups(boxes: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Group id of every GT, identical boxes (to 0.01 px) of one class sharing a group.

    Only one copy of a duplicated annotation can own a prior, so the copies are
    scored as one object (positives summed, best IoU over the copies).
    """
    key = torch.cat([(boxes * 100).round(), labels[:, None].to(boxes)], dim=1)
    uniq, group = torch.unique(key, dim=0, return_inverse=True)
    return group, uniq.size(0)


def assignment_stats(
    assigner: object, samples: Sequence[Sample], device: str
) -> dict[str, np.ndarray]:
    """Per-object and per-positive arrays of one assigner over the cached samples."""
    gt_size, gt_dense, gt_npos, gt_best = [], [], [], []
    pos_iou, pos_off, pos_stride, pos_size = [], [], [], []
    n_prior = n_ign = n_dup = 0
    for s in samples:
        pred = s.pred.to(device)
        gt = s.gt.to(device)
        ign = s.ignore.to(device) if s.ignore is not None else None
        with torch.no_grad():
            res = assigner.assign(pred, gt, ign)
        inds = res.gt_inds
        G = len(gt)
        gtb = _boxes(gt.bboxes)
        size = (gtb[:, 2] * gtb[:, 3]).clamp(min=0).sqrt()
        pos = (inds > 0).nonzero().squeeze(1)
        owner = inds[pos] - 1
        iou = (
            box_iou_rotated(_boxes(pred.bboxes)[pos], gtb[owner], aligned=True)
            if pos.numel()
            else gtb.new_zeros(0)
        )
        group, U = duplicate_groups(gtb, gt.labels)
        best = gtb.new_zeros(U).scatter_reduce(
            0, group[owner], iou, reduce='amax', include_self=True
        )
        priors = pred.priors
        off = (priors[pos, :2].float() - gtb[owner, :2]).norm(dim=1) / size[owner].clamp(min=1)
        gt_size.append(size.new_zeros(U).scatter_(0, group, size).cpu())
        gt_dense.append(torch.full((U,), U))
        gt_npos.append(torch.bincount(group[owner], minlength=U).cpu())
        gt_best.append(best.cpu())
        n_dup += G - U
        pos_iou.append(iou.cpu())
        pos_off.append(off.cpu())
        pos_stride.append(priors[pos, 2].float().cpu() if priors.size(1) > 2 else off.new_zeros(0))
        pos_size.append(size[owner].cpu())
        n_prior += inds.numel()
        n_ign += int((inds < 0).sum())
    cat = lambda xs: torch.cat(xs).numpy()  # noqa: E731
    return dict(
        gt_size=cat(gt_size),
        gt_dense=cat(gt_dense),
        gt_npos=cat(gt_npos),
        gt_best=cat(gt_best),
        pos_iou=cat(pos_iou),
        pos_off=cat(pos_off),
        pos_stride=cat(pos_stride),
        pos_size=cat(pos_size),
        ignored=np.array(n_ign / max(1, n_prior)),
        duplicates=np.array(n_dup),
    )


def bin_names(edges: Sequence[float]) -> list[str]:
    """``(8, 12)`` -> ``['<8', '8-12', '>=12']``."""
    names = [f'<{edges[0]:g}']
    names += [f'{a:g}-{b:g}' for a, b in itertools.pairwise(edges)]
    return names + [f'>={edges[-1]:g}']


def summarize(stats: dict[str, np.ndarray], size_edges: Sequence[float]) -> dict:
    """Scalar metrics, overall and per GT-size bin and image-density bin."""
    npos, best = stats['gt_npos'], stats['gt_best']

    def block(m: np.ndarray, pm: np.ndarray | None = None) -> dict:
        if not m.any():
            return dict(gts=0)
        out = dict(
            gts=int(m.sum()),
            starved=float((npos[m] == 0).mean()),
            pos_per_gt=float(npos[m].mean()),
            coverage=float(best[m].mean()),
        )
        if pm is not None and pm.any():
            out['pos_iou'] = float(stats['pos_iou'][pm].mean())
            out['score'] = out['coverage'] * out['pos_iou']
        return out

    size_bin = np.digitize(stats['gt_size'], size_edges)
    pos_bin = np.digitize(stats['pos_size'], size_edges)
    dense_bin = np.digitize(stats['gt_dense'], DENSITY_EDGES)
    out = block(np.ones_like(npos, dtype=bool), np.ones_like(stats['pos_iou'], dtype=bool))
    out['offset'] = float(stats['pos_off'].mean()) if stats['pos_off'].size else float('nan')
    out['ignored'] = float(stats['ignored'])
    out['duplicates'] = int(stats['duplicates'])
    out['by_size'] = {
        name: block(size_bin == i, pos_bin == i) for i, name in enumerate(bin_names(size_edges))
    }
    dense_names = bin_names(DENSITY_EDGES)
    out['by_density'] = {
        f'{name} GT/img': block(dense_bin == i) for i, name in enumerate(dense_names)
    }
    strides = stats['pos_stride']
    if strides.size and strides.max() > 0:
        levels = np.unique(strides)
        out['by_level'] = {
            name: {
                f's{int(s)}': float((strides[pos_bin == i] == s).mean())
                for s in levels
                if (pos_bin == i).any()
            }
            for i, name in enumerate(bin_names(size_edges))
        }
    return out


def parse_caps(specs: Sequence[str] | None, names: Sequence[str]) -> dict[str, float]:
    """``['<8:0.02', 'all:0.01']`` -> ``{'<8': 0.02, 'all': 0.01}``; bins from ``names``."""
    caps = {}
    for spec in specs or ():
        bin_name, sep, share = spec.rpartition(':')
        if not sep or (bin_name != 'all' and bin_name not in names):
            raise ValueError(f'--max-starved {spec!r}: expected BIN:SHARE, BIN in all, {names}')
        caps[bin_name] = float(share)
    return caps


def violations(summary: dict, caps: dict[str, float]) -> list[str]:
    """The starvation caps a candidate's summary exceeds, e.g. ``['<8 6.0% > 2%']``."""
    out = []
    for bin_name, cap in caps.items():
        src = summary if bin_name == 'all' else summary['by_size'].get(bin_name, {})
        v = src.get('starved')
        if v is not None and v > cap:
            out.append(f'{bin_name} {100 * v:.1f}% > {100 * cap:g}%')
    return out


def rank(
    results: list[dict], key: str, focus: str | None, caps: dict[str, float] | None = None
) -> list[dict]:
    """Order by ``key`` (overall or on the ``focus`` size bin), best first.

    With ``caps``, every result gets a ``violations`` list and the results
    within all caps come first.
    """
    for r in results:
        r['violations'] = violations(r['summary'], caps or {})

    def value(r: dict) -> tuple[bool, float]:
        src = r['summary']['by_size'].get(focus, {}) if focus else r['summary']
        v = src.get(key, float('nan'))
        if math.isnan(v):
            return bool(r['violations']), math.inf
        return bool(r['violations']), v if key == 'starved' else -v

    return sorted(results, key=value)


# ----------------------------------------------------------------------------- report


def _pct(v: float | None) -> str:
    return '-' if v is None or math.isnan(v) else f'{100 * v:.1f}'


def _num(v: float | None, nd: int = 2) -> str:
    return '-' if v is None or math.isnan(v) else f'{v:.{nd}f}'


def write_markdown(
    path: str, results: list[dict], args: argparse.Namespace, n_images: int, n_gts: int
) -> None:
    """Ranked overview, per-size table and sensitivity table."""
    names = bin_names(args.size_edges)
    focus = args.focus_size
    lines = [
        f'# Assigner probe: {osp.basename(args.config)}',
        '',
        f'- checkpoint: `{args.checkpoint or "none (initialisation)"}`',
        f'- {n_images} training images, {n_gts} GTs'
        + (f' ({results[0]["summary"]["duplicates"]} exact duplicates merged)' if results else '')
        + '; sizes are sqrt(w*h) in px',
        f'- ranked by `{args.rank_by}`' + (f' on GTs {focus} px' if focus else ''),
    ]
    caps = getattr(args, 'max_starved', None)
    if caps:
        lines.append(
            '- starvation caps (`--max-starved`): '
            + ', '.join(f'`{c}`' for c in caps)
            + '; candidates over a cap (marked ✗) are ranked last'
        )
    lines += [
        '',
        '## Ranking',
        '',
        '| # | candidate | score | coverage | starved % | pos/GT | pos IoU | offset | ignored % '
        + ''.join(f'| starved {n} ' for n in names)
        + '|',
        '|---|---|---|---|---|---|---|---|---' + '|---' * len(names) + '|',
    ]
    for i, r in enumerate(results, 1):
        s = r['summary']
        name = r['name'] + (f' ✗ ({"; ".join(r["violations"])})' if r.get('violations') else '')
        lines.append(
            f'| {i} | {name} | {_num(s.get("score"), 3)} | {_num(s["coverage"], 3)} '
            f'| {_pct(s["starved"])} '
            f'| {_num(s["pos_per_gt"], 1)} | {_num(s.get("pos_iou"), 3)} '
            f'| {_num(s["offset"])} | {_pct(s["ignored"])} '
            + ''.join(f'| {_pct(s["by_size"][n].get("starved"))} ' for n in names)
            + '|'
        )
    lines += [
        '',
        '## Positives per GT and their IoU, by GT size',
        '',
        '| candidate ' + ''.join(f'| pos/GT {n} | IoU {n} ' for n in names) + '|',
        '|---' + '|---|---' * len(names) + '|',
    ]
    for r in results:
        bs = r['summary']['by_size']
        lines.append(
            f'| {r["name"]} '
            + ''.join(
                f'| {_num(bs[n].get("pos_per_gt"), 1)} | {_num(bs[n].get("pos_iou"), 3)} '
                for n in names
            )
            + '|'
        )
    dens = list(results[0]['summary']['by_density']) if results else []
    lines += [
        '',
        '## Starved GTs (%) by image density',
        '',
        '| candidate ' + ''.join(f'| {d} ' for d in dens) + '|',
        '|---' + '|---' * len(dens) + '|',
    ]
    for r in results:
        bd = r['summary']['by_density']
        lines.append(
            f'| {r["name"]} ' + ''.join(f'| {_pct(bd[d].get("starved"))} ' for d in dens) + '|'
        )
    if any('by_level' in r['summary'] for r in results):
        lines += ['', '## Share of positives per FPN level (stride), by GT size', '']
        for r in results:
            bl = r['summary'].get('by_level')
            if not bl:
                continue
            cells = '; '.join(
                f'{n}: ' + ' '.join(f'{k} {100 * v:.0f}%' for k, v in bl[n].items())
                for n in names
                if bl.get(n)
            )
            lines.append(f'- **{r["name"]}**: {cells}')
    lines += [
        '',
        'Metrics: see the docstring of `tools/assigner_tuning/assigner_probe.py` and',
        '`tools/assigner_tuning/README.md`. The probe ranks what each assigner *trains on*;',
        'confirm the top candidates with short training runs (`assigner_search.py`).',
        '',
    ]
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


def plot_sensitivity(path: str, results: list[dict], focus: str | None) -> bool:
    """Small multiples, one panel per (family, swept key): score, coverage and pos IoU.

    All three are fractions on one shared y-axis, so panels compare directly.
    """
    groups: dict[tuple[str, str], list[tuple[object, dict]]] = {}
    base = {r['family']: r for r in results if not r['changed']}
    for r in results:
        if len(r['changed']) != 1:
            continue
        ((key, value),) = r['changed'].items()
        groups.setdefault((r['family'], key), []).append((value, r))
    panels = []
    for (family, key), pts in groups.items():
        base_value = get_path(base[family]['cfg'], key) if family in base else None
        if family in base:
            pts = [*pts, (base_value, base[family])]
        pts = [(v, r) for v, r in pts if isinstance(v, int | float) and not isinstance(v, bool)]
        if len(pts) > 1:
            panels.append((family, key, base_value, sorted(pts, key=lambda p: p[0])))
    panels.sort(key=lambda p: (p[0], p[1]))
    if not panels:
        return False
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def get(r: dict, k: str) -> float:
        src = r['summary']['by_size'].get(focus, {}) if focus else r['summary']
        return src.get(k, float('nan'))

    series = (('score', '#2a6fdb', 'o'), ('coverage', '#d9622b', 's'), ('pos_iou', '#2f9e6e', '^'))
    vals = [get(r, k) for *_, pts in panels for _, r in pts for k, _, _ in series]
    vals = [v for v in vals if not math.isnan(v)]
    lo, hi = min(vals), max(vals)
    pad = 0.05 * (hi - lo or 1)
    cols = min(4, len(panels))
    rows = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.7 * rows), squeeze=False)
    ink, muted = '#1f1f1f', '#6b6b6b'
    for ax, (family, key, base_value, pts) in zip(axes.flat, panels, strict=False):
        xs = [p[0] for p in pts]
        for k, color, marker in series:
            ax.plot(
                xs,
                [get(r, k) for _, r in pts],
                marker=marker,
                color=color,
                lw=2,
                ms=6,
                label=k.replace('_', ' '),
            )
        if base_value is not None:
            ax.axvline(base_value, color=muted, lw=1, ls=':')
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(f'{family}: {key}', fontsize=9, color=ink)
        ax.tick_params(labelsize=7, colors=muted)
        ax.grid(axis='y', color='#e4e4e4', lw=0.8)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    for ax in list(axes.flat)[len(panels) :]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', ncol=3, fontsize=8, frameon=False)
    where = f'GTs {focus} px' if focus else 'all GTs'
    fig.suptitle(
        f'Assigner sensitivity ({where}); dotted: config value', fontsize=10, x=0.02, ha='left'
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


# ----------------------------------------------------------------------------- main


def build_dataset(cfg: Config, split: str) -> object:
    """Training images through the training pipeline (augmentations included)."""
    loader = cfg.get(f'{split}_dataloader')
    ds_cfg = copy.deepcopy(loader.dataset)
    return DATASETS.build(ds_cfg)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command line."""
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('config')
    p.add_argument('--checkpoint', help='trained weights (strongly advised, see docstring)')
    p.add_argument('--sweep', action='append', default=[], help='key=v1,v2 on the config assigner')
    p.add_argument('--spec', help='YAML file of assigner families to compare')
    p.add_argument('--no-config-assigner', action='store_true', help='probe only --spec')
    p.add_argument('--mode', choices=('oat', 'grid'), default='oat')
    p.add_argument('--num-images', type=int, default=120)
    p.add_argument('--split', default='train', help='dataloader whose dataset is sampled')
    p.add_argument('--head', default='bbox_head', help='model attribute holding the assigner')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--size-edges', type=float, nargs='+', default=list(DEFAULT_SIZE_EDGES))
    p.add_argument('--rank-by', choices=RANK_KEYS, default='score')
    p.add_argument('--focus-size', help='size bin to rank on, e.g. "<8" or "8-12"')
    p.add_argument(
        '--max-starved',
        action='append',
        help='BIN:SHARE cap on starved GTs, e.g. "<8:0.02" or "all:0.01" (repeatable)',
    )
    p.add_argument('--out-dir', help='default work_dirs/assigner_probe/<config name>')
    p.add_argument('--rerank', action='store_true', help='re-rank <out-dir>/probe.json only')
    p.add_argument('--cfg-options', nargs='+', action=DictAction)
    return p.parse_args(argv)


def gather_candidates(cfg: Config, args: argparse.Namespace) -> list[Candidate]:
    """Config assigner (+ ``--sweep``) and ``--spec`` families."""
    cands: list[Candidate] = []
    if not args.no_config_assigner:
        base = cfg.model.train_cfg.assigner.to_dict()
        family = base['type'].split('.')[-1]
        cands += expand_family(family, base, parse_sweep(args.sweep), args.mode)
    if args.spec:
        cands += load_spec(args.spec, args.mode)
    names = [c.name for c in cands]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise ValueError(f'duplicate candidate names {sorted(dup)}; rename the spec families')
    return cands


def main(argv: Sequence[str] | None = None) -> None:
    """Probe every candidate and write probe.md / probe.json / sensitivity.png."""
    args = parse_args(argv)
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    init_default_scope(cfg.get('default_scope', 'mmrotate'))
    if args.focus_size and args.focus_size not in bin_names(args.size_edges):
        raise ValueError(f'--focus-size must be one of {bin_names(args.size_edges)}')
    parse_caps(args.max_starved, bin_names(args.size_edges))  # fail before the forward passes
    out_dir = args.out_dir or osp.join(
        'work_dirs', 'assigner_probe', osp.splitext(osp.basename(args.config))[0]
    )
    os.makedirs(out_dir, exist_ok=True)
    if args.rerank:
        rerank(out_dir, args)
        return

    cands = gather_candidates(cfg, args)
    assigners: list[tuple[Candidate, object]] = [(c, TASK_UTILS.build(c.cfg)) for c in cands]
    print(f'{len(cands)} candidates')

    dataset = build_dataset(cfg, args.split)
    indices = pick_images(dataset, args.num_images, args.seed)
    t0 = time.time()
    samples = collect_samples(
        cfg, args.checkpoint, indices, dataset, args.device, args.head, args.seed
    )
    n_gts = sum(len(s.gt) for s in samples)
    print(f'{len(samples)} images, {n_gts} GTs cached in {time.time() - t0:.0f}s')

    results = []
    for c, assigner in assigners:
        t1 = time.time()
        stats = assignment_stats(assigner, samples, args.device)
        summary = summarize(stats, args.size_edges)
        results.append(
            dict(name=c.name, family=c.family, cfg=c.cfg, changed=c.changed, summary=summary)
        )
        print(
            f'  {c.name}: coverage {summary["coverage"]:.3f}, starved '
            f'{100 * summary["starved"]:.1f}%, pos IoU {summary.get("pos_iou", float("nan")):.3f}'
            f' ({time.time() - t1:.0f}s)',
            flush=True,
        )

    write_outputs(out_dir, results, args, len(samples), n_gts)


def write_outputs(
    out_dir: str, results: list[dict], args: argparse.Namespace, n_images: int, n_gts: int
) -> None:
    """probe.md, probe.json and sensitivity.png of the (unranked) results."""
    caps = parse_caps(args.max_starved, bin_names(args.size_edges))
    ranked = rank(results, args.rank_by, args.focus_size, caps)
    write_markdown(osp.join(out_dir, 'probe.md'), ranked, args, n_images, n_gts)
    with open(osp.join(out_dir, 'probe.json'), 'w') as f:
        json.dump(
            dict(
                config=args.config,
                checkpoint=args.checkpoint,
                images=n_images,
                gts=n_gts,
                results=ranked,
            ),
            f,
            indent=1,
            default=str,
        )
    plotted = plot_sensitivity(osp.join(out_dir, 'sensitivity.png'), results, args.focus_size)
    print(f'wrote {out_dir}/probe.md, probe.json' + (', sensitivity.png' if plotted else ''))


def rerank(out_dir: str, args: argparse.Namespace) -> None:
    """Rewrite the outputs of an earlier probe (other ``--rank-by`` / ``--focus-size``)."""
    with open(osp.join(out_dir, 'probe.json')) as f:
        data = json.load(f)
    for r in data['results']:  # probes written before ``score`` existed
        for block in (r['summary'], *r['summary']['by_size'].values()):
            if 'pos_iou' in block:
                block['score'] = block['coverage'] * block['pos_iou']
    args.checkpoint = data['checkpoint']
    write_outputs(out_dir, data['results'], args, data['images'], data['gts'])


if __name__ == '__main__':
    main()
