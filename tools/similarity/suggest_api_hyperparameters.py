# Copyright (c) OpenMMLab. All rights reserved.
"""Suggest similarity-service settings for an mmrotate model from its own val/test data.

    python -m tools.similarity.suggest_api_hyperparameters CONFIG WEIGHTS [--split test]

Runs the model over a sample of the config's ``test`` (or ``val``) dataset, exactly as the
API's similarity service would (same input size, padding and feature code in ``api/``), and
scores candidate feature settings with the qgis_annotations similarity measures
(``tools/similarity/metrics.py``):

- stage A, *representation*: every source (backbone, neck, head cls / reg features) x
  {concat of all levels, mean of all levels, each single level}, centred, at ``--base-stride``;
- stage B, *resolution*: the ``--top-k`` best of A x ``--strides``;
- stage C, *compression*: the best of B x {none, center, PCA to each of ``--dims``, whitened
  PCA} x {float32, int8}.

The images are split in two halves: the *fit* half fits the centring / PCA, the ridge
probe and the class prototypes, the *eval* half is scored. Besides retrieval APs, every
candidate is scored for **false alarms per image** (FPPI): each class's score map over
whole eval images (cosine to the class prototype, as the brush does, and the ridge model,
as trained classes do) is read as a detector whose local maxima are detections. Pass
``--rank-by fppi`` to choose by false alarms instead of AP, ``--no-false-alarms`` to skip.

Writes ``suggestions.md``, ``suggestions.json`` and ``similarity_profile.npz`` (the winner,
with its fitted projection) to ``--out``, and prints the command that serves it.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.registry import init_default_scope
from torch import Tensor

from api import tiling
from api.features import (
    FeatureExtractor,
    FeatureSpec,
    Moments,
    Projector,
    batch_inputs,
    fuse,
)
from api.model import LoadedModel, load
from api.profile import save_profile
from api.settings import parse_size
from mmrotate.registry import DATASETS
from mmrotate.structures.bbox import rbox2qbox
from tools.similarity import metrics

SOURCES = ('backbone', 'neck', 'head_cls', 'head_reg')


# --------------------------------------------------------------------------- data
@dataclass
class Sample:
    """One model window of a dataset image, with its objects in window pixels."""

    img: np.ndarray  # BGR uint8, at most the input size
    boxes: np.ndarray  # [K, 5] cx, cy, w, h, angle
    labels: np.ndarray  # [K]
    fit: bool
    name: str = ''


def _unwrap(ds_cfg: dict) -> dict:
    while 'pipeline' not in ds_cfg and ('dataset' in ds_cfg or 'datasets' in ds_cfg):
        ds_cfg = ds_cfg.get('dataset') or ds_cfg['datasets'][0]
    return ds_cfg


def build_dataset(cfg: Config, split: str, data_root: str | None) -> object:
    """The config's ``{split}_dataloader`` dataset, loading images and boxes only."""
    loader = cfg.get(f'{split}_dataloader')
    if not loader:
        raise SystemExit(f'the config has no {split}_dataloader')
    ds_cfg = copy.deepcopy(_unwrap(loader['dataset']))
    if data_root:
        ds_cfg['data_root'] = data_root
    keep = ('LoadImageFromFile', 'LoadAnnotations', 'ConvertBoxType')
    pipeline = [
        s for s in ds_cfg.get('pipeline', []) if str(s.get('type', '')).split('.')[-1] in keep
    ]
    kinds = [str(s['type']).split('.')[-1] for s in pipeline]
    if 'LoadImageFromFile' not in kinds:
        pipeline.insert(0, {'type': 'mmdet.LoadImageFromFile'})
    if 'LoadAnnotations' not in kinds:
        pipeline.insert(1, {'type': 'mmdet.LoadAnnotations', 'with_bbox': True, 'box_type': 'qbox'})
    ds_cfg['pipeline'] = pipeline
    ds_cfg['test_mode'] = False
    ds_cfg.pop('filter_cfg', None)
    init_default_scope(cfg.get('default_scope', 'mmrotate'))
    return DATASETS.build(ds_cfg)


def _rboxes(results: dict) -> tuple[np.ndarray, np.ndarray]:
    boxes = results.get('gt_bboxes')
    labels = results.get('gt_bboxes_labels')
    if boxes is None or labels is None or len(labels) == 0:
        return np.zeros((0, 5), np.float32), np.zeros((0,), np.int64)
    if hasattr(boxes, 'convert_to'):
        boxes = boxes.convert_to('rbox')
    t = boxes.tensor if hasattr(boxes, 'tensor') else torch.as_tensor(boxes)
    b = t.float().numpy().reshape(-1, 5)
    lab = np.asarray(labels, np.int64)
    ignore = results.get('gt_ignore_flags')
    if ignore is not None and len(ignore) == len(lab):
        keep = ~np.asarray(ignore, bool)
        b, lab = b[keep], lab[keep]
    return b, lab


def sample_images(
    dataset: object, lm: LoadedModel, num: int, seed: int
) -> tuple[list[Sample], int]:
    """``num`` windows with at least one object, alternately assigned to fit / eval."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(dataset))  # type: ignore[arg-type]
    win_w, win_h = lm.input_size
    out: list[Sample] = []
    tried = 0
    for idx in order:
        if len(out) >= num:
            break
        tried += 1
        results = dataset[int(idx)]  # type: ignore[index]
        img = results['img']
        boxes, labels = _rboxes(results)
        if not len(labels):
            continue
        h, w = img.shape[:2]
        x0 = y0 = 0
        if w > win_w or h > win_h:  # a window around a random object
            cx, cy = boxes[rng.integers(len(boxes)), :2]
            x0 = int(np.clip(cx - win_w / 2, 0, max(0, w - win_w)))
            y0 = int(np.clip(cy - win_h / 2, 0, max(0, h - win_h)))
        crop = img[y0 : y0 + win_h, x0 : x0 + win_w]
        b = boxes.copy()
        b[:, 0] -= x0
        b[:, 1] -= y0
        ch, cw = crop.shape[:2]
        inside = (b[:, 0] >= 0) & (b[:, 1] >= 0) & (b[:, 0] < cw) & (b[:, 1] < ch)
        if not inside.any():
            continue
        out.append(
            Sample(
                img=np.ascontiguousarray(crop),
                boxes=b[inside],
                labels=labels[inside],
                fit=len(out) % 2 == 0,
                name=str(results.get('img_path', idx)),
            )
        )
    return out, tried


# --------------------------------------------------------------------------- cells
@dataclass
class Cells:
    """Which grid cells of one sample are objects, background and fit cells at a stride."""

    obj: np.ndarray  # [K, 2] (row, col)
    obj_labels: np.ndarray
    obj_sizes: np.ndarray  # sqrt(w * h), model px
    bg: np.ndarray  # [M, 2]
    fit: np.ndarray  # [F, 2]


def object_map(s: Sample, stride: int) -> np.ndarray:
    """``[ceil(h/s), ceil(w/s)]`` int32: the index of the object covering each cell, -1 on
    background. Boxes are grown by one cell, so even a sub-cell object owns a cell."""
    h, w = s.img.shape[:2]
    out = np.full((math.ceil(h / stride), math.ceil(w / stride)), -1, np.int32)
    if not len(s.boxes):
        return out
    grown = torch.as_tensor(s.boxes.copy())
    grown[:, 2:4] += stride
    quads = rbox2qbox(grown).numpy().reshape(-1, 4, 2) / stride - 0.5
    # largest first, so a small object on a big one keeps its own cells
    for i in np.argsort(-(s.boxes[:, 2] * s.boxes[:, 3])):
        cv2.fillPoly(out, [np.round(quads[i]).astype(np.int32)], int(i))
    return out


def make_cells(
    s: Sample, stride: int, max_objects: int, n_bg: int, n_fit: int, rng: np.random.Generator
) -> Cells:
    h, w = s.img.shape[:2]
    hc, wc = math.ceil(h / stride), math.ceil(w / stride)
    k = len(s.labels)
    pick = rng.permutation(k)[: min(k, max_objects)]
    b = s.boxes[pick]
    rows = np.clip((b[:, 1] // stride).astype(int), 0, hc - 1)
    cols = np.clip((b[:, 0] // stride).astype(int), 0, wc - 1)
    # background: cells outside every (slightly grown) box
    free = np.argwhere(object_map(s, stride) < 0)
    bg = free[rng.permutation(len(free))[: min(len(free), n_bg)]]
    allc = np.argwhere(np.ones((hc, wc), bool))
    fit = allc[rng.permutation(len(allc))[: min(len(allc), n_fit)]]
    return Cells(
        obj=np.stack([rows, cols], 1),
        obj_labels=s.labels[pick],
        obj_sizes=np.sqrt(np.abs(b[:, 2] * b[:, 3])),
        bg=bg,
        fit=fit,
    )


# --------------------------------------------------------------------------- candidates
@dataclass
class Candidate:
    spec: FeatureSpec
    stage: str
    moments: Moments | None = None
    obj: dict[bool, list[Tensor]] = field(default_factory=lambda: {True: [], False: []})
    obj_y: dict[bool, list[Tensor]] = field(default_factory=lambda: {True: [], False: []})
    obj_size: dict[bool, list[Tensor]] = field(default_factory=lambda: {True: [], False: []})
    bg: dict[bool, list[Tensor]] = field(default_factory=lambda: {True: [], False: []})

    def take(self, x: Tensor, cells: Cells, fit: bool) -> None:
        """``x`` ``[h, w, D]`` (L2-normalised if the spec says so) -> stored samples."""
        o = torch.as_tensor(cells.obj, device=x.device)
        b = torch.as_tensor(cells.bg, device=x.device)
        self.obj[fit].append(x[o[:, 0], o[:, 1]].half().cpu())
        self.obj_y[fit].append(torch.as_tensor(cells.obj_labels))
        self.obj_size[fit].append(torch.as_tensor(cells.obj_sizes, dtype=torch.float32))
        if len(b):
            self.bg[fit].append(x[b[:, 0], b[:, 1]].half().cpu())
        if fit:
            f = torch.as_tensor(cells.fit, device=x.device)
            v = x[f[:, 0], f[:, 1]]
            if self.moments is None:
                self.moments = Moments.zeros(v.shape[1], x.device)
            self.moments.add(v)

    def tensors(self, fit: bool, device: torch.device) -> tuple[Tensor, ...]:
        def cat(xs: list[Tensor], dim: int | None = None) -> Tensor:
            if xs:
                return torch.cat(xs).to(device)
            return torch.zeros((0, dim or 1), device=device)

        dim = self.obj[True][0].shape[1] if self.obj[True] else 1
        return (
            cat(self.obj[fit], dim).float(),
            cat(self.obj_y[fit]).long(),
            cat(self.obj_size[fit]).float(),
            cat(self.bg[fit], dim).float(),
        )


def collect(
    lm: LoadedModel,
    samples: Sequence[Sample],
    candidates: Sequence[Candidate],
    args: argparse.Namespace,
) -> float:
    """One forward pass over ``samples`` feeding every candidate. Returns ms per window."""
    extractor = FeatureExtractor(lm.model)
    parts = list(dict.fromkeys(p for c in candidates for p in c.spec.parts))
    cell_cache: dict[tuple[int, int], Cells] = {}
    t_model = 0.0
    for start in range(0, len(samples), args.batch):
        batch = samples[start : start + args.batch]
        crops = [lm.window_image(s.img) for s in batch]
        with torch.inference_mode(), lm.autocast():
            t0 = time.perf_counter()
            inputs = batch_inputs(lm.model, crops)
            captured = extractor.capture(inputs, parts)
            if inputs.is_cuda:
                torch.cuda.synchronize()
            t_model += time.perf_counter() - t0
            for cand in candidates:
                stride = cand.spec.out_stride
                fused = fuse(captured, cand.spec, tuple(inputs.shape[-2:])).float()
                x = fused.permute(0, 2, 3, 1)
                if cand.spec.l2_before_projection:
                    x = F.normalize(x, dim=-1)
                for b, s in enumerate(batch):
                    key = (start + b, stride)
                    if key not in cell_cache:
                        rng = np.random.default_rng([args.seed, start + b, stride])
                        cell_cache[key] = make_cells(
                            s, stride, args.max_objects, args.bg_per_image, args.fit_cells, rng
                        )
                    cand.take(x[b], cell_cache[key], s.fit)
    return 1000 * t_model / max(1, len(samples))


@dataclass
class Scorer:
    """One evaluated variant (candidate + projection + dtype) and its fitted class models,
    with the peaks of its class maps collected by :func:`false_alarm_pass`."""

    cand: Candidate
    projector: Projector | None
    dtype: str
    prototypes: Tensor  # [C, K] unit class means of the fit objects (the brush)
    ridge: Tensor  # [K + 1, C] (trained classes)
    has_fit: Tensor  # [C] bool
    row: dict = field(default_factory=dict)
    # per kind and class: [(object scores, false-alarm values, false-alarm codes)] per image
    events: dict[str, list[list[tuple[np.ndarray, ...]]]] = field(default_factory=dict)

    def transform(self, x: Tensor) -> Tensor:
        if self.projector is not None:
            x = self.projector.apply(x)
        x = F.normalize(x, dim=-1)
        if self.dtype == 'int8':
            x = torch.clamp(torch.round(x * 127), -127, 127) / 127
        return x

    def class_maps(self, x: Tensor) -> dict[str, Tensor]:
        """``[h, w, D]`` (pre-projection) -> ``{'brush' | 'ridge': [C, h, w]}``."""
        y = self.transform(x)
        return {
            'brush': (y @ self.prototypes.T).permute(2, 0, 1),
            'ridge': metrics.ridge_scores(self.ridge, y).permute(2, 0, 1),
        }


def evaluate(
    cand: Candidate,
    projection: str,
    dim: int | None,
    whiten: bool,
    dtype: str,
    num_classes: int,
    device: torch.device,
    max_queries: int,
) -> Scorer:
    projector = None
    if projection != 'none':
        assert cand.moments is not None
        projector = Projector.from_moments(cand.moments, projection, dim, whiten)
    fo, fy, _, fb = cand.tensors(True, device)
    eo, ey, es, eb = cand.tensors(False, device)
    scorer = Scorer(cand, projector, dtype, fo[:0], fo[:0], fy[:0].bool())
    fo, fb, eo, eb = (scorer.transform(t) for t in (fo, fb, eo, eb))
    row: dict = {}
    row.update(metrics.object_ap(eo, ey, es, eb, max_queries=max_queries, num_classes=num_classes))
    fit_x = torch.cat([fo, fb])
    fit_y = torch.cat([fy, torch.full((fb.shape[0],), -1, device=device, dtype=torch.long)])
    ev_x = torch.cat([eo, eb])
    ev_y = torch.cat([ey, torch.full((eb.shape[0],), -1, device=device, dtype=torch.long)])
    row.update(metrics.ridge_ap(fit_x, fit_y, ev_x, ev_y, num_classes))
    scorer.ridge = metrics.ridge_fit(fit_x, fit_y, num_classes)
    scorer.has_fit = metrics.class_presence(fy, num_classes)
    protos = torch.zeros(num_classes, fo.shape[1], device=device)
    protos.index_add_(0, fy, fo)
    scorer.prototypes = F.normalize(protos, dim=-1)
    row.update(metrics.geometry(eo, ey, eb))
    if dtype == 'int8':
        row.update(metrics.int8_agreement(_float_units(cand, projector, device)))
    out_dim = eo.shape[1]
    row['dim'] = int(out_dim)
    row['score'] = float(np.nanmean([row.get('obj_ap', np.nan), row.get('ridge_ap', np.nan)]))
    row['kb_per_512'] = metrics.grid_bytes_per_512(cand.spec.out_stride, out_dim, dtype) / 1024
    if projector is not None and projector.explained_variance is not None and cand.moments:
        total = float(cand.moments.explained().sum())
        row['var_kept'] = float(projector.explained_variance.sum()) / max(total, 1e-12)
    scorer.row = row
    return scorer


KINDS = ('brush', 'ridge')


def false_alarm_pass(
    lm: LoadedModel,
    samples: Sequence[Sample],
    scorers: Sequence[Scorer],
    args: argparse.Namespace,
) -> None:
    """Score whole eval images with every scorer's class maps; adds FPPI metrics to rows.

    One forward pass over the eval half. Every object's best cell on its class map is a hit;
    per image, class and kind the ``--fa-max-peaks`` highest local maxima off the class's
    objects are its false alarms, so an FPPI above that is reported as unreachable.
    """
    if not scorers:
        return
    evals = [s for s in samples if not s.fit]
    num_classes = len(lm.classes)
    extractor = FeatureExtractor(lm.model)
    by_cand: dict[int, list[Scorer]] = {}
    for sc in scorers:
        by_cand.setdefault(id(sc.cand), []).append(sc)
        sc.events = {k: [[] for _ in range(num_classes)] for k in KINDS}
    cands = [group[0].cand for group in by_cand.values()]
    parts = list(dict.fromkeys(p for c in cands for p in c.spec.parts))
    n_gt = np.zeros(num_classes, int)
    for s in evals:
        n_gt += np.bincount(s.labels, minlength=num_classes)[:num_classes]
    for start in range(0, len(evals), args.batch):
        batch = evals[start : start + args.batch]
        crops = [lm.window_image(s.img) for s in batch]
        with torch.inference_mode(), lm.autocast():
            inputs = batch_inputs(lm.model, crops)
            captured = extractor.capture(inputs, parts)
            for group in by_cand.values():
                cand = group[0].cand
                stride = cand.spec.out_stride
                window = metrics.peak_window(stride, args.fa_peak_px)
                x = fuse(captured, cand.spec, tuple(inputs.shape[-2:])).float()
                x = x.permute(0, 2, 3, 1)
                if cand.spec.l2_before_projection:
                    x = F.normalize(x, dim=-1)
                for b, s in enumerate(batch):
                    omap = torch.as_tensor(object_map(s, stride), device=x.device).long()
                    hc, wc = omap.shape
                    labels = torch.as_tensor(s.labels, device=x.device).long()
                    for sc in group:
                        for kind, maps in sc.class_maps(x[b, :hc, :wc]).items():
                            obj, vals, codes, ok = metrics.class_events(
                                maps.float(), omap, labels, window, args.fa_max_peaks
                            )
                            obj, vals, codes, ok = (t.cpu().numpy() for t in (obj, vals, codes, ok))
                            for c in range(num_classes):
                                sc.events[kind][c].append(
                                    (obj[s.labels == c], vals[c][ok[c]], codes[c][ok[c]])
                                )
    for sc in scorers:
        for kind in KINDS:
            curves = []
            for c in range(num_classes):
                if n_gt[c] < args.fa_min_objects or not bool(sc.has_fit[c]):
                    curves.append(None)
                    continue
                obj, vals, codes = (
                    np.concatenate([e[i] for e in sc.events[kind][c]]) for i in range(3)
                )
                curves.append(metrics.fppi_curve(obj, vals, codes, int(n_gt[c]), len(evals)))
            summary = metrics.fppi_summary(curves, args.fa_recalls, args.fa_fppi)
            sc.row.update({f'{k}_{kind}': v for k, v in summary.items()})
        first = f'fppi@r{args.fa_recalls[0]:g}'
        sc.row['fppi'] = float(np.mean([sc.row.get(f'{first}_{k}', math.nan) for k in KINDS]))
        sc.events = {}


def rank_value(row: dict, rank_by: str) -> float:
    """Higher is better: the AP score, or minus the headline FPPI."""
    if rank_by == 'fppi':
        v = row.get('fppi', math.nan)
        return -v if not math.isnan(v) else -math.inf
    return row['score']


def _float_units(cand: Candidate, projector: Projector | None, device: torch.device) -> Tensor:
    eo, _, _, eb = cand.tensors(False, device)
    x = torch.cat([eo, eb])
    if projector is not None:
        x = projector.apply(x)
    return F.normalize(x, dim=-1)


# --------------------------------------------------------------------------- stages
def probe_sources(
    lm: LoadedModel, sample: Sample, sources: Iterable[str]
) -> dict[str, list[tuple[int, int]]]:
    """``{source: [(channels, stride), ...]}`` for every source the model has."""
    extractor = FeatureExtractor(lm.model)
    out = {}
    with torch.inference_mode(), lm.autocast():
        inputs = batch_inputs(lm.model, [lm.window_image(sample.img)])
        H = inputs.shape[-2]
        for src in sources:
            try:
                maps = extractor.capture(inputs, [(src, None)])[src]
            except (ValueError, RuntimeError, AttributeError) as e:
                print(f'  skip source {src}: {e}')
                continue
            out[src] = [(int(m.shape[1]), round(H / m.shape[-2])) for m in maps]
    return out


def stage_a_specs(info: dict[str, list[tuple[int, int]]], stride: int) -> list[FeatureSpec]:
    specs = []
    for src, levels in info.items():
        n = len(levels)
        base = FeatureSpec(source=src, out_stride=stride, projection='center', dtype='float32')
        if n > 1:
            specs.append(base.replace(fusion='concat'))
            if len({c for c, _ in levels}) == 1:
                specs.append(base.replace(fusion='mean'))
        for lv in range(n):
            specs.append(base.replace(fusion='single', levels=(lv,)))
    return specs


def fmt(v: object, nd: int = 3) -> str:
    if isinstance(v, float):
        return '-' if math.isnan(v) else f'{v:.{nd}f}'
    return str(v)


def table(rows: list[dict], extra: Sequence[str] = ()) -> str:
    cols = ['tag', 'dim', 'score', 'obj_ap', 'ridge_ap']
    cols += [f'obj_ap_{b}' for b in metrics.bucket_names()]
    cols += ['bg_paircos', 'same_cos', 'other_cos', *extra, 'kb_per_512']
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for r in rows:
        lines.append('| ' + ' | '.join(fmt(r.get(c, float('nan'))) for c in cols) + ' |')
    return '\n'.join(lines)


def fa_columns(args_like: dict) -> list[str]:
    cols = []
    for kind in KINDS:
        cols.append(f'map_ap_{kind}')
        cols += [f'fppi@r{r:g}_{kind}' for r in args_like['fa_recalls']]
        cols += [f'r@fppi{p:g}_{kind}' for p in args_like['fa_fppi']]
        cols.append(f'fa_other_class_{kind}')
    return cols


def fa_table(rows: list[dict], args_like: dict) -> str:
    cols = ['tag', 'fppi', *fa_columns(args_like)]
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, float('nan'))
            cells.append('>cap' if isinstance(v, float) and math.isinf(v) else fmt(v, 2))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def per_class_table(rep: dict, row: dict | None = None) -> str:
    """Per class: AP, recall at each false-alarm budget and false alarms needed for each
    recall, for the brush and for trained classes (``row`` defaults to the recommendation).
    """
    row = row or rep['recommendation']
    fa = rep.get('false_alarms')
    cols, keys = ['class', 'eval objects'], []
    for kind, ap_key in (('brush', 'obj_ap_per_class'), ('ridge', 'ridge_ap_per_class')):
        cols.append(f'{kind} AP')
        keys.append((ap_key, 3))
        if fa:
            cols.append(f'{kind} map AP')
            keys.append((f'map_ap_per_class_{kind}', 3))
            for p in fa['fa_fppi']:
                cols.append(f'{kind} R@FA{p:g}')
                keys.append((f'r@fppi{p:g}_per_class_{kind}', 2))
            for r in fa['fa_recalls']:
                cols.append(f'{kind} FA@R{r:g}')
                keys.append((f'fppi@r{r:g}_per_class_{kind}', 2))
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for c, name in enumerate(rep['classes']):
        cells = [name, str(rep['eval_objects'][c])]
        for key, nd in keys:
            vals = row.get(key) or []
            v = vals[c] if c < len(vals) else None
            if v is None or (isinstance(v, float) and math.isnan(v)):
                cells.append('-')
            elif math.isinf(v):
                cells.append('>cap')
            else:
                cells.append(f'{v:.{nd}f}')
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def best_within(rows: list[dict], tolerance: float, rank_by: str = 'score') -> dict:
    """The cheapest row (smallest grid) whose rank value is within ``tolerance`` of the
    best: absolute for the AP score, relative (a fraction) for FPPI."""
    top = max(rank_value(r, rank_by) for r in rows)
    if rank_by == 'fppi' and math.isfinite(top):
        floor = top * (1 + tolerance)  # top is -fppi: allow (1 + tol) x the lowest FPPI
    else:
        floor = top - tolerance
    close = [r for r in rows if rank_value(r, rank_by) >= floor]
    return min(close, key=lambda r: (r['kb_per_512'], -rank_value(r, rank_by)))


# --------------------------------------------------------------------------- main
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument('config')
    p.add_argument('weights')
    p.add_argument('--split', default='test', choices=['test', 'val'])
    p.add_argument('--data-root', help="override the dataset's data_root")
    p.add_argument('--num-images', type=int, default=200)
    p.add_argument('--max-objects', type=int, default=64, help='object cells per image')
    p.add_argument('--bg-per-image', type=int, default=50, help='background cells per image')
    p.add_argument('--fit-cells', type=int, default=512, help='PCA fit cells per fit image')
    p.add_argument('--max-queries', type=int, default=2000)
    p.add_argument('--sources', nargs='+', default=list(SOURCES), choices=SOURCES)
    p.add_argument('--base-stride', type=int, default=8)
    p.add_argument('--strides', nargs='+', type=int, default=[4, 8, 16])
    p.add_argument('--dims', nargs='+', type=int, default=[64, 128, 256])
    p.add_argument('--top-k', type=int, default=3, help='stage A winners taken to stage B')
    p.add_argument(
        '--tolerance',
        type=float,
        default=0.005,
        help='prefer the smallest grid among results this close to the best score',
    )
    p.add_argument(
        '--wire-dtype',
        default='int8',
        choices=['int8', 'float16', 'float32'],
        help='dtype the recommendation must use (qgis_annotations reads int8 only)',
    )
    p.add_argument(
        '--rank-by',
        default='score',
        choices=['score', 'fppi'],
        help='score: mean of obj AP and ridge AP; fppi: fewest false alarms per image at the '
        'first --fa-recalls point (mean of brush and ridge)',
    )
    p.add_argument(
        '--fa-tolerance',
        type=float,
        default=0.1,
        help='with --rank-by fppi: prefer the smallest grid within this fraction of the best',
    )
    p.add_argument('--no-false-alarms', action='store_true', help='skip the FPPI pass')
    p.add_argument('--fa-recalls', nargs='+', type=float, default=[0.5, 0.7])
    p.add_argument('--fa-fppi', nargs='+', type=float, default=[1.0, 10.0])
    p.add_argument(
        '--fa-peak-px',
        type=float,
        default=24.0,
        help='a score-map peak suppresses others within about this many model px',
    )
    p.add_argument(
        '--fa-max-peaks',
        type=int,
        default=100,
        help='false-alarm peaks kept per image and class (an FPPI above it is unreachable)',
    )
    p.add_argument(
        '--fa-min-objects',
        type=int,
        default=10,
        help='classes with fewer eval objects are left out of the FPPI means',
    )
    p.add_argument('--input-size', help='as the API flag (default: from the test pipeline)')
    p.add_argument('--pad-mode', default='square', choices=['square', 'divisor'])
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--precision', default='fp32', choices=['fp32', 'fp16', 'bf16'])
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--gsd', type=float, help='model GSD, only copied into the printed command')
    p.add_argument('--out', help='output dir (default work_dirs/similarity/<config stem>)')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> dict:
    t_start = time.time()
    if args.no_false_alarms:
        args.rank_by = 'score'
    out_dir = Path(args.out or Path('work_dirs/similarity') / Path(args.config).stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    lm = load(
        args.config,
        args.weights,
        device=args.device,
        precision=args.precision,
        input_size=parse_size(args.input_size),
        pad_mode=args.pad_mode,
    )
    device = torch.device(args.device)
    for side in lm.input_size:
        tiling.check_divisible(side, args.base_stride, 'input size', '--base-stride')
    strides = [s for s in args.strides if all(side % s == 0 for side in lm.input_size)]
    print(f'model {lm.model_name}: input {lm.input_size} ({lm.input_size_source})')

    dataset = build_dataset(lm.cfg, args.split, args.data_root)
    samples, tried = sample_images(dataset, lm, args.num_images, args.seed)
    if len(samples) < 4:
        raise SystemExit(f'only {len(samples)} images with objects found (tried {tried})')
    n_obj = sum(len(s.labels) for s in samples)
    print(f'{len(samples)} images ({tried} tried), {n_obj} objects; half fit, half eval')
    nc = len(lm.classes)

    info = probe_sources(lm, samples[0], args.sources)
    print('sources:', {k: [f'{c}ch/s{s}' for c, s in v] for k, v in info.items()})
    report: dict = {
        'config': args.config,
        'weights': args.weights,
        'split': args.split,
        'images': len(samples),
        'objects': n_obj,
        'input_size': list(lm.input_size),
        'pad_mode': lm.pad_mode,
        'classes': lm.classes,
        'eval_objects': np.bincount(
            np.concatenate([s.labels for s in samples if not s.fit]), minlength=nc
        )[:nc].tolist(),
        'sources': info,
        'rank_by': args.rank_by,
        'false_alarms': None
        if args.no_false_alarms
        else {
            'fa_recalls': args.fa_recalls,
            'fa_fppi': args.fa_fppi,
            'peak_px': args.fa_peak_px,
            'max_peaks': args.fa_max_peaks,
            'eval_images': sum(1 for s in samples if not s.fit),
        },
        'stages': {},
    }

    def run_stage(name: str, specs: list[FeatureSpec]) -> tuple[list[dict], list[Candidate]]:
        cands = [Candidate(spec=s, stage=name) for s in specs]
        ms = collect(lm, samples, cands, args)
        scorers = [
            evaluate(c, 'center', None, False, 'float32', nc, device, args.max_queries)
            for c in cands
        ]
        if not args.no_false_alarms:
            false_alarm_pass(lm, samples, scorers, args)
        rows = []
        for c, sc in zip(cands, scorers, strict=True):
            sc.row['tag'] = c.spec.tag()
            sc.row['spec'] = c.spec.to_dict()
            rows.append(sc.row)
        print(f'stage {name}: {len(cands)} candidates, model {ms:.1f} ms/window')
        report['ms_per_window'] = ms
        return rows, cands

    # A: representation
    rows_a, _ = run_stage('A', stage_a_specs(info, args.base_stride))
    rows_a.sort(key=lambda r: -rank_value(r, args.rank_by))
    report['stages']['A'] = rows_a
    for r in rows_a[:5]:
        print(f'  {r["tag"]:40s} score {fmt(r["score"])}  fppi {fmt(r.get("fppi"), 2)}')

    # B: resolution
    specs_b, seen = [], set()
    for r in rows_a[: args.top_k]:
        for s in strides:
            spec = FeatureSpec.from_dict(r['spec']).replace(out_stride=s)
            if spec.tag() not in seen:
                seen.add(spec.tag())
                specs_b.append(spec)
    rows_b, cands_b = run_stage('B', specs_b)
    order = sorted(range(len(rows_b)), key=lambda i: -rank_value(rows_b[i], args.rank_by))
    rows_b = [rows_b[i] for i in order]
    cands_b = [cands_b[i] for i in order]
    report['stages']['B'] = rows_b
    for r in rows_b[:5]:
        print(f'  {r["tag"]:40s} score {fmt(r["score"])}  fppi {fmt(r.get("fppi"), 2)}')

    # C: compression, on the moments and samples already collected for B's winner
    best = cands_b[0]
    D = rows_b[0]['dim']
    variants: list[tuple[str, int | None, bool]] = [('none', None, False), ('center', None, False)]
    variants += [('pca', d, False) for d in args.dims if d < D]
    wdim = min(128, D)
    variants.append(('pca', wdim, True))
    scorers_c = []
    dtypes = ('float32', args.wire_dtype) if args.wire_dtype != 'float32' else ('float32',)
    for proj, dim, whiten in variants:
        for dtype in dtypes:
            sc = evaluate(best, proj, dim, whiten, dtype, nc, device, args.max_queries)
            spec = best.spec.replace(projection=proj, dtype=dtype)
            label = proj if proj != 'pca' else f'pca{dim}' + ('w' if whiten else '')
            sc.row['tag'] = f'{spec.tag()}{"" if proj != "pca" else label[3:]}-{dtype}'
            sc.row['projection'] = label
            sc.row['dtype'] = dtype
            sc.row['spec'] = spec.to_dict()
            scorers_c.append(sc)
    if not args.no_false_alarms:
        false_alarm_pass(lm, samples, scorers_c, args)
    rows_c = [sc.row for sc in scorers_c]
    projectors = {sc.row['tag']: sc.projector for sc in scorers_c}
    report['stages']['C'] = rows_c
    wire_rows = [r for r in rows_c if r['dtype'] == args.wire_dtype]
    rank_by = args.rank_by
    tolerance = args.fa_tolerance if rank_by == 'fppi' else args.tolerance
    choice = best_within(wire_rows, tolerance, rank_by)
    projector = projectors[choice['tag']]
    spec = FeatureSpec.from_dict(choice['spec'])
    report['recommendation'] = choice
    print(
        f'recommended: {choice["tag"]} (score {fmt(choice["score"])}, '
        f'fppi {fmt(choice.get("fppi"), 2)}, dim {choice["dim"]}, ranked by {rank_by})'
    )

    meta = {
        'tool': 'tools/similarity/suggest_api_hyperparameters.py',
        'created': datetime.now().isoformat(timespec='seconds'),
        'config': args.config,
        'weights': args.weights,
        'weights_digest': lm.weights_digest,
        'split': args.split,
        'images': len(samples),
        'input_size': list(lm.input_size),
        'pad_mode': lm.pad_mode,
        'gsd_m': args.gsd,
        'metrics': {k: v for k, v in choice.items() if k != 'spec'},
    }
    profile_path = save_profile(out_dir / 'similarity_profile.npz', spec, projector, meta)
    gsd = args.gsd if args.gsd is not None else '<GSD_M>'
    command = (
        f'python -m api --config {args.config} --weights {args.weights} --gsd {gsd} '
        f'--similarity-profile {profile_path}'
    )
    if lm.input_size_source == 'flag':
        command += f' --input-size {lm.input_size[0]}x{lm.input_size[1]}'
    if lm.pad_mode != 'square':
        command += f' --pad-mode {lm.pad_mode}'
    report['command'] = command
    report['profile'] = str(profile_path)
    report['seconds'] = time.time() - t_start
    (out_dir / 'suggestions.json').write_text(json.dumps(report, indent=1, default=str))
    (out_dir / 'suggestions.md').write_text(markdown(report))
    print('per class, recommendation:\n' + per_class_table(report))
    print(f'wrote {out_dir}/suggestions.md, suggestions.json, similarity_profile.npz')
    print('serve it with:\n  ' + command)
    return report


def markdown(rep: dict) -> str:
    rec = rep['recommendation']
    lines = [
        f'# Similarity settings for `{Path(rep["config"]).name}`',
        '',
        f'- weights: `{rep["weights"]}`',
        f'- data: {rep["images"]} images of the `{rep["split"]}` split, {rep["objects"]} objects '
        '(half fit the projection and ridge probe, half are scored)',
        f'- input {rep["input_size"][0]}x{rep["input_size"][1]}, pad `{rep["pad_mode"]}`, '
        f'{rep.get("ms_per_window", float("nan")):.1f} ms per window',
        f'- sources: {", ".join(f"{k} ({len(v)} levels)" for k, v in rep["sources"].items())}',
        '',
        '## Recommendation',
        '',
        f'**`{rec["tag"]}`** (ranked by {rep.get("rank_by", "score")}): score '
        f'{fmt(rec["score"])} (obj AP {fmt(rec.get("obj_ap"))}, ridge AP '
        f'{fmt(rec.get("ridge_ap"))}), false alarms per image {fmt(rec.get("fppi"), 2)}, '
        f'{rec["dim"]}-d, {fmt(rec["kb_per_512"], 0)} KB per 512 px tile.',
        '',
        '```bash',
        rep['command'],
        '```',
        '',
        '**Reading the scores.** *obj AP*: the brush (cosine to one painted object). '
        '*ridge AP*: trained classes (a linear model on paint). *score* is their mean. '
        '*bg_paircos* near 0 means centred features (raw ones crowd together and every '
        "cosine looks high); *same_cos* is what the plugin's Nystroem kernel expects to be "
        'roughly 0.76-0.90 for alike things. Size buckets are sqrt(area) in model px.',
        '',
        '## Per class (recommendation)',
        '',
        per_class_table(rep),
        '',
        '*AP*: retrieval AP (brush: each object queries its class against other classes and '
        'background; ridge: trained-class scores). *map AP*: the class score map over whole '
        'eval images read as a detector (precision at each found object). *R@FA N*: recall '
        'within N false alarms per image. *FA@R r*: false alarms per image needed to find a '
        'fraction r of the objects (`>cap`: not within the false alarms kept). Every stage '
        "row in suggestions.json has the same lists (`*_per_class`, in the classes' order).",
        '',
        '## Stage A: where the features come from (centred, float)',
        '',
        table(rep['stages']['A']),
        '',
        '## Stage B: output stride',
        '',
        table(rep['stages']['B']),
        '',
        '## Stage C: projection and wire dtype',
        '',
        table(rep['stages']['C'], extra=('var_kept', 'int8_cos', 'int8_top5_iou')),
        '',
    ]
    fa = rep.get('false_alarms')
    if fa:
        lines += [
            '## False alarms per image',
            '',
            f"Each class's score map over the {fa['eval_images']} eval images is thresholded "
            'as an annotator would: *brush* scores cosine to the class mean of the fit objects '
            '(the painted example), *ridge* is the trained-class model. At a threshold, an '
            'object is *found* when any of its cells is above it; every local maximum above it '
            f"(one per ~{fa['peak_px']:g} px) on the background or on another class's object "
            'is one *false alarm*. `fppi@rR`: false alarms per image needed to find a fraction '
            'R of the objects (mean over classes; `>cap` = some class does not get there '
            f'within {fa["max_peaks"]} false alarms per image). `r@fppiN`: recall within N '
            'false alarms per image. `fa_other_class`: share of the false alarms on objects of '
            f'other classes (the rest is background), at r{fa["fa_recalls"][0]:g}. `fppi` is '
            f'the mean of brush and ridge at r{fa["fa_recalls"][0]:g}.',
            '',
        ]
        for stage in ('A', 'B', 'C'):
            lines += [f'### Stage {stage}', '', fa_table(rep['stages'][stage], fa), '']
    return '\n'.join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == '__main__':
    main()
