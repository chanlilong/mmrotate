# Copyright (c) OpenMMLab. All rights reserved.
"""Quality measures for similarity features, after qgis_annotations' similarity benchmarks
(``scripts/bench_similarity_onnx.py``, ``bench_similarity_features.py``):

- **obj AP**: the Similarity brush's prototype mode. Every object queries the other objects
  of its class against the other classes plus background cells, under cosine.
- **ridge AP**: the plugin's trained classes. One-vs-rest ridge (lambda = 100 on unit
  features) fitted on one half of the images, scored on the other.
- **geometry**: mean pairwise cosine of background cells (anisotropy) and mean same-class
  cosine (the plugin's Nystroem gamma = 2 assumes roughly 0.76-0.90).
- **int8 agreement**: mean cosine of the int8 vectors to the float ones, and the top-5 %
  IoU of similarity maps (the reference service's ship rule is >= 0.9).
- **per class**: every AP, recall and FPPI is also reported per class (``*_per_class``
  lists in class order), next to the means over classes.
- **false alarms per image**: every class's score map (brush prototype or ridge model) over
  whole held-out images, thresholded as an annotator would. At a threshold, an object of the
  class is *found* when any of its cells scores above it; every local maximum above it on
  the background or on another class's object is one *false alarm* (a separate blob).
  Reported like ``tools/soda_dinov3/false_alarms.py``: FPPI needed for a recall, recall
  within an FPPI budget.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

SIZE_BUCKETS = (16, 32, 64)  # sqrt(area) edges in model px


def average_precision_rows(scores: Tensor, positives: Tensor) -> Tensor:
    """AP of every row: ``scores`` ``[Q, N]``, ``positives`` bool ``[Q, N]`` -> ``[Q]``.

    Rows without positives give NaN.
    """
    order = torch.argsort(scores, dim=1, descending=True)
    pos = torch.gather(positives, 1, order).float()
    hits = pos.cumsum(1)
    ranks = torch.arange(1, scores.shape[1] + 1, device=scores.device, dtype=torch.float32)
    precision = hits / ranks
    npos = pos.sum(1)
    ap = (precision * pos).sum(1) / npos.clamp_min(1)
    ap[npos == 0] = float('nan')
    return ap


def size_bucket(sizes: Tensor) -> Tensor:
    edges = torch.as_tensor(SIZE_BUCKETS, dtype=sizes.dtype, device=sizes.device)
    return torch.bucketize(sizes, edges)


def bucket_names() -> list[str]:
    e = SIZE_BUCKETS
    return [f'<{e[0]}'] + [f'{a}-{b}' for a, b in zip(e[:-1], e[1:], strict=True)] + [f'>{e[-1]}']


def object_ap(
    obj: Tensor,
    labels: Tensor,
    sizes: Tensor,
    bg: Tensor,
    max_queries: int = 2000,
    seed: int = 0,
    num_classes: int | None = None,
) -> dict:
    """Prototype retrieval AP (cosine), overall, per object-size bucket and per class
    (``obj_ap_per_class``, the mean over that class's queries; None without queries).

    ``obj`` ``[N, D]`` and ``bg`` ``[M, D]`` are unit vectors of the evaluation images.
    """
    n = obj.shape[0]
    if n < 2:
        return {'obj_ap': float('nan')}
    gen = torch.Generator().manual_seed(seed)
    q_idx = torch.randperm(n, generator=gen)[: min(n, max_queries)].to(obj.device)
    cand = torch.cat([obj, bg])
    cand_labels = torch.cat([labels, torch.full((bg.shape[0],), -1, device=labels.device)])
    aps = []
    for chunk in q_idx.split(512):
        scores = obj[chunk] @ cand.T
        positives = cand_labels[None, :] == labels[chunk, None]
        scores[torch.arange(len(chunk)), chunk] = -math.inf  # the query itself
        positives[torch.arange(len(chunk)), chunk] = False
        aps.append(average_precision_rows(scores, positives))
    ap = torch.cat(aps)
    ok = ~torch.isnan(ap)
    out = {'obj_ap': float(ap[ok].mean()) if ok.any() else float('nan')}
    buckets = size_bucket(sizes[q_idx])
    for b, name in enumerate(bucket_names()):
        sel = ok & (buckets == b)
        out[f'obj_ap_{name}'] = float(ap[sel].mean()) if sel.any() else float('nan')
    if num_classes is not None:
        q_labels = labels[q_idx]
        per = []
        for c in range(num_classes):
            sel = ok & (q_labels == c)
            per.append(float(ap[sel].mean()) if sel.any() else None)
        out['obj_ap_per_class'] = per
    return out


def ridge_fit(x: Tensor, y: Tensor, num_classes: int, lam: float = 100.0) -> Tensor:
    """One-vs-rest ridge on unit features (labels ``-1`` = background) -> ``[D + 1, C]``
    weights, the last row being the (unpenalised) bias."""
    xb = torch.cat([x, torch.ones_like(x[:, :1])], 1).double()
    t = torch.zeros(x.shape[0], num_classes, dtype=torch.float64, device=x.device)
    fg = y >= 0
    t[fg, y[fg]] = 1.0
    reg = lam * torch.eye(xb.shape[1], dtype=torch.float64, device=x.device)
    reg[-1, -1] = 0.0
    return torch.linalg.solve(xb.T @ xb + reg, xb.T @ t).float()


def ridge_scores(w: Tensor, x: Tensor) -> Tensor:
    """``[..., D]`` features -> ``[..., C]`` ridge scores."""
    return x @ w[:-1] + w[-1]


def class_presence(y: Tensor, num_classes: int) -> Tensor:
    has = torch.zeros(num_classes, dtype=torch.bool, device=y.device)
    has[y[y >= 0].unique()] = True
    return has


def ridge_ap(
    fit_x: Tensor,
    fit_y: Tensor,
    eval_x: Tensor,
    eval_y: Tensor,
    num_classes: int,
    lam: float = 100.0,
) -> dict:
    """One-vs-rest ridge on unit features (labels ``-1`` = background), mean AP per class."""
    w = ridge_fit(fit_x, fit_y, num_classes, lam)
    scores = ridge_scores(w, eval_x).T  # [C, M]
    positives = eval_y[None, :] == torch.arange(num_classes, device=eval_y.device)[:, None]
    has_fit = class_presence(fit_y, num_classes)
    ap = average_precision_rows(scores, positives)
    ok = ~torch.isnan(ap) & has_fit
    return {
        'ridge_ap': float(ap[ok].mean()) if ok.any() else float('nan'),
        'ridge_classes': int(ok.sum()),
        'ridge_ap_per_class': [float(a) if o else None for a, o in zip(ap, ok, strict=True)],
    }


def geometry(obj: Tensor, labels: Tensor, bg: Tensor, max_n: int = 2000) -> dict:
    """Anisotropy (mean background pairwise cosine) and same / other-class object cosine."""
    out = {}
    b = bg[:max_n]
    if b.shape[0] > 1:
        g = b @ b.T
        n = g.shape[0]
        out['bg_paircos'] = float((g.sum() - g.diagonal().sum()) / (n * (n - 1)))
    o, lab = obj[:max_n], labels[:max_n]
    if o.shape[0] > 1:
        g = o @ o.T
        same = lab[:, None] == lab[None, :]
        eye = torch.eye(g.shape[0], dtype=torch.bool, device=g.device)
        s, d = same & ~eye, ~same
        out['same_cos'] = float(g[s].mean()) if s.any() else float('nan')
        out['other_cos'] = float(g[d].mean()) if d.any() else float('nan')
    return out


def int8_agreement(unit: Tensor, n_queries: int = 32, top: float = 0.05, seed: int = 0) -> dict:
    """How much ``round(v * 127)`` changes things: mean cosine and top-5 % map IoU."""
    q8 = torch.clamp(torch.round(unit * 127), -127, 127) / 127
    cos = torch.nn.functional.cosine_similarity(unit, q8, dim=-1)
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(unit.shape[0], generator=gen)[:n_queries].to(unit.device)
    k = max(1, int(round(unit.shape[0] * top)))
    s_f = unit[idx] @ unit.T
    s_q = q8[idx] @ q8.T
    top_f = torch.topk(s_f, k, dim=1).indices
    top_q = torch.topk(s_q, k, dim=1).indices
    ious = []
    for a, b in zip(top_f.tolist(), top_q.tolist(), strict=True):
        sa, sb = set(a), set(b)
        ious.append(len(sa & sb) / len(sa | sb))
    return {
        'int8_cos': float(cos.mean()),
        'int8_cos_min': float(cos.min()),
        'int8_top5_iou': float(np.mean(ious)),
    }


def grid_bytes_per_512(stride_src: int, dim: int, dtype: str) -> int:
    """Bytes of one 512 x 512 source-pixel tile's grid on the wire (before base64)."""
    per = {'int8': 1, 'float16': 2, 'float32': 4}[dtype]
    return math.ceil(512 / stride_src) ** 2 * dim * per


# --------------------------------------------------------------------------- false alarms
BACKGROUND, OTHER_CLASS = -2, -1  # codes of a false-alarm peak


def peak_window(stride: int, peak_px: float) -> int:
    """Odd max-pool window (cells) spanning about ``peak_px`` model px (at least 3, so a
    blob of several cells is one false alarm, not one per cell)."""
    return max(3, 2 * int(peak_px // (2 * stride)) + 1)


def class_events(
    scores: Tensor, obj_map: Tensor, obj_labels: Tensor, window: int, max_fa: int
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Hits and false alarms of every class map of one image.

    Args:
        scores: ``[C, h, w]`` class score maps.
        obj_map: ``[h, w]`` long, ``-1`` on background, else the index of the object
            (into ``obj_labels``) covering the cell.
        obj_labels: ``[K]`` class of every object.
        window: odd max-pool window in cells: a false alarm is a local maximum.
        max_fa: keep the highest this many false-alarm peaks per class.

    Returns:
        ``obj_scores`` ``[K]``: each object's best cell on its own class's map (``-inf``
        when no cell is left to it); ``fa_values``, ``fa_codes`` (:data:`OTHER_CLASS` or
        :data:`BACKGROUND`) and ``fa_valid``, each ``[C, N]``.
    """
    C = scores.shape[0]
    flat_ids = obj_map.reshape(-1)
    K = obj_labels.numel()
    obj_scores = torch.full((K,), -math.inf, device=scores.device)
    on_obj = flat_ids >= 0
    if K and on_obj.any():
        ids = flat_ids[on_obj]
        cells = torch.nonzero(on_obj).squeeze(1)
        vals = scores.reshape(C, -1)[obj_labels[ids], cells]
        obj_scores = obj_scores.scatter_reduce(0, ids, vals, reduce='amax')
    pooled = F.max_pool2d(scores[None], window, stride=1, padding=window // 2)[0]
    peak = scores >= pooled
    cell_cls = torch.full_like(flat_ids, -1)
    if K:
        cell_cls[on_obj] = obj_labels[flat_ids[on_obj]]
    own = cell_cls[None, :] == torch.arange(C, device=scores.device)[:, None]  # [C, h*w]
    flat = scores.reshape(C, -1)
    fa = peak.reshape(C, -1) & ~own
    masked = torch.where(fa, flat, torch.full_like(flat, -math.inf))
    fa_values, idx = masked.topk(min(max_fa, masked.shape[1]), dim=1)
    fa_codes = torch.where(
        flat_ids[idx] >= 0,
        torch.full_like(idx, OTHER_CLASS),
        torch.full_like(idx, BACKGROUND),
    )
    return obj_scores, fa_values, fa_codes, torch.isfinite(fa_values)


def fppi_curve(
    obj_scores: np.ndarray, fa_values: np.ndarray, fa_codes: np.ndarray, n_gt: int, n_images: int
) -> dict:
    """Recall / FPPI as the threshold falls through every hit and false alarm of a class."""
    obj_scores = obj_scores[np.isfinite(obj_scores)]
    values = np.concatenate([obj_scores, fa_values])
    codes = np.concatenate([np.zeros(len(obj_scores), int), fa_codes])
    order = np.argsort(-values, kind='stable')
    codes = codes[order]
    fp = codes < 0
    tp = codes == 0
    cum_tp, cum_fp = np.cumsum(tp), np.cumsum(fp)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1)
    return {
        'recall': cum_tp / max(n_gt, 1),
        'fppi': cum_fp / max(n_images, 1),
        'other_class': np.cumsum(codes == OTHER_CLASS),
        'fp': cum_fp,
        # AP of the score map as a detector: mean precision at each found object (objects
        # never found add 0). False alarms beyond --fa-max-peaks are not seen.
        'map_ap': float(precision[tp].sum() / max(n_gt, 1)),
    }


def fppi_summary(
    curves: list[dict | None],
    recall_points: Sequence[float],
    fppi_points: Sequence[float],
) -> dict:
    """Mean over classes of the FPPI needed for each recall and of recall within each FPPI
    budget. A class that never reaches a recall within the false alarms kept counts as
    ``inf`` in the mean; ``fppi@rR_reached`` says how many classes did."""
    out: dict = {}
    used = [c for c in curves if c is not None]
    if not used:
        return out
    for r in recall_points:
        vals = []
        for c in used:
            ok = c['recall'] >= r
            vals.append(float(c['fppi'][ok].min()) if ok.any() else math.inf)
        out[f'fppi@r{r:g}'] = float(np.mean(vals))
        out[f'fppi@r{r:g}_reached'] = int(np.isfinite(vals).sum())
        out[f'fppi@r{r:g}_per_class'] = _spread(curves, vals)
    for p in fppi_points:
        vals = []
        for c in used:
            ok = c['fppi'] <= p
            vals.append(float(c['recall'][ok].max()) if ok.any() else 0.0)
        out[f'r@fppi{p:g}'] = float(np.mean(vals))
        out[f'r@fppi{p:g}_per_class'] = _spread(curves, vals)
    map_aps = [c['map_ap'] for c in used]
    out['map_ap'] = float(np.mean(map_aps))
    out['map_ap_per_class'] = _spread(curves, map_aps)
    # what the false alarms are, at the first recall point: other classes vs background
    r0 = recall_points[0]
    shares = []
    for c in used:
        ok = np.flatnonzero(c['recall'] >= r0)
        i = ok[0] if len(ok) else len(c['recall']) - 1
        if len(c['fp']) and c['fp'][i] > 0:
            shares.append(c['other_class'][i] / c['fp'][i])
    out['fa_other_class'] = float(np.mean(shares)) if shares else math.nan
    out['fa_classes'] = len(used)
    return out


def _spread(curves: list[dict | None], vals: list[float]) -> list[float | None]:
    """``vals`` (one per used curve) back onto the positions of ``curves``."""
    it = iter(vals)
    return [None if c is None else next(it) for c in curves]
