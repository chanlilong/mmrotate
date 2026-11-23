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
"""

from __future__ import annotations

import math

import numpy as np
import torch
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
) -> dict:
    """Prototype retrieval AP (cosine), overall and per object-size bucket.

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
    return out


def ridge_ap(
    fit_x: Tensor,
    fit_y: Tensor,
    eval_x: Tensor,
    eval_y: Tensor,
    num_classes: int,
    lam: float = 100.0,
) -> dict:
    """One-vs-rest ridge on unit features (labels ``-1`` = background), mean AP per class."""
    x = torch.cat([fit_x, torch.ones_like(fit_x[:, :1])], 1).double()
    y = torch.zeros(fit_x.shape[0], num_classes, dtype=torch.float64, device=fit_x.device)
    fg = fit_y >= 0
    y[fg, fit_y[fg]] = 1.0
    reg = lam * torch.eye(x.shape[1], dtype=torch.float64, device=x.device)
    reg[-1, -1] = 0.0  # no penalty on the bias
    w = torch.linalg.solve(x.T @ x + reg, x.T @ y)
    xe = torch.cat([eval_x, torch.ones_like(eval_x[:, :1])], 1).double()
    scores = (xe @ w).float().T  # [C, M]
    positives = eval_y[None, :] == torch.arange(num_classes, device=eval_y.device)[:, None]
    has_fit = torch.zeros(num_classes, dtype=torch.bool, device=fit_y.device)
    has_fit[fit_y[fg].unique()] = True
    ap = average_precision_rows(scores, positives)
    ok = ~torch.isnan(ap) & has_fit
    return {
        'ridge_ap': float(ap[ok].mean()) if ok.any() else float('nan'),
        'ridge_classes': int(ok.sum()),
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
