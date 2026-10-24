# Copyright (c) OpenMMLab. All rights reserved.
"""Fast rotated AP over several IoU thresholds and size ranges at once.

Same matching rule as mmrotate's ``tpfp_default`` (VOC style): every
detection is matched to the GT it overlaps most; walking detections by
descending score, the first one reaching ``iou_thr`` on a GT is a TP and later
ones are FPs. Ignored GTs and GTs outside the evaluated size range make their
matched detections neither TP nor FP; unmatched detections are FPs only when
their own size is inside the range. The rotated IoU matrix of an image is
computed once (on the GPU when available) and the matching is vectorised
over thresholds and ranges. Everything runs in-process: forking workers after
CUDA / OpenMP initialisation can deadlock inside a training run.
"""

from collections.abc import Sequence

import numpy as np
import torch
from mmcv.ops import box_iou_rotated
from mmdet.evaluation.functional import average_precision

ImageResult = tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]


def _area_in(boxes: np.ndarray, area_ranges: np.ndarray) -> np.ndarray:
    """``(K, n)`` mask: box area (``w * h``) inside ``[lo, hi)`` of each range."""
    area = boxes[:, 2] * boxes[:, 3]
    return (area[None] >= area_ranges[:, :1]) & (area[None] < area_ranges[:, 1:])


def tpfp_multi(
    dets: np.ndarray,
    gts: np.ndarray,
    gts_ignore: np.ndarray,
    iou_thrs: np.ndarray,
    area_ranges: np.ndarray,
    device: str = 'cpu',
) -> tuple[np.ndarray, np.ndarray]:
    """TP/FP flags of shape ``(T, K, m)`` for one image and one class.

    Args:
        dets: ``(m, 6)`` rotated boxes ``(cx, cy, w, h, a, score)``.
        gts: ``(n, 5)`` rotated GT boxes.
        gts_ignore: ``(k, 5)`` ignored GT boxes.
        iou_thrs: ``(T,)`` IoU thresholds.
        area_ranges: ``(K, 2)`` area (not sqrt) ranges.
        device: Device for the rotated IoU.
    """
    T, K, m = len(iou_thrs), len(area_ranges), len(dets)
    tp = np.zeros((T, K, m), dtype=bool)
    fp = np.zeros((T, K, m), dtype=bool)
    if m == 0:
        return tp, fp
    det_in = _area_in(dets, area_ranges)  # (K, m)
    all_gts = np.concatenate([gts.reshape(-1, 5), gts_ignore.reshape(-1, 5)])
    if len(all_gts) == 0:
        fp[:] = det_in[None]
        return tp, fp
    gt_bad = ~_area_in(all_gts, area_ranges)  # (K, n + k)
    gt_bad[:, len(gts) :] = True
    ious = (
        box_iou_rotated(
            torch.from_numpy(dets[:, :5]).float().to(device),
            torch.from_numpy(all_gts).float().to(device),
        )
        .cpu()
        .numpy()
    )
    ious_max = ious.max(axis=1)
    ious_arg = ious.argmax(axis=1)
    order = np.argsort(-dets[:, 5], kind='stable')
    for t, thr in enumerate(iou_thrs):
        matched = ious_max >= thr
        for k in range(K):
            good = matched & ~gt_bad[k, ious_arg]
            idx = order[good[order]]  # good matches in score order
            _, first = np.unique(ious_arg[idx], return_index=True)
            fp[t, k, idx] = True
            fp[t, k, idx[first]] = False
            tp[t, k, idx[first]] = True
            fp[t, k] |= ~matched & det_in[k]
    return tp, fp


def _eval_image(
    det_per_cls: Sequence[np.ndarray],
    gt: dict,
    iou_thrs: np.ndarray,
    area_ranges: np.ndarray,
    device: str,
) -> ImageResult:
    tps, fps, scores, num_gts = [], [], [], []
    for c, dets in enumerate(det_per_cls):
        mask = gt['labels'] == c
        gts = gt['bboxes'][mask]
        ign_mask = gt.get('labels_ignore', np.zeros(0, dtype=int)) == c
        gts_ign = gt.get('bboxes_ignore', np.zeros((0, 5)))[ign_mask]
        tp, fp = tpfp_multi(dets, gts, gts_ign, iou_thrs, area_ranges, device)
        tps.append(tp)
        fps.append(fp)
        scores.append(dets[:, 5] if len(dets) else np.zeros(0))
        num_gts.append(_area_in(gts.reshape(-1, 5), area_ranges).sum(axis=1))
    return tps, fps, scores, num_gts


def eval_rotated_ap(
    det_results: Sequence[Sequence[np.ndarray]],
    annotations: Sequence[dict],
    iou_thrs: Sequence[float],
    scale_ranges: Sequence[Sequence[float]],
    mode: str = 'area',
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-class AP for every IoU threshold and ``sqrt(area)`` range.

    Args:
        det_results: Per image, per class ``(m, 6)`` rotated detections.
        annotations: Per image dict with ``bboxes`` ``(n, 5)``, ``labels`` and
            optionally ``bboxes_ignore`` / ``labels_ignore``.
        iou_thrs: IoU thresholds.
        scale_ranges: ``(lo, hi)`` ranges on ``sqrt(w * h)``.
        mode: ``'area'`` or ``'11points'`` AP.
        device: Device for the rotated IoU, CUDA when available by default.

    Returns:
        ``ap`` of shape ``(T, K, C)`` and ``num_gts`` of shape ``(K, C)``.
    """
    iou_arr = np.asarray(iou_thrs, dtype=np.float64)
    area_ranges = np.asarray(scale_ranges, dtype=np.float64) ** 2
    num_classes = len(det_results[0]) if det_results else 0
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    per_img = [
        _eval_image(d, a, iou_arr, area_ranges, device)
        for d, a in zip(det_results, annotations, strict=True)
    ]

    T, K = len(iou_arr), len(area_ranges)
    ap = np.zeros((T, K, num_classes))
    num_gts = np.zeros((K, num_classes), dtype=np.int64)
    for c in range(num_classes):
        scores = np.concatenate([r[2][c] for r in per_img])
        order = np.argsort(-scores, kind='stable')
        tp = np.concatenate([r[0][c] for r in per_img], axis=2)[..., order]
        fp = np.concatenate([r[1][c] for r in per_img], axis=2)[..., order]
        num_gts[:, c] = np.sum([r[3][c] for r in per_img], axis=0)
        if len(scores) == 0:
            continue
        eps = np.finfo(np.float32).eps
        for t in range(T):  # per threshold to bound memory on dense classes
            tp_cum = np.cumsum(tp[t], axis=1, dtype=np.float64)
            fp_cum = np.cumsum(fp[t], axis=1, dtype=np.float64)
            recalls = tp_cum / np.maximum(num_gts[:, c, None], eps)
            precisions = tp_cum / np.maximum(tp_cum + fp_cum, eps)
            ap[t, :, c] = average_precision(recalls, precisions, mode)
    return ap, num_gts
