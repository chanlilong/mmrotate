# Copyright (c) OpenMMLab. All rights reserved.
"""Compare TensorRT detector outputs with the torch model's."""

from collections.abc import Sequence

import torch
from mmcv.ops import box_iou_rotated
from torch import Tensor

from .export import flatten_outputs


def raw_output_errors(ref: object, test: object) -> dict:
    """Max abs / relative error over all (nested) head outputs; ``inf`` if any is NaN / inf."""
    ref_leaves, _ = flatten_outputs(ref)
    test_leaves, _ = flatten_outputs(test)
    max_err, max_ref, rel, finite = 0.0, 0.0, 0.0, True
    for r, t in zip(ref_leaves, test_leaves, strict=True):
        r, t = r.float(), t.float().to(r.device)
        if not torch.isfinite(t).all():  # max() would silently skip NaNs
            finite = False
            max_err = rel = float('inf')
            continue
        err = (r - t).abs().max().item()
        scale = r.abs().max().item()
        max_err, max_ref = max(max_err, err), max(max_ref, scale)
        rel = max(rel, err / max(scale, 1e-6))
    return dict(max_abs_err=max_err, max_abs_ref=max_ref, max_rel_err=rel, finite=finite)


def _match(ref: dict, test: dict, score_thr: float, iou_thr: float) -> dict:
    keep = ref['scores'] >= score_thr
    ref_boxes, ref_scores, ref_labels = (ref[k][keep] for k in ('bboxes', 'scores', 'labels'))
    t_keep = test['scores'] >= score_thr * 0.5  # near-threshold boxes may cross it
    t_boxes, t_scores, t_labels = (test[k][t_keep] for k in ('bboxes', 'scores', 'labels'))
    n_ref = len(ref_boxes)
    if n_ref == 0:
        return dict(n_ref=0, n_test=int((test['scores'] >= score_thr).sum()), matched=0)
    if len(t_boxes) == 0:
        return dict(n_ref=n_ref, n_test=0, matched=0)
    ious = box_iou_rotated(ref_boxes.float().cuda(), t_boxes.float().cuda()).cpu()
    ious[ref_labels[:, None] != t_labels[None, :]] = 0
    best, idx = ious.max(dim=1)
    hit = best >= iou_thr
    return dict(
        n_ref=n_ref,
        n_test=int((test['scores'] >= score_thr).sum()),
        matched=int(hit.sum()),
        iou_sum=float(best[hit].sum()),
        score_diff_sum=float((ref_scores[hit] - t_scores[idx[hit]]).abs().sum()),
    )


def detection_agreement(
    ref: Sequence[dict], test: Sequence[dict], score_thr: float = 0.3, iou_thr: float = 0.5
) -> dict:
    """How well TensorRT detections reproduce the torch detections.

    Each torch detection with ``score >= score_thr`` is matched to the
    same-class TensorRT detection of highest rotated IoU.

    Args:
        ref / test: Per image ``dict(bboxes=(N, 5), scores=(N,), labels=(N,))``.

    Returns:
        ``recall`` (share of torch detections matched at ``iou_thr``),
        ``mean_iou`` and ``mean_score_diff`` of the matches, and the detection
        counts above ``score_thr``.
    """
    total = dict(n_ref=0, n_test=0, matched=0, iou_sum=0.0, score_diff_sum=0.0)
    for r, t in zip(ref, test, strict=True):
        for k, v in _match(r, t, score_thr, iou_thr).items():
            total[k] += v
    matched = max(total['matched'], 1)
    return dict(
        score_thr=score_thr,
        torch_dets=total['n_ref'],
        trt_dets=total['n_test'],
        recall=total['matched'] / max(total['n_ref'], 1),
        mean_iou=total['iou_sum'] / matched,
        mean_score_diff=total['score_diff_sum'] / matched,
    )


def instances_to_dict(pred_instances: object) -> dict[str, Tensor]:
    """``pred_instances`` -> CPU dict of bboxes / scores / labels."""
    bboxes = pred_instances.bboxes
    bboxes = bboxes.tensor if hasattr(bboxes, 'tensor') else bboxes
    return dict(
        bboxes=bboxes.detach().float().cpu(),
        scores=pred_instances.scores.detach().float().cpu(),
        labels=pred_instances.labels.detach().cpu(),
    )


def to_cpu(outputs: object) -> object:
    """Nested tuples of tensors -> same structure on the CPU."""
    if isinstance(outputs, Tensor):
        return outputs.detach().cpu()
    return tuple(to_cpu(o) for o in outputs)
