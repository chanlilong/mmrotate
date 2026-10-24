# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch
from mmengine.structures import InstanceData

from mmrotate.evaluation import eval_rbbox_map
from mmrotate.evaluation.metrics import SODAMetric
from mmrotate.evaluation.metrics.soda.soda_eval import eval_rotated_ap, tpfp_multi

CLASSES = ('a', 'b')


def _random_boxes(rng: np.random.Generator, n: int, size: tuple[float, float]) -> np.ndarray:
    xy = rng.uniform(0, 200, (n, 2))
    wh = rng.uniform(*size, (n, 2))
    angle = rng.uniform(-np.pi / 2, np.pi / 2, (n, 1))
    return np.concatenate([xy, wh, angle], axis=1)


def _random_dataset(seed: int, num_imgs: int = 12) -> tuple[list, list]:
    """GTs plus jittered (partly wrong-class) detections and noise boxes."""
    rng = np.random.default_rng(seed)
    dets, anns = [], []
    for _ in range(num_imgs):
        n = int(rng.integers(0, 15))
        gts = _random_boxes(rng, n, (4, 45))
        labels = rng.integers(0, 2, n)
        jitter = gts + rng.normal(0, 1.5, gts.shape) * [1, 1, 0.3, 0.3, 0.05]
        noise = _random_boxes(rng, 5, (4, 45))
        boxes = np.concatenate([jitter, noise])
        box_labels = np.concatenate([labels, rng.integers(0, 2, 5)])
        scores = rng.uniform(0, 1, len(boxes))
        per_cls = [
            np.concatenate([boxes[box_labels == c], scores[box_labels == c, None]], axis=1)
            for c in range(2)
        ]
        dets.append(per_cls)
        anns.append(
            dict(
                bboxes=gts,
                labels=labels,
                bboxes_ignore=np.zeros((0, 5)),
                labels_ignore=np.zeros(0, dtype=np.int64),
            )
        )
    return dets, anns


@pytest.mark.parametrize('iou_thr', [0.3, 0.5, 0.75])
def test_matches_eval_rbbox_map(iou_thr: float) -> None:
    dets, anns = _random_dataset(0)
    ref, cls_results = eval_rbbox_map(
        dets, anns, iou_thr=iou_thr, use_07_metric=False, logger='silent', nproc=1
    )
    ap, num_gts = eval_rotated_ap(dets, anns, [iou_thr], [(0, 1e5)])
    ref_cls = np.array([r['ap'] for r in cls_results])
    np.testing.assert_allclose(ap[0, 0], ref_cls, atol=1e-6)
    assert num_gts[0].tolist() == [r['num_gts'] for r in cls_results]
    assert ap[0, 0].mean() == pytest.approx(ref, abs=1e-6)


def test_tpfp_size_ranges() -> None:
    # GT 0 is 10x10 (bin 0-12), GT 1 is 30x30 (bin 20-32).
    gts = np.array([[20, 20, 10, 10, 0], [100, 100, 30, 30, 0]], dtype=np.float64)
    dets = np.array(
        [
            [20, 20, 10, 10, 0, 0.9],  # TP on the small GT
            [100, 100, 30, 30, 0, 0.8],  # TP on the large GT
            [20, 20, 10, 10, 0, 0.7],  # duplicate of the small GT
            [300, 300, 10, 10, 0, 0.6],  # small unmatched box
        ]
    )
    ranges = np.array([(0, 1e5), (0, 12), (20, 32)], dtype=np.float64) ** 2
    tp, fp = tpfp_multi(dets, gts, np.zeros((0, 5)), np.array([0.5]), ranges)
    assert tp[0, 0].tolist() == [True, True, False, False]
    assert fp[0, 0].tolist() == [False, False, True, True]
    # small bin: the large-GT match is ignored, small FPs count
    assert tp[0, 1].tolist() == [True, False, False, False]
    assert fp[0, 1].tolist() == [False, False, True, True]
    # large bin: only the large-GT match counts; small dets are out of range
    assert tp[0, 2].tolist() == [False, True, False, False]
    assert fp[0, 2].tolist() == [False, False, False, False]


def test_ignored_gt_is_neither_tp_nor_fp() -> None:
    ign = np.array([[20, 20, 10, 10, 0]], dtype=np.float64)
    dets = np.array([[20, 20, 10, 10, 0, 0.9]])
    tp, fp = tpfp_multi(dets, np.zeros((0, 5)), ign, np.array([0.5]), np.array([[0, 1e10]]))
    assert not tp.any() and not fp.any()


def test_soda_metric_perfect_predictions() -> None:
    metric = SODAMetric()
    metric.dataset_meta = dict(classes=CLASSES)
    # one GT per SODA bin (sqrt area 8, 16, 25, 36) plus one larger object
    sides = [8, 16, 25, 36, 60]
    boxes = torch.tensor([[50.0 * i + 40, 60, s, s, 0.1] for i, s in enumerate(sides)])
    labels = torch.tensor([0, 1, 0, 1, 0])
    data_sample = dict(
        img_id='p0',
        gt_instances=dict(bboxes=boxes, labels=labels),
        ignored_instances=dict(bboxes=torch.zeros((0, 5)), labels=torch.zeros(0).long()),
        pred_instances=InstanceData(bboxes=boxes, scores=torch.ones(5), labels=labels).to_dict(),
    )
    metric.process({}, [data_sample])
    res = metric.evaluate(1)
    for key in ('mAP', 'AP50', 'AP75', 'AP_eS', 'AP_rS', 'AP_gS', 'AP_N', 'AP_a', 'AP_b'):
        assert res[f'soda/{key}'] == pytest.approx(1.0), key
