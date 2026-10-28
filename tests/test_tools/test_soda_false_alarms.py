# Copyright (c) OpenMMLab. All rights reserved.
"""Tests for tools/soda_dinov3/false_alarms.py (FP taxonomy and FPPI-recall)."""

import importlib.util
import os.path as osp
from types import ModuleType

import numpy as np
import pytest

_PATH = osp.join(osp.dirname(__file__), '..', '..', 'tools', 'soda_dinov3', 'false_alarms.py')


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location('soda_false_alarms', _PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fa = _load()
DUP, CLS, LOC, CLS_LOC, BKG = (1 + fa.FP_TYPES.index(t) for t in fa.FP_TYPES)


def _box(cx: float, cy: float, w: float = 10.0, h: float = 10.0) -> list[float]:
    return [cx, cy, w, h, 0.0]


def test_classify_detections_taxonomy() -> None:
    gts = np.array([_box(50, 50), _box(100, 100)])  # class 0, class 1
    gt_labels = np.array([0, 1])
    ignore, ignore_labels = np.array([_box(200, 200)]), np.array([0])
    dets = np.array(
        [
            _box(50, 50),  # TP on GT 0
            _box(50, 50),  # same GT again: dup
            _box(100, 100),  # class 0 on the class-1 GT: cls
            _box(55, 50),  # IoU 1/3 with GT 0: loc
            _box(55, 50),  # class 1, IoU 1/3 with the class-0 GT: cls+loc
            _box(300, 300),  # nothing near: bkg
            _box(200, 200),  # on the ignored GT
        ]
    )
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
    labels = np.array([0, 0, 0, 0, 1, 0, 0])
    codes = fa.classify_detections(dets, scores, labels, gts, gt_labels, ignore, ignore_labels)
    assert codes.tolist() == [fa.TP, DUP, CLS, LOC, CLS_LOC, BKG, fa.IGNORED]


def test_classify_detections_score_order() -> None:
    gts, gt_labels = np.array([_box(50, 50)]), np.array([0])
    dets = np.array([_box(50, 50), _box(51, 50)])
    codes = fa.classify_detections(dets, np.array([0.2, 0.9]), np.array([0, 0]), gts, gt_labels)
    # The higher-scoring detection takes the GT even though it overlaps less.
    assert codes.tolist() == [DUP, fa.TP]


def test_classify_detections_no_gt() -> None:
    dets = np.array([_box(50, 50)])
    codes = fa.classify_detections(
        dets, np.array([0.5]), np.array([0]), np.zeros((0, 5)), np.zeros(0, dtype=np.int64)
    )
    assert codes.tolist() == [BKG]
    empty = fa.classify_detections(
        np.zeros((0, 5)),
        np.zeros(0),
        np.zeros(0, dtype=np.int64),
        np.array([_box(1, 1)]),
        np.array([0]),
    )
    assert empty.shape == (0,)


def test_curve_and_operating_point() -> None:
    # 2 images (the second without GT), 4 GTs, detections sorted by score.
    run = dict(
        score=np.array([0.9, 0.8, 0.6, 0.4, 0.2]),
        code=np.array([fa.TP, BKG, fa.TP, DUP, BKG], dtype=np.int8),
        label=np.array([0, 1, 0, 0, 1]),
        size=np.array([0, 1, 0, 0, 4]),
        img=np.array([0, 1, 0, 0, 1], dtype=np.int32),
        gt_per_class=np.array([4] + [0] * (len(fa.CLASSES) - 1)),
        empty=np.array([False, True]),
    )
    cv = fa.curve(run)
    np.testing.assert_allclose(cv['recall'], [0.25, 0.25, 0.5, 0.5, 0.5])
    np.testing.assert_allclose(cv['fppi'], [0.0, 0.5, 0.5, 1.0, 1.5])

    op = fa.operating_point(run, 0.5)
    assert (op['tp'], op['fp'], op['fn']) == (2, 1, 2)
    assert op['fppi'] == pytest.approx(0.5)
    assert op['fppi_empty'] == pytest.approx(1.0)
    assert op['img_with_fp'] == pytest.approx(0.5)
    assert op['fp_types']['bkg'] == 1 and op['fp_types']['dup'] == 0
    assert op['fp_per_class']['helicopter'] == 1
    assert op['fp_per_size']['rS'] == 1

    summary = fa.summarise(run)
    assert summary['recall_at_fppi']['1.0'] == pytest.approx(0.5)
    assert summary['fppi_at_recall']['0.5'] == pytest.approx(0.5)
    assert summary['fppi_at_recall']['0.8'] is None


def test_short_name() -> None:
    assert fa.short_name('vits-dinov3-comer_rtmdet-p2-nwd-rka-6e-soda') == 'comer nwd-rka'
    assert fa.short_name('convnext-t-dinov3_rtmdet-p2-6e-soda') == 'convnext-t'
    assert fa.short_name('vits-dinov3-vitdet-lora_rtmdet-p2-6e-soda') == 'vitdet-lora'
