# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch
from mmdet.structures.bbox import HorizontalBoxes
from mmdet.structures.mask import PolygonMasks

from mmrotate.datasets.transforms import FilterOutsideBoxes
from mmrotate.registry import TRANSFORMS
from mmrotate.structures.bbox import RotatedBoxes, rbbox_overlaps

IMG_H, IMG_W = 100, 100


def _square_poly(cx: float, cy: float, s: float) -> list[np.ndarray]:
    h = s / 2
    return [np.array([cx - h, cy - h, cx + h, cy - h, cx + h, cy + h, cx - h, cy + h])]


def _toy_results() -> dict:
    # 0: fully inside (IoF 1), 1: half outside (IoF 0.5), 2: fully outside (IoF 0)
    boxes = np.array(
        [[50, 50, 20, 10, 0.0], [100, 50, 20, 10, 0.0], [150, 50, 10, 10, 0.0]],
        dtype=np.float32,
    )
    return dict(
        img_shape=(IMG_H, IMG_W),
        gt_bboxes=RotatedBoxes(boxes),
        gt_bboxes_labels=np.array([0, 1, 2], dtype=np.int64),
        gt_ignore_flags=np.array([False, True, False]),
        gt_masks=PolygonMasks(
            [_square_poly(50, 50, 10), _square_poly(100, 50, 10), _square_poly(150, 50, 10)],
            IMG_H,
            IMG_W,
        ),
    )


def test_filter_keeps_aligned_annotations() -> None:
    results = FilterOutsideBoxes(iof_thr=0.45)(_toy_results())
    assert len(results['gt_bboxes']) == 2
    np.testing.assert_array_equal(results['gt_bboxes_labels'], [0, 1])
    np.testing.assert_array_equal(results['gt_ignore_flags'], [False, True])
    assert len(results['gt_masks']) == 2
    np.testing.assert_allclose(results['gt_masks'].masks[1][0][:2], [95, 45])

    results = FilterOutsideBoxes(iof_thr=0.6)(_toy_results())
    np.testing.assert_array_equal(results['gt_bboxes_labels'], [0])
    assert len(results['gt_masks']) == 1


def test_rotated_corner_box_iof() -> None:
    # A square rotated 45 deg centred on the image corner is 1/4 inside.
    box = torch.tensor([[0.0, 0.0, 20.0, 20.0, np.pi / 4]])
    img_box = torch.tensor([[IMG_W / 2, IMG_H / 2, IMG_W, IMG_H, 0.0]])
    iof = rbbox_overlaps(box, img_box, mode='iof')[0, 0].item()
    assert iof == pytest.approx(0.25, abs=1e-3)

    results = dict(img_shape=(IMG_H, IMG_W), gt_bboxes=RotatedBoxes(box))
    assert len(FilterOutsideBoxes(iof_thr=0.2)(dict(results))['gt_bboxes']) == 1
    assert len(FilterOutsideBoxes(iof_thr=0.3)(dict(results))['gt_bboxes']) == 0


def test_hbox_keeps_type() -> None:
    boxes = HorizontalBoxes(torch.tensor([[40.0, 40.0, 60.0, 60.0], [90.0, 40.0, 110.0, 60.0]]))
    results = dict(img_shape=(IMG_H, IMG_W, 3), gt_bboxes=boxes)
    results = FilterOutsideBoxes(iof_thr=0.6)(results)
    assert isinstance(results['gt_bboxes'], HorizontalBoxes)
    assert len(results['gt_bboxes']) == 1


def test_empty_and_keep_empty() -> None:
    empty = dict(
        img_shape=(IMG_H, IMG_W),
        gt_bboxes=RotatedBoxes(torch.zeros((0, 5))),
        gt_bboxes_labels=np.zeros((0,), dtype=np.int64),
    )
    out = FilterOutsideBoxes(keep_empty=False)(empty)
    assert out is not None and len(out['gt_bboxes']) == 0

    out = FilterOutsideBoxes(iof_thr=1.0)(dict(img_shape=(IMG_H, IMG_W)))
    assert out is not None

    results = _toy_results()
    results['gt_bboxes'] = results['gt_bboxes'][[2]]
    results['gt_bboxes_labels'] = results['gt_bboxes_labels'][[2]]
    results['gt_ignore_flags'] = results['gt_ignore_flags'][[2]]
    results['gt_masks'] = results['gt_masks'][[2]]
    assert FilterOutsideBoxes(keep_empty=False)(dict(results)) is None
    out = FilterOutsideBoxes(keep_empty=True)(dict(results))
    assert out is not None and len(out['gt_bboxes']) == 0 and len(out['gt_masks']) == 0


def test_build_and_repr() -> None:
    transform = TRANSFORMS.build(dict(type='FilterOutsideBoxes', iof_thr=0.3, keep_empty=False))
    assert isinstance(transform, FilterOutsideBoxes)
    assert repr(transform) == 'FilterOutsideBoxes(iof_thr=0.3, keep_empty=False)'
    with pytest.raises(AssertionError):
        FilterOutsideBoxes(iof_thr=1.5)
