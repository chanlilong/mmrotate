# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch

from mmrotate.models.task_modules.nwd_rka import RBboxNWD2D, rbbox_nwd
from mmrotate.structures.bbox import RotatedBoxes


def aitod_nwd(b1: torch.Tensor, b2: torch.Tensor, constant: float) -> torch.Tensor:
    """Pairwise NWD of mmdet-aitod ``bbox_overlaps(mode='nwd')`` (xyxy)."""
    c1 = (b1[:, None, :2] + b1[:, None, 2:]) / 2
    c2 = (b2[None, :, :2] + b2[None, :, 2:]) / 2
    center = ((c1 - c2) ** 2).sum(-1)
    wh1 = b1[:, None, 2:] - b1[:, None, :2]
    wh2 = b2[None, :, 2:] - b2[None, :, :2]
    wh = ((wh1 - wh2) ** 2).sum(-1) / 4
    return torch.exp(-torch.sqrt(center + wh) / constant)


def random_rboxes(num: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    xy = torch.rand(num, 2, generator=gen) * 100
    wh = torch.rand(num, 2, generator=gen) * 30 + 2
    theta = (torch.rand(num, 1, generator=gen) - 0.5) * math.pi
    return torch.cat([xy, wh, theta], dim=1)


def test_horizontal_matches_reference() -> None:
    boxes = random_rboxes(7, 0)
    boxes[:, 4] = 0
    xyxy = torch.cat([boxes[:, :2] - boxes[:, 2:4] / 2, boxes[:, :2] + boxes[:, 2:4] / 2], 1)
    expected = aitod_nwd(xyxy, xyxy[:5], 12.7)
    # The general (rotated) closed form cancels in float32 for identical boxes,
    # W2 ~ 1e-5 instead of 0, hence the absolute tolerance.
    torch.testing.assert_close(rbbox_nwd(boxes, boxes[:5], 12.7), expected, atol=1e-3, rtol=0)
    torch.testing.assert_close(rbbox_nwd(xyxy, xyxy[:5], 12.7), expected, atol=1e-3, rtol=0)
    # mixed horizontal / rotated inputs
    torch.testing.assert_close(rbbox_nwd(xyxy, boxes[:5], 12.7), expected, atol=1e-3, rtol=0)


def test_identity_and_range() -> None:
    boxes = random_rboxes(6, 1)
    nwd = rbbox_nwd(boxes, boxes)
    torch.testing.assert_close(nwd.diagonal(), torch.ones(6), atol=1e-3, rtol=0)
    assert ((nwd > 0) & (nwd <= 1)).all()


def test_rotation_invariance() -> None:
    b1, b2 = random_rboxes(5, 2), random_rboxes(4, 3)
    phi = 0.7
    rot = torch.tensor([[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]])

    def rotate(b: torch.Tensor) -> torch.Tensor:
        return torch.cat([b[:, :2] @ rot.T, b[:, 2:4], b[:, 4:] + phi], dim=1)

    torch.testing.assert_close(
        rbbox_nwd(b1, b2), rbbox_nwd(rotate(b1), rotate(b2)), atol=1e-5, rtol=1e-4
    )


def test_angle_representation_invariance() -> None:
    b1, b2 = random_rboxes(5, 4), random_rboxes(4, 5)
    # (w, h, theta) and (h, w, theta + pi/2) describe the same box.
    b1_swapped = torch.cat([b1[:, :2], b1[:, 3:4], b1[:, 2:3], b1[:, 4:] + math.pi / 2], 1)
    torch.testing.assert_close(rbbox_nwd(b1, b2), rbbox_nwd(b1_swapped, b2), atol=1e-5, rtol=1e-4)


def test_aligned_and_empty() -> None:
    b1, b2 = random_rboxes(5, 6), random_rboxes(5, 7)
    torch.testing.assert_close(rbbox_nwd(b1, b2, is_aligned=True), rbbox_nwd(b1, b2).diagonal())
    assert rbbox_nwd(b1[:0], b2).shape == (0, 5)
    assert rbbox_nwd(b1[:0], b2[:0], is_aligned=True).shape == (0,)


def test_calculator() -> None:
    calc = RBboxNWD2D(constant=10.0)
    b1, b2 = random_rboxes(5, 8), random_rboxes(3, 9)
    expected = rbbox_nwd(b1, b2, 10.0)
    torch.testing.assert_close(calc(RotatedBoxes(b1), RotatedBoxes(b2)), expected)
    # default mode='iou' of mmdet assigners still yields NWD; score column is stripped
    scored = torch.cat([b1, torch.rand(5, 1)], dim=1)
    torch.testing.assert_close(calc(scored, b2, mode='iou'), expected)
    iof = calc(b1, b2, mode='iof')
    assert iof.shape == (5, 3)
    assert ((iof >= 0) & (iof <= 1 + 1e-5)).all()
    assert 'constant=10.0' in repr(calc)
