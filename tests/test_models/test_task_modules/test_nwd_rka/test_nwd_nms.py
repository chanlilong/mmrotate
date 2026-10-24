# Copyright (c) OpenMMLab. All rights reserved.
import math
import os

import pytest
import torch
from mmcv.ops import batched_nms

from mmrotate.models.task_modules.nwd_rka import box_to_gaussian, nwd_nms, rbbox_nwd
from mmrotate.models.task_modules.nwd_rka.nwd_nms import HAS_TRITON, nwd_nms_mask_pytorch

TRITON_INTERPRET = os.environ.get('TRITON_INTERPRET') == '1'


def random_rboxes(num: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    xy = torch.rand(num, 2, generator=gen) * 60
    wh = torch.rand(num, 2, generator=gen) * 20 + 2
    theta = (torch.rand(num, 1, generator=gen) - 0.5) * math.pi
    return torch.cat([xy, wh, theta], dim=1)


def brute_force_nms(
    boxes: torch.Tensor, scores: torch.Tensor, thr: float, constant: float
) -> torch.Tensor:
    order = scores.argsort(descending=True)
    nwd = rbbox_nwd(boxes, boxes, constant)
    keep: list[int] = []
    for i in order.tolist():
        if all(nwd[k, i] <= thr for k in keep):
            keep.append(i)
    return torch.tensor(keep, dtype=torch.long)


@pytest.mark.parametrize('num', [0, 1, 63, 64, 65, 200])
def test_matches_brute_force(num: int) -> None:
    boxes = random_rboxes(num, num)
    scores = torch.rand(num, generator=torch.Generator().manual_seed(1))
    dets, keep = nwd_nms(boxes, scores, 0.5, constant=8.0, use_triton=False)
    assert torch.equal(keep, brute_force_nms(boxes, scores, 0.5, 8.0))
    assert dets.shape == (keep.numel(), 6)
    torch.testing.assert_close(dets[:, -1], scores[keep])


def test_horizontal_boxes() -> None:
    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=torch.float32)
    scores = torch.tensor([0.9, 0.8, 0.7])
    dets, keep = nwd_nms(boxes, scores, 0.85)
    assert keep.tolist() == [0, 2]
    assert dets.shape == (2, 5)


def test_batched_nms_resolves_type() -> None:
    boxes = torch.tensor(
        [[10, 10, 8, 8, 0], [10.5, 10, 8, 8, 0], [10.5, 10, 8, 8, 0]], dtype=torch.float32
    )
    scores = torch.tensor([0.9, 0.8, 0.7])
    labels = torch.tensor([0, 0, 1])
    dets, keep = batched_nms(
        boxes, scores, labels, dict(type='nwd_nms', iou_threshold=0.8, constant=12.8)
    )
    # box 1 is a duplicate of box 0 (same class); box 2 has another class
    assert sorted(keep.tolist()) == [0, 2]
    assert dets.shape == (2, 6)


@pytest.mark.skipif(
    not (HAS_TRITON and (torch.cuda.is_available() or TRITON_INTERPRET)),
    reason='requires triton with CUDA or TRITON_INTERPRET=1',
)
@pytest.mark.parametrize('num', [1, 64, 130])
def test_triton_mask_matches_pytorch(num: int) -> None:
    from mmrotate.models.task_modules.nwd_rka.nwd_nms_triton import nwd_nms_mask_triton

    device = 'cuda' if torch.cuda.is_available() and not TRITON_INTERPRET else 'cpu'
    gauss = box_to_gaussian(random_rboxes(num, 3)).to(device)
    expected = nwd_nms_mask_pytorch(gauss, 0.4, 10.0)
    assert torch.equal(nwd_nms_mask_triton(gauss, 0.4, 10.0), expected)
