# Copyright (c) OpenMMLab. All rights reserved.
import math

import pytest
import torch
from torch import Tensor

from mmrotate.models.task_modules.dcfl import RBboxGJSD2D, gaussian_density, rbbox_gjsd
from mmrotate.models.task_modules.nwd_rka import box_to_gaussian
from mmrotate.structures.bbox import RotatedBoxes


def _ref_xy_sigma(xywhr: Tensor) -> tuple[Tensor, Tensor]:
    """``xy_wh_r_2_xy_sigma`` of mmrotate-dcfl's rotate_metric_calculator.py."""
    _shape = xywhr.shape
    xy = xywhr[..., :2]
    wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
    r = xywhr[..., 4]
    cos_r, sin_r = torch.cos(r), torch.sin(r)
    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    S = 0.5 * torch.diag_embed(wh)
    sigma = R.bmm(S.square()).bmm(R.permute(0, 2, 1)).reshape(_shape[:-1] + (2, 2))
    return xy, sigma


def _inv(m: Tensor) -> Tensor:
    inv = torch.stack((m[..., 1, 1], -m[..., 0, 1], -m[..., 1, 0], m[..., 0, 0]), dim=-1)
    return inv.reshape(m.shape) / m.det()[..., None, None]


def _ref_gjsd(boxes1: Tensor, boxes2: Tensor, alpha: float = 0.5) -> Tensor:
    """``get_gjsd`` of mmrotate-dcfl (CPU branch), condensed."""
    xy_p, sigma_p = _ref_xy_sigma(boxes1)
    xy_t, sigma_t = _ref_xy_sigma(boxes2)
    sigma_p_inv, sigma_t_inv = _inv(sigma_p), _inv(sigma_t)
    xy_p = xy_p[:, None, :, None]
    xy_t = xy_t[None, :, :, None]
    sigma_p, sigma_p_inv = sigma_p[:, None], sigma_p_inv[:, None]
    sigma_t, sigma_t_inv = sigma_t[None], sigma_t_inv[None]
    sigma_alpha = _inv((1 - alpha) * sigma_p_inv + alpha * sigma_t_inv)
    sigma_alpha_inv = _inv(sigma_alpha)
    mu_alpha = sigma_alpha.matmul(
        (1 - alpha) * sigma_p_inv.matmul(xy_p) + alpha * sigma_t_inv.matmul(xy_t)
    )
    first = (
        (1 - alpha) * xy_p.transpose(-1, -2).matmul(sigma_p_inv).matmul(xy_p)
        + alpha * xy_t.transpose(-1, -2).matmul(sigma_t_inv).matmul(xy_t)
        - mu_alpha.transpose(-1, -2).matmul(sigma_alpha_inv).matmul(mu_alpha)
    )
    second = (sigma_p.det() ** (1 - alpha) * sigma_t.det() ** alpha / sigma_alpha.det()).log()
    return 0.5 * (first.squeeze(-1).squeeze(-1) + second)


def _boxes(n: int, seed: int) -> Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.cat(
        [
            torch.rand(n, 2, generator=g) * 100,
            torch.rand(n, 2, generator=g) * 40 + 0.5,
            (torch.rand(n, 1, generator=g) - 0.5) * 3,
        ],
        dim=1,
    )


def test_matches_reference() -> None:
    boxes1, boxes2 = _boxes(13, 0), _boxes(17, 1)
    ref = 1 / (1 + _ref_gjsd(boxes1.double(), boxes2.double()))
    out = rbbox_gjsd(boxes1, boxes2)
    assert out.shape == (13, 17) and out.dtype == torch.float32
    torch.testing.assert_close(out.double(), ref, rtol=1e-5, atol=1e-6)


def test_symmetric_identity_and_range() -> None:
    boxes1, boxes2 = _boxes(9, 2), _boxes(11, 3)
    torch.testing.assert_close(rbbox_gjsd(boxes1, boxes2), rbbox_gjsd(boxes2, boxes1).T)
    torch.testing.assert_close(rbbox_gjsd(boxes1, boxes1).diag(), torch.ones(9))
    sim = rbbox_gjsd(boxes1, boxes2)
    assert ((sim > 0) & (sim <= 1)).all()
    # a rotation by pi is the same Gaussian
    flipped = boxes1.clone()
    flipped[:, 4] += torch.pi
    torch.testing.assert_close(rbbox_gjsd(boxes1, flipped, is_aligned=True), torch.ones(9))


def test_aligned_and_horizontal() -> None:
    boxes1, boxes2 = _boxes(8, 4), _boxes(8, 5)
    torch.testing.assert_close(
        rbbox_gjsd(boxes1, boxes2, is_aligned=True), rbbox_gjsd(boxes1, boxes2).diag()
    )
    hbox = torch.tensor([[0.0, 0, 10, 4]])
    rbox = torch.tensor([[5.0, 2, 10, 4, 0]])
    torch.testing.assert_close(rbbox_gjsd(hbox, rbox), torch.ones(1, 1))


def test_empty() -> None:
    assert rbbox_gjsd(torch.zeros(0, 5), _boxes(3, 0)).shape == (0, 3)
    assert rbbox_gjsd(torch.zeros(0, 5), torch.zeros(0, 5), is_aligned=True).shape == (0,)


def test_tiny_far_boxes_stay_finite() -> None:
    """The closed form has no absolute-coordinate terms to cancel."""
    boxes = torch.tensor([[5000.0, 5000, 2, 2, 0.3], [5001.0, 5000, 2, 2, 0.3]])
    sim = rbbox_gjsd(boxes[:1], boxes[1:])
    # Sigma = I for both, so GJSD = 1/4 * d^T (2 I)^-1 d + 1/2 * ln(det(2 I) / 4) = 1/8
    torch.testing.assert_close(sim, torch.tensor([[1 / (1 + 0.125)]]))


def test_calculator() -> None:
    calc = RBboxGJSD2D()
    boxes1, boxes2 = _boxes(4, 6), _boxes(5, 7)
    torch.testing.assert_close(
        calc(RotatedBoxes(boxes1), RotatedBoxes(boxes2)), rbbox_gjsd(boxes1, boxes2)
    )
    scored = torch.cat([boxes1, torch.rand(4, 1)], dim=1)
    torch.testing.assert_close(calc(scored, boxes2), rbbox_gjsd(boxes1, boxes2))
    if torch.cuda.is_available():
        iof = calc(boxes1.cuda(), boxes2.cuda(), mode='iof')
        assert iof.shape == (4, 5) and ((iof >= 0) & (iof <= 1)).all()


@pytest.mark.parametrize('theta', [0.0, 0.7])
def test_gaussian_density(theta: float) -> None:
    box = torch.tensor([[10.0, 20, 8, 4, theta]])
    gauss = box_to_gaussian(box)
    # one standard deviation (w / 2 = 4 px) along the box's width axis
    direction = torch.tensor([[math.cos(theta), math.sin(theta)]])
    point = box[:, :2] + 4 * direction
    torch.testing.assert_close(
        gaussian_density(point, box[:, :2], gauss), torch.exp(torch.tensor([-0.5]))
    )
    torch.testing.assert_close(gaussian_density(box[:, :2], box[:, :2], gauss), torch.ones(1))
