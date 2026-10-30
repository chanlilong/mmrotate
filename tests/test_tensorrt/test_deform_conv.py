# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from torchvision.ops import deform_conv2d

from mmrotate.tensorrt.deform_conv import modulated_deform_conv2d_gridsample


@pytest.mark.parametrize(
    'stride,padding,dilation,groups,deform_groups',
    [
        (1, 1, 1, 1, 1),
        (1, 3, 3, 1, 1),  # the DCFL heads' dilated DCN
        (2, 1, 1, 1, 1),
        (1, 2, 2, 2, 2),
        (1, 0, 1, 1, 4),
    ],
)
@pytest.mark.parametrize('with_mask', [True, False])
def test_matches_torchvision(
    stride: int, padding: int, dilation: int, groups: int, deform_groups: int, with_mask: bool
) -> None:
    torch.manual_seed(0)
    B, C, H, W, out_ch, k = 2, 8, 11, 13, 6, 3
    x = torch.randn(B, C, H, W)
    weight = torch.randn(out_ch, C // groups, k, k)
    bias = torch.randn(out_ch)
    Ho = (H + 2 * padding - dilation * (k - 1) - 1) // stride + 1
    Wo = (W + 2 * padding - dilation * (k - 1) - 1) // stride + 1
    # Offsets of a few pixels, so some samples fall outside the image.
    offset = 3 * torch.randn(B, deform_groups * 2 * k * k, Ho, Wo)
    mask = torch.rand(B, deform_groups * k * k, Ho, Wo) if with_mask else None
    kwargs = dict(stride=(stride,) * 2, padding=(padding,) * 2, dilation=(dilation,) * 2)
    ref = deform_conv2d(x, offset, weight, bias, mask=mask, **kwargs)
    out = modulated_deform_conv2d_gridsample(
        x, offset, mask, weight, bias, groups=groups, deform_groups=deform_groups, **kwargs
    )
    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
