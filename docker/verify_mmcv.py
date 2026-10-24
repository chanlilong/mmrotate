# Copyright (c) OpenMMLab. All rights reserved.
"""GPU smoke test for the dev image: run inside the container on the GPU host.

    python docker/verify_mmcv.py

Checks that the mmcv CUDA ops mmrotate relies on actually run on the GPU, that the
Triton MSDA kernel used by the C-RADIOv4 / ViT-CoMer backbone compiles, and that
mmrotate imports (which runs its mmcv / mmdet / mmengine version asserts).
"""

from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def check_versions() -> None:
    import mmcv
    import mmdet
    import mmengine
    from mmcv.ops import get_compiler_version, get_compiling_cuda_version

    import mmrotate

    print(f'torch {torch.__version__} (CUDA {torch.version.cuda})')
    print(
        f'mmcv {mmcv.__version__} (ops built with CUDA {get_compiling_cuda_version()}, '
        f'{get_compiler_version()})'
    )
    print(
        f'mmengine {mmengine.__version__} | mmdet {mmdet.__version__} | '
        f'mmrotate {mmrotate.__version__}'
    )
    assert torch.cuda.is_available(), 'CUDA not available: run with --gpus all'
    major, minor = torch.cuda.get_device_capability()
    print(f'GPU: {torch.cuda.get_device_name()} (sm_{major}{minor})')


def check_mmcv_ops() -> None:
    from mmcv.ops import DeformConv2d, box_iou_rotated, min_area_polygons, nms_rotated

    dev = torch.device('cuda')
    boxes = torch.tensor(
        [[10.0, 10.0, 8.0, 4.0, 0.1], [10.5, 10.0, 8.0, 4.0, 0.1], [40.0, 40.0, 6.0, 6.0, 0.0]],
        device=dev,
    )
    scores = torch.tensor([0.9, 0.8, 0.7], device=dev)
    _, keep = nms_rotated(boxes, scores, iou_threshold=0.5)
    assert keep.tolist() == [0, 2], f'nms_rotated kept {keep.tolist()}'

    ious = box_iou_rotated(boxes, boxes)
    assert torch.allclose(ious.diagonal(), torch.ones(3, device=dev), atol=1e-4), ious

    # 9 (x, y) points per row: the corners of a 4x2 rectangle plus interior points.
    points = torch.tensor(
        [[0, 0, 4, 0, 4, 2, 0, 2, 1, 1, 2, 1, 3, 1, 2, 0.5, 2, 1.5]],
        dtype=torch.float32,
        device=dev,
    )
    assert min_area_polygons(points).shape == (1, 8)

    conv = DeformConv2d(4, 8, kernel_size=3, padding=1).to(dev)
    x = torch.randn(2, 4, 16, 16, device=dev)
    offset = torch.zeros(2, 18, 16, 16, device=dev)
    conv(x, offset).sum().backward()
    print('mmcv CUDA ops: nms_rotated, box_iou_rotated, min_area_polygons, DeformConv2d OK')


def check_triton_msda() -> None:
    from mmrotate.models.backbones.cradiov4.ms_deform_attn import (
        msda_pytorch,
        multi_scale_deformable_attn,
    )

    dev = torch.device('cuda')
    shapes = torch.tensor([[8, 8], [4, 4]], device=dev)
    B, N, H, C, L, P = 2, 5, 2, 16, 2, 4
    value = torch.randn(B, int((shapes[:, 0] * shapes[:, 1]).sum()), H, C, device=dev)
    loc = torch.rand(B, N, H, L, P, 2, device=dev)
    attn = torch.rand(B, N, H, L, P, device=dev).softmax(-1)
    out = multi_scale_deformable_attn(value, shapes, loc, attn, backend='triton')
    ref = msda_pytorch(value, shapes, loc, attn)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
    print('Triton MSDA kernel OK')


def check_config() -> None:
    from mmengine.config import Config

    cfg = Config.fromfile(REPO_ROOT / 'configs/rotated_rtmdet/rotated_rtmdet_s-3x-dota.py')
    assert cfg.pretty_text  # exercises yapf, which has broken mmengine before
    print('mmengine Config OK')


def main() -> None:
    check_versions()
    check_mmcv_ops()
    check_triton_msda()
    check_config()
    print('All checks passed.')


if __name__ == '__main__':
    main()
