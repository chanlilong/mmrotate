# Copyright (c) OpenMMLab. All rights reserved.
"""Paths, skip markers and qgis_annotations contract checks shared by the API tests."""

from __future__ import annotations

import base64
import math
import os.path as osp
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
TINY_CONFIG = str(REPO / 'configs/rotated_rtmdet/rotated_rtmdet_tiny-3x-dota.py')
SODA = REPO / 'work_dirs/soda_dinov3'
VITDET = SODA / 'vits-dinov3-vitdet_rtmdet-p2-6e-soda'
DCFL = SODA / 'convnext-t-dinov3_rtmdet-p2-dcfl-6e-soda'
SODA_TEST = Path('/mnt/zfs/datasets/eo/soda-a/split_800/test')

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')


def run_dir_files(run: Path) -> tuple[str, str]:
    return str(run / f'{run.name}.yaml'), str(run / 'epoch_6.pth')


def needs_run(run: Path) -> pytest.MarkDecorator:
    cfg, ckpt = run_dir_files(run)
    return pytest.mark.skipif(
        not (osp.exists(cfg) and osp.exists(ckpt)), reason=f'no checkpoint in {run}'
    )


# ----------------------------------------------------------------- contract checks
# Mirror of the qgis_annotations client checks (server/inference/remote.py,
# services.probe): what a registered service must answer to be listed healthy.
def check_detector_health(h: dict) -> None:
    assert h['ready'] is True
    assert isinstance(h['model_name'], str) and h['model_name']
    assert isinstance(h['classes'], list) and all(isinstance(c, str) for c in h['classes'])
    assert h.get('gsd_m') is None or isinstance(h['gsd_m'], int | float)
    assert isinstance(h['native_size'], int) and h['native_size'] > 0
    assert isinstance(h['max_batch'], int) and h['max_batch'] >= 1
    assert 'class_agnostic_nms' in h['capabilities']
    assert h['model_version']


def check_similarity_health(h: dict) -> None:
    assert h['ready'] is True
    assert 'embed' in h['capabilities']
    for key in ('native_size', 'out_stride', 'feature_dim'):
        assert isinstance(h[key], int) and h[key] > 0, key
    assert h['native_size'] % h['out_stride'] == 0
    assert h['model_version'] and h['feature_space'] and len(h['feature_space']) <= 200


def decode_grid(resp: dict) -> np.ndarray:
    dtype = {'int8': np.int8, 'float16': np.float16, 'float32': np.float32}[resp['dtype']]
    raw = base64.b64decode(resp['data'])
    arr = np.frombuffer(raw, dtype).reshape(resp['height'], resp['width'], resp['dim'])
    return arr.astype(np.float32) / resp['scale']


def expected_cells(h: int, w: int, stride: int) -> tuple[int, int]:
    return math.ceil(h / stride), math.ceil(w / stride)


def random_tile(h: int, w: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)
