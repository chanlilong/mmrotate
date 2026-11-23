# Copyright (c) OpenMMLab. All rights reserved.
"""Shared fixtures for the REST API tests.

CPU tests use a randomly initialised ``rotated_rtmdet_tiny`` (DOTA config, 1024 input, so
nothing here leans on SODA's 800 px). GPU tests use real checkpoints from ``work_dirs/`` and
skip when CUDA or the checkpoint is missing.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
from api_test_utils import SODA_TEST, TINY_CONFIG, run_dir_files

from api.model import LoadedModel, load


@pytest.fixture(scope='session')
def tiny_lm() -> LoadedModel:
    torch.manual_seed(0)
    return load(TINY_CONFIG, None, device='cpu')


@pytest.fixture(scope='session')
def load_run() -> Callable[..., LoadedModel]:
    cache: dict[tuple, LoadedModel] = {}

    def _load(run: Path, **kwargs: object) -> LoadedModel:
        key = (str(run), tuple(sorted(kwargs.items())))
        if key not in cache:
            cfg, ckpt = run_dir_files(run)
            cache[key] = load(cfg, ckpt, device='cuda:0', **kwargs)  # type: ignore[arg-type]
        return cache[key]

    return _load


@pytest.fixture(scope='session')
def soda_patches() -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """A few SODA-A test patches with objects: ``(bgr, quads [K, 8], class names)``."""
    import cv2

    if not SODA_TEST.exists():
        pytest.skip('no SODA-A test split')
    out = []
    for ann in sorted((SODA_TEST / 'annfiles').glob('*.txt'))[:400]:
        rows = [line.split() for line in ann.read_text().splitlines() if len(line.split()) >= 9]
        if len(rows) < 10:
            continue
        img = cv2.imread(str(SODA_TEST / 'images' / f'{ann.stem}.jpg'))
        if img is None:
            continue
        quads = np.array([[float(v) for v in r[:8]] for r in rows], np.float32)
        out.append((img, quads, np.array([r[8] for r in rows])))
        if len(out) == 4:
            break
    if not out:
        pytest.skip('no SODA-A test patch with objects')
    return out
