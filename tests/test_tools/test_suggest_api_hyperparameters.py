# Copyright (c) OpenMMLab. All rights reserved.
"""tools/similarity/suggest_api_hyperparameters.py end to end.

CPU: random-init rotated_rtmdet_tiny on generated proxy DOTA tiles. GPU: the SODA-A ViTDet
checkpoint on a handful of test patches. Both check that the written profile loads in the
API's similarity service.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from api.model import load
from api.profile import load_profile
from api.similarity import SimilarityService
from tools.similarity import metrics
from tools.similarity.suggest_api_hyperparameters import main

REPO = Path(__file__).resolve().parents[2]
TINY = REPO / 'configs/rotated_rtmdet/rotated_rtmdet_tiny-3x-dota.py'
VITDET = REPO / 'work_dirs/soda_dinov3/vits-dinov3-vitdet_rtmdet-p2-6e-soda'


@pytest.fixture(scope='module')
def proxy_dota(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp('proxy_dota')
    subprocess.run(
        [
            sys.executable,
            str(REPO / 'tools/data/dota/make_proxy_dota.py'),
            '--out',
            str(out),
            '--num-train',
            '6',
            '--num-test',
            '1',
            '--size',
            '512',
        ],
        check=True,
    )
    return out


def _check_outputs(out: Path, config: str, weights: str | None, device: str) -> dict:
    report = json.loads((out / 'suggestions.json').read_text())
    assert set(report['stages']) == {'A', 'B', 'C'}
    assert report['recommendation']['dtype'] == 'int8'
    assert '--similarity-profile' in report['command']
    md = (out / 'suggestions.md').read_text()
    assert '## Recommendation' in md and 'Stage C' in md
    spec, projector, meta = load_profile(out / 'similarity_profile.npz')
    assert meta['input_size'] == report['input_size']
    lm = load(config, weights, device=device)
    svc = SimilarityService(lm, spec, projector, profile_meta=meta)
    svc.self_test()
    assert svc.feature_dim == report['recommendation']['dim']
    return report


def test_suggest_cpu_proxy(proxy_dota: Path, tmp_path: Path) -> None:
    out = tmp_path / 'sug'
    main(
        [
            str(TINY),
            '',
            '--split',
            'val',
            '--data-root',
            f'{proxy_dota}/',
            '--num-images',
            '6',
            '--device',
            'cpu',
            '--sources',
            'neck',
            'head_cls',
            '--strides',
            '8',
            '16',
            '--dims',
            '32',
            '--top-k',
            '2',
            '--out',
            str(out),
        ]
    )
    report = _check_outputs(out, str(TINY), None, 'cpu')
    tags = [r['tag'] for r in report['stages']['A']]
    assert any(t.startswith('neck-concat') for t in tags)
    assert any(t.startswith('head_cls-single-L2') for t in tags)
    assert {r['projection'] for r in report['stages']['C']} >= {'none', 'center', 'pca32'}


def test_metrics_basics() -> None:
    scores = torch.tensor([[0.9, 0.8, 0.1], [0.1, 0.2, 0.9]])
    pos = torch.tensor([[True, False, True], [False, False, False]])
    ap = metrics.average_precision_rows(scores, pos)
    assert ap[0] == pytest.approx((1 + 2 / 3) / 2) and torch.isnan(ap[1])
    # perfectly separable classes -> AP 1
    rng = np.random.default_rng(0)
    centres = torch.nn.functional.normalize(torch.randn(3, 16), dim=-1)
    labels = torch.as_tensor(rng.integers(0, 3, 60))
    obj = torch.nn.functional.normalize(centres[labels] + 0.01 * torch.randn(60, 16), dim=-1)
    bg = torch.nn.functional.normalize(-centres.sum(0, keepdim=True).repeat(20, 1), dim=-1)
    res = metrics.object_ap(obj, labels, torch.full((60,), 20.0), bg)
    assert res['obj_ap'] > 0.99
    r = metrics.ridge_ap(obj, labels, obj, labels, 3)
    assert r['ridge_ap'] > 0.99 and r['ridge_classes'] == 3
    agree = metrics.int8_agreement(obj)
    assert agree['int8_cos'] > 0.99
    assert metrics.grid_bytes_per_512(8, 128, 'int8') == 64 * 64 * 128


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
@pytest.mark.skipif(not (VITDET / 'epoch_6.pth').exists(), reason='no ViTDet checkpoint')
@pytest.mark.skipif(
    not Path('/mnt/zfs/datasets/eo/soda-a/split_800/test').exists(), reason='no SODA-A'
)
def test_suggest_gpu_vitdet(tmp_path: Path) -> None:
    config = str(VITDET / f'{VITDET.name}.yaml')
    weights = str(VITDET / 'epoch_6.pth')
    out = tmp_path / 'sug'
    main(
        [
            config,
            weights,
            '--num-images',
            '12',
            '--strides',
            '8',
            '16',
            '--dims',
            '64',
            '--top-k',
            '1',
            '--gsd',
            '0.5',
            '--out',
            str(out),
        ]
    )
    report = _check_outputs(out, config, weights, 'cuda:0')
    assert '--gsd 0.5' in report['command']
    assert report['recommendation']['score'] > 0.1
