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
    # false alarms per image, for both scorers, in every stage
    for stage in ('A', 'B', 'C'):
        for row in report['stages'][stage]:
            for kind in ('brush', 'ridge'):
                assert f'fppi@r0.5_{kind}' in row and f'r@fppi1_{kind}' in row
                assert 0 <= row[f'r@fppi10_{kind}'] <= 1
    md = (out / 'suggestions.md').read_text()
    assert '## False alarms per image' in md and '## Per class (recommendation)' in md
    # per-class AP, recall and false alarms, in class order, for every row
    nc = len(report['classes'])
    assert len(report['eval_objects']) == nc
    for row in report['stages']['C']:
        for key in ('obj_ap_per_class', 'ridge_ap_per_class'):
            assert len(row[key]) == nc
        for kind in ('brush', 'ridge'):
            for key in ('map_ap_per_class', 'r@fppi1_per_class', 'fppi@r0.5_per_class'):
                assert len(row[f'{key}_{kind}']) == nc
            assert 0 <= row[f'map_ap_{kind}'] <= 1
    for name in report['classes']:
        if report['eval_objects'][report['classes'].index(name)]:
            assert f'| {name} |' in md


def test_suggest_rank_by_fppi(proxy_dota: Path, tmp_path: Path) -> None:
    out = tmp_path / 'sug'
    common = ['--split', 'val', '--data-root', f'{proxy_dota}/', '--num-images', '6']
    common += ['--device', 'cpu', '--sources', 'neck', '--strides', '8', '--dims', '32']
    common += ['--top-k', '1', '--fa-min-objects', '1']
    main([str(TINY), '', *common, '--rank-by', 'fppi', '--out', str(out)])
    report = json.loads((out / 'suggestions.json').read_text())
    assert report['rank_by'] == 'fppi'
    wire = [r for r in report['stages']['C'] if r['dtype'] == 'int8']
    best = min(r['fppi'] for r in wire)
    assert report['recommendation']['fppi'] <= best * 1.1 + 1e-9
    out2 = tmp_path / 'nofa'
    main([str(TINY), '', *common, '--no-false-alarms', '--rank-by', 'fppi', '--out', str(out2)])
    report = json.loads((out2 / 'suggestions.json').read_text())
    assert report['rank_by'] == 'score' and report['false_alarms'] is None
    assert 'fppi' not in report['recommendation']


def test_class_events_and_fppi() -> None:
    # 2 classes on a 6x6 grid; object 0 (class 0) covers (1, 1) and (1, 2), object 1
    # (class 1) covers (4, 4)
    obj_map = torch.full((6, 6), -1, dtype=torch.long)
    obj_map[1, 1], obj_map[1, 2], obj_map[4, 4] = 0, 0, 1
    labels = torch.tensor([0, 1])
    scores = torch.zeros(2, 6, 6)
    scores[0, 1, 2] = 0.9  # object 0's best cell on its own map
    scores[0, 4, 4] = 0.8  # class-0 blob on the class-1 object: other-class false alarm
    scores[0, 1, 5] = 0.7  # class-0 blob on background
    scores[0, 0, 5] = 0.6  # same blob, suppressed by its 0.7 maximum (window 3)
    scores[1, 4, 4] = 0.5  # object 1 on its own map
    obj, vals, codes, ok = metrics.class_events(scores, obj_map, labels, 3, 10)
    np.testing.assert_allclose(obj.numpy(), [0.9, 0.5])
    kept = zip(vals[0][ok[0]], codes[0][ok[0]], strict=True)
    fa0 = {(round(float(v), 2), int(c)) for v, c in kept}
    assert (0.8, metrics.OTHER_CLASS) in fa0 and (0.7, metrics.BACKGROUND) in fa0
    assert all(v not in (0.9, 0.6) for v, _ in fa0)  # own object / suppressed: not alarms
    assert metrics.peak_window(8, 24) == 3 and metrics.peak_window(4, 24) == 7
    assert metrics.peak_window(16, 24) == 3
    # curve: hits at 0.9 and 0.5, false alarms at 0.8 (other class) and 0.7 (background)
    curve = metrics.fppi_curve(
        np.array([0.9, 0.5]), np.array([0.8, 0.7]), np.array([-1, -2]), n_gt=2, n_images=2
    )
    np.testing.assert_allclose(curve['recall'], [0.5, 0.5, 0.5, 1.0])
    np.testing.assert_allclose(curve['fppi'], [0, 0.5, 1.0, 1.0])
    summary = metrics.fppi_summary([curve, None], [0.5, 1.0], [0.5])
    assert summary['fppi@r0.5'] == 0 and summary['fppi@r1'] == 1.0
    assert summary['r@fppi0.5'] == 0.5 and summary['fa_classes'] == 1
    assert summary['fppi@r0.5_per_class'] == [0.0, None]
    assert summary['r@fppi0.5_per_class'] == [0.5, None]
    # map AP: object 1 found at precision 1/1, object 2 at precision 2/4
    assert curve['map_ap'] == pytest.approx((1 + 0.5) / 2) == summary['map_ap']
    never = metrics.fppi_curve(np.array([]), np.array([0.9]), np.array([-2]), n_gt=1, n_images=1)
    assert metrics.fppi_summary([never], [0.5], [1])['fppi@r0.5'] == float('inf')


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
    res = metrics.object_ap(obj, labels, torch.full((60,), 20.0), bg, num_classes=4)
    assert res['obj_ap'] > 0.99
    assert all(a > 0.99 for a in res['obj_ap_per_class'][:3])
    assert res['obj_ap_per_class'][3] is None  # no object of class 3
    r = metrics.ridge_ap(obj, labels, obj, labels, 3)
    assert r['ridge_ap'] > 0.99 and r['ridge_classes'] == 3
    assert len(r['ridge_ap_per_class']) == 3
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
