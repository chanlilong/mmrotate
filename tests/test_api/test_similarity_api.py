# Copyright (c) OpenMMLab. All rights reserved.
"""The similarity service (``/similarity/health``, ``/similarity/embed``)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from api_test_utils import (
    DCFL,
    VITDET,
    check_similarity_health,
    decode_grid,
    expected_cells,
    needs_cuda,
    needs_run,
    random_tile,
)
from fastapi.testclient import TestClient

from api.app import build_services, create_app
from api.codec import encode_png
from api.features import FeatureSpec, Projector
from api.model import LoadedModel
from api.profile import save_profile
from api.settings import Settings, SimilarityOverrides
from api.similarity import SimilarityService


def _settings(lm: LoadedModel, **kwargs: object) -> Settings:
    return Settings(config=lm.config_path, weights=lm.weights_path, **kwargs)  # type: ignore[arg-type]


@pytest.fixture(scope='module')
def tiny_client(tiny_lm: LoadedModel) -> TestClient:
    app = create_app(_settings(tiny_lm, services=('similarity',)), lm=tiny_lm)
    return TestClient(app)


def test_health_contract(tiny_client: TestClient) -> None:
    h = tiny_client.get('/similarity/health').json()
    check_similarity_health(h)
    assert h == tiny_client.get('/health').json()  # single service: also at the root
    assert h['native_size'] == 1024 and h['out_stride'] == 8 and h['feature_dim'] == 96
    assert h['dtype'] == 'int8' and h['scale'] == 1.0
    # no profile, no flags: the finest level of the head's classification features
    assert h['features']['source'] == 'head_cls' and h['features']['levels'] == [0]


@pytest.mark.parametrize(('h', 'w'), [(1024, 1024), (300, 1024), (257, 33), (8, 8)])
def test_embed_shapes_and_norms(tiny_client: TestClient, h: int, w: int) -> None:
    r = tiny_client.post('/similarity/embed', json={'image': encode_png(random_tile(h, w))})
    assert r.status_code == 200, r.text
    body = r.json()
    assert (body['height'], body['width']) == expected_cells(h, w, 8)
    assert body['stride'] == 8 and body['dtype'] == 'int8' and body['scale'] == 127
    grid = decode_grid(body)
    norms = np.linalg.norm(grid, axis=-1)
    assert norms.min() > 0.95 and norms.max() < 1.05


def test_embed_errors(tiny_client: TestClient) -> None:
    big = encode_png(random_tile(1025, 64))
    r = tiny_client.post('/similarity/embed', json={'image': big})
    assert r.status_code == 400 and 'native_size' in r.json()['detail']
    r = tiny_client.post('/similarity/embed', json={'image': 'aGVsbG8='})
    assert r.status_code == 400
    r = tiny_client.post('/similarity/embed', json={})
    assert r.status_code == 422


def test_gsd_is_advertised_not_applied(tiny_lm: LoadedModel) -> None:
    """``gsd_m`` is reported for the qgis server to resample to; the service never rescales."""
    plain = SimilarityService(tiny_lm)
    svc = SimilarityService(tiny_lm, gsd_m=0.5)
    h = svc.health().model_dump(exclude_none=True)
    check_similarity_health(h)
    assert h['gsd_m'] == 0.5 and h['scale'] == 1.0
    assert h['out_stride'] == 8 and h['native_size'] == 1024
    # qgis names the space +gsd<g> itself, so the service's own name is GSD-free
    assert svc.feature_space == plain.feature_space and 'gsd' not in svc.feature_space
    assert 'gsd_m' not in plain.health().model_dump(exclude_none=True)
    out = svc.embed(random_tile(640, 333))
    assert (out.height, out.width) == expected_cells(640, 333, 8)


def test_similarity_gsd_setting(tiny_lm: LoadedModel) -> None:
    """``--similarity-gsd``: default ``--gsd``, a number overrides, ``none`` = native pixels;
    the detector keeps ``--gsd`` either way."""
    for sim_gsd, want in ((None, 0.5), (0.3, 0.3), ('none', None)):
        settings = _settings(tiny_lm, gsd_m=0.5, similarity_gsd_m=sim_gsd)
        det, sim = build_services(settings, tiny_lm)
        assert det is not None and sim is not None
        assert det.health().gsd_m == 0.5
        h = sim.health().model_dump(exclude_none=True)
        check_similarity_health(h)
        assert h.get('gsd_m') == want, sim_gsd


def test_large_tile_is_stitched(tiny_lm: LoadedModel) -> None:
    """A tile bigger than the input runs as aligned windows; the grid covers it exactly and
    its first window's interior matches the tile embedded on its own."""
    svc = SimilarityService(tiny_lm, tile_size=1600, max_batch=4)
    tile = random_tile(1600, 1200)
    grid = svc.embed_unit(tile)
    assert grid.shape[:2] == expected_cells(1600, 1200, 8)
    alone = svc.embed_unit(tile[:1024, :1024])
    inner = (slice(0, 64), slice(0, 64))  # well inside the first window's core
    cos = (grid[inner] * alone[inner]).sum(-1)
    assert cos.min() > 0.999


def test_overrides_and_feature_space(tiny_lm: LoadedModel) -> None:
    base = SimilarityService(tiny_lm)
    head = SimilarityService(tiny_lm, FeatureSpec(source='head_cls', fusion='concat'))
    assert head.feature_dim == 3 * 96 and head.feature_space != base.feature_space
    f16 = SimilarityService(tiny_lm, FeatureSpec(dtype='float16'))
    out = f16.embed(random_tile(64, 64))
    assert out.dtype == 'float16' and out.scale == 1
    assert not f16.feature_space.endswith('-int8') and base.feature_space.endswith('-int8')
    with pytest.raises(ValueError, match='profile'):
        SimilarityService(tiny_lm, FeatureSpec(projection='pca'))
    with pytest.raises(ValueError, match='out_stride'):
        SimilarityService(tiny_lm, FeatureSpec(out_stride=48))


def test_profile_through_settings(tiny_lm: LoadedModel, tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    proj = Projector.fit(rng.normal(size=(400, 288)), 'pca', dim=32)
    spec = FeatureSpec(fusion='concat', projection='pca')
    path = save_profile(tmp_path / 'p.npz', spec, proj, {'input_size': [800, 800]})
    settings = _settings(
        tiny_lm,
        services=('similarity',),
        similarity_profile=str(path),
        sim=SimilarityOverrides(out_stride=16),
        self_test=False,
    )
    client = TestClient(create_app(settings, lm=tiny_lm))
    h = client.get('/health').json()
    check_similarity_health(h)
    assert h['feature_dim'] == 32 and h['out_stride'] == 16
    assert h['features']['projector']['kind'] == 'pca'
    r = client.post('/embed', json={'image': encode_png(random_tile(100, 70))})
    assert (r.json()['height'], r.json()['width'], r.json()['dim']) == (7, 5, 32)


# ----------------------------------------------------------------- real models (GPU)
@needs_cuda
@needs_run(VITDET)
def test_vitdet_embed(load_run: Callable[..., LoadedModel], soda_patches: list) -> None:
    lm = load_run(VITDET)
    client = TestClient(create_app(_settings(lm, services=('similarity',)), lm=lm))
    h = client.get('/health').json()
    check_similarity_health(h)
    assert h['native_size'] == 800 and h['feature_dim'] == 128
    img = soda_patches[0][0]
    r = client.post('/embed', json={'image': encode_png(np.ascontiguousarray(img[:300]))})
    body = r.json()
    assert (body['height'], body['width']) == (38, 100)
    norms = np.linalg.norm(decode_grid(body), axis=-1)
    assert norms.min() > 0.95


@needs_cuda
@needs_run(VITDET)
@pytest.mark.parametrize('pad_mode', ['square', 'divisor'])
def test_vitdet_crop_alignment(
    load_run: Callable[..., LoadedModel], soda_patches: list, pad_mode: str
) -> None:
    """A crop's cells line up with the same cells of the whole patch.

    The ViT attends globally, so a cell's vector depends on the whole window (and on the
    padding of a short crop): agreement is high but not 1. What must hold is the grid
    alignment, i.e. the aligned offset agrees better than any one-cell shift.
    """
    lm = load_run(VITDET, pad_mode=pad_mode)
    svc = SimilarityService(lm)
    img = soda_patches[0][0]
    full = svc.embed_unit(img)
    y0, x0, h, w = 104, 200, 400, 480  # multiples of the stride
    crop = svc.embed_unit(np.ascontiguousarray(img[y0 : y0 + h, x0 : x0 + w]))
    m = 8  # cells; edge cells see a different context
    a = crop[m:-m, m:-m]
    cy, cx = y0 // 8 + m, x0 // 8 + m

    def agreement(dy: int, dx: int) -> float:
        b = full[cy + dy : cy + dy + a.shape[0], cx + dx : cx + dx + a.shape[1]]
        return float(np.median((a * b).sum(-1)))

    aligned = agreement(0, 0)
    shifted = [agreement(dy, dx) for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))]
    print(f'pad_mode={pad_mode}: aligned {aligned:.3f}, one-cell shifts {shifted}')
    assert aligned > max(shifted)
    assert aligned > 0.7


@needs_cuda
@pytest.mark.parametrize('run', [VITDET, DCFL], ids=['sepbn', 'dcfl'])
@pytest.mark.parametrize('source', ['backbone', 'head_cls', 'head_reg', 'head_cls@0'])
def test_real_sources(
    load_run: Callable[..., LoadedModel], soda_patches: list, run: Path, source: str
) -> None:
    if not (run / 'epoch_6.pth').exists():
        pytest.skip(f'no checkpoint in {run}')
    lm = load_run(run)
    fusion = 'concat' if source == 'backbone' else 'mean'
    svc = SimilarityService(lm, FeatureSpec(source=source, fusion=fusion))
    svc.self_test()
    out = svc.embed_unit(soda_patches[0][0][:256, :512])
    assert out.shape[:2] == (32, 64)
