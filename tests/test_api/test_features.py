# Copyright (c) OpenMMLab. All rights reserved.
"""Feature extraction, fusion, projection and profiles (api/features.py, api/profile.py).

CPU, on a randomly initialised rotated_rtmdet_tiny (3 levels, strides 8/16/32).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from api_test_utils import random_tile

from api.features import (
    FeatureExtractor,
    FeatureSpec,
    Moments,
    Projector,
    batch_inputs,
    default_spec,
    dequantize,
    fuse,
    parse_source,
    postprocess,
    quantize,
    spec_digest,
)
from api.model import LoadedModel
from api.profile import load_profile, save_profile


@pytest.fixture(scope='module')
def captured(tiny_lm: LoadedModel) -> tuple[dict, tuple[int, int]]:
    inputs = batch_inputs(tiny_lm.model, [random_tile(256, 320)])
    parts = parse_source('backbone+neck+head_cls+head_reg+head_cls@0+head_reg@-1')
    with torch.inference_mode():
        out = FeatureExtractor(tiny_lm.model).capture(inputs, parts)
    return out, tuple(inputs.shape[-2:])


def test_capture_every_source(captured: tuple) -> None:
    maps, hw = captured
    assert hw == (256, 320)
    for key in ('backbone', 'neck', 'head_cls', 'head_reg', 'head_cls@0', 'head_reg@-1'):
        assert len(maps[key]) == 3, key
        # level order = finest first (strides 8, 16, 32)
        assert [m.shape[-2] for m in maps[key]] == [32, 16, 8], key


def test_head_hooks_match_manual_forward(tiny_lm: LoadedModel, captured: tuple) -> None:
    maps, _ = captured
    head = tiny_lm.model.bbox_head
    with torch.inference_mode():
        for lvl, x in enumerate(maps['neck']):
            cls_feat = x
            for layer in head.cls_convs[lvl]:
                cls_feat = layer(cls_feat)
            torch.testing.assert_close(maps['head_cls'][lvl], cls_feat)
            first = head.cls_convs[lvl][0](x)
            torch.testing.assert_close(maps['head_cls@0'][lvl], first)
            # head_reg@-1 is the last reg conv == the regression predictor's input
            torch.testing.assert_close(maps['head_reg@-1'][lvl], maps['head_reg'][lvl])


@pytest.mark.parametrize('stride', [4, 8, 16, 32])
@pytest.mark.parametrize(
    ('source', 'fusion', 'levels', 'dim'),
    [
        ('neck', 'mean', None, 96),
        ('neck', 'concat', None, 288),
        ('neck', 'single', (1,), 96),
        ('backbone', 'concat', None, 96 + 192 + 384),
        ('backbone', 'concat', (0, 2), 96 + 384),
        ('head_cls+head_reg', 'concat', (0,), 192),
        ('head_cls+head_reg', 'mean', None, 96),
    ],
)
def test_fuse_shapes(
    captured: tuple, source: str, fusion: str, levels: tuple | None, dim: int, stride: int
) -> None:
    maps, hw = captured
    spec = FeatureSpec(source=source, fusion=fusion, levels=levels, out_stride=stride)
    fused = fuse(maps, spec, hw)
    assert fused.shape == (1, dim, -(-hw[0] // stride), -(-hw[1] // stride))
    unit = postprocess(fused, spec, None)
    torch.testing.assert_close(unit.norm(dim=-1), torch.ones(unit.shape[:-1]))


def test_fuse_errors(captured: tuple) -> None:
    maps, hw = captured
    with pytest.raises(ValueError, match='equal channels'):
        fuse(maps, FeatureSpec(source='backbone', fusion='mean'), hw)
    with pytest.raises(ValueError, match='exactly one'):
        fuse(maps, FeatureSpec(source='neck', fusion='single'), hw)
    with pytest.raises(ValueError, match='out of range'):
        fuse(maps, FeatureSpec(source='neck', levels=(5,)), hw)
    with pytest.raises(ValueError):
        FeatureSpec(source='nope')
    with pytest.raises(ValueError):
        FeatureSpec(source='neck@1')
    with pytest.raises(ValueError):
        postprocess(fuse(maps, FeatureSpec(), hw), FeatureSpec(projection='pca'), None)


def test_projector_matches_numpy_pca() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 12)) @ rng.normal(size=(12, 12))
    p = Projector.fit(x, 'pca', dim=4)
    xc = x - x.mean(0)
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    np.testing.assert_allclose(np.abs(p.components), np.abs(vt[:4]), atol=1e-4)
    np.testing.assert_allclose(p.explained_variance, s[:4] ** 2 / (len(x) - 1), rtol=1e-4)
    # streamed moments give the same basis
    m = Moments.zeros(12)
    for chunk in np.array_split(x, 7):
        m.add(torch.as_tensor(chunk))
    q = Projector.from_moments(m, 'pca', dim=4)
    np.testing.assert_allclose(p.components, q.components, atol=1e-5)
    c = Projector.fit(x, 'center')
    assert c.components is None and c.out_dim == 12
    y = c.apply(torch.as_tensor(x, dtype=torch.float32))
    assert y.mean(0).abs().max() < 1e-4
    w = Projector.fit(x, 'pca', dim=4, whiten=True)
    yw = w.apply(torch.as_tensor(x, dtype=torch.float32))
    np.testing.assert_allclose(yw.var(0).numpy(), np.ones(4), rtol=1e-2)


def test_quantize_roundtrip() -> None:
    unit = torch.nn.functional.normalize(torch.randn(10, 7, 128), dim=-1).numpy()
    q, scale = quantize(unit, 'int8')
    assert q.dtype == np.int8 and scale == 127
    back = dequantize(q, scale)
    cos = (back * unit).sum(-1) / np.linalg.norm(back, axis=-1)
    assert cos.min() > 0.99
    f16, s16 = quantize(unit, 'float16')
    assert f16.dtype == np.float16 and s16 == 1


def test_profile_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    p = Projector.fit(rng.normal(size=(200, 16)), 'pca', dim=8, whiten=True)
    spec = FeatureSpec(source='head_cls', fusion='concat', levels=(0, 1), projection='pca')
    path = save_profile(tmp_path / 'p.npz', spec, p, {'input_size': [800, 800]})
    spec2, p2, meta = load_profile(path)
    assert spec2 == spec and meta == {'input_size': [800, 800]}
    assert p2 is not None and p2.whiten and p2.digest() == p.digest()
    assert spec_digest(spec, p) == spec_digest(spec2, p2)
    p3 = Projector.fit(rng.normal(size=(200, 16)), 'pca', dim=8)
    assert spec_digest(spec, p) != spec_digest(spec, p3)
    assert spec_digest(spec, p) != spec_digest(spec.replace(out_stride=4), p)


def test_spec_dict_roundtrip() -> None:
    spec = FeatureSpec(source='head_cls@1+neck', levels=[0, 2], fusion='concat')
    assert FeatureSpec.from_dict(spec.to_dict()) == spec
    assert spec.with_overrides({'out_stride': 16}).out_stride == 16
    assert spec.tag() == 'head_clsc1_neck-concat-L02-s8'


def test_default_spec(tiny_lm: LoadedModel) -> None:
    assert default_spec(tiny_lm.model) == FeatureSpec(
        source='head_cls', fusion='single', levels=(0,)
    )

    class NoHead(torch.nn.Module):
        bbox_head = torch.nn.Conv2d(1, 1, 1)  # no cls predictor to hook

    assert default_spec(NoHead()) == FeatureSpec()  # neck mean
