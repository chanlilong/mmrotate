# Copyright (c) OpenMMLab. All rights reserved.
"""Image-size handling (api/tiling.py) and settings: CPU only, no weights."""

from __future__ import annotations

import numpy as np
import pytest
from api_test_utils import TINY_CONFIG
from mmengine.config import Config

from api import tiling
from api.settings import Settings, parse_size, settings_from_args


def _cfg(pipeline: list[dict], divisor: int = 32) -> Config:
    return Config(
        {
            'model': {'data_preprocessor': {'pad_size_divisor': divisor}},
            'test_dataloader': {'dataset': {'type': 'DOTADataset', 'pipeline': pipeline}},
        }
    )


LOAD = {'type': 'mmdet.LoadImageFromFile'}
ANN = {'type': 'mmdet.LoadAnnotations', 'with_bbox': True, 'box_type': 'qbox'}
PACK = {'type': 'mmdet.PackDetInputs', 'meta_keys': ('img_id', 'img_path', 'ori_shape')}


@pytest.mark.parametrize(
    ('pipeline', 'size'),
    [
        ([LOAD, {'type': 'mmdet.Resize', 'scale': (800, 800), 'keep_ratio': True}, PACK], 800),
        ([LOAD, {'type': 'mmdet.Resize', 'scale': (1333, 800), 'keep_ratio': True}, PACK], 800),
        ([LOAD, {'type': 'mmdet.Resize', 'scale': 640}, PACK], 640),
        (
            [
                LOAD,
                {'type': 'mmdet.Resize', 'scale': (1024, 1024), 'keep_ratio': True},
                {'type': 'mmdet.Pad', 'size': (1024, 1024)},
                PACK,
            ],
            1024,
        ),
        ([LOAD, PACK], tiling.DEFAULT_INPUT_SIZE),
    ],
)
def test_detect_input_size(pipeline: list[dict], size: int) -> None:
    (w, h), source = tiling.detect_input_size(_cfg(pipeline))
    assert (w, h) == (size, size)
    assert source


def test_detect_input_size_rect_without_keep_ratio() -> None:
    cfg = _cfg([LOAD, {'type': 'mmdet.Resize', 'scale': (1024, 768), 'keep_ratio': False}])
    assert tiling.detect_input_size(cfg)[0] == (1024, 768)


def test_detect_input_size_dota_config() -> None:
    size, source = tiling.detect_input_size(Config.fromfile(TINY_CONFIG))
    assert size == (1024, 1024) and 'Pad' in source


def test_inference_pipeline_drops_resize_in_none_mode() -> None:
    pipe = [LOAD, {'type': 'mmdet.Resize', 'scale': (800, 800), 'keep_ratio': True}, ANN, PACK]
    kinds = [s['type'] for s in tiling.inference_pipeline_cfg(_cfg(pipe), 'none')]
    assert kinds == ['mmdet.LoadImageFromNDArray', 'mmdet.PackDetInputs']
    kinds = [s['type'] for s in tiling.inference_pipeline_cfg(_cfg(pipe), 'config')]
    assert kinds == ['mmdet.LoadImageFromNDArray', 'mmdet.Resize', 'mmdet.PackDetInputs']
    pack = tiling.inference_pipeline_cfg(_cfg(pipe), 'none')[-1]
    assert 'scale_factor' in pack['meta_keys']


def test_resize_mode_none_never_upsamples(tiny_lm: object) -> None:
    """A small tile reaches the network at its own size (padded, not enlarged)."""
    small = np.zeros((300, 200, 3), np.uint8)
    data = tiny_lm.collate([small])  # type: ignore[attr-defined]
    assert tuple(data['inputs'][0].shape[-2:]) == (300, 200)
    assert data['data_samples'][0].scale_factor == (1.0, 1.0)


def test_check_divisible() -> None:
    tiling.check_divisible(800, 32, 'x', 'hint')
    with pytest.raises(ValueError, match='use 992 or 1024'):
        tiling.check_divisible(1000, 32, 'input width', 'pad_size_divisor')


def test_window_starts_and_cores() -> None:
    assert tiling.window_starts(800, 800, 128) == [0]
    assert tiling.window_starts(500, 800, 128) == [0]
    starts = tiling.window_starts(1600, 800, 128)
    assert starts == [0, 672, 1344]
    cores = tiling.core_ranges(starts, 800, 1600)
    assert cores[0][0] == 0 and cores[-1][1] == 1600
    assert all(a[1] == b[0] for a, b in zip(cores[:-1], cores[1:], strict=True))
    # aligned starts and boundaries
    starts = tiling.window_starts(2000, 800, 100, align=8)
    assert all(s % 8 == 0 for s in starts) and starts[-1] + 800 >= 2000
    cores = tiling.core_ranges(starts, 800, 2000, align=8)
    assert all(lo % 8 == 0 for lo, _ in cores)
    with pytest.raises(ValueError):
        tiling.window_starts(2000, 800, 800)


def test_plan_windows_cover_tile() -> None:
    wins = tiling.plan_windows(1000, 1700, 800, 800, 128)
    covered = np.zeros((1000, 1700), bool)
    for w in wins:
        assert w.width <= 800 and w.height <= 800
        covered[w.y0 : w.y0 + w.height, w.x0 : w.x0 + w.width] = True
    assert covered.all()


def test_pad_to() -> None:
    out = tiling.pad_to(np.ones((3, 5, 3), np.uint8), 8, 8)
    assert out.shape == (8, 8, 3) and out[:3, :5].all() and not out[3:].any()
    with pytest.raises(ValueError):
        tiling.pad_to(np.ones((9, 5, 3), np.uint8), 8, 8)


# ----------------------------------------------------------------- settings
def test_parse_size() -> None:
    assert parse_size('800') == (800, 800)
    assert parse_size('1024x768') == (1024, 768)
    assert parse_size(None) is None
    with pytest.raises(ValueError):
        parse_size('0')


def test_settings_env_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('MMROTATE_API_CONFIG', 'c.yaml')
    monkeypatch.setenv('MMROTATE_API_WEIGHTS', 'w.pth')
    monkeypatch.setenv('MMROTATE_API_GSD', '0.5')
    monkeypatch.setenv('MMROTATE_API_INPUT_SIZE', '1024')
    monkeypatch.setenv('MMROTATE_API_SERVICES', 'similarity')
    monkeypatch.setenv('MMROTATE_API_SIM_LEVELS', '0,2')
    s = settings_from_args(['--gsd', '0.3'])  # the flag wins over the environment
    assert (s.config, s.weights, s.gsd_m) == ('c.yaml', 'w.pth', 0.3)
    assert s.input_size == (1024, 1024) and s.services == ('similarity',)
    assert s.sim.as_dict() == {'levels': [0, 2]}


def test_similarity_gsd_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    base = ['--config', 'c.yaml', '--weights', 'w.pth', '--gsd', '0.5']
    s = settings_from_args(base)
    assert s.similarity_gsd_m is None and s.similarity_gsd() == 0.5  # follows --gsd
    s = settings_from_args([*base, '--similarity-gsd', 'None'])
    assert s.similarity_gsd_m == 'none' and s.similarity_gsd() is None  # native pixels
    monkeypatch.setenv('MMROTATE_API_SIMILARITY_GSD', '0.3')
    s = settings_from_args(base)
    assert s.similarity_gsd() == 0.3 and s.gsd_m == 0.5  # the detector keeps --gsd
    with pytest.raises(ValueError):
        settings_from_args([*base, '--similarity-gsd', 'abc'])
    assert Settings(config='c', weights='w', similarity_gsd_m=0.3).similarity_gsd() == 0.3


def test_settings_validation() -> None:
    with pytest.raises(ValueError):
        Settings(config='c', weights='w', services=('nope',))
    with pytest.raises(ValueError):
        Settings(config='c', weights='w', similarity_gsd_m=-1.0)
    with pytest.raises(ValueError):
        Settings(config='c', weights='w', gsd_m=0.0)
    with pytest.raises(ValueError):
        Settings(config='c', weights='w', pad_mode='odd')
