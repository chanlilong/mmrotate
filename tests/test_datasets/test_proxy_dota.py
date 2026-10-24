# Copyright (c) OpenMMLab. All rights reserved.
import importlib.util
import os.path as osp
import sys
from pathlib import Path
from types import ModuleType

import pytest

from mmrotate.datasets import DOTADataset

SCRIPT = osp.join(osp.dirname(__file__), '..', '..', 'tools', 'data', 'dota', 'make_proxy_dota.py')


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location('make_proxy_dota', SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_proxy_classes_match_dota() -> None:
    assert _load_script().DOTA_CLASSES == DOTADataset.METAINFO['classes']


def test_make_proxy_dota(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = _load_script()
    args = [
        'make_proxy_dota.py',
        '--out',
        str(tmp_path),
        '--num-train',
        '3',
        '--num-test',
        '1',
        '--size',
        '128',
        '--max-objects',
        '8',
    ]
    monkeypatch.setattr(sys, 'argv', args)
    script.main()

    assert len(list((tmp_path / 'trainval' / 'images').glob('*.png'))) == 3
    assert len(list((tmp_path / 'trainval' / 'annfiles').glob('*.txt'))) == 3
    assert len(list((tmp_path / 'test' / 'images').glob('*.png'))) == 1

    dataset = DOTADataset(
        data_root=str(tmp_path),
        ann_file='trainval/annfiles/',
        data_prefix=dict(img_path='trainval/images/'),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=[],
    )
    assert len(dataset) == 3
    for info in dataset:
        assert info['instances']
        for inst in info['instances']:
            assert len(inst['bbox']) == 8
            assert all(0 <= v <= 128 for v in inst['bbox'])
            assert 0 <= inst['bbox_label'] < 15


def test_class_style_is_distinct() -> None:
    script = _load_script()
    styles = {script.class_style(i) for i in range(len(script.DOTA_CLASSES))}
    assert len(styles) == len(script.DOTA_CLASSES)
