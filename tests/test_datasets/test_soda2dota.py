# Copyright (c) OpenMMLab. All rights reserved.
import importlib.util
import json
import os
import os.path as osp
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / 'tools/data/soda_a/soda2dota.py'


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location('soda2dota', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_sample(root: Path, name: str, anns: list[dict], split: str = 'train') -> None:
    (root / 'Images').mkdir(parents=True, exist_ok=True)
    (root / 'Annotations' / split).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(root / 'Images' / f'{name}.jpg'), np.full((64, 80, 3), 255, np.uint8))
    data = dict(
        images=dict(file_name=f'{name}.jpg', height=64, width=80, id=1),
        annotations=anns,
        categories=[dict(id=i, name=n) for i, n in enumerate(_load_script().SODA_A_CLASSES)]
        + [dict(id=9, name='ignore')],
    )
    with open(root / 'Annotations' / split / f'{name}.json', 'w') as f:
        json.dump(data, f)


def test_poly_to_quad() -> None:
    mod = _load_script()
    quad = mod.poly_to_quad([0, 0, 10, 0, 10, 5, 0, 5])
    assert quad.shape == (4, 2)
    hexagon = [0, 2, 2, 0, 8, 0, 10, 2, 8, 4, 2, 4]
    rect = mod.poly_to_quad(hexagon)
    assert rect.shape == (4, 2)
    assert cv2.contourArea(rect) == np.float32(40.0)
    assert mod.poly_to_quad([0, 0, 1, 1]) is None


def test_convert(tmp_path: Path) -> None:
    src, out = tmp_path / 'soda', tmp_path / 'out'
    anns = [
        dict(poly=[2, 2, 12, 2, 12, 8, 2, 8], area=60.0, category_id=2, id=1),
        dict(poly=[20, 22, 22, 20, 28, 20, 30, 22, 28, 24, 22, 24], area=36.0, category_id=0, id=2),
        dict(poly=[40, 30, 70, 30, 70, 60, 40, 60], area=None, category_id=9, id=3),
    ]
    _write_sample(src, '00001', anns)
    _write_sample(src, '00002', anns[:1])
    subprocess.run(
        [sys.executable, str(SCRIPT), str(src), str(out), '--splits', 'train', '--nproc', '1'],
        check=True,
    )
    lines = (out / 'train/labelTxt/00001.txt').read_text().splitlines()
    assert len(lines) == 2  # the ignore region is not an object
    assert lines[0].split()[-2:] == ['small-vehicle', '0']
    assert lines[1].split()[-2:] == ['airplane', '0']
    assert len(lines[1].split()) == 10  # hexagon -> 4 points

    painted = cv2.imread(str(out / 'train/images/00001.jpg'))
    assert np.abs(painted[45, 55].astype(int) - [104, 116, 124]).max() <= 3
    assert painted[10, 5].min() >= 250  # untouched elsewhere
    # an image without ignore regions is just linked
    assert osp.islink(out / 'train/images/00002.jpg')
    assert os.readlink(out / 'train/images/00002.jpg') == str(src / 'Images/00002.jpg')
