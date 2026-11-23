# Copyright (c) OpenMMLab. All rights reserved.
"""Image-size handling shared by the detector, the similarity service and the suggest tool.

Nothing here assumes a particular dataset. The model input size is detected from the
config's test pipeline when it is not given, every size is validated up front, and tiles
larger than the input are run as overlapping windows of the input size.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from mmengine.config import Config, ConfigDict

DEFAULT_INPUT_SIZE = 1024

# Pipeline steps that only deal with annotations; never part of inference.
_ANNOTATION_STEPS = ('LoadAnnotations', 'LoadPanopticAnnotations', 'ConvertBoxType')
_RESIZE_STEPS = ('Resize', 'RResize', 'FixShapeResize', 'RandomResize')
_LOAD_STEPS = ('LoadImageFromFile', 'LoadPatchFromNDArray', 'LoadImageFromNDArray')


def _kind(step: dict) -> str:
    return str(step.get('type', '')).split('.')[-1]


def test_pipeline_cfg(cfg: Config) -> list[dict]:
    """The test pipeline: ``test_pipeline``, else the test/val dataloader's dataset pipeline."""
    if cfg.get('test_pipeline'):
        return list(cfg.test_pipeline)
    for loader in ('test_dataloader', 'val_dataloader'):
        dl = cfg.get(loader)
        if not dl:
            continue
        dataset = dl.get('dataset', {})
        # wrappers (ConcatDataset, RepeatDataset, ...) keep the real one under dataset/datasets
        while 'pipeline' not in dataset and ('dataset' in dataset or 'datasets' in dataset):
            dataset = dataset.get('dataset') or dataset['datasets'][0]
        if dataset.get('pipeline'):
            return list(dataset['pipeline'])
    return []


def detect_input_size(cfg: Config) -> tuple[tuple[int, int], str]:
    """``((w, h), where-it-came-from)`` of the model input, from the test pipeline.

    Order: a ``Pad`` with a fixed ``size``; a square ``Resize``/``RResize`` ``scale``; the
    shorter side of a rectangular keep-ratio scale (a square tile of that side fits inside
    it unchanged); else :data:`DEFAULT_INPUT_SIZE`.
    """
    pipeline = test_pipeline_cfg(cfg)
    for step in pipeline:
        if _kind(step) == 'Pad' and step.get('size') is not None:
            size = step['size']
            w, h = (size, size) if isinstance(size, int) else (int(size[0]), int(size[1]))
            return (w, h), 'test pipeline Pad.size'
    for step in pipeline:
        if _kind(step) in _RESIZE_STEPS and step.get('scale') is not None:
            scale = step['scale']
            if isinstance(scale, int):
                return (scale, scale), f'test pipeline {_kind(step)}.scale'
            if isinstance(scale, Sequence) and len(scale) == 2 and isinstance(scale[0], int):
                w, h = int(scale[0]), int(scale[1])
                if w == h or not step.get('keep_ratio', False):
                    return (w, h), f'test pipeline {_kind(step)}.scale'
                side = min(w, h)
                return (side, side), f'shorter side of test pipeline {_kind(step)}.scale {w}x{h}'
    return (DEFAULT_INPUT_SIZE, DEFAULT_INPUT_SIZE), f'default {DEFAULT_INPUT_SIZE}'


def inference_pipeline_cfg(cfg: Config, resize_mode: str = 'none') -> list[dict]:
    """The test pipeline adapted to in-memory tiles.

    ``LoadImageFromFile`` becomes ``mmdet.LoadImageFromNDArray`` and annotation steps are
    dropped. With ``resize_mode='none'`` the ``Resize``/``Pad`` steps are dropped too: the
    service pads tiles itself and never rescales them (a keep-ratio ``Resize`` would silently
    enlarge a short or small tile and change its GSD). ``'config'`` keeps them.
    """
    out: list[dict] = []
    has_load = False
    for step in test_pipeline_cfg(cfg):
        step = copy.deepcopy(dict(step))
        kind = _kind(step)
        if kind in _ANNOTATION_STEPS:
            continue
        if kind in _LOAD_STEPS:
            out.append({'type': 'mmdet.LoadImageFromNDArray'})
            has_load = True
            continue
        if resize_mode == 'none' and (kind in _RESIZE_STEPS or kind == 'Pad'):
            continue
        out.append(step)
    if not has_load:
        out.insert(0, {'type': 'mmdet.LoadImageFromNDArray'})
    if not any(_kind(s) == 'PackDetInputs' for s in out):
        out.append(
            {
                'type': 'mmdet.PackDetInputs',
                'meta_keys': ('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'),
            }
        )
    for step in out:
        if _kind(step) == 'PackDetInputs' and step.get('meta_keys') is not None:
            keys = tuple(k for k in step['meta_keys'] if k != 'instances')
            if 'scale_factor' not in keys:
                keys += ('scale_factor',)  # needed to map boxes back from a Resize
            step['meta_keys'] = keys
    return [ConfigDict(s) for s in out]


def pad_size_divisor(cfg: Config) -> int:
    dp = cfg.model.get('data_preprocessor') or {}
    return int(dp.get('pad_size_divisor', 1) or 1)


def pad_to(img: np.ndarray, height: int, width: int) -> np.ndarray:
    """Zero-pad ``img`` bottom-right to ``(height, width)`` (never crops)."""
    h, w = img.shape[:2]
    if h == height and w == width:
        return img
    if h > height or w > width:
        raise ValueError(f'pad_to: image {w}x{h} larger than {width}x{height}')
    out = np.zeros((height, width) + img.shape[2:], dtype=img.dtype)
    out[:h, :w] = img
    return out


def window_starts(total: int, window: int, overlap: int, align: int = 1) -> list[int]:
    """Start offsets of windows of ``window`` px covering ``[0, total)``.

    Consecutive windows overlap by at least ``overlap`` px; every start is a multiple of
    ``align``. The last window may run past ``total`` (it is padded), so starts stay aligned.
    """
    if window <= 0:
        raise ValueError('window must be > 0')
    if total <= window:
        return [0]
    step = window - overlap
    step = (step // align) * align
    if step <= 0:
        raise ValueError(
            f'overlap {overlap} leaves no step for window {window} (aligned to {align})'
        )
    starts = [0]
    while starts[-1] + window < total:
        starts.append(starts[-1] + step)
    return starts


def core_ranges(starts: Sequence[int], window: int, total: int, align: int = 1) -> list[tuple]:
    """``[(lo, hi), ...]``: the part of ``[0, total)`` each window is trusted for.

    Boundaries sit at the middle of each overlap (snapped to ``align``), so every pixel/cell
    is taken from the window where it is farthest from an edge, and the ranges tile
    ``[0, total)`` exactly.
    """
    out = []
    for i, s in enumerate(starts):
        lo = 0 if i == 0 else out[-1][1]
        if i == len(starts) - 1:
            hi = total
        else:
            mid = (starts[i + 1] + s + window) / 2
            hi = int(round(mid / align)) * align
            hi = max(lo, min(hi, total))
        out.append((lo, hi))
    return out


@dataclass(frozen=True)
class Window:
    """One model window cut from a tile: its offset and the region of the tile it covers."""

    x0: int
    y0: int
    width: int
    height: int


def plan_windows(
    height: int, width: int, win_h: int, win_w: int, overlap: int, align: int = 1
) -> list[Window]:
    """Overlapping windows of at most ``win_h x win_w`` covering a ``height x width`` tile."""
    ys = window_starts(height, win_h, overlap, align)
    xs = window_starts(width, win_w, overlap, align)
    return [
        Window(x0=x, y0=y, width=min(win_w, width - x), height=min(win_h, height - y))
        for y in ys
        for x in xs
    ]


def check_divisible(value: int, divisor: int, what: str, hint: str) -> None:
    if divisor > 1 and value % divisor:
        lower = (value // divisor) * divisor
        upper = lower + divisor
        raise ValueError(
            f'{what} = {value} is not a multiple of {divisor} ({hint}); '
            f'use {lower} or {upper} instead'
        )


def snap_down(value: int, multiple: int) -> int:
    return max(multiple, (value // multiple) * multiple)
