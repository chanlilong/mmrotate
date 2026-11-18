# Copyright (c) OpenMMLab. All rights reserved.
"""Parquet result bridge: exact round trips, padding dedup, file rendezvous."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
import pytest
import torch
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData

from mmrotate.accelerate.result_bridge import (
    ParquetResultBridge,
    decode,
    encode,
    results_to_table,
    table_to_results,
)


def assert_same(a: object, b: object) -> None:
    assert type(a) is type(b), (type(a), type(b))
    if isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape  # type: ignore[union-attr]
        np.testing.assert_array_equal(a, b)  # type: ignore[arg-type]
    elif isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape  # type: ignore[union-attr]
        assert torch.equal(a, b)  # type: ignore[arg-type]
    elif isinstance(a, dict):
        assert list(a) == list(b)  # type: ignore[arg-type]
        for k in a:
            assert_same(a[k], b[k])  # type: ignore[index]
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)  # type: ignore[arg-type]
        for x, y in zip(a, b, strict=True):  # type: ignore[call-overload]
            assert_same(x, y)
    elif isinstance(a, float) and np.isnan(a):
        assert np.isnan(b)  # type: ignore[arg-type]
    else:
        assert a == b


def dota_like_result(seed: int) -> tuple[dict, dict]:
    """The (ann, result) tuple DOTAMetric / SODAMetric append per image."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(0, 6))
    ann = {
        'bboxes': rng.random((n, 5)).astype(np.float32),
        'labels': rng.integers(0, 15, n).astype(np.int64),
        'bboxes_ignore': np.zeros((0, 5), np.float32),
        'labels_ignore': np.zeros((0,), np.int64),
    }
    result = {
        'img_id': f'P{seed:04d}__1024__0___0',
        'bboxes': rng.random((n, 5)).astype(np.float32),
        'scores': rng.random(n).astype(np.float32),
        'labels': rng.integers(0, 15, n).astype(np.int64),
        'pred_bbox_scores': [rng.random((k, 6)).astype(np.float32) for k in (0, 2, 1)],
    }
    return ann, result


@pytest.mark.parametrize(
    'obj',
    [
        None,
        True,
        3,
        -2.5,
        float('nan'),
        'x',
        b'\x00\x01',
        np.float32(1.5),
        np.int64(7),
        np.zeros((0, 5), np.float32),
        np.arange(6, dtype='>i4').reshape(2, 3),  # big-endian survives
        np.array(4.0),  # 0-d
        np.array([1, 'a', None], dtype=object),
        torch.arange(6, dtype=torch.float32).reshape(2, 3),
        torch.tensor([1.5, -2.0], dtype=torch.bfloat16),
        torch.tensor([True, False]),
        {1: 'int key', (2, 3): 'tuple key', 'k': [1, (2, None)]},
        dota_like_result(3),
    ],
)
def test_round_trip(obj: object) -> None:
    spec, arrays = encode(obj)
    assert_same(obj, decode(spec, arrays))


def test_round_trip_data_element() -> None:
    sample = DetDataSample(metainfo={'img_id': 'a', 'ori_shape': (4, 4)})
    sample.pred_instances = InstanceData(
        bboxes=torch.rand(3, 5), scores=torch.rand(3), labels=torch.tensor([0, 2, 1])
    )
    out = decode(*encode(sample))
    assert isinstance(out, DetDataSample)
    assert out.img_id == 'a' and out.ori_shape == (4, 4)
    assert torch.equal(out.pred_instances.bboxes, sample.pred_instances.bboxes)
    assert torch.equal(out.pred_instances.labels, sample.pred_instances.labels)


def test_unsupported_type_names_the_path() -> None:
    with pytest.raises(TypeError, match=r"result\['a'\]\[1\]"):
        encode({'a': [1, object()]})


def test_merge_drops_sampler_padding_and_restores_order() -> None:
    # DefaultSampler(shuffle=False), 7 samples, 3 ranks: indices padded to 9.
    n, world = 7, 3
    padded = list(range(n)) + list(range(9 - n))
    parts = []
    for rank in range(world):
        idx = padded[rank::world]
        results = [dota_like_result(i) for i in idx]
        parts.append(results_to_table([(results, idx)], rank))
    import pyarrow as pa

    merged = table_to_results(pa.concat_tables(parts[::-1]), 1)[0]
    assert len(merged) == n
    for i, res in enumerate(merged):
        assert_same(res, dota_like_result(i))


def test_merge_keeps_multiple_results_per_sample() -> None:
    import pyarrow as pa

    t0 = results_to_table([(['a0', 'a1', 'c'], [0, 0, 2])], 0)
    t1 = results_to_table([(['b', 'a0-dup', 'a1-dup'], [1, 0, 0])], 1)
    assert table_to_results(pa.concat_tables([t1, t0]), 1)[0] == ['a0', 'a1', 'b', 'c']


def test_bridge_threads(tmp_path: Path) -> None:
    """Three 'ranks' rendezvous through files; all get rank 0's metrics."""
    world, n = 3, 8
    padded = list(range(n)) + [0]
    out: dict[int, object] = {}
    errors: list[BaseException] = []

    def compute(merged: list[list[object]]) -> dict:
        return {'n': len(merged[0]), 'first': merged[0][0], 'arr': np.arange(3)}

    def run(rank: int) -> None:
        try:
            idx = padded[rank::world]
            bridge = ParquetResultBridge(str(tmp_path / 'round'), rank, world, timeout_s=30)
            out[rank] = bridge.run([([f's{i}' for i in idx], idx)], compute)
            bridge.cleanup(wait_s=30)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=run, args=(r,)) for r in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    for r in range(world):
        assert out[r]['n'] == n and out[r]['first'] == 's0'  # type: ignore[index]
        np.testing.assert_array_equal(out[r]['arr'], np.arange(3))  # type: ignore[index]
    assert not os.path.exists(tmp_path / 'round')  # cleaned up after the acks


def test_bridge_error_unblocks_other_ranks(tmp_path: Path) -> None:
    errors: dict[int, BaseException] = {}

    def compute(merged: list[list[object]]) -> dict:
        raise ValueError('metric exploded')

    def run(rank: int) -> None:
        try:
            ParquetResultBridge(str(tmp_path), rank, 2, timeout_s=30).run([([1], [rank])], compute)
        except BaseException as e:  # noqa: BLE001
            errors[rank] = e

    threads = [threading.Thread(target=run, args=(r,)) for r in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert isinstance(errors[0], ValueError)
    assert isinstance(errors[1], RuntimeError) and 'metric exploded' in str(errors[1])


def test_bridge_timeout_names_missing_ranks(tmp_path: Path) -> None:
    bridge = ParquetResultBridge(str(tmp_path), 0, 2, timeout_s=1)
    with pytest.raises(TimeoutError, match='part_00001'):
        bridge.run([([1], [0])], lambda merged: {})
