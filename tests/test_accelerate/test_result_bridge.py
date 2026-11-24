# Copyright (c) OpenMMLab. All rights reserved.
"""Parquet result bridge: exact round trips, padding dedup, file rendezvous."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
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
    publish_status,
    read_status,
    results_to_table,
    sweep_old_rounds,
    table_to_results,
    wait_for,
    write_table,
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


def test_rank0_never_arriving_fails_on_the_arrival_timeout(tmp_path: Path) -> None:
    """Rank 1 must not sit out the long metric timeout when rank 0 never shows up."""
    status = tmp_path / '_ranks'
    publish_status(str(status), 0, 'val:ready', 'old_round')
    bridge = ParquetResultBridge(
        str(tmp_path / 'round'),
        1,
        2,
        timeout_s=3600,
        arrive_timeout_s=0.5,
        status_dir=str(status),
        round_tag='this_round',
    )
    start = time.monotonic()
    with pytest.raises(TimeoutError) as info:
        bridge.run([([1], [1])], lambda merged: {})
    assert time.monotonic() - start < 30
    msg = str(info.value)
    assert 'part_00000' in msg and 'rank 0 never reached this evaluation round' in msg
    assert 'phase=val:ready' in msg and 'round=old_round' in msg


def test_slow_metric_still_gets_the_full_timeout(tmp_path: Path) -> None:
    """Once rank 0 has arrived, a metric slower than arrive_timeout_s is fine."""
    out: dict[int, object] = {}
    errors: list[BaseException] = []

    def compute(merged: list[list[object]]) -> dict:
        time.sleep(1.5)
        return {'n': len(merged[0])}

    def run(rank: int) -> None:
        try:
            bridge = ParquetResultBridge(
                str(tmp_path / 'round'), rank, 2, timeout_s=30, arrive_timeout_s=0.5
            )
            out[rank] = bridge.run([([rank], [rank])], compute)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=run, args=(r,)) for r in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    assert out[0] == out[1] == {'n': 2}


def test_rank0_done_before_rank1_polls(tmp_path: Path) -> None:
    """Rank 0 deletes its part after the metric; metrics.done counts as arrived."""
    root = tmp_path / 'round'
    root.mkdir()
    part1 = root / 'part_00001.parquet'
    write_table(results_to_table([([1], [1])], 1), str(part1))
    (root / 'part_00001.parquet.done').touch()
    r0 = ParquetResultBridge(str(root), 0, 2, timeout_s=5)
    assert r0.run([([0], [0])], lambda merged: {'n': len(merged[0])}) == {'n': 2}
    assert not (root / 'part_00000.parquet.done').exists()
    r1 = ParquetResultBridge(str(root), 1, 2, timeout_s=5, arrive_timeout_s=0.5)
    assert r1.run([([1], [1])], lambda merged: {}) == {'n': 2}


def test_wait_for_reports_progress(tmp_path: Path) -> None:
    calls: list[tuple[list[str], float]] = []
    missing = str(tmp_path / 'never')
    with pytest.raises(TimeoutError, match='extra text'):
        wait_for(
            lambda: [missing],
            0.6,
            'testing',
            poll_s=0.05,
            on_slow=lambda m, t: calls.append((m, t)),
            report_every_s=0.1,
            on_timeout=lambda m: 'extra text',
        )
    assert len(calls) >= 3 and all(m == [missing] for m, _ in calls)


def test_sweep_keeps_status_dir(tmp_path: Path) -> None:
    publish_status(str(tmp_path / '_ranks'), 0, 'val:ready')
    (tmp_path / 'val_old').mkdir()
    (tmp_path / 'val_new').mkdir()
    sweep_old_rounds(str(tmp_path), keep=str(tmp_path / 'val_new'))
    assert sorted(os.listdir(tmp_path)) == ['_ranks', 'val_new']
    info = read_status(str(tmp_path / '_ranks'), 0)
    assert info is not None and info['pid'] == os.getpid() and info['phase'] == 'val:ready'


@pytest.mark.skipif(not hasattr(os, 'kill') or sys.platform == 'win32', reason='POSIX signals')
def test_timeout_dumps_the_missing_peers_stack(tmp_path: Path) -> None:
    """The waiting rank makes a stuck rank 0 on the same host print its Python stack."""
    status = tmp_path / '_ranks'
    code = (
        'import sys, time\n'
        'from mmrotate.accelerate.result_bridge import publish_status\n'
        'def stuck_in_checkpoint_saving():\n'
        '    time.sleep(60)\n'
        f'publish_status({str(status)!r}, 0, "val:ready", "r1")\n'
        'print("ready", flush=True)\n'
        'stuck_in_checkpoint_saving()\n'
    )
    child = subprocess.Popen(
        [sys.executable, '-c', code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        for line in child.stdout:  # skip import-time noise (DeepSpeed's accelerator warning)
            if line.strip() == 'ready':
                break
        else:
            pytest.fail('child exited before publishing its status')
        bridge = ParquetResultBridge(
            str(tmp_path / 'round'),
            1,
            2,
            timeout_s=3600,
            arrive_timeout_s=0.5,
            status_dir=str(status),
            round_tag='r1',
        )
        with pytest.raises(TimeoutError, match=f'SIGUSR1 sent to pid {child.pid}'):
            bridge.run([([1], [1])], lambda merged: {})
    finally:
        child.kill()
        _, err = child.communicate(timeout=30)
    assert 'stuck_in_checkpoint_saving' in err, err


@pytest.mark.skipif(not hasattr(os, 'kill') or sys.platform == 'win32', reason='POSIX signals')
def test_stack_dump_survives_a_native_handler_takeover() -> None:
    """Triton's LLVM replaces the SIGUSR1 handler in C; re-enabling must take it back."""
    code = (
        'import ctypes, os, signal, time\n'
        'from mmrotate.accelerate.result_bridge import enable_stack_dumps\n'
        'assert enable_stack_dumps()\n'
        'libc = ctypes.CDLL(None)\n'
        "libc.signal(signal.SIGUSR1, 1)  # SIG_IGN behind Python's back, like LLVM does\n"
        'os.kill(os.getpid(), signal.SIGUSR1); time.sleep(0.2)\n'
        'print("ignored", flush=True)\n'
        'assert enable_stack_dumps()\n'
        'def after_takeover():\n'
        '    os.kill(os.getpid(), signal.SIGUSR1); time.sleep(0.2)\n'
        'after_takeover()\n'
    )
    proc = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert 'ignored' in proc.stdout
    assert proc.stderr.count('Current thread') == 1, proc.stderr
    assert 'after_takeover' in proc.stderr
