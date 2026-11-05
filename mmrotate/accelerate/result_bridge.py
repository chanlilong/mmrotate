# Copyright (c) OpenMMLab. All rights reserved.
"""Parquet bridge for evaluation results between ranks (no pickle, no NCCL).

mmengine's ``BaseMetric.evaluate`` gathers every rank's ``metric.results`` with
``collect_results`` (pickles over NCCL or a pickle file per rank in a tmpdir)
and then broadcasts the metrics. Rank 0 computes the metric while the other
ranks sit in a collective, so a slow metric (DOTA / SODA-A mAP with patch
merging) can trip the NCCL watchdog.

This bridge moves the same data through files instead:

1. every rank encodes its ``metric.results`` (any nesting of dict / list /
   tuple / ndarray / tensor / scalar / str / None / data element) into a
   parquet file, one row per result with the dataset index it belongs to;
2. rank 0 waits for all parts, drops the duplicates the distributed sampler
   pads in (by dataset index), restores dataset order, decodes and runs
   ``metric.compute_metrics``;
3. rank 0 writes the metrics as a one-row parquet file, the other ranks poll
   for it. No collective is open while the metric is computed.

Arrays are stored as raw bytes plus dtype / shape in a JSON spec, so decoding
is exact and nothing is unpickled.
"""

from __future__ import annotations

import importlib
import json
import os
import os.path as osp
import shutil
import time
import traceback
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from mmengine.structures import BaseDataElement

SCHEMA = pa.schema(
    [
        ('metric', pa.int32()),
        ('index', pa.int64()),
        ('rank', pa.int32()),
        ('seq', pa.int64()),
        ('spec', pa.large_string()),
        ('arrays', pa.list_(pa.large_binary())),
    ]
)

_TORCH_DTYPES = {
    str(dt): dt
    for dt in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.bool,
        torch.complex64,
        torch.complex128,
    )
}


# --------------------------------------------------------------------------- encoding


def encode(obj: object) -> tuple[str, list[bytes]]:
    """Encode ``obj`` as a JSON spec plus a list of raw array buffers."""
    arrays: list[bytes] = []
    spec = _encode(obj, arrays, 'result')
    return json.dumps(spec, allow_nan=True, separators=(',', ':')), arrays


def decode(spec: str, arrays: Sequence[bytes]) -> object:
    """Inverse of :func:`encode`."""
    return _decode(json.loads(spec), arrays)


def _put_array(arr: np.ndarray, arrays: list[bytes]) -> int:
    arrays.append(np.ascontiguousarray(arr).tobytes())
    return len(arrays) - 1


def _encode(obj: object, arrays: list[bytes], path: str) -> dict[str, Any]:
    # bool before int: bool is a subclass of int.
    if obj is None:
        return {'t': 'none'}
    if isinstance(obj, bool):
        return {'t': 'bool', 'v': obj}
    if isinstance(obj, int):
        return {'t': 'int', 'v': obj}
    if isinstance(obj, float):
        return {'t': 'float', 'v': obj}
    if isinstance(obj, str):
        return {'t': 'str', 'v': obj}
    if isinstance(obj, bytes):
        return {'t': 'bytes', 'i': _put_array(np.frombuffer(obj, np.uint8), arrays)}
    if isinstance(obj, np.ndarray):
        if obj.dtype.hasobject:
            flat = [_encode(x, arrays, f'{path}[{i}]') for i, x in enumerate(obj.ravel().tolist())]
            return {'t': 'ndobj', 'shape': list(obj.shape), 'v': flat}
        return {
            't': 'nd',
            'dtype': obj.dtype.str,
            'shape': list(obj.shape),
            'i': _put_array(obj, arrays),
        }
    if isinstance(obj, np.generic):
        arr = np.asarray(obj)
        return {'t': 'npscalar', 'dtype': arr.dtype.str, 'i': _put_array(arr, arrays)}
    if isinstance(obj, torch.Tensor):
        t = obj.detach().cpu()
        dtype = str(t.dtype)
        if dtype not in _TORCH_DTYPES:
            raise TypeError(f'{path}: unsupported tensor dtype {t.dtype}')
        # numpy has no bfloat16: store bf16 as its raw int16 bit pattern.
        arr = t.view(torch.int16).numpy() if t.dtype == torch.bfloat16 else t.numpy()
        return {
            't': 'tensor',
            'dtype': dtype,
            'shape': list(t.shape),
            'i': _put_array(arr, arrays),
        }
    if isinstance(obj, BaseDataElement):
        cls = type(obj)
        return {
            't': 'element',
            'cls': f'{cls.__module__}:{cls.__qualname__}',
            'meta': _encode(dict(obj.metainfo_items()), arrays, f'{path}.metainfo'),
            'data': _encode(dict(obj.items()), arrays, f'{path}.data'),
        }
    if isinstance(obj, dict):
        keys = list(obj.keys())
        return {
            't': 'dict',
            'k': [_encode(k, arrays, f'{path}.<key>') for k in keys],
            'v': [_encode(obj[k], arrays, f'{path}[{k!r}]') for k in keys],
        }
    if isinstance(obj, (list, tuple)):
        return {
            't': 'tuple' if isinstance(obj, tuple) else 'list',
            'v': [_encode(x, arrays, f'{path}[{i}]') for i, x in enumerate(obj)],
        }
    raise TypeError(
        f'{path}: cannot encode {type(obj).__name__} for the parquet result bridge. '
        'Metric results must be built from dict / list / tuple / numpy / torch / '
        'python scalars / str / None / BaseDataElement.'
    )


def _array(node: dict[str, Any], arrays: Sequence[bytes]) -> np.ndarray:
    dtype = np.dtype(node['dtype'])
    arr = np.frombuffer(arrays[node['i']], dtype=dtype)
    return arr.reshape(node.get('shape', [])).copy()  # copy: writable, owns its memory


def _decode(node: dict[str, Any], arrays: Sequence[bytes]) -> object:
    t = node['t']
    if t == 'none':
        return None
    if t in ('bool', 'int', 'float', 'str'):
        return node['v']
    if t == 'bytes':
        return bytes(arrays[node['i']])
    if t == 'nd':
        return _array(node, arrays)
    if t == 'ndobj':
        out = np.empty(len(node['v']), dtype=object)
        for i, x in enumerate(node['v']):
            out[i] = _decode(x, arrays)
        return out.reshape(node['shape'])
    if t == 'npscalar':
        return _array(node, arrays)[()]
    if t == 'tensor':
        dtype = _TORCH_DTYPES[node['dtype']]
        if dtype == torch.bfloat16:
            raw = np.frombuffer(arrays[node['i']], dtype=np.int16).copy()
            return torch.from_numpy(raw).view(torch.bfloat16).reshape(node['shape'])
        np_dtype = torch.empty((), dtype=dtype).numpy().dtype
        raw = np.frombuffer(arrays[node['i']], dtype=np_dtype).copy()
        return torch.from_numpy(raw).reshape(node['shape'])
    if t == 'element':
        module_name, qualname = node['cls'].split(':')
        cls: Any = importlib.import_module(module_name)
        for part in qualname.split('.'):
            cls = getattr(cls, part)
        if not (isinstance(cls, type) and issubclass(cls, BaseDataElement)):
            raise TypeError(f'{node["cls"]} is not a BaseDataElement')
        return cls(metainfo=_decode(node['meta'], arrays), **_decode(node['data'], arrays))
    if t == 'dict':
        keys = [_decode(k, arrays) for k in node['k']]
        return {k: _decode(v, arrays) for k, v in zip(keys, node['v'], strict=True)}
    if t == 'list':
        return [_decode(x, arrays) for x in node['v']]
    if t == 'tuple':
        return tuple(_decode(x, arrays) for x in node['v'])
    raise ValueError(f'unknown spec node type {t!r}')


# --------------------------------------------------------------------------- tables


def results_to_table(
    per_metric: Sequence[tuple[Sequence[Any], Sequence[int]]], rank: int
) -> pa.Table:
    """One row per result: ``per_metric[m] = (results, dataset_indices)``."""
    cols: dict[str, list[Any]] = {name: [] for name in SCHEMA.names}
    for m, (results, indices) in enumerate(per_metric):
        if len(results) != len(indices):
            raise ValueError(f'metric {m}: {len(results)} results but {len(indices)} indices')
        for seq, (res, idx) in enumerate(zip(results, indices, strict=True)):
            spec, arrays = encode(res)
            cols['metric'].append(m)
            cols['index'].append(int(idx))
            cols['rank'].append(rank)
            cols['seq'].append(seq)
            cols['spec'].append(spec)
            cols['arrays'].append(arrays)
    return pa.Table.from_pydict(cols, schema=SCHEMA)


def table_to_results(table: pa.Table, num_metrics: int) -> list[list[Any]]:
    """Merge parts into dataset order, one result list per metric.

    The distributed sampler pads the last batches with samples another rank
    also processes; for each dataset index only the rows of the lowest rank
    that holds it are kept (all of them, in order, so a metric may append more
    than one result per sample).
    """
    metric = table.column('metric').to_numpy()
    index = table.column('index').to_numpy()
    rank = table.column('rank').to_numpy()
    seq = table.column('seq').to_numpy()
    specs = table.column('spec').to_pylist()
    arrays = table.column('arrays').to_pylist()

    out: list[list[Any]] = []
    for m in range(num_metrics):
        rows = np.flatnonzero(metric == m)
        owner: dict[int, int] = {}
        for r in rows:
            i = int(index[r])
            owner[i] = min(owner.get(i, int(rank[r])), int(rank[r]))
        keep = [r for r in rows if owner[int(index[r])] == int(rank[r])]
        keep.sort(key=lambda r: (int(index[r]), int(seq[r])))
        out.append([decode(specs[r], arrays[r]) for r in keep])
    return out


def write_table(table: pa.Table, path: str) -> None:
    """Write atomically: readers never see a half-written file."""
    tmp = f'{path}.tmp.{os.getpid()}'
    pq.write_table(table, tmp)
    os.replace(tmp, path)


def touch(path: str, text: str = '') -> None:
    tmp = f'{path}.tmp.{os.getpid()}'
    with open(tmp, 'w') as f:
        f.write(text)
    os.replace(tmp, path)


def wait_for(
    paths: Callable[[], list[str]],
    timeout_s: float,
    what: str,
    error_file: str | None = None,
    poll_s: float = 0.5,
) -> None:
    """Poll until every path in ``paths()`` exists (shared-filesystem rendezvous)."""
    deadline = time.monotonic() + timeout_s
    while True:
        missing = [p for p in paths() if not osp.exists(p)]
        if not missing:
            return
        if error_file is not None and osp.exists(error_file):
            with open(error_file) as f:
                raise RuntimeError(f'rank 0 failed while {what}:\n{f.read()}')
        if time.monotonic() > deadline:
            raise TimeoutError(
                f'timed out after {timeout_s:.0f}s while {what}; still missing: {missing[:4]}'
                f'{" ..." if len(missing) > 4 else ""}. The bridge directory must be on a '
                'filesystem every rank can see (set accelerator.eval.dir on multi-node runs).'
            )
        time.sleep(poll_s)


class ParquetResultBridge:
    """File rendezvous for one evaluation round.

    Args:
        root: Directory for this round, unique per round and identical on every
            rank (e.g. ``<work_dir>/accelerate_eval/val_<timestamp>_<step>``).
        rank: This process's rank.
        world_size: Number of processes.
        timeout_s: How long to wait for the other ranks / the metric.
        keep_files: Keep the parts after the metric is computed (debugging).
    """

    def __init__(
        self,
        root: str,
        rank: int,
        world_size: int,
        timeout_s: float = 4 * 3600.0,
        keep_files: bool = False,
    ) -> None:
        self.root = root
        self.rank = rank
        self.world_size = world_size
        self.timeout_s = timeout_s
        self.keep_files = keep_files

    def _part(self, r: int) -> str:
        return osp.join(self.root, f'part_{r:05d}.parquet')

    @property
    def _metrics(self) -> str:
        return osp.join(self.root, 'metrics.parquet')

    @property
    def _error(self) -> str:
        return osp.join(self.root, 'error.txt')

    def run(
        self,
        per_metric: Sequence[tuple[Sequence[Any], Sequence[int]]],
        compute: Callable[[list[list[Any]]], dict[str, Any]],
    ) -> dict[str, Any]:
        """Exchange results, run ``compute`` on rank 0 and return its metrics on all ranks."""
        os.makedirs(self.root, exist_ok=True)
        write_table(results_to_table(per_metric, self.rank), self._part(self.rank))
        touch(self._part(self.rank) + '.done')

        if self.rank == 0:
            try:
                wait_for(
                    lambda: [self._part(r) + '.done' for r in range(self.world_size)],
                    self.timeout_s,
                    'waiting for the result parts of all ranks',
                )
                table = pa.concat_tables(
                    [pq.read_table(self._part(r)) for r in range(self.world_size)]
                )
                metrics = compute(table_to_results(table, len(per_metric)))
                spec, arrays = encode(metrics)
                write_table(
                    pa.Table.from_pydict(
                        {'spec': [spec], 'arrays': [arrays]},
                        schema=pa.schema(
                            [('spec', pa.large_string()), ('arrays', pa.list_(pa.large_binary()))]
                        ),
                    ),
                    self._metrics,
                )
            except BaseException:
                # Unblock the other ranks with the reason instead of a timeout.
                touch(self._error, traceback.format_exc())
                raise
            if not self.keep_files:
                for r in range(self.world_size):
                    for p in (self._part(r), self._part(r) + '.done'):
                        if osp.exists(p):
                            os.remove(p)
            touch(self._metrics + '.done')
            return metrics

        wait_for(
            lambda: [self._metrics + '.done'],
            self.timeout_s,
            'waiting for rank 0 to compute the metrics',
            error_file=self._error,
        )
        row = pq.read_table(self._metrics).to_pylist()[0]
        metrics = decode(row['spec'], row['arrays'])
        touch(osp.join(self.root, f'ack_{self.rank:05d}'))
        return metrics

    def cleanup(self, wait_s: float = 300.0) -> None:
        """Rank 0: remove the round's directory once every rank has read the metrics."""
        if self.rank != 0 or self.keep_files:
            return
        try:
            wait_for(
                lambda: [osp.join(self.root, f'ack_{r:05d}') for r in range(1, self.world_size)],
                wait_s,
                'waiting for the metric acks',
            )
        except TimeoutError:
            return  # leave it; the next round's sweep removes it
        shutil.rmtree(self.root, ignore_errors=True)


def sweep_old_rounds(parent: str, keep: str) -> None:
    """Rank 0: remove leftover round directories of earlier evaluations."""
    if not osp.isdir(parent):
        return
    for name in os.listdir(parent):
        path = osp.join(parent, name)
        if path != keep and osp.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
