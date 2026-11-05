# Copyright (c) OpenMMLab. All rights reserved.
"""Validation / test loops that gather metric results through parquet files.

The loops are mmengine's :class:`ValLoop` / :class:`TestLoop` unchanged (same
hooks, val loss, fp16 flag); only the evaluator is swapped for
:class:`BridgedEvaluator`, which tags every metric result with its dataset
index and gathers through :class:`ParquetResultBridge` instead of
``collect_results``. See ``result_bridge.py`` for why.
"""

from __future__ import annotations

import contextlib
import logging
import os.path as osp
from collections.abc import Iterator, Sequence
from typing import Any

from mmengine.dist import get_dist_info
from mmengine.evaluator import Evaluator
from mmengine.logging import print_log
from mmengine.runner import BaseLoop, Runner, TestLoop, ValLoop
from mmengine.structures import BaseDataElement
from torch.utils.data import DataLoader

from mmrotate.registry import LOOPS

from .result_bridge import (
    ParquetResultBridge,
    results_to_table,
    sweep_old_rounds,
    table_to_results,
)


def rank_indices(dataloader: DataLoader) -> list[int]:
    """Dataset indices this rank's ``dataloader`` yields, in iteration order.

    The samplers mmengine / mmdet use for evaluation (``DefaultSampler``,
    ``InfiniteSampler`` with shuffle off, ...) are deterministic for a given
    seed and epoch, so iterating them here gives the same order as the loop.
    """
    if dataloader.batch_sampler is not None:
        return [int(i) for batch in dataloader.batch_sampler for i in batch]
    if dataloader.sampler is not None:
        return [int(i) for i in dataloader.sampler]
    return list(range(len(dataloader.dataset)))  # type: ignore[arg-type]


class BridgedEvaluator(Evaluator):
    """:class:`Evaluator` whose ``evaluate`` gathers through parquet files.

    Args:
        metrics: The metrics of the evaluator being replaced.
        bridge_root: Parent directory of the per-round bridge directories.
        stage: ``'val'`` or ``'test'`` (part of the round directory name).
        timeout_s: Rendezvous timeout.
        keep_files: Keep the parquet parts (debugging).
    """

    def __init__(
        self,
        metrics: Sequence[Any],
        bridge_root: str,
        stage: str,
        timeout_s: float,
        keep_files: bool = False,
    ) -> None:
        super().__init__(list(metrics))
        self.bridge_root = bridge_root
        self.stage = stage
        self.timeout_s = timeout_s
        self.keep_files = keep_files
        self._indices: list[int] = []
        self._cursor = 0
        self._result_index: list[list[int]] = [[] for _ in self.metrics]
        self._round_tag = ''

    def begin(self, indices: list[int], round_tag: str) -> None:
        """Start a round: ``indices`` are the dataset indices this rank will process."""
        self._indices = indices
        self._cursor = 0
        self._result_index = [[] for _ in self.metrics]
        self._round_tag = round_tag
        for metric in self.metrics:
            metric.results.clear()

    def process(
        self,
        data_samples: Sequence[BaseDataElement],
        data_batch: dict | tuple | list | None = None,
    ) -> None:
        samples = [s.to_dict() if isinstance(s, BaseDataElement) else s for s in data_samples]
        n = len(samples)
        idx = self._indices[self._cursor : self._cursor + n]
        if len(idx) != n:
            raise RuntimeError(
                f'{self.stage}: the loop produced more samples ({self._cursor + n}) than the '
                f'sampler yields on this rank ({len(self._indices)}).'
            )
        self._cursor += n
        for m, metric in enumerate(self.metrics):
            before = len(metric.results)
            metric.process(data_batch, samples)
            added = len(metric.results) - before
            if added == n:
                self._result_index[m].extend(idx)
            elif added and n == 1:
                self._result_index[m].extend(idx * added)
            elif added:
                raise RuntimeError(
                    f'{type(metric).__name__}.process appended {added} results for a batch of '
                    f'{n} samples, so results cannot be mapped to dataset indices. Use '
                    'batch_size=1 for this metric or accelerator.eval.bridge=mmengine.'
                )

    def evaluate(self, size: int) -> dict:
        if self._cursor != len(self._indices):
            print_log(
                f'{self.stage}: processed {self._cursor} samples, the sampler yields '
                f'{len(self._indices)}',
                logger='current',
                level=logging.WARNING,
            )
        rank, world_size = get_dist_info()
        per_metric = [
            (list(metric.results), self._result_index[m]) for m, metric in enumerate(self.metrics)
        ]

        def compute(merged: list[list[Any]]) -> dict[str, Any]:
            out: dict[str, Any] = {}
            for metric, results in zip(self.metrics, merged, strict=True):
                if len(results) < size:
                    print_log(
                        f'{type(metric).__name__}: {len(results)} results for a dataset of '
                        f'{size} samples (drop_last, or samples without results)',
                        logger='current',
                        level=logging.WARNING,
                    )
                values = metric.compute_metrics(results)
                if metric.prefix:
                    values = {'/'.join((metric.prefix, k)): v for k, v in values.items()}
                for k in values:
                    if k in out:
                        raise ValueError(
                            f'There are multiple evaluation results with the same metric name {k}. '
                            'Please make sure all metrics have different prefixes.'
                        )
                out.update(values)
            return out

        try:
            if world_size == 1:
                # Same merge (dedup + dataset order) without the files.
                merged = table_to_results(results_to_table(per_metric, 0), len(self.metrics))
                return compute(merged)
            parent = osp.join(self.bridge_root, 'accelerate_eval')
            root = osp.join(parent, f'{self.stage}_{self._round_tag}')
            if rank == 0:
                sweep_old_rounds(parent, keep=root)
            bridge = ParquetResultBridge(root, rank, world_size, self.timeout_s, self.keep_files)
            metrics = bridge.run(per_metric, compute)
            bridge.cleanup()
            return metrics
        finally:
            for metric in self.metrics:
                metric.results.clear()
            self._result_index = [[] for _ in self.metrics]


class _BridgedLoopMixin:
    """Swaps the loop's evaluator for a :class:`BridgedEvaluator`."""

    stage: str = 'val'
    runner: Any
    dataloader: DataLoader
    evaluator: Evaluator
    _round = 0

    def _install_bridge(self) -> None:
        cfg = self.runner.cfg.get('accelerator', None) or {}
        eval_cfg = dict(cfg.get('eval', None) or {})
        if eval_cfg.get('bridge', 'parquet') != 'parquet':
            return  # 'mmengine': keep the stock collect_results path
        meta = self.evaluator.dataset_meta
        self.evaluator = BridgedEvaluator(
            self.evaluator.metrics,
            bridge_root=eval_cfg.get('dir') or self.runner.work_dir,
            stage=self.stage,
            timeout_s=float(eval_cfg.get('timeout_minutes', 240)) * 60.0,
            keep_files=bool(eval_cfg.get('keep_files', False)),
        )
        if meta is not None:
            self.evaluator.dataset_meta = meta

    def _begin_round(self) -> None:
        if isinstance(self.evaluator, BridgedEvaluator):
            self._round += 1
            # Not runner.epoch / runner.iter: those build the train loop (and its
            # dataloader) on first access, which breaks test-only runs.
            train_loop = getattr(self.runner, '_train_loop', None)
            epoch, it = (
                (train_loop.epoch, train_loop.iter) if isinstance(train_loop, BaseLoop) else (0, 0)
            )
            tag = f'{self.runner.timestamp}_e{epoch}_i{it}_r{self._round}'
            self.evaluator.begin(rank_indices(self.dataloader), tag)

    @contextlib.contextmanager
    def _eval_context(self) -> Iterator[None]:
        model = self.runner.model
        ctx = getattr(model, 'eval_context', None)
        with ctx() if ctx is not None else contextlib.nullcontext():
            yield


@LOOPS.register_module()
class AccelerateValLoop(_BridgedLoopMixin, ValLoop):
    """:class:`ValLoop` with the parquet result bridge (``accelerator.eval``)."""

    stage = 'val'

    def __init__(
        self,
        runner: Runner,
        dataloader: DataLoader | dict,
        evaluator: Evaluator | dict | list,
        fp16: bool = False,
    ) -> None:
        super().__init__(runner, dataloader, evaluator, fp16)
        self._install_bridge()

    def run(self) -> dict:
        self._begin_round()
        with self._eval_context():
            return super().run()


@LOOPS.register_module()
class AccelerateTestLoop(_BridgedLoopMixin, TestLoop):
    """:class:`TestLoop` with the parquet result bridge (``accelerator.eval``)."""

    stage = 'test'

    def __init__(
        self,
        runner: Runner,
        dataloader: DataLoader | dict,
        evaluator: Evaluator | dict | list,
        fp16: bool = False,
    ) -> None:
        super().__init__(runner, dataloader, evaluator, fp16)
        self._install_bridge()

    def run(self) -> dict:
        self._begin_round()
        with self._eval_context():
            return super().run()
