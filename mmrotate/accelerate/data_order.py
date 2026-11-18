# Copyright (c) OpenMMLab. All rights reserved.
"""Make sure every rank sees the dataset in the same order.

The distributed sampler hands rank ``r`` the indices ``r, r + W, ...`` of one
shared permutation, which is only correct if index ``i`` is the same sample on
every rank. Datasets that list files with an unsorted ``glob`` (e.g.
``DOTADataset``) can disagree between processes / nodes, so some samples are
then seen twice per epoch and others never, and evaluation mixes up images.

:class:`ConsistentOrderDataset` wraps a dataset, compares a fingerprint of its
sample list across ranks and, if the ranks disagree, re-orders the samples by a
stable key (``img_path``, ...) on every rank. If the ranks still disagree they
hold different samples and a clear error is raised.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Sized

import torch
import torch.distributed as dist
from mmengine.logging import print_log
from torch.utils.data import Dataset

from mmrotate.registry import DATASETS

SORT_KEYS = ('img_path', 'file_name', 'filename', 'img_id', 'image_id', 'ann_path')


def sample_key(info: object) -> str:
    """A stable, rank-independent key for one sample's data info."""
    if isinstance(info, dict):
        for k in SORT_KEYS:
            if k in info and info[k] is not None:
                return f'{k}={info[k]}'
    try:
        return json.dumps(info, sort_keys=True, default=repr)
    except (TypeError, ValueError):
        return repr(info)


def _all_keys(dataset: Sized) -> list[str] | None:
    get = getattr(dataset, 'get_data_info', None)
    if get is None:
        return None
    return [sample_key(get(i)) for i in range(len(dataset))]


def _fingerprint(keys: list[str] | None, length: int) -> int:
    h = hashlib.blake2b(digest_size=8)
    h.update(str(length).encode())
    for k in keys or ():
        h.update(k.encode())
        h.update(b'\0')
    return int.from_bytes(h.digest(), 'little', signed=True)


def _gather_int(value: int) -> list[int]:
    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
        return [value]
    device = torch.device('cpu')
    if dist.get_backend() == 'nccl':
        device = torch.device('cuda', torch.cuda.current_device())
    t = torch.tensor([value], dtype=torch.int64, device=device)
    out = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(out, t)
    return [int(x.item()) for x in out]


@DATASETS.register_module()
class ConsistentOrderDataset:
    """Wrap ``dataset`` so that sample ``i`` is the same on every rank.

    Args:
        dataset: The dataset or its config.
        sort: ``'auto'`` re-orders only when the ranks disagree (single-process
            runs and consistent datasets keep their order), ``'always'`` always
            sorts by the sample key, ``'never'`` only checks and raises.
        gather_fn: Override for the cross-rank gather (tests).
    """

    def __init__(
        self,
        dataset: Dataset | dict,
        sort: str = 'auto',
        gather_fn: Callable[[int], list[int]] | None = None,
    ) -> None:
        if sort not in ('auto', 'always', 'never'):
            raise ValueError(f'sort must be auto / always / never, got {sort!r}')
        if isinstance(dataset, dict):
            dataset = DATASETS.build(dataset)
        if hasattr(dataset, 'full_init'):
            dataset.full_init()
        self.dataset = dataset
        gather = gather_fn or _gather_int

        keys = _all_keys(dataset)
        order = list(range(len(dataset)))
        if keys is None:
            print_log(
                f'{type(dataset).__name__} has no get_data_info; checking only its length '
                'across ranks',
                logger='current',
                level=logging.WARNING,
            )
        fps = gather(_fingerprint(keys, len(dataset)))
        if keys is not None and (sort == 'always' or (sort == 'auto' and len(set(fps)) > 1)):
            if len(set(fps)) > 1:
                print_log(
                    f'{type(dataset).__name__}: the ranks list the samples in different orders; '
                    'sorting them by their file path so that every rank agrees',
                    logger='current',
                    level=logging.WARNING,
                )
            order = sorted(order, key=lambda i: keys[i])
            fps = gather(_fingerprint([keys[i] for i in order], len(dataset)))
        if len(set(fps)) > 1:
            raise RuntimeError(
                f'{type(dataset).__name__}: the ranks hold different samples (fingerprints '
                f'{fps}). Every rank must load the same annotation files; check that the data '
                'root is the same shared path on every node and that dataset `indices` / '
                'filters do not depend on file listing order.'
            )
        self._order = order
        self._identity = order == list(range(len(dataset)))

    def _map(self, idx: int) -> int:
        return idx if self._identity else self._order[idx]

    @property
    def metainfo(self) -> dict:
        return getattr(self.dataset, 'metainfo', {})

    def full_init(self) -> None:
        """Already initialised in ``__init__``."""

    def __len__(self) -> int:
        return len(self._order)

    def __getitem__(self, idx: int) -> object:
        return self.dataset[self._map(idx)]

    def get_data_info(self, idx: int) -> dict:
        return self.dataset.get_data_info(self._map(idx))

    def get_cat_ids(self, idx: int) -> list[int]:
        # mmdet's ClassAwareSampler / class-balanced wrappers index by sample.
        return self.dataset.get_cat_ids(self._map(idx))

    def __getattr__(self, name: str) -> object:
        # Only reached for attributes the wrapper lacks (pipeline, test_mode, ...).
        if name in ('dataset', '_order', '_identity'):
            raise AttributeError(name)
        return getattr(self.dataset, name)

    def __repr__(self) -> str:
        return f'ConsistentOrderDataset(identity={self._identity}, dataset={self.dataset!r})'


def unwrap_dataset(dataset: Dataset) -> Dataset:
    """The dataset inside a :class:`ConsistentOrderDataset` (or ``dataset`` itself)."""
    return dataset.dataset if isinstance(dataset, ConsistentOrderDataset) else dataset


__all__ = ['ConsistentOrderDataset', 'sample_key', 'unwrap_dataset']
