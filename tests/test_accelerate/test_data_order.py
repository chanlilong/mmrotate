# Copyright (c) OpenMMLab. All rights reserved.
"""ConsistentOrderDataset: rank disagreement is fixed or reported."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from mmrotate.accelerate.data_order import ConsistentOrderDataset, _fingerprint, sample_key


class ListDataset:
    metainfo = {'classes': ('a',)}

    def __init__(self, paths: list[str]) -> None:
        self.paths = paths
        self.pipeline = 'pipe'

    def __len__(self) -> int:
        return len(self.paths)

    def get_data_info(self, idx: int) -> dict:
        return {'img_path': self.paths[idx], 'idx': idx}

    def __getitem__(self, idx: int) -> str:
        return self.paths[idx]

    def get_cat_ids(self, idx: int) -> list[int]:
        return [ord(self.paths[idx])]


def other_rank(paths: list[str]) -> Callable[[int], list[int]]:
    """Fake gather: this rank's fingerprint plus one from a rank listing ``paths``."""
    other = _fingerprint([sample_key({'img_path': p}) for p in paths], len(paths))

    def gather(value: int) -> list[int]:
        return [value, other]

    return gather


def test_consistent_order_is_kept() -> None:
    paths = ['c', 'a', 'b']
    ds = ConsistentOrderDataset(ListDataset(paths), gather_fn=other_rank(paths))
    assert [ds[i] for i in range(3)] == paths
    assert ds.metainfo == {'classes': ('a',)} and ds.pipeline == 'pipe'


def test_disagreeing_order_is_sorted() -> None:
    ds = ConsistentOrderDataset(ListDataset(['c', 'a', 'b']), gather_fn=other_rank(['a', 'b', 'c']))
    assert [ds[i] for i in range(3)] == ['a', 'b', 'c']
    assert ds.get_data_info(0) == {'img_path': 'a', 'idx': 1}
    assert ds.get_cat_ids(0) == [ord('a')]


def test_different_files_raise() -> None:
    with pytest.raises(RuntimeError, match='different samples'):
        ConsistentOrderDataset(ListDataset(['a', 'b']), gather_fn=other_rank(['a', 'x']))


def test_never_only_checks() -> None:
    with pytest.raises(RuntimeError, match='different samples'):
        ConsistentOrderDataset(
            ListDataset(['b', 'a']), sort='never', gather_fn=other_rank(['a', 'b'])
        )


def test_always_sorts() -> None:
    ds = ConsistentOrderDataset(ListDataset(['b', 'a']), sort='always', gather_fn=lambda v: [v])
    assert [ds[i] for i in range(2)] == ['a', 'b']
