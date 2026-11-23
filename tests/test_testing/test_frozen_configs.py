# Copyright (c) OpenMMLab. All rights reserved.
"""The frozen test configs load on their own and match ``configs/`` when present."""

from pathlib import Path

import pytest

from mmrotate.testing.frozen_configs import (
    CONFIG_ROOT,
    FORK_CONFIG_DIRS,
    FULL,
    SNAPSHOTS,
    frozen_config,
    real_config,
    snapshot,
)


@pytest.mark.parametrize('rel', SNAPSHOTS)
def test_snapshot_loads_standalone(rel: str) -> None:
    cfg = frozen_config(rel)
    assert '_base_' not in cfg
    assert cfg.model.type


@pytest.mark.parametrize('rel', [r for r in SNAPSHOTS if Path(r).parts[0] in FORK_CONFIG_DIRS])
def test_snapshot_up_to_date(rel: str) -> None:
    real = real_config(rel)
    assert frozen_config(rel).to_dict() == snapshot(real, rel in FULL), (
        f'tests/frozen_configs snapshot of configs/{rel} is stale; '
        'run `python -m mmrotate.testing.frozen_configs`'
    )


def test_every_fork_config_folder_snapshot_complete() -> None:
    """New configs in the snapshotted folders that tests glob must get a snapshot too."""
    for sub in ('soda_dinov3', 'comer_lite'):
        if not (CONFIG_ROOT / sub).is_dir():
            pytest.skip(f'configs/{sub} not present')
        real = {f'{sub}/{p.name}' for p in (CONFIG_ROOT / sub).glob('*.yaml')}
        assert real <= set(SNAPSHOTS), sorted(real - set(SNAPSHOTS))
