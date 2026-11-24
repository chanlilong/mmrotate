# Copyright (c) OpenMMLab. All rights reserved.
"""Frozen config snapshots so the fork's tests run without ``./configs``.

Tests that only need a config to build a model (or to feed a function) load a
fully resolved, ``_base_``-free snapshot from ``tests/frozen_configs/`` through
:func:`frozen_config` / :func:`frozen_detector_cfg` / :func:`frozen_config_path`.
Tests that check the config files themselves use :func:`real_config`, which
skips when ``configs/<rel>`` is missing (e.g. after a merge into a fork that
does not ship these configs).

Regenerate the snapshots after changing a config::

    python -m mmrotate.testing.frozen_configs          # rewrite every snapshot
    python -m mmrotate.testing.frozen_configs --check  # fail if one is stale
"""

from __future__ import annotations

import copy
from pathlib import Path

from mmengine.config import Config, ConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = REPO_ROOT / 'configs'
FROZEN_ROOT = REPO_ROOT / 'tests' / 'frozen_configs'

# Keys kept in a trimmed snapshot (the model tests only need these).
TRIMMED_KEYS = ('default_scope', 'model', 'optim_wrapper')


def _folder(sub: str, *names: str) -> tuple[str, ...]:
    return tuple(f'{sub}/{name}' for name in names)


SNAPSHOTS: tuple[str, ...] = (
    *_folder(
        'cradio',
        'cradiov4-h-comer-lora_rtmdet-p2-3x-dota_ms.yaml',
        'cradiov4-h-comer-lora_rtmdet-proxy.yaml',
        'cradiov4-h-comer_rtmdet-3x-dota_ms.yaml',
        'cradiov4-h-comer_rtmdet-dcfl-3x-dota_ms.yaml',
        'cradiov4-h-comer_rtmdet-p2-3x-dota_ms.yaml',
        'cradiov4-h-comer_rtmdet-p2-nwd-dsl-rka-3x-dota_ms.yaml',
        'cradiov4-h-comer_rtmdet-p2-nwd-dsl-rka-lowmem-3x-dota_ms.yaml',
        'cradiov4-h-comer_rtmdet-p2-nwd-dsl-rka-lowmem-proxy.yaml',
        'cradiov4-h-comer_rtmdet-proxy.yaml',
        'cradiov4-h_rtmdet-3x-dota_ms.yaml',
        'cradiov4-h-vitdet_rtmdet-3x-dota_ms.yaml',
        'cradiov4-h-vitdet_rtmdet-p2-3x-dota_ms.yaml',
        'cradiov4-h-vitdet_rtmdet-proxy.yaml',
        'cradiov4-so400m-comer-lora_rtmdet-p2-3x-dota_ms.yaml',
        'cradiov4-so400m-comer-lora_rtmdet-proxy.yaml',
        'cradiov4-so400m-comer_rtmdet-3x-dota_ms.yaml',
        'cradiov4-so400m-comer_rtmdet-p2-3x-dota_ms.yaml',
        'cradiov4-so400m-comer_rtmdet-proxy.yaml',
    ),
    *_folder(
        'dcfl',
        'rotated-retinanet-dcfl-le135_r50_fpn_3x_dota.yaml',
        'rotated_rtmdet_l-dcfl-3x-dota_ms.yaml',
        'rotated_rtmdet_tiny-dcfl-proxy.yaml',
    ),
    *_folder(
        'soda_dinov3',
        'convnext-t-dinov3_rtmdet-p2-6e-soda.yaml',
        'convnext-t-dinov3_rtmdet-p2-dcfl-6e-soda.yaml',
        'convnext-t-dinov3_rtmdet-p2-dcflhead-nwd-dsl-rka-6e-soda.yaml',
        'vitl-lingbot-comer-lora_rtmdet-p2-6e-soda.yaml',
        'vitl-lingbot-comer_rtmdet-p2-6e-soda.yaml',
        'vits-dinov3-comer-lora_rtmdet-p2-6e-soda.yaml',
        'vits-dinov3-comer-noinj_rtmdet-p2-6e-soda.yaml',
        'vits-dinov3-comer_rtmdet-p2-6e-soda.yaml',
        'vits-dinov3-comer_rtmdet-p2-dcfl-6e-soda.yaml',
        'vits-dinov3-comer_rtmdet-p2-nwd-dsl-rka-6e-soda.yaml',
        'vits-dinov3-comer_rtmdet-p2-nwd-rka-6e-soda.yaml',
        'vits-dinov3-vitdet-lora_rtmdet-p2-6e-soda.yaml',
        'vits-dinov3-vitdet_rtmdet-p2-6e-soda.yaml',
        'vits-lingbot-comer-lora_rtmdet-p2-6e-soda.yaml',
        'vits-lingbot-comer_rtmdet-p2-6e-soda.yaml',
    ),
    *_folder(
        'comer_lite',
        'vits-dinov3-comer-cfinter-late_rtmdet-p2-6e-soda.yaml',
        'vits-dinov3-comer-extra0_rtmdet-p2-6e-soda.yaml',
        'vits-dinov3-comer-lite_rtmdet-p2-6e-soda.yaml',
        'vits-dinov3-comer-ratio05_rtmdet-p2-6e-soda.yaml',
        'vits-dinov3-comer-slim_rtmdet-p2-6e-soda.yaml',
    ),
    *_folder(
        'nwd_rka',
        'oriented-rcnn-le90_r50_fpn_nwd-rka-1x_dota.py',
        'rotated-retinanet-rbox-le90_r50_fpn_nwd-rka-1x_dota.py',
        'rotated-rtmdet-l_nwd-dsl-rka-3x-dota.py',
        'rotated-rtmdet-l_nwd-rka-3x-dota.py',
        'rotated-rtmdet-tiny_nwd-rka-proxy.py',
    ),
    'rotated_rtmdet/rotated_rtmdet_tiny-3x-dota.py',
)

# Snapshots kept whole: their pipelines / dataloaders / hooks are used too.
FULL: frozenset[str] = frozenset(
    {
        'rotated_rtmdet/rotated_rtmdet_tiny-3x-dota.py',
        'cradio/cradiov4-h-comer_rtmdet-3x-dota_ms.yaml',
        'cradio/cradiov4-h-comer_rtmdet-proxy.yaml',
        'cradio/cradiov4-h-comer_rtmdet-p2-nwd-dsl-rka-lowmem-proxy.yaml',
        'soda_dinov3/convnext-t-dinov3_rtmdet-p2-6e-soda.yaml',
    }
)

# Config folders added by this fork. Upstream configs (e.g. rotated_rtmdet) may
# legitimately differ in a fork the tests are merged into, so the drift check
# only covers these.
FORK_CONFIG_DIRS: frozenset[str] = frozenset(
    {'cradio', 'dcfl', 'soda_dinov3', 'comer_lite', 'nwd_rka'}
)


def frozen_config_path(rel: str) -> str:
    """Path of the snapshot of ``configs/<rel>`` (always stored as ``.yaml``)."""
    path = (FROZEN_ROOT / rel).with_suffix('.yaml')
    if not path.is_file():
        raise FileNotFoundError(
            f'No frozen snapshot of configs/{rel} at {path}. Add it to SNAPSHOTS and run '
            '`python -m mmrotate.testing.frozen_configs`.'
        )
    return str(path)


def frozen_config(rel: str) -> Config:
    """Snapshot of ``configs/<rel>``; never reads ``./configs``."""
    return Config.fromfile(frozen_config_path(rel))


def frozen_detector_cfg(rel: str) -> ConfigDict:
    """``model`` of the snapshot; drop-in for :func:`mmrotate.testing.get_detector_cfg`."""
    return copy.deepcopy(frozen_config(rel).model)


def frozen_names(sub: str) -> list[str]:
    """Sorted original file names of the snapshots of ``configs/<sub>/``."""
    return sorted(Path(rel).name for rel in SNAPSHOTS if rel.startswith(f'{sub}/'))


def real_config_dir(sub: str) -> Path:
    """``configs/<sub>``, or skip the calling test when it is missing."""
    path = CONFIG_ROOT / sub
    if not path.is_dir():
        import pytest

        pytest.skip(f'configs/{sub} not present')
    return path


def real_config(rel: str) -> Config:
    """``configs/<rel>`` itself, or skip the calling test when it is missing."""
    path = CONFIG_ROOT / rel
    if not path.is_file():
        import pytest

        pytest.skip(f'configs/{rel} not present')
    return Config.fromfile(path)


def snapshot(cfg: Config, full: bool) -> dict:
    """The resolved config as stored in a snapshot (trimmed unless ``full``)."""
    data = cfg.to_dict()
    if full:
        return data
    return {k: data[k] for k in TRIMMED_KEYS if k in data}
