# Copyright (c) OpenMMLab. All rights reserved.
"""Similarity profiles: a :class:`FeatureSpec`, its fitted :class:`Projector` and provenance,
in one ``.npz`` written by ``tools/similarity/suggest_api_hyperparameters.py``."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .features import FeatureSpec, Projector

PROFILE_VERSION = 1


def save_profile(
    path: str | Path, spec: FeatureSpec, projector: Projector | None, meta: dict | None = None
) -> Path:
    """Write ``spec`` + ``projector`` + ``meta`` (input size, pad mode, metrics, ...)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        'version': np.array(PROFILE_VERSION),
        'spec_json': np.array(json.dumps(spec.to_dict())),
        'meta_json': np.array(json.dumps(meta or {}, default=str)),
    }
    if projector is not None:
        arrays['mean'] = projector.mean.astype(np.float32)
        arrays['whiten'] = np.array(bool(projector.whiten))
        if projector.components is not None:
            arrays['components'] = projector.components.astype(np.float32)
        if projector.explained_variance is not None:
            arrays['explained_variance'] = projector.explained_variance.astype(np.float32)
    with open(path, 'wb') as f:
        np.savez(f, **arrays)
    return path


def load_profile(path: str | Path) -> tuple[FeatureSpec, Projector | None, dict]:
    """``(spec, projector or None, meta)`` from a profile written by :func:`save_profile`."""
    with np.load(path, allow_pickle=False) as z:
        version = int(z['version'])
        if version > PROFILE_VERSION:
            raise ValueError(f'{path}: profile version {version} is newer than this service')
        spec = FeatureSpec.from_dict(json.loads(str(z['spec_json'])))
        meta = json.loads(str(z['meta_json']))
        projector = None
        if 'mean' in z:
            projector = Projector(
                mean=z['mean'].astype(np.float32),
                components=z['components'].astype(np.float32) if 'components' in z else None,
                explained_variance=(
                    z['explained_variance'].astype(np.float32)
                    if 'explained_variance' in z
                    else None
                ),
                whiten=bool(z['whiten']) if 'whiten' in z else False,
            )
    return spec, projector, meta
