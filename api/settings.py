# Copyright (c) OpenMMLab. All rights reserved.
"""Service settings: command-line flags with ``MMROTATE_API_*`` environment fallbacks.

Every flag ``--foo-bar`` can also be given as ``MMROTATE_API_FOO_BAR`` (handy in docker compose).
A flag on the command line wins over the environment.
"""

from __future__ import annotations

import argparse
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

ENV_PREFIX = 'MMROTATE_API_'
SERVICES = ('detector', 'similarity')
NATIVE = 'none'  # --similarity-gsd none: no gsd_m, tiles at source pixels


def parse_gsd(value: object) -> float | Literal['none'] | None:
    """``--similarity-gsd``: empty -> follow ``--gsd``; ``none``/``native`` -> native px."""
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    if text.lower() in (NATIVE, 'native'):
        return NATIVE
    try:
        return float(text)
    except ValueError:
        raise ValueError(
            f'--similarity-gsd {text!r}: use metres per pixel, or none for native pixels'
        ) from None


def parse_size(value: str | int | Sequence[int] | None) -> tuple[int, int] | None:
    """``'800'`` / ``'1024x800'`` / ``800`` / ``(w, h)`` -> ``(w, h)``; ``None`` stays ``None``."""
    if value is None or value == '':
        return None
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, str):
        parts = value.lower().replace(',', 'x').split('x')
        nums = [int(p) for p in parts if p.strip()]
    else:
        nums = [int(v) for v in value]
    if len(nums) == 1:
        nums = nums * 2
    if len(nums) != 2 or min(nums) <= 0:
        raise ValueError(f'Bad size {value!r}: expected N or WxH with positive integers')
    return (nums[0], nums[1])


def _bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() in ('1', 'true', 'yes', 'on')


@dataclass
class SimilarityOverrides:
    """``--sim-*`` flags: override single fields of the similarity :class:`FeatureSpec`."""

    source: str | None = None
    levels: str | None = None
    fusion: str | None = None
    level_norm: str | None = None
    out_stride: int | None = None
    projection: str | None = None
    dtype: str | None = None

    def as_dict(self) -> dict[str, object]:
        out: dict[str, object] = {}
        for key, value in self.__dict__.items():
            if value is None:
                continue
            if key == 'levels':
                text = str(value).strip().lower()
                out[key] = None if text in ('all', '') else [int(v) for v in text.split(',')]
            else:
                out[key] = value
        return out


@dataclass
class Settings:
    """Everything the service needs; see ``python -m api --help`` for the meaning of each."""

    config: str
    weights: str
    gsd_m: float | None = None
    similarity_gsd_m: float | Literal['none'] | None = None
    services: tuple[str, ...] = SERVICES
    device: str = 'cuda:0'
    precision: str = 'fp32'
    model_name: str | None = None
    # image size handling
    input_size: tuple[int, int] | None = None
    resize_mode: str = 'none'
    pad_mode: str = 'square'
    detector_tile_size: int | None = None
    similarity_tile_size: int | None = None
    patch_overlap: int = 128
    max_batch: int = 8
    max_side_px: int | None = None
    # similarity
    similarity_profile: str | None = None
    sim: SimilarityOverrides = field(default_factory=SimilarityOverrides)
    model_version: str | None = None
    feature_space: str | None = None
    # runtime
    self_test: bool = True
    host: str = '0.0.0.0'
    port: int = 8200
    log_level: str = 'info'

    def __post_init__(self) -> None:
        unknown = set(self.services) - set(SERVICES)
        if unknown or not self.services:
            raise ValueError(f'--services must be a subset of {SERVICES}, got {self.services}')
        if self.resize_mode not in ('none', 'config'):
            raise ValueError(f'--resize-mode must be none or config, got {self.resize_mode!r}')
        if self.pad_mode not in ('square', 'divisor'):
            raise ValueError(f'--pad-mode must be square or divisor, got {self.pad_mode!r}')
        if self.precision not in ('fp32', 'fp16', 'bf16'):
            raise ValueError(f'--precision must be fp32, fp16 or bf16, got {self.precision!r}')
        if self.patch_overlap < 0:
            raise ValueError('--patch-overlap must be >= 0')
        if self.max_batch < 1:
            raise ValueError('--max-batch must be >= 1')
        if self.gsd_m is not None and not 0 < self.gsd_m < math.inf:
            raise ValueError('--gsd must be > 0')
        sim = self.similarity_gsd_m
        if sim is not None and sim != NATIVE:
            if isinstance(sim, str) or not 0 < sim < math.inf:
                raise ValueError(f'--similarity-gsd must be > 0 or {NATIVE!r}, got {sim!r}')

    def similarity_gsd(self) -> float | None:
        """The GSD the similarity ``/health`` reports as ``gsd_m`` (``None``: native pixels).

        The qgis_annotations server resamples every image to it before ``/embed``.
        """
        if self.similarity_gsd_m == NATIVE:
            return None
        if self.similarity_gsd_m is not None:
            return float(self.similarity_gsd_m)
        return self.gsd_m


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='python -m api',
        description='Serve an mmrotate single-stage detector over the qgis_annotations detector '
        '(/detect) and similarity (/embed) REST contracts.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    def env(name: str, default: object = None) -> object:
        return os.environ.get(ENV_PREFIX + name.upper(), default)

    g = p.add_argument_group('model')
    g.add_argument('--config', default=env('config'), help='mmrotate config (.py or .yaml)')
    g.add_argument('--weights', default=env('weights'), help='checkpoint (.pth)')
    g.add_argument(
        '--gsd',
        type=float,
        default=env('gsd'),
        help='metres per pixel the model was trained at; both services report it as gsd_m '
        'and the qgis server resamples the imagery to it',
    )
    g.add_argument('--model-name', default=env('model_name'), help='name shown to annotators')
    g.add_argument('--device', default=env('device', 'cuda:0'))
    g.add_argument(
        '--precision',
        default=env('precision', 'fp32'),
        choices=['fp32', 'fp16', 'bf16'],
        help='autocast dtype for the forward pass',
    )
    g.add_argument(
        '--services',
        default=env('services', ','.join(SERVICES)),
        help='comma list of services to enable: detector, similarity',
    )

    g = p.add_argument_group('image size')
    g.add_argument(
        '--input-size',
        default=env('input_size'),
        help='model input N or WxH (model px); default: from the test pipeline, else 1024',
    )
    g.add_argument(
        '--resize-mode',
        default=env('resize_mode', 'none'),
        choices=['none', 'config'],
        help="none: tiles at native pixels (GSD-correct); config: keep the config's Resize",
    )
    g.add_argument(
        '--pad-mode',
        default=env('pad_mode', 'square'),
        choices=['square', 'divisor'],
        help='square: pad every window to the input size; divisor: pad to pad_size_divisor only',
    )
    g.add_argument(
        '--detector-tile-size',
        type=int,
        default=env('detector_tile_size'),
        help='native_size advertised to the detector client; default: the input size. '
        'Larger tiles are run as overlapping windows of the input size',
    )
    g.add_argument(
        '--similarity-tile-size',
        type=int,
        default=env('similarity_tile_size'),
        help='native_size advertised to the similarity client (source px); default: '
        'input size / scale',
    )
    g.add_argument(
        '--patch-overlap',
        type=int,
        default=env('patch_overlap', 128),
        help='overlap (model px) between windows when a tile is larger than the input size',
    )
    g.add_argument('--max-batch', type=int, default=env('max_batch', 8))
    g.add_argument(
        '--max-side-px',
        type=int,
        default=env('max_side_px'),
        help='refuse tiles with a larger side; default: 2 x the tile size',
    )

    g = p.add_argument_group('similarity')
    g.add_argument(
        '--similarity-gsd',
        default=env('similarity_gsd'),
        help='GSD the similarity /health reports as gsd_m (the qgis server resamples tiles to '
        f'it). Default: --gsd; {NATIVE!r} = native pixels',
    )
    g.add_argument(
        '--similarity-profile',
        default=env('similarity_profile'),
        help='profile .npz written by tools/similarity/suggest_api_hyperparameters.py',
    )
    g.add_argument(
        '--sim-source',
        default=env('sim_source'),
        help='backbone | neck | head_cls | head_reg | head_cls@I | a+b',
    )
    g.add_argument('--sim-levels', default=env('sim_levels'), help="e.g. '0,1,2' or 'all'")
    g.add_argument('--sim-fusion', default=env('sim_fusion'), help='concat | mean | single')
    g.add_argument('--sim-level-norm', default=env('sim_level_norm'), help='l2 | none')
    g.add_argument('--sim-out-stride', type=int, default=env('sim_out_stride'))
    g.add_argument(
        '--sim-projection',
        default=env('sim_projection'),
        help='none | center | pca (center/pca need a profile)',
    )
    g.add_argument('--sim-dtype', default=env('sim_dtype'), help='int8 | float16 | float32')
    g.add_argument('--model-version', default=env('model_version'), help='override')
    g.add_argument('--feature-space', default=env('feature_space'), help='override')

    g = p.add_argument_group('server')
    g.add_argument('--host', default=env('host', '0.0.0.0'))
    g.add_argument('--port', type=int, default=env('port', 8200))
    g.add_argument('--log-level', default=env('log_level', 'info'))
    g.add_argument(
        '--no-self-test',
        action='store_true',
        default=not _bool(env('self_test', 'true')),
        help='skip the boot self-test',
    )
    return p


def settings_from_args(argv: Sequence[str] | None = None) -> Settings:
    parser = build_parser()
    a = parser.parse_args(argv)
    if not a.config or not a.weights:
        parser.error('--config and --weights are required (or MMROTATE_API_CONFIG / _WEIGHTS)')

    def opt_int(v: object) -> int | None:
        return None if v in (None, '') else int(v)  # type: ignore[arg-type]

    def opt_float(v: object) -> float | None:
        return None if v in (None, '') else float(v)  # type: ignore[arg-type]

    return Settings(
        config=a.config,
        weights=a.weights,
        gsd_m=opt_float(a.gsd),
        similarity_gsd_m=parse_gsd(a.similarity_gsd),
        services=tuple(s.strip() for s in str(a.services).split(',') if s.strip()),
        device=a.device,
        precision=a.precision,
        model_name=a.model_name,
        input_size=parse_size(a.input_size),
        resize_mode=a.resize_mode,
        pad_mode=a.pad_mode,
        detector_tile_size=opt_int(a.detector_tile_size),
        similarity_tile_size=opt_int(a.similarity_tile_size),
        patch_overlap=int(a.patch_overlap),
        max_batch=int(a.max_batch),
        max_side_px=opt_int(a.max_side_px),
        similarity_profile=a.similarity_profile or None,
        sim=SimilarityOverrides(
            source=a.sim_source,
            levels=a.sim_levels,
            fusion=a.sim_fusion,
            level_norm=a.sim_level_norm,
            out_stride=opt_int(a.sim_out_stride),
            projection=a.sim_projection,
            dtype=a.sim_dtype,
        ),
        model_version=a.model_version,
        feature_space=a.feature_space,
        self_test=not a.no_self_test,
        host=a.host,
        port=int(a.port),
        log_level=a.log_level,
    )
