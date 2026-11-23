# Copyright (c) OpenMMLab. All rights reserved.
"""Dense features from an mmrotate single-stage detector, for the similarity service.

Where the features come from (:class:`FeatureSpec`):

- ``backbone``: the backbone's multi-scale outputs;
- ``neck``: the neck's pyramid (e.g. ``CSPNeXtPAFPN``);
- ``head_cls`` / ``head_reg``: the input of the head's classification / regression
  predictor on every level, i.e. the output of the last cls / reg conv tower layer;
- ``head_cls@I`` / ``head_reg@I``: the output of the I-th stacked conv of that tower.

Several sources can be joined with ``+`` (``head_cls+head_reg``). The selected levels are
resampled to one ``out_stride`` (area pooling for finer levels, bilinear for coarser ones),
optionally L2-normalised per level, then concatenated or averaged. A :class:`Projector`
(centring or PCA, optionally whitened) can follow; the result is always L2-normalised and,
on the wire, quantised (``int8`` by default).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, replace

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

SOURCES = ('backbone', 'neck', 'head_cls', 'head_reg')
FUSIONS = ('concat', 'mean', 'single')
PROJECTIONS = ('none', 'center', 'pca')
DTYPES = ('int8', 'float16', 'float32')

# Prediction layers of the common single-stage heads; their input is the tower's output.
_CLS_PREDICTORS = ('rtm_cls', 'retina_cls', 'conv_cls', 'atss_cls', 'gfl_cls', 'vfnet_cls')
_REG_PREDICTORS = ('rtm_reg', 'retina_reg', 'conv_reg', 'atss_reg', 'gfl_reg', 'vfnet_reg')

Part = tuple[str, int | None]


def parse_source(source: str) -> list[Part]:
    """``'head_cls@1+neck'`` -> ``[('head_cls', 1), ('neck', None)]``."""
    parts: list[Part] = []
    for token in source.split('+'):
        token = token.strip()
        name, _, idx = token.partition('@')
        if name not in SOURCES:
            raise ValueError(f'unknown feature source {name!r}; choose from {SOURCES}')
        if idx and not name.startswith('head'):
            raise ValueError(f'"@I" selects a head conv layer; {name!r} has none')
        parts.append((name, int(idx) if idx else None))
    if not parts:
        raise ValueError('empty feature source')
    return parts


def part_key(part: Part) -> str:
    return part[0] if part[1] is None else f'{part[0]}@{part[1]}'


@dataclass(frozen=True)
class FeatureSpec:
    """How a similarity vector is built (everything that changes it, except the PCA)."""

    source: str = 'neck'
    levels: tuple[int, ...] | None = None  # None = every level
    fusion: str = 'mean'
    level_norm: str = 'l2'
    out_stride: int = 8
    projection: str = 'none'
    l2_before_projection: bool = True
    dtype: str = 'int8'

    def __post_init__(self) -> None:
        parse_source(self.source)
        if self.fusion not in FUSIONS:
            raise ValueError(f'fusion must be one of {FUSIONS}, got {self.fusion!r}')
        if self.level_norm not in ('l2', 'none'):
            raise ValueError(f'level_norm must be l2 or none, got {self.level_norm!r}')
        if self.projection not in PROJECTIONS:
            raise ValueError(f'projection must be one of {PROJECTIONS}, got {self.projection!r}')
        if self.dtype not in DTYPES:
            raise ValueError(f'dtype must be one of {DTYPES}, got {self.dtype!r}')
        if self.out_stride < 1:
            raise ValueError('out_stride must be >= 1')
        if self.levels is not None:
            object.__setattr__(self, 'levels', tuple(int(v) for v in self.levels))

    @property
    def parts(self) -> list[Part]:
        return parse_source(self.source)

    def to_dict(self) -> dict:
        d = asdict(self)
        d['levels'] = None if self.levels is None else list(self.levels)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> FeatureSpec:
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if known.get('levels') is not None:
            known['levels'] = tuple(known['levels'])
        return cls(**known)

    def with_overrides(self, overrides: dict) -> FeatureSpec:
        if not overrides:
            return self
        d = self.to_dict()
        d.update(overrides)
        return FeatureSpec.from_dict(d)

    def tag(self) -> str:
        """Short readable name, e.g. ``neck-mean-L0123-s8``."""
        src = self.source.replace('+', '_').replace('@', 'c')
        lv = 'Lall' if self.levels is None else 'L' + ''.join(str(v) for v in self.levels)
        proj = '' if self.projection == 'none' else f'-{self.projection}'
        return f'{src}-{self.fusion}-{lv}-s{self.out_stride}{proj}'

    def replace(self, **kwargs: object) -> FeatureSpec:
        return replace(self, **kwargs)


def default_spec(model: nn.Module) -> FeatureSpec:
    """The spec served when no profile and no ``--sim-*`` flag is given.

    The head's classification features at the finest level won on every SODA-A model
    measured with ``tools/similarity/suggest_api_hyperparameters.py`` (score 0.92-0.95
    against 0.57-0.71 for the neck mean; see api/README.md). Models without a recognised
    classification predictor fall back to the mean of the neck levels.
    """
    try:
        FeatureExtractor(model)._head_targets(('head_cls', None))
    except ValueError:
        return FeatureSpec()
    return FeatureSpec(source='head_cls', fusion='single', levels=(0,))


# --------------------------------------------------------------------------- capture
def batch_inputs(model: nn.Module, crops: Sequence[np.ndarray]) -> Tensor:
    """BGR uint8 ``[H, W, 3]`` crops -> the normalised, padded batch the backbone sees.

    Bypasses the test pipeline on purpose: the similarity service resamples tiles itself
    (``scale``), so a config ``Resize`` must never touch them.
    """
    tensors = [torch.from_numpy(np.ascontiguousarray(c.transpose(2, 0, 1))) for c in crops]
    return model.data_preprocessor({'inputs': tensors, 'data_samples': None}, False)['inputs']


def _as_list(x: object) -> list[Tensor]:
    if isinstance(x, Tensor):
        return [x]
    return [t for t in x if isinstance(t, Tensor)]  # type: ignore[union-attr]


class FeatureExtractor:
    """Runs the model as far as the requested sources need and returns per-level maps."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    @property
    def head(self) -> nn.Module:
        head = getattr(self.model, 'bbox_head', None)
        if head is None:
            raise ValueError(
                f'{type(self.model).__name__} has no bbox_head: head feature sources need a '
                'single-stage detector'
            )
        return head

    def _head_targets(self, part: Part) -> tuple[list[tuple[nn.Module, int | None]], bool]:
        """Modules to hook for ``part``: ``([(module, level or None)], use_input)``.

        ``level`` is None for a module shared across levels (then the call order is the
        level order). ``use_input`` hooks the module's input instead of its output.
        """
        head = self.head
        name, idx = part
        kind = 'cls' if name == 'head_cls' else 'reg'
        if idx is None:
            for attr in _CLS_PREDICTORS if kind == 'cls' else _REG_PREDICTORS:
                mod = getattr(head, attr, None)
                if mod is None:
                    continue
                if isinstance(mod, nn.ModuleList):
                    return [(m, lvl) for lvl, m in enumerate(mod)], True
                return [(mod, None)], True
            raise ValueError(
                f'{type(head).__name__} has none of the {kind} predictors '
                f'{_CLS_PREDICTORS if kind == "cls" else _REG_PREDICTORS}'
            )
        towers = getattr(head, f'{kind}_convs', None)
        if not isinstance(towers, nn.ModuleList) or not len(towers):
            raise ValueError(f'{type(head).__name__} has no {kind}_convs to pick layer {idx} from')
        if isinstance(towers[0], nn.ModuleList):  # per-level towers (SepBN heads)
            depth = len(towers[0])
            if not -depth <= idx < depth:
                raise ValueError(f'{kind}_convs has {depth} layers per level; @{idx} is out')
            return [(tower[idx], lvl) for lvl, tower in enumerate(towers)], False
        depth = len(towers)
        if not -depth <= idx < depth:
            raise ValueError(f'{kind}_convs has {depth} layers; @{idx} is out of range')
        return [(towers[idx], None)], False

    def capture(self, inputs: Tensor, parts: Iterable[Part]) -> dict[str, list[Tensor]]:
        """Per-level feature maps ``[B, C, h, w]`` for every requested part."""
        parts = list(dict.fromkeys(parts))
        names = {p[0] for p in parts}
        model = self.model
        out: dict[str, list[Tensor]] = {}
        feats = model.backbone(inputs)
        if 'backbone' in names:
            out['backbone'] = _as_list(feats)
        head_parts = [p for p in parts if p[0].startswith('head')]
        if 'neck' not in names and not head_parts:
            return out
        if getattr(model, 'with_neck', False) or getattr(model, 'neck', None) is not None:
            feats = model.neck(feats)
        neck = _as_list(feats)
        if 'neck' in names:
            out['neck'] = neck
        if not head_parts:
            return out

        store: dict[str, dict[int, Tensor] | list[Tensor]] = {}
        handles = []

        def make_hook(
            key: str, level: int | None, use_input: bool
        ) -> object:  # returns a hook callable
            def hook(_mod: nn.Module, args: tuple, output: object = None) -> None:
                t = args[0] if use_input else output
                bucket = store[key]
                if level is None:
                    bucket.append(t)  # type: ignore[union-attr]
                else:
                    bucket[level] = t  # type: ignore[index]

            return hook

        for part in head_parts:
            key = part_key(part)
            targets, use_input = self._head_targets(part)
            store[key] = [] if targets[0][1] is None else {}
            for mod, level in targets:
                hook = make_hook(key, level, use_input)
                if use_input:
                    handles.append(mod.register_forward_pre_hook(hook))  # type: ignore[arg-type]
                else:
                    handles.append(mod.register_forward_hook(hook))  # type: ignore[arg-type]
        try:
            self.head(tuple(neck))
        finally:
            for h in handles:
                h.remove()
        for key, bucket in store.items():
            maps = bucket if isinstance(bucket, list) else [bucket[k] for k in sorted(bucket)]
            if len(maps) != len(neck):
                raise RuntimeError(
                    f'{key}: captured {len(maps)} maps for {len(neck)} pyramid levels'
                )
            out[key] = list(maps)
        return out


def select_levels(maps: Sequence[Tensor], levels: Sequence[int] | None) -> list[Tensor]:
    if levels is None:
        return list(maps)
    n = len(maps)
    for lv in levels:
        if not -n <= lv < n:
            raise ValueError(f'level {lv} out of range: the source has {n} levels')
    return [maps[lv] for lv in levels]


def fuse(captured: dict[str, list[Tensor]], spec: FeatureSpec, input_hw: tuple[int, int]) -> Tensor:
    """Captured maps -> one ``[B, D, ceil(H/s), ceil(W/s)]`` float map at ``spec.out_stride``."""
    H, W = input_hw
    s = spec.out_stride
    hc, wc = -(-H // s), -(-W // s)
    maps: list[Tensor] = []
    for part in spec.parts:
        maps.extend(select_levels(captured[part_key(part)], spec.levels))
    resampled = []
    for m in maps:
        m = m.float()
        if spec.level_norm == 'l2':
            m = F.normalize(m, dim=1)
        if tuple(m.shape[-2:]) != (hc, wc):
            if m.shape[-2] >= hc and m.shape[-1] >= wc:
                m = F.adaptive_avg_pool2d(m, (hc, wc))
            else:
                m = F.interpolate(m, size=(hc, wc), mode='bilinear', align_corners=False)
        resampled.append(m)
    if spec.fusion == 'single':
        if len(resampled) != 1:
            raise ValueError(f'fusion=single needs exactly one map, got {len(resampled)}')
        return resampled[0]
    if spec.fusion == 'mean':
        chans = {m.shape[1] for m in resampled}
        if len(chans) != 1:
            raise ValueError(f'fusion=mean needs equal channels per level, got {sorted(chans)}')
        return torch.stack(resampled).mean(0)
    return torch.cat(resampled, dim=1)


# --------------------------------------------------------------------------- projection
@dataclass
class Moments:
    """Running ``n``, ``sum x`` and ``sum x x^T`` (float64) for fitting a projection."""

    n: int
    total: Tensor
    outer: Tensor

    @classmethod
    def zeros(cls, dim: int, device: torch.device | str = 'cpu') -> Moments:
        return cls(
            0,
            torch.zeros(dim, dtype=torch.float64, device=device),
            torch.zeros(dim, dim, dtype=torch.float64, device=device),
        )

    @classmethod
    def of(cls, x: Tensor) -> Moments:
        m = cls.zeros(x.shape[1], x.device)
        m.add(x)
        return m

    def add(self, x: Tensor) -> None:
        x = x.to(self.total.device, torch.float64)
        self.n += int(x.shape[0])
        self.total += x.sum(0)
        self.outer += x.T @ x

    def explained(self) -> Tensor:
        """Eigenvalues of the covariance, largest first (to report kept variance)."""
        mean = self.total / self.n
        cov = (self.outer - self.n * torch.outer(mean, mean)) / max(1, self.n - 1)
        return torch.linalg.eigvalsh(cov).flip(0).clamp_min(0)


@dataclass
class Projector:
    """``x -> (x - mean) @ components.T * scale`` (centring when ``components`` is None)."""

    mean: np.ndarray
    components: np.ndarray | None = None
    explained_variance: np.ndarray | None = None
    whiten: bool = False

    @property
    def out_dim(self) -> int:
        return int(self.mean.shape[0] if self.components is None else self.components.shape[0])

    @property
    def kind(self) -> str:
        return 'center' if self.components is None else 'pca'

    def apply(self, x: Tensor) -> Tensor:
        mean = torch.as_tensor(self.mean, device=x.device, dtype=x.dtype)
        x = x - mean
        if self.components is not None:
            comp = torch.as_tensor(self.components, device=x.device, dtype=x.dtype)
            x = x @ comp.T
            if self.whiten and self.explained_variance is not None:
                var = torch.as_tensor(self.explained_variance, device=x.device, dtype=x.dtype)
                x = x / torch.sqrt(var + 1e-12)
        return x

    def digest(self) -> str:
        h = hashlib.blake2b(digest_size=4)
        h.update(np.ascontiguousarray(self.mean, np.float32).tobytes())
        if self.components is not None:
            h.update(np.ascontiguousarray(self.components, np.float32).tobytes())
        h.update(b'w' if self.whiten else b'-')
        return h.hexdigest()

    @classmethod
    def fit(
        cls, x: Tensor | np.ndarray, kind: str, dim: int | None = None, whiten: bool = False
    ) -> Projector:
        """Fit on ``[N, D]`` vectors (already L2-normalised if the spec says so)."""
        xt = torch.as_tensor(x, dtype=torch.float64)
        return cls.from_moments(Moments.of(xt), kind, dim, whiten)

    @classmethod
    def from_moments(
        cls, m: Moments, kind: str, dim: int | None = None, whiten: bool = False
    ) -> Projector:
        """Fit from streamed first and second moments (no need to keep the vectors)."""
        if m.n < 2:
            raise ValueError('need at least 2 vectors to fit a projection')
        mean = m.total / m.n
        if kind == 'center':
            return cls(mean=mean.float().cpu().numpy())
        if kind != 'pca':
            raise ValueError(f'unknown projection {kind!r}')
        cov = (m.outer - m.n * torch.outer(mean, mean)) / (m.n - 1)
        evals, evecs = torch.linalg.eigh(cov)
        order = torch.argsort(evals, descending=True)
        k = min(int(dim or cov.shape[0]), cov.shape[0])
        evals, evecs = evals[order][:k].clamp_min(0), evecs[:, order][:, :k]
        # deterministic sign: largest-magnitude loading of each axis is positive
        idx = evecs.abs().argmax(0)
        signs = torch.sign(evecs[idx, torch.arange(k)])
        signs[signs == 0] = 1
        evecs = evecs * signs
        return cls(
            mean=mean.float().cpu().numpy(),
            components=evecs.T.float().cpu().numpy(),
            explained_variance=evals.float().cpu().numpy(),
            whiten=whiten,
        )


def postprocess(fused: Tensor, spec: FeatureSpec, projector: Projector | None) -> Tensor:
    """``[B, D, h, w]`` -> ``[B, h, w, K]`` unit vectors (projection included)."""
    x = fused.permute(0, 2, 3, 1).float()
    if spec.l2_before_projection:
        x = F.normalize(x, dim=-1)
    if spec.projection != 'none':
        if projector is None:
            raise ValueError(f'projection={spec.projection} needs a fitted projector (a profile)')
        x = projector.apply(x)
    return F.normalize(x, dim=-1)


def quantize(unit: np.ndarray, dtype: str) -> tuple[np.ndarray, float]:
    """Unit vectors -> ``(wire array, scale)``; ``int8`` is ``round(v * 127)`` clipped."""
    if dtype == 'int8':
        return np.clip(np.rint(unit * 127.0), -127, 127).astype(np.int8), 127.0
    if dtype == 'float16':
        return unit.astype(np.float16), 1.0
    return unit.astype(np.float32), 1.0


def dequantize(arr: np.ndarray, scale: float) -> np.ndarray:
    return arr.astype(np.float32) / float(scale)


def spec_digest(spec: FeatureSpec, projector: Projector | None, extra: dict | None = None) -> str:
    h = hashlib.blake2b(digest_size=4)
    h.update(json.dumps(spec.to_dict(), sort_keys=True).encode())
    if projector is not None and spec.projection != 'none':
        h.update(projector.digest().encode())
    if extra:
        h.update(json.dumps(extra, sort_keys=True).encode())
    return h.hexdigest()
