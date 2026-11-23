# Copyright (c) OpenMMLab. All rights reserved.
"""The similarity service: ``GET /health`` and ``POST /embed`` of qgis_annotations'
``docs/SIMILARITY_API.md``."""

from __future__ import annotations

import base64
import logging
import math
from collections.abc import Callable

import numpy as np
import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import tiling
from .codec import BadImageError, decode_png
from .features import (
    FeatureExtractor,
    FeatureSpec,
    Projector,
    batch_inputs,
    default_spec,
    dequantize,
    fuse,
    postprocess,
    quantize,
    spec_digest,
)
from .model import LoadedModel

log = logging.getLogger('api')


# --------------------------------------------------------------------------- schemas
class SimilarityHealth(BaseModel):
    """``GET /health`` of a similarity service (SIMILARITY_API.md section 2)."""

    kind: str = 'similarity'
    capabilities: list[str] = Field(default_factory=lambda: ['embed'])
    ready: bool = True
    backend: str = 'mmrotate-api'
    native_size: int = Field(description='largest tile side the server sends, in source px')
    out_stride: int = Field(description='source px per output cell; divides native_size')
    feature_dim: int
    model_version: str
    feature_space: str
    scale: float = Field(1.0, description='always 1: the service never resamples inside')
    patch_size: int | None = None
    model_name: str
    dtype: str = 'int8'
    precision: str
    device: str
    gsd_m: float | None = Field(
        None,
        description='m/px the qgis server resamples every image to before /embed (it adds '
        '+gsd<g> to the feature space); absent: tiles are source pixels',
    )
    input_size: list[int]
    pad_mode: str
    patch_overlap: int
    features: dict = Field(description='the feature spec and projection (informational)')


class EmbedRequest(BaseModel):
    image: str = Field(
        description=(
            'base64 PNG, RGB, at most native_size a side: source pixels, or '
            'resampled to gsd_m by the server when /health reports one'
        )
    )


class EmbedResponse(BaseModel):
    height: int = Field(description='ceil(H / stride)')
    width: int = Field(description='ceil(W / stride)')
    dim: int
    stride: int
    dtype: str = 'int8'
    scale: float = Field(127.0, description='divide the values by it to get unit vectors')
    data: str = Field(description='base64 of the C-order [height, width, dim] array')


# --------------------------------------------------------------------------- service
class SimilarityService:
    """Embeds one tile into a grid of unit vectors at ``out_stride``.

    ``gsd_m`` is only advertised: the qgis_annotations server resamples every image to it
    before ``/embed``, so a tile arrives at the scale the model wants and is never resampled
    here (``scale`` is always 1).
    """

    def __init__(
        self,
        lm: LoadedModel,
        spec: FeatureSpec | None = None,
        projector: Projector | None = None,
        *,
        gsd_m: float | None = None,
        tile_size: int | None = None,
        patch_overlap: int = 128,
        max_batch: int = 8,
        model_version: str | None = None,
        feature_space: str | None = None,
        profile_meta: dict | None = None,
    ) -> None:
        self.lm = lm
        self.spec = spec or default_spec(lm.model)
        self.projector = projector
        self.max_batch = max_batch
        self.extractor = FeatureExtractor(lm.model)
        self.gsd_m = gsd_m
        s = self.spec.out_stride
        if self.spec.projection != 'none' and projector is None:
            raise ValueError(
                f'--sim-projection {self.spec.projection} needs a fitted projection: pass '
                '--similarity-profile (tools/similarity/suggest_api_hyperparameters.py writes one)'
            )
        if projector is not None and self.spec.projection != 'none':
            if projector.kind != self.spec.projection:
                raise ValueError(
                    f'profile projection is {projector.kind}, spec asks for {self.spec.projection}'
                )
        for side, what in ((lm.input_w, 'input width'), (lm.input_h, 'input height')):
            tiling.check_divisible(side, s, what, f'similarity out_stride {s}; set --input-size')

        tile = int(tile_size or min(lm.input_size))
        snapped = tiling.snap_down(tile, s)
        if snapped != tile:
            log.info('similarity tile size %d snapped to %d (out_stride)', tile, snapped)
        self.tile_size = snapped
        overlap = int(round(patch_overlap / s)) * s
        self.patch_overlap = overlap
        if self.tile_size > min(lm.input_size):
            tiling.window_starts(self.tile_size, min(lm.input_size), overlap, s)
        if profile_meta:
            for key, mine in (('input_size', list(lm.input_size)), ('pad_mode', lm.pad_mode)):
                theirs = profile_meta.get(key)
                if theirs is not None and list(theirs) != list(mine) and theirs != mine:
                    log.warning(
                        'profile was tuned with %s=%s but the service runs %s=%s',
                        key,
                        theirs,
                        key,
                        mine,
                    )
        self.feature_dim = self._probe_dim()
        extra = {'input': list(lm.input_size), 'pad': lm.pad_mode}
        if self.tile_size > min(lm.input_size):
            extra['overlap'] = overlap
        digest = spec_digest(self.spec, projector, extra)
        name = lm.model_name[:60]
        space = (
            f'{name}+w{lm.weights_digest}+{self.spec.tag()}+f{digest}'
            f'-m{lm.input_w}x{lm.input_h}s{s}d{self.feature_dim}'
        )
        if self.spec.dtype == 'int8':
            space += '-int8'
        self.feature_space = feature_space or space[:200]
        self.model_version = model_version or f'{space}-{lm.precision}-t{self.tile_size}'

    # ---------------------------------------------------------------- core
    def _forward(self, crops: list[np.ndarray]) -> torch.Tensor:
        """Model-px BGR crops (each at most the input size) -> ``[B, h, w, D]`` unit vectors."""
        lm = self.lm
        padded = [lm.window_image(c) for c in crops]
        with lm.lock, torch.inference_mode(), lm.autocast():
            inputs = batch_inputs(lm.model, padded)
            captured = self.extractor.capture(inputs, self.spec.parts)
            fused = fuse(captured, self.spec, tuple(inputs.shape[-2:]))
            return postprocess(fused, self.spec, self.projector).float()

    def _probe_dim(self) -> int:
        s = self.spec.out_stride
        probe = np.full((s * 4, s * 4, 3), 114, np.uint8)
        return int(self._forward([probe]).shape[-1])

    def embed_unit(self, tile: np.ndarray) -> np.ndarray:
        """BGR uint8 tile -> ``[ceil(h/s), ceil(w/s), D]`` float32 unit vectors."""
        H, W = tile.shape[:2]
        s = self.spec.out_stride
        win_h, win_w = self.lm.input_h, self.lm.input_w
        ys = tiling.window_starts(H, win_h, self.patch_overlap, s)
        xs = tiling.window_starts(W, win_w, self.patch_overlap, s)
        y_core = tiling.core_ranges(ys, win_h, H, s)
        x_core = tiling.core_ranges(xs, win_w, W, s)
        jobs = [(iy, ix) for iy in range(len(ys)) for ix in range(len(xs))]
        grid = np.zeros((math.ceil(H / s), math.ceil(W / s), self.feature_dim), np.float32)
        for start in range(0, len(jobs), self.max_batch):
            chunk = jobs[start : start + self.max_batch]
            crops = [tile[ys[iy] : ys[iy] + win_h, xs[ix] : xs[ix] + win_w] for iy, ix in chunk]
            feats = self._forward(crops).cpu().numpy()
            for (iy, ix), f in zip(chunk, feats, strict=True):
                (ylo, yhi), (xlo, xhi) = y_core[iy], x_core[ix]
                cy0, cy1 = ylo // s, math.ceil(yhi / s)
                cx0, cx1 = xlo // s, math.ceil(xhi / s)
                oy, ox = ys[iy] // s, xs[ix] // s
                grid[cy0:cy1, cx0:cx1] = f[cy0 - oy : cy1 - oy, cx0 - ox : cx1 - ox]
        return grid

    def embed(self, tile: np.ndarray) -> EmbedResponse:
        h, w = tile.shape[:2]
        if max(h, w) > self.tile_size:
            raise ValueError(f'tile {w}x{h} is larger than native_size {self.tile_size}')
        unit = self.embed_unit(tile)
        arr, qscale = quantize(unit, self.spec.dtype)
        return EmbedResponse(
            height=arr.shape[0],
            width=arr.shape[1],
            dim=arr.shape[2],
            stride=self.spec.out_stride,
            dtype=self.spec.dtype,
            scale=qscale,
            data=base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode(),
        )

    # ---------------------------------------------------------------- health
    def health(self) -> SimilarityHealth:
        lm = self.lm
        proj = None
        if self.projector is not None and self.spec.projection != 'none':
            proj = {
                'kind': self.projector.kind,
                'dim': self.projector.out_dim,
                'whiten': self.projector.whiten,
                'digest': self.projector.digest(),
            }
        return SimilarityHealth(
            native_size=self.tile_size,
            out_stride=self.spec.out_stride,
            feature_dim=self.feature_dim,
            model_version=self.model_version,
            feature_space=self.feature_space,
            patch_size=self.spec.out_stride,
            model_name=lm.model_name,
            dtype=self.spec.dtype,
            precision=lm.precision,
            device=lm.device,
            gsd_m=self.gsd_m,
            input_size=list(lm.input_size),
            pad_mode=lm.pad_mode,
            patch_overlap=self.patch_overlap,
            features={**self.spec.to_dict(), 'projector': proj},
        )

    def self_test(self) -> None:
        """A random full tile and a short one: grid shape, unit norms, no zero vectors."""
        rng = np.random.default_rng(0)
        n, s = self.tile_size, self.spec.out_stride
        for h, w in ((n, n), (max(1, n // 3) + 5, n)):
            tile = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
            out = self.embed(tile)
            want = (math.ceil(h / s), math.ceil(w / s))
            if (out.height, out.width) != want or out.dim != self.feature_dim:
                raise RuntimeError(
                    f'similarity self-test: grid {out.height}x{out.width}x{out.dim} for a '
                    f'{w}x{h} tile, expected {want[0]}x{want[1]}x{self.feature_dim}'
                )
            dtype = {'int8': np.int8, 'float16': np.float16, 'float32': np.float32}[out.dtype]
            arr = np.frombuffer(base64.b64decode(out.data), dtype).reshape(want + (out.dim,))
            norms = np.linalg.norm(dequantize(arr, out.scale), axis=-1)
            if norms.min() < 0.8 or norms.max() > 1.2:
                raise RuntimeError(
                    f'similarity self-test: vector norms {norms.min():.3f}..{norms.max():.3f}'
                )
        log.info(
            'similarity self-test passed (tile %d, stride %d, dim %d, space %s, %s)',
            self.tile_size,
            s,
            self.feature_dim,
            self.feature_space,
            f'gsd_m {self.gsd_m:g}: the server resamples to it'
            if self.gsd_m
            else 'no gsd_m: tiles at source pixels',
        )


# --------------------------------------------------------------------------- router
def make_router(get_service: Callable[[], SimilarityService]) -> APIRouter:
    router = APIRouter(tags=['similarity'])

    @router.get(
        '/health',
        response_model=SimilarityHealth,
        response_model_exclude_none=True,
        summary='Similarity service identity, tile size, stride and feature space',
    )
    def health() -> SimilarityHealth:
        return get_service().health()

    @router.post(
        '/embed',
        response_model=EmbedResponse,
        summary='Grid of unit feature vectors for one tile (at gsd_m when /health has one)',
        responses={400: {'description': 'bad PNG or tile larger than native_size'}},
    )
    def embed(req: EmbedRequest) -> EmbedResponse:
        svc = get_service()
        try:
            tile = decode_png(req.image)
        except BadImageError as e:
            raise HTTPException(400, detail=str(e)) from e
        h, w = tile.shape[:2]
        if max(h, w) > svc.tile_size:
            raise HTTPException(
                400, detail=f'tile {w}x{h} is larger than native_size {svc.tile_size}'
            )
        try:
            return svc.embed(tile)
        except Exception as e:  # noqa: BLE001 - report the model's failure to the client
            log.exception('embed failed')
            raise HTTPException(500, detail=f'{type(e).__name__}: {e}'[:500]) from e

    return router
