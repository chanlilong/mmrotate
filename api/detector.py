# Copyright (c) OpenMMLab. All rights reserved.
"""The detector service: ``GET /health`` and ``POST /detect`` of qgis_annotations'
``docs/DETECTOR_API.md``."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

import numpy as np
import torch
from fastapi import APIRouter, HTTPException
from mmcv.ops import batched_nms
from pydantic import BaseModel, Field

from mmrotate.structures.bbox import rbox2qbox

from . import tiling
from .codec import BadImageError, decode_png
from .model import LoadedModel

log = logging.getLogger('api')


# --------------------------------------------------------------------------- schemas
class DetectorHealth(BaseModel):
    """``GET /health`` of a fixed-class detector (DETECTOR_API.md section 3)."""

    ready: bool = True
    kind: str = 'detector'
    backend: str = 'mmrotate-api'
    capabilities: list[str] = Field(default_factory=lambda: ['class_agnostic_nms'])
    model_name: str
    model_version: str
    classes: list[str]
    gsd_m: float | None = Field(None, description='metres per pixel the model was trained at')
    native_size: int = Field(description='tile side (px) the client pads every tile to')
    max_batch: int
    max_side_px: int
    nms_iou: float | None = Field(None, description="the config's own NMS IoU")
    weights_path: str = ''
    config_path: str = ''
    weights_size: int | None = None
    weights_mtime: int | None = None
    input_size: list[int] = Field(description='model input [w, h] in model px')
    resize_mode: str
    pad_mode: str
    patch_overlap: int
    precision: str
    device: str


class DetectRequest(BaseModel):
    images: list[str] = Field(
        min_length=1, description='base64 PNG tiles, 8-bit RGB, at most native_size a side'
    )
    score_thr: float = Field(0.3, ge=0, le=1, description='drop instances scoring below it')
    nms_iou: float | None = Field(
        None, ge=0, le=1, description="NMS IoU for this batch; null = the model's own"
    )
    class_agnostic: bool | None = Field(
        None, description="true = NMS across classes; null = the model's own rule"
    )


class Instance(BaseModel):
    quad: list[list[float]] = Field(
        min_length=4, max_length=4, description='four [x, y] corners in tile pixels'
    )
    score: float = Field(ge=0, le=1)
    label: str


class TileResult(BaseModel):
    instances: list[Instance]


class DetectResponse(BaseModel):
    results: list[TileResult] = Field(description='one per image, in request order')


# --------------------------------------------------------------------------- service
def _box_tensor(bboxes: object) -> torch.Tensor:
    return bboxes.tensor if hasattr(bboxes, 'tensor') else bboxes  # type: ignore[return-value]


def _quads(boxes: torch.Tensor) -> torch.Tensor:
    """``[N, 5]`` rotated (cx, cy, w, h, angle) or ``[N, 4]`` (x1, y1, x2, y2) -> ``[N, 4, 2]``."""
    if boxes.shape[-1] == 5:
        return rbox2qbox(boxes).reshape(-1, 4, 2)
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([x1, y1, x2, y1, x2, y2, x1, y2], -1).reshape(-1, 4, 2)


def _centres_sizes(boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if boxes.shape[-1] == 5:
        return boxes[:, :2], boxes[:, 2:4]
    return (boxes[:, :2] + boxes[:, 2:]) / 2, boxes[:, 2:] - boxes[:, :2]


class DetectorService:
    """Batched, windowed detection with per-request NMS overrides."""

    def __init__(
        self,
        lm: LoadedModel,
        *,
        gsd_m: float | None = None,
        tile_size: int | None = None,
        max_batch: int = 8,
        max_side_px: int | None = None,
        patch_overlap: int = 128,
        model_version: str | None = None,
    ) -> None:
        self.lm = lm
        self.gsd_m = gsd_m
        self.tile_size = int(tile_size or max(lm.input_size))
        self.max_batch = max_batch
        self.max_side_px = int(max_side_px or 2 * self.tile_size)
        self.patch_overlap = patch_overlap
        if self.tile_size > min(lm.input_size):
            # the tile is cut into windows of the input size; check the overlap leaves a step
            tiling.window_starts(self.tile_size, min(lm.input_size), patch_overlap)
        if lm.resize_mode == 'config' and self.tile_size != max(lm.input_size):
            log.warning(
                "--resize-mode config with tile size %d != input size %s: the config's Resize "
                'rescales every tile, so the model no longer sees gsd_m',
                self.tile_size,
                lm.input_size,
            )
        self.model_version = model_version or f'{lm.model_name}@{lm.weights_digest}'

    # ---------------------------------------------------------------- health
    def health(self) -> DetectorHealth:
        lm = self.lm
        size = mtime = None
        if lm.weights_path and os.path.exists(lm.weights_path):
            st = os.stat(lm.weights_path)
            size, mtime = int(st.st_size), int(st.st_mtime)
        return DetectorHealth(
            model_name=lm.model_name,
            model_version=self.model_version,
            classes=lm.classes,
            gsd_m=self.gsd_m,
            native_size=self.tile_size,
            max_batch=self.max_batch,
            max_side_px=self.max_side_px,
            nms_iou=lm.config_nms_iou(),
            weights_path=lm.weights_path,
            config_path=lm.config_path,
            weights_size=size,
            weights_mtime=mtime,
            input_size=list(lm.input_size),
            resize_mode=lm.resize_mode,
            pad_mode=lm.pad_mode,
            patch_overlap=self.patch_overlap,
            precision=lm.precision,
            device=lm.device,
        )

    # ---------------------------------------------------------------- inference
    def _windows(self, img: np.ndarray) -> list[tiling.Window]:
        h, w = img.shape[:2]
        lm = self.lm
        if lm.resize_mode == 'config':
            return [tiling.Window(0, 0, w, h)]  # the config's Resize handles any size
        return tiling.plan_windows(h, w, lm.input_h, lm.input_w, self.patch_overlap)

    def predict(
        self,
        images: list[np.ndarray],
        nms_iou: float | None = None,
        class_agnostic: bool | None = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """BGR uint8 tiles -> per tile ``(boxes, scores, labels)`` in tile pixels (all scores)."""
        lm = self.lm
        jobs = [(i, win) for i, img in enumerate(images) for win in self._windows(img)]
        outs: list = []
        with (
            lm.lock,
            torch.inference_mode(),
            lm.nms_override(nms_iou, class_agnostic),
            lm.autocast(),
        ):
            for start in range(0, len(jobs), self.max_batch):
                chunk = jobs[start : start + self.max_batch]
                crops = []
                for i, win in chunk:
                    crop = images[i][win.y0 : win.y0 + win.height, win.x0 : win.x0 + win.width]
                    crops.append(lm.window_image(crop) if lm.resize_mode == 'none' else crop)
                outs.extend(lm.model.test_step(lm.collate(crops)))
        per_image: list[list[tuple[tiling.Window, object]]] = [[] for _ in images]
        for (i, win), out in zip(jobs, outs, strict=True):
            per_image[i].append((win, out.pred_instances))
        results = []
        nms_cfg = lm.nms_cfg(nms_iou, class_agnostic)
        for parts in per_image:
            boxes, scores, labels = [], [], []
            for win, inst in parts:
                b = _box_tensor(inst.bboxes).float().clone()
                b[:, 0] += win.x0
                b[:, 1] += win.y0
                if b.shape[-1] == 4:
                    b[:, 2] += win.x0
                    b[:, 3] += win.y0
                boxes.append(b)
                scores.append(inst.scores.float())
                labels.append(inst.labels)
            b, s, lab = torch.cat(boxes), torch.cat(scores), torch.cat(labels)
            if len(parts) > 1 and b.numel():
                _, keep = batched_nms(b, s, lab, nms_cfg)
                b, s, lab = b[keep], s[keep], lab[keep]
            results.append((b, s, lab))
        return results

    def detect(
        self,
        images: list[np.ndarray],
        score_thr: float,
        nms_iou: float | None = None,
        class_agnostic: bool | None = None,
    ) -> list[list[Instance]]:
        """BGR uint8 tiles -> the instances of each tile, as the contract wants them."""
        results = []
        for img, (b, s, lab) in zip(
            images, self.predict(images, nms_iou, class_agnostic), strict=True
        ):
            h, w = img.shape[:2]
            centre, size = _centres_sizes(b)
            ok = (
                (s >= score_thr)
                & (size.min(-1).values >= 1)
                & (centre[:, 0] >= 0)
                & (centre[:, 1] >= 0)
                & (centre[:, 0] < w)
                & (centre[:, 1] < h)
            )
            b, s, lab = b[ok], s[ok], lab[ok]
            order = torch.argsort(s, descending=True)
            quads = _quads(b[order]).cpu().tolist()
            names = self.lm.classes
            results.append(
                [
                    Instance(
                        quad=q,
                        score=float(min(1.0, max(0.0, sc))),
                        label=names[int(li)] if 0 <= int(li) < len(names) else '',
                    )
                    for q, sc, li in zip(
                        quads, s[order].cpu().tolist(), lab[order].cpu().tolist(), strict=True
                    )
                ]
            )
        return results

    def self_test(self) -> None:
        """Two grey tiles (one full, one short) through ``detect``; raises on a broken setup."""
        full = np.full((self.tile_size, self.tile_size, 3), 114, np.uint8)
        short = np.full((max(32, self.tile_size // 3), self.tile_size, 3), 114, np.uint8)
        out = self.detect([full, short], score_thr=0.99)
        if len(out) != 2:
            raise RuntimeError(f'detector self-test: {len(out)} results for 2 tiles')
        log.info('detector self-test passed (tile %d)', self.tile_size)


# --------------------------------------------------------------------------- router
def _decode_all(images: list[str], max_side: int) -> list[np.ndarray]:
    out = []
    for k, data in enumerate(images):
        try:
            img = decode_png(data)
        except BadImageError as e:
            raise HTTPException(400, detail=f'images[{k}]: {e}') from e
        if max(img.shape[:2]) > max_side:
            raise HTTPException(
                400,
                detail=f'images[{k}] is {img.shape[1]}x{img.shape[0]}, over max_side_px {max_side}',
            )
        out.append(img)
    return out


def make_router(get_service: Callable[[], DetectorService]) -> APIRouter:
    router = APIRouter(tags=['detector'])

    @router.get(
        '/health',
        response_model=DetectorHealth,
        response_model_exclude_none=True,
        summary='Detector identity, classes, GSD and tile sizes',
    )
    def health() -> DetectorHealth:
        return get_service().health()

    @router.post(
        '/detect',
        response_model=DetectResponse,
        summary='Rotated boxes for a batch of tiles',
        responses={400: {'description': 'too many tiles, bad PNG or tile too large'}},
    )
    def detect(req: DetectRequest) -> DetectResponse:
        svc = get_service()
        if len(req.images) > svc.max_batch:
            raise HTTPException(
                400, detail=f'{len(req.images)} tiles, over max_batch {svc.max_batch}'
            )
        images = _decode_all(req.images, svc.max_side_px)
        try:
            results = svc.detect(images, req.score_thr, req.nms_iou, req.class_agnostic)
        except Exception as e:  # noqa: BLE001 - report the model's failure to the client
            log.exception('detect failed')
            raise HTTPException(500, detail=f'{type(e).__name__}: {e}'[:500]) from e
        return DetectResponse(results=[TileResult(instances=r) for r in results])

    return router
