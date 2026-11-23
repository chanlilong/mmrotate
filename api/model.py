# Copyright (c) OpenMMLab. All rights reserved.
"""Load an mmrotate config + checkpoint once and run it under one lock."""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from mmcv.transforms import Compose
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from torch import nn

from mmrotate.registry import DATASETS, MODELS
from mmrotate.utils import register_all_modules

from . import tiling

log = logging.getLogger('api')


def build_model(cfg: Config, checkpoint: str | None, device: str) -> tuple[nn.Module, dict]:
    """Build ``cfg.model`` in eval mode and load ``checkpoint`` into it.

    Same recipe as ``mmrotate/tensorrt/export.py:build_detector`` (copied rather than imported
    so the service does not pull in TensorRT): the backbone's own pretrained weights are not
    loaded, everything comes from the detector checkpoint. Returns ``(model, ckpt_meta)``.
    """
    register_all_modules(init_default_scope=False)
    init_default_scope(cfg.get('default_scope', 'mmrotate'))
    model_cfg = cfg.model.copy()
    backbone = model_cfg.get('backbone')
    if isinstance(backbone, dict):
        backbone = backbone.copy()
        if 'checkpoint' in backbone:
            backbone['checkpoint'] = None
        backbone.pop('init_cfg', None)
        if 'pretrained' in backbone:
            backbone['pretrained'] = None
        model_cfg['backbone'] = backbone
    model = MODELS.build(model_cfg)
    meta: dict = {}
    if checkpoint is not None:
        ckpt = load_checkpoint(model, checkpoint, map_location='cpu', strict=False, logger='silent')
        meta = ckpt.get('meta', {}) or {}
    model.cfg = cfg
    model = revert_sync_batchnorm(model)
    return model.to(device).eval(), meta


def weights_digest(path: str, head_bytes: int = 1 << 20) -> str:
    """Short digest of a checkpoint: the first MiB plus the size (fast on 500 MB files)."""
    size = os.path.getsize(path)
    h = hashlib.blake2b(digest_size=4)
    with open(path, 'rb') as f:
        h.update(f.read(head_bytes))
    h.update(str(size).encode())
    return h.hexdigest()


def _head_num_classes(model: nn.Module) -> int | None:
    head = getattr(model, 'bbox_head', None)
    return getattr(head, 'num_classes', None)


def _dataset_class_names(cfg: Config) -> list[str] | None:
    """The default ``METAINFO['classes']`` of the config's dataset class (e.g. DOTADataset)."""
    for loader in ('test_dataloader', 'val_dataloader', 'train_dataloader'):
        ds = (cfg.get(loader) or {}).get('dataset', {})
        while ds and 'type' in ds and ('dataset' in ds or 'datasets' in ds):
            ds = ds.get('dataset') or ds['datasets'][0]
        if not ds or 'type' not in ds:
            continue
        ds_type = ds['type']
        ds_cls = DATASETS.get(ds_type) if isinstance(ds_type, str) else ds_type
        classes = (getattr(ds_cls, 'METAINFO', None) or {}).get('classes')
        if classes:
            return list(classes)
    return None


def resolve_classes(model: nn.Module, cfg: Config, meta: dict) -> list[str]:
    """Class names from the checkpoint's ``dataset_meta``, else the config; checked against
    the head's ``num_classes`` (a silent COCO-80 fallback is the classic failure)."""
    classes = None
    dataset_meta = meta.get('dataset_meta') or {}
    if dataset_meta.get('classes'):
        classes = list(dataset_meta['classes'])
    elif (cfg.get('metainfo') or {}).get('classes'):
        classes = list(cfg.metainfo.classes)
    else:
        for loader in ('test_dataloader', 'val_dataloader', 'train_dataloader'):
            ds = (cfg.get(loader) or {}).get('dataset', {})
            if (ds.get('metainfo') or {}).get('classes'):
                classes = list(ds['metainfo']['classes'])
                break
        if classes is None:
            classes = _dataset_class_names(cfg)
    num = _head_num_classes(model)
    if not classes:
        raise ValueError(
            'No class names: the checkpoint has no meta.dataset_meta.classes and the config '
            'has no metainfo.classes'
        )
    if num is not None and len(classes) != num:
        raise ValueError(
            f'{len(classes)} class names ({classes}) but the head predicts {num} classes'
        )
    return [str(c) for c in classes]


@dataclass
class LoadedModel:
    """The model plus everything derived from its config, shared by both services."""

    cfg: Config
    model: nn.Module
    classes: list[str]
    config_path: str
    weights_path: str
    weights_digest: str
    model_name: str
    device: str
    precision: str
    input_size: tuple[int, int]  # (w, h), model px
    input_size_source: str
    resize_mode: str
    pad_mode: str
    pad_divisor: int
    pipeline: Compose
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def input_w(self) -> int:
        return self.input_size[0]

    @property
    def input_h(self) -> int:
        return self.input_size[1]

    @contextlib.contextmanager
    def autocast(self) -> Iterator[None]:
        dev = torch.device(self.device)
        if self.precision == 'fp32' or dev.type != 'cuda':
            yield
            return
        dtype = torch.float16 if self.precision == 'fp16' else torch.bfloat16
        with torch.autocast(device_type='cuda', dtype=dtype):
            yield

    def window_image(self, img: np.ndarray) -> np.ndarray:
        """Pad one window (at most the input size) as ``--pad-mode`` says."""
        if self.pad_mode == 'square':
            return tiling.pad_to(img, self.input_h, self.input_w)
        return img

    def collate(self, images: list[np.ndarray]) -> dict:
        """BGR uint8 windows -> the ``test_step`` input dict (pipeline + pseudo_collate)."""
        samples = [
            # scale_factor 1: no Resize in the pipeline (a config Resize overwrites it)
            self.pipeline({'img': im, 'img_id': i, 'img_path': '', 'scale_factor': (1.0, 1.0)})
            for i, im in enumerate(images)
        ]
        return pseudo_collate(samples)

    def preprocess(self, images: list[np.ndarray]) -> torch.Tensor:
        """BGR uint8 windows -> the normalised, padded batch tensor the backbone sees."""
        data = self.model.data_preprocessor(self.collate(images), False)
        return data['inputs']

    @contextlib.contextmanager
    def nms_override(self, iou: float | None, class_agnostic: bool | None) -> Iterator[None]:
        """Temporarily set the NMS IoU / class-agnostic flag of the model's ``test_cfg``.

        Only safe under :attr:`lock`, which every caller holds for the whole batch.
        """
        cfgs = []
        for holder in (getattr(self.model, 'bbox_head', None), self.model):
            test_cfg = getattr(holder, 'test_cfg', None)
            if test_cfg is not None and test_cfg.get('nms') is not None:
                if all(c is not test_cfg.nms for c in cfgs):
                    cfgs.append(test_cfg.nms)
        saved = [dict(c) for c in cfgs]
        try:
            for c in cfgs:
                if iou is not None:
                    c['iou_threshold'] = float(iou)
                if class_agnostic is not None:
                    c['class_agnostic'] = bool(class_agnostic)
            yield
        finally:
            for c, old in zip(cfgs, saved, strict=True):
                c.clear()
                c.update(old)

    def nms_cfg(self, iou: float | None = None, class_agnostic: bool | None = None) -> dict:
        """The NMS config used to merge windows (the model's own, with the overrides)."""
        test_cfg = getattr(getattr(self.model, 'bbox_head', None), 'test_cfg', None)
        if test_cfg is None:
            test_cfg = getattr(self.model, 'test_cfg', None) or {}
        nms = dict(test_cfg.get('nms') or {'type': 'nms_rotated', 'iou_threshold': 0.1})
        if iou is not None:
            nms['iou_threshold'] = float(iou)
        if class_agnostic is not None:
            nms['class_agnostic'] = bool(class_agnostic)
        return nms

    def config_nms_iou(self) -> float | None:
        value = self.nms_cfg().get('iou_threshold')
        return None if value is None else float(value)


def load(
    config: str,
    weights: str | None,
    *,
    device: str = 'cuda:0',
    precision: str = 'fp32',
    model_name: str | None = None,
    input_size: tuple[int, int] | None = None,
    resize_mode: str = 'none',
    pad_mode: str = 'square',
) -> LoadedModel:
    """Load the model and derive and validate the image-size settings.

    ``weights`` None or empty builds a randomly initialised model (smoke tests only).
    """
    weights = weights or None
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError(f'--device {device} requested but CUDA is not available')
    cfg = Config.fromfile(config)
    model, meta = build_model(cfg, weights, device)
    classes = resolve_classes(model, cfg, meta)
    model.dataset_meta = {'classes': tuple(classes)}

    if input_size is None:
        input_size, source = tiling.detect_input_size(cfg)
    else:
        source = 'flag'
    divisor = tiling.pad_size_divisor(cfg)
    for side, what in ((input_size[0], 'input width'), (input_size[1], 'input height')):
        tiling.check_divisible(
            side, divisor, what, "the data preprocessor's pad_size_divisor; set --input-size"
        )
    init_default_scope(cfg.get('default_scope', 'mmrotate'))
    pipeline = Compose(tiling.inference_pipeline_cfg(cfg, resize_mode))
    lm = LoadedModel(
        cfg=cfg,
        model=model,
        classes=classes,
        config_path=str(config),
        weights_path=str(weights or ''),
        weights_digest=weights_digest(weights) if weights else 'random',
        model_name=model_name or Path(config).stem,
        device=device,
        precision=precision,
        input_size=input_size,
        input_size_source=source,
        resize_mode=resize_mode,
        pad_mode=pad_mode,
        pad_divisor=divisor,
        pipeline=pipeline,
    )
    log.info(
        'model %s: %d classes, input %dx%d (%s), resize_mode=%s, pad_mode=%s, device=%s',
        lm.model_name,
        len(classes),
        input_size[0],
        input_size[1],
        source,
        resize_mode,
        pad_mode,
        device,
    )
    return lm
