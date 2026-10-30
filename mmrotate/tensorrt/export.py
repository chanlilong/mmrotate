# Copyright (c) OpenMMLab. All rights reserved.
"""Torch model -> ONNX for the TensorRT engine of a single-stage detector.

The engine computes ``detector._forward(inputs)`` (backbone, neck and
``bbox_head.forward``). Its nested tuple output (e.g. ``(cls_scores,
bbox_preds, angle_preds)`` with one tensor per level) is flattened to named
ONNX outputs, and the nesting is stored as a JSON-able spec so the runtime can
rebuild it and call the unchanged ``bbox_head.predict_by_feat``.
"""

import copy
import warnings
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path

import torch
from mmdet.models.detectors import SingleStageDetector
from mmengine.config import Config
from mmengine.model import revert_sync_batchnorm
from mmengine.runner import load_checkpoint
from torch import Tensor, nn

from mmrotate.registry import MODELS
from mmrotate.utils import register_all_modules

# Nesting of the engine outputs: ``None`` is a tensor leaf, a list a tuple.
OutputSpec = list | None
INPUT_NAME = 'inputs'


def flatten_outputs(outputs: object) -> tuple[list[Tensor], OutputSpec]:
    """Flatten nested tuples/lists of tensors; returns the leaves and spec."""
    if isinstance(outputs, Tensor):
        return [outputs], None
    if not isinstance(outputs, list | tuple):
        raise TypeError(f'Cannot export a {type(outputs).__name__} output')
    leaves: list[Tensor] = []
    spec = []
    for item in outputs:
        item_leaves, item_spec = flatten_outputs(item)
        leaves += item_leaves
        spec.append(item_spec)
    return leaves, spec


def unflatten_outputs(leaves: Sequence[Tensor], spec: OutputSpec) -> object:
    """Inverse of :func:`flatten_outputs` (tuples for every list)."""
    it = iter(leaves)

    def build(node: OutputSpec) -> object:
        if node is None:
            return next(it)
        return tuple(build(child) for child in node)

    out = build(spec)
    if next(it, None) is not None:
        raise ValueError('More tensors than the output spec describes')
    return out


def output_names(spec: OutputSpec, prefix: str = 'out') -> list[str]:
    """Names of the flattened outputs, e.g. ``out_0_2`` for ``outs[0][2]``."""
    if spec is None:
        return [prefix]
    names = []
    for i, child in enumerate(spec):
        names += output_names(child, f'{prefix}_{i}')
    return names


class ExportWrapper(nn.Module):
    """``detector._forward`` with flat outputs, for ``torch.onnx.export``."""

    def __init__(self, detector: SingleStageDetector) -> None:
        super().__init__()
        self.detector = detector

    def forward(self, inputs: Tensor) -> tuple[Tensor, ...]:
        leaves, _ = flatten_outputs(self.detector._forward(inputs))
        return tuple(leaves)


def check_single_stage(model: nn.Module) -> None:
    """Raise unless ``model`` is a :class:`SingleStageDetector`."""
    if not isinstance(model, SingleStageDetector):
        raise TypeError(
            f'TensorRT export supports SingleStageDetector subclasses only, '
            f'got {type(model).__name__}'
        )


def build_detector(
    cfg: Config, checkpoint: str | None, device: str = 'cuda'
) -> SingleStageDetector:
    """Build ``cfg.model`` in eval mode and load ``checkpoint`` into it.

    The backbone's own pretrained weights are not loaded (``checkpoint`` /
    ``init_cfg`` are cleared): everything comes from the detector checkpoint.
    """
    register_all_modules()
    model_cfg = copy.deepcopy(cfg.model)
    backbone = model_cfg.get('backbone')
    if isinstance(backbone, dict):
        if 'checkpoint' in backbone:
            backbone.checkpoint = None
        backbone.pop('init_cfg', None)
    model = MODELS.build(model_cfg)
    check_single_stage(model)
    if checkpoint is not None:
        ckpt = load_checkpoint(model, checkpoint, map_location='cpu', strict=False)
        meta = ckpt.get('meta', {})
        if 'dataset_meta' in meta:
            model.dataset_meta = meta['dataset_meta']
    else:
        warnings.warn('No checkpoint given: exporting randomly initialised weights', stacklevel=2)
    model = revert_sync_batchnorm(model)
    return model.to(device).eval()


def infer_input_shape(cfg: Config) -> tuple[int, int]:
    """Network input ``(H, W)`` from the test pipeline (``Pad`` or ``Resize``)."""
    pipeline = cfg.test_dataloader.dataset.get('pipeline', [])
    size = None
    for step in pipeline:
        kind = str(step.get('type', '')).split('.')[-1]
        if kind == 'Pad' and step.get('size') is not None:
            size = step['size']  # (w, h)
        elif kind == 'Resize' and size is None and step.get('scale') is not None:
            size = step['scale']  # (w, h); keep_ratio fits inside this box
    if size is None:
        raise ValueError('Cannot infer the input size from the test pipeline, pass --input-shape')
    if isinstance(size, int):
        size = (size, size)
    w, h = size
    divisor = cfg.model.get('data_preprocessor', {}).get('pad_size_divisor', 1) or 1
    return -(-int(h) // divisor) * divisor, -(-int(w) // divisor) * divisor


@torch.no_grad()
def export_onnx(
    model: SingleStageDetector,
    onnx_file: str | Path,
    input_shape: tuple[int, int, int, int],
    opset: int = 19,
    autocast_dtype: torch.dtype | None = None,
) -> tuple[OutputSpec, list[str]]:
    """Export ``model._forward`` with a static input shape.

    Call inside :func:`~mmrotate.tensorrt.export_patches.export_mode`. Uses
    the TorchScript exporter: the backbones' shape-dependent Python logic
    (padding, position-embedding interpolation, deformable reference points)
    is folded into constants for the given shape.

    Args:
        autocast_dtype: Trace under ``torch.autocast`` with this dtype (mixed
            precision graph), ``None`` for fp32.

    Returns:
        The output nesting spec and the ONNX output names.
    """
    check_single_stage(model)
    device = next(model.parameters()).device
    dummy = torch.randn(input_shape, device=device)
    autocast = (
        torch.autocast(device.type, dtype=autocast_dtype) if autocast_dtype else nullcontext()
    )
    with autocast:
        _, spec = flatten_outputs(model._forward(dummy))
    names = output_names(spec)
    onnx_file = Path(onnx_file)
    onnx_file.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings(), autocast:
        warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
        torch.onnx.export(
            ExportWrapper(model).train(model.training),  # export restores this mode
            (dummy,),
            str(onnx_file),
            dynamo=False,
            input_names=[INPUT_NAME],
            output_names=names,
            opset_version=opset,
            do_constant_folding=True,
        )
    return spec, names
