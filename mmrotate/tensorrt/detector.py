# Copyright (c) OpenMMLab. All rights reserved.
"""Inference-only single-stage detector backed by a TensorRT engine."""

import json
import math
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from mmdet.models.detectors import SingleStageDetector
from mmdet.models.detectors.base import BaseDetector
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from torch import Tensor

from mmrotate.registry import MODELS

from .export import unflatten_outputs
from .runtime import TRTEngine

META_FILE = 'trt_meta.json'


@MODELS.register_module()
class TRTSingleStageDetector(SingleStageDetector):
    """A :class:`SingleStageDetector` whose backbone, neck and head forward
    run in a TensorRT engine built by ``tools/tensorrt/convert.py``.

    Post-processing (decoding, score filtering, NMS, rescaling) is the
    original head's ``predict_by_feat``, so ``predict`` returns the same
    ``DetDataSample`` / ``pred_instances`` as the torch model. Only the head
    is built, and its weights are unused: the engine holds all parameters.

    The engine has a static input shape. Inputs smaller than it are padded
    bottom/right with zeros (as ``DetDataPreprocessor`` pads a batch), larger
    ones raise. Batches are run in chunks of the engine batch size.

    Args:
        engine_file (str): Serialized TensorRT engine.
        bbox_head (dict): Config of the original head (for post-processing).
        meta_file (str, optional): ``trt_meta.json`` written with the engine.
            Defaults to the one next to ``engine_file``.
        data_preprocessor (dict, optional): The original data preprocessor.
        test_cfg (dict, optional): The original ``test_cfg``.
    """

    def __init__(
        self,
        engine_file: str,
        bbox_head: ConfigType,
        meta_file: str | None = None,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        # Skip SingleStageDetector.__init__: there is no backbone or neck.
        BaseDetector.__init__(self, data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.engine_file = str(engine_file)
        meta_file = meta_file or str(Path(engine_file).with_name(META_FILE))
        self.meta = json.loads(Path(meta_file).read_text())
        self.output_spec = self.meta['output_spec']
        if self.meta.get('dataset_meta'):
            self.dataset_meta = self.meta['dataset_meta']
        # Follows .to()/.cuda() so the engine is loaded on the model's device.
        self.register_buffer('_device_anchor', torch.empty(0), persistent=False)
        self._engine: TRTEngine | None = None
        self._warned_pad = False

    @property
    def engine(self) -> TRTEngine:
        """The engine, loaded on first use on the model's device."""
        device = self._device_anchor.device
        if self._engine is None or self._engine.device != device:
            if device.type != 'cuda':
                raise RuntimeError('Move the model to a CUDA device before running it')
            self._check_environment(device)
            self._engine = TRTEngine(self.engine_file, device)
        return self._engine

    def _check_environment(self, device: torch.device) -> None:
        import tensorrt as trt

        built = self.meta.get('build', {})
        cc = '.'.join(map(str, torch.cuda.get_device_capability(device)))
        if built.get('tensorrt') and built['tensorrt'] != trt.__version__:
            warnings.warn(
                f'Engine built with TensorRT {built["tensorrt"]}, running {trt.__version__}',
                stacklevel=3,
            )
        if built.get('compute_capability') and built['compute_capability'] != cc:
            warnings.warn(
                f'Engine built for compute capability {built["compute_capability"]} '
                f'({built.get("gpu")}), running on {cc}',
                stacklevel=3,
            )

    def extract_feat(self, batch_inputs: Tensor) -> tuple[Tensor]:
        """Not available: the backbone and neck are inside the engine."""
        raise NotImplementedError(
            'TRTSingleStageDetector has no separate features; use _forward / predict'
        )

    def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> dict:
        """Not available: the detector is inference only."""
        raise NotImplementedError('TRTSingleStageDetector is inference only')

    def _pad_to_engine(self, batch_inputs: Tensor) -> Tensor:
        _, c, h, w = self.engine.input_shape()
        in_c, in_h, in_w = batch_inputs.shape[1:]
        if in_c != c or in_h > h or in_w > w:
            raise ValueError(
                f'Input of shape {tuple(batch_inputs.shape[1:])} does not fit the engine '
                f'input {(c, h, w)}; convert again with a larger --input-shape'
            )
        if (in_h, in_w) != (h, w):
            if not self._warned_pad:
                warnings.warn(
                    f'Padding inputs from {(in_h, in_w)} to the engine size {(h, w)}; '
                    'results can differ slightly from the torch model at that size',
                    stacklevel=3,
                )
                self._warned_pad = True
            batch_inputs = F.pad(batch_inputs, (0, w - in_w, 0, h - in_h))
        return batch_inputs

    def _forward(self, batch_inputs: Tensor, batch_data_samples: OptSampleList = None) -> tuple:
        """Engine outputs, nested as ``bbox_head.forward`` returns them (fp32)."""
        batch_inputs = self._pad_to_engine(batch_inputs)
        engine_bs = self.engine.input_shape()[0]
        num = batch_inputs.shape[0]
        chunks: list[list[Tensor]] = []
        for i in range(math.ceil(num / engine_bs)):
            chunk = batch_inputs[i * engine_bs : (i + 1) * engine_bs]
            real = chunk.shape[0]
            if real < engine_bs:
                chunk = F.pad(chunk, (0, 0, 0, 0, 0, 0, 0, engine_bs - real))
            chunks.append([out[:real].float() for out in self.engine(chunk)])
        leaves = [
            torch.cat(outs) if len(outs) > 1 else outs[0] for outs in zip(*chunks, strict=True)
        ]
        return unflatten_outputs(leaves, self.output_spec)

    def predict(
        self, batch_inputs: Tensor, batch_data_samples: SampleList, rescale: bool = True
    ) -> SampleList:
        """Same as :meth:`SingleStageDetector.predict`, head forward in TensorRT."""
        outs = self._forward(batch_inputs)
        batch_img_metas = [data_sample.metainfo for data_sample in batch_data_samples]
        results_list = self.bbox_head.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, rescale=rescale
        )
        return self.add_pred_to_datasample(batch_data_samples, results_list)
