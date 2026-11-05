# Copyright (c) OpenMMLab. All rights reserved.
"""mmengine model-wrapper around the module accelerate prepared."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

from mmengine.optim import OptimWrapper
from torch import nn

from mmrotate.registry import MODEL_WRAPPERS


@MODEL_WRAPPERS.register_module()
class AccelerateModelWrapper(nn.Module):
    """Gives the prepared engine the ``train_step`` / ``val_step`` interface.

    ``accelerator.prepare`` returns a torch DDP module, a DeepSpeed engine or
    the model itself, none of which has mmengine's step methods. This wrapper
    mirrors :class:`MMDistributedDataParallel`: the training forward goes
    through the engine (gradient-sync hooks, ZeRO-3 parameter gathering,
    DeepSpeed autocast), evaluation calls the model directly.

    Being registered in ``MODEL_WRAPPERS`` makes ``is_model_wrapper`` true, so
    hooks and checkpointing use :attr:`module`, the raw model.

    Args:
        module: The raw (unwrapped) model.
        engine: What ``accelerator.prepare(module)`` returned.
        zero3: The engine is DeepSpeed ZeRO-3 (parameters are partitioned).
    """

    def __init__(self, module: nn.Module, engine: nn.Module, zero3: bool = False) -> None:
        super().__init__()
        self.module = module
        # Not registered as a submodule: it contains `module`, which would
        # otherwise appear twice in state_dict / parameters.
        self.__dict__['engine'] = engine
        self.zero3 = zero3
        # Set by ZeRO3EMAHook: the full (not partitioned) EMA model, used for evaluation
        # exactly where the stock EMAHook would swap the EMA weights in.
        self.__dict__['eval_module'] = None

    def set_eval_module(self, module: nn.Module | None) -> None:
        self.__dict__['eval_module'] = module

    @property
    def data_preprocessor(self) -> nn.Module:
        return self.module.data_preprocessor  # type: ignore[return-value]

    def train(self, mode: bool = True) -> AccelerateModelWrapper:
        super().train(mode)
        engine = self.__dict__['engine']
        if engine is not self.module:
            engine.train(mode)
        return self

    def _run_forward(self, data: dict | tuple | list, mode: str) -> dict | list:
        engine = self.__dict__['engine']
        if isinstance(data, dict):
            return engine(**data, mode=mode)
        if isinstance(data, (list, tuple)):
            return engine(*data, mode=mode)
        raise TypeError(f'Output of `data_preprocessor` should be list, tuple or dict, got {data}')

    def train_step(self, data: dict | tuple | list, optim_wrapper: OptimWrapper) -> dict:
        engine = self.__dict__['engine']
        with optim_wrapper.optim_context(engine):
            data = self.module.data_preprocessor(data, training=True)  # type: ignore[operator]
            losses = self._run_forward(data, mode='loss')
        parsed_loss, log_vars = self.module.parse_losses(losses)  # type: ignore[operator]
        optim_wrapper.update_params(parsed_loss)
        return log_vars

    def _eval_model(self) -> nn.Module:
        return self.__dict__['eval_module'] or self.module

    def val_step(self, data: dict | tuple | list) -> list:
        return self._eval_model().val_step(data)  # type: ignore[operator]

    def test_step(self, data: dict | tuple | list) -> list:
        return self._eval_model().test_step(data)  # type: ignore[operator]

    @contextlib.contextmanager
    def eval_context(self) -> Iterator[None]:
        """ZeRO-3 without an EMA model: gather the full parameters for the whole round."""
        if not self.zero3 or self.__dict__['eval_module'] is not None:
            yield
            return
        import deepspeed

        params = list(self.module.parameters())
        with deepspeed.zero.GatheredParameters(params, modifier_rank=None):
            yield

    def forward(self, *args: object, **kwargs: object) -> object:
        return self.__dict__['engine'](*args, **kwargs)
