# Copyright (c) OpenMMLab. All rights reserved.
"""``OptimWrapper`` that routes backward / clipping / stepping through accelerate."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import torch
from accelerate import Accelerator
from mmengine.optim import OptimWrapper
from mmengine.registry import OPTIM_WRAPPERS  # mmrotate has no child registry for these
from torch import nn


@OPTIM_WRAPPERS.register_module()
class AccelerateOptimWrapper(OptimWrapper):
    """Optimizer wrapper for :class:`AccelerateRunner`.

    Built from the config's ``optim_wrapper`` (``OptimWrapper`` /
    ``AmpOptimWrapper`` are converted automatically, so custom constructors
    such as ``MuonAdamWOptimWrapperConstructor`` keep working) and bound to the
    ``Accelerator`` by the runner once the model is prepared.

    * ``optim_context``: ``accelerator.accumulate`` (gradient sync only on the
      last micro-batch) + ``torch.autocast`` (not under DeepSpeed, which
      autocasts itself), wrapped around the data preprocessor and forward,
      exactly where ``AmpOptimWrapper`` autocasts.
    * ``update_params``: ``accelerator.backward`` (DeepSpeed: engine backward
      + step; fp16: GradScaler), then on sync steps clip, step and zero_grad.
    * Gradient accumulation is accelerate's (``accelerator.gradient_accumulation_steps``),
      so ``accumulative_counts`` is always 1 here.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        accumulative_counts: int = 1,
        clip_grad: dict | None = None,
    ) -> None:
        if accumulative_counts != 1:
            raise ValueError(
                'AccelerateOptimWrapper: use accelerator.gradient_accumulation_steps for '
                'gradient accumulation'
            )
        super().__init__(optimizer, accumulative_counts=1, clip_grad=clip_grad)
        self.accelerator: Accelerator | None = None
        self.autocast_dtype: torch.dtype | None = None
        self.is_deepspeed = False

    def bind(
        self, accelerator: Accelerator, autocast_dtype: torch.dtype | None, is_deepspeed: bool
    ) -> None:
        """Attach the accelerator (called by the runner after ``prepare``)."""
        self.accelerator = accelerator
        self.autocast_dtype = autocast_dtype
        self.is_deepspeed = is_deepspeed

    def _require_bound(self) -> Accelerator:
        if self.accelerator is None:
            raise RuntimeError('AccelerateOptimWrapper is used before AccelerateRunner bound it')
        return self.accelerator

    @contextlib.contextmanager
    def optim_context(self, model: nn.Module) -> Iterator[None]:
        accelerator = self._require_bound()
        with contextlib.ExitStack() as stack:
            stack.enter_context(accelerator.accumulate(model))
            if self.autocast_dtype is not None:
                stack.enter_context(
                    torch.autocast(device_type=accelerator.device.type, dtype=self.autocast_dtype)
                )
            yield

    def update_params(  # type: ignore[override]
        self,
        loss: torch.Tensor,
        step_kwargs: dict | None = None,
        zero_kwargs: dict | None = None,
    ) -> None:
        self.backward(loss)
        if self._require_bound().sync_gradients:
            self.step(**(step_kwargs or {}))
            self.zero_grad(**(zero_kwargs or {}))

    def backward(self, loss: torch.Tensor, **kwargs: object) -> None:
        self._require_bound().backward(loss, **kwargs)
        self._inner_count += 1

    def step(self, **kwargs: object) -> None:
        accelerator = self._require_bound()
        if self.clip_grad_kwargs and not self.is_deepspeed:
            params = [p for g in self.optimizer.param_groups for p in g['params']]
            params = [p for p in params if p.requires_grad and p.grad is not None]
            if params:
                # OptimWrapper.__init__ popped 'type' and picked clip_func from it.
                if self.clip_func is torch.nn.utils.clip_grad_value_:
                    accelerator.clip_grad_value_(params, **self.clip_grad_kwargs)
                else:
                    norm = accelerator.clip_grad_norm_(params, **self.clip_grad_kwargs)
                    if norm is not None:
                        self.message_hub.update_scalar('train/grad_norm', float(norm))
        self.optimizer.step(**kwargs)
        if self.is_deepspeed:
            # DeepSpeed clipped inside its step (gradient_clipping); report the norm it saw.
            norm = accelerator.clip_grad_norm_([], 1.0)
            if norm is not None:
                self.message_hub.update_scalar('train/grad_norm', float(norm))

    def zero_grad(self, **kwargs: object) -> None:
        self.optimizer.zero_grad(**kwargs)

    def state_dict(self) -> dict:
        """Optimizer (+ GradScaler) state; DeepSpeed keeps its sharded state in ``save_state``.

        ``base_param_settings`` (mmengine's extra group that schedulers update and
        ``base_lr`` is logged from) is kept in both cases, as ``OptimWrapper`` does.
        """
        base = getattr(self, 'base_param_settings', None)
        if self.is_deepspeed:
            return {} if base is None else {'base_param_settings': base}
        state = self.optimizer.state_dict()
        if base is not None:
            state['base_param_settings'] = base
        scaler = getattr(self.accelerator, 'scaler', None)
        if scaler is not None:
            state['loss_scaler'] = scaler.state_dict()  # same key as AmpOptimWrapper
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        state_dict = dict(state_dict)
        base = state_dict.pop('base_param_settings', None)
        if base is not None:
            self.base_param_settings = base
        if self.is_deepspeed or not state_dict:
            return
        scaler_state = state_dict.pop('loss_scaler', None)
        self.optimizer.load_state_dict(state_dict)
        scaler = getattr(self.accelerator, 'scaler', None)
        if scaler is not None and scaler_state is not None:
            scaler.load_state_dict(scaler_state)
