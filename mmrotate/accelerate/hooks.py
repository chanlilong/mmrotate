# Copyright (c) OpenMMLab. All rights reserved.
"""Hooks for :class:`AccelerateRunner`."""

from __future__ import annotations

import itertools

import torch
from mmengine.hooks import DistSamplerSeedHook, EMAHook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner

from mmrotate.registry import HOOKS


@HOOKS.register_module()
class AccelerateDistSamplerSeedHook(DistSamplerSeedHook):
    """Alias of mmengine's :class:`DistSamplerSeedHook`, kept for older configs.

    :class:`AccelerateRunner` never hands the dataloaders to accelerate, so the
    stock hook already finds the ``DefaultSampler`` and calls ``set_epoch`` on
    it; nothing else is needed.
    """


@HOOKS.register_module()
class ZeRO3EMAHook(EMAHook):
    """:class:`EMAHook` for DeepSpeed ZeRO-3, where parameters are partitioned.

    :class:`AccelerateRunner` swaps it in for ``EMAHook`` automatically under
    ZeRO-3; the constructor arguments are the same. Differences from the
    stock hook:

    * The EMA copy is made in ``before_run``, before the model is partitioned,
      so it holds the full weights.
    * Each update gathers only the trainable parameters (frozen ones never
      change, so their EMA equals the initial copy) and the buffers, which
      ZeRO does not partition.
    * Evaluation runs the EMA model directly (``AccelerateModelWrapper``'s
      eval module) instead of swapping weights into the partitioned model,
      which has the same effect.
    * Checkpoints: the runner gathers a full ``state_dict`` before
      ``before_save_checkpoint``, so the stock state-dict swap works as is.
    """

    def before_train(self, runner: Runner) -> None:
        super().before_train(runner)
        names = [n for n, p in self.src_model.named_parameters() if p.requires_grad]
        if self.ema_model.update_buffers:
            names += [n for n, _ in self.src_model.named_buffers()]
        # Non-persistent buffers (e.g. the data preprocessor's mean / std) are not averaged.
        self._dynamic = [n for n in names if n in self.ema_model.avg_parameters]
        self.ema_model.module.eval()
        if is_model_wrapper(runner.model) and hasattr(runner.model, 'set_eval_module'):
            runner.model.set_eval_module(self.ema_model.module)

    @torch.no_grad()
    def after_train_iter(
        self,
        runner: Runner,
        batch_idx: int,
        data_batch: dict | tuple | list | None = None,
        outputs: dict | None = None,
    ) -> None:
        import deepspeed

        trainable = [p for p in self.src_model.parameters() if p.requires_grad]
        with deepspeed.zero.GatheredParameters(trainable, modifier_rank=None):
            src = dict(
                itertools.chain(self.src_model.named_parameters(), self.src_model.named_buffers())
            )
            avg = self.ema_model.avg_parameters
            started = self._ema_started(runner)
            ema = self.ema_model
            for k in self._dynamic:
                p_avg, p_src = avg[k], src[k]  # avg_parameters: the EMA module's own keys
                if not started or ema.steps == 0:
                    p_avg.data.copy_(p_src.data)
                elif ema.steps % ema.interval == 0 and p_avg.dtype.is_floating_point:
                    ema.avg_func(p_avg.data, p_src.data.to(p_avg.device), ema.steps)
            if started:
                ema.steps += 1

    def before_val_epoch(self, runner: Runner) -> None:
        """No swap: the wrapper evaluates the EMA model directly."""

    def after_val_epoch(self, runner: Runner, metrics: dict | None = None) -> None:
        """No swap: the wrapper evaluates the EMA model directly."""

    def before_test_epoch(self, runner: Runner) -> None:
        """No swap: the wrapper evaluates the EMA model directly."""

    def after_test_epoch(self, runner: Runner, metrics: dict | None = None) -> None:
        """No swap: the wrapper evaluates the EMA model directly."""
