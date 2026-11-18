# Copyright (c) OpenMMLab. All rights reserved.
"""mmengine ``Runner`` that trains through HF accelerate (DDP / DeepSpeed ZeRO / AMP / fp8)."""

from __future__ import annotations

import contextlib
import copy
import os
import os.path as osp
import shutil
from collections.abc import Callable, Iterator, Mapping

import torch
import torch.distributed as dist
from mmengine.model import (
    convert_sync_batchnorm,
    is_model_wrapper,
    revert_sync_batchnorm,
)
from mmengine.optim import OptimWrapper
from mmengine.runner import Runner
from mmengine.runner.checkpoint import _load_checkpoint
from torch import nn
from torch.utils.data import DataLoader

from mmrotate.registry import RUNNERS

from .accelerate_optim_wrapper import AccelerateOptimWrapper
from .accelerator_builder import (
    AccelerateSettings,
    _type_name,
    build_accelerator,
    resolve_settings,
)
from .model_wrapper import AccelerateModelWrapper

_LOOP_SWAP = {
    'ValLoop': 'mmrotate.AccelerateValLoop',
    'TestLoop': 'mmrotate.AccelerateTestLoop',
}
_OPTIM_WRAPPER_TYPES = ('OptimWrapper', 'AmpOptimWrapper', 'AccelerateOptimWrapper')
_AMP_ONLY_KEYS = ('dtype', 'loss_scale', 'use_fsdp', 'accumulative_counts')


def _swap_loop(loop_cfg: object, default: str) -> object:
    """``ValLoop`` / ``TestLoop`` (or no type) -> the bridged loop; others untouched."""
    if not isinstance(loop_cfg, Mapping):
        return loop_cfg
    loop_cfg = copy.deepcopy(dict(loop_cfg))
    name = _type_name(loop_cfg.get('type', default))
    if name in _LOOP_SWAP:
        loop_cfg['type'] = _LOOP_SWAP[name]
    return loop_cfg


def _convert_optim_wrapper(cfg: object) -> object:
    if not isinstance(cfg, Mapping):
        return cfg
    cfg = copy.deepcopy(dict(cfg))
    if 'optimizer' not in cfg:
        raise NotImplementedError(
            'AccelerateRunner supports one optimizer (optim_wrapper.optimizer); multi-optimizer '
            'OptimWrapperDict configs are not supported'
        )
    name = _type_name(cfg.get('type', 'OptimWrapper'))
    if name not in _OPTIM_WRAPPER_TYPES:
        raise ValueError(
            f'AccelerateRunner replaces the optimizer wrapper; {name} is not supported '
            f'(use one of {_OPTIM_WRAPPER_TYPES})'
        )
    cfg['type'] = 'AccelerateOptimWrapper'  # registered in mmengine's root registry
    for key in _AMP_ONLY_KEYS:  # precision / accumulation live in `accelerator:` now
        cfg.pop(key, None)
    return cfg


def _swap_ema_hooks(hooks: list | None) -> list | None:
    if not hooks:
        return hooks
    out = []
    for h in hooks:
        if isinstance(h, Mapping) and _type_name(h.get('type', '')) == 'EMAHook':
            h = dict(copy.deepcopy(h), type='mmrotate.ZeRO3EMAHook')
        out.append(h)
    return out


@contextlib.contextmanager
def _state_dict_override(module: nn.Module | None, state: dict | None) -> Iterator[None]:
    """Make ``module.state_dict()`` return ``state`` (the gathered ZeRO-3 weights)."""
    if module is None or state is None:
        yield
        return
    object.__setattr__(module, 'state_dict', lambda *args, **kwargs: state)
    try:
        yield
    finally:
        del module.__dict__['state_dict']


@RUNNERS.register_module()
class AccelerateRunner(Runner):
    """``Runner`` whose model and optimizer are prepared by ``accelerate``.

    Use it through ``tools/accelerate/train.py`` (or ``runner_type:
    mmrotate.AccelerateRunner``) under ``accelerate launch``; the training
    recipe goes in the config's ``accelerator:`` block (see
    ``docs/en/user_guides/accelerate_training.md``).

    What it changes compared with ``Runner``:

    * The ``Accelerator`` is created first, so it initialises
      ``torch.distributed``; mmengine then sees an initialised process group,
      skips its own ``init_dist`` and every distributed-aware component
      (``DefaultSampler``, ``DistSamplerSeedHook``, SyncBN, logging) works
      unchanged.
    * Dataloaders are **not** passed to ``accelerator.prepare``: mmengine's
      ``DefaultSampler`` already shards by rank, so preparing them would shard
      twice. Datasets are wrapped in :class:`ConsistentOrderDataset` so every
      rank agrees on the sample order.
    * The model is not DDP-wrapped by mmengine; ``accelerator.prepare(model,
      optimizer)`` runs once the optimizer, schedulers, hooks (EMA copy) and
      weights are ready, and the result is wrapped in
      :class:`AccelerateModelWrapper`.
    * ``OptimWrapper`` / ``AmpOptimWrapper`` become
      :class:`AccelerateOptimWrapper`; ``ValLoop`` / ``TestLoop`` become the
      parquet-bridged loops; ``EMAHook`` becomes :class:`ZeRO3EMAHook` under
      ZeRO-3.
    * Checkpoints stay mmengine ``.pth`` files that the stock ``tools/test.py``
      loads (ZeRO-3 weights are gathered first). Under DeepSpeed the sharded
      optimizer state is saved next to them with ``accelerator.save_state``
      and restored on ``--resume``.

    Args:
        accelerator: The ``accelerator:`` block; read from ``cfg`` when None.
        *args, **kwargs: ``Runner`` arguments.
    """

    def __init__(self, *args: object, accelerator: dict | None = None, **kwargs: object) -> None:
        cfg = kwargs.get('cfg', None) or {}
        acc_cfg = accelerator if accelerator is not None else cfg.get('accelerator', None)
        self.accel_settings: AccelerateSettings = resolve_settings(
            {
                'accelerator': acc_cfg,
                'optim_wrapper': kwargs.get('optim_wrapper', None),
                'train_dataloader': kwargs.get('train_dataloader', None),
                'model_wrapper_cfg': kwargs.get('model_wrapper_cfg', None),
                'find_unused_parameters': cfg.get('find_unused_parameters', False),
            }
        )
        self.accelerator = build_accelerator(self.accel_settings)
        self._accel_prepared = False
        self._skip_model_weights = False
        self._loaded_meta: dict = {}

        is_dist = dist.is_available() and dist.is_initialized()
        kwargs['launcher'] = 'pytorch' if is_dist else 'none'
        if kwargs.get('optim_wrapper', None) is not None:
            kwargs['optim_wrapper'] = _convert_optim_wrapper(kwargs['optim_wrapper'])
        if kwargs.get('val_cfg', None) is not None:
            kwargs['val_cfg'] = _swap_loop(kwargs['val_cfg'], 'ValLoop')
        if kwargs.get('test_cfg', None) is not None:
            kwargs['test_cfg'] = _swap_loop(kwargs['test_cfg'], 'TestLoop')
        if self.accel_settings.zero_stage == 3:
            kwargs['custom_hooks'] = _swap_ema_hooks(kwargs.get('custom_hooks', None))
        super().__init__(*args, **kwargs)

        s = self.accel_settings
        state = self.accelerator.state
        self.logger.info(
            'AccelerateRunner: '
            f'distributed_type={state.distributed_type}, num_processes={state.num_processes}, '
            f'mixed_precision={s.mixed_precision}, '
            f'deepspeed_zero_stage={s.zero_stage if s.use_deepspeed else None}, '
            f'gradient_accumulation_steps={s.gradient_accumulation_steps}'
        )

    # ------------------------------------------------------------------ build

    def wrap_model(self, model_wrapper_cfg: dict | None, model: nn.Module) -> nn.Module:
        """Device placement and SyncBN handling only; accelerate does the wrapping."""
        if is_model_wrapper(model):
            return model
        if model_wrapper_cfg:
            self.logger.warning(
                'model_wrapper_cfg is ignored by AccelerateRunner (accelerate wraps the model); '
                'put DDP options in accelerator.ddp_kwargs'
            )
        model = model.to(self.accelerator.device)
        if not self.distributed:
            self.logger.info(
                'Single process: SyncBatchNorm (SyncBN) layers are reverted to BatchNormXd.'
            )
            return revert_sync_batchnorm(model)
        sync_bn = self.cfg.get('sync_bn', None)
        if sync_bn is not None:
            model = convert_sync_batchnorm(model, sync_bn)
        s = self.accel_settings
        if s.zero_stage == 3 and s.zero3_sync_bn == 'revert' and _has_sync_bn(model):
            # ZeRO-3 gathers each module's parameters from inside autograd during
            # backward; SyncBN's backward all-reduce runs on the same communicator.
            # Data-dependent graphs (loss terms, checkpointing) can make ranks reach
            # the two in different orders: a silent NCCL deadlock until the timeout.
            self.logger.warning(
                'ZeRO-3: SyncBatchNorm layers (including those mmdet RTMDet adds when '
                'world_size > 1) are reverted to per-GPU BatchNorm, because SyncBN collectives '
                'can deadlock against ZeRO-3 parameter gathers. Use '
                'accelerator.zero3_sync_bn=keep to override, or ZeRO-2 to keep SyncBN.'
            )
            model = revert_sync_batchnorm(model)
        return model

    def build_dataloader(  # type: ignore[override]
        self,
        dataloader: DataLoader | dict,
        seed: int | None = None,
        diff_rank_seed: bool = False,
    ) -> DataLoader:
        """Wrap the dataset in ``ConsistentOrderDataset`` (``accelerator.data_order``)."""
        order = self.accel_settings.data_order
        if isinstance(dataloader, Mapping) and order != 'off':
            dataloader = dict(dataloader)
            ds = dataloader.get('dataset', None)
            if (
                isinstance(ds, Mapping)
                and _type_name(ds.get('type', '')) != 'ConsistentOrderDataset'
            ):
                dataloader['dataset'] = dict(
                    type='mmrotate.ConsistentOrderDataset', dataset=ds, sort=order
                )
        return Runner.build_dataloader(dataloader, seed=seed, diff_rank_seed=diff_rank_seed)

    def build_optim_wrapper(self, optim_wrapper: OptimWrapper | dict) -> OptimWrapper:
        optim_wrapper = super().build_optim_wrapper(optim_wrapper)
        if not isinstance(optim_wrapper, AccelerateOptimWrapper):
            raise TypeError(
                f'AccelerateRunner needs an AccelerateOptimWrapper, got {type(optim_wrapper)}'
            )
        s = self.accel_settings
        # Bound now (not after prepare) so a resumed GradScaler state lands in accelerate's scaler.
        optim_wrapper.bind(self.accelerator, s.autocast_dtype, s.use_deepspeed)
        return optim_wrapper

    # ------------------------------------------------------------------ prepare

    def load_or_resume(self) -> None:
        """Prepare the model + optimizer around mmengine's checkpoint loading.

        ``load_from`` weights go into the raw model before ``prepare`` (ZeRO-3
        partitions it); a DeepSpeed resume needs the engine first, so it
        prepares before loading.
        """
        training = isinstance(self.optim_wrapper, OptimWrapper) and not self._accel_prepared
        if training and self.accel_settings.use_deepspeed and self._resume:
            self._prepare()
            super().load_or_resume()
            return
        super().load_or_resume()
        if training:
            self._prepare()

    def _prepare(self) -> None:
        s = self.accel_settings
        optim_wrapper = self.optim_wrapper
        assert isinstance(optim_wrapper, AccelerateOptimWrapper)
        model = self.model.module if is_model_wrapper(self.model) else self.model

        optimizer = optim_wrapper.optimizer
        if s.offload_optimizer:
            optimizer = _to_cpu_adam(optimizer)
            optim_wrapper.optimizer = optimizer
        if s.zero3_leaf_modules:
            _set_zero3_leaf_modules(model, s.zero3_leaf_modules)

        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        engine, prepared = self.accelerator.prepare(model, optimizer)
        if '_original_forward' in model.__dict__:
            # accelerate wrapped forward in autocast; the optim wrapper autocasts the
            # training step itself (like AmpOptimWrapper) and evaluation stays fp32.
            model.forward = model.__dict__.pop('_original_forward')
        optim_wrapper.optimizer = prepared
        optim_wrapper.bind(self.accelerator, s.autocast_dtype, s.use_deepspeed)
        self.model = AccelerateModelWrapper(model, engine, zero3=s.zero_stage == 3)
        self._accel_prepared = True
        n_fp8 = sum(type(m).__name__ == 'Float8Linear' for m in model.modules())
        self.logger.info(
            f'accelerate prepared {type(engine).__name__} with {type(prepared).__name__} '
            f'({n_train / 1e6:.2f}M trainable parameters'
            + (f', {n_fp8} nn.Linear -> Float8Linear' if s.mixed_precision == 'fp8' else '')
            + ')'
        )

    # ------------------------------------------------------------------ checkpoints

    def save_checkpoint(  # type: ignore[override]
        self,
        out_dir: str,
        filename: str,
        file_client_args: dict | None = None,
        save_optimizer: bool = True,
        save_param_scheduler: bool = True,
        meta: dict | None = None,
        by_epoch: bool = True,
        backend_args: dict | None = None,
    ) -> None:
        """Runs on every rank (DeepSpeed saving and ZeRO-3 gathering are collective)."""
        meta = dict(meta) if meta else {}
        wrapper = self.model if isinstance(self.model, AccelerateModelWrapper) else None
        s = self.accel_settings
        if wrapper is not None and s.use_deepspeed and save_optimizer:
            stem = osp.splitext(osp.basename(filename))[0]
            state_dir = osp.join(out_dir, 'accelerate_state', stem)
            self.accelerator.save_state(state_dir)
            meta['accelerate_state'] = osp.relpath(state_dir, out_dir)
        full_state = None
        if wrapper is not None and s.zero_stage == 3:
            full_state = self.accelerator.get_state_dict(wrapper.__dict__['engine'], unwrap=True)
        with _state_dict_override(wrapper.module if wrapper is not None else None, full_state):
            super().save_checkpoint(  # master_only
                out_dir,
                filename,
                file_client_args=file_client_args,
                save_optimizer=save_optimizer,
                save_param_scheduler=save_param_scheduler,
                meta=meta,
                by_epoch=by_epoch,
                backend_args=backend_args,
            )
        if self.accelerator.is_main_process and s.use_deepspeed:
            _prune_states(out_dir)

    def load_checkpoint(  # type: ignore[override]
        self,
        filename: str,
        map_location: str | Callable[..., object] = 'cpu',
        strict: bool = False,
        revise_keys: list | None = None,
    ) -> dict:
        if revise_keys is None:
            revise_keys = [(r'^module.', '')]
        if self._skip_model_weights:
            # ZeRO-3 resume: the weights come from the DeepSpeed state instead.
            checkpoint = _load_checkpoint(filename, map_location=map_location)
            self.call_hook('after_load_checkpoint', checkpoint=checkpoint)
            self._has_loaded = True
        else:
            checkpoint = super().load_checkpoint(filename, map_location, strict, revise_keys)
        self._loaded_meta = dict(checkpoint.get('meta', {}) or {})
        return checkpoint

    def resume(  # type: ignore[override]
        self,
        filename: str,
        resume_optimizer: bool = True,
        resume_param_scheduler: bool = True,
        map_location: str | Callable[..., object] = 'default',
    ) -> None:
        s = self.accel_settings
        deepspeed_resume = s.use_deepspeed and isinstance(self.model, AccelerateModelWrapper)
        self._skip_model_weights = deepspeed_resume and s.zero_stage == 3
        try:
            super().resume(filename, resume_optimizer, resume_param_scheduler, map_location)
        finally:
            self._skip_model_weights = False
        if not deepspeed_resume:
            return
        rel = self._loaded_meta.get('accelerate_state', None)
        state_dir = osp.join(osp.dirname(filename), rel) if rel else None
        if state_dir and osp.isdir(state_dir) and resume_optimizer:
            self.accelerator.load_state(state_dir)
            self.logger.info(f'restored the DeepSpeed state from {state_dir}')
        elif s.zero_stage == 3:
            raise RuntimeError(
                f'{filename} has no DeepSpeed state next to it ({state_dir}); a ZeRO-3 run '
                'cannot resume without it. Use load_from to start from its weights instead.'
            )
        else:
            self.logger.warning(
                f'no DeepSpeed state found for {filename}: weights resumed, optimizer state reset'
            )


def _prune_states(out_dir: str) -> None:
    """Drop DeepSpeed state dirs whose ``.pth`` the CheckpointHook has already removed."""
    root = osp.join(out_dir, 'accelerate_state')
    if not osp.isdir(root):
        return
    for stem in os.listdir(root):
        if not osp.exists(osp.join(out_dir, f'{stem}.pth')):
            shutil.rmtree(osp.join(root, stem), ignore_errors=True)


def _to_cpu_adam(optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
    """AdamW / Adam -> DeepSpeedCPUAdam with the same param groups (ZeRO optimizer offload)."""
    from deepspeed.ops.adam import DeepSpeedCPUAdam

    d = optimizer.defaults
    if d.get('amsgrad', False):
        raise ValueError('DeepSpeedCPUAdam does not support amsgrad')
    return DeepSpeedCPUAdam(
        [dict(g) for g in optimizer.param_groups],
        lr=d['lr'],
        betas=tuple(d['betas']),
        eps=d['eps'],
        weight_decay=d['weight_decay'],
        adamw_mode=isinstance(optimizer, torch.optim.AdamW),
    )


def _set_zero3_leaf_modules(model: nn.Module, names: list[str]) -> None:
    from deepspeed.utils import set_z3_leaf_modules

    classes = {type(m) for m in model.modules() if type(m).__name__ in names}
    missing = set(names) - {c.__name__ for c in classes}
    if missing:
        raise ValueError(f'zero3_leaf_modules: no module of class {sorted(missing)} in the model')
    set_z3_leaf_modules(model, list(classes))


def _has_sync_bn(model: nn.Module) -> bool:
    return any(isinstance(m, nn.SyncBatchNorm) for m in model.modules())
