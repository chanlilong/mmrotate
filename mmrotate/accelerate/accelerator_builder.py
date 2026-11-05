# Copyright (c) OpenMMLab. All rights reserved.
"""Turn the ``accelerator:`` block of an mmengine config into an ``Accelerator``.

The mmengine config owns the training recipe (precision, ZeRO stage, gradient
accumulation, DDP options) so that an experiment is reproducible from its
config alone; the ``accelerate launch`` config only describes the topology
(number of machines / processes, addresses). Explicit config values win over
the launcher's ``--mixed_precision`` / ``--use_deepspeed`` flags.

Precision is always *autocast with fp32 master weights*, as with mmengine's
``AmpOptimWrapper``:

* DDP / single GPU: ``torch.autocast`` around the data preprocessor + forward
  (fp16 adds accelerate's GradScaler);
* DeepSpeed: DeepSpeed's own ``torch_autocast`` section. DeepSpeed's
  ``bf16`` / ``fp16`` modes cast the whole module (BN statistics, EMA, the fp32
  output of the data preprocessor), which detectors do not survive, so they
  are rejected;
* fp8: torchao ``Float8Linear`` for eligible ``nn.Linear`` layers inside a bf16
  autocast (DDP / single GPU only).
"""

from __future__ import annotations

import copy
import json
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

if TYPE_CHECKING:
    from accelerate import Accelerator

ACCEL_KEYS = frozenset(
    {
        'mixed_precision',
        'gradient_accumulation_steps',
        'deepspeed',
        'zero3_leaf_modules',
        'zero3_sync_bn',
        'fp8',
        'ddp_kwargs',
        'timeout_minutes',
        'eval',
        'data_order',
    }
)
MUON_OPTIMIZERS = frozenset({'MuonAdamW', 'Muon'})
ADAM_OPTIMIZERS = frozenset({'AdamW', 'Adam'})
_AMP_DTYPES = {
    'bfloat16': 'bf16',
    'bf16': 'bf16',
    'torch.bfloat16': 'bf16',
    'float16': 'fp16',
    'fp16': 'fp16',
    'half': 'fp16',
    'torch.float16': 'fp16',
}


@dataclass
class AccelerateSettings:
    """Resolved ``accelerator:`` block."""

    mixed_precision: str = 'no'  # no | bf16 | fp16 | fp8
    gradient_accumulation_steps: int = 1
    deepspeed: dict[str, Any] | None = None
    zero3_leaf_modules: list[str] = field(default_factory=list)
    zero3_sync_bn: str = 'revert'  # revert | keep
    fp8: dict[str, Any] = field(default_factory=dict)
    ddp_kwargs: dict[str, Any] = field(default_factory=dict)
    timeout_minutes: float = 60.0
    data_order: str = 'auto'  # auto | always | never | off

    @property
    def use_deepspeed(self) -> bool:
        return self.deepspeed is not None

    @property
    def zero_stage(self) -> int:
        if self.deepspeed is None:
            return -1
        return int(self.deepspeed.get('zero_optimization', {}).get('stage', 0))

    @property
    def offload_optimizer(self) -> bool:
        off = (self.deepspeed or {}).get('zero_optimization', {}).get('offload_optimizer') or {}
        return str(off.get('device', 'none')).lower() in ('cpu', 'nvme')

    @property
    def autocast_dtype(self) -> torch.dtype | None:
        """The autocast dtype the optim wrapper applies itself (None: none / DeepSpeed's)."""
        if self.use_deepspeed:
            return None
        return {'bf16': torch.bfloat16, 'fp8': torch.bfloat16, 'fp16': torch.float16}.get(
            self.mixed_precision
        )


def _type_name(t: object) -> str:
    if isinstance(t, type):
        return t.__name__
    return str(t).rsplit('.', 1)[-1]


def infer_mixed_precision(optim_wrapper_cfg: Mapping[str, Any] | None) -> str:
    """``AmpOptimWrapper`` + ``dtype`` -> ``bf16`` / ``fp16``, anything else -> ``no``."""
    if not optim_wrapper_cfg:
        return 'no'
    if _type_name(optim_wrapper_cfg.get('type', 'OptimWrapper')) != 'AmpOptimWrapper':
        return 'no'
    dtype = optim_wrapper_cfg.get('dtype', None)
    if dtype is None:
        return 'fp16'  # AmpOptimWrapper's default
    return _AMP_DTYPES.get(str(dtype), 'fp16')


def optimizer_type(optim_wrapper_cfg: Mapping[str, Any] | None) -> str:
    opt = (optim_wrapper_cfg or {}).get('optimizer', None) or {}
    if isinstance(opt, Mapping):
        return _type_name(opt.get('type', ''))
    return type(opt).__name__


def _is_auto(value: object) -> bool:
    return value is None or value == 'auto'


def build_deepspeed_config(
    ds_cfg: Mapping[str, Any] | str,
    *,
    micro_batch_size: int | None,
    gradient_accumulation_steps: int,
    clip_grad: Mapping[str, Any] | None,
    mixed_precision: str,
    optimizer_name: str,
) -> dict[str, Any]:
    """Validate a DeepSpeed config and fill what mmengine already knows.

    Fills ``train_micro_batch_size_per_gpu`` (dataloader batch size; the
    dataloaders are not handed to accelerate, so it cannot infer it),
    ``gradient_accumulation_steps``, ``gradient_clipping`` (from
    ``optim_wrapper.clip_grad``), ``torch_autocast`` (from the precision) and
    the ZeRO-3 consolidated-weights flag used to write ``.pth`` checkpoints.
    """
    if isinstance(ds_cfg, str):
        with open(ds_cfg) as f:
            ds = json.load(f)
    else:
        ds = copy.deepcopy(dict(ds_cfg))
    ds = json.loads(json.dumps(ds))  # mmengine ConfigDict -> plain dicts

    for key in ('optimizer', 'scheduler'):
        if key in ds:
            raise ValueError(
                f'DeepSpeed config: remove "{key}"; mmengine builds the optimizer '
                '(optim_wrapper) and the schedulers (param_scheduler).'
            )
    for key in ('bf16', 'fp16'):
        section = ds.get(key)
        if section is None:
            continue
        enabled = section.get('enabled', False) if isinstance(section, dict) else section
        if enabled is True or enabled == 'true':
            raise ValueError(
                f'DeepSpeed config: "{key}.enabled" casts the whole model to {key}, which breaks '
                'BN statistics, EMA and the fp32 data preprocessor of mmdet detectors. Remove it '
                'and set accelerator.mixed_precision instead (DeepSpeed then runs torch_autocast '
                'with fp32 master weights).'
            )
        ds.pop(key)  # 'auto' / false entries (e.g. from accelerate's launcher)

    zero = ds.setdefault('zero_optimization', {})
    stage = int(zero.get('stage', 0))
    if stage not in (0, 1, 2, 3):
        raise ValueError(f'DeepSpeed ZeRO stage must be 0-3, got {stage}')
    zero['stage'] = stage

    if mixed_precision == 'fp8':
        raise ValueError(
            'fp8 with DeepSpeed needs TransformerEngine / MS-AMP, which this integration does not '
            'use. Use fp8 with DDP (no accelerator.deepspeed), or bf16 with DeepSpeed.'
        )
    if mixed_precision == 'fp16' and stage > 0:
        raise ValueError(
            'fp16 + DeepSpeed ZeRO has no loss scaling with torch_autocast (DeepSpeed only scales '
            'for ZeRO stage 0). Use mixed_precision: bf16.'
        )
    if optimizer_name in MUON_OPTIMIZERS and stage > 0:
        raise ValueError(
            f'{optimizer_name} cannot run under DeepSpeed ZeRO stage {stage}: ZeRO flattens and '
            'partitions the parameters into 1-D shards, but Muon orthogonalises each full weight '
            'matrix. Train Muon with DDP (remove accelerator.deepspeed) or ZeRO stage 0.'
        )
    off = zero.get('offload_optimizer') or {}
    if str(off.get('device', 'none')).lower() in ('cpu', 'nvme'):
        if stage == 0:
            raise ValueError('offload_optimizer needs ZeRO stage >= 1')
        if optimizer_name not in ADAM_OPTIMIZERS:
            raise ValueError(
                f'offload_optimizer swaps the optimizer for DeepSpeedCPUAdam; {optimizer_name} '
                'cannot be swapped. Use AdamW / Adam or disable the offload.'
            )

    if _is_auto(ds.get('train_micro_batch_size_per_gpu')):
        if micro_batch_size is None:
            raise ValueError('set train_micro_batch_size_per_gpu in the DeepSpeed config')
        ds['train_micro_batch_size_per_gpu'] = int(micro_batch_size)
    if _is_auto(ds.get('gradient_accumulation_steps')):
        ds['gradient_accumulation_steps'] = int(gradient_accumulation_steps)
    elif int(ds['gradient_accumulation_steps']) != int(gradient_accumulation_steps):
        raise ValueError(
            f'DeepSpeed gradient_accumulation_steps={ds["gradient_accumulation_steps"]} differs '
            f'from accelerator.gradient_accumulation_steps={gradient_accumulation_steps}'
        )
    if _is_auto(ds.get('train_batch_size')):
        ds.pop('train_batch_size', None)  # accelerate derives it from the world size
    if _is_auto(ds.get('gradient_clipping')):
        max_norm = 0.0
        if clip_grad:
            if clip_grad.get('type', 'norm') != 'norm' or float(clip_grad.get('norm_type', 2)) != 2:
                raise ValueError(
                    'DeepSpeed only clips by the global L2 norm; clip_grad must be '
                    f'{{max_norm: x, norm_type: 2}}, got {dict(clip_grad)}'
                )
            max_norm = float(clip_grad['max_norm'])
        ds['gradient_clipping'] = max_norm

    if mixed_precision in ('bf16', 'fp16') and 'torch_autocast' not in ds:
        ds['torch_autocast'] = {
            'enabled': True,
            'dtype': 'bfloat16' if mixed_precision == 'bf16' else 'float16',
        }
    if stage == 3:
        zero.setdefault('stage3_gather_16bit_weights_on_model_save', True)
    ds['zero_allow_untested_optimizer'] = True
    ds.setdefault('steps_per_print', 10**9)  # mmengine's LoggerHook does the logging
    ds.setdefault('wall_clock_breakdown', False)
    return ds


def resolve_settings(cfg: Mapping[str, Any]) -> AccelerateSettings:
    """Resolve the ``accelerator:`` block of a full mmengine config."""
    acc = dict(cfg.get('accelerator', None) or {})
    unknown = set(acc) - ACCEL_KEYS
    if unknown:
        raise KeyError(f'unknown accelerator keys {sorted(unknown)}; valid: {sorted(ACCEL_KEYS)}')
    optim_cfg = cfg.get('optim_wrapper', None) or {}

    mp = acc.get('mixed_precision', None)
    if mp is None:
        mp = infer_mixed_precision(optim_cfg)
        env = os.environ.get('ACCELERATE_MIXED_PRECISION', 'no')
        if mp == 'no' and env in ('bf16', 'fp16', 'fp8'):
            mp = env  # accelerate launch --mixed_precision, when the config says nothing
    mp = str(mp)
    if mp not in ('no', 'bf16', 'fp16', 'fp8'):
        raise ValueError(f'accelerator.mixed_precision must be no / bf16 / fp16 / fp8, got {mp}')

    ga = int(acc.get('gradient_accumulation_steps', 1))
    if ga < 1:
        raise ValueError('gradient_accumulation_steps must be >= 1')
    if int(optim_cfg.get('accumulative_counts', 1) or 1) > 1:
        if 'gradient_accumulation_steps' in acc:
            raise ValueError(
                'set either optim_wrapper.accumulative_counts or '
                'accelerator.gradient_accumulation_steps, not both'
            )
        ga = int(optim_cfg['accumulative_counts'])

    ds_cfg = acc.get('deepspeed', None)
    if ds_cfg is None and os.environ.get('ACCELERATE_USE_DEEPSPEED', 'false').lower() == 'true':
        # `accelerate launch --use_deepspeed ...`: take its config through the same checks.
        from accelerate.utils import DeepSpeedPlugin

        ds_cfg = DeepSpeedPlugin().deepspeed_config
    deepspeed = None
    if ds_cfg is not None:
        train_dl = cfg.get('train_dataloader', None) or {}
        deepspeed = build_deepspeed_config(
            ds_cfg,
            micro_batch_size=train_dl.get('batch_size', None),
            gradient_accumulation_steps=ga,
            clip_grad=optim_cfg.get('clip_grad', None),
            mixed_precision=mp,
            optimizer_name=optimizer_type(optim_cfg),
        )

    ddp_kwargs = dict(acc.get('ddp_kwargs', None) or {})
    wrapper_cfg = cfg.get('model_wrapper_cfg', None) or {}
    ddp_kwargs.setdefault(
        'find_unused_parameters',
        bool(wrapper_cfg.get('find_unused_parameters', cfg.get('find_unused_parameters', False))),
    )
    ddp_kwargs.setdefault('broadcast_buffers', bool(wrapper_cfg.get('broadcast_buffers', False)))

    data_order = str(acc.get('data_order', 'auto'))
    if data_order not in ('auto', 'always', 'never', 'off'):
        raise ValueError('accelerator.data_order must be auto / always / never / off')
    leaf = acc.get('zero3_leaf_modules', None) or []
    if leaf and (deepspeed is None or int(deepspeed['zero_optimization']['stage']) != 3):
        leaf = []  # only meaningful for ZeRO-3
    zero3_sync_bn = str(acc.get('zero3_sync_bn', 'revert'))
    if zero3_sync_bn not in ('revert', 'keep'):
        raise ValueError('accelerator.zero3_sync_bn must be revert / keep')
    return AccelerateSettings(
        mixed_precision=mp,
        gradient_accumulation_steps=ga,
        deepspeed=deepspeed,
        zero3_leaf_modules=[str(x) for x in leaf],
        zero3_sync_bn=zero3_sync_bn,
        fp8=dict(acc.get('fp8', None) or {}),
        ddp_kwargs=ddp_kwargs,
        timeout_minutes=float(acc.get('timeout_minutes', 60)),
        data_order=data_order,
    )


def fp8_module_filter(exclude: list[str], min_dim: int = 16) -> Callable[[nn.Module, str], bool]:
    """torchao filter: ``nn.Linear`` with both dims divisible by 16, not matching ``exclude``."""
    patterns = [re.compile(p) for p in exclude]

    def keep(module: nn.Module, fqn: str) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        if module.in_features % 16 or module.out_features % 16:
            return False
        if min(module.in_features, module.out_features) < min_dim:
            return False
        return not any(p.search(fqn) for p in patterns)

    return keep


def single_process_rendezvous() -> None:
    """Env for a 1-process group when no launcher set one (``python train.py`` + DeepSpeed).

    DeepSpeed always needs ``torch.distributed``; without RANK / WORLD_SIZE / MASTER_ADDR it
    falls back to MPI discovery (mpi4py), which the plain single-GPU case should not need.
    """
    if 'RANK' in os.environ or 'OMPI_COMM_WORLD_RANK' in os.environ:
        return
    import socket

    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    os.environ.update(
        RANK='0',
        LOCAL_RANK='0',
        WORLD_SIZE='1',
        LOCAL_WORLD_SIZE='1',
        MASTER_ADDR='127.0.0.1',
        MASTER_PORT=os.environ.get('MASTER_PORT', str(port)),
    )


def build_accelerator(settings: AccelerateSettings) -> Accelerator:
    """Create the ``Accelerator``; it initialises ``torch.distributed`` when multi-process."""
    from accelerate import Accelerator
    from accelerate.utils import (
        DeepSpeedPlugin,
        DistributedDataParallelKwargs,
        InitProcessGroupKwargs,
    )

    handlers: list[Any] = [
        DistributedDataParallelKwargs(**settings.ddp_kwargs),
        InitProcessGroupKwargs(timeout=timedelta(minutes=settings.timeout_minutes)),
    ]
    mixed_precision = settings.mixed_precision
    if mixed_precision == 'fp8':
        from accelerate.utils import AORecipeKwargs
        from torchao.float8 import Float8LinearConfig

        fp8 = dict(settings.fp8)
        recipe = fp8.pop('recipe', 'tensorwise')
        exclude = list(fp8.pop('exclude', []))
        min_dim = int(fp8.pop('min_dim', 16))
        if fp8:
            raise KeyError(f'unknown accelerator.fp8 keys {sorted(fp8)}')
        handlers.append(
            AORecipeKwargs(
                config=Float8LinearConfig.from_recipe_name(recipe),
                module_filter_func=fp8_module_filter(exclude, min_dim),
            )
        )
    plugin = None
    if settings.deepspeed is not None:
        single_process_rendezvous()
        plugin = DeepSpeedPlugin(hf_ds_config=settings.deepspeed, zero3_init_flag=False)
        mixed_precision = 'no'  # DeepSpeed's torch_autocast section handles it
    return Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        deepspeed_plugin=plugin,
        kwargs_handlers=handlers,
        step_scheduler_with_optimizer=False,
    )
