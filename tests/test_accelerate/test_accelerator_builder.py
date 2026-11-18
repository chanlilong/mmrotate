# Copyright (c) OpenMMLab. All rights reserved.
"""Config resolution of the ``accelerator:`` block (no processes started)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from mmrotate.accelerate.accelerator_builder import (
    build_deepspeed_config,
    fp8_module_filter,
    infer_mixed_precision,
    resolve_settings,
)

ADAMW = {'type': 'AdamW', 'lr': 1e-3}


def cfg(**acc: object) -> dict:
    return {
        'accelerator': acc,
        'optim_wrapper': {
            'type': 'AmpOptimWrapper',
            'dtype': 'bfloat16',
            'optimizer': ADAMW,
            'clip_grad': {'max_norm': 35, 'norm_type': 2},
        },
        'train_dataloader': {'batch_size': 4},
    }


@pytest.mark.parametrize(
    ('wrapper', 'expected'),
    [
        ({'type': 'OptimWrapper'}, 'no'),
        ({'type': 'AmpOptimWrapper'}, 'fp16'),
        ({'type': 'AmpOptimWrapper', 'dtype': 'bfloat16'}, 'bf16'),
        ({'type': 'mmengine.AmpOptimWrapper', 'dtype': 'float16'}, 'fp16'),
        (None, 'no'),
    ],
)
def test_infer_mixed_precision(wrapper: dict | None, expected: str) -> None:
    assert infer_mixed_precision(wrapper) == expected


def test_defaults_follow_amp_optim_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('ACCELERATE_MIXED_PRECISION', raising=False)
    s = resolve_settings(cfg())
    assert s.mixed_precision == 'bf16' and s.autocast_dtype == torch.bfloat16
    assert not s.use_deepspeed and s.zero_stage == -1
    assert s.ddp_kwargs == {'find_unused_parameters': False, 'broadcast_buffers': False}


def test_explicit_precision_wins_over_launcher(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('ACCELERATE_MIXED_PRECISION', 'fp16')
    assert resolve_settings(cfg(mixed_precision='no')).mixed_precision == 'no'
    plain = cfg()
    plain['optim_wrapper'] = {'optimizer': ADAMW}
    assert resolve_settings(plain).mixed_precision == 'fp16'  # config silent -> launcher


def test_accumulative_counts_maps_to_accelerate() -> None:
    c = cfg()
    c['optim_wrapper']['accumulative_counts'] = 4
    assert resolve_settings(c).gradient_accumulation_steps == 4
    with pytest.raises(ValueError, match='not both'):
        resolve_settings({**c, 'accelerator': {'gradient_accumulation_steps': 2}})


def test_unknown_key() -> None:
    with pytest.raises(KeyError, match='mixed_precison'):
        resolve_settings(cfg(mixed_precison='bf16'))


@pytest.mark.parametrize('stage', [0, 1, 2, 3])
def test_deepspeed_autofill(stage: int) -> None:
    s = resolve_settings(cfg(deepspeed={'zero_optimization': {'stage': stage}}))
    ds = s.deepspeed
    assert ds is not None and s.zero_stage == stage
    assert ds['train_micro_batch_size_per_gpu'] == 4
    assert ds['gradient_accumulation_steps'] == 1
    assert ds['gradient_clipping'] == 35.0
    assert ds['torch_autocast'] == {'enabled': True, 'dtype': 'bfloat16'}
    assert 'bf16' not in ds and 'fp16' not in ds
    assert s.autocast_dtype is None  # DeepSpeed autocasts, not the optim wrapper
    if stage == 3:
        assert ds['zero_optimization']['stage3_gather_16bit_weights_on_model_save'] is True


def test_deepspeed_auto_entries_from_launcher_are_dropped() -> None:
    ds = build_deepspeed_config(
        {'bf16': {'enabled': 'auto'}, 'fp16': {'enabled': False}, 'train_batch_size': 'auto'},
        micro_batch_size=2,
        gradient_accumulation_steps=2,
        clip_grad=None,
        mixed_precision='bf16',
        optimizer_name='AdamW',
    )
    assert 'bf16' not in ds and 'fp16' not in ds and 'train_batch_size' not in ds
    assert ds['gradient_clipping'] == 0.0 and ds['gradient_accumulation_steps'] == 2


@pytest.mark.parametrize(
    ('ds', 'mp', 'opt', 'match'),
    [
        ({'bf16': {'enabled': True}}, 'bf16', 'AdamW', 'casts the whole model'),
        ({'optimizer': {'type': 'Adam'}}, 'bf16', 'AdamW', 'remove "optimizer"'),
        ({'zero_optimization': {'stage': 2}}, 'bf16', 'MuonAdamW', 'Muon'),
        ({'zero_optimization': {'stage': 1}}, 'fp16', 'AdamW', 'no loss scaling'),
        ({}, 'fp8', 'AdamW', 'TransformerEngine'),
        ({'zero_optimization': {'stage': 4}}, 'bf16', 'AdamW', 'stage must be 0-3'),
        (
            {'zero_optimization': {'stage': 2, 'offload_optimizer': {'device': 'cpu'}}},
            'bf16',
            'SGD',
            'DeepSpeedCPUAdam',
        ),
        ({'gradient_accumulation_steps': 3}, 'bf16', 'AdamW', 'differs'),
    ],
)
def test_deepspeed_guards(ds: dict, mp: str, opt: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        build_deepspeed_config(
            ds,
            micro_batch_size=2,
            gradient_accumulation_steps=1,
            clip_grad=None,
            mixed_precision=mp,
            optimizer_name=opt,
        )


def test_muon_allowed_with_zero0() -> None:
    ds = build_deepspeed_config(
        {'zero_optimization': {'stage': 0}},
        micro_batch_size=2,
        gradient_accumulation_steps=1,
        clip_grad=None,
        mixed_precision='bf16',
        optimizer_name='MuonAdamW',
    )
    assert ds['zero_optimization']['stage'] == 0


def test_clip_by_value_rejected_for_deepspeed() -> None:
    with pytest.raises(ValueError, match='L2 norm'):
        build_deepspeed_config(
            {},
            micro_batch_size=2,
            gradient_accumulation_steps=1,
            clip_grad={'type': 'value', 'clip_value': 1.0},
            mixed_precision='bf16',
            optimizer_name='AdamW',
        )


def test_fp8_filter() -> None:
    keep = fp8_module_filter(exclude=[r'\.head\.'])
    assert keep(nn.Linear(64, 128), 'backbone.fc1')
    assert not keep(nn.Linear(64, 128), 'model.head.fc')
    assert not keep(nn.Linear(60, 128), 'backbone.fc1')  # not divisible by 16
    assert not keep(nn.Conv2d(64, 64, 1), 'backbone.conv')


def test_single_process_rendezvous(monkeypatch: pytest.MonkeyPatch) -> None:
    from mmrotate.accelerate.accelerator_builder import single_process_rendezvous

    for k in ('RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT', 'LOCAL_RANK'):
        monkeypatch.delenv(k, raising=False)
    single_process_rendezvous()
    import os

    assert os.environ['RANK'] == '0' and os.environ['WORLD_SIZE'] == '1'
    assert int(os.environ['MASTER_PORT']) > 0
    monkeypatch.setenv('RANK', '3')
    monkeypatch.setenv('WORLD_SIZE', '8')
    single_process_rendezvous()  # a launcher's env is left alone
    assert os.environ['RANK'] == '3' and os.environ['WORLD_SIZE'] == '8'


def test_runner_config_rewrites() -> None:
    from mmrotate.accelerate.accelerate_runner import (
        _convert_optim_wrapper,
        _swap_ema_hooks,
        _swap_loop,
    )

    amp = {
        'type': 'AmpOptimWrapper',
        'dtype': 'bfloat16',
        'loss_scale': 'dynamic',
        'accumulative_counts': 2,
        'optimizer': ADAMW,
        'constructor': 'MuonAdamWOptimWrapperConstructor',
        'clip_grad': {'max_norm': 1},
    }
    out = _convert_optim_wrapper(amp)
    assert out == {
        'type': 'AccelerateOptimWrapper',
        'optimizer': ADAMW,
        'constructor': 'MuonAdamWOptimWrapperConstructor',
        'clip_grad': {'max_norm': 1},
    }
    with pytest.raises(ValueError, match='ApexOptimWrapper'):
        _convert_optim_wrapper({'type': 'ApexOptimWrapper', 'optimizer': ADAMW})
    with pytest.raises(NotImplementedError):
        _convert_optim_wrapper({'generator': {'optimizer': ADAMW}})

    assert _swap_loop({'type': 'ValLoop', 'fp16': True}, 'ValLoop') == {
        'type': 'mmrotate.AccelerateValLoop',
        'fp16': True,
    }
    assert _swap_loop({}, 'TestLoop') == {'type': 'mmrotate.AccelerateTestLoop'}
    assert _swap_loop({'type': 'MyLoop'}, 'ValLoop') == {'type': 'MyLoop'}

    hooks = [{'type': 'EMAHook', 'momentum': 2e-4}, {'type': 'NumClassCheckHook'}]
    assert _swap_ema_hooks(hooks) == [
        {'type': 'mmrotate.ZeRO3EMAHook', 'momentum': 2e-4},
        {'type': 'NumClassCheckHook'},
    ]


def test_zero3_sync_bn_setting() -> None:
    from mmrotate.accelerate.accelerate_runner import _has_sync_bn

    z3 = {'zero_optimization': {'stage': 3}}
    assert resolve_settings(cfg(deepspeed=z3)).zero3_sync_bn == 'revert'
    assert resolve_settings(cfg(deepspeed=z3, zero3_sync_bn='keep')).zero3_sync_bn == 'keep'
    with pytest.raises(ValueError, match='zero3_sync_bn'):
        resolve_settings(cfg(zero3_sync_bn='yes'))
    assert _has_sync_bn(nn.Sequential(nn.Conv2d(3, 4, 1), nn.SyncBatchNorm(4)))
    assert not _has_sync_bn(nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4)))


@pytest.mark.parametrize('deepspeed', [False, True])
def test_optim_wrapper_keeps_base_param_settings(deepspeed: bool) -> None:
    """Resuming restores mmengine's base-lr group (else ``base_lr`` restarts from scratch)."""
    from mmrotate.accelerate.accelerate_optim_wrapper import AccelerateOptimWrapper

    def make() -> AccelerateOptimWrapper:
        a, b = nn.Linear(2, 2), nn.Linear(2, 2)
        opt = torch.optim.SGD([{'params': a.parameters()}, {'params': b.parameters()}], lr=0.1)
        wrapper = AccelerateOptimWrapper(opt)
        wrapper.is_deepspeed = deepspeed
        return wrapper

    src = make()
    assert src.base_param_settings is not None  # two groups -> mmengine adds it
    src.base_param_settings['lr'] = 0.025  # what a scheduler would have set
    state = src.state_dict()
    assert state['base_param_settings']['lr'] == 0.025
    dst = make()
    dst.load_state_dict(state)
    assert dst.get_lr()['base_lr'] == [0.025]
