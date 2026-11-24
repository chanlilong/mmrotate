# Copyright (c) OpenMMLab. All rights reserved.
"""FrozenSharedExpMomentumEMA averages like ExpMomentumEMA but shares frozen parameters."""

import copy

import torch
from mmdet.models.layers import ExpMomentumEMA
from torch import nn

from mmrotate.models.layers import FrozenSharedExpMomentumEMA
from mmrotate.registry import MODELS
from mmrotate.utils import register_all_modules

EMA_KW = dict(momentum=0.0002, update_buffers=True)


def _model() -> nn.Module:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.Linear(8, 2))
    model[0].requires_grad_(False)  # the frozen "backbone"
    return model


def _train(model: nn.Module, emas: list[nn.Module], steps: int = 5) -> None:
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    for _ in range(steps):
        optimizer.zero_grad()
        model(torch.randn(16, 4)).square().mean().backward()
        optimizer.step()
        for ema in emas:
            ema.update_parameters(model)


def test_matches_exp_momentum_ema() -> None:
    register_all_modules()
    model = _model()
    frozen_before = copy.deepcopy(model[0].state_dict())
    ref = ExpMomentumEMA(model, **EMA_KW)
    ema = MODELS.build(
        dict(type='FrozenSharedExpMomentumEMA', **EMA_KW), default_args=dict(model=model)
    )
    assert isinstance(ema, FrozenSharedExpMomentumEMA)
    assert ema.shared_parameters == {'0.weight', '0.bias'}
    _train(model, [ref, ema])

    # Same keys; trainable parameters and buffers averaged exactly like ExpMomentumEMA.
    assert list(ema.state_dict()) == list(ref.state_dict())
    for key, value in ema.state_dict().items():
        if not key.startswith('module.0.'):
            assert torch.equal(value, ref.state_dict()[key]), key
    # Frozen parameters: one storage, untouched by the updates.
    for name, p in model[0].named_parameters():
        assert getattr(ema.module[0], name).data_ptr() == p.data_ptr()
        assert torch.equal(p, frozen_before[name])
    assert not any(p.requires_grad for p in ema.module.parameters())


def test_swap_and_load() -> None:
    model = _model()
    ema = FrozenSharedExpMomentumEMA(model, **EMA_KW)
    _train(model, [ema])
    frozen = model[0].weight.clone()

    # EMAHook._swap_ema_parameters: swap, validate, swap back.
    def swap() -> None:
        for p_avg, p_src in zip(ema.module.parameters(), model.parameters(), strict=True):
            tmp = p_avg.data.clone()
            p_avg.data.copy_(p_src.data)
            p_src.data.copy_(tmp)

    raw, averaged = model[2].weight.clone(), ema.module[2].weight.clone()
    swap()
    assert torch.equal(model[2].weight, averaged) and torch.equal(model[0].weight, frozen)
    swap()
    assert torch.equal(model[2].weight, raw) and torch.equal(ema.module[2].weight, averaged)

    # Resuming: a checkpoint's EMA copy of the frozen weights does not overwrite the model's.
    state = {k: v.clone() for k, v in ema.state_dict().items()}
    state['module.0.weight'] += 1.0
    state['module.2.weight'] += 1.0
    ema.load_state_dict(state, strict=False)
    assert torch.equal(model[0].weight, frozen)
    assert torch.equal(ema.module[2].weight, averaged + 1.0)
