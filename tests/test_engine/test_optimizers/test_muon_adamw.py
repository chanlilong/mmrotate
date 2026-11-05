# Copyright (c) OpenMMLab. All rights reserved.
"""Tests for the hybrid MuonAdamW optimizer and its routing constructor."""

import copy

import pytest
import torch
from mmengine.optim import build_optim_wrapper
from mmengine.optim.scheduler import LinearLR
from mmengine.registry import init_default_scope
from torch import Tensor, nn

from mmrotate.engine.optimizers import MuonAdamW
from mmrotate.engine.optimizers.muon_adamw import DEFAULT_NS_COEFFICIENTS


class ToyLoRA(nn.Module):
    """Stand-in for cradiov4 ``LoRALinear``: not an ``nn.Linear``, 2D factors."""

    def __init__(self, dim: int, r: int) -> None:
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(r, dim) * 0.1)
        self.lora_B = nn.Parameter(torch.randn(dim, r) * 0.1)

    def forward(self, x: Tensor) -> Tensor:
        return x + x @ self.lora_A.t() @ self.lora_B.t()


class ToyDetector(nn.Module):
    """Every layer kind of an RTMDet / CoMer model, at toy size."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(32, 32)
        self.small_linear = nn.Linear(32, 8)  # min dim 8 < muon_min_dim
        self.lora_lo = ToyLoRA(32, r=4)
        self.lora_hi = ToyLoRA(32, r=16)
        self.level_embed = nn.Parameter(torch.zeros(3, 32))  # 2D but not a layer
        self.pw = nn.Conv2d(32, 32, 1)
        self.grouped_pw = nn.Conv2d(32, 32, 1, groups=4)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.dw = nn.Conv2d(32, 32, 3, padding=1, groups=32)
        self.bn = nn.BatchNorm2d(32)
        self.rtm_cls = nn.Conv2d(32, 16, 1)
        self.frozen = nn.Linear(32, 32)
        self.frozen.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        tokens = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        tokens = self.lora_hi(self.lora_lo(self.frozen(self.linear(tokens))))
        out = tokens.transpose(1, 2).reshape(x.shape)
        out = self.bn(self.dw(self.conv3(self.grouped_pw(self.pw(out)))))
        small = self.small_linear(out.mean((2, 3)))
        out = out + self.level_embed.sum(0)[:, None, None]  # after BN, which would cancel it
        return self.rtm_cls(out).mean() + small.mean()


EXPECTED_MUON = {
    'linear.weight',
    'lora_hi.lora_A',
    'lora_hi.lora_B',
    'pw.weight',
}


def _optim_cfg(**paramwise: object) -> dict:
    return dict(
        type='OptimWrapper',
        constructor='MuonAdamWOptimWrapperConstructor',
        optimizer=dict(type='MuonAdamW', lr=1e-3, weight_decay=0.05),
        paramwise_cfg=dict(
            norm_decay_mult=0,
            bias_decay_mult=0,
            bypass_duplicate=True,
            muon_include_keys=['lora_A', 'lora_B'],
            muon_exclude_keys=['rtm_cls'],
            muon_min_dim=16,
            **paramwise,
        ),
    )


def _routing(optimizer: torch.optim.Optimizer, model: nn.Module) -> dict[str, bool]:
    flags = {id(p): g['use_muon'] for g in optimizer.param_groups for p in g['params']}
    return {n: flags[id(p)] for n, p in model.named_parameters() if id(p) in flags}


def _backward(model: nn.Module, seed: int) -> None:
    torch.manual_seed(seed)
    model.zero_grad()
    model(torch.randn(2, 32, 4, 4)).backward()


@pytest.fixture(autouse=True)
def _scope() -> None:
    init_default_scope('mmrotate')


def test_constructor_routing() -> None:
    model = ToyDetector()
    wrapper = build_optim_wrapper(model, _optim_cfg())
    routing = _routing(wrapper.optimizer, model)
    assert {n for n, muon in routing.items() if muon} == EXPECTED_MUON
    assert 'frozen.weight' not in routing  # frozen params are not handed to the optimizer
    assert routing['level_embed'] is False
    assert routing['grouped_pw.weight'] is False
    assert routing['rtm_cls.weight'] is False
    assert routing['lora_lo.lora_A'] is False  # rank 4 < muon_min_dim
    # paramwise_cfg still applies: no decay on norm and bias params.
    groups = {id(g['params'][0]): g for g in wrapper.optimizer.param_groups}
    assert groups[id(model.bn.weight)]['weight_decay'] == 0
    assert groups[id(model.pw.bias)]['weight_decay'] == 0
    assert groups[id(model.pw.weight)]['weight_decay'] == pytest.approx(0.05)


def test_custom_keys_can_force_adamw() -> None:
    model = ToyDetector()
    wrapper = build_optim_wrapper(
        model, _optim_cfg(custom_keys={'pw.weight': dict(use_muon=False)})
    )
    assert _routing(wrapper.optimizer, model)['pw.weight'] is False


def test_constructor_without_paramwise_cfg() -> None:
    model = ToyDetector()
    cfg = _optim_cfg()
    cfg.pop('paramwise_cfg')
    routing = _routing(build_optim_wrapper(model, cfg).optimizer, model)
    assert {n for n, muon in routing.items() if muon} == {
        'linear.weight',
        'pw.weight',
        'rtm_cls.weight',
    }


def test_shape_fallback_with_default_constructor() -> None:
    model = ToyDetector()
    wrapper = build_optim_wrapper(model, dict(optimizer=dict(type='MuonAdamW', lr=1e-3)))
    routing = _routing(wrapper.optimizer, model)
    assert routing['linear.weight'] and routing['pw.weight'] and routing['grouped_pw.weight']
    assert not routing['conv3.weight'] and not routing['dw.weight'] and not routing['pw.bias']
    assert routing['level_embed']  # the shape rule can't tell; the constructor can


def test_rejects_non_matrix_muon_param() -> None:
    with pytest.raises(ValueError, match='2D or 1x1'):
        MuonAdamW([{'params': [nn.Conv2d(4, 4, 3).weight], 'use_muon': True}])


@pytest.mark.parametrize('shape', [(48, 32), (48, 32, 1, 1)])
def test_muon_matches_torch_muon(shape: tuple[int, ...]) -> None:
    torch.manual_seed(0)
    weight = nn.Parameter(torch.randn(shape))
    ref = nn.Parameter(weight.detach().reshape(48, 32).clone())
    ours = MuonAdamW([{'params': [weight], 'use_muon': True}], lr=0.02, weight_decay=0.1)
    torch_muon = torch.optim.Muon([ref], lr=0.02, weight_decay=0.1, adjust_lr_fn='match_rms_adamw')
    assert ours.param_groups[0]['ns_coefficients'] == DEFAULT_NS_COEFFICIENTS
    for _ in range(3):
        grad = torch.randn(48, 32)
        weight.grad = grad.reshape(shape).clone()
        ref.grad = grad.clone()
        ours.step()
        torch_muon.step()
    torch.testing.assert_close(weight.detach().reshape(48, 32), ref.detach())


def test_adamw_matches_torch_adamw() -> None:
    torch.manual_seed(0)
    conv = nn.Parameter(torch.randn(8, 8, 3, 3))
    bias = nn.Parameter(torch.randn(8))
    refs = [nn.Parameter(p.detach().clone()) for p in (conv, bias)]
    ours = MuonAdamW([conv, bias], lr=1e-2, weight_decay=0.05)
    assert all(not g['use_muon'] for g in ours.param_groups)
    torch_adamw = torch.optim.AdamW(refs, lr=1e-2, weight_decay=0.05)
    for _ in range(3):
        for p, r in zip((conv, bias), refs, strict=True):
            p.grad = torch.randn_like(p)
            r.grad = p.grad.clone()
        ours.step()
        torch_adamw.step()
    for p, r in zip((conv, bias), refs, strict=True):
        torch.testing.assert_close(p.detach(), r.detach())


def test_skips_params_without_grad() -> None:
    weight = nn.Parameter(torch.randn(16, 16))
    before = weight.detach().clone()
    optimizer = MuonAdamW([{'params': [weight], 'use_muon': True}])
    optimizer.step()
    torch.testing.assert_close(weight.detach(), before)
    assert len(optimizer.state) == 0


def test_lr_mults() -> None:
    weight = nn.Parameter(torch.randn(16, 16))
    bias = nn.Parameter(torch.randn(16))
    optimizer = MuonAdamW(
        [weight, bias], lr=1e-3, weight_decay=0.1, muon_lr_mult=2.0, muon_decay_mult=0.5
    )
    groups = {g['use_muon']: g for g in optimizer.param_groups}
    assert groups[True]['lr'] == pytest.approx(2e-3)
    assert groups[True]['weight_decay'] == pytest.approx(0.05)
    assert groups[False]['lr'] == pytest.approx(1e-3)


def test_scheduler_reaches_both_branches() -> None:
    model = ToyDetector()
    wrapper = build_optim_wrapper(model, _optim_cfg())
    scheduler = LinearLR(wrapper, start_factor=0.5, by_epoch=False, begin=0, end=3)
    for group in wrapper.optimizer.param_groups:
        assert group['lr'] == pytest.approx(5e-4)
    _backward(model, 0)
    wrapper.step()
    scheduler.step()
    for group in wrapper.optimizer.param_groups:
        assert group['lr'] == pytest.approx(7.5e-4)


def test_state_dict_round_trip() -> None:
    model = ToyDetector()
    wrapper = build_optim_wrapper(model, _optim_cfg())
    for seed in range(2):
        _backward(model, seed)
        wrapper.step()
    resumed_model = copy.deepcopy(model)
    resumed = build_optim_wrapper(resumed_model, _optim_cfg())
    resumed.load_state_dict(copy.deepcopy(wrapper.state_dict()))
    assert 'momentum_buffer' in resumed.optimizer.state[resumed_model.pw.weight]
    assert 'exp_avg' in resumed.optimizer.state[resumed_model.conv3.weight]
    for m, w in ((model, wrapper), (resumed_model, resumed)):
        _backward(m, 7)
        w.step()
    for (name, p), r in zip(model.named_parameters(), resumed_model.parameters(), strict=True):
        torch.testing.assert_close(p, r, msg=name)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_amp_bf16_wrapper() -> None:
    model = ToyDetector().cuda()
    cfg = _optim_cfg()
    cfg.update(type='AmpOptimWrapper', dtype='bfloat16')
    wrapper = build_optim_wrapper(model, cfg)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    for _ in range(2):
        with wrapper.optim_context(model):
            loss = model(torch.randn(2, 32, 4, 4, device='cuda'))
        wrapper.update_params(loss)
    params = dict(model.named_parameters())
    for name, p in params.items():
        assert torch.isfinite(p).all(), name
    # Both branches stepped (other params only get BN-cancelled, ~zero grads in this toy).
    for name in ('linear.weight', 'pw.weight', 'conv3.weight', 'rtm_cls.weight', 'level_embed'):
        assert not torch.equal(params[name], before[name]), name
