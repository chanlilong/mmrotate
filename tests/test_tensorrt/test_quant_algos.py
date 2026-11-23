# Copyright (c) OpenMMLab. All rights reserved.
"""INT8 refinement algorithms (AWQ scale / clip, GPTQ, sensitivity) on small CPU models."""

import copy

import pytest
import torch
from torch import Tensor, nn

pytest.importorskip('modelopt')

from mmrotate.tensorrt import precision as prec  # noqa: E402
from mmrotate.tensorrt.quant import (  # noqa: E402
    auto_exclude,
    awq_clip,
    awq_scale,
    gptq,
    layer_sensitivity,
    quant_layers,
    refine,
    select_layers,
)
from mmrotate.tensorrt.quant.common import fake_quant, pre_scale, quantizers_disabled  # noqa: E402
from mmrotate.tensorrt.quant.gptq import gptq_round, reconstruction_error  # noqa: E402

OUTLIER = 200.0


class TinyNet(nn.Module):
    """conv -> SiLU -> conv -> SiLU -> conv with one outlier channel after the first conv.

    Channel 0 of the first conv is ``OUTLIER`` times larger than the others and
    the second conv divides it back, as a channel with a large folded
    BatchNorm scale would behave: harmless in float, fatal for a per-tensor
    activation quantizer.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.conv3 = nn.Conv2d(8, 4, 1)
        self.act = nn.SiLU()
        with torch.no_grad():
            self.conv1.weight[0] *= OUTLIER
            self.conv1.bias[0] *= OUTLIER
            self.conv2.weight[:, 0] /= OUTLIER

    def forward(self, x: Tensor) -> Tensor:
        return self.conv3(self.act(self.conv2(self.act(self.conv1(x)))))

    _forward = forward


def _batches(n: int = 4) -> list[Tensor]:
    gen = torch.Generator().manual_seed(0)
    return [torch.randn(2, 3, 16, 16, generator=gen) for _ in range(n)]


def _quantized(seed: int = 0) -> tuple[nn.Module, list[Tensor]]:
    torch.manual_seed(seed)
    batches = _batches()
    model = prec.quantize(TinyNet().eval(), 'int8', batches, exclude=())
    return model, batches


@torch.no_grad()
def _float_out(model: nn.Module, batches: list[Tensor]) -> Tensor:
    with quantizers_disabled(model):
        return torch.cat([model(x) for x in batches])


@torch.no_grad()
def _quant_err(model: nn.Module, batches: list[Tensor]) -> float:
    ref = _float_out(model, batches)
    out = torch.cat([model(x) for x in batches])
    return float((out - ref).square().mean() / ref.var())


def test_awq_scale_keeps_float_model_and_fixes_outlier_channel() -> None:
    model, batches = _quantized()
    ref = _float_out(model, batches)
    err_before = _quant_err(model, batches)
    report = awq_scale(model, batches)
    scaled = {r['layer'] for r in report if r['alpha'] is not None}
    assert 'conv2' in scaled
    layer = next(layer for layer in quant_layers(model) if layer.name == 'conv2')
    scale = pre_scale(layer).flatten()
    assert scale[0] < 0.1 * scale[1:].min()  # the outlier channel is scaled down
    assert torch.allclose(_float_out(model, batches), ref, rtol=1e-4, atol=1e-5)
    assert _quant_err(model, batches) < 0.2 * err_before


def test_awq_clip_only_shrinks_ranges_and_never_raises_layer_error() -> None:
    model, batches = _quantized()
    before = {layer.name: layer.weight_quantizer.amax.clone() for layer in quant_layers(model)}
    report = awq_clip(model, batches)
    assert len(report) == 3
    for r in report:
        assert r['err_after'] <= r['err_before']
        assert 0.5 <= r['mean_ratio'] <= 1.0
    for layer in quant_layers(model):
        assert bool((layer.weight_quantizer.amax <= before[layer.name] + 1e-9).all())


def test_gptq_round_beats_nearest_on_correlated_inputs() -> None:
    gen = torch.Generator().manual_seed(0)
    d, out = 48, 16
    mixing = torch.randn(d, 6, generator=gen)  # inputs live close to a 6-d subspace
    x = mixing @ torch.randn(6, 4000, generator=gen) + 0.05 * torch.randn(d, 4000, generator=gen)
    hessian = (x @ x.t()).double() / x.shape[1]
    weight = torch.randn(out, d, generator=gen)
    amax = weight.abs().amax(1, keepdim=True)
    nearest = fake_quant(weight, amax)
    rounded = gptq_round(weight, hessian, amax)
    steps = rounded / (amax / 127)
    assert torch.allclose(steps, steps.round(), atol=1e-4)  # on the quantizer grid
    assert float(steps.abs().max()) <= 128
    err_nearest = reconstruction_error(weight, nearest, hessian)
    assert reconstruction_error(weight, rounded, hessian) < 0.5 * err_nearest


def test_gptq_keeps_weights_on_the_quantizer_grid() -> None:
    model, batches = _quantized()
    report = gptq(model, batches)
    assert [r['layer'] for r in report] == ['conv1', 'conv2', 'conv3']
    for r in report:
        assert r['err_gptq'] <= r['err_nearest']
    for layer in quant_layers(model):
        weight = layer.module.weight.detach()
        amax = layer.weight_quantizer.amax
        assert torch.allclose(fake_quant(weight, amax), weight, atol=1e-6)


def test_select_layers_fills_the_budget_from_the_least_sensitive() -> None:
    sensitivity = dict(a=0.001, b=0.5, c=0.004, d=0.02)
    assert select_layers(sensitivity, budget=0.01) == ['d', 'b']
    assert select_layers(sensitivity, budget=1.0) == []
    assert sorted(select_layers(sensitivity, budget=0.0)) == ['a', 'b', 'c', 'd']


def test_sensitivity_finds_the_outlier_layer_and_auto_exclude_restores_it() -> None:
    model, batches = _quantized()
    originals = {layer.name: layer.module.weight.detach().clone() for layer in quant_layers(model)}
    sensitivity = layer_sensitivity(model, batches)
    assert max(sensitivity, key=sensitivity.get) == 'conv2'
    budget = sum(v for k, v in sensitivity.items() if k != 'conv2') * 1.01
    result = auto_exclude(model, batches, budget, originals)
    assert result['excluded'] == ['conv2']
    assert [layer.name for layer in quant_layers(model)] == ['conv1', 'conv3']
    assert not model.conv2.input_quantizer.is_enabled
    assert result['noise'] < sensitivity['conv2']


def test_refine_runs_every_step_and_lowers_the_output_error() -> None:
    model, batches = _quantized()
    reference = copy.deepcopy(model)
    err_before = _quant_err(reference, batches)
    report = refine(model, batches, ('awq-scale', 'awq-clip', 'gptq'), exclude_budget=None)
    assert report['awq_scale']['scaled'] >= 1
    assert report['gptq']['error_ratio'] <= 1.0
    assert _quant_err(model, batches) < 0.2 * err_before


def test_refine_rejects_unknown_algorithms_and_fp8() -> None:
    model, batches = _quantized()
    with pytest.raises(ValueError, match='Unknown quantization algorithms'):
        refine(model, batches, ('adaround',))
    with pytest.raises(ValueError, match='int8 only'):
        prec.quantize(TinyNet().eval(), 'fp8', batches, exclude=(), algos=('gptq',))


@torch.no_grad()
def test_int8_export_forward_applies_the_awq_input_scale() -> None:
    model, batches = _quantized()
    awq_scale(model, batches)
    x = batches[0]
    expected = model(x)
    with prec.int8_export_mode(model):
        exported = model(x)
    assert torch.allclose(exported, expected, rtol=1e-4, atol=1e-5)
