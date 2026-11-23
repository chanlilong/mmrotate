# Copyright (c) OpenMMLab. All rights reserved.
"""Fused attention export: forms, numerics probe, ONNX ``Attention`` nodes, TensorRT fusion."""

from collections.abc import Callable
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from trt_test_utils import convert_tiny

from mmrotate.tensorrt import init_trt_detector
from mmrotate.tensorrt.attention import (
    ATTENTION_FORMS,
    ATTENTION_OPSET,
    AttentionExporter,
    AttentionStats,
    patch_sdpa,
    plan_attention,
    probe_attention,
    resolve_attention,
    sdpa_fused,
    upgrade_attention_nodes,
)
from mmrotate.tensorrt.verify import raw_output_errors

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')


class TwoAttentions(nn.Module):
    """Two attention calls; the first one has q / k of ``gain`` times the scale."""

    def __init__(self, dim: int = 16, heads: int = 2, gain: float = 1.0) -> None:
        super().__init__()
        self.heads = heads
        self.gain = gain
        self.qkv = nn.Linear(dim, dim * 3)

    def _split(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        return qkv[0], qkv[1], qkv[2]

    def forward(self, x: Tensor) -> Tensor:
        q, k, v = self._split(x)
        x = F.scaled_dot_product_attention(q * self.gain, k * self.gain, v)
        x = x.transpose(1, 2).flatten(2)
        q, k, v = self._split(x)
        return F.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(2)

    _forward = forward


def _qkv(dtype: torch.dtype = torch.float32) -> tuple[Tensor, Tensor, Tensor]:
    gen = torch.Generator().manual_seed(0)
    return tuple(torch.randn(2, 3, 10, 8, generator=gen).to(dtype) for _ in range(3))


@pytest.mark.parametrize('form', ATTENTION_FORMS)
@pytest.mark.parametrize('scale', [None, 0.2])
def test_forms_equal_torch_attention(form: str, scale: float | None) -> None:
    q, k, v = _qkv()
    expected = F.scaled_dot_product_attention(q, k, v, scale=scale)
    out = AttentionExporter(form)(q, k, v, scale=scale)
    assert torch.allclose(out, expected, atol=1e-5)


@pytest.mark.parametrize('form', ATTENTION_FORMS)
def test_masked_attention_is_exact_in_every_form(form: str) -> None:
    q, k, v = _qkv()
    mask = torch.ones(10, 10, dtype=torch.bool).tril()
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    assert torch.allclose(AttentionExporter(form)(q, k, v, attn_mask=mask), expected, atol=1e-5)
    additive = torch.zeros(10, 10).masked_fill(~mask, float('-inf'))
    assert torch.allclose(AttentionExporter(form)(q, k, v, attn_mask=additive), expected, atol=1e-5)


def test_fused_forward_is_fp32_math_in_any_dtype() -> None:
    q, k, v = _qkv(torch.float16)
    q = q * 3000  # logits far beyond fp16
    expected = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
    out = sdpa_fused(q, k, v)
    assert out.dtype == torch.float16
    assert torch.isfinite(out).all()
    assert torch.allclose(out.float(), expected, atol=2e-3)


def test_exporter_follows_the_plan_and_validates_it() -> None:
    exporter = AttentionExporter(['legacy', 'fused'])
    q, k, v = _qkv()
    for _ in range(5):
        exporter(q, k, v)
    assert exporter.calls == 5
    with pytest.raises(ValueError, match='Unknown attention forms'):
        AttentionExporter(['fused', 'flash'])
    with pytest.raises(ValueError, match='Unknown attention forms'):
        AttentionExporter([])
    with pytest.raises(NotImplementedError):
        AttentionExporter('fused')(q, k, v, is_causal=True)


def test_patch_sdpa_is_undone() -> None:
    original = F.scaled_dot_product_attention
    with patch_sdpa(AttentionExporter('legacy')):
        assert F.scaled_dot_product_attention is not original
    assert F.scaled_dot_product_attention is original


def test_probe_measures_each_call_and_plans_fp32_accumulation_for_overflow() -> None:
    torch.manual_seed(0)
    model = TwoAttentions(gain=3000.0).eval()
    batches = [torch.randn(2, 12, 16), torch.randn(1, 12, 16)]
    stats = probe_attention(model, batches)
    assert [s.index for s in stats] == [0, 1]
    assert stats[0].tokens == 12 and stats[0].head_dim == 8
    assert stats[0].max_q > 100 * stats[1].max_q
    assert not stats[0].fp16_finite and stats[0].fp16_err == float('inf')
    assert stats[1].fp16_finite and stats[1].fp16_err < 0.02
    assert stats[0].fp16_io_err < 0.05  # fp32 accumulation survives fp16 inputs
    assert plan_attention(stats, torch.float16) == ['fused-fp32acc', 'fused']
    assert plan_attention(stats, torch.bfloat16) == ['fused', 'fused']
    assert plan_attention(stats, None) == ['legacy', 'legacy']


def test_plan_keeps_masked_calls_unfused() -> None:
    base = dict(
        tokens=10, head_dim=8, max_q=1.0, max_k=1.0, max_logit=1.0, max_partial=1.0,
        fp16_finite=True, fp16_err=1e-4, fp16_io_err=1e-4,
    )  # fmt: skip
    stats = [AttentionStats(index=0, **base), AttentionStats(index=1, masked=True, **base)]
    assert plan_attention(stats, torch.float16) == ['fused', 'legacy']
    risky = AttentionStats(index=0, **{**base, 'max_partial': 30000.0})
    assert plan_attention([risky], torch.float16) == ['fused-fp32acc']
    assert plan_attention([risky], torch.float16, headroom=1.0) == ['fused']


def test_resolve_attention_modes() -> None:
    torch.manual_seed(0)
    model = TwoAttentions(gain=3000.0).eval()
    batches = [torch.randn(1, 12, 16)]
    plan, stats = resolve_attention('auto', model, batches, torch.float16)
    assert plan == ['fused-fp32acc', 'fused']
    assert [s['form'] for s in stats] == plan and stats[0]['max_logit'] > 1e4
    plan, _ = resolve_attention('legacy', model, batches, torch.float16)
    assert plan == ['legacy', 'legacy']
    with pytest.raises(ValueError, match='Unknown attention mode'):
        resolve_attention('flash', model, batches, torch.float16)


class _QKVModule(nn.Module):
    def __init__(self, plan: list[str]) -> None:
        super().__init__()
        self.plan = plan

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        x = F.scaled_dot_product_attention(q, k, v)
        return F.scaled_dot_product_attention(x, k, v)


def _export(module: nn.Module, args: tuple, file: Path, plan: list[str]) -> None:
    with torch.no_grad(), patch_sdpa(AttentionExporter(plan)):
        torch.onnx.export(
            module, args, str(file), dynamo=False, opset_version=19,
            input_names=['q', 'k', 'v'], output_names=['o'], custom_opsets={'mmrotate': 1},
        )  # fmt: skip


def test_export_gives_standard_attention_nodes_with_fp32_casts(tmp_path: Path) -> None:
    onnx = pytest.importorskip('onnx')
    file = tmp_path / 'att.onnx'
    plan = ['fused-fp32acc', 'fused']
    _export(_QKVModule(plan), _qkv(torch.float16), file, plan)
    assert upgrade_attention_nodes(str(file)) == 2
    assert upgrade_attention_nodes(str(file)) == 0  # idempotent
    model = onnx.load(str(file))
    onnx.checker.check_model(model)
    assert {o.domain: o.version for o in model.opset_import}[''] == ATTENTION_OPSET
    producers = {out: node for node in model.graph.node for out in node.output}
    consumers = {inp: node for node in model.graph.node for inp in node.input}
    first, second = (n for n in model.graph.node if n.op_type == 'Attention')
    assert first.domain == second.domain == ''
    scale = {a.name: a.f for a in first.attribute}['scale']
    assert scale == pytest.approx(8**-0.5)
    # fused-fp32acc: q, k, v cast to fp32, the result cast back to fp16.
    for name in first.input:
        cast = producers[name]
        assert cast.op_type == 'Cast' and cast.attribute[0].i == onnx.TensorProto.FLOAT
    back = consumers[first.output[0]]
    assert back.op_type == 'Cast' and back.attribute[0].i == onnx.TensorProto.FLOAT16
    # fused: on the fp16 tensors (the tracer may add same-type casts, which are no-ops).
    for name, source in zip(second.input, (back.output[0], 'k', 'v'), strict=True):
        while name != source:
            cast = producers[name]
            assert cast.op_type == 'Cast' and cast.attribute[0].i == onnx.TensorProto.FLOAT16
            name = cast.input[0]


def test_upgrade_renames_grid_sample_modes_and_skips_plain_graphs(tmp_path: Path) -> None:
    onnx = pytest.importorskip('onnx')

    class Sampler(nn.Module):
        def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
            grid = torch.zeros(q.shape[0], 4, 4, 2)
            x = F.grid_sample(q, grid, mode='bilinear', align_corners=False)
            return x.sum() + F.scaled_dot_product_attention(q, k, v)

    plain = tmp_path / 'plain.onnx'
    _export(Sampler(), _qkv(), plain, ['legacy'])
    before = plain.read_bytes()
    assert upgrade_attention_nodes(str(plain)) == 0
    assert plain.read_bytes() == before

    fused = tmp_path / 'fused.onnx'
    _export(Sampler(), _qkv(), fused, ['fused'])
    assert upgrade_attention_nodes(str(fused)) == 1
    model = onnx.load(str(fused))
    onnx.checker.check_model(model)
    sample = next(n for n in model.graph.node if n.op_type == 'GridSample')
    assert {a.name: a.s for a in sample.attribute if a.name == 'mode'}['mode'] == b'linear'


@needs_cuda
@pytest.mark.parametrize('attention', ['auto', 'fused', 'fused-fp32acc'])
def test_tensorrt_fuses_every_attention_call(
    attention: str, tiny_detector: Callable[..., nn.Module], tmp_path: Path
) -> None:
    pytest.importorskip('tensorrt')
    model = tiny_detector('radio', 'cuda')
    shape = (2, 3, 128, 128)
    meta = convert_tiny(model, 'radio', tmp_path, shape, precision='fp16', attention=attention)
    calls = len(meta['attention'])
    assert calls == 4  # one per ViT block of the tiny model
    assert all(a['form'] != 'legacy' for a in meta['attention'])
    assert meta['fusion']['mha'] == calls
    x = torch.randn(shape, device='cuda')
    with torch.no_grad():
        errors = raw_output_errors(
            model._forward(x), init_trt_detector(tmp_path, 'cuda')._forward(x)
        )
    assert errors['finite']
    assert errors['max_abs_err'] < 0.05 * errors['max_abs_ref'], errors


@needs_cuda
def test_legacy_attention_is_not_fused(
    tiny_detector: Callable[..., nn.Module], tmp_path: Path
) -> None:
    pytest.importorskip('tensorrt')
    model = tiny_detector('radio', 'cuda')
    meta = convert_tiny(model, 'radio', tmp_path, (2, 3, 128, 128), precision='fp16')
    assert len(meta['attention']) == 4
    assert meta['fusion']['mha'] == 0
