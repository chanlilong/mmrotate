# Copyright (c) OpenMMLab. All rights reserved.
"""ViT-CoMer deformable attention as a TensorRT plugin around the Triton kernel."""

from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn
from trt_test_utils import convert_tiny

from mmrotate.tensorrt.plugins.msda import PLUGIN_NAME, PLUGIN_NAMESPACE, strip_hidden_params

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')

PTX = """\
.version 8.8
.target sm_86
.visible .entry kern(
\t.param .u64 .ptr .global .align 1 kern_param_0,
\t.param .u32 kern_param_1,
\t.param .u64 .ptr .global .align 1 kern_param_2,
\t.param .u64 .ptr .global .align 1 kern_param_3,
\t.param .u64 .ptr .global .align 1 kern_param_4
)
.reqntid 32
{
\tld.param.u64 \t%rd1, [kern_param_0];
\tld.param.u32 \t%r1, [kern_param_1];
\tld.param.u64 \t%rd2, [kern_param_2];
\tret;
}
"""


def test_strip_hidden_params_drops_the_unused_trailing_parameters() -> None:
    stripped = strip_hidden_params(PTX, 'kern', 3)
    assert 'kern_param_3' not in stripped and 'kern_param_4' not in stripped
    assert '\t.param .u64 .ptr .global .align 1 kern_param_2\n)' in stripped
    assert stripped.count('\t.param ') == 3
    assert stripped.endswith(PTX[PTX.index('.reqntid') :])
    assert strip_hidden_params(PTX, 'kern', 5) == PTX


def test_strip_hidden_params_refuses_used_or_missing_parameters() -> None:
    used = PTX.replace('\tret;', '\tld.param.u64 \t%rd3, [kern_param_3];\n\tret;')
    with pytest.raises(ValueError, match='is used'):
        strip_hidden_params(used, 'kern', 3)
    with pytest.raises(ValueError, match='No entry'):
        strip_hidden_params(PTX, 'other', 3)
    with pytest.raises(ValueError, match='expected 6'):
        strip_hidden_params(PTX, 'kern', 6)


def _msda_inputs(dtype: torch.dtype, ref_levels: int = 1) -> tuple[Tensor, ...]:
    gen = torch.Generator().manual_seed(0)
    B, H, C, L, P = 2, 2, 16, 3, 4
    shapes = torch.tensor([[12, 10], [6, 5], [3, 3]], device='cuda')
    num_in = int((shapes[:, 0] * shapes[:, 1]).sum())
    N = 77
    value = torch.randn(B, num_in, H, C, generator=gen).to('cuda', dtype)
    ref = torch.rand(1, N, ref_levels, 2, generator=gen).cuda()
    offsets = (3 * torch.randn(B, N, H, L, P, 2, generator=gen)).to('cuda', dtype)
    logits = torch.randn(B, N, H, L * P, generator=gen).to('cuda', dtype)
    return value, shapes, ref, offsets, logits


# One unit in the last place of each dtype around 1, for results that differ in rounding only.
_ULP = {torch.float32: 2e-6, torch.float16: 2e-3, torch.bfloat16: 2e-2}


@needs_cuda
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('ref_levels', [1, 3])
def test_trt_order_kernel_matches_the_training_kernel(dtype: torch.dtype, ref_levels: int) -> None:
    triton = pytest.importorskip('triton')
    from mmrotate.models.backbones.cradiov4 import msda_triton_fused as fused
    from mmrotate.tensorrt.plugins.msda_kernel import msda_fused_fwd_trt_kernel

    value, shapes, ref, offsets, logits = _msda_inputs(dtype, ref_levels)
    expected = fused.msda_fused_fwd(value, shapes, ref, offsets, logits)
    tuned = fused._msda_fused_fwd_kernel.best_config
    B, num_in, H, C = value.shape
    _, N, _, L, P, _ = offsets.shape
    out = torch.empty_like(expected)

    def run(block_n: int, num_warps: int) -> Tensor:
        out.zero_()
        msda_fused_fwd_trt_kernel[(triton.cdiv(N, block_n), B, H)](
            value, shapes, ref, offsets, logits, N, num_in, 0, out,
            H=H, C=C, L=L, P=P, LR=ref_levels,
            BLOCK_LP=triton.next_power_of_2(L * P), BLOCK_C=triton.next_power_of_2(C),
            BLOCK_N=block_n, num_warps=num_warps,
        )  # fmt: skip
        return out

    # Same launch configuration: the same machine code, bit for bit.
    assert torch.equal(run(int(tuned.kwargs['BLOCK_N']), int(tuned.num_warps)), expected)
    # Other configurations vectorise differently and may round the last bit differently.
    for block_n, num_warps in ((1, 1), (4, 2), (16, 4)):
        diff = (run(block_n, num_warps).float() - expected.float()).abs().max()
        assert float(diff) <= _ULP[dtype] * max(1.0, float(expected.float().abs().max()))


@needs_cuda
def test_compiled_kernel_has_exactly_the_tensorrt_parameters() -> None:
    pytest.importorskip('triton')
    from mmrotate.tensorrt.plugins.msda import compile_msda_kernel

    dtypes = dict(value='fp16', shapes='i64', ref='fp32', off='fp16', logit='fp16', out='fp16')
    name, ptx, shared = compile_msda_kernel(
        dtypes, H=2, C=16, L=3, P=4, LR=1, block_n=2, num_warps=1
    )
    assert name == 'msda_fused_fwd_trt_kernel'
    entry = ptx[ptx.index(f'.entry {name}(') : ptx.index('\n)', ptx.index(f'.entry {name}('))]
    assert entry.count('.param') == 9  # 5 inputs, 3 scalars, 1 output
    assert shared >= 0


class _OneMSDA(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from mmrotate.models.backbones.cradiov4.ms_deform_attn import MSDeformAttn

        self.attn = MSDeformAttn(d_model=32, n_levels=3, n_heads=2, n_points=4)
        nn.init.normal_(self.attn.sampling_offsets.weight, std=0.05)
        nn.init.normal_(self.attn.attention_weights.weight, std=0.05)
        self.shapes = [(12, 10), (6, 5), (3, 3)]
        self.register_buffer('ref', torch.rand(1, 159, 1, 2))

    def forward(self, query: Tensor, feat: Tensor) -> Tensor:
        shapes = torch.as_tensor(self.shapes, dtype=torch.long, device=query.device)
        return self.attn(query, self.ref, feat, shapes, None)


@needs_cuda
def test_export_gives_one_plugin_node_with_the_kernel_inputs(tmp_path: Path) -> None:
    onnx = pytest.importorskip('onnx')
    pytest.importorskip('triton')
    from mmrotate.models.backbones.cradiov4 import ms_deform_attn
    from mmrotate.tensorrt.plugins import export_msda_as_plugin

    torch.manual_seed(0)
    module = _OneMSDA().cuda().eval()
    args = (torch.randn(2, 159, 32, device='cuda'), torch.randn(2, 159, 32, device='cuda'))
    original = ms_deform_attn.FusedMSDAFunction
    with torch.no_grad():
        expected = module(*args)
        with export_msda_as_plugin():
            assert torch.equal(module(*args), expected)  # same kernel while exporting
            torch.onnx.export(
                module, args, str(tmp_path / 'm.onnx'), dynamo=False, opset_version=19,
                custom_opsets={PLUGIN_NAMESPACE: 1},
            )  # fmt: skip
    assert ms_deform_attn.FusedMSDAFunction is original
    graph = onnx.load(str(tmp_path / 'm.onnx')).graph
    nodes = [n for n in graph.node if n.op_type == PLUGIN_NAME]
    assert len(nodes) == 1 and nodes[0].domain == PLUGIN_NAMESPACE
    assert len(nodes[0].input) == 5
    attrs = {a.name: onnx.helper.get_attribute_value(a) for a in nodes[0].attribute}
    assert attrs['plugin_namespace'] == PLUGIN_NAMESPACE.encode()
    assert attrs['block_n'] >= 1 and attrs['num_warps'] >= 1
    assert not any(n.op_type == 'GridSample' for n in graph.node)


@needs_cuda
@pytest.mark.parametrize('plugins', ['aot', 'jit'])
@pytest.mark.parametrize('precision', ['fp32', 'fp16'])
def test_plugin_engine_matches_torch(
    plugins: str, precision: str, tiny_detector: Callable[..., nn.Module], tmp_path: Path
) -> None:
    pytest.importorskip('tensorrt')
    pytest.importorskip('triton')
    from mmrotate.tensorrt import init_trt_detector
    from mmrotate.tensorrt.export_patches import has_msda
    from mmrotate.tensorrt.verify import raw_output_errors

    torch.backends.cudnn.allow_tf32 = False
    model = tiny_detector('comer', 'cuda')
    assert has_msda(model)
    num_msda = sum(type(m).__name__ == 'MSDeformAttn' for m in model.modules())
    shape = (2, 3, 128, 128)
    meta = convert_tiny(model, 'comer', tmp_path, shape, precision=precision, plugins=plugins)
    assert meta['fusion']['plugins'] == num_msda > 0
    x = torch.randn(shape, device='cuda')
    with torch.no_grad():
        errors = raw_output_errors(
            model._forward(x), init_trt_detector(tmp_path, 'cuda')._forward(x)
        )
    assert errors['finite']
    if precision == 'fp32':
        assert errors['max_rel_err'] < 1e-3, errors
    else:
        assert errors['max_abs_err'] < 0.05 * errors['max_abs_ref'], errors
