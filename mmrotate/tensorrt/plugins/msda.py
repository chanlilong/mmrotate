# Copyright (c) OpenMMLab. All rights reserved.
"""ViT-CoMer ``MSDeformAttn`` as one TensorRT plugin node running the Triton kernel.

Export side (:func:`export_msda_as_plugin`): the fused Triton autograd
function of ``MSDeformAttn`` is swapped for :class:`MSDAExportFunction`, which
still runs the Triton kernel in torch (so tracing and calibration see real
values) and exports as a single ``mmrotate::MSDAFused`` ONNX node with the
inputs of the kernel: value, spatial shapes, reference points, raw sampling
offsets and raw attention logits. The softmax, the sampling locations, the
bilinear sampling and the weighted sum all happen inside the kernel.

TensorRT side (:func:`register_plugins`): a ``tensorrt.plugin`` plugin with
two implementations of that node,

- ``aot``: the kernel is compiled by Triton to PTX when the engine is built
  and embedded in the engine; TensorRT launches it itself.
- ``jit``: TensorRT calls back into Python, which launches the kernel through
  Triton. The engine then needs this module imported, and Triton, at run time.

Both read value / offsets / logits in the dtype of the graph (fp16, bf16 or
fp32) and compute in fp32, as the training kernel does.
"""

import logging
import re
from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

PLUGIN_NAMESPACE = 'mmrotate'
PLUGIN_NAME = 'MSDAFused'
PLUGIN_ID = f'{PLUGIN_NAMESPACE}::{PLUGIN_NAME}'
PLUGIN_MODES = ('aot', 'jit')
# Launch config used when the kernel was not autotuned for a shape (CPU-side tests).
_DEFAULT_TUNING = (2, 1)
# (H, C, L, P, LR, value dtype name) -> (BLOCK_N, num_warps), filled while tracing.
_TUNED: dict[tuple, tuple[int, int]] = {}


@contextmanager
def _tracing_paused() -> Iterator[None]:
    """Pause the TorchScript tracer, under which tensor shapes are tensors.

    The kernel launch needs plain ints (compile-time constants, grid sizes);
    the tracer records the surrounding ``autograd.Function`` call as one node
    either way.
    """
    state = torch._C._get_tracing_state()
    torch._C._set_tracing_state(None)
    try:
        yield
    finally:
        torch._C._set_tracing_state(state)


class _MSDAOp(torch.autograd.Function):
    """The fused Triton forward, exported as one ``MSDAFused`` node.

    ``dims`` is ``(B, N, H, C, L, P, LR)`` as plain ints: inside ``symbolic``
    the tensor shapes of a traced model are often unknown.
    """

    @staticmethod
    def forward(
        ctx: object,
        value: Tensor,
        spatial_shapes: Tensor,
        reference_points: Tensor,
        offsets: Tensor,
        logits: Tensor,
        dims: tuple[int, ...],
        dtype: str,
    ) -> Tensor:
        from mmrotate.models.backbones.cradiov4 import msda_triton_fused as fused

        with _tracing_paused():
            out = fused.msda_fused_fwd(value, spatial_shapes, reference_points, offsets, logits)
            best = getattr(fused._msda_fused_fwd_kernel, 'best_config', None)
            if best is not None:
                _TUNED[(*dims[2:], dtype)] = (int(best.kwargs['BLOCK_N']), int(best.num_warps))
        return out

    @staticmethod
    def symbolic(
        g: object,
        value: object,
        spatial_shapes: object,
        reference_points: object,
        offsets: object,
        logits: object,
        dims: tuple[int, ...],
        dtype: str,
    ) -> object:
        B, N, H, C = dims[:4]
        block_n, num_warps = _TUNED.get((*dims[2:], dtype), _DEFAULT_TUNING)
        out = g.op(
            f'{PLUGIN_NAMESPACE}::{PLUGIN_NAME}',
            value,
            spatial_shapes,
            reference_points,
            offsets,
            logits,
            block_n_i=block_n,
            num_warps_i=num_warps,
            plugin_namespace_s=PLUGIN_NAMESPACE,
            plugin_version_s='1',
        )
        out.setType(value.type().with_sizes([B, N, H, C]))
        return out


class MSDAExportFunction:
    """Stand-in for ``FusedMSDAFunction`` while exporting (same ``apply`` call)."""

    @staticmethod
    def apply(
        value: Tensor,
        spatial_shapes: Tensor,
        reference_points: Tensor,
        offsets: Tensor,
        logits: Tensor,
    ) -> Tensor:
        # int(): under the TorchScript tracer shape entries are traced tensors.
        B, _, H, C = (int(n) for n in value.shape)
        N, L, P = int(offsets.shape[1]), int(offsets.shape[3]), int(offsets.shape[4])
        dims = (B, N, H, C, L, P, int(reference_points.shape[2]))
        dtype = str(value.dtype).replace('torch.', '')
        return _MSDAOp.apply(value, spatial_shapes, reference_points, offsets, logits, dims, dtype)


@contextmanager
def export_msda_as_plugin() -> Iterator[None]:
    """Make ``MSDeformAttn`` (fused Triton path) export as ``MSDAFused`` nodes."""
    from mmrotate.models.backbones.cradiov4 import ms_deform_attn

    if not ms_deform_attn.HAS_TRITON:
        raise RuntimeError('The MSDA plugin export needs Triton; use the gridsample export')
    original = ms_deform_attn.FusedMSDAFunction
    ms_deform_attn.FusedMSDAFunction = MSDAExportFunction
    try:
        yield
    finally:
        ms_deform_attn.FusedMSDAFunction = original


def strip_hidden_params(ptx: str, kernel_name: str, num_params: int) -> str:
    """Drop the trailing kernel parameters Triton adds to every kernel.

    Triton >= 3.3 appends ``global_scratch`` / ``profile_scratch`` pointers to
    the kernel signature. TensorRT passes exactly the declared tensors and
    scalars, so the extra parameters are removed from the PTX entry; they must
    be unused (Triton leaves them unused unless the kernel needs scratch
    memory).
    """
    pattern = re.compile(r'(\.entry\s+' + re.escape(kernel_name) + r'\s*\()(.*?)(\n\))', re.S)
    match = pattern.search(ptx)
    if match is None:
        raise ValueError(f'No entry {kernel_name} in the PTX')
    params = [p.strip().rstrip(',') for p in match.group(2).strip().splitlines() if p.strip()]
    if len(params) < num_params:
        raise ValueError(f'{kernel_name} has {len(params)} parameters, expected {num_params}')
    body = ptx[match.end() :]
    for extra in params[num_params:]:
        name = extra.split()[-1]
        if re.search(re.escape(name) + r'\b', body):
            raise ValueError(f'Hidden kernel parameter {name} is used; cannot strip it')
    kept = ',\n'.join('\t' + p for p in params[:num_params])
    return ptx[: match.start()] + match.group(1) + '\n' + kept + match.group(3) + body


_TRITON_TYPES = {'HALF': 'fp16', 'BF16': 'bf16', 'FLOAT': 'fp32', 'INT64': 'i64', 'INT32': 'i32'}
_SIGNATURE_ORDER = (
    'value_ptr', 'shapes_ptr', 'ref_ptr', 'off_ptr', 'logit_ptr', 'N', 'I', 'ref_sb', 'out_ptr',
)  # fmt: skip


def _triton_type(dtype: object) -> str:
    name = getattr(dtype, 'name', str(dtype))
    if name not in _TRITON_TYPES:
        raise TypeError(f'MSDAFused does not support {name} tensors')
    return _TRITON_TYPES[name]


def compile_msda_kernel(
    dtypes: dict[str, str],
    H: int,
    C: int,
    L: int,
    P: int,
    LR: int,
    block_n: int,
    num_warps: int,
) -> tuple[str, str, int]:
    """Compile the forward kernel to PTX for TensorRT.

    Args:
        dtypes: Triton element types of ``value``, ``shapes``, ``ref``,
            ``off``, ``logit`` and ``out`` (e.g. ``'fp16'``, ``'i64'``).
        H: Heads.
        C: Channels per head.
        L: Feature levels.
        P: Sampling points per head and level.
        LR: Levels of the reference points (1 or ``L``).
        block_n: Queries per program.
        num_warps: Warps per program.

    Returns:
        The kernel name, its PTX (hidden parameters stripped) and the shared
        memory it needs in bytes.
    """
    import triton
    from triton.compiler.compiler import make_backend
    from triton.runtime import driver

    from .msda_kernel import msda_fused_fwd_trt_kernel

    signature = {
        'value_ptr': f'*{dtypes["value"]}',
        'shapes_ptr': f'*{dtypes["shapes"]}',
        'ref_ptr': f'*{dtypes["ref"]}',
        'off_ptr': f'*{dtypes["off"]}',
        'logit_ptr': f'*{dtypes["logit"]}',
        'N': 'i32',
        'I': 'i32',
        'ref_sb': 'i32',
        'out_ptr': f'*{dtypes["out"]}',
    }
    assert tuple(signature) == _SIGNATURE_ORDER
    # Triton's JIT marks 16-byte aligned pointers ("D"), which is what lets the
    # compiler vectorise the value gathers (~10x faster kernel). An ahead-of-time
    # compile has no tensors to look at, so the alignment is declared here;
    # TensorRT and torch both allocate tensors with a larger alignment.
    backend = make_backend(driver.active.get_current_target())
    aligned = backend.parse_attr('D')
    attrs = {(i,): aligned for i, ty in enumerate(signature.values()) if ty.startswith('*')}
    src = triton.compiler.ASTSource(
        fn=msda_fused_fwd_trt_kernel,
        signature=signature,
        constexprs=dict(
            H=H, C=C, L=L, P=P, LR=LR,
            BLOCK_LP=triton.next_power_of_2(L * P),
            BLOCK_C=triton.next_power_of_2(C),
            BLOCK_N=block_n,
        ),
        attrs=attrs,
    )  # fmt: skip
    compiled = triton.compile(src, options={'num_warps': num_warps})
    meta = compiled.metadata
    if meta.global_scratch_size or meta.profile_scratch_size:
        raise RuntimeError('The MSDA kernel needs Triton scratch memory, which TensorRT lacks')
    ptx = strip_hidden_params(compiled.asm['ptx'], meta.name, len(signature))
    return meta.name, ptx, int(meta.shared)


class _CudaArray:
    """``__cuda_array_interface__`` view of a TensorRT plugin tensor."""

    def __init__(self, ptr: int, shape: tuple[int, ...], typestr: str) -> None:
        self.__cuda_array_interface__ = dict(
            shape=shape, typestr=typestr, data=(ptr, False), version=3, strides=None
        )


_TORCH_VIEWS = {
    'HALF': ('<f2', torch.float16),
    'BF16': ('<i2', torch.bfloat16),
    'FLOAT': ('<f4', torch.float32),
    'INT64': ('<i8', torch.int64),
    'INT32': ('<i4', torch.int32),
}


def _as_torch(tensor: object) -> Tensor:
    """A torch view of a ``tensorrt.plugin.Tensor`` (bf16 goes through int16)."""
    typestr, dtype = _TORCH_VIEWS[tensor.dtype.name]
    array = _CudaArray(int(tensor.data_ptr), tuple(tensor.shape), typestr)
    return torch.as_tensor(array, device='cuda').view(dtype)


_REGISTERED = False


def register_plugins() -> None:
    """Register the ``mmrotate::MSDAFused`` plugin with TensorRT (idempotent)."""
    global _REGISTERED
    if _REGISTERED:
        return
    import tensorrt.plugin as trtp
    import triton

    from .msda_kernel import msda_fused_fwd_trt_kernel

    @trtp.register(PLUGIN_ID)
    def msda_desc(
        value: trtp.TensorDesc,
        shapes: trtp.TensorDesc,
        ref: trtp.TensorDesc,
        offsets: trtp.TensorDesc,
        logits: trtp.TensorDesc,
        block_n: int,
        num_warps: int,
    ) -> trtp.TensorDesc:
        out_shape = (
            value.shape_expr[0],
            offsets.shape_expr[1],
            value.shape_expr[2],
            value.shape_expr[3],
        )
        return trtp.from_shape_expr(out_shape, dtype=value.dtype)

    @trtp.impl(PLUGIN_ID)
    def msda_impl(
        value: trtp.Tensor,
        shapes: trtp.Tensor,
        ref: trtp.Tensor,
        offsets: trtp.Tensor,
        logits: trtp.Tensor,
        block_n: int,
        num_warps: int,
        outputs: tuple[trtp.Tensor],
        stream: int,
    ) -> None:
        B, I, H, C = value.shape  # noqa: E741
        _, N, _, L, P, _ = offsets.shape
        LR = ref.shape[2]
        ref_sb = 0 if ref.shape[0] == 1 else N * LR * 2
        with torch.cuda.stream(torch.cuda.ExternalStream(stream)):
            msda_fused_fwd_trt_kernel[(triton.cdiv(N, block_n), B, H)](
                _as_torch(value), _as_torch(shapes), _as_torch(ref), _as_torch(offsets),
                _as_torch(logits), N, I, ref_sb, _as_torch(outputs[0]),
                H=H, C=C, L=L, P=P, LR=LR,
                BLOCK_LP=triton.next_power_of_2(L * P),
                BLOCK_C=triton.next_power_of_2(C),
                BLOCK_N=block_n,
                num_warps=num_warps,
            )  # fmt: skip

    @trtp.aot_impl(PLUGIN_ID)
    def msda_aot_impl(
        value: trtp.TensorDesc,
        shapes: trtp.TensorDesc,
        ref: trtp.TensorDesc,
        offsets: trtp.TensorDesc,
        logits: trtp.TensorDesc,
        block_n: int,
        num_warps: int,
        outputs: tuple[trtp.TensorDesc],
        tactic: int,
    ) -> tuple[str | bytes, str | bytes, trtp.KernelLaunchParams, trtp.SymExprs]:
        def const(expr: object) -> int:
            if not expr.is_constant:
                raise ValueError('MSDAFused needs static head, channel, level and point counts')
            return int(expr.constant_value())

        H, C = const(value.shape_expr[2]), const(value.shape_expr[3])
        L, P = const(offsets.shape_expr[3]), const(offsets.shape_expr[4])
        LR = const(ref.shape_expr[2])
        name, ptx, shared = compile_msda_kernel(
            dict(
                value=_triton_type(value.dtype),
                shapes=_triton_type(shapes.dtype),
                ref=_triton_type(ref.dtype),
                off=_triton_type(offsets.dtype),
                logit=_triton_type(logits.dtype),
                out=_triton_type(outputs[0].dtype),
            ),
            H, C, L, P, LR, block_n, num_warps,
        )  # fmt: skip
        N = offsets.shape_expr[1]
        launch = trtp.KernelLaunchParams()
        launch.grid_x = trtp.cdiv(N, block_n)
        launch.grid_y = value.shape_expr[0]
        launch.grid_z = H
        launch.block_x = num_warps * 32
        launch.shared_mem = shared
        extra = trtp.SymIntExprs(3)
        extra[0] = trtp.SymInt32(N)
        extra[1] = trtp.SymInt32(value.shape_expr[1])
        extra[2] = trtp.SymInt32(0 if const(ref.shape_expr[0]) == 1 else N * (LR * 2))
        return name, ptx, launch, extra

    _REGISTERED = True
    logger.debug('Registered TensorRT plugin %s', PLUGIN_ID)
