# Copyright (c) OpenMMLab. All rights reserved.
"""Export forms of ``F.scaled_dot_product_attention`` that TensorRT fuses.

TensorRT runs attention as one fused kernel (an ``mha`` layer in the engine:
tiled, ``O(S)`` memory, tensor cores) instead of two batched matmuls around a
softmax over a materialised ``S x S`` logit tensor. Measured on an RTX 3090 for
one DINOv3 ViT-S block at 800x800 (batch 8, 2505 tokens): 1.0-1.5 ms fused
against 8.5 ms unfused. The fused kernel exists for fp16 and bf16 tensors (not
fp32; int8 only up to 512 tokens), so three forms are exported:

``fused``
    An ONNX ``Attention`` node (opset 23) on the low-precision tensors of the
    graph, which the TensorRT parser maps to its attention layer. The fastest
    form. In fp16 it also accumulates in fp16: logits beyond the fp16 range
    give NaNs, large logits lose precision.
``fused-fp32acc``
    The same node on q, k and v cast to fp32, result cast back. TensorRT
    (>= 11.3) recognises the casts as "low-precision I/O, fp32 accumulation"
    and still runs the fused kernel (~1.5x the time of ``fused``), with the
    accuracy of fp32 attention. For attention whose logits do not fit fp16:
    the first block of DINOv3 ViTs has q / k activations of ~2000 and logits in
    the millions.
``legacy``
    Plain ONNX ops with fp32 logits and softmax and the second matmul in the
    low-precision dtype (the form of the first version of this package). Never
    fused; used for fp32 engines, masked attention and as the baseline.

The TorchScript exporter stops at opset 20, so the fused forms are exported as
a placeholder node that :func:`upgrade_attention_nodes` turns into a standard
``Attention`` node while raising the model's opset.

:func:`probe_attention` measures every attention call of a model on real
images and :func:`plan_attention` picks ``fused`` or ``fused-fp32acc`` per call.
"""

import logging
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

logger = logging.getLogger(__name__)

ATTENTION_FORMS = ('fused', 'fused-fp32acc', 'legacy')
ATTENTION_MODES = ('auto', *ATTENTION_FORMS)
# Placeholder node of the fused forms, see upgrade_attention_nodes.
PLACEHOLDER_DOMAIN = 'mmrotate'
PLACEHOLDER_OP = 'Attention'
ATTENTION_OPSET = 23
_FP16_MAX = 65504.0


def _scale(query: Tensor, scale: float | None) -> float:
    # int(): under the TorchScript tracer a shape entry is a traced tensor.
    return float(int(query.shape[-1]) ** -0.5 if scale is None else scale)


def _check_plain(dropout_p: float, is_causal: bool, enable_gqa: bool) -> None:
    if is_causal or dropout_p or enable_gqa:
        raise NotImplementedError('Only plain inference attention can be exported')


def _mask(logits: Tensor, attn_mask: Tensor | None) -> Tensor:
    if attn_mask is None:
        return logits
    if attn_mask.dtype == torch.bool:
        return logits.masked_fill(~attn_mask, float('-inf'))
    return logits + attn_mask.to(logits.dtype)


def _plain_attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None, s: float) -> Tensor:
    return _mask((q * s) @ k.transpose(-2, -1), mask).softmax(dim=-1) @ v


def _low_dtype(x: Tensor) -> torch.dtype:
    """The dtype the surrounding graph runs in (the autocast dtype, if on)."""
    if torch.is_autocast_enabled(x.device.type):
        return torch.get_autocast_dtype(x.device.type)
    return x.dtype


class _AttentionOp(torch.autograd.Function):
    """Unmasked attention exported as one ``Attention`` placeholder node.

    The torch forward is always computed in fp32 (then cast to the input
    dtype), so tracing and calibration see the values the fused TensorRT
    kernel is meant to produce, whatever precision the node runs in.
    """

    @staticmethod
    def forward(ctx: object, query: Tensor, key: Tensor, value: Tensor, scale: float) -> Tensor:
        out = _plain_attention(query.float(), key.float(), value.float(), None, scale)
        return out.to(value.dtype)

    @staticmethod
    def symbolic(g: object, query: object, key: object, value: object, scale: float) -> object:
        out = g.op(f'{PLACEHOLDER_DOMAIN}::{PLACEHOLDER_OP}', query, key, value, scale_f=scale)
        q_sizes, v_sizes = query.type().sizes(), value.type().sizes()
        if q_sizes is not None and v_sizes is not None:
            out.setType(value.type().with_sizes([*q_sizes[:-1], v_sizes[-1]]))
        return out


def sdpa_legacy(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> Tensor:
    """fp32 logits and softmax, second matmul in ``value.dtype`` (not fused)."""
    _check_plain(dropout_p, is_causal, enable_gqa)
    with torch.autocast(query.device.type, enabled=False):
        logits = _mask(
            (query.float() * _scale(query, scale)) @ key.float().transpose(-2, -1), attn_mask
        )
        attn = logits.softmax(dim=-1)
    return attn.to(value.dtype) @ value


def sdpa_fused(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> Tensor:
    """Attention node on the low-precision tensors of the graph."""
    _check_plain(dropout_p, is_causal, enable_gqa)
    if attn_mask is not None:  # masks are not exported in fused form
        return sdpa_legacy(query, key, value, attn_mask, scale=scale)
    dtype = _low_dtype(query)
    with torch.autocast(query.device.type, enabled=False):
        return _AttentionOp.apply(
            query.to(dtype), key.to(dtype), value.to(dtype), _scale(query, scale)
        )


def sdpa_fused_fp32acc(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> Tensor:
    """Attention node on fp32 casts of q, k, v; the result is cast back."""
    _check_plain(dropout_p, is_causal, enable_gqa)
    if attn_mask is not None:
        return sdpa_legacy(query, key, value, attn_mask, scale=scale)
    dtype = _low_dtype(query)
    with torch.autocast(query.device.type, enabled=False):
        out = _AttentionOp.apply(query.float(), key.float(), value.float(), _scale(query, scale))
        return out.to(dtype)


_FORMS: dict[str, Callable[..., Tensor]] = {
    'fused': sdpa_fused,
    'fused-fp32acc': sdpa_fused_fp32acc,
    'legacy': sdpa_legacy,
}


class AttentionExporter:
    """Drop-in for ``F.scaled_dot_product_attention`` following a per-call plan.

    Args:
        plan: One form for every call, or one form per attention call of a
            forward pass, in call order (the order is fixed for a traced,
            static-shape model). Calls beyond the plan wrap around, so the
            same exporter serves any number of forward passes.
    """

    def __init__(self, plan: str | Sequence[str]) -> None:
        forms = [plan] if isinstance(plan, str) else list(plan)
        unknown = sorted(set(forms) - set(ATTENTION_FORMS))
        if unknown or not forms:
            raise ValueError(f'Unknown attention forms {unknown}; choose from {ATTENTION_FORMS}')
        self.plan = forms
        self.calls = 0

    def __call__(
        self, query: Tensor, key: Tensor, value: Tensor, *args: object, **kwargs: object
    ) -> Tensor:
        form = self.plan[self.calls % len(self.plan)]
        self.calls += 1
        return _FORMS[form](query, key, value, *args, **kwargs)


@contextmanager
def patch_sdpa(fn: Callable[..., Tensor]) -> Iterator[None]:
    """Replace ``F.scaled_dot_product_attention`` by ``fn`` inside the context."""
    original = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = fn
    try:
        yield
    finally:
        F.scaled_dot_product_attention = original


@dataclass
class AttentionStats:
    """Numerics of one attention call, the worst case over the probed images.

    Errors are RMS errors against fp32 attention, relative to the output RMS.

    Attributes:
        index: Position of the call in the forward pass.
        tokens: Sequence length of the keys.
        head_dim: Channels per head.
        max_q: Largest ``|q|`` (before the ``1 / sqrt(d)`` scale).
        max_k: Largest ``|k|``.
        max_logit: Largest ``|logit|``.
        max_partial: Largest ``sum_i |q_i k_i|``, a bound on every partial sum
            of an fp16 accumulation.
        fp16_finite: An all-fp16 evaluation stayed finite.
        fp16_err: Error of the all-fp16 evaluation (``fused`` in an fp16 graph).
        fp16_io_err: Error of fp32 math on fp16-rounded q, k, v
            (``fused-fp32acc``): the floor for an fp16 graph.
        masked: The call has an attention mask (exported in ``legacy`` form).
    """

    index: int
    tokens: int
    head_dim: int
    max_q: float
    max_k: float
    max_logit: float
    max_partial: float
    fp16_finite: bool
    fp16_err: float
    fp16_io_err: float
    masked: bool = False

    def merge(self, other: 'AttentionStats') -> 'AttentionStats':
        """Worst case of two observations of the same call."""
        return AttentionStats(
            index=self.index,
            tokens=max(self.tokens, other.tokens),
            head_dim=self.head_dim,
            max_q=max(self.max_q, other.max_q),
            max_k=max(self.max_k, other.max_k),
            max_logit=max(self.max_logit, other.max_logit),
            max_partial=max(self.max_partial, other.max_partial),
            fp16_finite=self.fp16_finite and other.fp16_finite,
            fp16_err=max(self.fp16_err, other.fp16_err),
            fp16_io_err=max(self.fp16_io_err, other.fp16_io_err),
            masked=self.masked or other.masked,
        )


class _Probe:
    """Records :class:`AttentionStats` and returns the fp32 result."""

    def __init__(self) -> None:
        self.stats: list[AttentionStats] = []
        self._index = 0

    def new_forward(self) -> None:
        self._index = 0

    def __call__(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> Tensor:
        _check_plain(dropout_p, is_causal, enable_gqa)
        s = _scale(query, scale)
        with torch.autocast(query.device.type, enabled=False):
            q, k, v = query.float(), key.float(), value.float()
            logits = _mask((q * s) @ k.transpose(-2, -1), attn_mask)
            out = logits.softmax(dim=-1) @ v
            rms = out.square().mean().sqrt().clamp_min(1e-12)

            def rel_err(x: Tensor) -> float:
                if not bool(torch.isfinite(x).all()):
                    return float('inf')
                return float((x.float() - out).square().mean().sqrt() / rms)

            qh, kh, vh = q.half(), k.half(), v.half()
            fp16_err = rel_err(_plain_attention(qh, kh, vh, attn_mask, s))
            finite_logits = logits[torch.isfinite(logits)]
            new = AttentionStats(
                index=self._index,
                tokens=key.shape[-2],
                head_dim=key.shape[-1],
                max_q=float(q.abs().max()),
                max_k=float(k.abs().max()),
                max_logit=float(finite_logits.abs().max()) if finite_logits.numel() else 0.0,
                max_partial=float(((q.abs() * s) @ k.abs().transpose(-2, -1)).max()),
                fp16_finite=fp16_err != float('inf'),
                fp16_err=fp16_err,
                fp16_io_err=rel_err(
                    _plain_attention(qh.float(), kh.float(), vh.float(), attn_mask, s)
                ),
                masked=attn_mask is not None,
            )
        if self._index < len(self.stats):
            self.stats[self._index] = self.stats[self._index].merge(new)
        else:
            self.stats.append(new)
        self._index += 1
        return out.to(value.dtype)


@torch.no_grad()
def probe_attention(model: nn.Module, batches: Iterable[Tensor]) -> list[AttentionStats]:
    """Numerics of every attention call of ``model._forward`` over ``batches``.

    Images are run one at a time in fp32, so the probe needs about the memory
    of one image's attention logits.
    """
    device = next(model.parameters()).device
    probe = _Probe()
    with patch_sdpa(probe):
        for batch in batches:
            for image in batch:
                probe.new_forward()
                model._forward(image[None].to(device))
    return probe.stats


def plan_attention(
    stats: Sequence[AttentionStats],
    low_dtype: torch.dtype | None,
    headroom: float = 4.0,
    max_err: float = 0.02,
) -> list[str]:
    """One export form per attention call.

    fp32 graphs and masked calls get ``legacy``. In an fp16 graph a call is
    ``fused`` if an fp16 accumulation is safe, else ``fused-fp32acc``. bf16
    graphs use ``fused``: bf16 has the range of fp32 and TensorRT's bf16 kernel
    accumulates in fp32, so fp32 casts would add nothing (the error of a bf16
    graph comes from q, k and v being rounded to 8 bits of mantissa before the
    attention, 3-8x the error of fp16 I/O on DINOv3 blocks).

    Args:
        stats: Output of :func:`probe_attention`.
        low_dtype: Low-precision dtype of the graph (``None``: fp32 graph).
        headroom: An fp16 call is ``fused`` only if its largest partial sum,
            times this factor, stays below the fp16 maximum ...
        max_err: ... and its all-fp16 evaluation is within this RMS error of
            fp32 attention (relative to the output RMS).
    """
    plan = []
    for s in stats:
        if low_dtype is None or s.masked:
            plan.append('legacy')
        elif low_dtype != torch.float16:
            plan.append('fused')
        else:
            safe = s.fp16_finite and s.max_partial * headroom < _FP16_MAX and s.fp16_err <= max_err
            plan.append('fused' if safe else 'fused-fp32acc')
    return plan


def resolve_attention(
    mode: str,
    model: nn.Module,
    batches: Iterable[Tensor],
    low_dtype: torch.dtype | None,
) -> tuple[list[str], list[dict]]:
    """Export plan for ``mode`` and the probe statistics (as dicts, for the meta file).

    ``mode='auto'`` plans from the probe; any other mode uses that form for
    every call (the probe still runs, to count the calls and report numerics).
    """
    if mode not in ATTENTION_MODES:
        raise ValueError(f'Unknown attention mode {mode!r}; choose from {ATTENTION_MODES}')
    stats = probe_attention(model, batches)
    plan = plan_attention(stats, low_dtype) if mode == 'auto' else [mode] * len(stats)
    for s, form in zip(stats, plan, strict=True):
        logger.info(
            'attention %2d: %d tokens, max |q| %.0f |k| %.0f |logit| %.0f, '
            'fp16 err %.2g (fp16 I/O floor %.2g) -> %s',
            s.index, s.tokens, s.max_q, s.max_k, s.max_logit, s.fp16_err, s.fp16_io_err, form,
        )  # fmt: skip
    return plan, [asdict(s) | {'form': form} for s, form in zip(stats, plan, strict=True)]


# Ops whose meaning changed between the TorchScript exporter's opsets and
# ATTENTION_OPSET in a way a plain opset bump would silently get wrong.
_INCOMPATIBLE_OPS = ('DFT', 'GroupNormalization')
_GRID_SAMPLE_MODES = {b'bilinear': b'linear', b'bicubic': b'cubic'}


def upgrade_attention_nodes(onnx_file: str) -> int:
    """Turn the placeholder nodes of ``onnx_file`` into ONNX ``Attention`` nodes.

    Raises the default opset of the model to :data:`ATTENTION_OPSET` (the
    first with ``Attention``). The ops the exporter emits are unchanged up to
    that opset except ``GridSample``, whose ``mode`` values were renamed in
    opset 20 and are rewritten here. The file is rewritten in place; nothing is
    done (and ``onnx`` is not needed) for graphs without placeholder nodes.

    Returns:
        The number of attention nodes.
    """
    with open(onnx_file, 'rb') as f:
        if PLACEHOLDER_OP.encode() not in f.read():
            return 0
    import onnx

    model = onnx.load(onnx_file, load_external_data=False)
    nodes = [
        n
        for n in model.graph.node
        if n.domain == PLACEHOLDER_DOMAIN and n.op_type == PLACEHOLDER_OP
    ]
    if not nodes:
        return 0
    default = next(o for o in model.opset_import if o.domain in ('', 'ai.onnx'))
    if default.version < ATTENTION_OPSET:
        for node in model.graph.node:
            if node.domain not in ('', 'ai.onnx'):
                continue
            if node.op_type in _INCOMPATIBLE_OPS:
                raise NotImplementedError(
                    f'{node.op_type} changed after opset {default.version}; cannot raise the '
                    f'opset for fused attention. Export with --attention legacy.'
                )
            if node.op_type == 'GridSample' and default.version < 20:
                for attr in node.attribute:
                    if attr.name == 'mode':
                        attr.s = _GRID_SAMPLE_MODES.get(attr.s, attr.s)
        default.version = ATTENTION_OPSET
    for node in nodes:
        node.domain = ''
    onnx.save(model, onnx_file)
    return len(nodes)
