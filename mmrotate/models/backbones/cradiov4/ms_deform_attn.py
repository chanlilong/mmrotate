# Copyright (c) OpenMMLab. All rights reserved.
"""Multi-scale deformable attention (Deformable-DETR) for ViT-CoMer.

The sampling kernel is the Triton implementation vendored from
https://github.com/rziga/msda-triton (MIT), so no CUDA extension has to be
compiled. A pure PyTorch ``grid_sample`` path is used on CPU or on request.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.amp import custom_bwd, custom_fwd
from torch.autograd.function import Function, FunctionCtx, once_differentiable

try:
    from .msda_triton_kernels import (
        triton_multi_scale_deformable_attention_bwd,
        triton_multi_scale_deformable_attention_fwd,
    )

    HAS_TRITON = True
except ImportError:  # pragma: no cover - triton ships with CUDA torch builds
    HAS_TRITON = False

# Deformable-DETR semantics: zero padding, align_corners=False.
_PADDING_MODE = 'zeros'
_ALIGN_CORNERS = False


def msda_pytorch(
    value: Tensor,
    spatial_shapes: Tensor,
    sampling_locations: Tensor,
    attention_weights: Tensor,
) -> Tensor:
    """Reference implementation.

    Args:
        value: ``(B, I, H, C)`` flattened pyramid, ``I = sum(h_l * w_l)``.
        spatial_shapes: ``(L, 2)`` (h, w) per level.
        sampling_locations: ``(B, N, H, L, P, 2)`` normalised (x, y).
        attention_weights: ``(B, N, H, L, P)``.

    Returns:
        ``(B, N, H, C)``.
    """
    B, _, H, C = value.shape
    _, N, _, L, P, _ = sampling_locations.shape
    levels = value.split([int(h) * int(w) for h, w in spatial_shapes.tolist()], dim=1)
    grids = 2 * sampling_locations - 1
    samples = []
    for lvl, (h, w) in enumerate(spatial_shapes.tolist()):
        v = levels[lvl].permute(0, 2, 3, 1).reshape(B * H, C, int(h), int(w))
        g = grids[:, :, :, lvl].permute(0, 2, 1, 3, 4).reshape(B * H, N, P, 2)
        s = F.grid_sample(
            v, g, mode='bilinear', padding_mode=_PADDING_MODE, align_corners=_ALIGN_CORNERS
        )
        samples.append(s.reshape(B, H, C, N, P).permute(0, 3, 1, 4, 2))
    stacked = torch.stack(samples, dim=3)  # (B, N, H, L, P, C)
    return (attention_weights[..., None] * stacked).sum(dim=(3, 4))


class _TritonMSDAFunction(Function):
    @staticmethod
    @custom_fwd(device_type='cuda', cast_inputs=torch.float32)
    def forward(
        ctx: FunctionCtx,
        value: Tensor,
        spatial_shapes: Tensor,
        sampling_locations: Tensor,
        attention_weights: Tensor,
    ) -> Tensor:
        ctx.save_for_backward(value, spatial_shapes, sampling_locations, attention_weights)
        return triton_multi_scale_deformable_attention_fwd(
            value,
            spatial_shapes,
            sampling_locations,
            attention_weights,
            _PADDING_MODE,
            _ALIGN_CORNERS,
        )

    @staticmethod
    @once_differentiable
    @custom_bwd(device_type='cuda')
    def backward(ctx: FunctionCtx, grad_out: Tensor) -> tuple[Tensor | None, ...]:
        value, spatial_shapes, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_loc, grad_attn = triton_multi_scale_deformable_attention_bwd(
            grad_out.contiguous(),
            value,
            spatial_shapes,
            sampling_locations,
            attention_weights,
            _PADDING_MODE,
            _ALIGN_CORNERS,
        )
        return grad_value, None, grad_loc, grad_attn


def multi_scale_deformable_attn(
    value: Tensor,
    spatial_shapes: Tensor,
    sampling_locations: Tensor,
    attention_weights: Tensor,
    backend: str = 'auto',
) -> Tensor:
    """Dispatch to the Triton kernel (CUDA) or the PyTorch reference.

    ``backend`` is one of ``'auto'``, ``'triton'`` or ``'pytorch'``.
    """
    if backend not in ('auto', 'triton', 'pytorch'):
        raise ValueError(f'Unknown MSDA backend {backend!r}')
    use_triton = backend == 'triton' or (backend == 'auto' and HAS_TRITON and value.is_cuda)
    if use_triton:
        return _TritonMSDAFunction.apply(
            value, spatial_shapes, sampling_locations, attention_weights
        )
    dtype = value.dtype
    out = msda_pytorch(
        value.float(), spatial_shapes, sampling_locations.float(), attention_weights.float()
    )
    return out.to(dtype)


class MSDeformAttn(nn.Module):
    """Deformable-DETR attention with ViT-Adapter/CoMer's ``ratio`` option.

    Args:
        d_model: Query / feature dimension.
        n_levels: Number of feature levels.
        n_heads: Attention heads.
        n_points: Sampling points per head and level.
        ratio: Value projection width as a fraction of ``d_model``.
        backend: ``'auto'`` | ``'triton'`` | ``'pytorch'``.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_levels: int = 4,
        n_heads: int = 8,
        n_points: int = 4,
        ratio: float = 1.0,
        backend: str = 'auto',
    ) -> None:
        super().__init__()
        value_dim = int(d_model * ratio)
        if value_dim % n_heads != 0:
            raise ValueError(
                f'd_model*ratio ({value_dim}) must be divisible by n_heads ({n_heads})'
            )
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.ratio = ratio
        self.backend = backend
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, value_dim)
        self.output_proj = nn.Linear(value_dim, d_model)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid_init.view(-1))
        nn.init.constant_(self.attention_weights.weight.data, 0.0)
        nn.init.constant_(self.attention_weights.bias.data, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query: Tensor,
        reference_points: Tensor,
        input_flatten: Tensor,
        input_spatial_shapes: Tensor,
        input_level_start_index: Tensor,
        input_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Same signature as the Deformable-DETR / ViT-CoMer ``MSDeformAttn``.

        Args:
            query: ``(B, Lq, C)``.
            reference_points: ``(B, Lq, n_levels or 1, 2)`` normalised (x, y).
            input_flatten: ``(B, Lin, C)``.
            input_spatial_shapes: ``(n_levels, 2)`` (h, w).
            input_level_start_index: ``(n_levels,)``; kept for API parity.
            input_padding_mask: ``(B, Lin)``, True for padded elements.
        """
        del input_level_start_index  # derived from the shapes by the kernel
        B, Lq, _ = query.shape
        _, Lin, _ = input_flatten.shape
        H, L, P = self.n_heads, self.n_levels, self.n_points

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], 0.0)
        value = value.view(B, Lin, H, -1)
        offsets = self.sampling_offsets(query).view(B, Lq, H, L, P, 2)
        attn = self.attention_weights(query).view(B, Lq, H, L * P).softmax(-1)
        attn = attn.view(B, Lq, H, L, P)

        if reference_points.shape[-1] != 2:
            raise ValueError('Only 2-d reference points are supported')
        normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
        sampling_locations = (
            reference_points[:, :, None, :, None, :]
            + offsets / normalizer[None, None, None, :, None, :]
        )
        out = multi_scale_deformable_attn(
            value, input_spatial_shapes, sampling_locations, attn, self.backend
        )
        return self.output_proj(out.reshape(B, Lq, -1))
