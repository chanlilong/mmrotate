# Copyright (c) OpenMMLab. All rights reserved.
"""Dependency-free re-implementation of the LingBot-Vision ViT.

Re-implements the inference path of ``LingBotVisionTransformer`` from
https://github.com/robbyant/lingbot-vision (Apache-2.0), a DINOv3-style ViT
with storage (register) tokens, LayerScale and axial RoPE. Module and
parameter names mirror the released ``model.pt`` so the checkpoint loads with
strict key matching; :mod:`.checkpoint` folds the ``qkv.bias_mask`` buffer into
the bias and drops the (unused) ``mask_token``, so every linear is a plain
``nn.Linear`` (LoRA- and TensorRT-friendly).
"""

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn.bricks import DropPath
from torch import Tensor, nn

# Released architectures (``lingbot_vision/configs/lbot_vision_vit*.yaml``).
LINGBOT_ARCHS: dict[str, dict] = {
    'vit_small': dict(embed_dim=384, depth=12, num_heads=6),
    'vit_base': dict(embed_dim=768, depth=12, num_heads=12),
    'vit_large': dict(embed_dim=1024, depth=24, num_heads=16),
    'vit_giant2': dict(embed_dim=1536, depth=40, num_heads=24, ffn_layer='swiglu', qkv_bias=False),
}

RopeTables = tuple[Tensor, Tensor]


class RopePositionEmbedding(nn.Module):
    """Axial 2D rotary position embedding (no learnable parameters).

    Patch centres are normalised to ``[-1, 1]`` per axis (``'separate'``) or by
    the shorter / longer side. Height and width use independent frequencies.
    ``periods`` is a persistent buffer, as in the checkpoint. The tables are
    always computed in fp32, whatever the module dtype.

    Args:
        embed_dim: Token dimension.
        num_heads: Attention heads (the tables cover one head).
        base: Period base; ``periods = base ** (2 i / (D_head / 2))``.
        normalize_coords: ``'separate'`` | ``'min'`` | ``'max'``.
        shift_coords: Train-mode random coordinate shift (augmentation).
        jitter_coords: Train-mode random per-axis log-uniform scaling.
        rescale_coords: Train-mode random isotropic log-uniform scaling.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        base: float = 100.0,
        normalize_coords: str = 'separate',
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
    ) -> None:
        super().__init__()
        if embed_dim % (4 * num_heads):
            raise ValueError('embed_dim must be divisible by 4 * num_heads')
        if normalize_coords not in ('separate', 'min', 'max'):
            raise ValueError(f'Unknown normalize_coords: {normalize_coords}')
        self.d_head = embed_dim // num_heads
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords
        n = self.d_head // 4
        periods = base ** (2 * torch.arange(n, dtype=torch.float32) / (self.d_head // 2))
        self.register_buffer('periods', periods, persistent=True)

    def forward(self, h: int, w: int) -> RopeTables:
        """``(sin, cos)`` tables of shape ``(h * w, D_head)`` in fp32."""
        dd = dict(device=self.periods.device, dtype=torch.float32)
        if self.normalize_coords == 'separate':
            norm_h, norm_w = h, w
        else:
            norm_h = norm_w = min(h, w) if self.normalize_coords == 'min' else max(h, w)
        coords_h = torch.arange(0.5, h, **dd) / norm_h
        coords_w = torch.arange(0.5, w, **dd) / norm_w
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'), dim=-1)
        coords = 2.0 * coords.flatten(0, 1) - 1.0  # (h*w, 2)

        if self.training and self.shift_coords is not None:
            coords = coords + torch.empty(2, **dd).uniform_(-self.shift_coords, self.shift_coords)
        if self.training and self.jitter_coords is not None:
            log_j = math.log(self.jitter_coords)
            coords = coords * torch.empty(2, **dd).uniform_(-log_j, log_j).exp()
        if self.training and self.rescale_coords is not None:
            log_r = math.log(self.rescale_coords)
            coords = coords * torch.empty(1, **dd).uniform_(-log_r, log_r).exp()

        periods = self.periods.float()
        angles = 2 * math.pi * coords[:, :, None] / periods[None, None, :]  # (h*w, 2, D/4)
        angles = angles.flatten(1, 2).tile(2)  # (h*w, D)
        return torch.sin(angles), torch.cos(angles)


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: Tensor, rope: RopeTables) -> Tensor:
    """Rotate the trailing ``h*w`` tokens of ``x`` (B, heads, N, D); the prefix
    (cls + storage tokens) has no position and is left unchanged."""
    sin, cos = rope
    prefix = x.shape[-2] - sin.shape[-2]
    if prefix < 0:
        raise ValueError('fewer tokens than RoPE positions')
    dtype = x.dtype
    x = x.to(sin.dtype)
    patches = x[..., prefix:, :]
    patches = patches * cos + _rotate_half(patches) * sin
    return torch.cat([x[..., :prefix, :], patches], dim=-2).to(dtype)


class Attention(nn.Module):
    """Multi-head self attention with RoPE on q / k (fused ``qkv`` + ``proj``)."""

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, rope: RopeTables | None = None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if rope is not None:
            q, k = apply_rope(q, rope), apply_rope(k, rope)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class Mlp(nn.Module):
    """Two layer MLP with exact GELU."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class SwiGLUFFN(nn.Module):
    """SwiGLU FFN (ViT-g); hidden width ``2/3 * hidden_dim`` rounded up to ``align_to``."""

    def __init__(self, dim: int, hidden_dim: int, align_to: int = 8) -> None:
        super().__init__()
        d = int(hidden_dim * 2 / 3)
        d += -d % align_to
        self.w1 = nn.Linear(dim, d)
        self.w2 = nn.Linear(dim, d)
        self.w3 = nn.Linear(d, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), init_values))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class Block(nn.Module):
    """Pre-norm transformer block with LayerScale and RoPE attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        ffn_layer: str = 'mlp',
        layerscale_init: float = 1e-5,
        norm_eps: float = 1e-5,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = Attention(dim, num_heads, qkv_bias=qkv_bias)
        self.ls1 = LayerScale(dim, layerscale_init)
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        hidden = int(dim * ffn_ratio)
        if ffn_layer == 'mlp':
            self.mlp: nn.Module = Mlp(dim, hidden)
        elif ffn_layer == 'swiglu':
            self.mlp = SwiGLUFFN(dim, hidden)
        else:
            raise ValueError(f'Unknown ffn_layer: {ffn_layer}')
        self.ls2 = LayerScale(dim, layerscale_init)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor, rope: RopeTables | None = None) -> Tensor:
        x = x + self.drop_path(self.ls1(self.attn(self.norm1(x), rope)))
        return x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))


class PatchEmbed(nn.Module):
    """Non-overlapping conv patchify: ``(B, C, H, W) -> (B, h, w, D)``."""

    def __init__(self, patch_size: int, in_chans: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x).permute(0, 2, 3, 1)


class LingBotTrunk(nn.Module):
    """Container whose names match the released ``model.pt`` state dict."""

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        patch_size: int,
        in_chans: int,
        n_storage_tokens: int,
        ffn_ratio: float,
        qkv_bias: bool,
        ffn_layer: str,
        layerscale_init: float,
        norm_eps: float,
        drop_path_rate: float,
        rope_cfg: dict,
    ) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.storage_tokens = (
            nn.Parameter(torch.zeros(1, n_storage_tokens, embed_dim)) if n_storage_tokens else None
        )
        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)
        self.rope_embed = RopePositionEmbedding(embed_dim, num_heads, **rope_cfg)
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    ffn_ratio=ffn_ratio,
                    qkv_bias=qkv_bias,
                    ffn_layer=ffn_layer,
                    layerscale_init=layerscale_init,
                    norm_eps=norm_eps,
                    drop_path=drop_path_rate,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=norm_eps)

        # Random init (only matters without a checkpoint, e.g. in tests).
        nn.init.normal_(self.cls_token, std=0.02)
        if self.storage_tokens is not None:
            nn.init.normal_(self.storage_tokens, std=0.02)
        for m in self.blocks.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class LingBotViT(nn.Module):
    """LingBot-Vision ViT with the plain-ViT interface of the ViT-CoMer adapter.

    The pretrained modules live in ``self.model`` (:class:`LingBotTrunk`), whose
    state dict matches the ``model`` entry of the released ``model.pt``.
    ``patch_embed`` caches the RoPE tables of the current patch grid, which
    ``run_blocks`` passes to every block (like ``TimmViT``).

    Args:
        arch: One of :data:`LINGBOT_ARCHS` (``vit_small`` / ``vit_base`` /
            ``vit_large`` / ``vit_giant2``).
        patch_size: Patch size in pixels.
        in_chans: Input channels.
        n_storage_tokens: Register tokens after the cls token.
        ffn_ratio: Hidden / embed ratio of the FFN.
        layerscale_init: LayerScale init value (overwritten by the checkpoint).
        norm_eps: LayerNorm epsilon (``layernormbf16`` = 1e-5).
        rope_base: RoPE period base.
        rope_normalize_coords: RoPE coordinate normalisation.
        rope_shift_coords: Train-mode RoPE augmentation (off by default).
        rope_jitter_coords: Train-mode RoPE augmentation (off by default).
        rope_rescale_coords: Train-mode RoPE augmentation (pretraining used
            2; off by default so detection fine-tuning is deterministic).
        drop_path_rate: Stochastic depth (same rate for every block).
        with_cp: Gradient checkpointing per block (only when grads flow).
        **overrides: Override ``embed_dim`` / ``depth`` / ``num_heads`` /
            ``ffn_layer`` / ``qkv_bias`` of the preset (e.g. tiny test models).
    """

    def __init__(
        self,
        arch: str = 'vit_large',
        patch_size: int = 16,
        in_chans: int = 3,
        n_storage_tokens: int = 4,
        ffn_ratio: float = 4.0,
        layerscale_init: float = 1e-5,
        norm_eps: float = 1e-5,
        rope_base: float = 100.0,
        rope_normalize_coords: str = 'separate',
        rope_shift_coords: float | None = None,
        rope_jitter_coords: float | None = None,
        rope_rescale_coords: float | None = None,
        drop_path_rate: float = 0.0,
        with_cp: bool = False,
        **overrides: object,
    ) -> None:
        super().__init__()
        if arch not in LINGBOT_ARCHS:
            raise ValueError(f'Unknown arch {arch!r}, expected one of {sorted(LINGBOT_ARCHS)}')
        spec = {'ffn_layer': 'mlp', 'qkv_bias': True, **LINGBOT_ARCHS[arch]}
        unknown = set(overrides) - set(spec)
        if unknown:
            raise TypeError(f'Unexpected arguments {sorted(unknown)}')
        spec.update(overrides)

        self.embed_dim: int = int(spec['embed_dim'])
        self.depth: int = int(spec['depth'])
        self.patch_size = patch_size
        self.n_storage_tokens = n_storage_tokens
        self.with_cp = with_cp
        self.model = LingBotTrunk(
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=int(spec['num_heads']),
            patch_size=patch_size,
            in_chans=in_chans,
            n_storage_tokens=n_storage_tokens,
            ffn_ratio=ffn_ratio,
            qkv_bias=bool(spec['qkv_bias']),
            ffn_layer=str(spec['ffn_layer']),
            layerscale_init=layerscale_init,
            norm_eps=norm_eps,
            drop_path_rate=drop_path_rate,
            rope_cfg=dict(
                base=rope_base,
                normalize_coords=rope_normalize_coords,
                shift_coords=rope_shift_coords,
                jitter_coords=rope_jitter_coords,
                rescale_coords=rope_rescale_coords,
            ),
        )
        self._rope: RopeTables | None = None

    @property
    def blocks(self) -> nn.ModuleList:
        return self.model.blocks

    @property
    def norm(self) -> nn.LayerNorm:
        return self.model.norm

    @property
    def num_skip(self) -> int:
        """Prefix tokens (cls + storage) that carry no spatial position."""
        return 1 + self.n_storage_tokens

    def patch_embed(self, x: Tensor) -> tuple[Tensor, int, int]:
        """Image -> tokens ``(B, num_skip + h*w, C)`` and the patch grid."""
        if x.shape[-2] % self.patch_size or x.shape[-1] % self.patch_size:
            raise ValueError(
                f'Input size {tuple(x.shape[-2:])} must be a multiple of {self.patch_size}.'
            )
        m = self.model
        x = m.patch_embed(x)
        B, h, w, _ = x.shape
        prefix = [m.cls_token.expand(B, -1, -1)]
        if m.storage_tokens is not None:
            prefix.append(m.storage_tokens.expand(B, -1, -1))
        x = torch.cat([*prefix, x.flatten(1, 2)], dim=1)
        self._rope = m.rope_embed(h, w)
        return x, h, w

    @staticmethod
    def _run_block(blk: nn.Module, x: Tensor, rope: RopeTables | None) -> Tensor:
        return blk(x, rope)

    def run_blocks(self, x: Tensor, start: int, end: int) -> Tensor:
        """Run blocks ``start..end`` (inclusive) on the full token sequence."""
        for blk in self.blocks[start : end + 1]:
            if self.with_cp and x.requires_grad:
                x = cp.checkpoint(self._run_block, blk, x, self._rope, use_reentrant=False)
            else:
                x = self._run_block(blk, x, self._rope)
        return x

    def forward_features(
        self, x: Tensor, out_indices: Sequence[int], final_norm: bool = True
    ) -> list[Tensor]:
        """NCHW spatial features after each block in ``out_indices``.

        ``final_norm`` applies the last LayerNorm to every output, as the
        LingBot features are read after it.
        """
        B = x.shape[0]
        out_indices = [i % self.depth for i in out_indices]
        x, h, w = self.patch_embed(x)
        feats = {}
        last = -1
        for idx in sorted(set(out_indices)):
            x = self.run_blocks(x, last + 1, idx)
            last = idx
            y = self.norm(x) if final_norm else x
            feats[idx] = y[:, self.num_skip :].transpose(1, 2).reshape(B, -1, h, w)
        return [feats[i] for i in out_indices]
