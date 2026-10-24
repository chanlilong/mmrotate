# Copyright (c) OpenMMLab. All rights reserved.
"""Dependency-free re-implementation of the NVIDIA C-RADIO ViT.

Module and parameter names mirror the HuggingFace ``RADIOModel`` checkpoint
(after stripping the ``radio_model.`` prefix) so that the released
``model.safetensors`` loads with ``strict=True`` and no conversion.
"""

from collections.abc import Sequence

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn.bricks import DropPath
from torch import Tensor, nn

# OpenAI CLIP normalisation used by RADIO's default input conditioner.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class InputConditioner(nn.Module):
    """Normalises ``[0, 1]`` RGB images with the stored mean / std."""

    def __init__(
        self,
        mean: Sequence[float] = CLIP_MEAN,
        std: Sequence[float] = CLIP_STD,
    ) -> None:
        super().__init__()
        self.register_buffer('norm_mean', torch.tensor(mean).view(-1, 1, 1))
        self.register_buffer('norm_std', torch.tensor(std).view(-1, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        return (x - self.norm_mean) / self.norm_std


class Attention(nn.Module):
    """Multi-head self attention (timm layout: fused ``qkv`` + ``proj``)."""

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class Mlp(nn.Module):
    """Two layer MLP with exact GELU."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    """Pre-norm transformer block without layer scale."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads, qkv_bias=qkv_bias)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class ClsToken(nn.Module):
    """Learned summary (cls) tokens followed by register tokens."""

    def __init__(self, dim: int, num_tokens: int, num_registers: int) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.num_registers = num_registers
        self.token = nn.Parameter(torch.zeros(num_tokens + num_registers, dim))

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat([self.token.unsqueeze(0).expand(x.shape[0], -1, -1), x], dim=1)


class ViTPatchGenerator(nn.Module):
    """Patchify + linear embed + CPE position embedding + prefix tokens."""

    def __init__(
        self,
        patch_size: int,
        embed_dim: int,
        max_img_size: int,
        num_cls_tokens: int,
        num_registers: int,
        in_chans: int = 3,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.num_rows = self.num_cols = max_img_size // patch_size
        self.embedder = nn.Linear(in_chans * patch_size**2, embed_dim, bias=False)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_rows * self.num_cols, embed_dim))
        self.cls_token = ClsToken(embed_dim, num_cls_tokens, num_registers)

    @property
    def num_cls_tokens(self) -> int:
        return self.cls_token.num_tokens

    @property
    def num_skip(self) -> int:
        return self.cls_token.num_tokens + self.cls_token.num_registers

    def embed_patches(self, x: Tensor) -> tuple[Tensor, int, int]:
        B, C, H, W = x.shape
        p = self.patch_size
        h, w = H // p, W // p
        # (B, C, h, p, w, p) -> (B, h*w, C*p*p), feature order (c, py, px)
        x = x.reshape(B, C, h, p, w, p).permute(0, 2, 4, 1, 3, 5).reshape(B, h * w, C * p * p)
        return self.embedder(x), h, w

    def get_pos_embed(self, h: int, w: int) -> Tensor:
        """Deterministic (eval-mode) CPE position embedding for a h x w grid."""
        pos_embed = self.pos_embed
        if (h, w) == (self.num_rows, self.num_cols):
            return pos_embed
        pos_embed = pos_embed.reshape(1, self.num_rows, self.num_cols, -1).permute(0, 3, 1, 2)
        max_dim = max(h, w)
        pos_embed = F.interpolate(
            pos_embed.float(), size=(max_dim, max_dim), mode='bilinear', align_corners=False
        ).to(pos_embed.dtype)
        pos_embed = pos_embed[..., :h, :w]
        return pos_embed.flatten(2).transpose(1, 2)

    def forward(self, x: Tensor) -> tuple[Tensor, int, int]:
        """Return tokens ``(B, num_skip + h*w, C)`` and the patch grid size."""
        patches, h, w = self.embed_patches(x)
        patches = patches + self.get_pos_embed(h, w)
        return self.cls_token(patches), h, w


class _ViTTrunk(nn.Module):
    """Container whose attribute names match ``radio_model.model.*``."""

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        patch_size: int,
        max_img_size: int,
        num_cls_tokens: int,
        num_registers: int,
        drop_path_rate: float,
    ) -> None:
        super().__init__()
        self.patch_generator = ViTPatchGenerator(
            patch_size, embed_dim, max_img_size, num_cls_tokens, num_registers
        )
        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.Sequential(
            *[Block(embed_dim, num_heads, mlp_ratio, drop_path=dpr[i]) for i in range(depth)]
        )


class RADIOVisionTransformer(nn.Module):
    """C-RADIO vision transformer (defaults: C-RADIOv4-H).

    Args:
        embed_dim: Token dimension.
        depth: Number of transformer blocks.
        num_heads: Attention heads per block.
        mlp_ratio: Hidden / embed ratio of the MLP.
        patch_size: Patch size in pixels.
        max_img_size: Resolution the CPE position grid was trained for.
        num_cls_tokens: Number of summary tokens.
        num_registers: Number of register tokens.
        drop_path_rate: Stochastic depth rate (linearly increasing).
        normalize_input: Apply the CLIP input conditioner (expects ``[0, 1]``
            RGB). Disable when the data preprocessor already normalises.
        with_cp: Use gradient checkpointing for each block.
    """

    def __init__(
        self,
        embed_dim: int = 1280,
        depth: int = 32,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        patch_size: int = 16,
        max_img_size: int = 2048,
        num_cls_tokens: int = 3,
        num_registers: int = 7,
        drop_path_rate: float = 0.0,
        normalize_input: bool = False,
        with_cp: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.patch_size = patch_size
        self.normalize_input = normalize_input
        self.with_cp = with_cp
        self.input_conditioner = InputConditioner()
        self.model = _ViTTrunk(
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            patch_size,
            max_img_size,
            num_cls_tokens,
            num_registers,
            drop_path_rate,
        )
        self.register_buffer('summary_idxs', torch.tensor([0, 1], dtype=torch.int64))

    @property
    def blocks(self) -> nn.Sequential:
        return self.model.blocks

    @property
    def num_skip(self) -> int:
        return self.model.patch_generator.num_skip

    def patch_embed(self, x: Tensor) -> tuple[Tensor, int, int]:
        """Image -> tokens ``(B, num_skip + h*w, C)`` and the patch grid."""
        if x.shape[-2] % self.patch_size or x.shape[-1] % self.patch_size:
            raise ValueError(
                f'Input size {tuple(x.shape[-2:])} must be a multiple of {self.patch_size}.'
            )
        if self.normalize_input:
            x = self.input_conditioner(x)
        return self.model.patch_generator(x)

    def run_blocks(self, x: Tensor, start: int, end: int) -> Tensor:
        """Run blocks ``start..end`` (inclusive) on the full token sequence."""
        for blk in self.blocks[start : end + 1]:
            if self.with_cp and x.requires_grad:
                x = cp.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return x

    def forward_features(self, x: Tensor, out_indices: Sequence[int]) -> list[Tensor]:
        """Return NCHW spatial features after each block in ``out_indices``."""
        B = x.shape[0]
        out_indices = [i % self.depth for i in out_indices]
        x, h, w = self.patch_embed(x)
        outs = []
        last = -1
        for idx in sorted(set(out_indices)):
            x = self.run_blocks(x, last + 1, idx)
            last = idx
            outs.append(x[:, self.num_skip :].transpose(1, 2).reshape(B, -1, h, w))
        lookup = dict(zip(sorted(set(out_indices)), outs, strict=True))
        return [lookup[i] for i in out_indices]

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """HF-compatible output: ``(summary (B, 2C), features (B, h*w, C))``."""
        x, _, _ = self.patch_embed(x)
        x = self.run_blocks(x, 0, self.depth - 1)
        summary = x[:, self.summary_idxs].flatten(1)
        return summary, x[:, self.num_skip :]
