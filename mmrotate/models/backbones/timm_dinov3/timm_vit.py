# Copyright (c) OpenMMLab. All rights reserved.
"""timm ViT / DINOv3 (Eva + RoPE) wrapper with the plain-ViT interface used by
the ViT-CoMer adapter (``patch_embed`` / ``run_blocks`` / ``num_skip``)."""

from collections.abc import Sequence

import timm
import torch.utils.checkpoint as cp
from mmengine.logging import MMLogger
from torch import Tensor, nn

from ..cradiov4.checkpoint import resolve_checkpoint


def create_timm_model(
    model_name: str, checkpoint: str | None = None, **kwargs: object
) -> nn.Module:
    """Create a timm model, loading weights from a local file or HF snapshot dir.

    ``checkpoint`` may be a timm HF-hub folder (``model.safetensors`` /
    ``pytorch_model.bin``) or a weights file. timm's own pretrained loading is
    used, so key filtering (e.g. for ``features_only``) is handled by timm and
    nothing is downloaded.
    """
    if checkpoint is None:
        return timm.create_model(model_name, pretrained=False, **kwargs)
    file = resolve_checkpoint(checkpoint)
    model = timm.create_model(
        model_name, pretrained=True, pretrained_cfg_overlay=dict(file=file), **kwargs
    )
    MMLogger.get_current_instance().info(f'Loaded timm {model_name} weights from {file}')
    return model


class TimmViT(nn.Module):
    """A timm ViT exposing token-level access to its blocks.

    Works for ``VisionTransformer`` and ``Eva`` (DINOv3, with rotary position
    embeddings). ``patch_embed`` caches the RoPE of the current patch grid,
    which ``run_blocks`` then passes to every block.

    Args:
        model_name: timm model name, e.g. ``vit_small_patch16_dinov3``.
        checkpoint: Local weights (HF snapshot dir or file). ``None`` keeps
            the random init (tests).
        with_cp: Gradient checkpointing per block (only when grads flow).
        **kwargs: Extra ``timm.create_model`` kwargs.
    """

    def __init__(
        self,
        model_name: str = 'vit_small_patch16_dinov3',
        checkpoint: str | None = None,
        with_cp: bool = False,
        **kwargs: object,
    ) -> None:
        super().__init__()
        self.model = create_timm_model(
            model_name, checkpoint, num_classes=0, dynamic_img_size=True, **kwargs
        )
        if getattr(self.model, 'rope_mixed', False):
            raise NotImplementedError('mixed (per-depth) RoPE is not supported')
        self.with_cp = with_cp
        self.embed_dim: int = self.model.embed_dim
        self.patch_size: int = self.model.patch_embed.patch_size[0]
        self.num_skip: int = self.model.num_prefix_tokens
        self.depth = len(self.model.blocks)
        self._rope: Tensor | None = None

    @property
    def blocks(self) -> nn.Sequential:
        return self.model.blocks

    @property
    def norm(self) -> nn.Module:
        return self.model.norm

    def patch_embed(self, x: Tensor) -> tuple[Tensor, int, int]:
        """Image -> tokens ``(B, num_skip + h*w, C)`` and the patch grid."""
        if x.shape[-2] % self.patch_size or x.shape[-1] % self.patch_size:
            raise ValueError(
                f'Input size {tuple(x.shape[-2:])} must be a multiple of {self.patch_size}.'
            )
        x = self.model.patch_embed(x)  # (B, h, w, C) with dynamic_img_size
        h, w = x.shape[1], x.shape[2]
        out = self.model._pos_embed(x)
        x, self._rope = out if isinstance(out, tuple) else (out, None)
        x = self.model.norm_pre(x)
        return x, h, w

    def _run_block(self, blk: nn.Module, x: Tensor) -> Tensor:
        if self._rope is None:
            return blk(x)
        return blk(x, rope=self._rope)

    def run_blocks(self, x: Tensor, start: int, end: int) -> Tensor:
        """Run blocks ``start..end`` (inclusive) on the full token sequence."""
        for blk in self.blocks[start : end + 1]:
            if self.with_cp and x.requires_grad:
                x = cp.checkpoint(self._run_block, blk, x, use_reentrant=False)
            else:
                x = self._run_block(blk, x)
        return x

    def forward_features(
        self, x: Tensor, out_indices: Sequence[int], final_norm: bool = True
    ) -> list[Tensor]:
        """NCHW spatial features after each block in ``out_indices``.

        ``final_norm`` applies the ViT's last LayerNorm to every output, as the
        DINOv3 features are trained to be read after it.
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
