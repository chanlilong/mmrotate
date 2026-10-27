# Copyright (c) OpenMMLab. All rights reserved.
"""LoRA (https://arxiv.org/abs/2106.09685) for the frozen plain ViTs.

The pretrained ``weight`` / ``bias`` of each wrapped ``nn.Linear`` are kept
(same Parameter objects, same state-dict keys, still frozen); only the
low-rank ``lora_A`` / ``lora_B`` are trained. ``lora_B`` starts at zero, so a
freshly injected ViT computes exactly the pretrained function.
"""

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from mmengine.logging import MMLogger
from torch import Tensor, nn

DEFAULT_TARGETS = ('qkv', 'proj', 'fc1', 'fc2')


class LoRALinear(nn.Module):
    """``base(x) + B(A(dropout(x))) * alpha / r`` around an existing linear.

    Not an ``nn.Linear`` subclass on purpose: adapter ``_init_weights`` hooks
    that match ``nn.Linear`` must never re-initialise the pretrained weight.

    Args:
        base: The pretrained linear layer; its parameters are reused as is.
        r: LoRA rank.
        alpha: LoRA scaling numerator (``scaling = alpha / r``).
        dropout: Dropout on the LoRA branch input.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f'LoRA rank must be positive, got {r}')
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.weight = base.weight
        self.bias = base.bias
        factory = dict(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features, **factory))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, **factory))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        out = F.linear(x, self.weight, self.bias)
        lora = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
        return out + lora * self.scaling

    def extra_repr(self) -> str:
        return (
            f'in_features={self.in_features}, out_features={self.out_features}, '
            f'r={self.r}, alpha={self.alpha}'
        )


def inject_lora(
    vit: nn.Module,
    r: int = 8,
    alpha: float | None = None,
    dropout: float = 0.0,
    target_modules: Sequence[str] = DEFAULT_TARGETS,
    block_indices: Sequence[int] | None = None,
) -> int:
    """Wrap the ``target_modules`` linears of ``vit.blocks`` with :class:`LoRALinear`.

    Args:
        vit: A plain ViT exposing ``blocks`` (RADIO or timm wrapper).
        r: LoRA rank.
        alpha: LoRA scaling numerator, defaults to ``r`` (scaling 1).
        dropout: Dropout on the LoRA branch input.
        target_modules: Attribute names of the linears to adapt.
        block_indices: Blocks to adapt (negative indices allowed); all if
            ``None``.

    Returns:
        The number of wrapped linear layers.
    """
    blocks = vit.blocks
    depth = len(blocks)
    indices = range(depth) if block_indices is None else [i % depth for i in block_indices]
    targets = set(target_modules)
    alpha = float(r if alpha is None else alpha)

    wrapped, num_params = 0, 0
    for i in sorted(set(indices)):
        block = blocks[i]
        for name, module in list(block.named_modules()):
            parent_name, _, attr = name.rpartition('.')
            if attr not in targets or not isinstance(module, nn.Linear):
                continue
            parent = block.get_submodule(parent_name)
            # timm Eva with q/k/v biases runs ``F.linear(x, qkv.weight, bias)``
            # and would bypass the LoRA forward; the separate-bias path calls
            # ``qkv(x)`` and adds the same bias.
            if attr == 'qkv' and getattr(parent, 'q_bias', None) is not None:
                parent.qkv_bias_separate = True
            lora = LoRALinear(module, r, alpha, dropout)
            setattr(parent, attr, lora)
            wrapped += 1
            num_params += lora.lora_A.numel() + lora.lora_B.numel()

    if not wrapped:
        raise ValueError(f'No nn.Linear named {sorted(targets)} found in the ViT blocks')
    MMLogger.get_current_instance().info(
        f'LoRA r={r} alpha={alpha:g}: wrapped {wrapped} linear layers in {len(set(indices))} '
        f'blocks, {num_params / 1e6:.3f}M trainable params'
    )
    return wrapped


def has_lora(module: nn.Module) -> bool:
    return any(isinstance(m, LoRALinear) for m in module.modules())


def set_lora_train(module: nn.Module, mode: bool) -> None:
    """Set the train mode of the LoRA layers only (e.g. under a frozen ViT in eval)."""
    for m in module.modules():
        if isinstance(m, LoRALinear):
            m.train(mode)
