# Copyright (c) OpenMMLab. All rights reserved.
"""Load LingBot-Vision weights straight from a HuggingFace download.

The HF repos (``robbyant/lingbot-vision-vit-*``) ship ``model.pt`` holding
``{'model': state_dict}``. The state dict is converted for :class:`LingBotViT`:

* ``blocks.*.attn.qkv.bias_mask`` (1 / 0 / 1 over q / k / v, i.e. no key bias)
  is folded into ``qkv.bias``, so ``qkv`` is a plain ``nn.Linear``;
* ``mask_token`` (masked-image-modelling only) is dropped.
"""

import os.path as osp

import torch
from mmengine.logging import MMLogger
from torch import Tensor, nn

_WEIGHT_FILES = ('model.pt', 'model.safetensors', 'pytorch_model.bin')
_WRAPPER_KEYS = ('model', 'teacher', 'model_state', 'state_dict', 'backbone')
_DROPPED_KEYS = ('mask_token',)
_BIAS_MASK = '.bias_mask'


def resolve_lingbot_checkpoint(path: str) -> str:
    """Accept a HF model directory or a weights file and return the file."""
    if osp.isdir(path):
        for name in _WEIGHT_FILES:
            candidate = osp.join(path, name)
            if osp.isfile(candidate):
                return candidate
        raise FileNotFoundError(f'No {_WEIGHT_FILES} found in {path}')
    if not osp.isfile(path):
        raise FileNotFoundError(path)
    return path


def _unwrap(state: dict) -> dict:
    for key in _WRAPPER_KEYS:
        if isinstance(state.get(key), dict):
            return state[key]
    return state


def convert_lingbot_state_dict(state: dict) -> dict[str, Tensor]:
    """Unwrap, strip ``_orig_mod.`` / ``backbone.`` and fold ``bias_mask``."""
    state = _unwrap(state)
    state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    if any(k.startswith('backbone.') for k in state):
        state = {k[len('backbone.') :]: v for k, v in state.items() if k.startswith('backbone.')}
    out = {}
    for key, value in state.items():
        if key in _DROPPED_KEYS or key.endswith(_BIAS_MASK):
            continue
        mask = state.get(key + '_mask') if key.endswith('.bias') else None
        out[key] = value * mask.to(value.dtype) if mask is not None else value
    return out


def load_lingbot_state_dict(path: str) -> dict[str, Tensor]:
    """Read a LingBot-Vision checkpoint (``.pt`` / ``.bin`` / ``.safetensors``)."""
    path = resolve_lingbot_checkpoint(path)
    if path.endswith('.safetensors'):
        from safetensors.torch import load_file

        state = load_file(path, device='cpu')
    else:
        state = torch.load(path, map_location='cpu', weights_only=True)
    return convert_lingbot_state_dict(state)


def load_lingbot_weights(vit: nn.Module, checkpoint: str) -> None:
    """Load LingBot weights into ``vit.model`` with strict key matching.

    Only the ``lora_*`` parameters of an injected LoRA may be missing.
    """
    state = load_lingbot_state_dict(checkpoint)
    missing, unexpected = vit.model.load_state_dict(state, strict=False)
    missing = [k for k in missing if '.lora_' not in k]
    if missing or unexpected:
        raise RuntimeError(
            f'Error loading LingBot-Vision weights from {checkpoint}: '
            f'missing keys {missing}, unexpected keys {unexpected}'
        )
    MMLogger.get_current_instance().info(f'Loaded LingBot-Vision weights from {checkpoint}')
