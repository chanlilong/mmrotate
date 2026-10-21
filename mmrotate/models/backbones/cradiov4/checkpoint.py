# Copyright (c) OpenMMLab. All rights reserved.
"""Load C-RADIO weights straight from a HuggingFace download."""

import os.path as osp

import torch
from torch import Tensor

HF_PREFIX = 'radio_model.'
_WEIGHT_FILES = ('model.safetensors', 'pytorch_model.bin')


def resolve_checkpoint(path: str) -> str:
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


def load_radio_state_dict(path: str) -> dict[str, Tensor]:
    """Read a C-RADIO checkpoint and strip the HF ``radio_model.`` prefix.

    Supports ``model.safetensors`` (the HF release) as well as torch pickles
    (``.bin`` / ``.pth``) holding either a bare or a ``state_dict`` mapping.
    """
    path = resolve_checkpoint(path)
    if path.endswith('.safetensors'):
        from safetensors.torch import load_file

        state_dict = load_file(path, device='cpu')
    else:
        state_dict = torch.load(path, map_location='cpu', weights_only=True)
        state_dict = state_dict.get('state_dict', state_dict)
    return {
        (k[len(HF_PREFIX) :] if k.startswith(HF_PREFIX) else k): v for k, v in state_dict.items()
    }
