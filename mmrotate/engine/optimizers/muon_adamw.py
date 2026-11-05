# Copyright (c) OpenMMLab. All rights reserved.
"""Hybrid Muon + AdamW optimizer for detectors that mix linear and conv layers.

``torch.optim.Muon`` only accepts 2D parameters. A 1x1 convolution is a linear
map applied at every pixel, so its ``(out, in, 1, 1)`` weight is optimised by
Muon as the ``(out, in)`` matrix it really is. Everything else (k x k and
depthwise convs, norms, biases, embeddings, output layers) uses AdamW.

Two pieces, both usable from a config::

    optim_wrapper:
      constructor: MuonAdamWOptimWrapperConstructor   # routes by module type
      optimizer: {type: MuonAdamW, lr: 2.5e-4, weight_decay: 0.05}
      paramwise_cfg:
        muon_exclude_keys: [rtm_cls, rtm_reg, rtm_ang]

* :class:`MuonAdamWOptimWrapperConstructor` tags each param group with
  ``use_muon``, looking at the module that owns the parameter
  (``nn.Linear`` and 1x1, ``groups=1`` ``nn.Conv2d`` weights).
* :class:`MuonAdamW` runs torch's own Muon update on the tagged groups and
  torch's own AdamW on the rest. Without the constructor it falls back to a
  shape rule (2D, or 1x1 conv with more than one input channel), which cannot
  tell e.g. a 2D embedding ``nn.Parameter`` from a linear weight.
"""

import logging
from collections.abc import Callable, Iterable
from typing import Any

import torch
from mmengine.logging import print_log
from mmengine.optim import DefaultOptimWrapperConstructor, OptimWrapper
from torch import Tensor, nn

from mmrotate.registry import OPTIM_WRAPPER_CONSTRUCTORS, OPTIMIZERS

try:  # torch >= 2.9; the functional form lets us feed 2D views of 1x1 conv weights.
    from torch.optim._muon import muon as _torch_muon
except ImportError:  # pragma: no cover
    _torch_muon = None

# Keller Jordan's quintic Newton-Schulz coefficients (torch.optim.Muon defaults).
DEFAULT_NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)


def muon_compatible(param: Tensor) -> bool:
    """Whether ``param`` is a matrix: 2D, or a 1x1 conv weight ``(out, in, 1, 1)``."""
    return param.ndim == 2 or (param.ndim == 4 and param.shape[2] == 1 and param.shape[3] == 1)


def matrix_shape(param: Tensor) -> tuple[int, int]:
    """``(fan_out, fan_in)`` of the matrix Muon sees for ``param``."""
    return param.shape[0], param.numel() // param.shape[0]


def _default_use_muon(param: Tensor) -> bool:
    """Shape rule used when a group carries no ``use_muon`` flag."""
    return muon_compatible(param) and min(matrix_shape(param)) > 1


@OPTIMIZERS.register_module()
class MuonAdamW(torch.optim.AdamW):
    """Muon for matrix (linear / 1x1 conv) weights, AdamW for everything else.

    One optimizer, so mmengine param schedulers, ``paramwise_cfg``, AMP and
    checkpoints see a single object. Groups with ``use_muon=True`` get torch's
    Muon update (momentum, Newton-Schulz orthogonalisation, shape-adjusted lr,
    decoupled weight decay) on a 2D view of each weight; the other groups get
    torch's AdamW step unchanged.

    Args:
        params: Parameters or param groups. A group's ``use_muon`` key selects
            the branch; groups without it are split by the shape rule.
        lr: Learning rate (both branches, see ``muon_lr_mult``).
        betas: AdamW betas.
        eps: AdamW epsilon.
        weight_decay: Decoupled weight decay (both branches, see
            ``muon_decay_mult``).
        amsgrad: AdamW amsgrad.
        foreach: AdamW foreach implementation (``None`` picks automatically).
        muon_momentum: Muon momentum.
        nesterov: Nesterov momentum for Muon.
        ns_coefficients: Newton-Schulz polynomial coefficients.
        ns_steps: Newton-Schulz iterations.
        muon_eps: Muon normalisation epsilon.
        adjust_lr_fn: Muon lr scaling, one of ``'original'`` (Keller),
            ``'match_rms_adamw'`` (Moonshot: Muon reuses the AdamW lr and
            weight decay) or ``'spectral_unclamped'``.
        muon_lr_mult: Multiplier on the lr of Muon groups.
        muon_decay_mult: Multiplier on the weight decay of Muon groups.
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        *,
        foreach: bool | None = None,
        muon_momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: tuple[float, float, float] = DEFAULT_NS_COEFFICIENTS,
        ns_steps: int = 5,
        muon_eps: float = 1e-7,
        adjust_lr_fn: str | None = 'match_rms_adamw',
        muon_lr_mult: float = 1.0,
        muon_decay_mult: float = 1.0,
    ) -> None:
        if _torch_muon is None:
            raise ImportError(
                f'MuonAdamW needs torch >= 2.9 (torch.optim.Muon), got {torch.__version__}'
            )
        if adjust_lr_fn not in (None, 'original', 'match_rms_adamw', 'spectral_unclamped'):
            raise ValueError(f'Unsupported adjust_lr_fn: {adjust_lr_fn}')
        if not 0.0 <= muon_momentum < 1.0:
            raise ValueError(f'muon_momentum must be in [0, 1), got {muon_momentum}')

        groups = self._split_groups(params)
        for group in groups:
            if not group['use_muon']:
                continue
            for p in group['params']:
                if not muon_compatible(p):
                    raise ValueError(
                        f'Muon needs a 2D or 1x1 conv weight, got shape {tuple(p.shape)}'
                    )
            group['lr'] = group.get('lr', lr) * muon_lr_mult
            group['weight_decay'] = group.get('weight_decay', weight_decay) * muon_decay_mult

        # fused stays off: a fused Adam advertises _step_supports_amp_scaling, which makes
        # GradScaler hand the optimizer still-scaled grads that the Muon branch can't handle.
        super().__init__(
            groups,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            foreach=foreach,
            fused=False,
        )
        muon_defaults = dict(
            use_muon=False,
            muon_momentum=muon_momentum,
            nesterov=nesterov,
            ns_coefficients=tuple(ns_coefficients),
            ns_steps=ns_steps,
            muon_eps=muon_eps,
            adjust_lr_fn=adjust_lr_fn,
        )
        self.defaults.update(muon_defaults)
        for group in self.param_groups:
            for key, value in muon_defaults.items():
                group.setdefault(key, value)

    @staticmethod
    def _split_groups(
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Normalise ``params`` to groups that all carry a ``use_muon`` flag."""
        items = list(params)
        if not items:
            raise ValueError('MuonAdamW got an empty parameter list')
        if not isinstance(items[0], dict):
            items = [{'params': items}]
        groups = []
        for item in items:
            group = dict(item)
            group_params = group['params']
            group_params = (
                [group_params] if isinstance(group_params, Tensor) else list(group_params)
            )
            if 'use_muon' in group:
                group['params'] = group_params
                groups.append(group)
                continue
            muon = [p for p in group_params if _default_use_muon(p)]
            other = [p for p in group_params if not _default_use_muon(p)]
            for use_muon, subset in ((True, muon), (False, other)):
                if subset:
                    groups.append({**group, 'params': subset, 'use_muon': use_muon})
        return groups

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Perform one optimisation step on both branches."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        all_groups = self.param_groups
        adamw_groups = [g for g in all_groups if not g['use_muon']]
        if adamw_groups:
            # torch's AdamW iterates self.param_groups; hand it only its own groups. The
            # state dict is shared, so one state_dict() covers both branches.
            self.param_groups = adamw_groups
            try:
                super().step()
            finally:
                self.param_groups = all_groups
        for group in all_groups:
            if group['use_muon']:
                self._muon_group_step(group)
        return loss

    def _muon_group_step(self, group: dict[str, Any]) -> None:
        params: list[Tensor] = []
        grads: list[Tensor] = []
        bufs: list[Tensor] = []
        for p in group['params']:
            if p.grad is None:
                continue
            if p.grad.is_sparse:
                raise RuntimeError('MuonAdamW does not support sparse gradients')
            # A view, so the in-place update lands in the (out, in, 1, 1) conv weight.
            param_2d = p.view(matrix_shape(p))
            grad_2d = p.grad.reshape(param_2d.shape)
            state = self.state[p]
            if 'momentum_buffer' not in state:
                state['momentum_buffer'] = torch.zeros_like(grad_2d)
                # Adam.__setstate__ (run by load_state_dict) expects every state to have one.
                state['step'] = torch.tensor(0.0)
            state['step'] += 1
            params.append(param_2d)
            grads.append(grad_2d)
            bufs.append(state['momentum_buffer'])
        if not params:
            return
        _torch_muon(
            params,
            grads,
            bufs,
            lr=group['lr'],
            weight_decay=group['weight_decay'],
            momentum=group['muon_momentum'],
            nesterov=group['nesterov'],
            ns_coefficients=tuple(group['ns_coefficients']),
            ns_steps=group['ns_steps'],
            eps=group['muon_eps'],
            adjust_lr_fn=group['adjust_lr_fn'],
            has_complex=False,
        )


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class MuonAdamWOptimWrapperConstructor(DefaultOptimWrapperConstructor):
    """``DefaultOptimWrapperConstructor`` that routes params to Muon by module type.

    Muon gets the ``weight`` of every trainable ``nn.Linear`` (and subclasses)
    and of every ``nn.Conv2d`` with a 1x1 kernel and ``groups=1``. All other
    parameters get AdamW. The usual ``paramwise_cfg`` options keep working; a
    ``custom_keys`` entry may also set ``use_muon`` explicitly, which wins.

    Extra ``paramwise_cfg`` keys (removed before mmengine sees the config):

    * ``muon_include_keys`` (list[str]): parameter-name substrings sent to Muon
      when the parameter is a matrix, e.g. ``[lora_A, lora_B]``.
    * ``muon_exclude_keys`` (list[str]): parameter-name substrings forced to
      AdamW, e.g. output layers ``[rtm_cls, rtm_reg, rtm_ang]``. Wins over
      the type rule and ``muon_include_keys``.
    * ``muon_min_dim`` (int, default 16): matrices whose smaller side is below
      this go to AdamW (orthogonalising a thin matrix, e.g. a rank-8 LoRA
      factor, keeps little of the gradient).
    """

    def __init__(self, optim_wrapper_cfg: dict, paramwise_cfg: dict | None = None) -> None:
        paramwise_cfg = dict(paramwise_cfg or {})
        self.muon_include_keys = tuple(paramwise_cfg.pop('muon_include_keys', ()))
        self.muon_exclude_keys = tuple(paramwise_cfg.pop('muon_exclude_keys', ()))
        self.muon_min_dim = int(paramwise_cfg.pop('muon_min_dim', 16))
        # Without a paramwise_cfg mmengine passes model.parameters() untagged; any non-empty
        # dict makes it build per-parameter groups through add_params().
        super().__init__(optim_wrapper_cfg, paramwise_cfg or {'bypass_duplicate': True})
        self._muon_names: dict[int, str] = {}

    def __call__(self, model: nn.Module) -> OptimWrapper:
        if hasattr(model, 'module'):
            model = model.module
        self._muon_names = self.select_muon_params(model)
        return super().__call__(model)

    def select_muon_params(self, model: nn.Module) -> dict[int, str]:
        """Map ``id(param) -> name`` for every trainable parameter Muon should take."""
        selected: dict[int, str] = {}
        for module_name, module in model.named_modules():
            is_matrix_layer = isinstance(module, nn.Linear) or (
                isinstance(module, nn.Conv2d)
                and tuple(module.kernel_size) == (1, 1)
                and module.groups == 1
            )
            for param_name, param in module.named_parameters(recurse=False):
                name = f'{module_name}.{param_name}' if module_name else param_name
                if not param.requires_grad or id(param) in selected:
                    continue
                by_type = is_matrix_layer and param_name == 'weight'
                by_key = muon_compatible(param) and any(k in name for k in self.muon_include_keys)
                if not (by_type or by_key):
                    continue
                if any(k in name for k in self.muon_exclude_keys):
                    continue
                if min(matrix_shape(param)) < self.muon_min_dim:
                    continue
                selected[id(param)] = name
        return selected

    def add_params(
        self,
        params: list[dict],
        module: nn.Module,
        prefix: str = '',
        is_dcn_module: int | float | None = None,
    ) -> None:
        super().add_params(params, module, prefix, is_dcn_module)
        if prefix:  # recursive call on a child module; tag once at the root
            return
        for group in params:
            group.setdefault('use_muon', all(id(p) in self._muon_names for p in group['params']))
        self._log_routing(params)

    def _log_routing(self, params: list[dict]) -> None:
        counts = {True: [0, 0], False: [0, 0]}
        for group in params:
            for p in group['params']:
                counts[group['use_muon']][0] += 1
                counts[group['use_muon']][1] += p.numel()
        print_log(
            f'MuonAdamW routing: Muon {counts[True][0]} tensors ({counts[True][1] / 1e6:.2f}M '
            f'params), AdamW {counts[False][0]} tensors ({counts[False][1] / 1e6:.2f}M params)',
            logger='current',
        )
        muon_ids = {id(p) for g in params if g['use_muon'] for p in g['params']}
        for pid, name in self._muon_names.items():
            if pid in muon_ids:
                print_log(
                    f'MuonAdamW routing -- {name}: Muon', logger='current', level=logging.DEBUG
                )
