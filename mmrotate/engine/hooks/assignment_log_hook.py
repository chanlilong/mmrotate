# Copyright (c) OpenMMLab. All rights reserved.
"""Log what the label assigner gives every training object, epoch by epoch.

``tools/assigner_tuning/assigner_probe.py`` measures starvation on one
checkpoint; this hook measures it over a whole training run, so it tells a GT
that is *never* trained (geometry: no prior can reach it) from one that is
starved only sometimes (prediction-aware assigners change their choice as the
model changes). Add it to a config::

    custom_imports = dict(imports=['mmrotate.engine.hooks'], allow_failed_imports=False)
    custom_hooks = [..., dict(type='AssignmentLogHook')]

and summarise ``<work_dir>/assignment_log.npz`` with
``tools/assigner_tuning/assignment_log_report.py``.

The hook wraps ``<head>.assigner.assign`` and ``<head>._get_targets_single``
(to know the image) for the length of the training. Objects are identified by
image path and annotation order, which the pipeline must keep (flips and
rotations do, crops that drop boxes do not: images whose GT count changes are
skipped). Exact duplicate GTs (same class, same box to 0.01 px) are merged, as
in the probe. The file is rewritten after every epoch and holds:

- ``images`` (I,): image paths; ``seen`` (E, I): visits per epoch;
- ``obj_image``, ``obj_size`` (``sqrt(w * h)``, px), ``obj_label``,
  ``obj_density`` (objects in the image), all (N,);
- ``npos`` (E, N): positives per object, summed over the epoch's visits;
- ``best_iou`` (E, N): best rotated IoU between the predicted box of a
  positive and its GT in the epoch (0 when starved).
"""

from collections.abc import Callable

import numpy as np
import torch
from mmcv.ops import box_iou_rotated
from mmdet.models.task_modules import AssignResult
from mmengine.dist import get_rank, get_world_size
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner
from mmengine.structures import InstanceData

from mmrotate.registry import HOOKS


def _boxes(x: object) -> torch.Tensor:
    t = x.tensor if hasattr(x, 'tensor') else x
    return t[..., :5].detach().float()


@HOOKS.register_module()
class AssignmentLogHook(Hook):
    """Per-object, per-epoch positives and positive IoU of the training assigner.

    Args:
        head (str): Attribute of the model that holds the assigner.
        file_name (str): Output name in the work dir, without ``.npz``; a
            ``_rank{r}`` suffix is added in distributed training.
    """

    priority = 'VERY_LOW'

    def __init__(self, head: str = 'bbox_head', file_name: str = 'assignment_log') -> None:
        self.head_name = head
        self.file_name = file_name
        self._log: dict[str, dict[str, np.ndarray]] = {}
        self._meta: dict | None = None
        self._epoch = 0
        self._epochs = 1
        self._skipped = 0
        self._restore: list[Callable[[], None]] = []

    def before_train(self, runner: Runner) -> None:
        """Wrap the head's target computation and assigner."""
        model = runner.model.module if is_model_wrapper(runner.model) else runner.model
        head = getattr(model, self.head_name)
        assigner = head.assigner
        self._epochs = runner.max_epochs
        orig_targets = head._get_targets_single
        orig_assign = assigner.assign

        def targets(*args: object, **kwargs: object) -> tuple:
            # the image meta is the only dict with an ``img_path`` among the arguments
            metas = [a for a in (*args, *kwargs.values()) if isinstance(a, dict)]
            self._meta = next((m for m in metas if 'img_path' in m), None)
            try:
                return orig_targets(*args, **kwargs)
            finally:
                self._meta = None

        def assign(
            pred_instances: InstanceData,
            gt_instances: InstanceData,
            *args: object,
            **kwargs: object,
        ) -> AssignResult:
            result = orig_assign(pred_instances, gt_instances, *args, **kwargs)
            if self._meta is not None:
                self.record(self._meta['img_path'], pred_instances, gt_instances, result.gt_inds)
            return result

        head._get_targets_single = targets
        assigner.assign = assign
        self._restore = [
            lambda: delattr(head, '_get_targets_single'),
            lambda: delattr(assigner, 'assign'),
        ]

    def before_train_epoch(self, runner: Runner) -> None:
        """Remember the epoch the next assignments belong to."""
        self._epoch = min(runner.epoch, self._epochs - 1)

    def after_train_epoch(self, runner: Runner) -> None:
        """Write the log so far."""
        self.dump(runner.work_dir)
        if self._skipped:
            runner.logger.warning(
                f'AssignmentLogHook: {self._skipped} assignments skipped (GT count changed)'
            )

    def after_train(self, runner: Runner) -> None:
        """Remove the wrappers."""
        for undo in self._restore:
            undo()
        self._restore = []

    @torch.no_grad()
    def record(
        self,
        img_path: str,
        pred_instances: InstanceData,
        gt_instances: InstanceData,
        gt_inds: torch.Tensor,
    ) -> None:
        """Add one image's assignment (``gt_inds``: 0 negative, -1 ignored, k > 0 GT k-1)."""
        gtb = _boxes(gt_instances.bboxes)
        if gtb.size(0) == 0:
            return
        entry = self._log.get(img_path)
        if entry is None:
            labels = gt_instances.labels
            # object ids are fixed on the first visit and reused by annotation index
            key = torch.cat([(gtb * 100).round(), labels[:, None].to(gtb)], dim=1)
            _, group = torch.unique(key, dim=0, return_inverse=True)
            n_obj = int(group.max()) + 1
            size = (gtb[:, 2] * gtb[:, 3]).clamp(min=0).sqrt()
            entry = dict(
                group=group.cpu().numpy(),
                size=size.new_zeros(n_obj).scatter_(0, group, size).cpu().numpy(),
                label=labels.new_zeros(n_obj).scatter_(0, group, labels).cpu().numpy(),
                seen=np.zeros(self._epochs, np.int32),
                npos=np.zeros((self._epochs, n_obj), np.int32),
                best_iou=np.zeros((self._epochs, n_obj), np.float32),
            )
            self._log[img_path] = entry
        if entry['group'].size != gtb.size(0):
            self._skipped += 1
            return
        e = self._epoch
        entry['seen'][e] += 1
        pos = (gt_inds > 0).nonzero().squeeze(1)
        if pos.numel() == 0:
            return
        owner = gt_inds[pos] - 1
        iou = box_iou_rotated(_boxes(pred_instances.bboxes)[pos], gtb[owner], aligned=True)
        obj = torch.as_tensor(entry['group'], device=owner.device)[owner]
        n_obj = entry['npos'].shape[1]
        entry['npos'][e] += torch.bincount(obj, minlength=n_obj).cpu().numpy()
        best = iou.new_zeros(n_obj).scatter_reduce(0, obj, iou, reduce='amax', include_self=True)
        np.maximum(entry['best_iou'][e], best.cpu().numpy(), out=entry['best_iou'][e])

    def dump(self, work_dir: str) -> str:
        """Write ``<work_dir>/<file_name>[_rank{r}].npz``; returns its path."""
        suffix = f'_rank{get_rank()}' if get_world_size() > 1 else ''
        path = f'{work_dir}/{self.file_name}{suffix}.npz'
        images = list(self._log)
        entries = [self._log[k] for k in images]
        sizes = [e['size'].size for e in entries]

        def cat(key: str, axis: int = 0) -> np.ndarray:
            if not entries:
                return np.zeros((self._epochs, 0) if axis else 0)
            return np.concatenate([e[key] for e in entries], axis=axis)

        np.savez_compressed(
            path,
            images=np.array(images),
            seen=np.stack([e['seen'] for e in entries], 1)
            if entries
            else np.zeros((self._epochs, 0), np.int32),
            obj_image=np.repeat(np.arange(len(entries)), sizes),
            obj_size=cat('size'),
            obj_label=cat('label'),
            obj_density=np.repeat(sizes, sizes),
            npos=cat('npos', 1),
            best_iou=cat('best_iou', 1).astype(np.float16),
        )
        return path
