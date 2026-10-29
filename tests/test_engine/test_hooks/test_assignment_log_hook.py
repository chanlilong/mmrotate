# Copyright (c) OpenMMLab. All rights reserved.
"""Tests for AssignmentLogHook (per-object, per-epoch assignment log)."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from mmdet.models.task_modules import AssignResult
from mmengine.structures import InstanceData

from mmrotate.engine.hooks import AssignmentLogHook


class ScriptedAssigner:
    """Returns the ``gt_inds`` it is told to (0 negative, k > 0 GT k-1)."""

    def __init__(self) -> None:
        self.gt_inds = torch.zeros(0, dtype=torch.long)

    def assign(
        self,
        pred_instances: InstanceData,
        gt_instances: InstanceData,
        gt_instances_ignore: InstanceData | None = None,
    ) -> AssignResult:
        return AssignResult(len(gt_instances), self.gt_inds, None, labels=None)


class FakeHead:
    """The part of ``RotatedRTMDetHead`` the hook wraps."""

    def __init__(self) -> None:
        self.assigner = ScriptedAssigner()

    def _get_targets_single(
        self,
        cls_scores: torch.Tensor,
        bbox_preds: torch.Tensor,
        priors: torch.Tensor,
        valid_flags: torch.Tensor,
        gt_instances: InstanceData,
        img_meta: dict,
        gt_instances_ignore: InstanceData | None = None,
    ) -> tuple:
        pred = InstanceData(scores=cls_scores, bboxes=bbox_preds, priors=priors)
        return (self.assigner.assign(pred, gt_instances, gt_instances_ignore),)


def _step(head: FakeHead, gts: list[list[float]], gt_inds: list[int], path: str) -> None:
    gtb = torch.tensor(gts)
    owner = torch.tensor(gt_inds)
    # prior k predicts its GT shifted by 1 px (or a far-away box for negatives)
    preds = torch.tensor([[400.0, 400.0, 5.0, 5.0, 0.0]]).repeat(len(gt_inds), 1)
    preds[owner > 0] = gtb[owner[owner > 0] - 1] + torch.tensor([1.0, 0, 0, 0, 0])
    head.assigner.gt_inds = owner
    gt = InstanceData(bboxes=gtb, labels=torch.zeros(len(gts), dtype=torch.long))
    n = len(gt_inds)
    head._get_targets_single(
        torch.zeros(n, 2), preds, torch.zeros(n, 4), torch.ones(n), gt, {'img_path': path}
    )


def test_assignment_log_hook(tmp_path: Path) -> None:
    head = FakeHead()
    runner = SimpleNamespace(
        model=SimpleNamespace(bbox_head=head),
        max_epochs=2,
        epoch=0,
        work_dir=str(tmp_path),
        logger=MagicMock(),
    )
    hook = AssignmentLogHook()
    hook.before_train(runner)
    # GTs 1 and 2 are exact duplicates: one object
    gts = [[20.0, 20.0, 10.0, 10.0, 0.0], [44.0, 44.0, 6.0, 6.0, 0.0], [44.0, 44.0, 6.0, 6.0, 0.0]]
    hook.before_train_epoch(runner)
    _step(head, gts, [1, 1, 3, 0], 'a.png')  # both objects matched (one via the copy)
    _step(head, [[60.0, 60.0, 4.0, 4.0, 0.0]], [0, 0], 'b.png')  # starved
    hook.after_train_epoch(runner)
    runner.epoch = 1
    hook.before_train_epoch(runner)
    _step(head, gts, [0, 1, 0, 0], 'a.png')  # the small object is starved this time
    _step(head, gts[:2], [1], 'a.png')  # GT count changed: skipped
    hook.after_train_epoch(runner)
    hook.after_train(runner)
    # the wrappers are gone
    assert '_get_targets_single' not in vars(head) and 'assign' not in vars(head.assigner)

    log = dict(np.load(tmp_path / 'assignment_log.npz'))
    assert log['images'].tolist() == ['a.png', 'b.png']
    assert log['seen'].tolist() == [[1, 1], [1, 0]]  # (epoch, image)
    assert log['obj_image'].tolist() == [0, 0, 1]
    order = np.argsort(log['obj_size'][:2])  # a.png: 6 px object, 10 px object
    assert log['obj_size'][:2][order] == pytest.approx([6.0, 10.0])
    assert log['obj_density'].tolist() == [2, 2, 1]
    npos = log['npos'][:, :2][:, order]
    assert npos.tolist() == [[1, 2], [0, 1]]
    assert log['npos'][:, 2].tolist() == [0, 0]
    best = log['best_iou'][:, :2][:, order].astype(np.float32)
    assert best[0, 0] > 0.5 and best[1, 0] == 0 and (best[:, 1] > 0.8).all()
