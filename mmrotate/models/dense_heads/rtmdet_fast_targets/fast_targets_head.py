# Copyright (c) OpenMMLab. All rights reserved.
"""``RotatedRTMDetSepBNHead`` without the per-GT Python loop in its targets.

``_get_targets_single`` is adapted from
``mmrotate/models/dense_heads/rotated_rtmdet_head.py`` (Apache-2.0, OpenMMLab).
"""

import torch
from mmdet.models.task_modules import anchor_inside_flags
from mmdet.models.utils import unmap
from mmengine.structures import InstanceData
from torch import Tensor

from mmrotate.registry import MODELS

from ..rotated_rtmdet_head import RotatedRTMDetSepBNHead


@MODELS.register_module()
class FastTargetsRotatedRTMDetSepBNHead(RotatedRTMDetSepBNHead):
    """``RotatedRTMDetSepBNHead`` with the same targets, computed without a loop over GTs.

    The parent copies the assigned IoUs into ``assign_metrics`` one GT at a
    time (three small GPU operations per GT). Every positive belongs to exactly
    one GT, so one indexed copy over all positives writes the same values. Same
    parameters and state-dict keys as the parent.
    """

    def _get_targets_single(
        self,
        cls_scores: Tensor,
        bbox_preds: Tensor,
        flat_anchors: Tensor,
        valid_flags: Tensor,
        gt_instances: InstanceData,
        img_meta: dict,
        gt_instances_ignore: InstanceData | None = None,
        unmap_outputs: bool = True,
    ) -> tuple:
        """Targets of one image; see ``RotatedRTMDetHead._get_targets_single``."""
        inside_flags = anchor_inside_flags(
            flat_anchors, valid_flags, img_meta['img_shape'][:2], self.train_cfg['allowed_border']
        )
        if not inside_flags.any():
            return (None,) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        pred_instances = InstanceData(
            scores=cls_scores[inside_flags, :], bboxes=bbox_preds[inside_flags, :], priors=anchors
        )
        assign_result = self.assigner.assign(pred_instances, gt_instances, gt_instances_ignore)
        sampling_result = self.sampler.sample(assign_result, pred_instances, gt_instances)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = anchors.new_zeros((*anchors.size()[:-1], 5))
        labels = anchors.new_full((num_valid_anchors,), self.num_classes, dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        assign_metrics = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            # point-based
            pos_bbox_targets = sampling_result.pos_gt_bboxes
            pos_bbox_targets = pos_bbox_targets.regularize_boxes(self.angle_version)
            bbox_targets[pos_inds, :] = pos_bbox_targets

            labels[pos_inds] = sampling_result.pos_gt_labels
            if self.train_cfg['pos_weight'] <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg['pos_weight']
            # The parent's loop over the assigned GTs, as one indexed copy.
            assign_metrics[pos_inds] = assign_result.max_overlaps[pos_inds]
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors, inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            assign_metrics = unmap(assign_metrics, num_total_anchors, inside_flags)
        return (anchors, labels, label_weights, bbox_targets, assign_metrics, sampling_result)
