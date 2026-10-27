# Copyright (c) OpenMMLab. All rights reserved.
"""Rotated RetinaNet head of DCFL (Xu et al., CVPR 2023).

Ported from ``mmrotate/models/dense_heads/r_dcfl_head.py`` of
https://github.com/Chasel-Tsui/mmrotate-dcfl (Apache-2.0) to mmdet 3.x.
"""

import torch
from mmcv.cnn import ConvModule
from mmdet.models.task_modules.prior_generators import anchor_inside_flags
from mmdet.models.utils import unmap
from mmdet.structures.bbox import BaseBoxes, get_box_tensor
from mmdet.utils import InstanceList, OptConfigType, OptInstanceList
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor, nn

from mmrotate.registry import MODELS

from ..rotated_retina_head import RotatedRetinaHead
from .layers import DCFLModulatedDeformConv, DCFLSharedOffsetDeformConv, dynamic_prior_shift

# img_meta key under which ``get_targets`` hands each image's predictions to
# ``_get_targets_single`` (mmdet's ``get_targets`` only forwards per-image
# anchors, flags, GTs and metas).
_PRED_KEY = '_dcfl_preds'


@MODELS.register_module()
class RotatedDCFLHead(RotatedRetinaHead):
    """Rotated RetinaNet head with DCFL's dynamic priors.

    The last two convs of each tower are a dilated 3x3 conv and a DCNv2. The
    regression DCN predicts the offsets and masks, and the classification DCN
    reuses them. Before label assignment every anchor is moved by the mean
    offset of its location (the dynamic prior), and the assigner, normally
    :class:`C2FAssigner`, gets the shifted anchors together with the current
    scores and decoded boxes.

    The reference applies the fixed offsets ``(d - 1) * (i, j)`` to an mmcv
    ``DeformConv2d``. That is exactly a dilated conv, which is used here with
    the same weight shape and no bias.

    Differences from the reference implementation:

    - The shifted anchors are used for the assignment only. The reference
      also encoded regression targets against them (it uses
      ``reg_decoded_bbox=True``, where this does not matter) and decoded the
      predictions the assigner sees with them. Here the predictions come from
      the original anchors, as at test time.
    - The shift is ``offset * stride * prior_offset_scale``, which matches the
      reference's resolution-dependent ``img_h / 256 * 2 ** level`` at 1024
      input with the default scale of 0.5.

    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of input channels.
        stacked_convs (int): Tower depth including the dilated conv and the
            DCN. Must be at least 2. Defaults to 4.
        dilation_rate (int): Dilation of the dilated conv. Defaults to 3.
        prior_offset_scale (float): Scale of the prior shift in strides per
            feature-pixel offset. Defaults to 0.5.
        dcn_backend (str): ``'auto'``, ``'mmcv'`` or ``'torchvision'``.
            Defaults to ``'auto'``.
        conv_cfg (dict, optional): Conv config of the plain convs.
        norm_cfg (dict, optional): Norm config of the plain convs.
        **kwargs: ``RotatedRetinaHead`` arguments.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        stacked_convs: int = 4,
        dilation_rate: int = 3,
        prior_offset_scale: float = 0.5,
        dcn_backend: str = 'auto',
        conv_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = None,
        **kwargs: object,
    ) -> None:
        assert stacked_convs >= 2, 'DCFL replaces the last two tower convs'
        self.dilation_rate = dilation_rate
        self.prior_offset_scale = prior_offset_scale
        self.dcn_backend = dcn_backend
        self._dcfl_preds: tuple | None = None
        super().__init__(
            num_classes,
            in_channels,
            stacked_convs=stacked_convs,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            **kwargs,
        )

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        in_channels = self.in_channels
        for _ in range(self.stacked_convs - 2):
            for convs in (self.cls_convs, self.reg_convs):
                convs.append(
                    ConvModule(
                        in_channels,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                    )
                )
            in_channels = self.feat_channels
        d = self.dilation_rate
        for convs in (self.cls_convs, self.reg_convs):
            convs.append(
                nn.Conv2d(in_channels, self.feat_channels, 3, padding=d, dilation=d, bias=False)
            )
        dcn = dict(kernel_size=3, padding=1, bias=False, backend=self.dcn_backend)
        self.cls_convs.append(
            DCFLSharedOffsetDeformConv(self.feat_channels, self.feat_channels, **dcn)
        )
        self.reg_convs.append(
            DCFLModulatedDeformConv(self.feat_channels, self.feat_channels, **dcn)
        )
        self.retina_cls = nn.Conv2d(
            self.feat_channels, self.num_base_priors * self.cls_out_channels, 3, padding=1
        )
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_base_priors * self.bbox_coder.encode_size, 3, padding=1
        )

    def init_weights(self) -> None:
        """Initialize weights; the DCN offset branch starts at zero."""
        super().init_weights()
        self.reg_convs[-1].init_offset()

    def forward_single(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Forward a single scale level.

        Returns:
            tuple[Tensor, Tensor, Tensor]: Class scores, box deltas and the
            regression DCN offsets ``(B, 18, H, W)``.
        """
        reg_feat = x
        for reg_conv in self.reg_convs[:-1]:
            reg_feat = reg_conv(reg_feat)
        reg_feat, offset, mask = self.reg_convs[-1](reg_feat)

        cls_feat = x
        for cls_conv in self.cls_convs[:-1]:
            cls_feat = cls_conv(cls_feat)
        cls_feat = self.cls_convs[-1](cls_feat, offset, mask)
        return self.retina_cls(cls_feat), self.retina_reg(reg_feat), offset

    def loss_by_feat(
        self,
        cls_scores: list[Tensor],
        bbox_preds: list[Tensor],
        offsets: list[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: list[dict],
        batch_gt_instances_ignore: OptInstanceList = None,
    ) -> dict:
        """Loss of RetinaNet with the DCFL assignment (see ``AnchorHead``)."""
        self._dcfl_preds = (cls_scores, bbox_preds, offsets)
        try:
            return super().loss_by_feat(
                cls_scores,
                bbox_preds,
                batch_gt_instances,
                batch_img_metas,
                batch_gt_instances_ignore=batch_gt_instances_ignore,
            )
        finally:
            self._dcfl_preds = None

    def get_targets(
        self,
        anchor_list: list[list[Tensor]],
        valid_flag_list: list[list[Tensor]],
        batch_gt_instances: InstanceList,
        batch_img_metas: list[dict],
        batch_gt_instances_ignore: OptInstanceList = None,
        unmap_outputs: bool = True,
        return_sampling_results: bool = False,
    ) -> tuple:
        """``AnchorHead.get_targets`` that hands each image its flattened
        scores, box deltas and prior shifts (from ``loss_by_feat``)."""
        assert self._dcfl_preds is not None, 'get_targets is called from loss_by_feat'
        cls_scores, bbox_preds, offsets = self._dcfl_preds
        num_imgs = len(batch_img_metas)
        scores = torch.cat(
            [
                s.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels)
                for s in cls_scores
            ],
            dim=1,
        )
        deltas = torch.cat(
            [
                b.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.bbox_coder.encode_size)
                for b in bbox_preds
            ],
            dim=1,
        )
        shifts = torch.cat(
            [
                dynamic_prior_shift(o, stride, self.prior_offset_scale).repeat_interleave(
                    self.num_base_priors, dim=1
                )
                for o, stride in zip(offsets, self.prior_generator.strides, strict=True)
            ],
            dim=1,
        )
        img_metas = [
            {**meta, _PRED_KEY: (scores[i].detach(), deltas[i].detach(), shifts[i])}
            for i, meta in enumerate(batch_img_metas)
        ]
        return super().get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore,
            unmap_outputs=unmap_outputs,
            return_sampling_results=return_sampling_results,
        )

    def _get_targets_single(
        self,
        flat_anchors: Tensor | BaseBoxes,
        valid_flags: Tensor,
        gt_instances: InstanceData,
        img_meta: dict,
        gt_instances_ignore: InstanceData | None = None,
        unmap_outputs: bool = True,
    ) -> tuple:
        """``AnchorHead._get_targets_single`` whose assigner sees the shifted
        anchors, the scores and the decoded boxes."""
        scores, deltas, shift = img_meta[_PRED_KEY]
        inside_flags = anchor_inside_flags(
            flat_anchors, valid_flags, img_meta['img_shape'][:2], self.train_cfg['allowed_border']
        )
        if not inside_flags.any():
            raise ValueError(
                'There is no valid anchor inside the image boundary. Please '
                'check the image size and anchor sizes, or set '
                '``allowed_border`` to -1 to skip the condition.'
            )
        anchors = flat_anchors[inside_flags]
        dynamic = get_box_tensor(anchors).clone()
        dynamic[:, :2] += shift[inside_flags].to(dynamic.dtype)
        if isinstance(anchors, BaseBoxes):
            dynamic = type(anchors)(dynamic)
        pred_bboxes = get_box_tensor(self.bbox_coder.decode(anchors, deltas[inside_flags]))

        pred_instances = InstanceData(
            priors=dynamic, bboxes=pred_bboxes, scores=scores[inside_flags]
        )
        assign_result = self.assigner.assign(pred_instances, gt_instances, gt_instances_ignore)
        sampling_result = self.sampler.sample(
            assign_result, InstanceData(priors=anchors), gt_instances
        )

        num_valid_anchors = anchors.shape[0]
        target_dim = (
            gt_instances.bboxes.size(-1) if self.reg_decoded_bbox else self.bbox_coder.encode_size
        )
        bbox_targets = anchors.new_zeros(num_valid_anchors, target_dim)
        bbox_weights = anchors.new_zeros(num_valid_anchors, target_dim)
        labels = anchors.new_full((num_valid_anchors,), self.num_classes, dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_priors, sampling_result.pos_gt_bboxes
                )
            else:
                pos_bbox_targets = get_box_tensor(sampling_result.pos_gt_bboxes)
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            labels[pos_inds] = sampling_result.pos_gt_labels
            if self.train_cfg['pos_weight'] <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg['pos_weight']
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors, inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            pos_inds,
            neg_inds,
            sampling_result,
        )

    def predict_by_feat(
        self,
        cls_scores: list[Tensor],
        bbox_preds: list[Tensor],
        offsets: list[Tensor] | None = None,
        score_factors: list[Tensor] | None = None,
        batch_img_metas: list[dict] | None = None,
        cfg: ConfigDict | None = None,
        rescale: bool = False,
        with_nms: bool = True,
    ) -> InstanceList:
        """``AnchorHead.predict_by_feat``; the DCN offsets are not needed."""
        return super().predict_by_feat(
            cls_scores,
            bbox_preds,
            score_factors=score_factors,
            batch_img_metas=batch_img_metas,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
        )
