# Copyright (c) OpenMMLab. All rights reserved.
"""Rotated RTMDet (SepBN) head with DCFL's dynamic priors (Xu et al., CVPR 2023)."""

from mmcv.cnn import ConvModule, build_activation_layer, build_norm_layer
from mmdet.models import inverse_sigmoid
from mmdet.models.utils import sigmoid_geometric_mean
from mmdet.utils import InstanceList, OptInstanceList
from mmengine.config import ConfigDict
from torch import Tensor, nn

from mmrotate.registry import MODELS

from ..rotated_rtmdet_head import RotatedRTMDetSepBNHead
from .layers import DCFLModulatedDeformConv, DCFLSharedOffsetDeformConv, dynamic_prior_shift


@MODELS.register_module()
class RotatedRTMDetSepBNDCFLHead(RotatedRTMDetSepBNHead):
    """``RotatedRTMDetSepBNHead`` with DCFL's dynamic priors.

    DCFL (Xu et al., CVPR 2023) for the RTMDet head. After the usual
    ``stacked_convs`` of each tower come a dilated 3x3 ConvModule and a DCNv2,
    each followed by the level's own norm and activation, as in the SepBN
    head. The regression DCN predicts offsets and masks and the classification
    DCN reuses them. Conv and DCN weights are shared across levels when
    ``share_conv`` is set.

    For label assignment every point prior is moved by the mean DCN offset of
    its location, so the assigner (normally :class:`C2FAssigner` with
    ``point_prior_scale``) ranks the dynamic priors. Box decoding still uses
    the original points.

    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of input channels.
        dilation_rate (int): Dilation of the dilated conv. Defaults to 3.
        prior_offset_scale (float): Scale of the prior shift in strides per
            feature-pixel offset. 0.5 matches the reference at 1024 input.
            Defaults to 0.5.
        dcn_backend (str): ``'auto'``, ``'mmcv'`` or ``'torchvision'``.
            Defaults to ``'auto'``.
        **kwargs: ``RotatedRTMDetSepBNHead`` arguments.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        dilation_rate: int = 3,
        prior_offset_scale: float = 0.5,
        dcn_backend: str = 'auto',
        **kwargs: object,
    ) -> None:
        self.dilation_rate = dilation_rate
        self.prior_offset_scale = prior_offset_scale
        self.dcn_backend = dcn_backend
        self._dcfl_offsets: list[Tensor] | None = None
        super().__init__(num_classes, in_channels, **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        super()._init_layers()
        num_levels = len(self.prior_generator.strides)
        chn = self.feat_channels if self.stacked_convs > 0 else self.in_channels
        d = self.dilation_rate
        dcn = dict(kernel_size=3, padding=1, bias=False, backend=self.dcn_backend)
        self.cls_dilated_convs = nn.ModuleList()
        self.reg_dilated_convs = nn.ModuleList()
        self.cls_dcns = nn.ModuleList()
        self.reg_dcns = nn.ModuleList()
        self.cls_dcn_norms = nn.ModuleList()
        self.reg_dcn_norms = nn.ModuleList()
        for n in range(num_levels):
            for convs in (self.cls_dilated_convs, self.reg_dilated_convs):
                convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        padding=d,
                        dilation=d,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                    )
                )
            if self.share_conv and n > 0:
                self.cls_dilated_convs[n].conv = self.cls_dilated_convs[0].conv
                self.reg_dilated_convs[n].conv = self.reg_dilated_convs[0].conv
                self.cls_dcns.append(self.cls_dcns[0])
                self.reg_dcns.append(self.reg_dcns[0])
            else:
                self.cls_dcns.append(
                    DCFLSharedOffsetDeformConv(self.feat_channels, self.feat_channels, **dcn)
                )
                self.reg_dcns.append(
                    DCFLModulatedDeformConv(self.feat_channels, self.feat_channels, **dcn)
                )
            self.cls_dcn_norms.append(build_norm_layer(self.norm_cfg, self.feat_channels)[1])
            self.reg_dcn_norms.append(build_norm_layer(self.norm_cfg, self.feat_channels)[1])
        self.dcn_act = build_activation_layer(self.act_cfg)

    def init_weights(self) -> None:
        """Initialize weights; the DCN offset branches start at zero."""
        super().init_weights()
        for reg_dcn in self.reg_dcns:
            reg_dcn.init_offset()

    def forward(self, feats: tuple[Tensor, ...]) -> tuple:
        """Forward features from the upstream network.

        Returns:
            tuple: ``(cls_scores, bbox_preds, angle_preds, offsets)``, the
            first three as in ``RotatedRTMDetSepBNHead`` and ``offsets`` the
            regression DCN offsets ``(B, 18, H, W)`` of every level.
        """
        cls_scores = []
        bbox_preds = []
        angle_preds = []
        offsets = []
        for idx, (x, stride) in enumerate(zip(feats, self.prior_generator.strides, strict=True)):
            reg_feat = x
            for reg_layer in self.reg_convs[idx]:
                reg_feat = reg_layer(reg_feat)
            reg_feat = self.reg_dilated_convs[idx](reg_feat)
            reg_feat, offset, mask = self.reg_dcns[idx](reg_feat)
            reg_feat = self.dcn_act(self.reg_dcn_norms[idx](reg_feat))

            cls_feat = x
            for cls_layer in self.cls_convs[idx]:
                cls_feat = cls_layer(cls_feat)
            cls_feat = self.cls_dilated_convs[idx](cls_feat)
            cls_feat = self.cls_dcns[idx](cls_feat, offset, mask)
            cls_feat = self.dcn_act(self.cls_dcn_norms[idx](cls_feat))
            cls_score = self.rtm_cls[idx](cls_feat)

            if self.with_objectness:
                objectness = self.rtm_obj[idx](reg_feat)
                cls_score = inverse_sigmoid(sigmoid_geometric_mean(cls_score, objectness))
            if self.exp_on_reg:
                reg_dist = self.rtm_reg[idx](reg_feat).exp() * stride[0]
            else:
                reg_dist = self.rtm_reg[idx](reg_feat) * stride[0]

            cls_scores.append(cls_score)
            bbox_preds.append(reg_dist)
            angle_preds.append(self.rtm_ang[idx](reg_feat))
            offsets.append(offset)
        return tuple(cls_scores), tuple(bbox_preds), tuple(angle_preds), tuple(offsets)

    def loss_by_feat(
        self,
        cls_scores: list[Tensor],
        bbox_preds: list[Tensor],
        angle_preds: list[Tensor],
        offsets: list[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: list[dict],
        batch_gt_instances_ignore: OptInstanceList = None,
    ) -> dict:
        """``RotatedRTMDetHead.loss_by_feat`` with dynamic priors."""
        self._dcfl_offsets = list(offsets)
        try:
            return super().loss_by_feat(
                cls_scores,
                bbox_preds,
                angle_preds,
                batch_gt_instances,
                batch_img_metas,
                batch_gt_instances_ignore=batch_gt_instances_ignore,
            )
        finally:
            self._dcfl_offsets = None

    def get_targets(
        self,
        cls_scores: Tensor,
        bbox_preds: Tensor,
        anchor_list: list[list[Tensor]],
        valid_flag_list: list[list[Tensor]],
        batch_gt_instances: InstanceList,
        batch_img_metas: list[dict],
        batch_gt_instances_ignore: OptInstanceList = None,
        unmap_outputs: bool = True,
    ) -> tuple:
        """``RTMDetHead.get_targets`` on the priors shifted by the mean DCN
        offset (the decoded ``bbox_preds`` keep the original priors)."""
        assert self._dcfl_offsets is not None, 'get_targets is called from loss_by_feat'
        shifts = [
            dynamic_prior_shift(offset, stride, self.prior_offset_scale)
            for offset, stride in zip(self._dcfl_offsets, self.prior_generator.strides, strict=True)
        ]
        dynamic_anchor_list = []
        for img_id, anchors in enumerate(anchor_list):
            dynamic = []
            for lvl, priors in enumerate(anchors):
                priors = priors.clone()
                priors[:, :2] += shifts[lvl][img_id].to(priors.dtype)
                dynamic.append(priors)
            dynamic_anchor_list.append(dynamic)
        return super().get_targets(
            cls_scores,
            bbox_preds,
            dynamic_anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore,
            unmap_outputs=unmap_outputs,
        )

    def predict_by_feat(
        self,
        cls_scores: list[Tensor],
        bbox_preds: list[Tensor],
        angle_preds: list[Tensor],
        offsets: list[Tensor] | None = None,
        score_factors: list[Tensor] | None = None,
        batch_img_metas: list[dict] | None = None,
        cfg: ConfigDict | None = None,
        rescale: bool = False,
        with_nms: bool = True,
    ) -> InstanceList:
        """``RotatedRTMDetHead.predict_by_feat``; the offsets are not needed."""
        return super().predict_by_feat(
            cls_scores,
            bbox_preds,
            angle_preds,
            score_factors=score_factors,
            batch_img_metas=batch_img_metas,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
        )
