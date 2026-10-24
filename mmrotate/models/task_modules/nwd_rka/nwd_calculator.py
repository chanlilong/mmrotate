# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.structures.bbox import BaseBoxes, get_box_tensor
from torch import Tensor

from mmrotate.registry import TASK_UTILS
from mmrotate.structures.bbox import rbbox_overlaps

from .nwd import rbbox_nwd


def to_box_tensor(boxes: BaseBoxes | Tensor) -> Tensor:
    """Box tensor without a trailing score column (4 or 5 values per box)."""
    boxes = get_box_tensor(boxes)
    if boxes.size(-1) == 6:
        boxes = boxes[..., :5]
    return boxes


def to_rbox_tensor(boxes: Tensor) -> Tensor:
    """Convert ``(x1, y1, x2, y2)`` boxes to ``(cx, cy, w, h, 0)``."""
    if boxes.size(-1) == 5:
        return boxes
    xy = (boxes[..., :2] + boxes[..., 2:]) / 2
    wh = boxes[..., 2:] - boxes[..., :2]
    return torch.cat([xy, wh, wh.new_zeros((*wh.shape[:-1], 1))], dim=-1)


@TASK_UTILS.register_module()
class RBboxNWD2D:
    """Normalized Wasserstein Distance calculator for (rotated) boxes.

    A drop-in replacement for ``RBboxOverlaps2D`` (or ``BboxOverlaps2D``) in
    any assigner, e.g. ``MaxIoUAssigner``, ``DynamicSoftLabelAssigner``,
    ``RotatedATSSAssigner`` or :class:`RankingAssigner`. Horizontal and rotated
    boxes may be mixed (e.g. horizontal RPN anchors vs rotated GTs).

    Args:
        constant (float): Normalisation constant ``C`` of
            ``NWD = exp(-W2 / C)``, roughly the dataset's average absolute
            object size (see ``tools/analysis_tools/nwd_constant.py``).
            Defaults to 12.8.
    """

    def __init__(self, constant: float = 12.8) -> None:
        self.constant = constant

    def __call__(
        self,
        bboxes1: BaseBoxes | Tensor,
        bboxes2: BaseBoxes | Tensor,
        mode: str = 'nwd',
        is_aligned: bool = False,
    ) -> Tensor:
        """Compute NWD between two sets of boxes.

        Args:
            bboxes1 (BaseBoxes | Tensor): ``(m, 4|5|6)`` boxes.
            bboxes2 (BaseBoxes | Tensor): ``(n, 4|5|6)`` boxes.
            mode (str): ``'iof'`` returns rotated IoF (used by assigners for
                ignore regions); any other value (``'nwd'``, ``'iou'``) returns
                NWD so that assigners calling with their default ``mode='iou'``
                use NWD.
            is_aligned (bool): Whether to compute element-wise.

        Returns:
            Tensor: ``(m, n)`` or ``(m,)`` if aligned.
        """
        bboxes1 = to_box_tensor(bboxes1)
        bboxes2 = to_box_tensor(bboxes2)
        if mode == 'iof':
            return rbbox_overlaps(
                to_rbox_tensor(bboxes1), to_rbox_tensor(bboxes2), 'iof', is_aligned
            )
        return rbbox_nwd(bboxes1, bboxes2, self.constant, is_aligned)

    def __repr__(self) -> str:
        """str: a string describing the module."""
        return f'{self.__class__.__name__}(constant={self.constant})'
