# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.structures.bbox import BaseBoxes
from torch import Tensor

from mmrotate.registry import TASK_UTILS
from mmrotate.structures.bbox import rbbox_overlaps

from ..nwd_rka.nwd_calculator import to_box_tensor, to_rbox_tensor
from .gjsd import rbbox_gjsd


@TASK_UTILS.register_module()
class RBboxGJSD2D:
    """GJSD similarity ``1 / (1 + GJSD)`` calculator for (rotated) boxes.

    The prior-ranking metric of DCFL's :class:`C2FAssigner`, usable as a
    drop-in replacement for ``RBboxOverlaps2D`` in any assigner. Horizontal
    and rotated boxes may be mixed.
    """

    def __call__(
        self,
        bboxes1: BaseBoxes | Tensor,
        bboxes2: BaseBoxes | Tensor,
        mode: str = 'gjsd',
        is_aligned: bool = False,
    ) -> Tensor:
        """Compute the GJSD similarity between two sets of boxes.

        Args:
            bboxes1 (BaseBoxes | Tensor): ``(m, 4|5|6)`` boxes.
            bboxes2 (BaseBoxes | Tensor): ``(n, 4|5|6)`` boxes.
            mode (str): ``'iof'`` returns rotated IoF (used by assigners for
                ignore regions); any other value (``'gjsd'``, ``'iou'``)
                returns the GJSD similarity.
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
        return rbbox_gjsd(bboxes1, bboxes2, is_aligned)

    def __repr__(self) -> str:
        """str: a string describing the module."""
        return f'{self.__class__.__name__}()'
