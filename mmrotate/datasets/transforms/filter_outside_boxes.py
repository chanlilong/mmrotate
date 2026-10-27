# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.transforms import BaseTransform

from mmrotate.registry import TRANSFORMS
from mmrotate.structures.bbox import RotatedBoxes, rbbox_overlaps


@TRANSFORMS.register_module()
class FilterOutsideBoxes(BaseTransform):
    """Drop GT boxes that lie mostly outside the image.

    Geometric augmentations such as ``RandomRotate`` followed by
    ``mmdet.RandomShift`` can push rotated boxes partly or fully off the
    image, and ``RotatedBoxes.clip_`` is a no-op. This transform computes the
    IoF (intersection over the box's own area) between every GT box and an
    image-sized rotated box, and keeps only boxes with ``iof >= iof_thr``.

    Required Keys:

    - img_shape
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)

    Modified Keys:

    - gt_bboxes
    - gt_bboxes_labels
    - gt_ignore_flags
    - gt_masks

    Args:
        iof_thr (float): Minimum fraction of a box's area that must lie inside
            the image for the box to be kept. Defaults to 0.5.
        keep_empty (bool): Whether to return the results when every box is
            dropped. If False, ``None`` is returned so that the dataset
            resamples another image. Defaults to True.
    """

    def __init__(self, iof_thr: float = 0.5, keep_empty: bool = True) -> None:
        assert 0.0 <= iof_thr <= 1.0, f'iof_thr must be in [0, 1], got {iof_thr}'
        self.iof_thr = iof_thr
        self.keep_empty = keep_empty

    def transform(self, results: dict) -> dict | None:
        """Filter GT boxes (and aligned annotations) by IoF with the image."""
        boxes = results.get('gt_bboxes', None)
        if boxes is None or len(boxes) == 0:
            return results

        # results['img_shape'] may be (h, w, c) or (h, w)
        height, width = results['img_shape'][:2]
        img_box = RotatedBoxes(
            boxes.tensor.new_tensor([[width / 2, height / 2, width, height, 0.0]])
        )
        # ``RotatedBoxes.overlaps`` forwards an ``eps`` kwarg that
        # ``rbbox_overlaps`` does not accept, so call it directly.
        iof = rbbox_overlaps(boxes.convert_to('rbox').tensor, img_box.tensor, mode='iof')[:, 0]
        keep = (iof >= self.iof_thr).cpu().numpy()

        results['gt_bboxes'] = boxes[keep]
        if results.get('gt_ignore_flags', None) is not None:
            results['gt_ignore_flags'] = results['gt_ignore_flags'][keep]
        if results.get('gt_bboxes_labels', None) is not None:
            results['gt_bboxes_labels'] = results['gt_bboxes_labels'][keep]
        if results.get('gt_masks', None) is not None:
            results['gt_masks'] = results['gt_masks'][keep.nonzero()[0]]

        if not self.keep_empty and not keep.any():
            return None
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(iof_thr={self.iof_thr}, '
        repr_str += f'keep_empty={self.keep_empty})'
        return repr_str
