# Copyright (c) OpenMMLab. All rights reserved.
"""SODA-A style metric: COCO-like AP over IoU 0.50:0.95 with size bins.

SODA-A (Cheng et al., TPAMI 2023) reports AP, AP50, AP75 and AP over four
object-size bins defined on ``sqrt(area)`` in pixels: extremely small (eS,
<12), relatively small (rS, 12-20), generally small (gS, 20-32) and normal
(N, 32-40). This metric uses mmrotate's VOC-style rotated
matching (area-under-PR AP, see :mod:`.soda_eval`) for every IoU threshold,
with ground truths and detections outside a bin ignored for that bin, so it is
not bit-identical to the official COCO 101-point evaluation but uses the same
protocol and bins.
"""

from collections import OrderedDict
from collections.abc import Sequence

import numpy as np
from mmengine.logging import MMLogger

from mmrotate.registry import METRICS

from ..dota_metric import DOTAMetric
from .soda_eval import eval_rotated_ap

SODA_SCALE_RANGES = ((0, 12), (12, 20), (20, 32), (32, 40))
SODA_SCALE_NAMES = ('eS', 'rS', 'gS', 'N')
_ALL = (0, 1e5)


@METRICS.register_module()
class SODAMetric(DOTAMetric):
    """Rotated AP@[.50:.95] with SODA-A size bins.

    Returns ``mAP``, ``AP50``, ``AP75``, ``AP_eS``, ``AP_rS``, ``AP_gS``,
    ``AP_N`` (each bin averaged over the IoU thresholds) and the per-class
    ``mAP`` as ``AP_<class>``.

    Args:
        iou_thrs: IoU thresholds, COCO-style by default.
        scale_ranges: ``sqrt(area)`` bins, SODA-A's by default.
        scale_names: Name of each bin.
        eval_mode: ``'area'`` (default) or ``'11points'``.
        device: Device for the rotated IoUs (default: CUDA if available).
        **kwargs: Remaining :class:`DOTAMetric` arguments.
    """

    default_prefix: str | None = 'soda'

    def __init__(
        self,
        iou_thrs: Sequence[float] | None = None,
        scale_ranges: Sequence[Sequence[float]] = SODA_SCALE_RANGES,
        scale_names: Sequence[str] = SODA_SCALE_NAMES,
        eval_mode: str = 'area',
        device: str | None = None,
        **kwargs: object,
    ) -> None:
        if iou_thrs is None:
            iou_thrs = np.round(np.linspace(0.5, 0.95, 10), 2).tolist()
        super().__init__(
            iou_thrs=list(iou_thrs),
            scale_ranges=[tuple(r) for r in scale_ranges],
            eval_mode=eval_mode,
            **kwargs,
        )
        if len(scale_names) != len(scale_ranges):
            raise ValueError('scale_names and scale_ranges must have the same length')
        self.scale_names = tuple(scale_names)
        self.device = device

    def compute_metrics(self, results: list) -> dict:
        if self.format_only or self.merge_patches:
            return super().compute_metrics(results)
        logger = MMLogger.get_current_instance()
        gts, preds = zip(*results, strict=True)
        dets = [pred['pred_bbox_scores'] for pred in preds]
        classes = self.dataset_meta['classes']
        # range 0 = all sizes, then the SODA-A bins
        ap_tkc, num_gts_kc = eval_rotated_ap(
            dets,
            gts,
            iou_thrs=self.iou_thrs,
            scale_ranges=[_ALL, *self.scale_ranges],
            mode='11points' if self.use_07_metric else 'area',
            device=self.device,
        )
        # mean over the classes that have GTs in the range
        valid = num_gts_kc > 0
        ap = np.where(valid[None], ap_tkc, 0).sum(-1) / np.maximum(valid.sum(-1), 1)[None]
        cls_ap = ap_tkc[:, 0]  # (T, C), all sizes
        num_gts = num_gts_kc.T  # (C, K)

        eval_results: OrderedDict[str, float] = OrderedDict()
        eval_results['mAP'] = float(ap[:, 0].mean())
        for thr_name, thr in (('AP50', 0.5), ('AP75', 0.75)):
            if thr in self.iou_thrs:
                eval_results[thr_name] = float(ap[self.iou_thrs.index(thr), 0])
        for s, name in enumerate(self.scale_names, start=1):
            eval_results[f'AP_{name}'] = float(ap[:, s].mean())
        for c, name in enumerate(classes):
            eval_results[f'AP_{name}'] = float(cls_ap[:, c].mean())

        lines = [f'{"class":<16}{"gts":>9}{"mAP":>8}{"AP50":>8}']
        for c, name in enumerate(classes):
            lines.append(
                f'{name:<16}{num_gts[c, 0]:>9d}{cls_ap[:, c].mean():>8.3f}{cls_ap[0, c]:>8.3f}'
            )
        size_info = '  '.join(
            f'{n}({lo}-{hi}): gts={int(num_gts[:, s].sum())}'
            for s, (n, (lo, hi)) in enumerate(
                zip(self.scale_names, self.scale_ranges, strict=True), 1
            )
        )
        logger.info('\n' + '\n'.join(lines) + '\n' + size_info)
        return eval_results
