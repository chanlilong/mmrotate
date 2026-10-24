# Copyright (c) OpenMMLab. All rights reserved.
"""Estimate the NWD normalisation constant ``C`` for a dataset.

NWD-RKA sets ``C`` to the average absolute object size ``sqrt(w * h)`` of the
dataset (12.7 on AI-TOD). This prints statistics of ``sqrt(area)`` over the
non-ignored GT boxes of a config's training set, at annotation scale::

    python tools/analysis_tools/nwd_constant.py configs/nwd_rka/<config>.py

If the training pipeline rescales images, multiply by that scale factor.
"""

import argparse

import numpy as np
from mmengine.config import Config, ConfigDict, DictAction
from mmengine.utils import ProgressBar

from mmrotate.registry import DATASETS
from mmrotate.utils import register_all_modules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Estimate the NWD constant C')
    parser.add_argument('config', help='config whose train dataset is used')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override settings in the config, key=value format',
    )
    return parser.parse_args()


def box_area(bbox: list[float]) -> float:
    """Area of an ``(x1, y1, x2, y2)`` box or a polygon ``(x1, y1, ..., xn, yn)``."""
    if len(bbox) == 4:
        return max(bbox[2] - bbox[0], 0.0) * max(bbox[3] - bbox[1], 0.0)
    xs, ys = np.asarray(bbox[0::2]), np.asarray(bbox[1::2])
    return 0.5 * abs(float(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1))))


def innermost_dataset_cfg(cfg: ConfigDict) -> ConfigDict:
    """Unwrap RepeatDataset/ClassBalancedDataset-style wrappers."""
    while 'dataset' in cfg and 'ann_file' not in cfg:
        cfg = cfg.dataset
    return cfg


def main() -> None:
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    register_all_modules()

    dataset_cfg = innermost_dataset_cfg(cfg.train_dataloader.dataset)
    dataset_cfg.pipeline = []
    dataset = DATASETS.build(dataset_cfg)

    sizes = []
    progress = ProgressBar(len(dataset))
    for idx in range(len(dataset)):
        for instance in dataset.get_data_info(idx)['instances']:
            if not instance.get('ignore_flag', 0):
                sizes.append(box_area(instance['bbox']) ** 0.5)
        progress.update()
    if not sizes:
        raise RuntimeError('no GT boxes found')

    sizes_np = np.asarray(sizes)
    print(
        f'\n{sizes_np.size} boxes, sqrt(w*h): mean {sizes_np.mean():.2f}, '
        f'median {np.median(sizes_np):.2f}, '
        f'p10 {np.percentile(sizes_np, 10):.2f}, p90 {np.percentile(sizes_np, 90):.2f}'
    )
    print(f'Suggested NWD constant C = {sizes_np.mean():.1f}')


if __name__ == '__main__':
    main()
