# Copyright (c) OpenMMLab. All rights reserved.
"""Test a checkpoint on one or more GPUs with HF accelerate.

    python tools/accelerate/test.py <config> <checkpoint>
    accelerate launch --num_processes N tools/accelerate/test.py <config> <checkpoint>

The test set is sharded by mmengine's sampler; the per-rank results are
gathered through the parquet bridge (``accelerator.eval``), so the metrics are
identical to a single-process ``tools/test.py`` run. The checkpoint is a normal
mmengine ``.pth`` (``AccelerateRunner`` writes gathered weights also for
DeepSpeed ZeRO-3).
"""

from __future__ import annotations

import argparse
import os.path as osp

from mmdet.utils import register_all_modules as register_all_modules_mmdet
from mmengine.config import Config, DictAction

from mmrotate.accelerate import AccelerateRunner
from mmrotate.utils import register_all_modules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Test a detector with HF accelerate')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--work-dir', help='the dir to save logs and the bridge files')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override settings in the config, key=value (see tools/test.py)',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    register_all_modules_mmdet(init_default_scope=False)
    register_all_modules(init_default_scope=False)
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    cfg.load_from = args.checkpoint
    cfg.resume = False
    runner = AccelerateRunner.from_cfg(cfg)
    try:
        runner.test()
    finally:
        runner.accelerator.end_training()


if __name__ == '__main__':
    main()
