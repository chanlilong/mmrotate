# Copyright (c) OpenMMLab. All rights reserved.
"""Train with HF accelerate (DDP / DeepSpeed ZeRO / bf16 / fp16 / fp8).

    # one GPU
    python tools/accelerate/train.py <config>
    # N GPUs on this machine
    accelerate launch --num_processes N tools/accelerate/train.py <config>
    # topology from a launch config (multi-node: see slurm_train.sbatch)
    accelerate launch --config_file tools/accelerate/launch_configs/multi_gpu.yaml \\
        tools/accelerate/train.py <config>

Any mmrotate config works; the recipe (precision, ZeRO stage, gradient
accumulation) comes from its ``accelerator:`` block, or from the optimizer
wrapper (``AmpOptimWrapper`` + ``dtype: bfloat16`` -> bf16) when there is none.
See docs/en/user_guides/accelerate_training.md.
"""

from __future__ import annotations

import argparse
import os.path as osp

from mmdet.utils import register_all_modules as register_all_modules_mmdet
from mmengine.config import Config, DictAction

from mmrotate.accelerate import AccelerateRunner
from mmrotate.utils import register_all_modules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train a detector with HF accelerate')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='resume from the given checkpoint, or the latest one in work_dir if no path is given',
    )
    parser.add_argument(
        '--auto-scale-lr', action='store_true', help='enable automatically scaling LR'
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override settings in the config, key=value (see tools/train.py)',
    )
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Config:
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    if args.auto_scale_lr:
        if 'auto_scale_lr' not in cfg or 'base_batch_size' not in cfg.auto_scale_lr:
            raise RuntimeError('auto_scale_lr.base_batch_size is missing in the config')
        cfg.auto_scale_lr.enable = True
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume
    return cfg


def main() -> None:
    args = parse_args()
    register_all_modules_mmdet(init_default_scope=False)
    register_all_modules(init_default_scope=False)
    cfg = load_config(args)
    runner_type = cfg.get('runner_type', 'AccelerateRunner')
    if str(runner_type).rsplit('.', 1)[-1] != 'AccelerateRunner':
        raise ValueError(
            f'tools/accelerate/train.py runs AccelerateRunner, config asks {runner_type}'
        )
    runner = AccelerateRunner.from_cfg(cfg)
    try:
        runner.train()
    finally:
        runner.accelerator.end_training()


if __name__ == '__main__':
    main()
