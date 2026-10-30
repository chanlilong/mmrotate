# Copyright (c) OpenMMLab. All rights reserved.
"""Evaluate a TensorRT deploy directory with the config's test loop.

Same as ``tools/test.py`` but the model is the ``TRTSingleStageDetector`` of
``<deploy_dir>/deploy_cfg.py`` (no checkpoint: the weights are in the engine)::

    python tools/tensorrt/test_trt.py DEPLOY_DIR [--out preds.pkl] \\
        [--cfg-options test_dataloader.dataset.data_root=...]
"""

import argparse
import os.path as osp
from pathlib import Path

from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmdet.utils import register_all_modules as register_all_modules_mmdet
from mmengine.config import Config, DictAction
from mmengine.evaluator import DumpResults
from mmengine.runner import Runner

from mmrotate.tensorrt.deploy import DEPLOY_CFG
from mmrotate.utils import register_all_modules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Test a TensorRT deploy directory')
    parser.add_argument('deploy_dir', help='output of tools/tensorrt/convert.py')
    parser.add_argument('--work-dir', help='directory for logs and metrics')
    parser.add_argument('--out', help='dump predictions to a pickle file')
    parser.add_argument('--show-dir', help='save painted images to this directory')
    parser.add_argument('--show', action='store_true', help='show prediction results')
    parser.add_argument('--wait-time', type=float, default=2)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    register_all_modules_mmdet(init_default_scope=False)
    register_all_modules(init_default_scope=False)
    cfg = Config.fromfile(Path(args.deploy_dir) / DEPLOY_CFG)
    cfg.launcher = 'none'
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    cfg.work_dir = args.work_dir or osp.join(
        './work_dirs', 'tensorrt', Path(args.deploy_dir).resolve().name
    )
    cfg.load_from = None
    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)
    runner = Runner.from_cfg(cfg)
    if args.out is not None:
        if not args.out.endswith(('.pkl', '.pickle')):
            raise ValueError('The dump file must be a pkl file')
        runner.test_evaluator.metrics.append(DumpResults(out_file_path=args.out))
    runner.test()


if __name__ == '__main__':
    main()
