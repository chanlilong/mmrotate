# Copyright (c) OpenMMLab. All rights reserved.
"""Where the time of a TensorRT deploy directory goes.

Runs TensorRT's per-layer profiler on dataset images and prints the fusion
counts (fused attention ``mha`` layers, plugin layers), the time per model
component and the slowest layers::

    python tools/tensorrt/profile_engine.py DEPLOY_DIR [--images 32] [--top 15] [--out report.json]
"""

import argparse
import json
from pathlib import Path

import torch
from mmengine.config import Config

from mmrotate.tensorrt import engine_report, init_trt_detector
from mmrotate.tensorrt.calibration import build_input_loader, iter_batches
from mmrotate.tensorrt.deploy import DEPLOY_CFG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Per-layer profile of a TensorRT engine')
    parser.add_argument('deploy_dir')
    parser.add_argument('--images', type=int, default=32, help='dataset images to profile on')
    parser.add_argument('--split', default='val', choices=('train', 'val', 'test'))
    parser.add_argument('--top', type=int, default=15, help='slowest layers to list')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out', help='optional JSON output (all layer times)')
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    deploy_dir = Path(args.deploy_dir)
    cfg = Config.fromfile(deploy_dir / DEPLOY_CFG)
    model = init_trt_detector(deploy_dir, args.device)
    engine = model.engine
    shape = engine.input_shape()
    loader = build_input_loader(cfg, args.split, shuffle=False, num_workers=args.num_workers)
    batches = [
        inputs.to(args.device)
        for inputs, _ in iter_batches(loader, model.data_preprocessor, shape, args.images)
        if inputs.shape[0] == shape[0]
    ]
    layers = engine_report.layer_info(engine)
    times = engine_report.profile_layers(engine, batches)
    print(engine_report.format_report(times, layers, shape[0], top=args.top))
    if args.out:
        report = dict(
            fusion=engine_report.fusion_summary(layers),
            groups=engine_report.group_times(times, layers),
            layers=times,
        )
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
