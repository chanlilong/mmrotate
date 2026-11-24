# Copyright (c) OpenMMLab. All rights reserved.
"""Train-step time and peak GPU memory of a detector config on synthetic batches.

Builds the detector of every config with random weights (speed and memory do
not depend on them) and runs the real training step, ``model.train_step`` with
the config's optimiser wrapper (bf16 autocast, AdamW) followed by the config's
EMA update, on batches of ``--size``² images with ``--num-gts`` random rotated
boxes each. The learning rate is set to 0 so that every repetition sees the
same model.

For every (config, batch size, GTs per image) it reports:

* ``s/step`` and ``img/s``: median over ``--iters`` steps after ``--warmup``;
* ``peak GB``: ``max_memory_allocated`` over the timed steps (``reserved`` is
  what nvidia-smi would show), and ``static GB``: allocated between steps
  (weights, optimiser state, EMA);
* a split of the step into backbone / neck / head forward, assignment + loss
  and backward + optimiser + EMA, from a second pass that synchronises the GPU
  at every boundary (so it is slightly slower than the timed pass).

An out-of-memory case is reported as ``OOM`` and the run continues::

    python tools/cradio/bench_train_step.py CONFIG [CONFIG ...] \\
        --batch 2 4 --num-gts 50 200 1000 --out bench.json

``--conv-bench`` instead times single convolutions of the neck / adapter
(forward + backward) in NCHW and channels_last, for fp32, fp16 and bf16.
"""

import argparse
import json
import os.path as osp
import time
from collections import defaultdict
from collections.abc import Callable

import torch
from mmdet.structures import DetDataSample
from mmengine.config import Config, DictAction
from mmengine.optim import build_optim_wrapper
from mmengine.structures import InstanceData
from torch import Tensor, nn
from torch.profiler import ProfilerActivity, profile

from mmrotate.registry import MODELS
from mmrotate.structures.bbox import RotatedBoxes
from mmrotate.utils import register_all_modules

PHASES = ('backbone', 'neck', 'head', 'assign_loss', 'backward_optim_ema')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train-step time and memory on synthetic data')
    parser.add_argument('configs', nargs='*')
    parser.add_argument('--batch', type=int, nargs='+', default=[2])
    parser.add_argument('--num-gts', type=int, nargs='+', default=[200])
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument(
        '--gt-size', type=float, nargs=2, default=(8.0, 64.0), help='min / max box side in px'
    )
    parser.add_argument('--iters', type=int, default=6, help='timed steps per case')
    parser.add_argument('--warmup', type=int, default=2, help='untimed steps per case')
    parser.add_argument('--no-ema', action='store_true', help='skip the EMA update')
    parser.add_argument('--out', help='optional JSON output (a .md table is written next to it)')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    parser.add_argument('--conv-bench', action='store_true', help='time single convolutions')
    parser.add_argument(
        '--profile', action='store_true', help='print the operators with the most GPU time'
    )
    return parser.parse_args()


def synthetic_batch(
    batch: int, size: int, num_gts: int, gt_size: tuple[float, float], num_classes: int, seed: int
) -> dict:
    """``batch`` random images with ``num_gts`` random rotated boxes each."""
    gen = torch.Generator().manual_seed(seed)
    inputs, samples = [], []
    for _ in range(batch):
        inputs.append(torch.randint(0, 256, (3, size, size), generator=gen, dtype=torch.uint8))
        centre = torch.rand(num_gts, 2, generator=gen) * size
        side = gt_size[0] + torch.rand(num_gts, 2, generator=gen) * (gt_size[1] - gt_size[0])
        angle = (torch.rand(num_gts, 1, generator=gen) - 0.5) * torch.pi
        sample = DetDataSample(
            metainfo=dict(
                img_shape=(size, size),
                ori_shape=(size, size),
                pad_shape=(size, size),
                batch_input_shape=(size, size),
                scale_factor=(1.0, 1.0),
            )
        )
        sample.gt_instances = InstanceData(
            bboxes=RotatedBoxes(torch.cat([centre, side, angle], dim=1)),
            labels=torch.randint(0, num_classes, (num_gts,), generator=gen),
        )
        samples.append(sample)
    return dict(inputs=inputs, data_samples=samples)


class PhaseTimer:
    """Wall time of the detector's parts, synchronising the GPU at the boundaries."""

    def __init__(self, model: nn.Module) -> None:
        self.active = False
        self.times: dict[str, float] = defaultdict(float)
        self._start: dict[str, float] = {}
        for name in ('backbone', 'neck'):
            module = getattr(model, name)
            module.register_forward_pre_hook(self._pre(name))
            module.register_forward_hook(self._post(name))
        head = model.bbox_head
        head.register_forward_pre_hook(self._pre('head'))
        head.register_forward_hook(self._post('head'))
        head.loss_by_feat = self._wrap('assign_loss', head.loss_by_feat)

    def _now(self) -> float:
        torch.cuda.synchronize()
        return time.perf_counter()

    def _pre(self, name: str) -> Callable[..., None]:
        def hook(*_: object) -> None:
            if self.active:
                self._start[name] = self._now()

        return hook

    def _post(self, name: str) -> Callable[..., None]:
        def hook(*_: object) -> None:
            if self.active:
                self.times[name] += self._now() - self._start[name]

        return hook

    def _wrap(self, name: str, fn: Callable[..., object]) -> Callable[..., object]:
        def wrapped(*args: object, **kwargs: object) -> object:
            if not self.active:
                return fn(*args, **kwargs)
            start = self._now()
            out = fn(*args, **kwargs)
            self.times[name] += self._now() - start
            return out

        return wrapped


class Subject:
    """One config: detector, optimiser wrapper and EMA, built once."""

    def __init__(self, path: str, args: argparse.Namespace) -> None:
        cfg = Config.fromfile(path)
        if args.cfg_options:
            cfg.merge_from_dict(args.cfg_options)
        cfg.model.backbone.checkpoint = None
        cfg.optim_wrapper.optimizer.lr = 0.0
        torch.backends.cudnn.benchmark = bool(cfg.get('env_cfg', {}).get('cudnn_benchmark', False))
        self.name = osp.splitext(osp.basename(path))[0]
        self.num_classes = cfg.model.bbox_head.num_classes
        self.model = MODELS.build(cfg.model).cuda()
        self.model.init_weights()
        self.model.train()
        self.optim_wrapper = build_optim_wrapper(self.model, cfg.optim_wrapper)
        self.ema = None
        if not args.no_ema:
            for hook in cfg.get('custom_hooks', []):
                if hook['type'] == 'EMAHook':
                    ema_cfg = {k: v for k, v in hook.items() if k not in ('type', 'priority')}
                    ema_cfg['type'] = ema_cfg.pop('ema_type')
                    self.ema = MODELS.build(ema_cfg, default_args=dict(model=self.model))
        self.timer = PhaseTimer(self.model)

    def step(self, data: dict) -> None:
        self.model.train_step(data, self.optim_wrapper)
        if self.ema is not None:
            self.ema.update_parameters(self.model)

    def measure(self, batch: int, num_gts: int, args: argparse.Namespace) -> dict:
        result = dict(config=self.name, batch=batch, num_gts=num_gts)
        data = synthetic_batch(batch, args.size, num_gts, args.gt_size, self.num_classes, seed=0)
        try:
            for _ in range(args.warmup):
                self.step(data)
            torch.cuda.synchronize()
            static = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            times = []
            for _ in range(args.iters):
                start = time.perf_counter()
                self.step(data)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
            peak = torch.cuda.max_memory_allocated()
            reserved = torch.cuda.max_memory_reserved()

            self.timer.times.clear()
            self.timer.active = True
            start = self.timer._now()
            for _ in range(args.iters):
                self.step(data)
            total = (self.timer._now() - start) / args.iters
            self.timer.active = False
        except torch.OutOfMemoryError:
            self.timer.active = False
            self.optim_wrapper.zero_grad()
            del data
            torch.cuda.empty_cache()
            result['oom'] = True
            return result
        s_step = float(torch.tensor(times).median())
        phases = {k: self.timer.times[k] / args.iters for k in PHASES[:-1]}
        phases[PHASES[-1]] = total - sum(phases.values())
        result.update(
            s_step=round(s_step, 4),
            img_s=round(batch / s_step, 3),
            peak_GB=round(peak / 1e9, 2),
            reserved_GB=round(reserved / 1e9, 2),
            static_GB=round(static / 1e9, 2),
            phases_s={k: round(v, 4) for k, v in phases.items()},
        )
        return result

    def profile(self, batch: int, num_gts: int, args: argparse.Namespace, rows: int = 40) -> str:
        """Operators of one training step, sorted by their own GPU time."""
        data = synthetic_batch(batch, args.size, num_gts, args.gt_size, self.num_classes, seed=0)
        for _ in range(args.warmup):
            self.step(data)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            self.step(data)
            torch.cuda.synchronize()
        return prof.key_averages().table(
            sort_by='self_cuda_time_total', row_limit=rows, max_name_column_width=60
        )


def markdown(results: list[dict]) -> str:
    head = ['config', 'batch', 'GTs/img', 's/step', 'img/s', 'peak GB', 'reserved GB', 'static GB']
    lines = ['| ' + ' | '.join([*head, *PHASES]) + ' |', '|---|' + '---:|' * (len(head) - 1 + 5)]
    for r in results:
        row = [r['config'], str(r['batch']), str(r['num_gts'])]
        if r.get('oom'):
            row += ['OOM'] + [''] * (4 + len(PHASES))
        else:
            row += [f'{r["s_step"]:.3f}', f'{r["img_s"]:.2f}', f'{r["peak_GB"]:.1f}']
            row += [f'{r["reserved_GB"]:.1f}', f'{r["static_GB"]:.1f}']
            row += [f'{r["phases_s"][p]:.3f}' for p in PHASES]
        lines.append('| ' + ' | '.join(row) + ' |')
    return '\n'.join(lines)


def _time_op(module: nn.Module, x: Tensor, dtype: torch.dtype, iters: int = 20) -> float:
    """Median forward + backward time (ms) of ``module`` under ``dtype`` autocast."""
    x = x.clone().requires_grad_(True)

    def run() -> None:
        with torch.autocast('cuda', dtype=dtype, enabled=dtype != torch.float32):
            out = module(x)
        out.float().mean().backward()
        module.zero_grad(set_to_none=True)
        x.grad = None

    for _ in range(5):
        run()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        run()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return float(torch.tensor(times).median()) * 1e3


def conv_bench() -> str:
    """Single convolutions of the P2 neck / adapter, NCHW vs channels_last per dtype."""
    torch.backends.cudnn.benchmark = True
    ops: dict[str, tuple[Callable[[], nn.Module], tuple[int, int, int, int]]] = {
        'neck 3x3 640->640 @256²': (lambda: nn.Conv2d(640, 640, 3, padding=1), (2, 640, 256, 256)),
        'neck dw5x5 640 @256²': (
            lambda: nn.Conv2d(640, 640, 5, padding=2, groups=640),
            (2, 640, 256, 256),
        ),
        'neck 1x1 2560->640 @256²': (lambda: nn.Conv2d(2560, 640, 1), (2, 2560, 256, 256)),
        'neck 3x3 s2 1280->1280 @256²': (
            lambda: nn.Conv2d(1280, 1280, 3, stride=2, padding=1),
            (2, 1280, 256, 256),
        ),
        'head 3x3 256->256 @256²': (lambda: nn.Conv2d(256, 256, 3, padding=1), (2, 256, 256, 256)),
        'P2 convT 2x2 1280->1280 @128²': (
            lambda: nn.ConvTranspose2d(1280, 1280, 2, 2),
            (2, 1280, 128, 128),
        ),
        'ConvFFN dw3x3 320 @128²': (
            lambda: nn.Conv2d(320, 320, 3, padding=1, groups=320),
            (2, 320, 128, 128),
        ),
        'MRFP dw5x5 320 @128²': (
            lambda: nn.Conv2d(320, 320, 5, padding=2, groups=320),
            (2, 320, 128, 128),
        ),
    }
    dtypes = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}
    cols = [f'{d} {layout}' for d in dtypes for layout in ('NCHW', 'NHWC')]
    lines = [
        'forward + backward, ms (batch 2)\n',
        '| op | ' + ' | '.join(cols) + ' |',
        '|---|' + '---:|' * len(cols),
    ]
    for name, (build, shape) in ops.items():
        row = []
        for dtype in dtypes.values():
            for fmt in (torch.contiguous_format, torch.channels_last):
                module = build().cuda().to(memory_format=fmt)
                x = torch.randn(shape, device='cuda').contiguous(memory_format=fmt)
                row.append(f'{_time_op(module, x, dtype):.1f}')
                del module, x
                torch.cuda.empty_cache()
        lines.append(f'| {name} | ' + ' | '.join(row) + ' |')
        print(lines[-1], flush=True)
    return '\n'.join(lines)


def main() -> None:
    args = parse_args()
    register_all_modules()
    print(f'{torch.cuda.get_device_name()}, torch {torch.__version__}', flush=True)
    if args.conv_bench:
        md = conv_bench()
        results: list[dict] = []
    else:
        results = []
        for path in args.configs:
            subject = Subject(path, args)
            for batch in args.batch:
                for num_gts in args.num_gts:
                    if args.profile:
                        print(subject.profile(batch, num_gts, args), flush=True)
                        continue
                    results.append(subject.measure(batch, num_gts, args))
                    print(json.dumps(results[-1]), flush=True)
            del subject
            torch.cuda.empty_cache()
        md = markdown(results)
    print(md)
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)
        with open(osp.splitext(args.out)[0] + '.md', 'w') as f:
            f.write(md + '\n')


if __name__ == '__main__':
    main()
