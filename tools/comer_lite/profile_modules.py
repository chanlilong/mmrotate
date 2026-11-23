# Copyright (c) OpenMMLab. All rights reserved.
"""Per-module GPU time and GFLOP of ViT-CoMer detectors.

For every config, builds the detector (random weights, the timing does not
depend on them) and measures, at batch ``--batch`` and ``--size``² with bf16
autocast:

* **eval**: forward time per module (no grad), and the whole forward;
* **train**: forward time per module with autograd on, and the whole
  forward + backward step with the config's gradient checkpointing
  (``with_cp`` / ``vit_with_cp``; ``--no-cp`` turns both off, so the
  difference between the two runs is the recompute cost). The loss is a sum
  over the head outputs, so assignment and loss kernels are not timed, and
  there is no optimiser step. Backward is not split per module: autograd
  interleaves the CNN and ViT branches, so per-module backward windows overlap,
  and module hooks do not fire during checkpoint recompute;
* **GFLOP** per module and per image (``torch.utils.flop_counter``: matmul,
  conv and attention only; deformable sampling is not counted).

All models are built first and then timed round-robin for ``--rounds``
rounds; every reported time is the median over the rounds, so clock or power
drift during the run hits every model alike. Memory is the training-step peak
above what was allocated before it (activations, gradients, workspace).

Groups: ``spm``, ``mrfp``, ``cti_tov``, ``cti_toc``, ``extra_cti``,
``vit_blocks``, ``adapter_other`` (backbone minus the above: patch embed,
level embed, output glue), ``neck``, ``head``. Prints markdown tables and
writes JSON (+ ``.md``)::

    python tools/comer_lite/profile_modules.py configs/comer_lite/*.yaml \\
        configs/soda_dinov3/vits-dinov3-comer_rtmdet-p2-6e-soda.yaml --out profile.json
"""

import argparse
import json
import os.path as osp
import re
from collections import defaultdict
from collections.abc import Callable

import torch
from mmengine.config import Config, DictAction
from torch import Tensor, nn
from torch.utils.flop_counter import FlopCounterMode

from mmrotate.registry import MODELS
from mmrotate.utils import register_all_modules

ADAPTER = ('spm', 'mrfp', 'cti_tov', 'cti_toc', 'extra_cti', 'vit_blocks')
GROUPS = (*ADAPTER, 'adapter_other', 'neck', 'head')
# (regex on the module name, group); the first match wins.
PATTERNS = (
    (r'^backbone\.spm$', 'spm'),
    (r'^backbone\.interactions\.\d+\.mrfp$', 'mrfp'),
    (r'^backbone\.interactions\.\d+\.cti_tov$', 'cti_tov'),
    (r'^backbone\.interactions\.\d+\.cti_toc$', 'cti_toc'),
    (r'^backbone\.interactions\.\d+\.extra_CTIs\.\d+$', 'extra_cti'),
    (r'^backbone\.vit\.model\.blocks\.\d+$', 'vit_blocks'),
    (r'^backbone$', 'backbone'),
    (r'^neck$', 'neck'),
    (r'^bbox_head$', 'head'),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Per-module CoMer profile')
    parser.add_argument('configs', nargs='+')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--size', type=int, default=800)
    parser.add_argument('--iters', type=int, default=10, help='timed iterations per round')
    parser.add_argument('--warmup', type=int, default=3, help='untimed iterations per round')
    parser.add_argument('--rounds', type=int, default=5, help='round-robin rounds (median)')
    parser.add_argument('--no-cp', action='store_true', help='disable gradient checkpointing')
    parser.add_argument('--out', help='optional JSON output (a .md table is written next to it)')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    return parser.parse_args()


def _group(name: str) -> str | None:
    for pattern, group in PATTERNS:
        if re.match(pattern, name):
            return group
    return None


class ModuleTimer:
    """CUDA-event timing of the tracked modules' forward calls."""

    def __init__(self, model: nn.Module) -> None:
        self.events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self.open: dict[int, list[torch.cuda.Event]] = defaultdict(list)
        self.active = False
        self.handles = []
        for name, module in model.named_modules():
            group = _group(name)
            if group is not None:
                self.handles.append(module.register_forward_pre_hook(self._start))
                self.handles.append(module.register_forward_hook(self._stop(group)))

    def _start(self, module: nn.Module, *_: object) -> None:
        if self.active:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self.open[id(module)].append(event)

    def _stop(self, group: str) -> Callable[..., None]:
        def hook(module: nn.Module, *_: object) -> None:
            if self.active and self.open[id(module)]:
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                self.events.append((group, self.open[id(module)].pop(), end))

        return hook

    def collect(self, iters: int) -> dict[str, float]:
        torch.cuda.synchronize()
        out: dict[str, float] = defaultdict(float)
        for group, start, end in self.events:
            out[group] += start.elapsed_time(end) / iters
        self.events.clear()
        self.open.clear()
        return _adapter_other(out)

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()


def _adapter_other(values: dict[str, float]) -> dict[str, float]:
    values = dict(values)
    if 'backbone' in values:
        values['adapter_other'] = values.pop('backbone') - sum(values.get(g, 0.0) for g in ADAPTER)
    return values


def _flatten(o: object) -> list[Tensor]:
    if isinstance(o, Tensor):
        return [o]
    if isinstance(o, list | tuple):
        return [t for x in o for t in _flatten(x)]
    return []


def _forward(model: nn.Module, x: Tensor) -> list[Tensor]:
    """Backbone, neck and head forward (no loss, no NMS); every module via ``__call__``."""
    return _flatten(model.bbox_head(model.extract_feat(x)))


def flops_per_module(model: nn.Module, size: int) -> dict[str, float]:
    """GFLOP per image. Module names in the counter are prefixed by the root class name."""
    counter = FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        model(torch.randn(1, 3, size, size, device='cuda'), mode='tensor')
    counts = counter.get_flop_counts()
    gflop: dict[str, float] = defaultdict(float)
    for name, ops in counts.items():
        group = _group(name.split('.', 1)[1]) if '.' in name else None
        if group is not None:
            gflop[group] += sum(ops.values()) / 1e9
    gflop['total'] = sum(counts['Global'].values()) / 1e9
    # RTMDet calls ``bbox_head.forward`` directly, so the counter has no head entry.
    gflop['head'] = gflop['total'] - gflop['backbone'] - gflop['neck']
    return _adapter_other(gflop)


def _timed(
    fn: Callable[[], None], timer: ModuleTimer, warmup: int, iters: int
) -> tuple[float, dict[str, float]]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timer.active = True
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    timer.active = False
    times = timer.collect(iters)
    return start.elapsed_time(end) / iters, times


class Subject:
    """One detector under test: built once, then timed in every round."""

    def __init__(self, path: str, args: argparse.Namespace) -> None:
        cfg = Config.fromfile(path)
        if args.cfg_options:
            cfg.merge_from_dict(args.cfg_options)
        cfg.model.backbone.checkpoint = None
        if args.no_cp:
            cfg.model.backbone.update(with_cp=False, vit_with_cp=False)
        self.model = MODELS.build(cfg.model).cuda()
        self.model.init_weights()
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.name = osp.basename(path).replace('_rtmdet-p2-6e-soda.yaml', '')
        self.model.eval()
        self.gflop = flops_per_module(self.model, args.size)
        self.timer = ModuleTimer(self.model)
        self.rounds: dict[str, list] = defaultdict(list)

    def measure(self, x: Tensor, warmup: int, iters: int) -> None:
        model, params = self.model, self.params

        def eval_step() -> None:
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                _forward(model, x)

        def train_step() -> None:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = sum(t.float().mean() for t in _forward(model, x))
            loss.backward()
            for p in params:
                p.grad = None

        model.eval()
        ms, fwd = _timed(eval_step, self.timer, warmup, iters)
        self.rounds['eval_ms'].append(ms)
        self.rounds['eval_fwd'].append(fwd)

        model.train()
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        ms, fwd = _timed(train_step, self.timer, warmup, iters)
        self.rounds['train_ms'].append(ms)
        self.rounds['train_fwd'].append(fwd)
        self.rounds['train_act_mem_GB'].append((torch.cuda.max_memory_allocated() - before) / 1e9)

    def summary(self, args: argparse.Namespace) -> dict:
        def median(values: list[float]) -> float:
            return float(torch.tensor(values).median())

        out: dict = dict(
            config=self.name,
            batch=args.batch,
            size=args.size,
            cp=not args.no_cp,
            rounds=len(self.rounds['eval_ms']),
        )
        for key, values in self.rounds.items():
            if isinstance(values[0], dict):
                groups = {g for v in values for g in v}
                out[key] = {g: median([v.get(g, 0.0) for v in values]) for g in groups}
            else:
                out[key] = median(values)
                out[f'{key}_all'] = [round(v, 2) for v in values]
        out['train_act_mem_GB'] = round(out['train_act_mem_GB'], 2)
        out['gflop_per_img'] = self.gflop
        return out


def _fmt(v: float | None) -> str:
    return '-' if v is None else f'{v:.1f}'


def markdown(results: list[dict]) -> str:
    batch = results[0]['batch']
    lines = []
    for title, key in (
        (f'eval forward, ms per batch of {batch}', 'eval_fwd'),
        (f'train forward (autograd on), ms per batch of {batch}', 'train_fwd'),
        ('GFLOP per image', 'gflop_per_img'),
    ):
        cols = [*GROUPS, 'total'] if key == 'gflop_per_img' else list(GROUPS)
        lines += [f'\n### {title}\n', '| config | ' + ' | '.join(cols) + ' |']
        lines.append('|---|' + '---:|' * len(cols))
        for r in results:
            vals = [_fmt(r[key].get(c)) for c in cols]
            lines.append(f'| {r["config"]} | ' + ' | '.join(vals) + ' |')
    lines += [
        f'\n### totals (batch {batch})\n',
        '| config | grad ckpt | eval ms | eval img/s | train fwd ms | train fwd+bwd ms '
        '| train img/s (no loss / optimiser) | train activation GB |',
        '|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    for r in results:
        fwd = sum(r['train_fwd'].get(g, 0.0) for g in GROUPS)
        lines.append(
            f'| {r["config"]} | {r["cp"]} | {r["eval_ms"]:.1f} | '
            f'{r["batch"] * 1000 / r["eval_ms"]:.1f} | {fwd:.1f} | {r["train_ms"]:.1f} | '
            f'{r["batch"] * 1000 / r["train_ms"]:.1f} | {r["train_act_mem_GB"]} |'
        )
    return '\n'.join(lines)


def main() -> None:
    args = parse_args()
    register_all_modules()
    torch.backends.cudnn.benchmark = True
    subjects = [Subject(path, args) for path in args.configs]
    x = torch.randn(args.batch, 3, args.size, args.size, device='cuda')
    # Round-robin over the models, so clock / power drift hits them all alike;
    # every number is the median over the rounds.
    for r in range(args.rounds):
        for subject in subjects:
            subject.measure(x, args.warmup, args.iters)
            print(
                f'round {r}: {subject.name} eval {subject.rounds["eval_ms"][-1]:.1f} ms, '
                f'train {subject.rounds["train_ms"][-1]:.1f} ms',
                flush=True,
            )
    results = [subject.summary(args) for subject in subjects]
    print(json.dumps(results), flush=True)
    md = markdown(results)
    print(md)
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)
        with open(osp.splitext(args.out)[0] + '.md', 'w') as f:
            f.write(md + '\n')


if __name__ == '__main__':
    main()
