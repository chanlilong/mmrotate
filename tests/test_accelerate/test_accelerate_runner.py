# Copyright (c) OpenMMLab. All rights reserved.
"""AccelerateRunner end to end on a toy model.

Single process, and two CPU processes over gloo (``accelerate.test_utils.debug_launcher``)
to check the distributed behaviour that broke earlier integrations:

* no double sharding: each epoch the ranks together see every sample once;
* the shuffle changes between epochs (sampler ``set_epoch``);
* ranks that list the dataset in different orders are reconciled;
* evaluation with an odd dataset size: the sampler padding is dropped, every
  rank gets the same metrics, which match a single-process evaluation;
* checkpoints are plain mmengine files and resuming continues the run exactly.
"""

from __future__ import annotations

import json
import os
import os.path as osp
from pathlib import Path

import numpy as np
import pytest
import torch
from mmengine.config import Config
from mmengine.evaluator import BaseMetric
from mmengine.model import BaseModel
from mmengine.registry import init_default_scope
from mmengine.structures import BaseDataElement
from torch import nn

from mmrotate.registry import DATASETS, METRICS, MODELS

N_TRAIN = 20
N_VAL = 11  # odd: 2 ranks pad it to 12


@DATASETS.register_module(force=True)
class ToyAccelDataset:
    """``x_i = i / n``; ``reverse_on_rank1`` mimics an unsorted glob on another node."""

    metainfo = {'classes': ('toy',)}

    def __init__(self, n: int, reverse_on_rank1: bool = False) -> None:
        self.paths = [f'img_{i:03d}.png' for i in range(n)]
        if reverse_on_rank1 and int(os.environ.get('RANK', '0')) == 1:
            self.paths = self.paths[::-1]

    def __len__(self) -> int:
        return len(self.paths)

    def get_data_info(self, idx: int) -> dict:
        return {'img_path': self.paths[idx]}

    def __getitem__(self, idx: int) -> dict:
        path = self.paths[idx]
        x = int(path[4:7]) / len(self.paths)
        sample = BaseDataElement(metainfo={'img_path': path})
        return {'inputs': torch.tensor([x]), 'data_samples': sample}


@MODELS.register_module(force=True)
class ToyAccelModel(BaseModel):
    """``y = w * x`` regressing ``2x``; records the samples it trained on."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(1, 1, bias=False)
        self.seen: list[str] = []

    def forward(
        self, inputs: list[torch.Tensor], data_samples: list | None = None, mode: str = 'tensor'
    ) -> dict | list:
        x = torch.stack(inputs).float()
        y = self.lin(x)
        if mode == 'loss':
            self.seen += [s.img_path for s in data_samples or []]
            return {'loss': ((y - 2 * x) ** 2).mean()}
        out = []
        for s, v in zip(data_samples or [], y, strict=True):
            s = s.clone()
            s.pred = v.detach().cpu()
            out.append(s)
        return out


@METRICS.register_module(force=True)
class ToyAccelMetric(BaseMetric):
    def process(self, data_batch: object, data_samples: list) -> None:
        for s in data_samples:
            self.results.append({'path': s['img_path'], 'pred': s['pred'].numpy()})

    def compute_metrics(self, results: list) -> dict:
        paths = [r['path'] for r in results]
        return {
            'n': len(results),
            'unique': len(set(paths)),
            'ordered': float(paths == sorted(paths)),
            'pred_sum': float(np.sum([r['pred'] for r in results])),
        }


def make_cfg(work_dir: str, max_epochs: int = 2, **accelerator: object) -> Config:
    return Config(
        dict(
            default_scope='mmrotate',
            model=dict(type='ToyAccelModel'),
            work_dir=work_dir,
            train_dataloader=dict(
                dataset=dict(type='ToyAccelDataset', n=N_TRAIN, reverse_on_rank1=True),
                sampler=dict(type='DefaultSampler', shuffle=True),
                batch_size=2,
                num_workers=0,
                collate_fn=dict(type='pseudo_collate'),
            ),
            val_dataloader=dict(
                dataset=dict(type='ToyAccelDataset', n=N_VAL),
                sampler=dict(type='DefaultSampler', shuffle=False),
                batch_size=2,
                num_workers=0,
                collate_fn=dict(type='pseudo_collate'),
            ),
            val_evaluator=dict(type='ToyAccelMetric'),
            train_cfg=dict(by_epoch=True, max_epochs=max_epochs, val_interval=1),
            val_cfg=dict(type='ValLoop'),
            optim_wrapper=dict(optimizer=dict(type='SGD', lr=0.1, momentum=0.9)),
            param_scheduler=[dict(type='MultiStepLR', milestones=[1], gamma=0.5, by_epoch=True)],
            default_hooks=dict(
                checkpoint=dict(type='CheckpointHook', interval=1),
                logger=dict(type='LoggerHook', interval=1),
            ),
            custom_hooks=[dict(type='EMAHook', momentum=0.5)],
            randomness=dict(seed=0),
            accelerator=dict(mixed_precision='no', **accelerator),
            log_level='WARNING',
        )
    )


def build_runner(cfg: Config) -> object:
    from accelerate.state import AcceleratorState, GradientState, PartialState

    from mmrotate.accelerate import AccelerateRunner

    init_default_scope('mmrotate')
    # Fresh singletons: debug_launcher forks this (already initialised) process.
    AcceleratorState._reset_state(reset_partial_state=True)
    PartialState._reset_state()
    GradientState._reset_state()
    return AccelerateRunner.from_cfg(cfg)


def _dist_run(work_dir: str) -> None:
    import torch.distributed as dist

    cfg = make_cfg(work_dir)
    runner = build_runner(cfg)
    runner.train()
    rank = dist.get_rank()
    model = runner.model.module

    def scalar(key: str) -> float:
        return float(runner.message_hub.get_scalar(key).current())

    with open(osp.join(work_dir, f'seen_{rank}.json'), 'w') as f:
        json.dump(
            {
                'seen': model.seen,
                'val_n': scalar('val/n'),
                'val_unique': scalar('val/unique'),
                'val_ordered': scalar('val/ordered'),
                'pred_sum': scalar('val/pred_sum'),
                'weight': float(model.lin.weight.item()),
                'world': dist.get_world_size(),
            },
            f,
        )


def test_single_process_train_val_resume(tmp_path: Path) -> None:
    work = str(tmp_path / 'single')
    runner = build_runner(make_cfg(work, max_epochs=3))
    runner.train()
    full_w = float(runner.model.module.lin.weight.item())
    hub = runner.message_hub
    assert hub.get_scalar('val/n').current() == N_VAL
    assert hub.get_scalar('val/unique').current() == N_VAL
    assert hub.get_scalar('val/ordered').current() == 1.0
    # Each epoch is one pass over the data.
    seen = runner.model.module.seen
    assert len(seen) == 3 * N_TRAIN and len(set(seen[:N_TRAIN])) == N_TRAIN
    assert seen[:N_TRAIN] != seen[N_TRAIN : 2 * N_TRAIN]  # reshuffled per epoch
    ckpt = torch.load(osp.join(work, 'epoch_2.pth'), map_location='cpu', weights_only=False)
    assert {'state_dict', 'optimizer', 'param_schedulers', 'ema_state_dict'} <= set(ckpt)
    assert 'lin.weight' in ckpt['state_dict']

    # Resume from epoch 2 and finish epoch 3: same weights as the uninterrupted run.
    resume_work = str(tmp_path / 'resume')
    os.makedirs(resume_work)
    cfg = make_cfg(resume_work, max_epochs=3)
    cfg.resume = True
    cfg.load_from = osp.join(work, 'epoch_2.pth')
    runner2 = build_runner(cfg)
    runner2.train()
    assert runner2.model.module.lin.weight.item() == pytest.approx(full_w, rel=1e-6)


def test_test_only_run_does_not_build_train_data(tmp_path: Path) -> None:
    """``runner.test()`` must not touch ``train_dataloader`` (it may not exist on disk)."""
    work = str(tmp_path / 'train')
    build_runner(make_cfg(work, max_epochs=1)).train()
    cfg = make_cfg(str(tmp_path / 'test'))
    cfg.train_dataloader.dataset = dict(type='NoSuchDatasetAnywhere')
    cfg.test_dataloader = cfg.val_dataloader
    cfg.test_evaluator = cfg.val_evaluator
    cfg.test_cfg = dict(type='TestLoop')
    cfg.load_from = osp.join(work, 'epoch_1.pth')
    metrics = build_runner(cfg).test()
    assert metrics['n'] == metrics['unique'] == N_VAL


def test_two_processes_gloo(tmp_path: Path) -> None:
    from accelerate import debug_launcher

    work = str(tmp_path / 'dist')
    os.makedirs(work)
    debug_launcher(_dist_run, args=(work,), num_processes=2)
    runs = []
    for rank in range(2):
        with open(osp.join(work, f'seen_{rank}.json')) as f:
            runs.append(json.load(f))
    assert all(r['world'] == 2 for r in runs)
    per_epoch = N_TRAIN // 2
    for epoch in range(2):
        chunks = [r['seen'][epoch * per_epoch : (epoch + 1) * per_epoch] for r in runs]
        # No double sharding: the two ranks split the epoch, together covering it once.
        assert len(chunks[0]) == len(chunks[1]) == per_epoch
        assert set(chunks[0]).isdisjoint(chunks[1])
        assert set(chunks[0]) | set(chunks[1]) == {f'img_{i:03d}.png' for i in range(N_TRAIN)}
    assert runs[0]['seen'][:per_epoch] != runs[0]['seen'][per_epoch:]  # reshuffled
    # Evaluation: padding dropped, dataset order, identical metrics on both ranks.
    for r in runs:
        assert r['val_n'] == r['val_unique'] == N_VAL
        assert r['val_ordered'] == 1.0
    assert runs[0]['pred_sum'] == runs[1]['pred_sum']
    assert runs[0]['weight'] == runs[1]['weight']  # DDP kept the replicas in sync
    assert not os.listdir(osp.join(work, 'accelerate_eval')), 'bridge files left behind'
