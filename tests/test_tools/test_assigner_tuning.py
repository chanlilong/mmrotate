# Copyright (c) OpenMMLab. All rights reserved.
"""Tests for tools/assigner_tuning (assigner probe and successive-halving search)."""

import importlib.util
import json
import os.path as osp
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.structures import InstanceData

from mmrotate.registry import TASK_UTILS

_DIR = osp.join(osp.dirname(__file__), '..', '..', 'tools', 'assigner_tuning')


def _load(name: str) -> ModuleType:
    sys.path.insert(0, _DIR)  # assigner_search imports assigner_probe
    spec = importlib.util.spec_from_file_location(name, osp.join(_DIR, f'{name}.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = _load('assigner_probe')
search = _load('assigner_search')
log_report = _load('assignment_log_report')
init_default_scope('mmrotate')


def test_parse_sweep_and_paths() -> None:
    sweep = probe.parse_sweep(['topk=9,13', 'iou_calculator.constant=12.5,none', 'mode=a'])
    assert sweep == {'topk': [9, 13], 'iou_calculator.constant': [12.5, None], 'mode': ['a']}
    cfg: dict = {'iou_calculator': {'constant': 1}}
    probe.set_path(cfg, 'iou_calculator.constant', 2)
    probe.set_path(cfg, 'new.key', 3)
    assert probe.get_path(cfg, 'iou_calculator.constant') == 2
    assert probe.get_path(cfg, 'new.key') == 3
    assert probe.get_path(cfg, 'missing.key') is None
    with pytest.raises(ValueError):
        probe.parse_sweep(['topk'])


def test_expand_family_oat_and_grid() -> None:
    base = {'type': 'A', 'topk': 9, 'radius': 3.0}
    oat = probe.expand_family('a', base, {'topk': [5, 9, 13], 'radius': [1.0]}, 'oat')
    # the base plus every value that differs from it, one key at a time
    assert [c.changed for c in oat] == [{}, {'topk': 5}, {'topk': 13}, {'radius': 1.0}]
    assert oat[1].cfg == {'type': 'A', 'topk': 5, 'radius': 3.0}
    assert base['topk'] == 9  # not mutated
    grid = probe.expand_family('a', base, {'topk': [5, 9], 'radius': [1.0, 3.0]}, 'grid')
    assert len(grid) == 4  # base + 3 new combinations (topk=9, radius=3 is the base)


def test_canonical_ignores_spelled_out_defaults() -> None:
    short = {'type': 'mmdet.DynamicSoftLabelAssigner', 'topk': 13}
    full = {'type': 'mmdet.DynamicSoftLabelAssigner', 'topk': 13, 'soft_center_radius': 3.0}
    other = {'type': 'mmdet.DynamicSoftLabelAssigner', 'topk': 9}
    assert probe.canonical(short) == probe.canonical(full) != probe.canonical(other)
    # subclass arguments and the parent's defaults are both known
    rka = {'type': 'RankingDynamicSoftLabelAssigner', 'rka_topk': 3, 'topk': 13}
    assert probe.canonical(rka) == probe.canonical({'type': 'RankingDynamicSoftLabelAssigner'})


def test_load_spec_none_values(tmp_path: Path) -> None:
    spec = tmp_path / 'spec.yaml'
    spec.write_text(
        'c2f:\n  base: {type: C2FAssigner, constraint: dgmm}\n  sweep: {constraint: [none]}\n'
    )
    cands = probe.load_spec(str(spec), 'oat')
    assert [c.name for c in cands] == ['c2f', 'c2f: constraint=none']
    assert cands[1].cfg['constraint'] is None


def test_pick_images_stratified() -> None:
    class FakeDataset:
        def __init__(self, counts: list[int]) -> None:
            self.counts = counts

        def __len__(self) -> int:
            return len(self.counts)

        def get_data_info(self, k: int) -> dict:
            return {'instances': [{'ignore_flag': 0}] * self.counts[k]}

    counts = [10] * 50 + [100] * 50 + [300] * 50 + [600] * 2
    picked = probe.pick_images(FakeDataset(counts), 20, 0)
    got = np.array(counts)[picked]
    assert len(picked) == 20 and len(set(picked)) == 20
    assert (got == 600).sum() == 2  # the small bucket is taken whole, the rest share the quota
    assert {(got == c).sum() for c in (10, 100, 300)} <= {6, 7}


def _point_priors() -> torch.Tensor:
    ys, xs = torch.meshgrid(torch.arange(2.0, 64, 4), torch.arange(2.0, 64, 4), indexing='ij')
    xy = torch.stack([xs.flatten(), ys.flatten()], 1)
    return torch.cat([xy, torch.full_like(xy, 4.0)], 1)


def _sample(gts: list[list[float]], labels: list[int]) -> 'probe.Sample':
    priors = _point_priors()
    n = priors.size(0)
    gtb = torch.tensor(gts)
    # every prior predicts the box of its nearest GT, slightly shifted
    near = torch.cdist(priors[:, :2], gtb[:, :2]).argmin(1)
    bboxes = gtb[near].clone()
    bboxes[:, 0] += 1.0
    pred = InstanceData(priors=priors, bboxes=bboxes, scores=torch.zeros(n, 2))
    gt = InstanceData(bboxes=gtb, labels=torch.tensor(labels))
    return probe.Sample(pred, gt, None)


def test_assignment_stats_and_duplicates() -> None:
    assigner = TASK_UTILS.build(
        dict(
            type='RankingAssigner',
            topk=3,
            point_prior_scale=4.0,
            iou_calculator=dict(type='RBboxNWD2D', constant=8.0),
        )
    )
    gts = [[20.0, 20.0, 10.0, 10.0, 0.0], [44.0, 44.0, 6.0, 6.0, 0.0], [44.0, 44.0, 6.0, 6.0, 0.0]]
    stats = probe.assignment_stats(assigner, [_sample(gts, [0, 1, 1])], 'cpu')
    # the two identical GTs count as one object that owns all of their positives
    assert stats['duplicates'] == 1
    assert sorted(stats['gt_size'].round(3).tolist()) == [6.0, 10.0]
    assert sorted(stats['gt_npos'].tolist()) == [3, 3]
    assert stats['pos_iou'].size == 6 and (stats['pos_iou'] > 0.6).all()
    assert (stats['gt_best'] > 0.6).all()
    summary = probe.summarize(stats, (8.0,))
    assert summary['starved'] == 0 and summary['pos_per_gt'] == 3
    assert summary['score'] == pytest.approx(summary['coverage'] * summary['pos_iou'])
    assert summary['by_size']['<8']['gts'] == 1 and summary['by_size']['>=8']['gts'] == 1
    assert summary['duplicates'] == 1
    assert set(summary['by_level']['<8']) == {'s4'}


def test_starved_gt_is_counted() -> None:
    # topk=1 and two GTs on the same prior: the weaker one loses its only candidate
    assigner = TASK_UTILS.build(
        dict(
            type='RankingAssigner',
            topk=1,
            point_prior_scale=4.0,
            iou_calculator=dict(type='RBboxNWD2D', constant=8.0),
        )
    )
    gts = [[22.0, 22.0, 8.0, 8.0, 0.0], [22.5, 22.5, 6.0, 6.0, 0.0]]
    stats = probe.assignment_stats(assigner, [_sample(gts, [0, 0])], 'cpu')
    summary = probe.summarize(stats, (8.0,))
    assert summary['starved'] == 0.5
    assert sorted(stats['gt_best'].tolist())[0] == 0.0


def test_rank_orders_best_first() -> None:
    def res(name: str, cov: float, starved: float) -> dict:
        s = {'coverage': cov, 'starved': starved, 'by_size': {'<8': {'coverage': 1 - cov}}}
        return {'name': name, 'summary': s}

    rs = [res('a', 0.5, 0.2), res('b', 0.7, 0.3), res('c', 0.6, 0.1)]
    assert [r['name'] for r in probe.rank(rs, 'coverage', None)] == ['b', 'c', 'a']
    assert [r['name'] for r in probe.rank(rs, 'starved', None)] == ['c', 'a', 'b']
    assert [r['name'] for r in probe.rank(rs, 'coverage', '<8')] == ['a', 'c', 'b']


def test_search_trial_config_and_metric(tmp_path: Path) -> None:
    cfg = Config(
        dict(
            model=dict(train_cfg=dict(assigner=dict(type='A', topk=9))),
            train_dataloader=dict(dataset=dict(type='D')),
            train_cfg=dict(type='EpochBasedTrainLoop', max_epochs=6, val_interval=1),
        )
    )
    cand = probe.Candidate('b: topk=3', 'b', {'type': 'B', 'topk': 3})
    out = search.trial_config(cfg, cand, [1, 5], 'model.train_cfg.assigner')
    assert out.model.train_cfg.assigner.to_dict() == {'type': 'B', 'topk': 3}
    assert out.train_dataloader.dataset.indices == [1, 5]
    assert out.train_cfg.val_interval == 6
    assert cfg.model.train_cfg.assigner.type == 'A'  # input untouched
    assert search.slug('dsl: topk=3 iou_calculator.constant=16.6') == (
        'dsl__topk_3_iou_calculator.constant_16.6'
    )

    run = tmp_path / '20260101_000000' / 'vis_data'
    run.mkdir(parents=True)
    (run / 'scalars.json').write_text(
        '{"loss": 1.0, "step": 1}\n{"soda/mAP": 0.25, "step": 1}\n{"soda/mAP": 0.31, "step": 2}\n'
    )
    assert search.read_metric(str(tmp_path), 'soda/mAP') == pytest.approx(0.31)
    assert search.read_metric(str(tmp_path), 'dota/mAP') is None


def test_search_candidates_from_probe(tmp_path: Path) -> None:
    results = [
        dict(name=f'c{i}', family='c', cfg={'type': 'C', 'k': i}, changed={'k': i})
        for i in range(4)
    ]
    path = tmp_path / 'probe.json'
    path.write_text(json.dumps({'results': results}))
    assert [c.name for c in search.candidates_from_probe(str(path), 2, None)] == ['c0', 'c1']
    assert [c.cfg['k'] for c in search.candidates_from_probe(str(path), 2, ['c3'])] == [3]
    with pytest.raises(ValueError):
        search.candidates_from_probe(str(path), 2, ['nope'])


def _capped(name: str, score: float, starved: float, starved_lt8: float) -> dict:
    s = {'score': score, 'starved': starved, 'by_size': {'<8': {'starved': starved_lt8}}}
    return {'name': name, 'family': 'c', 'cfg': {'type': 'C'}, 'changed': {}, 'summary': s}


def test_max_starved_caps(tmp_path: Path) -> None:
    names = probe.bin_names((8.0,))
    assert probe.parse_caps(['<8:0.02', 'all:0.01'], names) == {'<8': 0.02, 'all': 0.01}
    assert probe.parse_caps(None, names) == {}
    for bad in ('<8', '<4:0.1'):
        with pytest.raises(ValueError):
            probe.parse_caps([bad], names)
    caps = {'<8': 0.02}
    rs = [_capped('dsl', 0.64, 0.008, 0.06), _capped('rka', 0.60, 0.0, 0.0)]
    assert probe.violations(rs[0]['summary'], caps) == ['<8 6.0% > 2%']
    # without caps the score decides, with them the starving candidate goes last
    assert [r['name'] for r in probe.rank(rs, 'score', None)] == ['dsl', 'rka']
    ranked = probe.rank(rs, 'score', None, caps)
    assert [r['name'] for r in ranked] == ['rka', 'dsl']
    assert ranked[1]['violations'] and not ranked[0]['violations']
    # the search takes its --top among the candidates within the caps
    path = tmp_path / 'probe.json'
    path.write_text(json.dumps({'results': rs}))
    picked = search.candidates_from_probe(str(path), 1, None, ['<8:0.02'])
    assert [c.name for c in picked] == ['rka']
    assert [c.name for c in search.candidates_from_probe(str(path), 1, None)] == ['dsl']


def _log(npos: list[list[int]], sizes: list[float], seen: list[list[int]]) -> dict:
    n_img = len(seen[0])
    return dict(
        images=np.array([f'{i}.png' for i in range(n_img)]),
        seen=np.array(seen),
        obj_image=np.arange(len(sizes)) % n_img,
        obj_size=np.array(sizes),
        obj_label=np.zeros(len(sizes), int),
        obj_density=np.ones(len(sizes), int),
        npos=np.array(npos),
        best_iou=(np.array(npos) > 0) * 0.5,
    )


def test_assignment_log_report(tmp_path: Path) -> None:
    # 4 tiny objects over 4 epochs: never / starved in the second half only / every epoch /
    # starved in epoch 1 only; one 20 px object always matched; images 0..4 seen every epoch
    npos = [[0, 1, 2, 0, 3], [0, 1, 2, 1, 3], [0, 0, 2, 1, 3], [0, 0, 2, 1, 3]]
    log = _log(npos, [4, 5, 6, 7, 20], [[1] * 5] * 4)
    s = log_report.summarize(log, (8.0,))
    assert s['epochs'] == [1, 2, 3, 4]
    tiny = s['by_size']['<8']
    assert tiny['objects'] == 4
    assert tiny['never'] == pytest.approx(0.25)
    assert tiny['late_never'] == pytest.approx(0.5)
    assert tiny['always'] == pytest.approx(0.25)
    assert tiny['starved_per_epoch'] == pytest.approx(7 / 16)
    assert tiny['starved_last'] == pytest.approx(0.5)
    assert tiny['never_so_far'] == pytest.approx([0.5, 0.25, 0.25, 0.25])
    assert tiny['best_iou'] == pytest.approx(0.5)
    assert s['by_size']['>=8']['never'] == 0 and s['all']['objects'] == 5

    # an epoch without visits (partial run) is dropped; an image missing an epoch is not counted
    partial = _log([[1, 0], [0, 0], [0, 0]], [4, 4], [[1, 1], [1, 0], [0, 0]])
    sp = log_report.summarize(partial, (8.0,))
    assert sp['epochs'] == [1, 2] and sp['all']['objects'] == 1 and sp['all']['never'] == 0

    # two rank files of one run are merged by image path
    run = tmp_path / 'run'
    run.mkdir()
    np.savez(run / 'assignment_log_rank0.npz', **_log([[0], [0]], [4.0], [[1], [0]]))
    np.savez(run / 'assignment_log_rank1.npz', **_log([[0], [2]], [4.0], [[0], [1]]))
    merged = log_report.load_log(str(run))
    assert merged['seen'].tolist() == [[1], [1]] and merged['npos'].tolist() == [[0], [2]]
    out = tmp_path / 'report.md'
    log_report.main([f'A={run}', str(run / 'assignment_log_rank0.npz'), '--out', str(out)])
    text = out.read_text()
    assert '| A | <8 | 1 |' in text and 'not matched in any epoch so far' in text
    assert json.loads((tmp_path / 'report.json').read_text())['A']['by_size']['<8']['never'] == 0
