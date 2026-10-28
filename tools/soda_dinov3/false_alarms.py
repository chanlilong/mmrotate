# Copyright (c) OpenMMLab. All rights reserved.
"""False-alarm (false-positive) analysis of the SODA-A DINOv3 runs.

Reads the test-split predictions of every run under ``work_root``
(``<run>/preds_test.pkl``, written by ``tools/test.py --out``, see
``run_experiments.sh dump``) and writes ``false_alarms.md``,
``false_alarms.json`` and ``fppi_recall.png``::

    python tools/soda_dinov3/false_alarms.py work_dirs/soda_dinov3

Detections are matched at IoU 0.5 with the rule of ``SODAMetric``
(``mmrotate/evaluation/metrics/soda/soda_eval.py``): per class, each detection
goes to the GT it overlaps most and, in score order, the first one reaching
the threshold is a TP. Detections whose best match is an ignored GT are
neither TP nor FP. Every FP gets one TIDE-style type (first that applies):

- ``dup``: IoU >= 0.5 with a same-class GT that an earlier detection took;
- ``cls``: IoU >= 0.5 with a GT of another class (right box, wrong label);
- ``loc``: 0.1 <= IoU < 0.5 with a same-class GT (poorly localised);
- ``cls+loc``: 0.1 <= IoU < 0.5 with a GT of another class;
- ``bkg``: IoU < 0.1 with every GT (a detection on background).

FPPI is false positives per 800x800 test patch.
"""

import argparse
import glob
import json
import os.path as osp
from collections.abc import Sequence

import numpy as np
import torch
from mmcv.ops import box_iou_rotated
from mmengine.fileio import load

IOU_FG = 0.5
IOU_BG = 0.1
TP, IGNORED = 0, -1
FP_TYPES = ('dup', 'cls', 'loc', 'cls+loc', 'bkg')  # codes 1..5
CLASSES = (
    'airplane',
    'helicopter',
    'small-vehicle',
    'large-vehicle',
    'ship',
    'container',
    'storage-tank',
    'swimming-pool',
    'windmill',
)
# SODA-A bins on sqrt(w * h), plus everything above them.
SIZE_EDGES = (12.0, 20.0, 32.0, 40.0)
SIZE_NAMES = ('eS', 'rS', 'gS', 'N', '>40')
FPPI_POINTS = (1.0, 2.0, 5.0, 10.0)
RECALL_POINTS = (0.5, 0.6, 0.7, 0.8)
FIXED_THR = 0.3


def _np(x: object) -> np.ndarray:
    """Tensor / ``BaseBoxes`` / array -> numpy."""
    x = getattr(x, 'tensor', x)
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _iou(a: np.ndarray, b: np.ndarray, device: str) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ious = box_iou_rotated(
        torch.from_numpy(a).float().to(device), torch.from_numpy(b).float().to(device)
    )
    return ious.cpu().numpy()


def classify_detections(
    dets: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    gts: np.ndarray,
    gt_labels: np.ndarray,
    ignore: np.ndarray | None = None,
    ignore_labels: np.ndarray | None = None,
    device: str = 'cpu',
) -> np.ndarray:
    """Outcome code of every detection of one image.

    Args:
        dets: ``(m, 5)`` rotated boxes ``(cx, cy, w, h, a)``.
        scores: ``(m,)`` scores.
        labels: ``(m,)`` class indices.
        gts: ``(n, 5)`` GT boxes.
        gt_labels: ``(n,)`` GT classes.
        ignore: ``(k, 5)`` ignored GT boxes.
        ignore_labels: ``(k,)`` their classes.
        device: Device for the rotated IoU.

    Returns:
        ``(m,)`` int8: 0 = TP, -1 = ignored, ``1 + FP_TYPES.index(type)`` for an FP.
    """
    m = len(dets)
    codes = np.full(m, len(FP_TYPES), dtype=np.int8)  # bkg unless shown otherwise
    if m == 0:
        return codes
    if ignore is None or ignore_labels is None:
        ignore, ignore_labels = np.zeros((0, 5)), np.zeros(0, dtype=np.int64)
    all_gts = np.concatenate([gts.reshape(-1, 5), ignore.reshape(-1, 5)])
    all_labels = np.concatenate([gt_labels, ignore_labels])
    n = len(gts)
    ious = _iou(dets, all_gts, device)  # (m, n + k)
    same = labels[:, None] == all_labels[None]
    iou_same = np.where(same, ious, 0.0)
    best = iou_same.argmax(axis=1) if ious.size else np.zeros(m, dtype=np.int64)
    best_iou = iou_same.max(axis=1) if ious.size else np.zeros(m)
    other = np.where(same[:, :n], 0.0, ious[:, :n]).max(axis=1) if n else np.zeros(m)

    matched = best_iou >= IOU_FG
    codes[matched & (best >= n)] = IGNORED
    order = np.argsort(-scores, kind='stable')
    idx = order[(matched & (best < n))[order]]  # GT matches in score order
    _, first = np.unique(best[idx], return_index=True)
    codes[idx] = 1  # dup
    codes[idx[first]] = TP

    rest = ~matched
    codes[rest & (other >= IOU_FG)] = 2  # cls
    codes[rest & (other < IOU_FG) & (best_iou >= IOU_BG)] = 3  # loc
    codes[rest & (other < IOU_FG) & (best_iou < IOU_BG) & (other >= IOU_BG)] = 4  # cls+loc
    return codes


def collect_run(pkl_path: str, device: str) -> dict:
    """Per-detection arrays of one run (score, code, class, size bin, image)."""
    samples = load(pkl_path)
    score, code, label, size, img = [], [], [], [], []
    gt_per_class = np.zeros(len(CLASSES), dtype=np.int64)
    empty = np.zeros(len(samples), dtype=bool)
    for i, s in enumerate(samples):
        pred, gt = s['pred_instances'], s.get('gt_instances') or {}
        ign = s.get('ignored_instances') or {}
        boxes = _np(pred['bboxes']).reshape(-1, 5)
        scores = _np(pred['scores']).astype(np.float32)
        labels = _np(pred['labels']).astype(np.int64)
        gts = _np(gt.get('bboxes', np.zeros((0, 5)))).reshape(-1, 5)
        gt_labels = _np(gt.get('labels', np.zeros(0))).astype(np.int64)
        gt_per_class += np.bincount(gt_labels, minlength=len(CLASSES))
        empty[i] = len(gts) == 0
        c = classify_detections(
            boxes,
            scores,
            labels,
            gts,
            gt_labels,
            _np(ign.get('bboxes', np.zeros((0, 5)))).reshape(-1, 5),
            _np(ign.get('labels', np.zeros(0))).astype(np.int64),
            device,
        )
        keep = c != IGNORED
        score.append(scores[keep])
        code.append(c[keep])
        label.append(labels[keep])
        size.append(
            np.digitize(np.sqrt(np.clip(boxes[keep, 2] * boxes[keep, 3], 0, None)), SIZE_EDGES)
        )
        img.append(np.full(int(keep.sum()), i, dtype=np.int32))
    cat = np.concatenate
    order = np.argsort(-cat(score), kind='stable')
    return dict(
        score=cat(score)[order],
        code=cat(code)[order],
        label=cat(label)[order],
        size=cat(size)[order],
        img=cat(img)[order],
        gt_per_class=gt_per_class,
        empty=empty,
    )


def curve(run: dict) -> dict[str, np.ndarray]:
    """Recall / FPPI / precision after each detection (score-descending)."""
    tp = np.cumsum(run['code'] == TP)
    fp = np.cumsum(run['code'] != TP)
    n_gt = max(int(run['gt_per_class'].sum()), 1)
    return dict(
        recall=tp / n_gt,
        fppi=fp / len(run['empty']),
        precision=tp / np.maximum(tp + fp, 1),
    )


def operating_point(run: dict, thr: float) -> dict:
    """Counts and FP breakdown for detections with ``score >= thr``."""
    keep = run['score'] >= thr
    code, label, size, img = (run[k][keep] for k in ('code', 'label', 'size', 'img'))
    is_fp = code != TP
    n_img, n_gt = len(run['empty']), int(run['gt_per_class'].sum())
    tp, fp = int((~is_fp).sum()), int(is_fp.sum())
    fp_img = np.bincount(img[is_fp], minlength=n_img)
    empty = run['empty']
    fp_cls = np.bincount(label[is_fp], minlength=len(CLASSES))
    tp_cls = np.bincount(label[~is_fp], minlength=len(CLASSES))
    return dict(
        thr=float(thr),
        tp=tp,
        fp=fp,
        fn=n_gt - tp,
        precision=tp / max(tp + fp, 1),
        recall=tp / max(n_gt, 1),
        fppi=fp / n_img,
        fppi_empty=float(fp_img[empty].mean()) if empty.any() else 0.0,
        img_with_fp=float((fp_img > 0).mean()),
        fp_types={t: int((code == i + 1).sum()) for i, t in enumerate(FP_TYPES)},
        fp_per_class={c: int(fp_cls[i]) for i, c in enumerate(CLASSES)},
        precision_per_class={
            c: float(tp_cls[i] / max(tp_cls[i] + fp_cls[i], 1)) for i, c in enumerate(CLASSES)
        },
        fp_per_size={s: int((size[is_fp] == i).sum()) for i, s in enumerate(SIZE_NAMES)},
    )


def summarise(run: dict) -> dict:
    cv = curve(run)
    f1 = 2 * cv['precision'] * cv['recall'] / np.maximum(cv['precision'] + cv['recall'], 1e-12)
    best = int(f1.argmax()) if len(f1) else 0
    thr_f1 = float(run['score'][best]) if len(f1) else 1.0

    def recall_at(fppi: float) -> float | None:
        ok = cv['fppi'] <= fppi
        return float(cv['recall'][ok].max()) if ok.any() else None

    def fppi_at(recall: float) -> float | None:
        ok = cv['recall'] >= recall
        return float(cv['fppi'][ok].min()) if ok.any() else None

    # Thin the curve for the json / plot.
    pick = np.unique(np.linspace(0, max(len(cv['fppi']) - 1, 0), 400).astype(int))
    return dict(
        num_images=len(run['empty']),
        num_empty_images=int(run['empty'].sum()),
        num_gts=int(run['gt_per_class'].sum()),
        gt_per_class={c: int(run['gt_per_class'][i]) for i, c in enumerate(CLASSES)},
        max_recall=float(cv['recall'][-1]) if len(f1) else 0.0,
        max_f1=float(f1[best]) if len(f1) else 0.0,
        recall_at_fppi={str(p): recall_at(p) for p in FPPI_POINTS},
        fppi_at_recall={str(r): fppi_at(r) for r in RECALL_POINTS},
        at_max_f1=operating_point(run, thr_f1),
        at_fixed=operating_point(run, FIXED_THR),
        curve=dict(fppi=cv['fppi'][pick].tolist(), recall=cv['recall'][pick].tolist()),
    )


def short_name(run: str) -> str:
    """``vits-dinov3-comer_rtmdet-p2-nwd-rka-6e-soda`` -> ``comer nwd-rka``."""
    name = run.replace('-6e-soda', '').replace('vits-dinov3-', '').replace('-dinov3', '')
    backbone, _, rest = name.partition('_rtmdet-p2')
    return f'{backbone} {rest.lstrip("-")}'.strip()


def _f(v: float | None, scale: float = 1.0, nd: int = 1) -> str:
    return '-' if v is None else f'{v * scale:.{nd}f}'


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    lines += ['| ' + ' | '.join(row) + ' |' for row in rows]
    return '\n'.join(lines)


def build_markdown(results: dict[str, dict]) -> str:
    any_r = next(iter(results.values()))
    out = [
        '# False alarms on the SODA-A test split\n',
        f'{any_r["num_images"]} patches ({any_r["num_empty_images"]} without GT), '
        f'{any_r["num_gts"]} GTs. TP = IoU >= 0.5 (SODAMetric matching), FPPI = false '
        'positives per 800x800 patch. Detections are the model outputs (score >= 0.05, '
        'max 1000 per patch).\n',
        '## Threshold-free: recall at a false-alarm budget, and FPPI needed for a recall\n',
    ]
    out.append(
        _table(
            [
                'run',
                'max F1',
                *(f'R@FPPI {p:g}' for p in FPPI_POINTS),
                *(f'FPPI@R {r:g}' for r in RECALL_POINTS),
                'max recall',
            ],
            [
                [
                    short_name(n),
                    _f(r['max_f1'], 100),
                    *(_f(r['recall_at_fppi'][str(p)], 100) for p in FPPI_POINTS),
                    *(_f(r['fppi_at_recall'][str(q)], 1, 2) for q in RECALL_POINTS),
                    _f(r['max_recall'], 100),
                ]
                for n, r in results.items()
            ],
        )
    )
    for key, title in (
        ('at_max_f1', "At each run's max-F1 score threshold"),
        ('at_fixed', f'At a fixed score threshold of {FIXED_THR}'),
    ):
        out.append(f'\n## {title}\n')
        rows = []
        for n, r in results.items():
            op = r[key]
            types = op['fp_types']
            rows.append(
                [
                    short_name(n),
                    f'{op["thr"]:.2f}',
                    _f(op['precision'], 100),
                    _f(op['recall'], 100),
                    str(op['fp']),
                    _f(op['fppi'], 1, 2),
                    _f(op['fppi_empty'], 1, 2),
                    _f(op['img_with_fp'], 100),
                    *(_f(types[t] / max(op['fp'], 1), 100) for t in FP_TYPES),
                ]
            )
        out.append(
            _table(
                [
                    'run',
                    'thr',
                    'P',
                    'R',
                    'FP',
                    'FPPI',
                    'FPPI empty',
                    '% patches w/ FP',
                    *(f'% {t}' for t in FP_TYPES),
                ],
                rows,
            )
        )
    out.append('\n## FP by predicted class (max-F1 threshold): FP count / precision %\n')
    out.append(
        _table(
            ['run', *CLASSES],
            [
                [
                    short_name(n),
                    *(
                        f'{r["at_max_f1"]["fp_per_class"][c]} / '
                        f'{_f(r["at_max_f1"]["precision_per_class"][c], 100)}'
                        for c in CLASSES
                    ),
                ]
                for n, r in results.items()
            ],
        )
    )
    out.append('\n## FP by predicted size, sqrt(w*h) px (max-F1 threshold): % of FPs\n')
    out.append(
        _table(
            ['run', *SIZE_NAMES],
            [
                [
                    short_name(n),
                    *(
                        _f(r['at_max_f1']['fp_per_size'][s] / max(r['at_max_f1']['fp'], 1), 100)
                        for s in SIZE_NAMES
                    ),
                ]
                for n, r in results.items()
            ],
        )
    )
    return '\n'.join(out) + '\n'


# Reference categorical palette (light), fixed order; colour follows the run.
PALETTE = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948')
BASELINES = ('comer', 'convnext-t')


def plot(results: dict[str, dict], path: str) -> None:
    """Recall vs FPPI; backbone runs and assigner / head runs in two panels."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = {n: short_name(n) for n in results}
    variant = [n for n, s in names.items() if any(k in s for k in ('nwd', 'dcfl'))]
    panels = [
        ('Backbone / adapter', [n for n in results if n not in variant]),
        (
            'Assigner / head (on CoMer and ConvNeXt-T)',
            [n for n, s in names.items() if s in BASELINES] + variant,
        ),
    ]
    panels = [(t, ns) for t, ns in panels if ns]
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 4.8), squeeze=False)
    for ax, (title, ns) in zip(axes[0], panels, strict=True):
        # Baselines keep slots 1 / 2 in both panels; other runs fill the rest in order.
        rest = iter(PALETTE[len(BASELINES) :])
        for n in ns:
            s = names[n]
            color = PALETTE[BASELINES.index(s)] if s in BASELINES else next(rest, '#8a8984')
            cv = results[n]['curve']
            ax.plot(cv['fppi'], np.asarray(cv['recall']) * 100, lw=2, color=color, label=s)
        ax.set_xscale('log')
        ax.set_xlim(0.1, None)
        ax.set_xlabel('false positives per patch (FPPI, log)', color='#52514e')
        ax.set_ylabel('recall @ IoU 0.5 (%)', color='#52514e')
        ax.set_title(title, color='#0b0b0b', loc='left', fontsize=11)
        ax.grid(True, which='major', color='#e4e3df', lw=0.8)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color('#b5b4ae')
        ax.tick_params(colors='#52514e')
        ax.legend(frameon=False, fontsize=9, loc='lower right', labelcolor='#0b0b0b')
    fig.set_facecolor('#fcfcfb')
    for ax in axes[0]:
        ax.set_facecolor('#fcfcfb')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='SODA-A false-alarm analysis')
    parser.add_argument('work_root')
    parser.add_argument('--device', default=None, help='rotated IoU device (default: cuda if any)')
    args = parser.parse_args()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}
    for pkl in sorted(glob.glob(osp.join(args.work_root, '*', 'preds_test.pkl'))):
        name = osp.basename(osp.dirname(pkl))
        print(f'[false alarms] {name}', flush=True)
        results[name] = summarise(collect_run(pkl, device))
    if not results:
        raise SystemExit(f'no */preds_test.pkl under {args.work_root}')
    markdown = build_markdown(results)
    with open(osp.join(args.work_root, 'false_alarms.md'), 'w') as f:
        f.write(markdown)
    with open(osp.join(args.work_root, 'false_alarms.json'), 'w') as f:
        json.dump(results, f, indent=2)
    plot(results, osp.join(args.work_root, 'fppi_recall.png'))
    print(markdown)


if __name__ == '__main__':
    main()
