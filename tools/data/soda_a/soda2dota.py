# Copyright (c) OpenMMLab. All rights reserved.
"""Convert SODA-A (per-image JSON) to DOTA-format txt annotations.

SODA-A layout::

    soda-a/Images/*.jpg
    soda-a/Annotations/{train,val,test}/*.json

Output layout (one folder per split, readable by ``img_split.py`` and
``DOTADataset``)::

    <out>/<split>/images/*.jpg      symlink, or a re-encoded copy if painted
    <out>/<split>/labelTxt/*.txt    ``x1 y1 x2 y2 x3 y3 x4 y4 <class> 0``

Objects annotated with a polygon of more than 4 points are converted to their
minimum-area rotated rectangle. ``ignore`` regions (category 9, ``area: null``)
are painted with ``--fill`` in the image, so they are neither trained on as
negatives nor counted as false positives at evaluation time, and no new
dataset class is needed.

Example::

    python tools/data/soda_a/soda2dota.py /mnt/nas/datasets/soda-a \
        /mnt/4TB/Datasets/soda-a/dota --nproc 16
"""

import argparse
import json
import os
import os.path as osp
from functools import partial
from multiprocessing import Pool

import cv2
import numpy as np

SODA_A_CLASSES = (
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
IGNORE_ID = 9
SPLITS = ('train', 'val', 'test')


def poly_to_quad(poly: list[float]) -> np.ndarray | None:
    """Return a ``(4, 2)`` quadrilateral, or ``None`` for degenerate input."""
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 3:
        return None
    if len(pts) == 4:
        return pts
    return cv2.boxPoints(cv2.minAreaRect(pts)).astype(np.float32)


def convert_one(
    json_path: str, img_root: str, out_img_dir: str, out_ann_dir: str, fill: tuple[int, ...]
) -> tuple[int, int, bool]:
    """Convert one SODA-A JSON. Returns ``(num_objects, num_ignore, painted)``."""
    with open(json_path) as f:
        data = json.load(f)
    file_name = data['images']['file_name']
    stem = osp.splitext(file_name)[0]
    names = {c['id']: c['name'] for c in data['categories']}

    lines, ignore_polys = [], []
    for ann in data['annotations']:
        cat = ann['category_id']
        if cat == IGNORE_ID or ann.get('area') is None:
            pts = np.asarray(ann['poly'], dtype=np.float32).reshape(-1, 2)
            if len(pts) >= 3:
                ignore_polys.append(np.round(pts).astype(np.int32))
            continue
        quad = poly_to_quad(ann['poly'])
        if quad is None:
            continue
        coords = ' '.join(f'{v:.2f}' for v in quad.reshape(-1))
        lines.append(f'{coords} {names[cat]} 0')

    with open(osp.join(out_ann_dir, stem + '.txt'), 'w') as f:
        f.write('\n'.join(lines) + ('\n' if lines else ''))

    src = osp.join(img_root, file_name)
    dst = osp.join(out_img_dir, file_name)
    if osp.lexists(dst):
        os.remove(dst)
    if ignore_polys:
        img = cv2.imread(src, cv2.IMREAD_COLOR)
        cv2.fillPoly(img, ignore_polys, color=tuple(int(v) for v in fill))
        cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        os.symlink(osp.abspath(src), dst)
    return len(lines), len(ignore_polys), bool(ignore_polys)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Convert SODA-A to DOTA format')
    parser.add_argument('src', help='SODA-A root with Images/ and Annotations/')
    parser.add_argument('out', help='output root, one sub-folder per split')
    parser.add_argument('--splits', nargs='+', default=list(SPLITS))
    parser.add_argument(
        '--fill',
        nargs=3,
        type=int,
        default=[104, 116, 124],
        help='BGR colour for ignore regions (matches the split padding)',
    )
    parser.add_argument('--nproc', type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    img_root = osp.join(args.src, 'Images')
    for split in args.splits:
        ann_dir = osp.join(args.src, 'Annotations', split)
        out_img_dir = osp.join(args.out, split, 'images')
        out_ann_dir = osp.join(args.out, split, 'labelTxt')
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_ann_dir, exist_ok=True)
        json_paths = sorted(
            osp.join(ann_dir, f) for f in os.listdir(ann_dir) if f.endswith('.json')
        )
        worker = partial(
            convert_one,
            img_root=img_root,
            out_img_dir=out_img_dir,
            out_ann_dir=out_ann_dir,
            fill=tuple(args.fill),
        )
        with Pool(args.nproc) as pool:
            stats = pool.map(worker, json_paths, chunksize=4)
        n_obj = sum(s[0] for s in stats)
        n_ign = sum(s[1] for s in stats)
        n_painted = sum(s[2] for s in stats)
        print(
            f'[{split}] images={len(stats)} objects={n_obj} '
            f'ignore_regions={n_ign} painted_images={n_painted}'
        )


if __name__ == '__main__':
    main()
