# Copyright (c) OpenMMLab. All rights reserved.
"""Build a small, deterministic validation subset from split SODA-A patches.

Keeps every patch of every ``--every``-th source image (sorted by name), so
the subset covers whole scenes rather than random patches. Files are
symlinked::

    python tools/data/soda_a/make_val_mini.py \
        /mnt/4TB/Datasets/soda-a/split_800/val /mnt/4TB/Datasets/soda-a/split_800/val_mini
"""

import argparse
import os
import os.path as osp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Make a val subset of split patches')
    parser.add_argument('src', help='split folder with images/ and annfiles/')
    parser.add_argument('dst', help='output folder')
    parser.add_argument('--every', type=int, default=8, help='keep every n-th source image')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ann_names = sorted(os.listdir(osp.join(args.src, 'annfiles')))
    sources = sorted({name.split('__')[0] for name in ann_names})
    keep = set(sources[:: args.every])
    img_names = {osp.splitext(n)[0]: n for n in os.listdir(osp.join(args.src, 'images'))}
    for sub in ('images', 'annfiles'):
        os.makedirs(osp.join(args.dst, sub), exist_ok=True)
    count = 0
    for ann in ann_names:
        stem = osp.splitext(ann)[0]
        if stem.split('__')[0] not in keep:
            continue
        for sub, name in (('annfiles', ann), ('images', img_names[stem])):
            src = osp.abspath(osp.join(args.src, sub, name))
            dst = osp.join(args.dst, sub, name)
            if not osp.lexists(dst):
                os.symlink(src, dst)
        count += 1
    print(f'{len(keep)}/{len(sources)} source images, {count} patches -> {args.dst}')


if __name__ == '__main__':
    main()
