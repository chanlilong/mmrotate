# Copyright (c) OpenMMLab. All rights reserved.
"""Generate a small synthetic dataset in split-DOTA format for smoke tests.

Images mimic panchromatic EO tiles (single band replicated to 3 channels)
with rotated rectangles whose size, aspect and brightness depend on the
class, so a detector can actually learn them. Layout::

    <out>/trainval/images/*.png, <out>/trainval/annfiles/*.txt
    <out>/test/images/*.png

Annotation lines: ``x1 y1 x2 y2 x3 y3 x4 y4 class difficulty``.
"""

import argparse
import os
import os.path as osp

import cv2
import numpy as np

DOTA_CLASSES = (
    'plane',
    'baseball-diamond',
    'bridge',
    'ground-track-field',
    'small-vehicle',
    'large-vehicle',
    'ship',
    'tennis-court',
    'basketball-court',
    'storage-tank',
    'soccer-ball-field',
    'roundabout',
    'harbor',
    'swimming-pool',
    'helicopter',
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', required=True, help='output directory')
    parser.add_argument('--num-train', type=int, default=32)
    parser.add_argument('--num-test', type=int, default=4)
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--max-objects', type=int, default=40)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def class_style(label: int) -> tuple[float, float, int]:
    """Deterministic (long side px, aspect ratio, brightness) per class."""
    long_side = 10.0 + 6.0 * (label % 8)
    aspect = 1.0 + 0.5 * (label % 4)
    brightness = 40 + int(200 * label / (len(DOTA_CLASSES) - 1))
    return long_side, aspect, brightness


def draw_image(
    rng: np.random.Generator, size: int, max_objects: int
) -> tuple[np.ndarray, list[str]]:
    img = rng.normal(110, 18, (size, size)).astype(np.float32)
    img = cv2.GaussianBlur(img, (0, 0), 3)
    lines = []
    for _ in range(int(rng.integers(max_objects // 4, max_objects + 1))):
        label = int(rng.integers(len(DOTA_CLASSES)))
        long_side, aspect, brightness = class_style(label)
        w = long_side * rng.uniform(0.85, 1.15)
        h = w / aspect
        cx, cy = rng.uniform(w, size - w, 2)
        angle = float(rng.uniform(0, 180))
        pts = cv2.boxPoints(((cx, cy), (w, h), angle))
        cv2.fillPoly(img, [np.round(pts).astype(np.int32)], float(brightness))
        coords = ' '.join(f'{v:.1f}' for v in pts.reshape(-1))
        lines.append(f'{coords} {DOTA_CLASSES[label]} 0')
    img += rng.normal(0, 4, img.shape).astype(np.float32)
    gray = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), lines


def write_split(
    rng: np.random.Generator, out: str, split: str, num: int, size: int, max_objects: int
) -> None:
    img_dir = osp.join(out, split, 'images')
    ann_dir = osp.join(out, split, 'annfiles')
    os.makedirs(img_dir, exist_ok=True)
    if split == 'trainval':
        os.makedirs(ann_dir, exist_ok=True)
    for i in range(num):
        name = f'P{i:04d}__1024__0___0'
        img, lines = draw_image(rng, size, max_objects)
        cv2.imwrite(osp.join(img_dir, f'{name}.png'), img)
        if split == 'trainval':
            with open(osp.join(ann_dir, f'{name}.txt'), 'w') as f:
                f.write('\n'.join(lines) + '\n')


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    write_split(rng, args.out, 'trainval', args.num_train, args.size, args.max_objects)
    write_split(rng, args.out, 'test', args.num_test, args.size, args.max_objects)
    print(f'Wrote proxy DOTA data to {args.out}')


if __name__ == '__main__':
    main()
