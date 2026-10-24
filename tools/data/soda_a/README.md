# Preparing SODA-A

[SODA-A](https://shaunyuan22.github.io/SODA/) (Cheng et al., *Towards Large-Scale Small Object Detection: Survey
and Benchmarks*, TPAMI 2023) ships as `Images/*.jpg` plus one JSON per image under
`Annotations/{train,val,test}/`. There are 2513 images of up to 4800×4800 pixels and about 870k oriented objects
in 9 classes, plus `ignore` regions. These tools convert it to the split DOTA format, so the stock `DOTADataset`
reads it without any new dataloader code.

```bash
# 1. JSON -> DOTA txt (x1 y1 ... x4 y4 class difficulty), ignore regions painted out
python tools/data/soda_a/soda2dota.py /mnt/nas/datasets/soda-a /mnt/4TB/Datasets/soda-a/dota --nproc 16

# 2. 800x800 patches with stride 650 (gap 150), the SODA-A protocol
for s in train val test; do
  python tools/data/dota/split/img_split.py --base-json tools/data/soda_a/split_configs/$s.json
done

# 3. a small, deterministic val subset (every 8th scene) for learning curves
python tools/data/soda_a/make_val_mini.py \
    /mnt/4TB/Datasets/soda-a/split_800/val /mnt/4TB/Datasets/soda-a/split_800/val_mini
```

The paths in the split configs are absolute; edit them to match your setup.

How the conversion works:

- **Ignore regions** (category 9, `area: null`) are filled with the split padding colour BGR (104, 116, 124). They are
  never seen as background during training, and they can't produce false positives at evaluation. Images with no
  ignore regions are symlinked to the originals, and painted images are re-encoded as JPEG at quality 95.
- **Polygons with more than 4 points** (about 1.3k objects) are replaced by their minimum-area rectangle.
- **Truncation:** objects cut by a patch border, with IoF ≥ 0.7, get difficulty `2` from `img_split.py`. They are
  still trained on and evaluated, because `DOTADataset` uses `diff_thr=100`.

Resulting sizes:

| split | images | objects | painted images | patches | non-empty patches |
|---|---|---|---|---|---|
| train | 1067 | 344,228 | 557 | 37,980 | 20,720 |
| val | 576 | 231,439 | 314 | 20,549 | 11,462 |
| val_mini | 72 | - | - | 2,588 | 1479 |
| test | 870 | 296,402 | 402 | 25,012 | 14,665 |

Evaluate with `SODAMetric` (`mmrotate/evaluation/metrics/soda/`). It reports AP@[.5:.95], AP50, AP75 and SODA-A's
size bins on `sqrt(area)`: eS < 12, rS 12–20, gS 20–32, N 32–40.
