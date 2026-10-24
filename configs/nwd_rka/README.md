# NWD-RKA

> [Detecting tiny objects in aerial images: A normalized Wasserstein distance and a new benchmark](https://doi.org/10.1016/j.isprsjprs.2022.06.002) (ISPRS J. 2022), reference code [Chasel-Tsui/mmdet-aitod](https://github.com/Chasel-Tsui/mmdet-aitod)

## Abstract

IoU is very sensitive to small location offsets of tiny objects, which makes IoU-based label assignment drop most of their
positives. NWD-RKA models boxes as 2D Gaussians, measures similarity with the Normalized Wasserstein Distance
`NWD = exp(-W2 / C)`, and assigns positives by ranking (RKA): each GT takes its top-k priors by NWD regardless of the
absolute value; priors whose best NWD is below 0.3 are negatives and the rest are ignored.

## Implementation

Everything lives in `mmrotate/models/task_modules/nwd_rka/` and is registered by name, so any config can use it:

| Registry name | What it is |
|---|---|
| `RBboxNWD2D` | IoU-calculator-compatible NWD for rotated **and** horizontal boxes (Gaussian W2), drop-in for any assigner |
| `RankingAssigner` | RKA for anchor priors (RetinaNet, RPN) and, with `point_prior_scale`, point priors (RTMDet) |
| `RankingDynamicSoftLabelAssigner` | RTMDet's DSL with each GT's NWD top-k priors added as candidates |
| `nms=dict(type='nwd_nms', ...)` | NWD-NMS (from the earlier NWD paper / `mmcv-nwd`), Triton kernel on CUDA, PyTorch fallback, no custom mmcv |

`C` should be close to the training set's mean `sqrt(w * h)`. Estimate it with
`python tools/analysis_tools/nwd_constant.py <config>`; the configs use the paper's AI-TOD value 12.8.

## Configs

| Detector | Config |
|---|---|
| Rotated RTMDet-l, pure RKA | [rotated-rtmdet-l_nwd-rka-3x-dota.py](./rotated-rtmdet-l_nwd-rka-3x-dota.py) |
| Rotated RTMDet-l, DSL + RKA | [rotated-rtmdet-l_nwd-dsl-rka-3x-dota.py](./rotated-rtmdet-l_nwd-dsl-rka-3x-dota.py) |
| Rotated RetinaNet | [rotated-retinanet-rbox-le90_r50_fpn_nwd-rka-1x_dota.py](./rotated-retinanet-rbox-le90_r50_fpn_nwd-rka-1x_dota.py) |
| Oriented R-CNN (RPN RKA + NWD-NMS) | [oriented-rcnn-le90_r50_fpn_nwd-rka-1x_dota.py](./oriented-rcnn-le90_r50_fpn_nwd-rka-1x_dota.py) |
| RTMDet-tiny smoke test | [rotated-rtmdet-tiny_nwd-rka-proxy.py](./rotated-rtmdet-tiny_nwd-rka-proxy.py) |

C-RADIOv4 variants are in `configs/cradio/*nwd*`. No benchmark results yet; `topk`, `point_prior_scale` and the
constant are untuned starting points.
