# DCFL

> [Dynamic Coarse-to-Fine Learning for Oriented Tiny Object Detection](https://arxiv.org/abs/2304.08876) (CVPR 2023), reference code [Chasel-Tsui/mmrotate-dcfl](https://github.com/Chasel-Tsui/mmrotate-dcfl)

## Abstract

Static priors and fixed-threshold assignment give tiny oriented objects few, poorly placed positives. DCFL makes both
sides dynamic:
- **Dynamic priors:** the head's regression branch ends in a DCNv2. The mean of its sampling offsets moves each prior
  before label assignment, so priors follow the features they sample.
- **Coarse-to-fine assignment:** each GT first takes its top-k priors by GJSD similarity between Gaussian-modelled
  boxes (coarse). It then keeps the top-q of those by prediction quality, `IoU(pred, gt) + score` (fine), filtered by a
  Gaussian-mixture constraint built from the GT and the mean of its positive priors.

## Implementation

The reference needs [MMCV-DCFL](https://github.com/Chasel-Tsui/MMCV-DCFL). That fork changes no CUDA kernels: it only
adds Python wrappers (`ModulatedDeformConvG` = `ModulatedDeformConv2dPack` returning its offsets and masks). Here the
stock `mmcv.ops.modulated_deform_conv2d` of official mmcv is used, with `torchvision.ops.deform_conv2d` as the fallback
(`dcn_backend='auto'|'mmcv'|'torchvision'`). Nothing needs compiling. The DCNs run in fp32 under bf16/fp16 autocast.

| Registry name | Where | What it is |
|---|---|---|
| `RotatedDCFLHead` | `mmrotate/models/dense_heads/dcfl/` | the paper's rotated RetinaNet head: last two tower convs are a dilated conv and a DCNv2, and the cls DCN reuses the reg DCN's offsets |
| `RotatedRTMDetSepBNDCFLHead` | `mmrotate/models/dense_heads/dcfl/` | the same dynamic priors on the rotated RTMDet SepBN head: a dilated conv and a DCNv2 (per-level BN + act) appended to both towers |
| `C2FAssigner` | `mmrotate/models/task_modules/dcfl/` | coarse-to-fine assigner, vectorised; works with anchors or, with `point_prior_scale`, point priors |
| `RBboxGJSD2D` | `mmrotate/models/task_modules/dcfl/` | GJSD similarity `1 / (1 + GJSD)`, closed form, as an IoU calculator |

The reference's fixed-offset `DeformConv2d` (constant offsets `(d - 1) * (i, j)`) is exactly a dilated conv, so it
is one. The reference scales the prior shift by `img_h / 256 * 2 ** level`. Here it is
`offset * stride * prior_offset_scale`, which equals the reference at its 1024 input with the default 0.5.

The class docstrings list the small deviations from the reference, all fixes of order- or padding-dependent behaviour:
- each GT ranks only its own candidates in the fine step;
- the constraint applies to positives only;
- assignment sees predictions decoded from the original priors.

## Configs

| Detector | Config |
|---|---|
| Rotated RetinaNet-R50, DOTA-v1.0, 3x (paper setting) | [rotated-retinanet-dcfl-le135_r50_fpn_3x_dota.yaml](./rotated-retinanet-dcfl-le135_r50_fpn_3x_dota.yaml) |
| Rotated RTMDet-l, DOTA-v1.0 multi-scale | [rotated_rtmdet_l-dcfl-3x-dota_ms.yaml](./rotated_rtmdet_l-dcfl-3x-dota_ms.yaml) |
| RTMDet-tiny smoke test on proxy data | [rotated_rtmdet_tiny-dcfl-proxy.yaml](./rotated_rtmdet_tiny-dcfl-proxy.yaml) |
| C-RADIOv4-H + ViT-CoMer, DOTA-v1.0 multi-scale | [../cradio/cradiov4-h-comer_rtmdet-dcfl-3x-dota_ms.yaml](../cradio/cradiov4-h-comer_rtmdet-dcfl-3x-dota_ms.yaml) |
| DINOv3 ViT-S + ViT-CoMer, SODA-A (P2) | [../soda_dinov3/vits-dinov3-comer_rtmdet-p2-dcfl-6e-soda.yaml](../soda_dinov3/vits-dinov3-comer_rtmdet-p2-dcfl-6e-soda.yaml) |

To add DCFL to another RTMDet config, swap the head type and the assigner:

```yaml
model:
  bbox_head: {type: RotatedRTMDetSepBNDCFLHead, dilation_rate: 3, prior_offset_scale: 0.5}
  train_cfg:
    assigner:
      _delete_: true
      type: C2FAssigner
      topk: 16
      topq: 12
      constraint: dgmm
      gauss_thr: 0.6
      point_prior_scale: 4.0          # point priors -> 4x-stride boxes for the GJSD ranking
      quality_calculator: {type: RBboxOverlaps2D}  # QFL soft target = rotated IoU of the prediction
```

`C2FAssigner` alone (without the DCFL head) also works with plain priors in any head that passes decoded `bboxes` and
`scores` to its assigner, e.g. the stock `RotatedRTMDetSepBNHead`.

## Reference results (from the paper's repo, mmrotate 0.x)

| Method | Backbone | DOTA-v2.0 AP50 | Angle | lr schd |
|---|---|---|---|---|
| RetinaNet-O | R50 | 46.68 | le135 | 1x |
| DCFL | R50 | 51.57 | le135 | 1x |
| DCFL | R101 | 52.54 | le135 | 1x |

These are not reproduced here yet. The RTMDet / ViT variants use the paper's DOTA hyper-parameters (`topk` 16, `topq` 12,
`gauss_thr` 0.6) untuned.

## Tests

```bash
pytest tests/test_models/test_task_modules/test_dcfl \
       tests/test_models/test_dense_heads/test_dcfl_heads.py \
       tests/test_models/test_detectors/test_dcfl_configs.py
```

## Citation

```latex
@InProceedings{Xu_2023_CVPR,
    author    = {Xu, Chang and Ding, Jian and Wang, Jinwang and Yang, Wen and Yu, Huai and Yu, Lei and Xia, Gui-Song},
    title     = {Dynamic Coarse-To-Fine Learning for Oriented Tiny Object Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {7318-7328}
}
```
