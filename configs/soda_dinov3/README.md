# Frozen DINOv3 backbones for small-object detection on SODA-A

ViT-S/16 with ViTDet vs ViT-CoMer adapters, against ConvNeXt-T, then NWD-RKA and NWD-DSL-RKA, with a rotated
RTMDet head at P2.

<!-- RESULTS -->

## Setup

### Questions
1. **H1:** a ViT-CoMer adapter gives better detection features than a ViTDet simple feature pyramid on the same
   frozen ViT.
2. **H2:** ViTDet is weak on small objects, because its stride-4/8 maps are only deconvolutions of the stride-16
   tokens.
3. **H3:** ViT-S + CoMer is on par with a CNN of similar size and the same pre-training (DINOv3 ConvNeXt-T, all four
   stages into the FPN).
4. **Phase 2:** once H1–H3 are settled, do the NWD-based label assigners (NWD-RKA, NWD-DSL-RKA) help the best ViT
   variant on tiny objects? The CNN is not run in this phase.

### Data
SODA-A, converted to split DOTA format by the tools in `tools/data/soda_a/` (see the README there):
- 800×800 patches with stride 650, the SODA-A protocol, and no resizing, so objects keep their native size;
- ignore regions painted out.

| split | role | patches |
|---|---|---|
| train | training, non-empty patches only | 20,720 |
| val_mini (every 8th val scene) | learning curves, every epoch (every 2 for ViTDet and ConvNeXt-T, which ran first) | 2,588 |
| test (fully annotated) | final numbers | 25,012 |

About 98% of the objects are under 32 px (√area). On the train patches √(wh) has a mean of 16.6 px, a median of
14.4 px and a 10th percentile of 8.0 px.

### Metric
`SODAMetric` (`mmrotate/evaluation/metrics/soda/`) reports:
- rotated AP@[.50:.95], AP50 and AP75;
- AP@[.50:.95] in SODA-A's size bins on √area: **eS** < 12, **rS** 12–20, **gS** 20–32, **N** 32–40 px.

Matching follows mmrotate's VOC-style rotated evaluator, with area-under-PR AP and no limit on detections per
image, applied per patch. The numbers are therefore comparable with each other but not bit-identical to the
official SODA-A COCO-style evaluation on merged full images.

### What is identical across runs
Everything except the feature extractor:
- **Backbone:** pre-trained weights frozen (`requires_grad=False`, eval mode).
- **Neck:** CSPNeXt-PAFPN taking **4 levels (P2–P5, strides 4/8/16/32)**, 128 channels, 1 CSP block per stage.
- **Head:** rotated RTMDet (`RotatedRTMDetSepBNHead`, 128 channels) with point priors at strides 4/8/16/32.
- **Loss and assigner:** QualityFocal loss + rotated IoU loss, with DynamicSoftLabelAssigner (topk 13). Phase 2
  changes only the assigner.
- **Optimiser:** AdamW, lr 4e-4, wd 0.05, batch 8, grad-clip 35. Warm-up over 500 iterations, then cosine to 2e-5.
  **6 epochs**, about 15.5k iterations.
- **Precision and weights:** bf16 autocast (`AmpOptimWrapper`, `dtype: bfloat16`). EMA (ExpMomentumEMA) weights are
  used for evaluation.
- **Augmentation:** random flips (horizontal, vertical, diagonal) only.
- **Seed:** 0. Seed 1 repeats of ViTDet and ConvNeXt-T measure run-to-run noise.

### Variants

| config | backbone (frozen) | trainable feature module |
|---|---|---|
| `vits-dinov3-vitdet_rtmdet-p2-6e-soda.yaml` | DINOv3 ViT-S/16 (21.6M) | ViTDet simple pyramid from the last block plus final norm: 2× deconv to stride 4, deconv to 8, identity at 16, max-pool to 32; 256 channels |
| `vits-dinov3-comer_rtmdet-p2-6e-soda.yaml` | DINOv3 ViT-S/16 (21.6M) | ViT-CoMer-S adapter: CNN spatial-prior branch plus CTI blocks after blocks 2/5/8/11, with 4 extra CTI; 384-channel outputs at strides 4/8/16/32 |
| `convnext-t-dinov3_rtmdet-p2-6e-soda.yaml` | DINOv3 ConvNeXt-T (27.8M) | none: all 4 stages (96/192/384/768 channels) go straight to the neck |
| `vits-dinov3-comer-lora_rtmdet-p2-6e-soda.yaml` | as CoMer, weights still frozen | as CoMer, plus LoRA (r = 8, α = 16) on every ViT block linear (`qkv`, `proj`, `fc1`, `fc2`, 0.59M params), so the ViT can adapt to the CNN → ViT injected features |
| `vits-dinov3-vitdet-lora_rtmdet-p2-6e-soda.yaml` | as ViTDet, weights still frozen | as ViTDet, plus the same LoRA: the control without injection (ViT blocks use gradient checkpointing) |
| `vits-dinov3-comer_rtmdet-p2-nwd-rka-6e-soda.yaml` | as CoMer | as CoMer, with the `RankingAssigner` (NWD top-9, NWD soft targets, C = 16.6) |
| `vits-dinov3-comer_rtmdet-p2-nwd-dsl-rka-6e-soda.yaml` | as CoMer | as CoMer, with the `RankingDynamicSoftLabelAssigner` (DSL using NWD plus 3 NWD-ranked extra candidates per GT, C = 16.6) |

C is the mean √(wh) of the SODA-A train boxes, from `tools/analysis_tools/nwd_constant.py`.

The backbones are implemented in `mmrotate/models/backbones/timm_dinov3/`:
- `TimmViTSimplePyramid`, `TimmViTCoMer` and `TimmFeatureBackbone` load timm weights from the local HF snapshot
  folders.
- `TimmViTCoMer` reuses the ViT-CoMer adapter of `CRADIOv4CoMer` unchanged. Only the plain ViT is swapped, for a timm
  DINOv3 Eva with RoPE.
- LoRA (`lora_cfg` on every ViT backbone, see `cradiov4/lora.py`) wraps the block linears without touching the
  pretrained weights: they keep their parameters and state-dict keys and stay frozen, and only `lora_A` / `lora_B` train.
  `lora_B` starts at zero, so training starts from exactly the frozen model. Set the rank, scaling, dropout, target layers
  and blocks in the config, e.g. `lora_cfg: {r: 16, alpha: 32, dropout: 0.05, target_modules: [qkv], block_indices: [8, 9, 10, 11]}`.

## Reproduce

```bash
# data (once): see tools/data/soda_a/README.md
tools/soda_dinov3/run_experiments.sh bench    # params / speed / memory -> work_dirs/soda_dinov3/bench.json
tools/soda_dinov3/run_experiments.sh phase1   # ViTDet, ConvNeXt-T, CoMer (seed 0)
tools/soda_dinov3/run_experiments.sh seeds    # seed-1 repeats
tools/soda_dinov3/run_experiments.sh phase2   # NWD-RKA, NWD-DSL-RKA on CoMer
tools/soda_dinov3/run_experiments.sh collect  # -> work_dirs/soda_dinov3/results.md
```

Each step runs in `mmrotate-dev:2026.10.24` and skips runs that already have test metrics. Tests:

```bash
pytest tests/test_models/test_backbones/test_timm_dinov3.py \
       tests/test_evaluation/test_metrics/test_soda_metric.py \
       tests/test_datasets/test_soda2dota.py \
       tests/test_models/test_detectors/test_soda_dinov3_configs.py
```
