# Frozen DINOv3 backbones for small-object detection on SODA-A

ViT-S/16 with ViTDet vs ViT-CoMer adapters, against ConvNeXt-T, then NWD-RKA and NWD-DSL-RKA, with a rotated
RTMDet head at P2.

<!-- RESULTS -->

## Results

Test split, AP × 100, seed 0 unless noted, with the configs' test settings (`nms_pre` 2000, `max_per_img` 1000).
Those settings cut recall in crowded patches, DCFL's most of all. For the comparison with dense-safe test settings,
which changes the DCFL verdict, see [Re-test with dense-safe settings](#re-test-with-dense-safe-test-settings).
Seed-to-seed noise is about ±0.1 mAP (ViTDet, ConvNeXt-T). The full
tables (all classes, val_mini curves, cost) are in `work_dirs/soda_dinov3/results.md`. The false-alarm tables are
in `work_dirs/soda_dinov3/false_alarms.md` and `fppi_recall.png`.

### Accuracy

| run | mAP | AP50 | AP75 | eS | rS | gS | N | small-vehicle | large-vehicle | val_mini ep 6 |
|---|---|---|---|---|---|---|---|---|---|---|
| ViTDet (2 seeds) | 17.5 | 48.0 | 7.5 | 1.2 | 7.8 | 24.4 | 27.7 | 5.3 | 8.8 | 17.3 |
| ViTDet + LoRA | 33.7 | 73.0 | 25.4 | 7.9 | 24.8 | 44.5 | 39.2 | 29.2 | 27.6 | 34.7 |
| ConvNeXt-T (2 seeds) | 36.0 | 74.0 | 30.3 | 11.8 | 28.0 | 44.1 | 38.0 | 33.0 | 26.0 | 37.9 |
| ConvNeXt-T + DCFL | 37.2 | 74.3 | 32.1 | 12.1 | 29.0 | 46.0 | 41.4 | 28.6 | 32.8 | 40.5 |
| CoMer, no injector | 37.0 | 75.5 | 31.5 | 11.9 | 28.4 | 44.1 | 40.2 | 34.0 | 32.1 | 38.8 |
| **CoMer** | **38.4** | 77.3 | **33.3** | 13.4 | 29.7 | 47.7 | 41.7 | **36.4** | 31.7 | 39.0 |
| CoMer + LoRA | 38.2 | **77.7** | 32.8 | **13.8** | **30.6** | 47.7 | 39.8 | 35.8 | **33.8** | 38.3 |
| CoMer + NWD-RKA | 35.6 | 73.9 | 29.2 | 12.3 | 26.7 | 44.1 | 39.6 | 30.9 | 25.2 | 36.4 |
| **CoMer + NWD-DSL-RKA** | **38.4** | 77.6 | 32.9 | 13.3 | 29.9 | **47.9** | 40.5 | 36.3 | 33.2 | **39.6** |
| CoMer + DCFL | 37.0 | 75.3 | 30.9 | 12.6 | 28.7 | 45.6 | **41.9** | 29.3 | 33.1 | 38.5 |

- **H1 (CoMer > ViTDet): yes, by a wide margin.** +20.9 mAP, and eS goes from 1.2 to 13.4. LoRA closes most of
  ViTDet's gap (33.7) but not all of it.
- **H2 (ViTDet weak on small objects): yes.** eS and rS are 1.2 and 7.8, against 24.4 for gS. A frozen ViT's
  stride-16 tokens deconvolved to strides 4/8 don't carry small-object detail, unless LoRA lets the ViT adapt.
- **H3 (ViT-S + CoMer ≈ ConvNeXt-T): CoMer is better** by +2.4 mAP, and on every size bin. Without the injector
  (CNN → ViT) it is only +1.0, so most of the gain needs the interaction. It costs about 4× the ConvNeXt-T training
  time.
- **LoRA on CoMer** is neutral overall (−0.2 mAP). It helps eS, rS and large vehicles, and costs some N and
  small-vehicle AP.

### Phase 2: assigners on CoMer (NWD-RKA, NWD-DSL-RKA, DCFL)

| | NWD-RKA | NWD-DSL-RKA | DCFL |
|---|---|---|---|
| head | stock RTMDet SepBN, static point priors | same | + dilated 3×3 conv + DCNv2 per tower; the classification DCN reuses the regression offsets |
| priors for assignment | static points as 4×-stride boxes | static points | points shifted by the mean DCN offset ("dynamic priors") |
| coarse metric | NWD (C = 16.6) | rotated IoU, + 3 NWD-ranked extra candidates per GT | GJSD (scale-free Gaussian JS divergence) |
| positives | fixed top-9 per GT, prediction-agnostic | DSL cost (IoU + class + centre), dynamic k | top-16 coarse → top-12 by IoU(pred, GT) + score → DGMM spatial constraint |
| soft target / box weight | rotated IoU of the prediction | rotated IoU (DSL) | rotated IoU of the prediction |
| mAP vs CoMer (config test settings) | −2.8 | ±0.0 | −1.4 |
| mAP vs CoMer (dense-safe test settings) | −2.0 | −0.1 | −0.3 |

- **NWD-RKA loses 2.8 mAP**, most of it at AP75 (−4.1), on small vehicles (−5.5) and on large vehicles (−6.5).
  A fixed, prediction-agnostic top-9 picks priors that are close in NWD but poorly placed for elongated boxes.
- **NWD-DSL-RKA ties the baseline** on test (38.4) and is the best val_mini run (39.6). DSL's prediction-aware cost
  repairs what RKA breaks, and the NWD extras add slightly more large-vehicle AP (+1.5).
- **DCFL: a test-setting artefact first, then a small loss on CoMer and a large gain on ConvNeXt-T.**
  - With the config test settings it drops 7.1 AP on small vehicles (63% of the test objects). That drop is the
    `nms_pre` cap (see below).
  - With dense-safe settings, CoMer + DCFL is 38.2, against 38.5 for CoMer. It has the best AP50 (78.2) and eS
    (13.8), and +1.3 on large vehicles, but the worst AP75 of the three (31.7 vs 33.3).
  - ConvNeXt-T + DCFL gains +2.1 over its own baseline (38.5 vs 36.4) and ties CoMer at a fraction of the cost.
- **DCFL needs the torchvision DCN.** With the mmcv `modulated_deform_conv2d` (mmcv 2.2.0, torch 2.14, Blackwell)
  the head never trained: val mAP was 0 after 2 epochs, with a flat box loss. Single forward and backward passes of
  the two backends agree, so the unit tests don't catch it. `dcfl/layers.py` now defaults to torchvision.

### False alarms

FP = a detection not matched at IoU 0.5 (same greedy matching as `SODAMetric`). FPPI = false positives per 800×800
patch, over all 30,974 test patches, 16,310 of them without objects.

| run | max recall | recall at 1 FPPI | FPPI at 60% recall | FPPI at 70% recall | FPPI at 80% recall | P / R at max F1 | % bkg FPs |
|---|---|---|---|---|---|---|---|
| ConvNeXt-T | 83.1 | 57.1 | 1.24 | 2.76 | 12.0 | 80.6 / 70.3 | 72 |
| ConvNeXt-T + DCFL | 74.2 | 55.0 | 1.64 | 5.89 | – | 80.7 / 64.2 | 76 |
| CoMer | 86.2 | **66.2** | **0.59** | 1.40 | 4.54 | **84.8** / 74.7 | 76 |
| CoMer + LoRA | 86.5 | 66.1 | 0.62 | **1.38** | **4.05** | 84.4 / **75.6** | 76 |
| CoMer + NWD-DSL-RKA | **86.6** | 65.7 | 0.64 | 1.41 | 4.24 | 84.2 / 75.4 | 78 |
| CoMer + NWD-RKA | 78.8 | 60.5 | 0.95 | 2.88 | – | 83.1 / 68.2 | 70 |
| CoMer + DCFL | 75.3 | 58.8 | 1.13 | 4.06 | – | 83.3 / 65.9 | 78 |
| ViTDet + LoRA | 79.4 | 59.6 | 1.03 | 2.67 | – | 83.1 / 68.9 | 63 |

- **CoMer, CoMer + LoRA and NWD-DSL-RKA are the cleanest.** At 70% recall they make about 1.4 false positives per
  patch, about half ConvNeXt-T's.
- **Most false alarms are background:** 70–78% are detections with IoU < 0.1 to every object. Localisation errors
  (same class, IoU 0.1–0.5) are 11–16%, and class confusion 7–15% (worst for ConvNeXt-T and NWD-RKA). There are no
  duplicates, because NMS at IoU 0.1 removes them.
- **ViTDet's false alarms are mostly localisation errors** (63%, or 26% with LoRA). Its boxes are found but badly
  placed.
- **NWD-RKA:** large-vehicle precision drops to 37.5% (7,042 FP), against 55–64% for the other CoMer runs.
- **DCFL looks good at its own threshold and bad at equal recall.** It makes the fewest false positives per patch
  at max F1 (2.20), but only because it outputs fewer boxes. Its max recall is 75%, and at 70% recall it needs 4.1
  false positives per patch.
- **Caveat: this table uses the config test settings (`nms_pre` 2000).** That cap is what limits DCFL's (and
  partly NWD-RKA's) max recall; see below. The DCFL and NWD-RKA rows therefore overstate false alarms at high recall.
  The false-alarm report has not yet been re-run with the dense-safe settings.

### Why DCFL misses objects: crowded patches

Recall over all detections (score ≥ 0.05), by number of objects in the test patch:

| objects / patch | patches | CoMer | NWD-DSL-RKA | CoMer + DCFL | ConvNeXt-T + DCFL | DCFL detections / patch |
|---|---|---|---|---|---|---|
| 1–49 | 12,422 | 89.2 | 89.6 | 89.5 | 90.3 | 51 |
| 50–199 | 1,670 | 87.4 | 87.6 | 86.3 | 86.4 | 207 |
| 200–499 | 455 | 85.9 | 86.3 | 71.8 | 71.6 | 304 |
| ≥ 500 | 117 | 80.9 | 81.8 | **44.6** | **37.6** | 460 (CoMer: 781) |

Up to about 200 objects per patch, DCFL matches the baselines. In car parks it outputs about 460 boxes for about
820 objects. Since small vehicles are 63% of the objects, this one failure explains DCFL's small-vehicle AP and its
recall ceiling.

**Cause: the test-time `nms_pre: 2000` cap, not the model.** The RTMDet head keeps the top `nms_pre` (prior,
class) scores **per level** before NMS. In a dense patch nearly every car is on P2, so with 2,000 candidates at P2
the number of objects that survive is 2,000 divided by the high-scoring priors per object. DCFL spends more of the
budget per object than the DSL models do, so it runs out first:

| ConvNeXt-T, test split | mAP | small-vehicle | recall, 200–499 objects/patch | recall, ≥ 500 | detections/patch, ≥ 500 |
|---|---|---|---|---|---|
| DCFL, `nms_pre` 2,000 | 37.2 | 28.6 | 71.6 | 37.6 | 429 |
| DCFL, `nms_pre` 30,000 | **38.4** | **36.3** | **88.8** | **79.7** | 977 (108 patches at the 1,000 cap) |
| baseline, `nms_pre` 2,000 | 36.1 | 32.8 | – | – | – |
| baseline, `nms_pre` 30,000 | 36.3 | 34.0 | 87.5 | 75.2 | 981 |

How the other suspects were ruled out:
- **NMS IoU 0.1 → 0.3:** +0.1 mAP for DCFL, small-vehicle AP unchanged. The missing cars never reach NMS.
- **"Last GT wins" candidate conflicts in `C2FAssigner`:** with the trained models on train patches, only 6–8% of
  coarse candidates are shared below 500 objects per patch, and no object loses all of them. C2F and DSL leave the
  same share of objects without positives (19%) only in the ≥ 500 bucket.
- **Train/test density shift:** real (191 train patches with ≥ 200 objects, against 572 in test), but it affects
  every model equally.

Why DCFL needs more candidates per object (likely; not measured at test time): C2F keeps up to 12 positives per
object, ranked by IoU + score. That trains a wider plateau of high scores than DSL's dynamic k: on train patches
C2F gives about 10 positives per object, DSL about 8. With the cap raised, DCFL's recall in dense patches is the
best of the ConvNeXt-T runs.

**A training-side weakness that remains: C2F starves the tiniest objects.** These are per-object positives of the
trained ConvNeXt-T models on 30 train patches per density bucket (11,212 objects), from
`assign_stats.py` (kept outside the repo):

| object √(wh) | objects | C2F (DCFL model): 0 positives | DSL on the same DCFL predictions | DSL (baseline model): 0 positives |
|---|---|---|---|---|
| < 8 px | 1,543 | **26.9%** (mean 2.5 positives) | 5.1% | 7.2% (mean 1.7) |
| 8–12 px | 2,891 | 9.5% | 8.1% | 8.1% |
| ≥ 12 px | 6,778 | ≈ 0% | ≈ 0% | ≈ 0% |

- **The DGMM constraint is the main culprit.** It rejects every candidate of 3.7% of all objects.
- **Crowding adds some.** In the 200–499 objects/patch bucket, 6.0% of objects get no positives under C2F, against
  1.1% under DSL on the same predictions.
- **Ruled out:** below 500 objects per patch, the "last GT wins" conflict rule takes all candidates from 0.6% of
  objects or fewer.
- **Why it matters downstream:** trucks at 0.7 m GSD have √(wh) of about 7–8 px, exactly the bin C2F starves (see
  the recommendation).

<!-- DCFL-RETEST -->

### Re-test with dense-safe test settings

The same checkpoints re-tested with `nms_pre` 30000 and `max_per_img` 3000 (AP × 100, change from the config
test settings in brackets):

| run | mAP | AP50 | AP75 | eS | rS | gS | N | small-vehicle | large-vehicle |
|---|---|---|---|---|---|---|---|---|---|
| ConvNeXt-T | 36.4 (+0.3) | 74.5 | 30.7 | 12.1 | 28.4 | 44.8 | 38.9 | 34.9 | 26.1 |
| **ConvNeXt-T + DCFL** | **38.5 (+1.3)** | 77.2 | 33.0 | 13.3 | **30.6** | 47.2 | 41.4 | 37.4 | **32.5** |
| **CoMer** | **38.5 (+0.1)** | 77.6 | **33.3** | 13.7 | 29.8 | 47.7 | 41.6 | **37.8** | 31.2 |
| **CoMer + NWD-DSL-RKA** | **38.5 (+0.1)** | 77.9 | 32.9 | 13.5 | 29.9 | **47.9** | 40.5 | 37.4 | **32.5** |
| CoMer + DCFL | 38.2 (+1.2) | **78.2** | 31.7 | **13.8** | 30.3 | 47.2 | **41.9** | 37.2 | **32.5** |
| CoMer + NWD-RKA | 36.5 (+0.9) | 76.0 | 29.8 | 13.1 | 27.8 | 44.9 | 39.6 | 36.6 | 24.9 |

- **The cap hurt the fixed-k assigners.** DCFL (+1.2 to +1.3) and NWD-RKA (+0.9) lose the most to it. The DSL
  models barely notice it (+0.1 to +0.3). NWD-RKA's small-vehicle deficit was mostly the cap (30.9 → 36.6). Its
  AP75 and large-vehicle losses are real.
- **On CoMer, DSL, NWD-DSL-RKA and DCFL are within 0.3 mAP**, which is close to the noise level. DCFL trades AP75
  (−1.6) for AP50 (+0.6) and large vehicles (+1.3).
- **On ConvNeXt-T, DCFL is a real gain** (+2.1 mAP, +6.4 large vehicle, +2.5 small vehicle). It brings the CNN level
  with CoMer at about 4× lower training cost and about 2.3× faster inference.
- **DCFL vs NWD-RKA (the original question): DCFL wins** by +1.7 mAP on CoMer, with +1.9 AP75 and +7.6
  large-vehicle AP.

### Predictions made before the DCFL runs, graded

| prediction | config test settings | dense-safe test settings |
|---|---|---|
| CoMer + DCFL ≈ 38.5–39.5, about tied with NWD-DSL-RKA | ❌ 37.0 | ⚠️ 38.2, 0.3 below NWD-DSL-RKA |
| DCFL beats NWD-RKA by about 3 mAP | ⚠️ +1.4 | ⚠️ +1.7 |
| ConvNeXt-T + DCFL +0.5 to +1.0 over ConvNeXt-T | ✅ +1.2 | ❌ underestimated: +2.1 (+2.4 over seed 0 at the config settings) |
| NWD-RKA has the most false alarms at matched recall | ❌ DCFL has more at 70% recall | not re-measured |
| DCFL adds background false alarms (DCN context) | ✅ slightly (78% vs 76%) | not re-measured |

Not predicted: the `nms_pre` interaction, and C2F starving objects under 8 px.

## Recommendation: C-RADIOv4, a larger dataset, and production vehicles at 0.5–0.7 m GSD

### What the target objects look like
Vehicles 9–13 m long (trucks, buses, articulated lorries) are about 2.5 m wide:

| GSD | length | width | √(wh) | aspect | SODA-A bin |
|---|---|---|---|---|---|
| 0.5 m | 18–26 px | 5 px | 9.5–11.4 px | 3.6–5.2 | eS (< 12) |
| 0.7 m | 13–19 px | 3.6 px | 6.8–8.2 px | 3.6–5.2 | eS |

SODA-A's own `large-vehicle` class is not the right proxy. Its median is 51 × 11.5 px and only 16% are in eS. The
targets are closer to SODA-A's `small-vehicle` (median 19.5 × 8.5 px, 38% eS), only half as wide. So the
**small-vehicle and eS columns** are the ones that transfer. With dense-safe test settings, the CoMer runs
(DSL, NWD-DSL-RKA, DCFL) lead those columns within 0.6 AP of each other, and ConvNeXt-T + DCFL matches them.
ViTDet without LoRA fails. At 0.7 m GSD, though, the trucks fall in the < 8 px bin, where C2F leaves 27% of
objects without positives. A 40 ft shipping container (12.2 × 2.4 m) has the
footprint of a 12 m truck; SODA-A's `container` class, the second-largest source of false positives, is the
look-alike.

### Recommended setup
Each item says whether it follows from SODA-A measurements here (**measured**) or is extrapolated (**extrapolated**).

1. **Backbone: C-RADIOv4-H + ViT-CoMer with the P2 level**, i.e. `configs/cradio/cradiov4-h-comer_rtmdet-p2-3x-dota_ms.yaml`.
   - *Measured:* CoMer beats a simple pyramid on the same frozen ViT by +20.9 mAP (eS 13.4 vs 1.2). It beats a CNN
     of the same size with the same head by +2.1 to +2.4 mAP, with half the false positives at 70% recall. A CNN
     with the DCFL head catches up with it (item 8).
   - *Extrapolated:* the trucks are at most 5 px wide, so **only the stride-4 level gives them more than one prior
     across the width**. Do not use the stride 8/16/32 C-RADIO configs.
   - *Extrapolated:* a larger ViT should help most with "what" errors (class confusion, background, container vs
     truck). The 4-px localisation comes from the CoMer CNN branch at stride 4, which C-RADIO doesn't change.
2. **Assigner: stock DSL as the default, NWD-DSL-RKA as the first ablation, DCFL as the second. Not NWD-RKA.**
   - *Measured:* NWD-DSL-RKA ties the baseline (38.5; best val_mini, +1.3 large-vehicle AP).
   - *Measured:* NWD-RKA costs 3.5 AP75 even with dense-safe settings, and halves large-vehicle precision. Its
     fixed top-9 suits elongated boxes badly.
   - *Measured:* on CoMer, DCFL ties (−0.3 mAP) but has the worst AP75 (−1.6). On thin boxes, localisation is what
     IoU-based acceptance punishes.
   - *Measured:* C2F leaves 27% of objects under 8 px without positives, against 7% for DSL. That bin is the 0.7 m
     trucks. At 0.5 m (9.5–11 px) the gap is small (9.5% vs 8.1%).
   - *Extrapolated:* if DCFL is tried at 0.7 m, relax the DGMM constraint first (`gauss_thr` above 0.6, or
     `constraint: null`) and check per-size positives with the assignment-stats script before training.
   - *Measured:* with a ConvNeXt-T backbone, DCFL is a clear win (+2.1 mAP). See item 8.
   - **Config fix needed.** `cradiov4-h-comer_rtmdet-nwd-dsl-rka-3x-dota_ms.yaml` still uses NWD for the DSL cost
     and soft labels (`iou_calculator: RBboxNWD2D`, C = 12.8). The SODA-A run that worked uses `RBboxOverlaps2D`
     there, with NWD only picking the extra candidates (commit 955f2e7: NWD soft targets rewarded negative-size
     boxes). C-RADIO has `exp_on_reg: true`, so negative sizes can't happen there, but the NWD-soft-label variant
     is untested. Align it with the SODA-A config, and recompute C on the target data
     (`tools/analysis_tools/nwd_constant.py`; about 7–11 for these trucks).
3. **Resolution: test 1.5× and 2× upsampling** (0.5–0.7 m → 0.25–0.47 m) against native resolution.
   - *Measured:* AP roughly doubles from eS to rS (13.4 → 29.7) and again to gS (47.7) for every model.
   - *Extrapolated:* upsampling adds no information, so the gain will be smaller than the bin-to-bin jump. But at
     3.6 px width a 1 px box error costs about 25% IoU, and 2× brings the trucks into the rS/gS bins at 4× the
     compute. Of the levers here this is likely the largest; measure it on the target data first.
4. **Training data for false alarms.**
   - *Measured:* 70–78% of false positives are background, 0.37–0.49 false positives per empty patch. Training
     here uses non-empty patches only (`filter_empty_gt=True`).
   - For wide-area production, add empty and hard-negative patches: shipping containers, rail wagons, building
     roofs and road markings at the target GSD. Label containers as their own class if possible; class confusion is
     already 7–15% of the false positives at SODA-A sizes.
5. **Test-time settings.**
   - *Measured:* **raise `nms_pre` and `max_per_img`.** `nms_pre` is a per-level cap, and in dense tiles nearly
     every object sits on P2. On SODA-A, `nms_pre` 30000 / `max_per_img` 3000 is worth +0.1 to +0.3 mAP for DSL
     models and +0.9 to +1.3 for NWD-RKA and DCFL (dense-patch recall from 38% to 80%).
   - *Extrapolated:* size them from the densest expected tile, at about 10× the max objects per tile for
     `nms_pre`. Apply this to every C-RADIO config; they all still use 2000. Post-processing time grows with the
     candidate count, so check latency.
   - NMS IoU 0.1 vs 0.3 made no difference on SODA-A (+0.1 mAP). Re-check on side-by-side parked trucks, which are
     3.6 px wide with gaps of about 1 px.
6. **Operating point and metric.**
   - Pick the threshold from the FPPI–recall curve on held-out target data (`tools/soda_dinov3/false_alarms.py`),
     not from mAP.
   - One 800 × 800 tile is 0.16 km² at 0.5 m and 0.31 km² at 0.7 m. 1.4 FP per tile (CoMer at 70% recall) is about
     9 or 4.5 false alarms per km². That only holds if the background is like SODA-A's.
   - Also report recall at IoU 0.3 or with centre-distance matching. At 4–5 px width, IoU 0.5 penalises boxes that
     are operationally correct.
7. **Larger dataset and schedule.**
   - *Measured:* runs were still improving at 6 epochs. LoRA on CoMer was neutral overall (eS +0.4, large vehicle
     +2.1, small vehicle −0.6).
   - *Extrapolated:* with more data and the 3× schedule, start frozen + CoMer and treat LoRA (r = 8 on the last
     blocks) as an ablation. It matters most when the target imagery differs from the pre-training data (sensor,
     off-nadir).
8. **Cost.**
   - At 800², ViT-S + CoMer runs at 59 img/s at test, against 134 for ConvNeXt-T (RTX 3090). C-RADIOv4-H is far
     larger.
   - *Measured:* if throughput rules C-RADIO out, **ConvNeXt-T + DCFL is the fallback**. With dense-safe settings it
     ties ViT-S + CoMer (38.5 mAP; small vehicle 37.4 vs 37.8; eS 13.3 vs 13.7) at about 4× lower training cost and
     about 2.3× faster inference. ConvNeXt-T + DSL is 2.1 mAP lower.
   - Its false-alarm curve with dense-safe settings has not been measured yet. The < 8 px starvation caveat from
     item 2 applies. Distilling C-RADIO + CoMer into it is the next step.

### What to revisit
- The crowded-scene "failure" of DCFL was the `nms_pre` cap. With it raised, DCFL is +2.1 mAP on ConvNeXt-T and
  +6.4 on large vehicles. What remains is C2F starving objects under 8 px. Fix that (relax DGMM, or C2F candidates
  plus DSL positives) and re-test on the target data at 0.7 m GSD.
- Re-run the false-alarm report with dense-safe test settings
  (`run_experiments.sh dump` with `CFG_OPTIONS="model.test_cfg.nms_pre=30000 model.test_cfg.max_per_img=3000"`,
  after moving the existing `preds_test.pkl` aside). The current tables understate DCFL's and NWD-RKA's recall.
- Seed noise is ±0.1 mAP here, but every result is a single 6-epoch run on one dataset. Confirm the top two
  choices (DSL vs NWD-DSL-RKA, native vs upsampled) on the target data before committing.

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
| `vits-dinov3-comer_rtmdet-p2-nwd-rka-6e-soda.yaml` | as CoMer | as CoMer, with the `RankingAssigner` (NWD top-9 ranking, rotated-IoU soft targets, C = 16.6) |
| `vits-dinov3-comer_rtmdet-p2-nwd-dsl-rka-6e-soda.yaml` | as CoMer | as CoMer, with the `RankingDynamicSoftLabelAssigner` (rotated-IoU DSL plus 3 NWD-ranked extra candidates per GT, C = 16.6) |
| `vits-dinov3-comer_rtmdet-p2-dcfl-6e-soda.yaml` | as CoMer | as CoMer, with DCFL: `RotatedRTMDetSepBNDCFLHead` (dilated conv + DCNv2 tower, DCN-shifted priors) and `C2FAssigner` (GJSD top-16, quality top-12, DGMM constraint; see `configs/dcfl/`) |
| `convnext-t-dinov3_rtmdet-p2-dcfl-6e-soda.yaml` | as ConvNeXt-T | as ConvNeXt-T, with the same DCFL head and assigner as the CoMer DCFL run: DCFL on the CNN reference |

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
tools/soda_dinov3/run_experiments.sh dcfl     # CoMer / ConvNeXt-T with DCFL
tools/soda_dinov3/run_experiments.sh collect  # -> work_dirs/soda_dinov3/results.md
# false alarms: test-split predictions (new runs dump them in their test step), then the FP report
tools/soda_dinov3/run_experiments.sh dump <config> [seed]   # -> <run>/preds_test.pkl
tools/soda_dinov3/run_experiments.sh falsealarms            # -> false_alarms.md / .json, fppi_recall.png
```

Each step runs in `mmrotate-dev:2026.10.24` and skips runs that already have test metrics. Tests:

```bash
pytest tests/test_models/test_backbones/test_timm_dinov3.py \
       tests/test_evaluation/test_metrics/test_soda_metric.py \
       tests/test_datasets/test_soda2dota.py \
       tests/test_models/test_detectors/test_soda_dinov3_configs.py
```
