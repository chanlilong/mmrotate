# CoMer-lite: how much of the ViT-S CoMer adapter does SODA-A need?

`mmrotate/tensorrt/COMER_LATENCY.md` shows ViT-S/16 DINOv3 + CoMer + RTMDet (P2-P5) costs 688 GFLOP per
800x800 image, against 221 for ConvNeXt-T. The adapter's interaction modules (`CTI_toV` x4, `CTI_toC` x4, extra
`Extractor_CTI` x4) and the 384-channel neck input account for most of the difference. This report cuts each
part on its own, retrains, and measures what each cut costs in mAP, recall and false alarms per image (FPPI).

**Short answer:** none of the adapter is dead weight on SODA-A. Every cut lost accuracy roughly in proportion
to the compute it removed: about 0.75 to 1.5 mAP per 10% of FLOPs saved. Only one cut is close to free on false
alarms, `deform_ratio: 0.5` (−7% FLOPs, FPPI unchanged, −0.5 mAP from looser boxes). The biggest planned saving,
narrower P2/P3 outputs into the neck, cost 2.9 mAP and erased most of CoMer's false-alarm advantage.

## 1. Setup

- Base config: `configs/soda_dinov3/vits-dinov3-comer_rtmdet-p2-6e-soda.yaml` (frozen ViT, 6 epochs, bf16 AMP,
  EMA, seed 0). Each variant in `configs/comer_lite/` changes only the backbone (and the neck input widths for
  `slim`); a test checks this.
- Test split: 30,974 patches, 517,534 objects. FPPI = false positives per 800x800 patch at IoU 0.5
  (`tools/soda_dinov3/false_alarms.py`).
- Seed noise is about ±0.1 mAP (earlier repeat runs). The FPPI noise wasn't measured; treat differences under
  ~0.05 FPPI at R0.7 as noise.
- All runs trained on node `may` (RTX PRO 6000 Blackwell Max-Q), mostly two at a time.

| run | config change | what it removes |
|---|---|---|
| baseline | — | — |
| `ratio05` | `deform_ratio: 0.5` | half the width of the deformable attention values (`value_proj`, `output_proj`) in all 12 attentions |
| `cfinter-late` | `cnn_feature_interaction: [F, F, T, T]` | CNN self-attention inside `CTI_toC` in stages 1-2 (2 of 8 CNN self-attention rounds) |
| `extra0` | `extra_num: 0` | the 4 extra CTIs after the last stage (4 of 8 CNN self-attention rounds) |
| `noinj` (earlier run) | no MRFP / `CTI_toV` | all 4 CNN→ViT injections |
| `slim` | `TimmViTCoMerLite`, `out_channels: [96, 192, 384, 384]` | P2/P3 width into the neck (ConvNeXt-T's widths) |
| `lite` | `TimmViTCoMerLite`: `deform_ratio: 0.5`, `extra_num: 1`, `dedup_norm: true` | the cheapest cuts together, 384-ch outputs kept |

`dedup_norm` removes the second LayerNorm that `CTI_toC` / `Extractor_CTI` apply to an input that
`MultiscaleExtractor` normalises again. It saves no FLOPs worth counting, only memory traffic.

## 2. Results

### Accuracy (test split, AP x 100)

| run | mAP | AP50 | AP75 | eS | small-veh | large-veh |
|---|---|---|---|---|---|---|
| baseline | **38.4** | 77.3 | 33.3 | 13.4 | 36.4 | 31.7 |
| `ratio05` | 37.9 (−0.5) | 77.0 | 32.4 | 13.2 | 36.6 | 31.9 |
| `cfinter-late` | 38.0 (−0.5) | 77.1 | 32.6 | 13.2 | 35.3 | 32.1 |
| `extra0` | 37.4 (−1.1) | 76.2 | 31.5 | 13.0 | 34.9 | 29.5 |
| `lite` | 37.1 (−1.3) | 76.0 | 31.7 | 12.7 | 35.5 | 28.7 |
| `noinj` | 37.0 (−1.4) | 75.5 | 31.5 | 11.9 | 34.0 | 32.1 |
| `slim` | 35.6 (−2.9) | 74.2 | 29.2 | 11.1 | 32.8 | 27.6 |
| ConvNeXt-T (reference) | 36.0 | | | | | |

### Recall and false alarms

| run | R@FPPI 1 | FPPI@R 0.6 | FPPI@R 0.7 | FPPI@R 0.8 | max recall |
|---|---|---|---|---|---|
| baseline | 66.2 | 0.59 | **1.40** | 4.54 | 86.2 |
| `ratio05` | 65.9 | 0.60 | **1.42** | 4.61 | 86.3 |
| `cfinter-late` | 65.1 | 0.67 | 1.51 | 4.66 | 86.1 |
| `lite` | 65.3 | 0.61 | 1.58 | 5.49 | 85.2 |
| `extra0` | 64.4 | 0.70 | 1.66 | 6.45 | 84.5 |
| `slim` | 61.4 | 0.90 | 2.09 | 8.42 | 83.2 |
| `noinj` | 60.5 | 0.97 | 2.22 | 8.39 | 83.9 |
| ConvNeXt-T (reference) | 57.1 | 1.24 | 2.76 | 12.00 | 83.1 |

### Cost

| run | GFLOP / img | Δ FLOPs | train img/s (from the run's log) |
|---|---|---|---|
| baseline | 688.5 | — | 14.2 |
| `ratio05` | 642.1 | −7% | 14.7 |
| `cfinter-late` | 664.8 | −3% | 15.6 |
| `extra0` | 633.2 | −8% | 17.9 |
| `lite` | ~612 | −11% | 18.1 |
| `noinj` | 625.2 | −9% | 19.0 |
| `slim` | 529.1 | −23% | 15.6 |

- GFLOP: `torch.utils.flop_counter` on the full detector at batch 1 (`tools/comer_lite/profile_modules.py`).
  `lite` was counted on CPU, which reads a fixed 115.6 GFLOP lower than the GPU count for every config (the
  differences between `baseline`, `ratio05` and `extra0` match the GPU numbers to 0.1 GFLOP), so it is
  reported on the GPU basis.
- Train img/s: batch 8 / median iteration time of epochs 2-6, including data loading, with another training on
  the neighbouring GPU. It's comparable between runs to about ±5%, not a clean benchmark (see section 6).

### Measured module breakdown (baseline, idle GPU, batch 8, 800x800, bf16, eval)

| spm | mrfp | cti_tov | cti_toc | extra CTIs | ViT blocks | other adapter glue | neck | head | total |
|---|---|---|---|---|---|---|---|---|---|
| 5.5 | 2.9 | 12.3 | 15.5 | **19.1** | 16.3 | 9.2 | 17.4 | 3.5 | 101.6 ms (79 img/s) |

The three interaction modules take 47 ms, almost half the forward pass and three times the frozen ViT. The 4
extra CTIs are the single most expensive module. The training step (forward + backward, adapter gradient
checkpointing on) took 429 ms for the same batch, with 19.3 GB of activations.

## 3. How much ViT↔CNN interaction is needed?

The adapter has two directions of interaction, and they cost very different amounts:

- **CNN→ViT (injection):** before each of the 4 ViT stages, MRFP + `CTI_toV` write CNN features into the ViT
  tokens. It is about 63 GFLOP (9%) and ~15 ms of the 102 ms forward. Its main cost is in training: the
  gradient has to flow back through the frozen ViT blocks to reach the earlier adapter layers.
- **ViT→CNN:** the ViT tokens are *added* to the stride-16 CNN tokens (`CTI_toC`) and to every output level
  (`add_vit_feature`). That is cheap. What is expensive is what follows each addition: a multi-scale **CNN
  self-attention** over all 13,125 CNN tokens. It runs 8 times (once in each of the 4 `CTI_toC`, once in
  each of the 4 extra CTIs) and costs about 35 ms of the 102 ms forward.

### CNN→ViT: needed, and the best value in the adapter

Removing it (`noinj`) costs 1.4 mAP, but the false-alarm numbers show the bigger effect: FPPI at 70% recall
goes from 1.40 to 2.22 (+59%), and at 80% recall from 4.5 to 8.4. That is most of the way to ConvNeXt-T
(2.76). The smallest objects suffer most (eS −1.5). **The injection is where CoMer's lower false-alarm rate
comes from.** It is also the cut that speeds up training most (+34% img/s), because without it no gradient
goes through the ViT.

Not measured: whether fewer than 4 injections would do (e.g. only before stages 1-2). That run was proposed
and not run.

### ViT→CNN: the CNN self-attention rounds are all doing work

Cutting the 8 CNN self-attention rounds lost accuracy steadily, with no sign of a plateau:

| CNN self-attention rounds | run | Δ mAP | FPPI@R0.7 | FPPI@R0.8 |
|---|---|---|---|---|
| 8 | baseline | — | 1.40 | 4.54 |
| 6 (drop stages 1-2) | `cfinter-late` | −0.5 | 1.51 | 4.66 |
| 5 (4 + 1 extra), also `deform_ratio 0.5` | `lite` | −1.3 (≈ −0.8 after removing `ratio05`'s −0.5) | 1.58 | 5.49 |
| 4 (no extra CTIs) | `extra0` | −1.1 | 1.66 | 6.45 |

That is about 0.25 mAP per round. The rounds after the last ViT stage (extra CTIs) matter most at high
recall: without them, max recall drops 1.7 points and FPPI at 80% recall rises 42%. Their job is to spread the
final ViT information through the CNN pyramid down to stride 8, which the 4 `CTI_toC` rounds can't do alone.
The extra CTIs are also the most expensive module (19 ms of 102), so this is a real trade, not a free cut.

### Is the P2 ViT-S CoMer too much?

The P2 level itself does no interaction. The stride-4 output is the CNN stem's stride-4 map plus an
upsampled stride-8 CNN map plus a 4x-upsampled ViT map. All attention happens at strides 8-32, and 76% of
the 13,125 attention queries are at stride 8. So "P2 CoMer" costs extra in two places:

1. the stride-8 CNN tokens in every attention (the bulk of the adapter's cost), and
2. the 384-channel P2 and P3 maps going into the neck (~150 GFLOP of the 688).

`slim` tested (2) by narrowing P2/P3 to ConvNeXt-T's 96/192 channels. It is the worst result of all:
−2.9 mAP, FPPI at 70% recall 2.09 (as bad as removing the injection), and large-vehicle precision down from
55% to 38%. The wide high-resolution features carry the ViT information to the neck, and narrowing them throws
most of it away. Whether 192/384 or 256/384 would be enough isn't measured. On this evidence, though, the P2
neck width is not excess.

(1) wasn't tested by dropping stride 8 from the attention. The `cfinter-late` and `extra0` results suggest it
would cost accuracy: the stride-8 CNN self-attention is most of what those runs removed.

**Verdict:** for SODA-A at 6 epochs, the P2 ViT-S CoMer isn't oversized. Every part we removed paid for its
compute, roughly in proportion to it. The only part that looks oversized is the *width* of the deformable
attention values (`deform_ratio`): halving it kept recall and FPPI and only loosened box tightness (AP75
−0.9).

Caveats: one seed per run, a 6-epoch schedule (a longer schedule could close some of the gaps for capacity
cuts like `slim` or `ratio05`), and SODA-A only (mostly small objects, where high-resolution features count
most).

## 4. Q2: ViT tokens as `CTI_toV` queries (analysis only, not trained)

The ViT-query experiment was dropped, so this is an estimate, not a measurement.

Today `CTI_toV` runs a multi-scale self-attention with all 13,125 CNN tokens as queries, then bilinearly pools
the stride-8 and stride-32 results back onto the 50x50 ViT grid and sums them. Most of the query-side work is
averaged away. With the 2,500 ViT tokens as queries (the ViT-Adapter "Injector" form, value = the 3-level CNN
pyramid):

- the query-side work (offset and weight projections, sampling, `output_proj`, ConvFFN) drops 5.25x. Only
  `value_proj` still runs on all 13,125 tokens;
- `CTI_toV` goes from ~47.5 to ~22 GFLOP (−26 GFLOP, −4% of the model), and its elementwise traffic from ~3.2
  to ~0.8 GB. The down/upsampling glue disappears;
- so `CTI_toV` itself would be 2-3x faster (12.3 ms → ~5 ms). **The whole model gets only ~5-8% faster,**
  because `CTI_toV` is 12% of the forward pass;
- risk: each ViT token sees the CNN branch through 72 samples instead of the pooled output of ~5 CNN queries,
  so a sparser view. Since `CTI_toV` is the part that carries CoMer's false-alarm advantage (section 3),
  this needs a training run before use. Cheap compensation if needed: more sampling points (`n_points: 8`) for
  these 2,500 queries only.

The inputs it needs already exist: `deform_inputs1` (stride-16 reference points and the 3-level shapes) is
passed into `CTIBlock.forward` but unused.

## 5. Recommendations

| goal | config | cost | accuracy |
|---|---|---|---|
| keep CoMer's accuracy | baseline | 688 GFLOP, 14.2 train img/s | 38.4 mAP, 1.40 FPPI@R0.7 |
| almost free | `ratio05` | −7% FLOPs, +4% train speed | −0.5 mAP (AP75), FPPI and recall unchanged |
| faster training, moderate loss | `lite` | −11% FLOPs, +27% train speed | −1.3 mAP, 1.58 FPPI@R0.7 (still far better than ConvNeXt-T's 2.76) |
| don't use | `slim` | −23% FLOPs, +10% train speed | −2.9 mAP, FPPI as bad as `noinj` |

- **If false alarms are the main criterion, use `ratio05`.** It is the only cut that keeps FPPI and recall.
  The saving is small (7%).
- **If training throughput matters more than the last 1.3 mAP, use `lite`.** It trains 27% faster, mostly
  because 3 of the 4 extra CTIs are gone. `extra0` trains as fast but loses more at high recall, so `lite` is
  the better trade.
- **Don't narrow the P2/P3 outputs** (this also revises item 2 in section 5 of `COMER_LATENCY.md`).
- To go much cheaper, the remaining options don't remove adapter capacity:
  - turn off gradient checkpointing for training on 96 GB cards (`with_cp: false`, `vit_with_cp: false`);
    the training step then skips the recomputed adapter forward;
  - `torch.compile` and TensorRT work from `COMER_LATENCY.md` section 5 (fusing the LayerNorm / cast / copy
    glue);
  - ViT-query `CTI_toV` (section 4), ~5-8% if it holds accuracy.

## 6. Not done / open

- **Clean speed benchmark.** The machine was shared with other users' trainings for the whole study, and
  a benchmark run under that load was unreliable (up to 40% drift between repeats). The train img/s above comes
  from each run's own log. `tools/comer_lite/slurm_bench.sbatch` measures train step, test throughput and
  per-module time for all 7 configs, with checkpointing on and off, and logs GPU clocks. Run it on an idle GPU:
  `IMAGE=artifactory.project.dso/srsep-docker/voltronix-mmrotate-dev:2026.10.19 WORK=work_dirs/comer_lite
  DATA_ROOT=/mnt/zfs/datasets/eo/soda-a/split_800/ GPU=<id> bash tools/comer_lite/slurm_bench.sbatch`.
- For reference, the combination planned first (`slim` + `ratio05` + `cfinter-late` + `extra_num 1` +
  `dedup_norm`) measured 437 GFLOP and 60 ms vs 102 ms per batch-8 forward on an idle GPU. It wasn't trained
  after `slim` lost 2.9 mAP on its own.
- Not run: fewer CNN→ViT injections, ViT-query `CTI_toV`, intermediate P2/P3 widths, longer schedules, more
  seeds, TensorRT fp16 of `lite`.

## 7. Files and how to reproduce

- `mmrotate/models/backbones/comer_lite/backbone.py`: `TimmViTCoMerLite(TimmViTCoMer)`, adds `out_channels`
  (narrow outputs, stride-4 path built at the narrow width) and `dedup_norm`. With default arguments it is
  identical to `TimmViTCoMer` (tested).
- `configs/comer_lite/*.yaml`: one config per run above.
- `tests/test_models/test_backbones/test_comer_lite.py`.
- `tools/comer_lite/profile_modules.py`: per-module GPU time and GFLOP; `slurm_bench.sbatch`: the cost benchmark.
- `tools/comer_lite/slurm_submit.sh round1|round2|bench`: queues the trainings on node `may` through
  `tools/soda_dinov3/slurm_run.sbatch`, then the false-alarm and results tables
  (`work_dirs/comer_lite/{results,false_alarms}.md`). The baseline and `noinj` runs are symlinked from
  `work_dirs/soda_dinov3/`.
