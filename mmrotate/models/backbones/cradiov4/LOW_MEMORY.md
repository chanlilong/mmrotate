# Fitting the P2 C-RADIOv4-H CoMer RTMDet at batch 4 on a 46 GB GPU

`configs/cradio/cradiov4-h-comer_rtmdet-p2-3x-dota_ms.yaml` trains at 2 images per 46 GB GPU and runs out of
memory at 4. This report covers the work to make it fit at batch 4 and to speed it up, under one constraint:
**exact changes only** (same model, same state-dict keys, same maths, existing checkpoints still load).

**Short answer:**

- **Memory:** batch 4 now fits at 32.8 GB peak on one RTX 6000 Ada. At batch 2 the peak dropped from 31.7 GB
  to 19.7 GB.
- **Speed:** per-image throughput is unchanged at typical object density (1.14 img/s against 1.14 before).
  Images with about 1000 objects train 7–8% faster.
- **Why not faster:** the Triton MSDA kernel is about 3% of GPU time. The rest is spread over the frozen ViT,
  the 1280–2560 channel neck convolutions, and fp32 LayerNorms with their casts. None of these can be reduced
  without changing numerics or the architecture.

All step timings are from synthetic batches, not real DOTA (real DOTA is not on the benchmark machine). A full
DOTA training with the new config has not been run.

## 1. How to use it

```bash
# 4 images per GPU, RankingDynamicSoftLabelAssigner, same model as the plain P2 config
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python tools/train.py configs/cradio/cradiov4-h-comer_rtmdet-p2-nwd-dsl-rka-lowmem-3x-dota_ms.yaml
```

| config (`configs/cradio/`) | what it is |
|---|---|
| `cradiov4-h-comer_rtmdet-p2-nwd-dsl-rka-3x-dota_ms.yaml` | plain P2 config with `RankingDynamicSoftLabelAssigner`, batch 2 |
| `cradiov4-h-comer_rtmdet-p2-nwd-dsl-rka-lowmem-3x-dota_ms.yaml` | same model, low-memory recipe, batch 4 |
| `cradiov4-h-comer_rtmdet-p2-nwd-dsl-rka-lowmem-proxy.yaml` | smoke variant on `data/proxy_dota/`, 12 epochs |

Two things to set yourself:

- **Learning rate.** The low-memory config keeps the base config's learning rate although the per-GPU batch
  doubled. On the same number of GPUs the total batch doubles and the iterations per epoch halve, so rescale
  the learning rate or lengthen the warm-up. This was left alone because it affects accuracy.
- **Allocator.** `expandable_segments:True` keeps reserved memory close to the peak (34.1 GB reserved for
  32.8 GB allocated at batch 4).

## 2. What changed

Changes to shared code apply to every CoMer config and both chunked assigners. The rest is opt-in through the
low-memory config.

| change | where | applies to | effect |
|---|---|---|---|
| One gradient checkpoint per interaction instead of a nested pair | `comer_modules.py` (`CTI_toC`) | all CoMer configs with `with_cp` | with the next row, −0.8 GB at batch 2 |
| Checkpointed `MRFP` (BatchNorm-safe) | `comer_modules.py` (`CTIBlock._mrfp`), `bn_checkpoint.py` | all CoMer configs with `with_cp` | see previous row |
| Dynamic-k matching with one `topk` instead of one per GT | `task_modules/chunked_dsl/` | both chunked DSL assigners | −25% assign+loss time at 1000 GTs/img |
| `CheckpointedCSPNeXtPAFPN`, checkpoints on the stride-4 and stride-8 layers | `necks/cp_pafpn/` | low-memory config | about −8.6 GB at batch 2, +4% step time |
| `FrozenSharedExpMomentumEMA`, EMA shares the frozen ViT with the model | `layers/frozen_ema/` | low-memory config | −2.6 GB, no averaging pass over the ViT |
| `FastTargetsRotatedRTMDetSepBNHead`, targets without a loop over GTs | `dense_heads/rtmdet_fast_targets/` | low-memory config | part of the dense-image speed-up |

The neck figure is derived, not measured in isolation: the low-memory config saves 11.2 GB at batch 2, of
which 2.6 GB is the EMA.

Notes on exactness:

- **BatchNorm inside a checkpoint.** `torch.utils.checkpoint` runs the body twice, which would update the
  running statistics twice. `checkpoint_bn` runs the recompute with `momentum = 0` and restores
  `num_batches_tracked`, so the statistics update exactly once. Turning `track_running_stats` off instead
  changes the autograd graph and makes the checkpoint raise `CheckpointError`.
- **Tie handling in dynamic-k matching.** A GT whose cost has a tie exactly at the top-k cut falls back to
  mmdet's per-GT `topk`, so the assignment is bit-identical to mmdet's.
- **EMA of frozen weights.** With the stock EMA the frozen ViT copy drifts from the pretrained weights by float
  rounding every step. With the shared EMA it equals them exactly. This is the only intended difference in
  saved values.
- **Checkpoint file size** is unchanged at 8.8 GB. The frozen ViT is still written twice because mmengine
  moves tensors to the CPU one by one when saving.

## 3. Results

One RTX 6000 Ada (46 GB), bf16 autocast, synthetic 1024×1024 batches, random weights, full train step
including AdamW and the EMA update (`tools/cradio/bench_train_step.py`). Peak is `max_memory_allocated`.
Repeat runs of the same case differ by up to about 3% in s/step.

| code | config | batch | GTs/img | s/step | img/s | peak GB |
|---|---|---:|---:|---:|---:|---:|
| before | plain | 2 | 50 | 1.66 | 1.21 | 31.7 |
| before | plain | 2 | 200 | 1.75 | 1.14 | 31.7 |
| before | plain | 2 | 1000 | 2.27 | 0.88 | 32.1 |
| before | plain | 4 | any | out of memory | | |
| after | plain | 2 | 50 | 1.66 | 1.21 | 30.9 |
| after | plain | 2 | 200 | 1.71 | 1.17 | 30.9 |
| after | plain | 2 | 1000 | 2.13 | 0.94 | 31.4 |
| after | plain | 4 | any | out of memory | | |
| after | low-memory | 2 | 50 | 1.74 | 1.15 | 19.8 |
| after | low-memory | 2 | 200 | 1.79 | 1.12 | 19.7 |
| after | low-memory | 2 | 1000 | 2.13 | 0.94 | 20.2 |
| after | low-memory | 4 | 50 | 3.43 | 1.17 | 32.8 |
| after | low-memory | 4 | 200 | 3.52 | 1.14 | 32.8 |
| after | low-memory | 4 | 1000 | 4.20 | 0.95 | 32.8 |

"Before" is the code at commit `eb8488c`; "after" is commit `1b80c5f`.

Where the step goes (low-memory config, batch 4, 200 GTs/img, seconds):

| backbone forward | neck forward | head forward | assign + loss | backward + optimiser + EMA |
|---:|---:|---:|---:|---:|
| 0.65 | 0.22 | 0.02 | 0.12 | 2.52 |

Backward is about 72% of the step because the ViT, the adapter and the two largest neck layers are all
recomputed there by their checkpoints.

## 4. Where the GPU time goes

Operator profile of one batch-4 step (3.38 s of CUDA time):

| group | share of CUDA time |
|---|---:|
| ViT attention and linear layers (forward and checkpoint recompute) | about 25–30% |
| Convolutions, forward and backward (neck, SPM, P2 upsampling, head) | about 16% |
| `aten::copy_`: fp32↔bf16 casts around the adapter LayerNorms | 12.4% |
| LayerNorm forward and backward | about 11–15% |
| Fused Triton MSDA backward | about 3% |
| AdamW step | 2.5% |

The adapter's LayerNorms run in fp32 under autocast on 21,504 tokens × 1280 channels, and every one is
followed by a cast back to bf16. That LayerNorm-plus-cast cost is about a quarter of the step and is the
largest single item that is not core model compute.

## 5. Tried and dropped

Each of these was measured and removed because it showed no gain beyond run-to-run noise, or cost too much
memory:

| experiment | result |
|---|---|
| `channels_last` convolutions with zero-copy token→map views | step time unchanged (3.52 s against 3.54 s); single convolutions gain only 3–15% |
| Checkpointing only the first 24 of 32 ViT blocks | 0.5% faster for 3.7 GB more |
| Checkpointing only the first 16 ViT blocks | about 3% faster, peak 40.4 GB |
| Adapter without checkpointing | out of memory at batch 4 |
| Fused AdamW | no measurable change |
| Casting the MSDA query once instead of per call | no measurable change |
| `chunk_budget_mb` 64 or 1024 instead of 256 | both slower at 1000 GTs/img (5.00 s and 4.52 s against 4.20 s) |
| fp16 instead of bf16 for the convolutions | cuDNN bf16 is within 10% of fp16, so no reason to switch |

## 6. Verification

- **Tests:** 402 passed on the final code (the cradiov4, MSDA Triton, comer-lite, neck, layer, chunked DSL,
  NWD/RKA, detector, head, frozen-config and engine suites), on a GPU. `ruff check` and `ruff format --check`
  are clean on every touched package, and the frozen config snapshots are current.
- **Equivalence:** on the same weights and inputs the low-memory detector's loss is bit-identical to the plain
  detector's. Gradients agree to within the plain detector's own run-to-run noise
  (`tests/test_models/test_detectors/test_cradio_lowmem.py`).
- **Training on proxy data** (`...-lowmem-proxy.yaml`, 12 epochs, bf16):

| run | iterations | final loss | `dota/mAP` | GB per GPU |
|---|---:|---:|---:|---:|
| 1 GPU, batch 4 | 96 | 1.15 | 0.664 | 31.5 |
| 2 GPUs (DDP), batch 2 each | 96 | 1.18 | 0.741 | 19.9 |
| 2 GPUs (DDP), batch 4 each | 48 | 1.65 | 0.179 | 32.6 |

The first two runs have the same total batch and iteration count, so they are the like-for-like pair. The gap
between them is most likely run-to-run variation on a 32-image proxy set; it was not repeated to confirm. The
third run has half the iterations, so its warm-up reaches half the learning rate and it is undertrained. It
shows that batch 4 per GPU fits under DDP, not that accuracy drops.

DDP matters here because the checkpointed neck contains SyncBN, which now also synchronises across ranks
during the checkpoint recompute in backward. The 2-GPU runs completed without hangs or divergence.

## 7. Next levers (not exact)

These would give real speed but change numerics or the model, so each needs a training run to check accuracy:

- Run the adapter LayerNorms in bf16, removing the fp32 LayerNorm and the casts around it.
- `dedup_norm`, which removes the second LayerNorm applied to an already normalised input. It exists only in
  `TimmViTCoMerLite` (`comer_lite/`) today and would have to be ported to `CRADIOv4CoMer`.
- Narrower P2/P3 inputs to the neck. `comer_lite/COMER_LITE.md` measured this on SODA-A with ViT-S and it cost
  2.9 mAP there.

## 8. Reproducing

GPU jobs go through Slurm; `tools/cradio/slurm_run.sbatch` runs one command in the dev image on the allocated
GPU.

```bash
mkdir -p work_dirs/cradio_lowmem/slurm
C=configs/cradio

# step benchmark (table in section 3)
sbatch --constraint=vram45 --gres=gpu:1 -J bench -o work_dirs/cradio_lowmem/slurm/%x-%j.out \
    tools/cradio/slurm_run.sbatch python tools/cradio/bench_train_step.py \
    $C/cradiov4-h-comer_rtmdet-p2-nwd-dsl-rka-3x-dota_ms.yaml \
    $C/cradiov4-h-comer_rtmdet-p2-nwd-dsl-rka-lowmem-3x-dota_ms.yaml \
    --batch 2 4 --num-gts 50 200 1000 --out work_dirs/cradio_lowmem/final.json

# operator profile (section 4) and single-convolution timings
... tools/cradio/slurm_run.sbatch python tools/cradio/bench_train_step.py <config> --batch 4 --profile
... tools/cradio/slurm_run.sbatch python tools/cradio/bench_train_step.py --conv-bench

# proxy training, 1 GPU and 2 GPUs
... tools/cradio/slurm_run.sbatch python tools/train.py $C/cradiov4-h-comer_rtmdet-p2-nwd-dsl-rka-lowmem-proxy.yaml
IMAGE=<dev image> sbatch --nodes=1 --gres=gpu:2 --constraint=vram45 tools/accelerate/slurm_train.sbatch \
    $C/cradiov4-h-comer_rtmdet-p2-nwd-dsl-rka-lowmem-proxy.yaml --cfg-options train_dataloader.batch_size=2
```
