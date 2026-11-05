# Training with HF accelerate (DDP, DeepSpeed ZeRO, bf16 / fp16 / fp8)

`mmrotate/accelerate` runs the normal mmengine training loop through
[HF accelerate](https://huggingface.co/docs/accelerate). It covers:

- single GPU, multi-GPU and multi-node DDP;
- DeepSpeed ZeRO stages 0-3, with optional CPU optimizer offload;
- bf16 / fp16 autocast and fp8 (torchao `Float8Linear`);
- every mmrotate optimizer, including `MuonAdamW`.

Any existing config runs under it unchanged. The training recipe goes in an
`accelerator:` block of the config, and the `accelerate launch` command only
describes the topology.

Contents: [Requirements](#requirements) · [Quick start](#quick-start) ·
[The `accelerator:` block](#the-accelerator-block) · [What can be combined](#what-can-be-combined) ·
[How it works](#how-it-works) · [Checkpoints and resuming](#checkpoints-and-resuming) ·
[Validation and test across ranks](#validation-and-test-across-ranks) ·
[Troubleshooting](#troubleshooting) · [Verified runs](#verified-runs) ·
[Merging into another fork](#merging-into-another-fork)

## Requirements

Use the dev image `voltronix-mmrotate-dev:2026.11.05` (built from
`mmrotate_build/docker/dockerfile-redhat`). It contains:

| package     | version          | needed for |
| ----------- | ---------------- | ---------- |
| accelerate  | 1.15.0           | everything |
| deepspeed   | 0.19.9           | `accelerator.deepspeed`. CPU Adam and fused Adam ops are prebuilt, because the runtime image has no nvcc |
| torchao     | 0.18.0+cu130     | `mixed_precision: fp8` (needs an sm_89+ GPU: Ada, Hopper or Blackwell) |
| pyarrow     | 26.0             | the parquet result bridge |

## Quick start

```bash
CFG=configs/accelerate/cradiov4-so400m-comer-lora_rtmdet-p2-3x-dota_ms_accel-bf16.yaml

# 1 GPU (no launcher needed, also with DeepSpeed)
python tools/accelerate/train.py $CFG

# N GPUs on one machine
accelerate launch --num_processes N tools/accelerate/train.py $CFG
accelerate launch --config_file tools/accelerate/launch_configs/multi_gpu.yaml \
    tools/accelerate/train.py $CFG

# multi-node: run on every node with its own rank
accelerate launch --config_file tools/accelerate/launch_configs/multi_node.yaml \
    --machine_rank $NODE_RANK --main_process_ip $MAIN_IP tools/accelerate/train.py $CFG

# slurm, one or more nodes (IMAGE=... runs inside the docker image)
IMAGE=artifactory.project.dso/srsep-docker/voltronix-mmrotate-dev:2026.11.05 \
    sbatch --nodes=1 --gres=gpu:2 tools/accelerate/slurm_train.sbatch $CFG

# resume from the latest checkpoint in work_dir (or --resume path/to/epoch_N.pth)
accelerate launch --num_processes N tools/accelerate/train.py $CFG --resume

# test on N GPUs (same metrics as the single-GPU tools/test.py)
accelerate launch --num_processes N tools/accelerate/test.py $CFG work_dirs/.../epoch_36.pth
```

`tools/accelerate/train.py` and `test.py` take the same `--work-dir`,
`--cfg-options` and `--auto-scale-lr` options as `tools/train.py` and
`tools/test.py`. The stock `tools/train.py` also works with configs that set
`runner_type: mmrotate.AccelerateRunner`.

Example configs, in `configs/accelerate/`:

| config | recipe |
| ------ | ------ |
| `cradiov4-so400m-comer-lora_rtmdet-p2-3x-dota_ms_accel-bf16.yaml` | DDP + bf16 |
| `..._accel-zero2.yaml` | DeepSpeed ZeRO-2 + bf16, with optional CPU optimizer offload |
| `..._accel-zero3.yaml` | DeepSpeed ZeRO-3 + bf16 |
| `..._accel-fp8.yaml` | DDP + fp8 (torchao), experimental |
| `cradiov4-so400m-comer-lora-muon_rtmdet-p2-3x-dota_ms_accel-bf16.yaml` | MuonAdamW, DDP + bf16 |
| `cradiov4-so400m-comer-lora-muon_..._accel-fp8.yaml` | MuonAdamW, DDP + fp8 |
| `soda/vits-dinov3-comer-lora_rtmdet-p2-6e-soda_accel-{bf16,zero2,zero3}.yaml`, `soda/...-muon_..._accel-bf16.yaml` | SODA-A versions |
| `proxy/*.yaml` | smoke tests on `data/proxy_dota` |

A config only adds the accelerate parts to the config it inherits from:

```yaml
_base_: ../cradio/cradiov4-so400m-comer-lora_rtmdet-p2-3x-dota_ms.yaml
runner_type: mmrotate.AccelerateRunner
custom_imports: {imports: [mmrotate.accelerate], allow_failed_imports: false}
accelerator:
  mixed_precision: bf16
```

To switch to DeepSpeed, add:

```yaml
accelerator:
  deepspeed:
    zero_optimization: {stage: 2, overlap_comm: true, contiguous_gradients: true}
```

The recipe can also be changed from the command line:
`--cfg-options accelerator.deepspeed.zero_optimization.stage=3`.

## The `accelerator:` block

| key | default | meaning |
| --- | ------- | ------- |
| `mixed_precision` | `null` | `no`, `bf16`, `fp16` or `fp8`. `null` is taken from `optim_wrapper` (`AmpOptimWrapper` + `dtype: bfloat16` gives `bf16`, plain `AmpOptimWrapper` gives `fp16`, `OptimWrapper` gives `no`). Only if that also says nothing is `accelerate launch --mixed_precision` used. |
| `gradient_accumulation_steps` | 1 | Done by accelerate (gradient sync only on the last micro-batch). `optim_wrapper.accumulative_counts` is mapped here. |
| `deepspeed` | `null` | A DeepSpeed config (a dict or a path to a json file). `null` means DDP, or a single plain process. See [DeepSpeed](#deepspeed). |
| `zero3_leaf_modules` | `[]` | ZeRO-3 only: module class names that DeepSpeed gathers as a whole. |
| `zero3_sync_bn` | `revert` | ZeRO-3 with 2+ processes: `revert` turns SyncBN layers back into per-GPU BN (see [below](#what-can-be-combined)); `keep` leaves them. |
| `fp8` | `{}` | `recipe` (`tensorwise`, `rowwise` or `rowwise_with_gw_hp`), `exclude` (regexes of module names to keep in bf16), `min_dim` (16). |
| `ddp_kwargs` | `find_unused_parameters: false`, `broadcast_buffers: false` | Passed to `torch.nn.parallel.DistributedDataParallel`. The defaults match mmengine's. `model_wrapper_cfg` is ignored. |
| `timeout_minutes` | 60 | NCCL process-group timeout. |
| `data_order` | `auto` | Cross-rank dataset order check. `auto` re-sorts only if the ranks disagree, `always` always sorts, `never` raises instead of sorting, `off` disables the check. |
| `eval.bridge` | `parquet` | How validation and test results are gathered: `parquet` (files) or `mmengine` (stock `collect_results`). |
| `eval.dir` | work_dir | Directory for the bridge files. It must be on a filesystem every node sees. |
| `eval.timeout_minutes` | 240 | How long ranks wait for each other's parts, and for rank 0's metric. |
| `eval.keep_files` | false | Keep the parquet parts, for debugging. |

`train_dataloader.batch_size` stays per GPU, as with mmengine DDP. The global
batch is `num_processes x batch_size x gradient_accumulation_steps`. Set
`auto_scale_lr.enable` (or pass `--auto-scale-lr`) to scale the learning rate
when the GPU count changes.

### DeepSpeed

The DeepSpeed config holds only what DeepSpeed itself needs. The runner fills in
the rest:

| DeepSpeed key | filled from |
| ------------- | ----------- |
| `train_micro_batch_size_per_gpu` | `train_dataloader.batch_size` |
| `gradient_accumulation_steps` | `accelerator.gradient_accumulation_steps` |
| `gradient_clipping` | `optim_wrapper.clip_grad.max_norm` (L2 only) |
| `torch_autocast` | `mixed_precision`. This is autocast with fp32 master weights, as with `AmpOptimWrapper`. |
| `zero_optimization.stage3_gather_16bit_weights_on_model_save` | `true` for ZeRO-3, so `.pth` checkpoints hold the full weights |

The runner rejects configs that cannot work, with a message that says why:

- `optimizer` or `scheduler` sections in the DeepSpeed config: mmengine builds both.
- `bf16.enabled` or `fp16.enabled`: these cast the whole detector to bf16/fp16,
  which breaks BN statistics, the EMA (momentum 2e-4 vanishes in bf16) and the
  fp32 data preprocessor.
- fp16 with ZeRO stage 1 or higher: DeepSpeed's autocast has no loss scaling there.
- fp8 with DeepSpeed: that would need TransformerEngine.
- MuonAdamW with ZeRO stage 1 or higher.

With `offload_optimizer: {device: cpu}`, AdamW or Adam is swapped for
`DeepSpeedCPUAdam` with the same parameter groups.

## What can be combined

|                         | 1 GPU | DDP | ZeRO-0 | ZeRO-1/2 | ZeRO-3 |
| ----------------------- | :---: | :-: | :----: | :------: | :----: |
| fp32 / bf16 autocast    | ✓ | ✓ | ✓ | ✓ | ✓ |
| fp16 autocast + scaler  | ✓ | ✓ | ✓ | ✗ | ✗ |
| fp8 (torchao)           | ✓ | ✓ | ✗ | ✗ | ✗ |
| AdamW / SGD / ...       | ✓ | ✓ | ✓ | ✓ | ✓ |
| MuonAdamW               | ✓ | ✓ | ✓ | ✗ | ✗ |
| EMAHook                 | ✓ | ✓ | ✓ | ✓ | ✓ (as `ZeRO3EMAHook`) |
| SyncBN                  | (BN) | ✓ | ✓ | ✓ | reverted to BN |
| CPU optimizer offload   |   |   |   | ✓ (AdamW/Adam) | ✓ (AdamW/Adam) |

Why MuonAdamW is DDP only: Muon orthogonalises each full weight matrix
(Newton-Schulz). ZeRO flattens the parameters and splits them into 1-D shards
across ranks, so no rank holds a whole matrix. DDP keeps every matrix whole on
every rank.

Why SyncBN is reverted under ZeRO-3: mmdet's `RTMDet` turns every BN into
SyncBN whenever `world_size > 1`, and the CoMer adapter defaults to SyncBN.
ZeRO-3 gathers each module's weights from inside autograd during backward, on the
same NCCL communicator that SyncBN's backward all-reduce uses. Ranks with
different data take different backward paths, so one rank can reach a SyncBN
all-reduce while another waits in a ZeRO-3 gather. Both then block until the
NCCL timeout (seen on 2 GPUs with SODA-A after about 45 iterations). Per-GPU BN
avoids this; with `batch_size` 8 per GPU the statistics are still reliable.

Which recipe to choose for these detectors: the ViT trunk is frozen, and the
trainable part (adapter, neck, head, LoRA) is small. So DDP + bf16 is the
fastest choice. ZeRO-2 saves the optimizer and gradient memory of the trainable
part. ZeRO-3 also shards the weights, but has to gather every sub-module on
each pass, which makes it several times slower. Use it only when the trainable
model does not fit on one GPU.

fp8 converts only `nn.Linear` layers whose dimensions are divisible by 16. In
the LoRA configs the ViT blocks are `LoRALinear`, so they stay bf16, and fp8
covers only the CoMer adapter and deformable-attention linears. torchao fp8 is
fastest under `torch.compile`; measure before using it. The log line
`accelerate prepared ... N nn.Linear -> Float8Linear` shows how many layers were
converted.

## How it works

```
accelerate launch  ──>  tools/accelerate/train.py  ──>  AccelerateRunner(cfg)
  1. Accelerator(...)                       inits torch.distributed (NCCL); mmengine sees it and
                                            skips its own init_dist (launcher forced to 'pytorch')
  2. Runner.__init__                        model -> device (SyncBN reverted on 1 process), hooks,
                                            OptimWrapper/AmpOptimWrapper -> AccelerateOptimWrapper,
                                            ValLoop/TestLoop -> AccelerateValLoop/TestLoop,
                                            EMAHook -> ZeRO3EMAHook (ZeRO-3 only)
  3. dataloaders (lazy)                     dataset wrapped in ConsistentOrderDataset;
                                            mmengine DefaultSampler shards by rank;
                                            NOT passed to accelerator.prepare
  4. train(): optimizer, schedulers, before_run (EMA copy), weight init, load_from
  5. load_or_resume -> accelerator.prepare(model, optimizer)  -> DDP / DeepSpeedEngine
                                            wrapped in AccelerateModelWrapper (train_step)
  6. EpochBasedTrainLoop (stock)            DistSamplerSeedHook (stock) reshuffles each epoch
       train_step: accumulate + autocast { data_preprocessor + engine forward } -> accelerator.backward
                   -> clip -> step -> zero_grad (on sync steps)
  7. AccelerateValLoop                      per-rank metric.process -> parquet parts -> rank 0 metric
  8. CheckpointHook -> save_checkpoint      mmengine .pth (+ DeepSpeed state dir)
```

This design fixes the problems of the earlier integration:

- **Double sharding.** The dataloaders are never prepared by accelerate.
  mmengine's `DefaultSampler` already gives every rank its own shard, and
  preparing the loader would shard it a second time.
- **Shuffle frozen across epochs.** The stock `DistSamplerSeedHook` finds the
  sampler again, because the loader is not wrapped. `AccelerateDistSamplerSeedHook`
  remains only as an alias, so configs from the other fork still load.
- **Ranks listing the dataset differently** (`DOTADataset` uses an unsorted
  `glob`). `ConsistentOrderDataset` compares a fingerprint of each rank's sample
  list. If the fingerprints differ, every rank re-sorts by `img_path`. If they
  still differ, the ranks hold different files, and the run stops with an error.
- **Autocast.** accelerate's own `forward` autocast patch is removed after
  `prepare`. The optim wrapper autocasts the data preprocessor and forward,
  exactly where `AmpOptimWrapper` does, so validation stays fp32 as with mmengine.

## Checkpoints and resuming

- Checkpoints are normal mmengine `.pth` files: `state_dict` (EMA weights when
  `EMAHook` is used), `ema_state_dict`, `meta`, `message_hub`,
  `param_schedulers`. Without DeepSpeed they also hold the optimizer state and,
  for fp16, the `loss_scaler`. The stock `tools/test.py`, `tools/train.py
  --resume` and deployment tools read them. With ZeRO-3, the full weights are
  gathered before saving.
- With DeepSpeed, the sharded optimizer state goes to
  `<work_dir>/accelerate_state/epoch_N/` (`accelerator.save_state`), and
  `meta.accelerate_state` points to it. `--resume` restores the `.pth` and then
  that directory. State directories whose `.pth` has been removed by
  `max_keep_ckpts` are pruned at the next save.
- `load_from` (initialising from weights, not resuming) loads into the
  unwrapped model before `prepare`, so it also works for ZeRO-3.
- Resuming from a checkpoint written by the plain mmengine runner works without
  DeepSpeed: the optimizer state and the AMP `loss_scaler` use the same keys.

## Validation and test across ranks

mmengine's `collect_results` pickles every rank's results and gathers them over
NCCL, or through pickle files in `/tmp`, which is not shared across nodes. Rank
0 then computes the metric while the other ranks wait inside a collective. A
slow metric (DOTA or SODA-A mAP, with patch merging) can therefore trip the
NCCL watchdog and leave the run hanging at validation.

`AccelerateValLoop` and `AccelerateTestLoop` keep mmengine's loop and hooks.
Only the gathering changes:

1. Each rank runs `metric.process` on its own shard. Each result is tagged with
   its dataset index, taken from the rank's batch sampler.
2. Each rank writes `part_<rank>.parquet` plus a `.done` marker to
   `<work_dir>/accelerate_eval/<round>/`. A part has one row per result. Arrays
   are stored as raw bytes, with dtype and shape in a JSON spec, so nothing is
   pickled and decoding is exact. The bridge works for any metric whose results
   are built from dicts, lists, tuples, numpy arrays, tensors, scalars, strings
   or data elements.
3. Rank 0 waits for all parts and keeps one copy of each dataset index. This
   drops the samples the sampler pads in to even out the ranks. It then restores
   dataset order, runs `metric.compute_metrics`, and writes `metrics.parquet`.
4. The other ranks poll for `metrics.parquet`, so no NCCL collective is open
   while rank 0 computes. If rank 0 fails, it writes `error.txt` and the other
   ranks raise with its traceback instead of hanging. The round directory is
   deleted once every rank has read the metrics.

The result is the same as a single-process evaluation. On the same checkpoint,
the stock `tools/test.py` and `tools/accelerate/test.py` print identical
per-class tables. For multi-node runs, `work_dir` (or `eval.dir`) must be on a
shared filesystem.

## Troubleshooting

| symptom | cause / fix |
| ------- | ----------- |
| `ModuleNotFoundError: mpi4py` with DeepSpeed | Fixed: without a launcher, the runner now sets up a 1-process rendezvous itself. Under `accelerate launch` the launcher sets it. |
| `The size of tensor a (0) must match ...` in backward with ZeRO-3 | A parameter was re-partitioned before its gradient arrived. Do not put the whole backbone in `zero3_leaf_modules`. |
| NCCL watchdog timeout under ZeRO-3, one rank in `SyncBatchNorm` backward (`ALLREDUCE`), the other in `_ALLGATHER_BASE` | SyncBN and ZeRO-3 collectives interleaved differently on the ranks. Fixed by the default `zero3_sync_bn: revert`; do not set `keep` with data-dependent models. |
| ZeRO-3 on 2+ GPUs: NCCL timeout with one rank in `_REDUCE_SCATTER_BASE` (`independent_gradient_partition_epilogue`) and the other in `_ALLGATHER_BASE` (`pre_sub_module_backward_function`) | Data-dependent backward: a module ran backward on one rank only. Not solved yet for RTMDet; use ZeRO-2 or DDP. Set `timeout_minutes` low (e.g. 10) while debugging so a hang fails fast. |
| `the ranks hold different samples` | Ranks read different annotation files: different data roots, or `indices` / filters applied to an unsorted listing. |
| validation `TimeoutError ... still missing part_000NN` | That rank died or is still evaluating, or `eval.dir` is not shared across nodes. Raise `eval.timeout_minutes` for very slow metrics. |
| `KeyError: unknown accelerator keys` | A typo in the `accelerator:` block; the message lists the valid keys. |
| `MuonAdamW cannot run under DeepSpeed ZeRO` | Use DDP (remove `accelerator.deepspeed`). |
| many `_pytree ... register_constant` warnings at start-up | Import-time warnings from torchao in this image. Harmless. |

## Verified runs

All runs used the `2026.11.05` image on RTX PRO 6000 Blackwell GPUs.

- **Unit tests** (CPU, `pytest tests/test_accelerate`, 63 tests):
  - result bridge round trips, padding dedup, file rendezvous, error and timeout handling;
  - DeepSpeed config autofill and guards;
  - data-order reconciliation;
  - toy-model runs: single process (train, val, checkpoint, resume equal to the
    uninterrupted run) and two gloo processes. The two-process run checks that
    there is no double sharding, that the shuffle changes per epoch, that a
    rank-reversed listing is fixed, that padded validation is deduplicated, that
    metrics are identical on both ranks, and that the bridge files are cleaned up.
- **1 GPU, proxy DOTA, C-RADIOv4-SO400M CoMer LoRA, 2 epochs:**

  | recipe | s/iter | memory (MB) | notes |
  | ------ | -----: | ----------: | ----- |
  | bf16 | 0.95 | 25839 | |
  | ZeRO-2 | 2.9 | 29916 | |
  | ZeRO-3 | 11.2 | 30627 | per-module gather |
  | fp8 | 2.2 | 25763 | not faster without `torch.compile` |
  | MuonAdamW | 2.7 | 25596 | |

  Only the bf16 time is steady-state (its epochs 3-4). The other runs were
  2 epochs long, so their times are end-of-epoch-2 values that still include
  warm-up, and are not comparable with bf16.

  All five checkpoints have the same EMA behaviour (32 EMA steps; EMA within
  0.03 % of the raw weights). The bf16 checkpoint gives mAP 0.1425 through both
  the stock `tools/test.py` and `tools/accelerate/test.py`, with identical
  per-class tables. The ZeRO-3 checkpoint loads in the stock `tools/test.py`.
- **2 GPUs (slurm on `may`, SODA-A ViT-S/16 DINOv3 CoMer LoRA, batch 8 per GPU,
  1600-image train subset, 2 epochs, validation on the full val split):**

  | recipe | s/iter | memory (MB) | val mAP / AP50 (epoch 2) | resume to epoch 3 |
  | ------ | -----: | ----------: | ------------------------ | ----------------- |
  | DDP bf16 | 0.62 | 13 872 | 0.0092 / 0.0481 | ✓ (0.0203 / 0.1045) |
  | ZeRO-2 | 0.89 | 20 998 | 0.0061 / 0.0350 | ✓, DeepSpeed state restored (0.0081 / 0.0453) |
  | MuonAdamW (DDP) | 0.63 | 13 841 | 0.0062 / 0.0364 | |
  | ZeRO-3 | ~4 | 17 200 | **hangs** in epoch 1 (see below) | |

  The mAP is low because these are 200-iteration smoke runs, and the recipes
  differ in numerics, so the runs are not expected to match each other.

  Bridge check: the DDP `epoch_2.pth` evaluated on `val_mini` (324 images) by
  the stock `tools/test.py` on 1 GPU and by `tools/accelerate/test.py` on 2 GPUs
  gives the same 16 SODA-A metrics, digit for digit. They also equal the
  validation logged during training.

  **ZeRO-3 on 2+ GPUs does not work with these RTMDet detectors yet.** The first
  hang was SyncBN against a ZeRO-3 gather; that is fixed (`zero3_sync_bn`).
  The second hang, after about 60 iterations, has one rank finished with backward
  and reduce-scattering gradients while the other still gathers one more weight
  (147 456 elements) for a module backward. So the ranks ran a different set of
  module backwards in that iteration: something in the model's backward depends
  on the data. ZeRO-3 needs the same sequence on every rank. ZeRO-3 works on
  1 GPU. For these models, whose trainable part is about 20 M parameters, ZeRO-2
  gives most of the memory saving anyway.

## Merging into another fork

Everything is in new paths:

```
mmrotate/accelerate/                 # package (+ ruff.toml)
tools/accelerate/                    # train.py, test.py, slurm_train.sbatch, launch_configs/
configs/accelerate/                  # example configs (cradio, muon, soda, proxy)
tests/test_accelerate/               # unit + toy end-to-end tests
docs/en/user_guides/accelerate_training.md
```

No upstream file is edited. The registration happens through
`custom_imports: [mmrotate.accelerate]`, and `tools/accelerate/*.py` imports the
package directly. The only change outside this repo is in `mmrotate_build`: one
new Dockerfile step (DeepSpeed + torchao) plus its checks in the verification
step.

The other fork already has its own `mmrotate/accelerate/` (`accelerate_loop.py`,
`accelerate_optim_wrapper.py`, `accelerate_runner.py`,
`accelerate_sampler_seed_hook.py`, `metrics_check.py`) and probably
`tools/train_accelerate.py`. Suggested merge:

1. Merge this branch and resolve the `mmrotate/accelerate/` add/add conflicts
   by taking this version of the folder wholesale
   (`git checkout --theirs -- mmrotate/accelerate` when this branch is being
   merged in). Then delete the files that no longer exist here:
   `accelerate_sampler_seed_hook.py` and `metrics_check.py`. Keeping parts of
   both versions would mix two incompatible designs. The old one prepares the
   dataloaders; this one deliberately does not.
2. Configs from that fork keep loading where they use the same registered
   names. These are `AccelerateDistSamplerSeedHook` (now an alias of the stock
   hook), `AccelerateValLoop` and `AccelerateRunner`. Check them for:
   - `runner_type: AccelerateRunner` without a scope: use
     `mmrotate.AccelerateRunner`, or launch with `tools/accelerate/train.py`;
   - extra arguments to `AccelerateValLoop`: it now takes the stock `ValLoop`
     arguments, and the bridge options live in `accelerator.eval`;
   - `MMDetDotaResult` / `to_parquet` usage: replaced by the generic
     `result_bridge.py`;
   - an `Accelerator(...)` config section: move it into the `accelerator:`
     block (see the table above).
3. `tools/train_accelerate.py` (run as `accelerate launch -m tools.train_accelerate`)
   can be removed in favour of `tools/accelerate/train.py`, or kept as a
   two-line wrapper around it.
4. If that fork sorted `data_list` inside `DOTADataset`, the edit can stay.
   `ConsistentOrderDataset` no longer needs it, because it reconciles the order
   for any dataset, but the sort is harmless.
5. Rebuild the image from the updated Dockerfile, or use `2026.11.05`.
