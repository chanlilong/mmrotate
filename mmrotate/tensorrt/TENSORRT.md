# TensorRT export for single-stage detectors

`mmrotate/tensorrt` converts an mmrotate / mmdet `SingleStageDetector` into a TensorRT engine,
optionally INT8 or FP8 quantized with calibration images from the config's dataset. The engine is
saved to a *deploy directory* and reloaded as `TRTSingleStageDetector`, an inference-only
`SingleStageDetector` subclass whose `predict()` returns the same `DetDataSample`s
(`pred_instances.bboxes (N, 5)`, `scores`, `labels`) as the torch model. It works with
`Runner.test()`, the evaluators and `mmdet.apis.inference_detector`.

Any single-stage detector whose `bbox_head.forward` returns (nested tuples of) tensors and whose
`predict_by_feat` takes them is supported. Tested: the rotated RTMDet heads (incl. the DCFL head)
with C-RADIOv4-H (plain, ViT-CoMer, ViTDet, LoRA), DINOv3 ViT-S (CoMer, ViTDet, LoRA) and DINOv3
ConvNeXt-T backbones. These are the model families of every config in `configs/cradio` and
`configs/soda_dinov3` (the others differ in training settings or a CoMer option). Two-stage and
refine-stage detectors (Faster R-CNN, Oriented R-CNN, S2ANet, R3Det, ...) are rejected: their RoI
ops and dynamic proposal counts have no static-shape TensorRT export.

Requirements: `tensorrt` (tested with `tensorrt-cu13==11.4.0.106`), `onnx` (fused attention
export) and, for INT8 / FP8, `nvidia-modelopt` (0.47.0, without the `[onnx]` extra). They are in
`voltronix-mmrotate-dev:2026.10.29` (see `mmrotate_build/docker/dockerfile-redhat`) and in the image
of `docker/Dockerfile.trt`, which adds them to the dev image of `docker/Dockerfile`:

```bash
docker build -f docker/Dockerfile.trt --build-arg BASE=mmrotate-dev -t mmrotate-trt docker
```

Fused attention needs TensorRT >= 11.3 (10.x: use `--attention legacy`), the deformable-attention
plugin needs `tensorrt.plugin` with ahead-of-time plugins (TensorRT >= 10.9) and Triton. Neither
needs a CUDA toolkit (`nvcc`).

## Guide: converting a model

All commands run from the repository root, inside an image with the requirements above. `CONFIG`
is the model config and `CKPT` its checkpoint, e.g.

```bash
CONFIG=configs/soda_dinov3/vits-dinov3-comer_rtmdet-p2-6e-soda.yaml
CKPT=work_dirs/soda_dinov3/vits-dinov3-comer_rtmdet-p2-6e-soda/epoch_6.pth
```

Data paths and other settings are overridden with `--cfg-options` as for `tools/test.py`
(e.g. `test_dataloader.dataset.data_root=...`); `convert.py` takes calibration and verification
images from `train/val/test_dataloader`.

### 1. Convert to fp16

```bash
python tools/tensorrt/convert.py $CONFIG $CKPT --out-dir work_dirs/trt/comer_fp16 \
    --precision fp16 --batch-size 8
```

This writes the deploy directory `work_dirs/trt/comer_fp16/{end2end.engine,trt_meta.json,
deploy_cfg.py}`. fp16 is the recommended precision: it matches the torch model's mAP on every
model tested and is the fastest accurate engine on an RTX 3090. Nothing else needs choosing:

- the input size comes from the test pipeline (`--input-shape H W` to override), the batch size is
  static (`--batch-size`; other batch sizes are run in chunks);
- ViT attention is exported in the form TensorRT fuses (`--attention auto`);
- ViT-CoMer deformable attention becomes a TensorRT plugin around the Triton kernel, embedded in
  the engine (`--msda auto`).

### 2. Check the conversion

`convert.py` compares the engine with the torch model on 8 validation images (`--verify N`) and
logs the report, which is also in `trt_meta.json`. Look at:

- `finite: true` and a `max_rel_err` around 1e-2 or below for fp16 (the engine computes the same
  function);
- `fusion`: `mha` should equal `attention_calls` for a ViT, and `plugins` the number of
  deformable-attention modules for ViT-CoMer. Fewer `mha` layers means slower attention; add
  `--require-fused-attention` to make that an error.

### 3. Evaluate and benchmark

```bash
# Dataset metrics with the config's test loop, as tools/test.py
python tools/tensorrt/test_trt.py work_dirs/trt/comer_fp16

# Latency on dataset images against torch (--e2e adds box decoding and rotated NMS)
python tools/tensorrt/bench.py $CONFIG $CKPT work_dirs/trt/comer_fp16 --e2e

# Time per model component, slowest layers, what TensorRT fused
python tools/tensorrt/profile_engine.py work_dirs/trt/comer_fp16
```

Compare the mAP of `test_trt.py` with `tools/test.py $CONFIG $CKPT` on the same split. Per-image
detection agreement is not a good accuracy check (see Checking a conversion).

### 4. Optional: INT8 quantization with AWQ or GPTQ

INT8 is worth trying when throughput matters more than the last mAP points, mainly for CNN
backbones. It needs `nvidia-modelopt` and calibration images (64 un-augmented images of the train
split by default: `--calib-split`, `--calib-num`). Build the variants next to the fp16 engine and
compare them with steps 2 and 3:

```bash
# a. Plain INT8: range calibration only. The fastest, and the baseline for the rest.
python tools/tensorrt/convert.py $CONFIG $CKPT --out-dir work_dirs/trt/m_int8 \
    --precision int8 --batch-size 8

# b. AWQ: activation-aware per-channel scaling (and optionally its weight clip search).
python tools/tensorrt/convert.py $CONFIG $CKPT --out-dir work_dirs/trt/m_int8_awq \
    --precision int8 --batch-size 8 --quant-algo awq-scale            # or awq-scale,awq-clip

# c. GPTQ: error-compensating weight rounding. Combines with AWQ: awq-scale,gptq
python tools/tensorrt/convert.py $CONFIG $CKPT --out-dir work_dirs/trt/m_int8_gptq \
    --precision int8 --batch-size 8 --quant-algo gptq

# d. AWQ + the most sensitive layers left in fp16 (CNN backbones only; refused for ViTs).
python tools/tensorrt/convert.py $CONFIG $CKPT --out-dir work_dirs/trt/m_int8_mixed \
    --precision int8 --batch-size 8 --quant-algo awq-scale --quant-auto-exclude 0.05

python tools/tensorrt/bench.py $CONFIG $CKPT work_dirs/trt/m_fp16 work_dirs/trt/m_int8* --e2e
for d in work_dirs/trt/m_int8*; do python tools/tensorrt/test_trt.py $d; done
```

How to read the result:

- The conversion log prints what each algorithm did (`AWQ scale: n of m layers scaled`, `GPTQ: ...
  reconstruction error x0.41 of round-to-nearest`, `Sensitivity: n of m layers kept in fp16`); the
  same is in `trt_meta.json` under `calibration.refine`.
- `verify.noise` is the error power of the head outputs relative to their power. It ranks the
  variants quickly (lower is better; an fp16 engine is ~1e-4), before running the evaluation.
  `verify.noise_torch_quantized` is the same for the fake-quantized torch model: the two should be
  close, and `convert.py` fails if the engine is much worse (see INT8 below).
- Only `test_trt.py` tells what the noise costs in mAP, and only `bench.py` whether the engine is
  actually faster than fp16: the AWQ multiply and mixed int8 / fp16 graphs have a cost.

See Results for what these give on the DINOv3 models: which algorithm helped, and how INT8
compares with fp16 in speed.

## Using an engine

In Python:

```python
from mmdet.apis import inference_detector
from mmrotate.tensorrt import init_trt_detector

model = init_trt_detector('work_dirs/trt/convnext_fp16', device='cuda:0')
result = inference_detector(model, 'image.png')  # DetDataSample, as for the torch model
```

Or as a model in any config: `deploy_cfg.py` is the source config with `model` replaced by

```python
model = dict(
    type='TRTSingleStageDetector',
    engine_file='{{ fileDirname }}/end2end.engine',
    meta_file='{{ fileDirname }}/trt_meta.json',
    bbox_head=...,
    data_preprocessor=...,
    test_cfg=...,
)
custom_imports = dict(imports=['mmrotate.tensorrt'], allow_failed_imports=False)
```

Paths are relative to the directory, so a deploy directory can be copied elsewhere. An engine
only runs on the GPU architecture and TensorRT version it was built with; `trt_meta.json` records
both, and loading warns on a mismatch.

### `convert.py` options

| option | default | |
|---|---|---|
| `--precision` | `fp16` | `fp32`, `fp16`, `bf16`, `int8`, `fp8` (FP8: compute capability 8.9+) |
| `--input-shape H W` | test pipeline | static engine input; from the `Pad` size, else the `Resize` scale, rounded up to `pad_size_divisor` |
| `--batch-size` | 1 | static engine batch; bigger batches are run in chunks, the last one padded |
| `--calib-split` / `--calib-num` | `train` / 64 | INT8 / FP8 calibration images (shuffled, seed 0) |
| `--calib-method` | `max` | `max`, `mse` (ModelOpt algorithms), `entropy`, `percentile` (activation histograms, INT8 only) |
| `--quant-exclude` | see below | fnmatch patterns of module names kept in fp16 |
| `--quant-algo` | none | INT8 refinement after calibration, comma-separated: `awq-scale`, `awq-clip`, `gptq` (see INT8 below) |
| `--quant-auto-exclude BUDGET` | off | INT8: leave the most sensitive layers in fp16 so the summed single-layer output noise stays below `BUDGET` (e.g. `0.01`) |
| `--quant-fp32-layers` | with auto-exclude | INT8: type the quantized layers fp32 instead of fp16 (needed for partially quantized graphs) |
| `--attention` | `auto` | ViT attention export: `auto`, `fused`, `fused-fp32acc`, `legacy` (see Attention below); always `legacy` for `fp32` |
| `--require-fused-attention` | off | fail unless TensorRT fused every attention call |
| `--msda` | `auto` | ViT-CoMer deformable attention: `aot` / `jit` TensorRT plugin around the Triton kernel, or `gridsample`; `auto` = `aot`, falling back to `jit` if the build fails |
| `--verify N` | 8 | compare raw outputs and detections with the torch model on N `--verify-split` images; stored in `trt_meta.json` |
| `--keep-onnx` | off | keep `end2end.onnx` (+ external weights) next to the engine |
| `--opt-level` | 1 for `fp32`, else 3 | TensorRT builder optimization level (see the fp32 note below) |
| `--workspace-gb` | unlimited | TensorRT workspace limit |

## How it works

- **Engine = `detector._forward(inputs)`**: backbone, neck and `bbox_head.forward`, i.e. the raw
  per-level outputs (`cls_scores`, `bbox_preds`, `angle_preds`, ...). Their nesting is stored in
  `trt_meta.json` (`output_spec`). `TRTSingleStageDetector.predict` rebuilds it and calls the
  unchanged `bbox_head.predict_by_feat` (decoding, score filtering, rotated NMS, rescaling), so the
  post-processing is the torch code itself.
- **Export** uses the TorchScript ONNX exporter (`dynamo=False`) at a static shape: the backbones'
  shape-dependent Python (pad to patch size, position-embedding interpolation, deformable
  reference points) folds into constants.
- **Export patches** (`export_patches.export_mode`, applied to module instances during export and
  undone afterwards, no model code changes):
  - ViT-CoMer `MSDeformAttn`: one `mmrotate::MSDAFused` plugin node per call, or (`--msda
    gridsample`) its `grid_sample` PyTorch path (see Deformable attention below);
  - DCFL head DCNv2 (`torchvision.ops.deform_conv2d`, no ONNX export): `deform_conv.py`, the same
    operation as `grid_sample` + a grouped 1x1 conv (tested against torchvision);
  - ViTDet `LayerNorm2d`: constant `normalized_shape`;
  - `F.scaled_dot_product_attention` (timm / RADIO ViTs): an ONNX `Attention` node that TensorRT
    runs as one fused kernel, or (`--attention legacy`, fp32 engines) explicit matmuls with fp32
    logits (see Attention below);
  - `bf16` only: `ConvTranspose2d` runs in fp32 (TensorRT 11 has no bf16 deconvolution).
- **Precision** is fixed in the ONNX graph; TensorRT builds a *strongly typed* network (the only
  mode in TensorRT 11, which has no FP16/INT8 builder flags or implicit INT8 calibrators):
  - `fp32`: TF32 disabled, so it reproduces the torch model to ~1e-5;
  - `fp16` / `bf16`: the graph is traced under `torch.autocast`, i.e. the same mixed precision as
    torch AMP (convs / matmuls in low precision, normalisation and softmax in fp32);
  - `int8` / `fp8`: ModelOpt post-training quantization in torch (per-channel weights,
    per-tensor activations of conv / linear layers), calibrated on GPU, then traced under fp16
    autocast. FP8 Q/DQ use ModelOpt's ONNX symbolics; INT8 Q/DQ use our own
    (`precision._Int8QDQ`), because ModelOpt's INT8 op segfaults the TorchScript exporter of torch
    2.14. Kept in fp16 by default (`DEFAULT_QUANT_EXCLUDE`): the RTMDet heads' final prediction convs
    (`*bbox_head.rtm_*`) and deformable-attention `sampling_offsets` / `attention_weights` (with
    these in FP8, TensorRT produced NaNs on a CoMer model); with fused attention also the
    projection after the attention (`*.attn.proj`, see INT8). Amax values whose fp16 scale would
    underflow are raised to the smallest normal scale.

ModelOpt prints a warning that it cannot build its CUDA extension (the runtime image has no
`nvcc`); it falls back to its PyTorch kernels, which only makes calibration a little slower.

## Attention

TensorRT has a fused attention kernel (an `mha` layer in the engine: tiled, no `S x S` logit
tensor in memory, tensor cores), but only for fp16 / bf16 tensors and only if it recognises the
attention. The first version of this package exported fp32 logits between low-precision matmuls,
which is never fused: every block materialised a `(B, heads, 2505, 2505)` fp32 tensor. Measured for
one DINOv3 ViT-S block at 800x800 on an RTX 3090 (batch 8, 2505 tokens, real activations; error is
the RMS error against fp32 attention relative to the output RMS):

| form | TensorRT layer | ms | error block 0 | block 3 | block 6 |
|---|---|---|---|---|---|
| `legacy`: fp32 logits + softmax, fp16 second matmul | 2 gemm + softmax | 8.5 | 0.67% | 0.57% | 0.05% |
| `fused`: `Attention` node on fp16 tensors | `mha` | 0.98 | NaN | 0.84% | 0.13% |
| `fused-fp32acc`: `Attention` node on fp32 casts of q, k, v | `mha` | 1.50 | 0.67% | 0.57% | 0.05% |
| `Attention` node on bf16 casts | `mha` | 1.51 | 2.3% | 4.5% | 0.43% |

- The fused forms are an ONNX `Attention` node (opset 23), which the TensorRT parser turns into
  its attention layer. With q, k and v cast to fp32 and the result cast back, TensorRT (>= 11.3)
  keeps the fused kernel and accumulates in fp32. The plain `MatMul -> Softmax -> MatMul` pattern
  is fused too in pure fp16, but not with the fp32 casts, so the node is used for both.
- The TorchScript exporter stops at opset 20: the node is exported as a placeholder and
  `attention.upgrade_attention_nodes` rewrites the file (opset 23, `GridSample` mode names).
- **Block 0 of the DINOv3 ViTs cannot run in fp16**: q reaches ~2000 and the logits 2.5 million.
  `--attention auto` (default) runs every attention call on calibration images
  (`attention.probe_attention`, stored per call in `trt_meta.json`) and exports `fused` where an
  fp16 accumulation is safe and accurate, `fused-fp32acc` elsewhere (block 0 here).
- bf16 does not overflow, but q, k and v are rounded to 8 bits of mantissa before the attention,
  which costs 3-8x the error of fp16 I/O. In a bf16 engine that rounding is there whatever the
  attention does, so bf16 engines use `fused`.
- `convert.py` counts the `mha` layers of the built engine (`fusion` in `trt_meta.json`), warns
  when there are fewer than attention calls, and fails with `--require-fused-attention`.
- Not fused: fp32 engines (no fp32 kernel), masked attention (exported in `legacy` form), INT8
  attention beyond 512 tokens (so the attention matmuls of INT8 engines stay fp16).

## Deformable attention (ViT-CoMer)

`MSDeformAttn` is exported as one `mmrotate::MSDAFused` node per call, taking what the fused Triton
kernel of `cradiov4/msda_triton_fused.py` takes: value, level shapes, reference points, raw
sampling offsets and raw attention logits. `plugins/msda.py` registers a `tensorrt.plugin` plugin
for it that runs **the same Triton kernel** (softmax, sampling locations, bilinear sampling and
weighted sum in one pass; fp16 / bf16 / fp32 I/O, fp32 math):

- `--msda aot` (default): the kernel is compiled by Triton to PTX when the engine is built and
  embedded in the engine. TensorRT launches it itself; the engine loads without this package or
  Triton. `BLOCK_N` / `num_warps` are the values Triton's autotuner picked on the build GPU.
- `--msda jit`: TensorRT calls back into Python, which launches the kernel through Triton. The
  engine needs `mmrotate.tensorrt.plugins` (imported by `TRTSingleStageDetector`) and Triton at
  run time. Same speed as `aot`.
- `--msda gridsample`: the previous export (`grid_sample` per level in fp32), standard ONNX ops.

Two details the ahead-of-time compile needs (`plugins/msda.py`):

- Triton >= 3.3 appends two scratch-memory pointers to every kernel signature. TensorRT passes
  only the declared tensors and scalars, so they are stripped from the PTX entry (they are unused).
- Triton's JIT marks 16-byte aligned pointers, which lets the compiler vectorise the value
  gathers. Without that hint the same kernel is 10x slower (2.7 ms against 0.28 ms per call), so
  the alignment is declared explicitly.

TensorRT 11.4 corrupts the heap when an execution context of an engine with `jit` plugins is
destroyed without ever having run. `TRTEngine` therefore creates its context on first use.

## INT8

`--precision int8` alone is ModelOpt's range calibration (`--calib-method`). Three refinement
algorithms and a mixed-precision search can be added (`mmrotate/tensorrt/quant`):

- `--quant-algo awq-scale`: AWQ's activation-aware scale search. Per layer, input channel `c` is
  divided by `s_c = act_range^a / weight_range^(1-a)` and its weights multiplied by `s_c`, with
  `a` searched to minimise the layer's W8A8 output error. The float layer is unchanged. The
  division stays in the graph as a per-channel multiply in front of the Q/DQ pair, which gives the
  effect of per-channel activation quantization (TensorRT only has per-tensor activation scales).
- `--quant-algo awq-clip`: AWQ's clip search, a per-output-channel shrink of the weight range.
- `--quant-algo gptq`: GPTQ rounding of linear and ungrouped conv weights (inverse-Hessian error
  compensation, layers in forward order with inputs from the model quantized so far).
- `--quant-auto-exclude BUDGET`: quantizes one layer at a time, measures the noise it causes at
  the head outputs (noise power / output power), and keeps layers quantized from the least
  sensitive on while the sum stays below `BUDGET`. The rest run in fp16 with their original
  weights. `trt_meta.json` lists the excluded layers.

`convert.py --verify` checks a quantized engine against the fake-quantized torch model (`noise`,
`noise_torch_quantized` in `trt_meta.json`) and **fails** when the engine is much noisier than
quantization explains. This catches a TensorRT behaviour of partially quantized graphs: when an
unquantized fp16 layer directly feeds a quantized one, TensorRT fuses the two into an fp16-input /
int8-output kernel that does not exist for some layers (build error) or gives wrong results (mAP
near 0). Three consequences:

- `--quant-auto-exclude` types all quantized layers fp32 (`--quant-fp32-layers`; they still run as
  int8 kernels) and should be used with `--quant-algo awq-scale`: the cast and the AWQ multiply in
  front of each Q node keep TensorRT from fusing across the boundary. That combination is what was
  tested (ConvNeXt-T); typing only the convolutions fp32 was not enough.
- With fused attention, the projection after the attention (`*.attn.proj`) is kept unquantized: a
  Q node directly on the attention output makes TensorRT build an int8 attention, which has no
  kernel beyond 512 tokens.
- `--quant-auto-exclude` is refused for models with fused attention: fp32-typed linear layers
  around the attention leave it without a kernel. For ViTs, use `--quant-algo awq-scale` alone.

## Checking a conversion

`convert.py --verify N` reports in `trt_meta.json`:

- `max_abs_err` / `max_rel_err` of the raw head outputs. This is the check that the engine
  computes the same function: ~1e-5 relative for fp32, ~1e-3 for fp16.
- `noise`: error power of the head outputs relative to their power (and, for quantized engines,
  the same for the fake-quantized torch model, see INT8).
- `fusion`: fused attention (`mha`) and plugin layers of the engine against the attention calls
  of the model.
- `detections`: share of torch detections (score >= 0.05 / 0.3) matched by a same-class TensorRT
  detection with rotated IoU >= 0.5. This is **not** expected to be 100%, even at fp32: with the
  configs' rotated NMS at IoU 0.1 on dense small objects, suppression chains flip under any
  perturbation. Gaussian noise of 1e-6 on the torch model's own outputs already drops it to ~96%.

Accuracy is therefore compared on dataset metrics with `test_trt.py` (results below).

## Results

### RTX 3090 (TensorRT 11.4, torch 2.14, this version)

SODA-A `val_mini` (2588 patches of 800x800), `tools/test.py` (torch fp32) vs `test_trt.py`; speed
from `bench.py --e2e` on dataset images, batch 8, in images / s. "Network" is `_forward`
(backbone, neck, head); "end to end" adds box decoding and rotated NMS, the same torch code for
every row. "old export" is `--attention legacy --msda gridsample`, the export of the first version
of this package. Checkpoints: ConvNeXt-T and ViTDet after 6 epochs, CoMer without injector
(`noinj`) after 1 epoch (so its absolute mAP is low; what matters is that it is reproduced).

| model | runtime | network | end to end | mAP x 100 |
|---|---|---:|---:|---:|
| ConvNeXt-T DINOv3 | torch fp32 | 56.2 | 48.3 | 38.48 |
| | torch fp16 AMP | 115.6 | 85.9 | |
| | TensorRT fp16 | 185.3 | 118.3 | 38.46 |
| ViT-S DINOv3 ViTDet | torch fp32 | 27.7 | 24.9 | 17.78 |
| | torch fp16 AMP | 66.0 | 51.9 | |
| | TensorRT fp16, old export | 53.9 | 43.6 | 17.81 |
| | TensorRT fp16 | 128.9 | 84.3 | 17.78 |
| ViT-S DINOv3 CoMer, no injector | torch fp32 | 17.5 | 16.2 | 16.59 |
| | torch fp16 AMP | 37.8 | 32.2 | |
| | TensorRT fp16, old export | 25.6 | 23.0 | 16.58 |
| | TensorRT fp16 | 66.5 | 51.4 | 16.58 |
| | TensorRT fp16, `--msda jit` | 66.5 | 51.2 | 16.59 |
| | TensorRT fp16, `--attention fused-fp32acc` | 63.0 | 49.5 | 16.57 |
| | TensorRT bf16 | 53.0 | 43.0 | 16.58 |
| ViT-S DINOv3 CoMer (full) \* | torch fp32 | 15.2 | 14.5 | |
| | torch fp16 AMP | 32.0 | 29.0 | |
| | TensorRT fp16, old export | 20.6 | 19.3 | |
| | TensorRT fp16 | 56.0 | 47.5 | |

\* No trained checkpoint on this machine: the full architecture with the no-injector weights (the
injector gates are zero-initialised), so only the speed is meaningful. Its engine matches the torch
model on the verification images.

At batch 1 the CoMer (no injector) network takes 16.3 ms in the TensorRT fp16 engine, 39.3 ms with
the old export and 34.3 ms in torch fp16 AMP.

- **fp16 is the recommended precision.** It reproduces the torch mAP on every model and is
  1.6-1.9x faster than torch fp16 AMP.
- **Fused attention and the deformable-attention plugin make the ViT engines 2.4-2.7x faster**
  than the old export, which was slower than torch AMP. Where the time went, for the full CoMer at
  batch 8 (`profile_engine.py`, ms per batch):

  | component | old export | this version |
  |---|---:|---:|
  | ViT attention (12 blocks, with qkv / proj) | 131.8 | 16.0 |
  | deformable attention (12 calls) | ~200 \*\* | 20.6 |
  | rest of the backbone (ViT MLPs and norms, adapter convs / norms / FFNs) | 66.1 | 75.0 |
  | neck | 24.3 | 24.2 |
  | head | 6.4 | 6.3 |
  | total while profiling | 439 | 159 |

  \*\* 10.5 ms of projections plus the `grid_sample` layers, which TensorRT names without a module
  scope (197 ms of unattributed layers, 10-11 ms per `grid_sample` block).

  What is left is spread over many layers: the largest single one takes 2.2 ms. See
  `COMER_LATENCY.md` for the structural costs (384-channel P2 neck, 13125 adapter tokens).
- `--msda jit` is as fast as the embedded kernel. `fused-fp32acc` for every block costs 5% and
  changes nothing in mAP: `auto` only needs it for block 0. bf16 is slower than fp16 here.

INT8 (same setup; calibrated on 64 train images with `max`; "fp16 layers" are the layers
`--quant-auto-exclude` left unquantized):

| model | INT8 variant | fp16 layers | network | end to end | mAP x 100 |
|---|---|---:|---:|---:|---:|
| ConvNeXt-T | fp16 engine, for reference | | 185.3 | 118.3 | 38.46 |
| | plain (`max` calibration) | 0 / 120 | 235.9 | 136.8 | 35.55 |
| | `gptq` | 0 | 235.7 | 135.5 | 35.36 |
| | `awq-scale` | 0 | 167.7 | 109.8 | 37.37 |
| | `awq-scale,awq-clip` | 0 | | | 37.24 |
| | `awq-scale,awq-clip,gptq` | 0 | | | 37.31 |
| | `awq-scale` + auto-exclude 0.05 | 13 | 107.4 | 80.7 | 38.16 - 38.36 \*\*\* |
| | `awq-scale` + auto-exclude 0.01 | 42 | 124.1 | 90.0 | 38.23 |
| | `awq-scale,awq-clip,gptq` + auto-exclude 0.01 | 41 | | | 38.29 |
| CoMer, no injector | fp16 engine, for reference | | 66.5 | 51.4 | 16.58 |
| | plain | 0 / 161 | 83.9 | 72.1 | 2.66 |
| | `gptq` | 0 | 83.7 | 71.9 | 2.70 |
| | `awq-scale` | 0 | 63.4 | 51.1 | 14.63 |
| | `awq-scale,gptq` | 0 | 63.4 | 51.0 | 14.68 |
| ViTDet | fp16 engine, for reference | | 128.9 | 84.3 | 17.78 |
| | plain, pyramid stages in fp16 \*\*\*\* | 0 / 99 | 171.2 | 112.7 | 1.20 |
| | `gptq`, pyramid stages in fp16 \*\*\*\* | 0 | 170.9 | 113.3 | 1.21 |
| | `awq-scale` | 0 / 107 | 132.6 | 86.1 | 16.33 |
| | `awq-scale,gptq` | 0 | 132.5 | 86.3 | 16.25 |

\*\*\* Two builds of the same graph.

\*\*\*\* `--quant-exclude` with `'*backbone.fpn.stages.*'` added, see the last finding.

- **INT8 does not beat fp16 on these models on an RTX 3090.** Plain INT8 is 1.26-1.32x faster on
  the network but loses 2.9 mAP on ConvNeXt-T and collapses on the ViTs (2.7 and 1.2 mAP). The variants
  that get the accuracy back are no faster than fp16: the AWQ multiply in front of each Q node and
  the fp32-typed mixed-precision graph cost more than int8 kernels save. The attention matmuls
  stay fp16 in any case (TensorRT's int8 attention stops at 512 tokens).
- **AWQ's scale search is what repairs the accuracy**: 35.55 -> 37.37 on ConvNeXt-T, 2.66 -> 14.63
  on CoMer, 1.20 -> 16.33 on ViTDet (still 1.1-2.0 below fp16). The loss comes from per-tensor activation ranges with outlier channels, which is what
  the per-channel scale removes.
- **GPTQ and the AWQ clip search change mAP by less than 0.2**, alone or on top of the scale
  search. GPTQ does what it is meant to (weight reconstruction error 0.36x of round-to-nearest on
  ConvNeXt-T, 0.11x on the ViTs), but at 8 bits weight rounding is not what costs accuracy here.
- **Leaving the sensitive layers in fp16 closes the rest of the gap on the CNN** (38.2-38.4 with
  13 of 120 layers in fp16). The sensitive layers are mostly the second linear layer of the
  ConvNeXt blocks. `--quant-auto-exclude` does not work for ViTs with fused attention (the engine
  fails to build; `convert.py` refuses the combination).
- Plain INT8 of ViTDet does not build on this GPU with the default exclusions: TensorRT has no int8
  kernel for a transposed convolution of the pyramid. Adding `'*backbone.fpn.stages.*'` to
  `--quant-exclude` keeps those stages in fp16 (the two rows marked above); with `awq-scale` the
  default exclusions build.

### RTX PRO 6000 Blackwell (TensorRT 11.4, first version of this package)

These tables predate fused attention, the deformable-attention plugin and the INT8 refinement
(old export, plain INT8 calibration), and are not comparable with the RTX 3090 numbers above.

SODA-A `val_mini` (2588 patches of 800x800), `tools/test.py` (torch fp32) vs `test_trt.py`, engines
with batch 8, built and run on an RTX PRO 6000 Blackwell (TensorRT 11.4). INT8 / FP8 calibrated on
64 train images with `max`. mAP@[.5:.95] x 100:

| model | torch | TRT fp32 | fp16 | bf16 | fp8 | int8 |
|---|---|---|---|---|---|---|
| ConvNeXt-T DINOv3 | 38.43 | 38.43 | 38.44 | 38.39 | 38.27 | 5.5 |
| ConvNeXt-T DINOv3 + DCFL head | 40.51 | 40.51 | 40.51 | 40.24 | 40.64 | |
| ViT-S DINOv3 ViTDet-LoRA | 34.70 | 34.70 | 34.69 | 34.61 | 34.62 | |
| ViT-S DINOv3 CoMer | 39.02 | 39.01 * | 38.98 | 38.92 | 38.34 | 11.1 |
| ViT-S DINOv3 CoMer-LoRA | 38.33 | 38.32 * | 38.33 | 38.37 | 38.21 | |

`_forward` latency, batch 8 at 800x800 (`bench.py`, images / s):

| model | torch fp32 | torch fp16 AMP | torch bf16 AMP | TRT fp32 | TRT fp16 | TRT bf16 | TRT fp8 | TRT int8 |
|---|---|---|---|---|---|---|---|---|
| ConvNeXt-T DINOv3 | 124 | 298 | 313 | 146 | 466 | 463 | 479 | 492 |
| ViT-S DINOv3 CoMer | 37 | 75 | 77 | 28 | 47 | 46 | 49 | 48 |

C-RADIOv4-H at 1024x1024 (CoMer at fp32 / fp16 / bf16 / int8 / fp8; plain, ViTDet-P2, CoMer-DCFL
and CoMer-LoRA-P2 at fp16) converts and builds in 1-3 minutes per engine; there is no trained
checkpoint to evaluate, the fp32 engine reproduces the torch outputs to 2e-5 relative.

Findings:

- **fp16 is the recommended precision**: it matches torch on every model above. bf16 is within
  0.3 mAP, FP8 (compute capability 8.9+) too except CoMer (-0.7), and neither is faster than fp16
  here.
- **Speed depends on the backbone.** ConvNeXt-T runs 1.6x faster than torch fp16 AMP. ViT-CoMer
  runs *slower* than torch AMP (47 vs 75 images / s): torch uses the fused Triton deformable
  attention and flash attention, while the engine has the `grid_sample` deformable attention
  (large `(B * heads, C, queries, points)` intermediates) and the explicit fp32-logit attention,
  which TensorRT does not fuse. For CoMer models, TensorRT is mainly useful for deploying without
  a PyTorch / Triton stack.
- **INT8 does not work for these DINOv3 models.** Per-tensor INT8 activations cannot represent
  their outlier channels (e.g. amax 157 at the neck input, where typical values are ~1): with
  weights-only INT8 the outputs stay close, with INT8 activations they do not, for every
  calibration method (`max`, `mse`, `entropy`, `percentile`) and with BatchNorm folded first. Use
  FP8, or check `trt_meta.json` / `test_trt.py` before deploying an INT8 engine of another model.
- \* **fp32 engines are built at optimization level 1.** The first ViT block of the DINOv3 CoMer
  models has q / k activations of ~2000 (softmax logits ~1e5), which makes the fp32 result
  sensitive to the kernels TensorRT picks. At level 3, the batch-8 CoMer engines were wrong on
  every build (max output error 31; 38.28 and 35.53 mAP), while levels 0 and 1 reproduce torch
  (error 4e-3; the table above). Level 1 costs 5-10% fp32 throughput (ConvNeXt-T, CoMer), so
  it is the default for `fp32` only; the fp32 latencies below are level-3 builds. `--verify`
  reports `max_abs_err` for every build, so check it when changing `--opt-level`.
