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

Requirements: `tensorrt` (tested with `tensorrt-cu13==11.4.0.106`) and, for INT8 / FP8,
`nvidia-modelopt` (0.47.0, without the `[onnx]` extra). Both are in
`voltronix-mmrotate-dev:2026.10.29` (see `mmrotate_build/docker/dockerfile-redhat`).

## Usage

```bash
# Convert (writes work_dirs/trt/convnext_fp16/{end2end.engine,trt_meta.json,deploy_cfg.py})
python tools/tensorrt/convert.py configs/soda_dinov3/convnext-t-dinov3_rtmdet-p2-6e-soda.yaml \
    work_dirs/soda_dinov3/convnext-t-dinov3_rtmdet-p2-6e-soda/epoch_6.pth \
    --out-dir work_dirs/trt/convnext_fp16 --precision fp16 --batch-size 8

# INT8 / FP8: calibrated on --calib-num images of --calib-split (test pipeline, no augmentation)
python tools/tensorrt/convert.py CONFIG CKPT --out-dir DIR --precision int8 \
    --calib-split train --calib-num 64 --calib-method max

# Dataset metrics, same as tools/test.py
python tools/tensorrt/test_trt.py work_dirs/trt/convnext_fp16 [--out preds.pkl]

# Latency, torch vs one or more engines
python tools/tensorrt/bench.py CONFIG CKPT work_dirs/trt/convnext_fp16 work_dirs/trt/convnext_int8
```

Data paths and other settings are overridden with `--cfg-options` as for `tools/test.py`
(e.g. `test_dataloader.dataset.data_root=...`); `convert.py` uses `train/val/test_dataloader`
for calibration and verification images.

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
| `--verify N` | 8 | compare raw outputs and detections with the torch model on N `--verify-split` images; stored in `trt_meta.json` |
| `--keep-onnx` | off | keep `end2end.onnx` (+ external weights) next to the engine |
| `--opt-level`, `--workspace-gb` | 3, unlimited | TensorRT builder settings |

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
  - ViT-CoMer `MSDeformAttn`: Triton kernels replaced by its `grid_sample` PyTorch path;
  - DCFL head DCNv2 (`torchvision.ops.deform_conv2d`, no ONNX export): `deform_conv.py`, the same
    operation as `grid_sample` + a grouped 1x1 conv (tested against torchvision);
  - ViTDet `LayerNorm2d`: constant `normalized_shape`;
  - `F.scaled_dot_product_attention` (timm / RADIO ViTs): `softmax((q * scale) @ k^T) @ v` with
    the logits and softmax in fp32. The DINOv3 ViTs have q / k activations of ~2000: with torch's
    own ONNX decomposition the fp16 logits overflow (NaN engines), and even the fp32 engine of
    ViT-S CoMer at batch 8 lost 0.7 mAP; with this form fp32 and fp16 match torch;
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
    these in FP8, TensorRT produced NaNs on a CoMer model). Amax values whose fp16 scale would
    underflow are raised to the smallest normal scale.

ModelOpt prints a warning that it cannot build its CUDA extension (the runtime image has no
`nvcc`); it falls back to its PyTorch kernels, which only makes calibration a little slower.

## Checking a conversion

`convert.py --verify N` reports in `trt_meta.json`:

- `max_abs_err` / `max_rel_err` of the raw head outputs. This is the check that the engine
  computes the same function: ~1e-5 relative for fp32, ~1e-3 for fp16.
- `detections`: share of torch detections (score >= 0.05 / 0.3) matched by a same-class TensorRT
  detection with rotated IoU >= 0.5. This is **not** expected to be 100%, even at fp32: with the
  configs' rotated NMS at IoU 0.1 on dense small objects, suppression chains flip under any
  perturbation. Gaussian noise of 1e-6 on the torch model's own outputs already drops it to ~96%.

Accuracy is therefore compared on dataset metrics with `test_trt.py` (results below).

## Results

SODA-A `val_mini` (2588 patches of 800x800), `tools/test.py` (torch fp32) vs `test_trt.py`, engines
with batch 8, built and run on an RTX PRO 6000 Blackwell (TensorRT 11.4). INT8 / FP8 calibrated on
64 train images with `max`. mAP@[.5:.95] x 100:

| model | torch | TRT fp32 | fp16 | bf16 | fp8 | int8 |
|---|---|---|---|---|---|---|
| ConvNeXt-T DINOv3 | 38.43 | 38.43 | 38.44 | 38.39 | 38.27 | 5.5 |
| ConvNeXt-T DINOv3 + DCFL head | 40.51 | 40.51 | 40.51 | 40.24 | 40.64 | |
| ViT-S DINOv3 ViTDet-LoRA | 34.70 | 34.70 | 34.69 | 34.61 | 34.62 | |
| ViT-S DINOv3 CoMer | 39.02 | 38.28 * | 38.98 | 38.92 | 38.34 | 11.1 |
| ViT-S DINOv3 CoMer-LoRA | 38.33 | 35.53 * | 38.33 | 38.37 | 38.21 | |

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
- \* **fp32 CoMer engines at batch 8 are unreliable.** The first ViT block of the DINOv3 CoMer
  models has q / k activations of ~2000 (softmax logits ~1e5), so the result depends on the fp32
  summation order of the TensorRT tactics picked at build time (pure fp32 FFMA kernels, no TF32):
  one build reproduced torch (39.01, max output error 4e-3), the build in the table did not (max
  error 31). Batch 1 and 2 builds matched torch. `--verify` shows it (`max_abs_err`); fp16 / bf16
  with the fp32 attention logits above are not affected and are what to deploy.
