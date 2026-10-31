# Why ViT-S CoMer is 4-10x slower than ConvNeXt-T

`TENSORRT.md` reports, at batch 8 and 800x800 (images / s):

| model | torch fp16 AMP | TRT fp16 |
|---|---|---|
| ConvNeXt-T DINOv3 | 298 | 466 |
| ViT-S DINOv3 CoMer | 75 | 47 |

So CoMer is 4x slower in torch even with the fused Triton deformable attention and flash
attention, and 10x slower under TensorRT. This note explains where the time goes. It comes from
reading the code plus a FLOP and memory-traffic count on CPU (no GPU was available). **No GPU
timings were taken**, so the per-component time estimates below are speculation to be confirmed
with a profiler (see the last section).

## 1. Work per image, by component

Method: both detectors were built in plain PyTorch on CPU with random weights at 1x3x800x800, using
the real `CRADIOv4CoMer` / `CTIBlock` / `MSDeformAttn` code (mmcv stubbed out, timm models
`vit_small_patch16_dinov3` and `convnext_tiny.dinov3_lvd1689m`) and a layer-for-layer mirror of
`CSPNeXtPAFPN` and `RotatedRTMDetSepBNHead` from `rtmdet-p2_dinov3_6e.yaml`.
GFLOP comes from `torch.utils.flop_counter` (matmul / conv / attention only). "Elementwise" counts
every kernel that is not a matmul, conv or view (LayerNorm, GELU, add, cat, interpolate,
grid_sample, ...) and the bytes it reads and writes, assuming fp16.

| model | component | GFLOP | # elementwise ops | elementwise GB |
|---|---|---:|---:|---:|
| ConvNeXt-T | backbone | 113.6 | 76 | 0.86 |
| | neck `[96, 192, 384, 768]` | 45.0 | 153 | 0.43 |
| | head | 62.9 | 48 | 0.22 |
| | **total** | **221.5** | **277** | **1.52** |
| ViT-S CoMer | ViT-S blocks (12) | 222.0 | 240 | 1.03 |
| | SPM CNN stem | 31.4 | 19 | 0.29 |
| | MRFP (x4) | 15.8 | 56 | 0.16 |
| | CTI_toV (x4) | 47.5 | 96 | 3.19 * |
| | CTI_toC (x4) | 47.5 | 196 | 3.40 * |
| | extra Extractor_CTI (x4) | 55.3 | 212 | 3.63 * |
| | output glue (ConvTranspose, add_vit_feature, BN) | 13.3 | 190 | 0.45 |
| | neck `[384, 384, 384, 384]` | 192.8 | 153 | 1.04 |
| | head | 62.9 | 48 | 0.22 |
| | **total** | **688.5** | **1210** | **13.41** |

\* These include the PyTorch `grid_sample` deformable attention path (8.0 GB for the 12
`MSDeformAttn` calls). The torch run with Triton keeps those intermediates in registers. The
TensorRT engine materialises them (section 4).

CoMer does **3.1x the FLOPs** and **~4x the non-matmul kernels** of ConvNeXt-T. Even without the
`grid_sample` intermediates it moves **~3.5x the elementwise bytes** (5.4 vs 1.5 GB per image).
The FLOP ratio alone explains most of the 4x torch gap. The rest is memory-bound adapter work.

## 2. The four reasons, largest first

### 2.1 ViT-S at 800x800 costs as much as the whole ConvNeXt detector

At 800x800 the ViT runs on 50x50 = 2500 tokens (+5 prefix tokens) in all 12 blocks. That is
222 GFLOP, the same as ConvNeXt-T backbone, neck and head combined. More than half of it
(~116 GFLOP) is the quadratic `QK^T` / `AV` attention. ConvNeXt is hierarchical: it works at
stride 4 with only 96 channels and keeps its widest stages at strides 16 and 32. A plain ViT keeps
384 channels at stride 16 for all 12 blocks, and each token attends to all 2500. Its width of 384
also makes the matmuls small (K = 384), which lowers tensor-core efficiency compared with
cuDNN's convolutions.

So a bare ViT-S at this resolution would already be about 2x slower than ConvNeXt-T. The Triton
kernels can't change that, because they only replace the deformable attention.

### 2.2 The neck runs at 384 channels on the 200x200 P2 map (~150 GFLOP extra)

`CRADIOv4CoMer` outputs `embed_dim` (384) channels at **every** level, so the config uses
`neck.in_channels: [384, 384, 384, 384]`. ConvNeXt-T gives `[96, 192, 384, 768]`. In
`CSPNeXtPAFPN` the P2 layers are sized by `in_channels[0]`:

- `top_down_blocks[-1]` = `CSPLayer(768 -> 384)` at 200x200,
- `downsamples[0]` = 3x3 stride-2 conv 384 -> 384 on the 200x200 map,
- `out_convs[0]` = 3x3 conv 384 -> 128 at 200x200.

The conv cost grows with the square of the channel count, so 4x the channels means about 16x the
FLOPs at P2. The neck goes from 45 to **193 GFLOP**, nearly the size of the ConvNeXt-T
backbone, and roughly two thirds of that is at P2. The 384-channel P2 map (8 x 384 x 200 x 200)
also goes through ConvTranspose, the 4x bilinear upsample of the ViT map, the add and the
BatchNorm in the backbone's output glue. Each of those passes is ~250 MB in fp16 at batch 8.

This cost has nothing to do with deformable attention, and it is the easiest one to remove
(section 5).

### 2.3 The adapter is ~200 GFLOP of memory-bound work on 13125 tokens

The CoMer branch works on `c` = stride 8/16/32 tokens = 10000 + 2500 + 625 = **13125 tokens**,
5.25x as many as the ViT. It runs **12 `MSDeformAttn` calls** (4 `CTI_toV`, 4 `CTI_toC`
`cfinter`, 4 extra `Extractor_CTI`) and each of them uses all 13125 tokens as queries:

- `CTI_toV` (`comer_modules.py:367`) uses the **CNN tokens as queries** (`query_norm(feat)`). It
  computes deformable attention and a ConvFFN for 13125 queries, then reduces them to the 2500 ViT
  tokens by bilinear down/upsampling and a sum. Upstream ViT-CoMer works the same way, but about
  80% of that work is averaged away.
- Each MSDA has `value_proj` and `output_proj` (384x384, `deform_ratio: 1.0`) plus offset and
  weight projections. That adds up to ~119 GFLOP over the 12 calls.

The FLOPs are moderate. What the counts show is that the adapter consists of **hundreds of small
memory-bound kernels**, about 700 non-matmul ops in the CTI modules against 76 in the whole
ConvNeXt-T backbone:

- **Four LayerNorms per CTI interaction on the same tensor.** `CTI_toC._interact` calls
  `self.query_norm(query)` and `self.feat_norm(query)`, then `MultiscaleExtractor` normalises both
  again (`self.query_norm(query)`, `self.feat_norm(feat)`). With `ffn_norm` and the extractors'
  norms that makes ~1.2 GB of LayerNorm traffic per image. Under AMP, `layer_norm` produces
  **fp32** outputs, which then get cast back to fp16 for the next linear layer, an extra
  read-write each time.
- **Layout shuffles.** Every `DWConv` / `MultiDWConv` (`_split_levels`) goes token -> NCHW
  (`transpose().reshape()`, a copy), runs three depthwise convs, then NCHW -> token
  (`flatten().transpose()`, another copy) and `torch.cat`. `MRFP` additionally splits channels in
  half and concatenates them. `CTIBlock.forward` and `_add_vit` re-slice and `cat` `c` (13125x384)
  at every stage.
- GELU, BatchNorm, DropPath (identity in eval), residual adds and the `gamma`-gated sum are all
  separate kernels in eager mode.

At batch 8 one pass over `c` in fp16 is ~80 MB, around 50 µs at ~1.6-1.8 TB/s. A few hundred such
passes come to **roughly 20-35 ms of the ~107 ms batch** (estimate). The deformable attention
gather is also latency-bound: each call gathers ~4 GB of corner values at batch 8 (13125 queries x
6 heads x 12 samples x 4 corners x 64 ch). That mostly hits L2, but with 12 calls it is plausibly
another ~5-10 ms. The Triton kernel makes the gather itself efficient. It does nothing about the
glue around it.

### 2.4 The whole thing is one long serial chain

ConvNeXt-T is 18 blocks of large, cuDNN-tuned convolutions. CoMer is strictly sequential
(`MRFP -> CTI_toV -> 3 ViT blocks -> CTI_toC -> ...`, with 4 extra CTIs at the end). Each step
depends on the previous one, so nothing runs concurrently, and on the order of 2000 kernels launch
per forward. Python and dispatcher overhead (`run_blocks` closures, `deform_inputs_only_one`
rebuilding reference points with `meshgrid` / `linspace` on every CTI call, `cat` of the prefix
tokens per stage) add a few ms on top in eager torch.

### Rough time budget (torch fp16 AMP, batch 8, speculative)

| | ConvNeXt-T (27 ms measured) | ViT-S CoMer (107 ms measured) |
|---|---|---|
| matmul / conv FLOPs | 1.8 TFLOP, ~12-15 ms | 5.5 TFLOP, ~35-45 ms (ViT ~12, neck ~10, adapter ~10, head ~4) |
| elementwise / layout | ~12 GB, ~7-10 ms | ~45-60 GB, ~25-35 ms |
| MSDA gathers | - | ~5-10 ms |
| launch / Python | small | ~5-10 ms |

## 3. Why the Triton kernels don't close the gap

The fused Triton MSDA (`msda_triton_fused.py`) speeds up **one op**, the sampling plus softmax plus
location arithmetic, which is perhaps 10% of CoMer's forward. Everything in sections 2.1, 2.2 and
most of 2.3 lies outside it: the ViT's quadratic attention, the 384-channel P2 neck, the
projections, the LayerNorms and the layout copies. Flash attention already makes the ViT attention
as fast as it can be in torch. The ViT just has a lot of it.

## 4. Why TensorRT makes CoMer *slower* (75 -> 47 images / s)

Torch fp16 takes ~107 ms per batch and TRT fp16 ~170 ms. The ~63 ms difference matches the two
export patches in `export_patches.py`:

1. **Explicit fp32 attention** (`_sdpa_fp32_logits`): `(q.float() * scale) @ k.float()^T`. The
   matmul runs in **fp32 without tensor cores**, and the logits are materialised. Per block at
   batch 8 that is 8 x 6 x 2505^2 x 4 B = **1.2 GB of logits**, written, read by softmax, written
   again and read by `@ v`, ~4-5 GB per block and ~50-60 GB over 12 blocks. That is **~30 ms** of
   pure memory traffic, plus ~460 GFLOP in fp32 SIMT. Torch's flash attention keeps the logits in
   SRAM with fp32 accumulation, so it never writes them.
2. **`grid_sample` deformable attention in fp32.** `multi_scale_deformable_attn`'s fallback
   calls `msda_pytorch(value.float(), ..., sampling_locations.float(), attention_weights.float())`,
   so this whole path is exported as **fp32 even in the fp16 engine**. Per level and call it
   permutes `value` into `(B*heads, C, h, w)` (a copy), writes a `(B*heads, C, 13125, 4)`
   grid_sample output (~0.65 GB fp32 per level at batch 8), then stacks it and multiplies-and-sums
   it with the weights. Across 3 levels and 12 calls that is **tens of GB, ~20-35 ms**, against
   one fused Triton kernel in torch.

TensorRT's gain on the rest of the graph (fused LayerNorm and elementwise ops, conv / BN fusion,
the neck) is real, the same 1.6x it gives ConvNeXt-T, but it is smaller than these two
regressions.

## 5. What to try (most impact for least risk first)

1. **Profile first** to confirm the budget above:
   - torch: `torch.profiler` with `record_function` ranges around `spm`, each `CTIBlock`
     part, `run_blocks`, neck and head, or `nsys profile --trace=cuda,nvtx python tools/tensorrt/bench.py ...`.
   - TRT: `trtexec --loadEngine=end2end.engine --dumpProfile --separateProfileRun --dumpLayerInfo`
     shows which layers (attention MatMul / Softmax, GridSample) dominate.
2. **Shrink the neck input** (retraining needed, new module / config only): project the CoMer
   outputs with a 1x1 conv (or mmdet's `ChannelMapper`) to e.g. `[96, 192, 384, 384]` or
   `[128, 128, 256, 384]` before `CSPNeXtPAFPN`. That saves up to ~150 GFLOP per image (~22% of the
   total) plus the 384-channel P2 traffic, and applies to torch and TRT alike. A cheaper variant:
   build the stride-4 output (`up`, `norm1`, `fc1`) at fewer channels.
3. **TRT attention:** have TensorRT run fused MHA instead of the explicit fp32 form. Keep q / k in
   fp16 or bf16 and let the fused kernel accumulate in fp32, as flash attention does in torch. The
   overflow in `TENSORRT.md` comes from fp16 *logits*, not from fp16 inputs (bf16 engines are
   within 0.1 mAP). Options: export the plain `MatMul -> Softmax -> MatMul` pattern in bf16 so TRT's
   MHA fusion recognises it, or use TensorRT's native attention layer if your TRT version has one.
   Whether a given build fuses can be checked in `--dumpLayerInfo`.
4. **TRT deformable attention:** add an MSDA TensorRT plugin (mmdeploy has a
   `MMCVMultiScaleDeformableAttention` plugin, and the Triton fused kernel's logic could be
   ported). At minimum, export the `grid_sample` path in fp16 instead of the hard-coded
   `.float()`, which halves its traffic, and check the accuracy with `--verify`.
5. **torch path:** `torch.compile` the adapter (`interactions`, `spm`, output glue). Inductor would
   fuse the duplicated LayerNorms with their casts, the cat / split / transpose copies, GELU / BN
   and the residual adds, and the Triton MSDA can stay as an opaque custom op. CUDA graphs
   (`mode='reduce-overhead'`) would also remove the ~2000-launch overhead at fixed shapes.
6. **Architecture changes** (retraining needed): use the 2500 ViT tokens as `CTI_toV` queries
   instead of all 13125 CNN tokens, drop or reduce the 4 extra CTIs (`extra_num`), or set
   `deform_ratio: 0.5` (as the C-RADIO configs do) to halve `value_proj` / `output_proj`. Section 1
   shows each of these as a share of the adapter's ~200 GFLOP.
