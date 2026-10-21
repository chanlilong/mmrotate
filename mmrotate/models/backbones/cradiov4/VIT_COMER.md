# How ViT-CoMer gets stride 4–32 feature maps out of a stride-16 ViT

This explains `CRADIOv4CoMer` (`vit_comer.py`) and its building blocks (`comer_modules.py`), with
small-object detection in mind. Shapes assume a 1024×1024 DOTA crop and C-RADIOv4-H
(`embed_dim = 1280`, 32 blocks, patch 16).

---

## 1. The problem: a ViT only has one resolution

A plain ViT cuts the image into 16×16 patches and keeps **one token per patch for every layer**.
A 1024² image becomes a 64×64 grid of tokens, all at stride 16. Nothing gets finer or coarser
as you go deeper.

A detector wants a **pyramid** (stride 4/8/16/32), and small objects need the fine levels:

```
  small vehicle in DOTA ≈ 10–20 px

  stride 16 token grid           stride 4 grid
  ┌────┬────┐                    ┌─┬─┬─┬─┬─┬─┬─┬─┐
  │ ▪  │    │   the car is       │ │ │ │ │ │ │ │ │
  │    │    │   ~1 token, mixed  ├─┼─┼─┼─┼─┼─┼─┼─┤   the car spans ~4×4 cells,
  ├────┼────┤   with road,       │ │▪│▪│ │ │ │ │ │   it has shape and edges, and
  │    │    │   shadow, other    ├─┼─┼─┼─┼─┼─┼─┼─┤   two cars parked side by side
  │    │    │   cars…            │ │▪│▪│ │ │ │ │ │   land in different cells
  └────┴────┘                    └─┴─┴─┴─┴─┴─┴─┴─┘
```

There are two common fixes:

| Approach | What it does | Used in |
|---|---|---|
| **Simple pyramid** (ViTDet-style) | Interpolate / pool the stride-16 map to 8, 32, … | `CRADIOv4` (`cradiov4.py`) |
| **Adapter** (ViT-Adapter, **ViT-CoMer**) | Run a small **CNN** in parallel that natively produces 8/16/32 maps, and let it trade information with the ViT through attention | `CRADIOv4CoMer` |

The simple pyramid can't recover detail that was lost inside a 16×16 patch: upsampling a blurry
map gives you a bigger blurry map. The adapter adds a second path that **never lost the detail in
the first place**.

---

## 2. The intuition in one sentence

> **The CNN supplies *where* (high-res spatial detail), the frozen ViT supplies *what*
> (large-model semantics), and deformable attention keeps the two in sync at every stage.**

Picture two people working on the same map:

- the **CNN** is a surveyor with a magnifying glass who sees every edge but doesn't know what
  anything is;
- the **ViT** is an expert who has seen billions of images and recognizes things, but squints at
  everything at 1/16 resolution.

Every 8 ViT layers they compare notes. The surveyor tells the expert "there are sharp edges here"
(CNN → ViT). The expert tells the surveyor "this blob is a vehicle, that one is a ship"
(ViT → CNN). At the end you read off the surveyor's high-res map, which is now annotated with the
expert's knowledge.

---

## 3. The full picture

```
 image 1024×1024
   │
   ├───────────────────────────────┐
   │                               │
   ▼                               ▼
 ┌────────────────────┐   ┌─────────────────────────────────────┐
 │ CNN branch ("SPM") │   │ C-RADIOv4-H ViT (frozen)            │
 │ spm = CNN(...)     │   │ patch_embed → 64×64 tokens, 1280-d  │
 │                    │   └─────────────────────────────────────┘
 │ c1 256×256 (s4)  ──┼─────────────────────────┐ (sits out the interaction)
 │ c2 128×128 (s8)  ┐ │                         │
 │ c3  64×64  (s16) ├─┼─ flatten+concat = "c"   │
 │ c4  32×32  (s32) ┘ │   21·1024 tokens        │
 └────────────────────┘                         │
                                                │
     c (CNN pyramid tokens)        x (ViT tokens, s16)
          │                              │
   ┌──────┴──────────── stage 1 ─────────┴───────────┐
   │  MRFP(c)          multi-kernel DW convs on c    │
   │  CTI_toV:  c ──deformable attn──► x += γ·Δ      │  CNN → ViT
   │  ViT blocks 0–7 on x   (frozen weights)         │
   │  CTI_toC:  x ──► added into c, then multi-scale │  ViT → CNN
   │            deformable self-attn over c          │
   └──────┬──────────────────────────────────┬───────┘
          │          stage 2: blocks 8–15     │
          │          stage 3: blocks 16–23    │
          │          stage 4: blocks 24–31    │
          │            + 4 extra CTI_toC      │
          ▼                                   ▼
     c → split back into c2, c3, c4      x (64×64)
          │                                   │
          │   c1 = ConvTranspose(c2) + c1     │
          │                                   │
          └──────── + bilinear-resized x ◄────┘   (add_vit_feature)
                         │
              SyncBN per level (norm1..norm4)
                         │
       s4 256² · s8 128² · s16 64² · s32 32²   (1280 ch each)
                         │
               CSPNeXtPAFPN → RTMDet head
```

Code: `CRADIOv4CoMer.forward` at `vit_comer.py:201`.

---

## 4. Step by step

### 4.1 CNN branch = spatial prior module (`CNN`, `comer_modules.py:559`)

A small ResNet-style stem:

```
3 → 64  conv s2      (512²)
64 → 64 conv ×2
maxpool s2           → c1  256²  stride 4,  64 ch
conv s2              → c2  128²  stride 8,  128 ch
conv s2              → c3   64²  stride 16, 256 ch
conv s2              → c4   32²  stride 32, 256 ch
```

Each level gets a 1×1 conv (`fc1..fc4`) to project it up to the ViT width (1280) so CNN and ViT
tokens live in the same space and can attend to each other. A learned `level_embed` is added to
c2/c3/c4 so attention can tell which scale a token came from.

c2, c3, c4 are flattened and concatenated into one long token sequence `c`:

```
c = [ c2: 128·128 = 16384 | c3: 64·64 = 4096 | c4: 32·32 = 1024 ]  = 21504 tokens
     \______ 16n ______/   \____ 4n ____/      \___ n ___/          n = H·W/4 = 1024
```

That's why the code keeps slicing at `16*n`, `20*n` with `n = N // 21` (`_split_levels`,
`CTI_toC._add_vit`).

This CNN is randomly initialised and trained from scratch. It's tiny next to ViT-H, but at stride
4/8 it can still see edges and corners that a 16×16 patch embedding averages away.

### 4.2 Why deformable attention?

Full attention between 21504 CNN tokens and 4096 ViT tokens would cost about 88M pairs per head per
layer. **Multi-scale deformable attention** (`MSDeformAttn`, from Deformable DETR) does this
instead:

```
for each query token q at reference point p:
    for each head (16) and each pyramid level (3):
        predict 4 offsets Δ from q              ← "where should I look?"
        bilinearly sample the value map at p+Δ
        weight by predicted attention weights
```

Each query only looks at `16 heads × 3 levels × 4 points` locations. That's linear in the number
of tokens, it works across scales, and the sampling points can drift toward wherever the useful
information is (for a small object, the object itself).

### 4.3 MRFP: multi-receptive-field feature pyramid (`MRFP`, `comer_modules.py:149`)

Before each interaction, `c` passes through a ConvFFN whose depthwise conv uses **3×3 on half the
channels and 5×5 on the other half**, separately per level. It's cheap local context mixing that
gives the CNN tokens several receptive fields. This is CoMer's main addition over ViT-Adapter.

### 4.4 CNN → ViT: `CTI_toV` (`comer_modules.py:367`)

```python
c = cat([c2, c3 + x, c4])                    # put ViT tokens into the s16 slot
c' = MSDeformAttn(query=c, value=c)          # multi-scale self-attn over the pyramid
c' = c' + ConvFFN(c')
s8  → bilinear ↓2  → 64×64
s16 →              → 64×64
s32 → bilinear ↑2  → 64×64
x = x + γ · (s8 + s16 + s32)                 # γ is initialised to 0
```

All three scales are squeezed down to the ViT's 64×64 grid and **added to the ViT tokens before
they go into the next 8 frozen blocks**. Fine detail from the stride-8 CNN level gets compressed
into the token, so the ViT's attention can reason about it.

`γ = 0` at init means that on step 0 the ViT sees exactly its pretrained input, so you start from
intact RADIO features. The adapter then learns how much to inject.

> **Frozen doesn't mean bypassed.** `freeze_vit=True` stops the ViT *weights* from updating, but
> gradients still flow *through* the ViT blocks back into `CTI_toV` and the CNN. The adapter
> learns to "prompt" the frozen ViT with spatial hints.

### 4.5 The ViT blocks (`run_blocks`, `vit_comer.py:222`)

Blocks `[start, end]` of RADIO run as usual on the (now slightly modified) tokens. RADIO's
cls + register tokens (`prefix`) are re-attached for the blocks and split off again afterwards,
so CoMer only ever sees the 64×64 spatial tokens.

### 4.6 ViT → CNN: `CTI_toC` (`comer_modules.py:227`)

```python
c = cat([c2, c3 + x, c4])                         # inject ViT semantics at s16
c = c + MSDeformAttn(query=c, value=c, 3 levels)  # cnn_feature_interaction
c = c + ConvFFN(c)
```

The ViT tokens are added **only into the stride-16 slot** (same grid, one-to-one). Then a
multi-scale deformable self-attention across the 3 CNN levels spreads that semantic signal: a
stride-8 token at some location samples the now semantically rich stride-16 map around itself
(and stride 32 for context). In effect it's a learned, content-aware upsampling of ViT semantics
onto the high-res CNN grid.

The last stage repeats this with 4 extra `Extractor_CTI` blocks (`use_extra_CTI`, `extra_num=4`),
since the final ViT layers carry the richest semantics.

### 4.7 Assembling the outputs (`vit_comer.py:229`)

```python
c2, c3, c4 = split(c)                          # 128², 64², 32²
c1 = ConvTranspose2d(c2) + c1                  # stride 4  (only if 0 in out_indices)

# add_vit_feature: also add the final ViT map directly, resized to each level
x3 = x (64²)
c1 += upsample(x3, ×4)
c2 += upsample(x3, ×2)
c3 += x3
c4 += downsample(x3, ×½)

outputs = SyncBN(c_i) for i in out_indices
```

Each output level is **CNN detail refined by attention, plus a skip connection of the final ViT
features**. The ViT skip guarantees every level carries the pretrained semantics, and the CNN path
adds what the ViT can't resolve.

---

## 5. Shapes at a glance (1024² input, ViT-H)

| Tensor | Grid | Stride | Tokens | Channels |
|---|---|---|---|---|
| ViT `x` | 64×64 | 16 | 4 096 | 1280 |
| CNN `c1` (stem) | 256×256 | 4 | 65 536 (never tokenised) | 64 → 1280 |
| CNN `c2` | 128×128 | 8 | 16 384 | 1280 |
| CNN `c3` | 64×64 | 16 | 4 096 | 1280 |
| CNN `c4` | 32×32 | 32 | 1 024 | 1280 |
| `c` (concat c2..c4) | – | – | 21 504 | 1280 |

`out_indices`: `0` → s4, `1` → s8, `2` → s16, `3` → s32.
`cradiov4-h-comer_rtmdet-3x-dota_ms.yaml` uses `[1, 2, 3]`; the `-p2-` config uses `[0, 1, 2, 3]`.

---

## 6. What this means for small objects (read before relying on P2)

**Stride 8 is the level that actually benefits from the interaction.** c2 takes part in every
CTI_toV / CTI_toC / MRFP step, so it's a well-fused map that combines CNN detail with ViT semantics.

**Stride 4 is much weaker than it looks.** c1 is **never part of `c`**, so it never goes through
deformable attention or MRFP. The stride-4 output is only:

```
c1_out = (stem conv features, 3 conv layers deep, 1×1-projected)
       + ConvTranspose(c2)            ← learned 2× upsample of the fused s8 map
       + bilinear ×4 of final ViT     ← blurry
```

That's still useful (raw edges plus upsampled semantics, like the P2 level in an FPN), but the
only high-frequency information it adds over stride 8 comes from a very shallow CNN. This is how
upstream ViT-CoMer does it too, because putting 65k stride-4 tokens into `c` would roughly
quadruple the adapter's cost.

**Practical levers, roughly ordered by bang for buck:**

1. **Upsample the input.** This is the most direct way to give a patch-16 ViT more pixels per
   object. A 2× upscaled crop makes the ViT effectively stride 8 relative to the original image,
   and every CoMer level moves down a stride with it. RADIO handles variable resolutions well. It
   costs about 4× compute.
2. **Try s8 + interaction first, then P2.** Compare `[1,2,3]` with `[0,1,2,3]` on your small
   classes before assuming P2 helps. With RTMDet's `DynamicSoftLabelAssigner`, a 10-px object at
   stride 8 still gets candidate points, just fewer of them.
3. **The P2 level is expensive at 1280 channels.** 256²×1280 in bf16 is about 168 MB per image
   just for that map, and `CSPNeXtPAFPN` then processes it at full width. If memory is tight,
   projecting backbone outputs to 256–512 channels before the neck costs little accuracy.
4. **If you want a real fused stride-4 level:** add c1 to the interaction, but only in the last
   stage or the extra CTIs, so the cost stays manageable. Do it as a subclass in a new file, per
   the repo's merge rules.
5. **`init_values` / `γ`:** keep it at 0 with a frozen ViT. It protects the pretrained features
   early in training.

---

## 7. Mental model recap

```
                 semantics (what)                    detail (where)
 frozen ViT  ──────────────────────►  CTI_toC  ──────────────────────►  CNN pyramid
   s16 only   ◄──────────────────────  CTI_toV  ◄──────────────────────  s8 / s16 / s32
                 spatial hints, γ-gated
                      × 4 stages (every 8 ViT blocks)

 output_level_k = CNN_k (attention-refined) + resize(final ViT, stride k)
 stride 4      = shallow CNN stem + upsampled s8 + upsampled ViT   (no attention)
```

The large pretrained model never has to work at high resolution. The adapter grows a
high-resolution "shadow" of it, keeps the two talking, and the detector reads the shadow.
