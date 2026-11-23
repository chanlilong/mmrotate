# mmrotate REST API for qgis_annotations

Serves an mmrotate **single-stage** detector (RTMDet and friends: the `work_dirs/soda_dinov3`
models, the stock `configs/rotated_rtmdet`, Retina-style heads, ...) over the two REST
contracts of qgis_annotations:

| Service | Endpoints | Contract | Registered in the admin as |
|---|---|---|---|
| Detector | `GET /detector/health`, `POST /detector/detect` | `docs/DETECTOR_API.md` | *Setup → AI services*, kind **fixed classes**, address `http://<host>:8200/detector` |
| Similarity | `GET /similarity/health`, `POST /similarity/embed` | `docs/SIMILARITY_API.md` | kind **Similarity features**, address `http://<host>:8200/similarity` |

One process loads the model once and serves both, so it sits in GPU memory once. When only
one service is enabled (`--services detector` or `--services similarity`) it is also served
at the root (`/health`, `/detect` or `/embed`), and the bare `http://<host>:8200` works as the
address. `GET /` lists what an instance serves. Swagger UI is at `/docs` and ReDoc at
`/redoc`. Both are **served from `api/static`**, so they work on an airgapped network.

## Quick start

You need the training config and the checkpoint, nothing else: the backbone's pretrained
weights are never downloaded, because everything comes from the checkpoint.

```bash
# in the repo's one image, docker/Dockerfile (or any env with this repo on PYTHONPATH plus
# docker/requirements/requirements-api.txt)
python -m api --config work_dirs/soda_dinov3/<run>/<run>.yaml \
              --weights work_dirs/soda_dinov3/<run>/epoch_6.pth \
              --gsd 0.5 --model-name soda-vitdet
```

Docker: the repo has one image, `voltronix-mmrotate-dev` (`docker/Dockerfile`), which includes
the API's packages from tag 2026.11.23. The service runs from the mounted repo, and `api/`
carries the Swagger UI / ReDoc assets, so nothing is fetched (airgapped-friendly):

```bash
MODEL_DIR=/path/to/model GSD_M=0.5 \
    docker compose -f docker/docker-compose.yaml --profile api up -d mmrotate-api
```

The `mmrotate-api` compose service is driven by `MMROTATE_API_*` environment variables.
`docker/README.md` has the equivalent `docker run` line, and the `docker save` / `docker load`
steps for carrying the image (a `.tar.gz` in `docker/`) to the airgapped side.
Every flag below has an environment form: `--detector-tile-size` is
`MMROTATE_API_DETECTOR_TILE_SIZE`, and a flag on the command line wins over the environment.

Start-up loads the model and runs a **self-test** (a grey and a random tile through each
service) before listening. A config and checkpoint that don't fit therefore fail at
`docker run`, not on an annotator's click. The class list comes from the checkpoint's
`meta.dataset_meta` (else the config, else the dataset class) and must match the head's
`num_classes`.

## GSD: feeding the model the scale it was trained at

`--gsd` is the metres per pixel the model was **trained** at.

- **Detector.** `/health` reports it as `gsd_m`. The qgis server then **resamples the
  imagery to that GSD before tiling** (when *Always run at this service's trained GSD* is
  ticked, the default). The service receives tiles already at the right scale and never
  rescales them.
- **Similarity.** The similarity contract carries no GSD: tiles arrive in the imagery's own
  source pixels.
  - Set `--imagery-gsd` to the campaign imagery's GSD. The service then resamples every tile
    by `scale = imagery_gsd / gsd` inside, exactly like the C-RADIO reference.
  - The scale is snapped so that `out_stride` is a whole number of source pixels; the snapped
    value is logged and reported in `/health`.
  - Without `--imagery-gsd` the scale is 1.

## Image size: options, auto-detected, validated

Nothing assumes SODA-A's 800 px.

| Flag | Default | Meaning |
|---|---|---|
| `--input-size N` or `WxH` | from the test pipeline (`Pad.size`, else `Resize.scale`; for a rectangular keep-ratio scale such as `(1333, 800)`, its shorter side), else 1024 | Window the network sees, in model px. Must be a multiple of `pad_size_divisor` (checked at start). |
| `--resize-mode` | `none` | `none` feeds tiles at their own pixels: the config's `Resize`/`Pad` are dropped, because a keep-ratio `Resize` would silently enlarge a short tile and change its GSD. `config` keeps them, for models trained on resized whole images. |
| `--pad-mode` | `square` | `square` zero-pads every window to the input size, which is the context the model trained on and is required by fixed-shape ViTs. `divisor` pads only to `pad_size_divisor`, which is faster on short tiles. |
| `--detector-tile-size` | input size | `native_size` in the detector's `/health`, i.e. the tile the client sends. A larger tile is cut into overlapping windows of the input size (`--patch-overlap`, 128 px); their boxes are merged with the model's own rotated NMS at the request's `nms_iou`. This cuts HTTP round trips on big GPUs. |
| `--similarity-tile-size` | input size / scale | `native_size` of the similarity service (source px, snapped to `out_stride`). Larger tiles run as windows whose core cells are stitched on the stride grid. |
| `--max-batch`, `--max-side-px` | 8, 2 × tile | Tiles per `/detect` request; the largest tile side accepted. |
| `--precision` | `fp32` | `fp16` / `bf16` autocast. |

## Similarity features

The reference C-RADIOv4 service embeds a ViT's last layer: L2, then PCA to 128-d, then L2
again, then int8. A detector has more to offer:

- a multi-scale **backbone**;
- a **neck** pyramid that already mixes the scales;
- a **head** whose cls and reg towers were trained to separate exactly the classes of
  interest.

`FeatureSpec` (`api/features.py`) covers all of these:

| Option (`--sim-*` flag) | Values | Base (with `--sim-*` flags) |
|---|---|---|
| `source` | `backbone`, `neck`, `head_cls`, `head_reg` (the final tower features, i.e. the predictors' input), `head_cls@I` / `head_reg@I` (the I-th stacked conv), or several joined with `+` | `neck` |
| `levels` | pyramid indices, e.g. `0,1`, or `all` | all |
| `fusion` | `concat`, `mean` (equal channels only), `single` | `mean` |
| `level_norm` | `l2` (each level unit length before fusing, so no level dominates by magnitude), `none` | `l2` |
| `out_stride` | grid stride in model px | 8 |
| `projection` | `none`, `center`, `pca` (center and PCA come from a profile, see below) | `none` |
| `dtype` | `int8` (qgis_annotations reads int8 only), `float16`, `float32` | `int8` |

**Default with no profile and no `--sim-*` flag:**

- `head_cls`, `single`, level 0, stride 8, no projection, int8: the head's classification
  features at the finest level.
- This choice won clearly on every model measured (table below).
- A model whose head has no recognised classification predictor falls back to the base,
  the mean of the neck levels.
- When you give `--sim-*` flags without a profile, they apply on top of the base column.

How fusion works:

- Every selected level is resampled to `out_stride`: area pooling for finer levels, bilinear
  interpolation for coarser ones.
- The levels are then concatenated or averaged.
- Then: L2 normalisation, an optional projection, a final L2 normalisation, then the wire
  dtype.

Every vector is unit length, as the brush and the trained classes require.

`feature_space` and `model_version` are derived from everything that changes the vectors:

- the weights digest;
- the spec;
- the PCA digest;
- the input size and padding;
- the stride, dimension and scale;
- int8.

So two instances report the same space only when their vectors are interchangeable. Override
them with `--feature-space` / `--model-version` only if you know they are.

### Multi-scale: interpolate, stack, PCA?

What the measurements say (`suggest_api_hyperparameters.py`, 200 SODA-A test patches,
7930 objects, centred features at stride 8):

| Model | head_cls, finest level | head_cls, concat all | backbone, concat all | neck, concat all | neck, mean |
|---|---|---|---|---|---|
| ViT-S CoMer | **0.943** | 0.912 | 0.847 | 0.739 | 0.689 |
| ViT-S ViTDet | **0.904** | 0.903 | 0.587 | 0.741 | 0.707 |
| ConvNeXt-T DCFL | **0.920** | 0.795 | 0.688 | 0.623 | 0.570 |

Score is the mean of obj AP (the brush) and ridge AP (trained classes). In stage B and C
on all three:

- stride 4 adds +0.001 to +0.015;
- PCA to 64-d costs at most 0.002;
- whitening hurts the brush (-0.03 to -0.07 obj AP) and barely moves ridge AP;
- centring beats no projection by only 0.001-0.004 on the head features, so the default
  (no profile) loses little;
- int8 is free (top-5 % IoU 0.96 to 0.97).

So:

- **Interpolating the scales to one stride and stacking them is supported** (`fusion=concat`)
  and is how backbone features should be used. Averaging levels with different meaning does
  worse, and PCA brings the 1000-1500-d stacks back to a sane size.
- **But here the stack never beat the single finest level of the head.** On these P2
  models, level 0 (stride 4) is where the small objects live, and the head's cls tower has
  already been trained to separate exactly these classes.
- **The neck is the weakest source** despite PAFPN's own fusion. Its features are tuned for
  both heads at once and are far less class-specific (same-class cosine 0.4-0.5 against
  0.88 for the head).
- **Caveat.** The benchmark scores retrieval of the model's own training classes, which
  favours head features. For classes the model never learned, the backbone may generalise
  better. If annotators paint such classes, consider serving a backbone profile, e.g.
  `--sim-source backbone --sim-fusion concat` with a PCA profile, from a second instance.
- **Stride.** 8 is the default: stride 4 gains at most 0.015 for 4 times the grid size.

Run the suggest tool for every new model rather than relying on these numbers.

### Choosing the settings: `tools/similarity/suggest_api_hyperparameters.py`

```bash
python -m tools.similarity.suggest_api_hyperparameters CONFIG WEIGHTS [--split test|val] \
    [--num-images 200] [--strides 4 8 16] [--dims 64 128 256] [--gsd 0.5] [--out DIR]
```

It runs the model over a sample of the config's own test (or val) data, exactly as the
service would (same `api/` code, input size and padding), and scores candidates in three
stages:

- **A, source:** every source × {concat of all levels, mean of all levels, each single
  level}, centred.
- **B, stride:** the best of A × `--strides`.
- **C, compression:** the best of B × {none, center, PCA to each of `--dims`, whitened PCA}
  × {float32, int8}.

The scores are the qgis_annotations similarity measures:

- **obj AP:** the brush. Every object queries other objects of its class against other
  classes plus background, under cosine. It is also given per object size.
- **ridge AP:** trained classes. One-vs-rest ridge, λ = 100, fitted on half of the images
  and scored on the other half.
- **bg_paircos:** anisotropy. Centring brings it to about 0.
- **same_cos:** the plugin's Nyström γ = 2 expects about 0.76–0.90.
- **int8 cosine / top-5 % IoU:** what quantisation costs.
- **False alarms per image (FPPI):** each class's score map over whole held-out images,
  thresholded as an annotator would, for both the brush (cosine to the class mean of the
  fit half's objects) and the trained classes (the ridge model).
  - At a threshold, an object is *found* when any of its cells scores above it.
  - Every local maximum above the threshold (one per `--fa-peak-px`, 24 px) on the
    background or on another class's object is one *false alarm*.
  - Reported like `tools/soda_dinov3/false_alarms.py`:
    - `FA@R r`: false alarms per image needed to find a fraction r of the objects
      (`--fa-recalls 0.5 0.7`);
    - `R@FA N`: recall within N false alarms per image (`--fa-fppi 1 10`);
    - `map AP`: the score map's AP as a detector;
    - the share of false alarms that fall on other classes rather than background.
  - At most `--fa-max-peaks` (100) false alarms are kept per image and class; beyond that
    the report says `>cap`. Classes with fewer than `--fa-min-objects` (10) eval objects
    are left out of the FPPI means.
  - `--no-false-alarms` skips this pass.
- **Per class:** every AP, recall and false-alarm figure is also reported per class.
  - `suggestions.md` has a per-class table for the recommendation (brush and ridge AP, map
    AP, R@FA and FA@R), and the run prints it.
  - Every row in `suggestions.json` carries the same lists (`*_per_class`, in the config's
    class order).

The recommendation is the cheapest int8 grid within `--tolerance` (0.005) of the best score.
With `--rank-by fppi` it is the cheapest int8 grid within `--fa-tolerance` (10 %) of the
fewest false alarms per image at the first `--fa-recalls` point; stages A and B are then
ranked that way too.

Example, per class for the recommended CoMer setting (`head_cls`, finest level, stride 4,
PCA 64, int8; 100 eval images):

| class | objects | brush AP | brush FA@R0.7 | ridge AP | ridge FA@R0.7 |
|---|---|---|---|---|---|
| airplane | 63 | 0.928 | 0.03 | 0.974 | 0.14 |
| small-vehicle | 2475 | 0.953 | 1.00 | 0.978 | 3.06 |
| container | 579 | 0.828 | 5.23 | 0.935 | 3.51 |
| ship | 413 | 0.993 | 0.14 | 0.972 | 0.45 |
| swimming-pool | 133 | 0.991 | 0.05 | 1.000 | 0.23 |

Retrieval AP alone hides where false alarms will annoy annotators: small vehicles and
containers. The per-class FA columns show it.

The tool writes three files to `work_dirs/similarity/<config>/`:

- `suggestions.md`: every table;
- `suggestions.json`;
- `similarity_profile.npz`: the spec plus its fitted centring / PCA.

It also prints the command:

```bash
python -m api --config CONFIG --weights WEIGHTS --gsd 0.5 \
    --similarity-profile work_dirs/similarity/<config>/similarity_profile.npz
```

`--sim-*` flags still override single fields of a loaded profile. Projection is part of the
profile and needs it. The service warns when the profile was tuned at another input size or
padding.

GPU runs on this cluster go through slurm:

```bash
sbatch --nodelist=hammond api/slurm_run.sbatch \
    python -m tools.similarity.suggest_api_hyperparameters CONFIG WEIGHTS --gsd 0.5
```

## Checking a running service

```bash
curl -s localhost:8200/ | python -m json.tool               # what is served, and where
curl -s localhost:8200/detector/health | python -m json.tool
curl -s localhost:8200/similarity/health | python -m json.tool
```

The round-trip snippets in `DETECTOR_API.md` §7 and `SIMILARITY_API.md` §7 work against
`/detector` and `/similarity`. So do the server's own clients:

- `RemoteDetector('http://host:8200/detector')`;
- `RemoteSimilarity('http://host:8200/similarity')`.

Both build their URLs as `base + '/health'`.

## Layout

| Path | What |
|---|---|
| `api/app.py` | `create_app`: load, build the services, self-test, mount routers and offline docs |
| `api/model.py` | config + checkpoint loading, class check, ndarray pipeline, GPU lock, NMS override |
| `api/tiling.py` | input-size detection and validation, windows, padding |
| `api/detector.py` | `/health`, `/detect` |
| `api/similarity.py` | `/health`, `/embed` |
| `api/features.py` | feature sources, fusion, projection, quantisation |
| `api/profile.py` | similarity profile `.npz` |
| `api/docs.py`, `api/static/` | Swagger UI 5.17.14 and ReDoc 2.1.5, vendored (see each folder's `SOURCE.md`, licences kept) |
| `api/settings.py` | flags and `MMROTATE_API_*` environment |
| `tools/similarity/` | the suggest tool and its metrics |
| `tests/test_api/`, `tests/test_tools/test_suggest_api_hyperparameters.py` | tests |

Tests:

- CPU tests use a random-init `rotated_rtmdet_tiny` (DOTA, 1024 input).
- GPU tests use the SODA-A checkpoints in `work_dirs/soda_dinov3` and skip without CUDA or
  without the checkpoint.

```bash
sbatch --nodelist=hammond api/slurm_run.sbatch \
    python -m pytest tests/test_api tests/test_tools/test_suggest_api_hyperparameters.py -q
```

Lint (new code is ruff-linted with full annotations, see `CLAUDE.md`):

```bash
ruff check api --config api/ruff.toml && ruff format api --config api/ruff.toml
ruff check tools/similarity --config tools/similarity/ruff.toml
ruff check tests/test_api --config tests/test_api/ruff.toml
```

## Limits

- **Single-stage detectors only for the head sources.** Two-stage models should still serve
  `/detect` and the backbone and neck sources, but that is untested.
- **No GSD on the similarity wire.** `--imagery-gsd` is one value per service. Imagery at
  very different GSDs needs one instance per GSD, or a contract change in qgis_annotations.
- **Context dependence.** ViT backbones attend globally, so a cell's vector depends on the
  whole window. For the SODA ViTDet model, an aligned 480 × 400 crop agrees with the same
  cells of the full 800 px patch at a median cosine of 0.89 with the default features (0.86
  with `--pad-mode divisor`, 0.84 for the neck mean;
  `tests/test_api/test_similarity_api.py::test_vitdet_crop_alignment`). The qgis server
  overlaps its tiles by 128 px and keeps each tile's core, which keeps seams mild.
- **No TensorRT.** The service runs PyTorch; the TensorRT engines (`mmrotate/tensorrt`) cannot
  return intermediate features. A TRT detector backend is a possible follow-up.
