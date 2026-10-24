# MMRotate dev image (x86_64, CUDA 13.0, Python 3.12)

`docker/Dockerfile` builds a development image for an **x86_64 host with an NVIDIA driver >= 580**:

| component | version |
| --- | --- |
| base | `pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime` (Ubuntu 24.04, system Python 3.12, no conda) |
| mmcv | 2.1.0, compiled from source with CUDA ops (in a `-devel` builder stage) |
| mmdet / mmengine | 3.3.0 / latest `<1.0` |
| mmrotate | this repo, editable install at `/mmrotate` |

CUDA 13.0 is the newest toolkit that driver 580 supports natively, so the `+PTX` JIT fallback also
works. CUDA 13 supports Turing (sm_75) and newer only.

The build has two stages. The `builder` stage, on the `-devel` base (~22 GB, includes `nvcc`),
compiles mmcv into a wheel. The final image installs that wheel on the `-runtime` base (~6 GB), which
makes it about 16 GB smaller. The final image installs `build-essential`, because Triton (used by
some torch 2.14 ops, `torch.compile` and this repo's kernels) compiles its kernel launchers with the
host C compiler at run time. It has **no CUDA toolkit**: there's no `nvcc`, so CUDA extensions can't be
compiled inside the container. If you need that, switch the final `FROM` to the `-devel` base.

## Build

Always run from the repository root, which is the build context:

```bash
# on the x86 GPU host
docker compose -f docker/docker-compose.yaml build
# or
docker build -f docker/Dockerfile -t mmrotate-dev .

# on an Apple Silicon Mac (emulated amd64; slow, use a narrow arch list for a quick check).
# Enabling "Use Rosetta for x86_64/amd64 emulation" in Docker Desktop makes this much faster.
docker buildx build --platform linux/amd64 -f docker/Dockerfile \
    --build-arg TORCH_CUDA_ARCH_LIST=8.6 --build-arg MAX_JOBS=2 -t mmrotate-dev --load .
```

Build args:

| arg | default | notes |
| --- | --- | --- |
| `TORCH_CUDA_ARCH_LIST` | `7.5 8.0 8.6 8.9 9.0 10.0 12.0+PTX` | narrow it to your GPUs for a faster build |
| `MAX_JOBS` | `8` | parallel nvcc jobs; each can use several GB of RAM |
| `PYTORCH` / `CUDA` / `CUDNN` | `2.14.1` / `13.0` / `9` | base image tag parts |
| `MMCV` / `MMDET` | `2.1.0` / `3.3.0` | mmdet 3.3 requires mmcv < 2.2 |

## Run

```bash
DATA_ROOT=/data docker compose -f docker/docker-compose.yaml run --rm mmrotate-dev
# inside the container
python docker/verify_mmcv.py   # GPU smoke test: mmcv CUDA ops, Triton MSDA, configs
```

The repo is bind-mounted over `/mmrotate`, so edits made on the host take effect right away.
`DATA_ROOT` (default `/mnt`) is mounted at `/mnt`.

## Why mmcv's CUDA ops sometimes go missing, and how this build prevents it

- **pip build isolation.** `pip install` builds in a fresh env without torch. mmcv's `setup.py` then
  prints `Skip building ext ops due to the absence of torch` and installs successfully with no
  `mmcv._ext`. This image builds with `--no-build-isolation`.
- **setuptools >= 80** drops `pkg_resources`, which mmcv 2.1's `setup.py` imports. This image pins `setuptools<80`.
- **C++ standard.** torch >= 2.10 headers require C++20, but mmcv 2.1.0's `setup.py` hard-codes
  `-std=c++17` for g++ and nvcc, so every op fails to compile.
  `mmcv_patches/0001-build-with-cxx20.patch` switches it to C++20.
- **No GPU at build time.** `FORCE_CUDA=1` and `MMCV_WITH_OPS=1` force the CUDA build anyway.

The build then **fails** unless `cuobjdump` finds kernels for every arch in `TORCH_CUDA_ARCH_LIST`
(checked in the builder), the runtime base has exactly the torch stack the wheel was built against,
`mmcv.ops.get_compiling_cuda_version()` matches torch's CUDA, and the rotated-box ops load from
`mmcv._ext`. Any source fixes mmcv needs for newer torch/CUDA go in `docker/mmcv_patches/`.

Other defaults: `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` (torch >= 2.6 changed the `torch.load` default,
which breaks mmengine checkpoint loading), and `/opt/constraints.txt` pins the image's torch stack so
later pip installs can't replace it.

Installs use plain pip rather than uv because uv segfaults under QEMU amd64 emulation. That would
break builds on an Apple Silicon Mac unless Docker Desktop's Rosetta emulation is enabled.
