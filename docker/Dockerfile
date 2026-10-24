# syntax=docker/dockerfile:1
# check=skip=FromPlatformFlagConstDisallowed
# MMRotate dev image: x86_64, PyTorch 2.14.1, CUDA 13.0 (NVIDIA driver >= 580), Python 3.12.
#
# Two stages: mmcv's CUDA ops are compiled into a wheel on the -devel base (nvcc, ~22 GB), and the
# wheel is installed into the much smaller -runtime base (~6 GB), which is what the final image is.
#
# Build from the repository root (the build context is the repo):
#   docker build -f docker/Dockerfile -t mmrotate-dev .
# See docker/README.md for build args and run commands.
ARG PYTORCH="2.14.1"
ARG CUDA="13.0"
ARG CUDNN="9"

# ================================================================ builder: mmcv wheel
# Pinned to amd64 so the image targets the x86 GPU box even when built on an arm64 Mac.
FROM --platform=linux/amd64 pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel AS builder

ARG MMCV="2.1.0"
# CUDA 13 supports sm_75 (Turing) and newer only. +PTX keeps newer GPUs working via JIT.
ARG TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.9 9.0 10.0 12.0+PTX"
ARG MAX_JOBS=8

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    MAX_JOBS=${MAX_JOBS} \
    FORCE_CUDA=1 \
    MMCV_WITH_OPS=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore

RUN apt-get update && apt-get install -y --no-install-recommends \
        ninja-build wget ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Pin the image's CUDA torch stack so no later install can replace it with a PyPI build.
# The runtime stage checks it has the exact same stack, since mmcv._ext is built against it.
RUN python - <<'EOF'
from importlib.metadata import PackageNotFoundError, version

pins = []
for name in ('torch', 'torchvision', 'torchaudio', 'triton', 'numpy'):
    try:
        pins.append(f'{name}=={version(name)}')
    except PackageNotFoundError:
        pass
open('/opt/constraints.txt', 'w').write('\n'.join(pins) + '\n')
print('\n'.join(pins))
EOF
ENV PIP_CONSTRAINT=/opt/constraints.txt

# Two ways the CUDA kernels silently go missing:
#  1. pip build isolation: the isolated build env has no torch, so mmcv's setup.py prints
#     "Skip building ext ops due to the absence of torch" and installs without mmcv._ext.
#     -> --no-build-isolation
#  2. setuptools >= 80 drops pkg_resources, which mmcv's setup.py imports. -> setuptools<80
# And torch >= 2.10 headers need C++20 while mmcv 2.1.0 forces -std=c++17, so every op fails
# to compile -> docker/mmcv_patches/0001-build-with-cxx20.patch.
# The check below fails the build if mmcv._ext lacks CUDA code for any requested arch.
RUN python -m pip install "setuptools<80" wheel ninja psutil "mmengine>=0.10.7,<1.0.0"
RUN wget -q https://github.com/open-mmlab/mmcv/archive/refs/tags/v${MMCV}.tar.gz -O /tmp/mmcv.tar.gz \
    && mkdir -p /opt/mmcv && tar -xzf /tmp/mmcv.tar.gz -C /opt/mmcv --strip-components=1 \
    && rm /tmp/mmcv.tar.gz
COPY docker/mmcv_patches/ /opt/mmcv_patches/
RUN cd /opt/mmcv \
    && for p in /opt/mmcv_patches/*.patch; do [ -e "$p" ] || continue; echo "applying $p"; patch -p1 < "$p"; done \
    && { python -m pip wheel --no-build-isolation --no-deps -w /wheels -v . 2>&1 \
         | tee /tmp/mmcv_build.log | grep --line-buffered -oE '^\s*\[[0-9]+/[0-9]+\] [^ ]+' \
         || { grep -E -B2 -A8 'error|FAILED' /tmp/mmcv_build.log | tail -n 200; exit 1; }; } \
    && if grep -q 'Skip building ext ops' /tmp/mmcv_build.log; then echo 'mmcv built without ops' >&2; exit 1; fi \
    && tail -n 3 /tmp/mmcv_build.log \
    && python -m pip install --no-deps /wheels/mmcv-*.whl
RUN python - <<'EOF'
import glob
import os
import subprocess

import mmcv

so = glob.glob(os.path.join(os.path.dirname(mmcv.__file__), '_ext*.so'))[0]
elf = subprocess.run(['cuobjdump', '--list-elf', so], capture_output=True, text=True, check=True).stdout
for arch in os.environ['TORCH_CUDA_ARCH_LIST'].replace(';', ' ').split():
    sm = 'sm_' + arch.replace('+PTX', '').replace('.', '')
    n = elf.count(f'.{sm}.')
    print(f'{sm}: {n} cubins')
    assert n > 0, f'no {sm} kernels in {so}'
EOF

# ================================================================ runtime: the dev image
FROM --platform=linux/amd64 pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-runtime

ARG MMDET="3.3.0"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# Triton (used by torch >= 2.14 ops, torch.compile and our own kernels) compiles its launchers with
# the host C compiler at run time -> build-essential. The Python headers ship with the base.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git wget ca-certificates libglib2.0-0 libgl1 ffmpeg screen tmux \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# mmcv._ext links against the builder's torch, so the runtime base must ship the same stack.
COPY --from=builder /opt/constraints.txt /opt/constraints.txt
RUN python - <<'EOF'
from importlib.metadata import version

for pin in open('/opt/constraints.txt').read().split():
    name, want = pin.split('==')
    have = version(name)
    assert have == want, f'{name} {have} in the runtime base, but mmcv was built against {want}'
print(open('/opt/constraints.txt').read(), end='')
EOF
ENV PIP_CONSTRAINT=/opt/constraints.txt

# ---------------------------------------------------------------- mmcv (prebuilt wheel)
RUN --mount=type=bind,from=builder,source=/wheels,target=/wheels \
    python -m pip install "setuptools<80" wheel psutil "mmengine>=0.10.7,<1.0.0" /wheels/mmcv-*.whl
RUN python - <<'EOF'
import mmcv
import torch
from mmcv.ops import get_compiler_version, get_compiling_cuda_version
from mmcv.utils import ext_loader

cuda = get_compiling_cuda_version()
print(f'mmcv {mmcv.__version__} | torch {torch.__version__} | compiled with CUDA {cuda} | {get_compiler_version()}')
assert cuda == torch.version.cuda, f'mmcv._ext built without CUDA (got {cuda!r}, torch has {torch.version.cuda})'

# Ops mmrotate relies on.
ext_loader.load_ext('_ext', [
    'nms_rotated', 'nms_quadri', 'box_iou_rotated', 'box_iou_quadri', 'convex_iou', 'convex_giou',
    'min_area_polygons', 'points_in_polygons_forward', 'active_rotated_filter_forward',
    'rotated_feature_align_forward', 'riroi_align_rotated_forward', 'roi_align_rotated_forward',
    'diff_iou_rotated_sort_vertices_forward', 'chamfer_distance_forward', 'deform_conv_forward',
])
EOF

# ---------------------------------------------------------------- mmdet + mmrotate deps
RUN python -m pip install "mmdet==${MMDET}"
WORKDIR /mmrotate
COPY requirements/runtime.txt /tmp/requirements-runtime.txt
RUN python -m pip install -r /tmp/requirements-runtime.txt \
        scipy scikit-learn e2cnn safetensors \
        pytest parameterized xdoctest kwarray ubelt ruff ipykernel

# ---------------------------------------------------------------- mmrotate (editable)
# The repo is bind-mounted over /mmrotate at dev time (docker/docker-compose.yaml), so the
# editable install picks up edits made on the host.
COPY . /mmrotate
RUN python -m pip install --no-deps --no-build-isolation \
        --config-settings editable_mode=compat -e /mmrotate \
    && python -c "import mmrotate, mmdet, mmcv, mmengine; print('mmrotate', mmrotate.__version__, '| mmdet', mmdet.__version__, '| mmcv', mmcv.__version__, '| mmengine', mmengine.__version__)"

CMD ["/bin/bash"]
