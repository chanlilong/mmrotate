# mmcv source patches

`*.patch` files here are applied with `patch -p1` to the mmcv source tree (in name order) before
mmcv is compiled in `docker/Dockerfile`. Use them for minimal fixes needed to build the pinned mmcv
release against newer PyTorch / CUDA. Each patch should say in its header what it fixes and why.
