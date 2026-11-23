# Check with Trivy

```bash
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock aquasec/trivy:0.69.0 --insecure image artifactory.project.dso/srsep-docker/voltronix-mmrotate-dev:2026.11.19
```

# voltronix-mmrotate-dev (docker/Dockerfile)

The only Docker build for this repo: development, training (`mmrotate/accelerate`, DeepSpeed,
torchao) and TensorRT export (`mmrotate/tensorrt`, `tools/tensorrt`) all use this one image.
It was moved here from the `mmrotate_build` repo (`docker/dockerfile-redhat`), and its
`certs/`, `confs/`, `setup_files/` and `requirements/` folders now sit next to the Dockerfile, so
the build context is `docker/`:

```bash
docker buildx build --load -f docker/Dockerfile \
    -t artifactory.project.dso/srsep-docker/voltronix-mmrotate-dev:2026.11.19 docker
```

| component | version |
| --- | --- |
| base | `nvidia/cuda:13.0.3-cudnn-{devel,runtime}-ubi9`, Miniforge Python 3.12 |
| torch / torchvision | 2.14.1 / 0.29.1 (cu130) |
| mmcv | 2.2.0, built from source with CUDA ops (`setup_files/setup_mmcv.py`, C++20) |
| mmengine / mmpretrain / mmdet | gitlab.ai-stack.dso forks (mmdet 3.3.0 from PyPI first) |
| TensorRT / ModelOpt / Polygraphy | `tensorrt-cu13` 11.4.0.106 / 0.47.0 / 0.53.6 |
| DeepSpeed / torchao | 0.19.9 (cpu_adam + fused_adam prebuilt) / 0.18.0+cu130 |

mmrotate itself is not installed. Mount the repo and put it on `PYTHONPATH`:

```bash
docker run --rm -it --gpus all --ipc host -e PYTHONPATH=/mmrotate -v "$PWD":/mmrotate -v /mnt:/mnt \
    -w /mmrotate artifactory.project.dso/srsep-docker/voltronix-mmrotate-dev:2026.11.19
# or
DATA_ROOT=/mnt docker compose -f docker/docker-compose.yaml run --rm mmrotate-dev
# inside the container
python docker/verify_mmcv.py   # GPU smoke test: mmcv CUDA ops, Triton MSDA, configs
```

- **2026.11.19**: moved into this repo, replacing the old `docker/Dockerfile` (mmcv 2.1.0 on
  `pytorch/pytorch`) and `docker/Dockerfile.trt`. Same packages as 2026.11.05; the verification
  step also checks `tensorrt.plugin` AOT support (needed by the MSDA plugin) and polygraphy / onnx.
- **2026.11.05**: adds `deepspeed==0.19.9` and `torchao==0.18.0+cu130` for `mmrotate/accelerate`
  (ZeRO 1/2/3, fp8). DeepSpeed's CPU Adam and fused Adam ops are compiled in the builder stage,
  which has nvcc. `DS_ACCELERATOR=cuda` is needed because there is no GPU during `docker build`.
- **2026.10.29**: adds `tensorrt-cu13==11.4.0.106`, `nvidia-modelopt==0.47.0` (without the `[onnx]`
  extra, which would pin onnx 1.21 and a CUDA 12 onnxruntime-gpu) and `polygraphy==0.53.6`, for
  mmrotate's TensorRT export and INT8/FP8 quantization (`mmrotate/tensorrt`, `tools/tensorrt`).
  The pip wheels bundle libnvinfer, so nothing else is needed at runtime.
- `requirements/requirements-dev.txt` is unpinned and no longer resolves to a working set (it pulled
  `six==1.7.3`, `boto3==0.0.3`, ..., which break `import mmdet`). It is installed with
  `-c requirements/constraints-dev.txt`, the package versions of the working 2026.10.19 image.
  Update that file deliberately when bumping packages.
- The `dnf install` steps use `--disablerepo=cuda`: none of their packages come from NVIDIA's CUDA
  repo, whose metadata was returning 404s and failing the build.
