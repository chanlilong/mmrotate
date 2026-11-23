# voltronix-mmrotate-dev: the one dev / training / TensorRT / REST API image for this repo (see docker/README.md).
# The build context is docker/ (certs/, confs/, setup_files/, requirements/ live next to this file):
#   docker buildx build --load -f docker/Dockerfile -t <registry>/voltronix-mmrotate-dev:<date> docker
# mmrotate itself is not installed: mount the repo at /mmrotate and set PYTHONPATH=/mmrotate.
# ====================================================================================
# Stage 1: Builder
#
# Uses Red Hat UBI 9. Installs CUDA Toolkit, Compilers, and builds dependencies.
# ====================================================================================
FROM nvidia/cuda:13.0.3-cudnn-devel-ubi9 AS builder

ARG CUDA_VERSION_MAJOR=13
ARG CUDA_VERSION_MINOR=0
# Corresponds to the RHEL repo versioning
ARG CUDA_PKG_VERSION=${CUDA_VERSION_MAJOR}-${CUDA_VERSION_MINOR}

# Set environment variables
ENV TORCH_CUDA_ARCH_LIST="7.5+PTX 8.6+PTX 8.9+PTX 9.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV UV_HTTP_TIMEOUT=3600
ENV SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
ENV REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
ENV FORCE_CUDA="1"
ENV MAX_JOBS="8"
# Add Conda to path immediately
ENV PATH="/opt/conda/bin:${PATH}"
# Set CUDA Home for compilation
ENV CUDA_HOME="/usr/local/cuda"
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# 1. Install Certificates (RHEL Location)
COPY certs/DSOCA50.pem /etc/pki/ca-trust/source/anchors/DSOCA50.pem
COPY certs/DSOCA60.pem /etc/pki/ca-trust/source/anchors/DSOCA60.pem
COPY certs/DSOVisibilityCA.pem /etc/pki/ca-trust/source/anchors/DSOVisibilityCA.crt
COPY certs/certbundle.crt  /etc/pki/ca-trust/source/anchors/certbundle.crt
RUN update-ca-trust


# 2. Install System Build Dependencies & CUDA Toolkit
# We need EPEL for ninja-build and screen
# RUN dnf install epel-release -y
# The CUDA yum repo is not needed for these packages (and its metadata is not always consistent).
RUN dnf install -y --disablerepo=cuda https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm
RUN dnf install -y --disablerepo=cuda \
    git wget gcc gcc-c++ make cmake \
    glib2 libSM libXext libXrender \
    ninja-build screen \
    && dnf clean all

# 3. Install Miniconda (Python 3.12)
RUN wget https://github.com/conda-forge/miniforge/releases/download/25.11.0-1/Miniforge3-25.11.0-1-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    conda install -y -c conda-forge python=3.12 gdal ffmpeg sqlite

# 4. Install uv and Python Base
RUN python -m pip install --upgrade pip && pip install uv

# WORKDIR /sam3
# RUN git clone https://github.com/facebookresearch/sam3.git
# RUN cd sam3 && uv pip install ".[train,dev]" --system --no-cache-dir --no-build-isolation

# 5. Install PyTorch (Matching CUDA version)
# Using the official PyTorch index
RUN uv pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130 --system --no-cache-dir

# 6. Install MM Dependencies
RUN git config --global http.sslVerify false
RUN uv pip install mmdet==3.3.0 --system --no-cache-dir
RUN uv pip install setuptools==76.1.0 --force-reinstall --no-cache-dir --no-build-isolation --system 
RUN uv pip install git+https://gitlab.ai-stack.dso/clilong/mmengine.git --system --no-build-isolation --no-cache-dir

# 7. Build MMCV from source
WORKDIR /mmcv
RUN wget https://github.com/open-mmlab/mmcv/archive/refs/tags/v2.2.0.tar.gz -O /mmcv.tar.gz \
    && tar -xzvf /mmcv.tar.gz -C / \
    && mv /mmcv-2.2.0 /mmcv \
    && rm /mmcv.tar.gz

RUN python -m uv pip install -r ./mmcv-2.2.0/requirements/optional.txt --system --no-build-isolation --no-cache-dir
COPY setup_files/setup_mmcv.py ./mmcv-2.2.0/setup.py


# Ensure CUDA is found during build
ENV FORCE_CUDA="1"
ENV MAX_JOBS="16"
RUN cd ./mmcv-2.2.0 && python setup.py install

# 8. Reinstall MM Engine (as per original dockerfile logic)
RUN uv pip install git+https://gitlab.ai-stack.dso/clilong/mmengine.git --system --no-build-isolation --force-reinstall --no-cache-dir

# 9. MMPretrain
RUN uv pip install setuptools==76.1.0 --force-reinstall --no-cache-dir --no-build-isolation --system
WORKDIR /mmpretrain
RUN uv pip install git+https://gitlab.ai-stack.dso/clilong/mmpretrain.git --system --no-build-isolation --force-reinstall --no-cache-dir

# 10. MMDetection & Requirements
WORKDIR /root
COPY ./requirements/requirements-dev.txt ./requirements.txt
COPY ./requirements/constraints-dev.txt ./constraints.txt
RUN uv pip install -r requirements.txt -c constraints.txt --system --no-cache-dir
RUN uv pip install git+https://gitlab.ai-stack.dso/clilong/mmdetection.git --system --no-cache-dir --no-build-isolation

# 11. TensorRT + ModelOpt for mmrotate's TensorRT export (mmrotate/tensorrt, tools/tensorrt).
# The pip wheels bundle libnvinfer, so the runtime stage needs nothing extra. ModelOpt without the
# [onnx] extra: that extra pins onnx 1.21 / a CUDA 12 onnxruntime-gpu; mmrotate quantizes in torch.
RUN uv pip install tensorrt-cu13==11.4.0.106 nvidia-modelopt==0.47.0 polygraphy==0.53.6 \
    --system --no-cache-dir

# 11b. DeepSpeed (ZeRO 1/2/3 through accelerate, mmrotate/accelerate) and torchao (fp8 training).
# DeepSpeed's CUDA/C++ ops need nvcc, which only this builder stage has, so the ops mmrotate can use
# are compiled here (CPU Adam for ZeRO optimizer offload, fused Adam) instead of JIT at runtime.
# 12.0 = Blackwell (RTX PRO 6000); the other ops stay unbuilt (DS_BUILD_OPS=0). DS_ACCELERATOR=cuda:
# there is no GPU during `docker build`, and DeepSpeed would otherwise fall back to its CPU accelerator.
RUN DS_ACCELERATOR=cuda TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0;12.0+PTX" \
    DS_BUILD_OPS=0 DS_BUILD_CPU_ADAM=1 DS_BUILD_FUSED_ADAM=1 \
    uv pip install deepspeed==0.19.9 --system --no-cache-dir --no-build-isolation
RUN uv pip install torchao==0.18.0+cu130 --index-url https://download.pytorch.org/whl/cu130 \
    --system --no-cache-dir --no-deps

# 11c. REST API for qgis_annotations (api/, `python -m api`): fastapi + uvicorn, and httpx for its
# tests. Last in the builder so the layers above stay cached; versions come from constraints-dev.
COPY ./requirements/requirements-api.txt ./requirements-api.txt
RUN uv pip install -r requirements-api.txt -c constraints.txt --system --no-cache-dir

# 12. Install GDAL and FFmpeg via Conda
# Installing ffmpeg here avoids needing RPMFusion in the base image
RUN CONDA_NO_PLUGINS=true conda clean --all -y

# ====================================================================================
# Stage 2: Final Image
#
# Uses UBI 9 Minimal. Copies the Conda environment and compiled artifacts.
# ====================================================================================
# FROM registry.access.redhat.com/ubi9/ubi:9.7-1767674301
FROM nvidia/cuda:13.0.3-cudnn-runtime-ubi9

# Set runtime environment variables
ENV SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
ENV REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
ENV PYTHONPATH=/mmdeploy/mmdeploy/build/lib${PYTHONPATH:+:${PYTHONPATH}}
ENV LD_LIBRARY_PATH=/mmdeploy/mmdeploy-dep/onnxruntime-linux-x64-1.8.1/lib/:$LD_LIBRARY_PATH

# GDAL Env Vars
ENV GDAL_DATA=/opt/conda/share/gdal
ENV PROJ_LIB=/opt/conda/share/proj
# Ensure system finds Conda libs and local libs
ENV LD_LIBRARY_PATH=/opt/conda/lib:/usr/local/lib:$LD_LIBRARY_PATH
ENV PATH="/opt/conda/bin:$PATH"

# 1. Install Certificates
COPY certs/DSOCA50.pem /etc/pki/ca-trust/source/anchors/DSOCA50.pem
COPY certs/DSOCA60.pem /etc/pki/ca-trust/source/anchors/DSOCA60.pem
COPY certs/DSOVisibilityCA.pem /etc/pki/ca-trust/source/anchors/DSOVisibilityCA.crt
COPY certs/certbundle.crt  /etc/pki/ca-trust/source/anchors/certbundle.crt
COPY certs/ai-stack-ca-cert.crt /etc/pki/ca-trust/source/anchors/ai-stack-ca-cert.crt
RUN update-ca-trust

# 2. Install Runtime Dependencies
# gcc is needed at runtime: Triton JIT-compiles its C launcher stubs on first kernel launch
RUN dnf install -y --disablerepo=cuda https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm && \
    dnf install -y --disablerepo=cuda \
    git glib2 libSM libXext libXrender screen wget shadow-utils gcc unzip jq \
    && dnf clean all
# 3. Copy Artifacts from Builder
COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /mmcv /mmcv
COPY --from=builder /mmpretrain /mmpretrain
# COPY --from=builder /mmdeploy /mmdeploy (Commented out in original)

# 4. Configure Pip
WORKDIR /root
RUN mkdir -p /etc/xdg/pip /root/.pip
COPY confs/pip_penguin.conf /root/.pip/pip.conf

# 5. Initialize Shell

RUN CONDA_NO_PLUGINS=true conda init bash

# 6. Install SonarScanner CLI (bundles its own JRE) for local analysis during dev
# The bundled JRE's cacerts is pointed at the system Java truststore so it trusts the DSO CAs
ARG SONAR_SCANNER_VERSION=8.3.0.16463
ENV SONAR_SCANNER_HOME=/opt/sonar-scanner
RUN wget -q https://binaries.sonarsource.com/Distribution/sonar-scanner-cli/sonar-scanner-cli-${SONAR_SCANNER_VERSION}-linux-x64.zip -O /tmp/sonar-scanner.zip && \
    unzip -q /tmp/sonar-scanner.zip -d /opt && \
    mv /opt/sonar-scanner-${SONAR_SCANNER_VERSION}-linux-x64 ${SONAR_SCANNER_HOME} && \
    rm /tmp/sonar-scanner.zip && \
    ln -sf /etc/pki/java/cacerts ${SONAR_SCANNER_HOME}/jre/lib/security/cacerts && \
    ln -s ${SONAR_SCANNER_HOME}/bin/sonar-scanner /usr/local/bin/sonar-scanner
COPY setup_files/sonar-dev-scan.sh /usr/local/bin/sonar-dev-scan
COPY setup_files/sonar_export_issues.py /usr/local/bin/sonar-export-issues
RUN chmod +x /usr/local/bin/sonar-dev-scan /usr/local/bin/sonar-export-issues

# 7. Verification
RUN echo "✅ Verifying installations..." && \
    python -c "import torch; print(f'PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')" && \
    python -c "import mmdet; print(f'MMDetection: {mmdet.__version__}')" && \
    python -c "import mmcv; print(f'MMCV: {mmcv.__version__}')" && \
    python -c "import mmcv.ops" && \
    python -c "import tensorrt; print(f'TensorRT: {tensorrt.__version__}')" && \
    python -c "import tensorrt.plugin as trtp; assert hasattr(trtp, 'aot_impl'), 'no AOT plugins (MSDA plugin needs TensorRT >= 10.9)'" && \
    python -c "import modelopt; import modelopt.torch.quantization; print(f'ModelOpt: {modelopt.__version__}')" && \
    python -c "import polygraphy, onnx, onnxruntime; print(f'Polygraphy: {polygraphy.__version__}, onnx: {onnx.__version__}')" && \
    python -c "import accelerate, deepspeed; print(f'accelerate: {accelerate.__version__}, DeepSpeed: {deepspeed.__version__}')" && \
    DS_ACCELERATOR=cuda python -c "from deepspeed.git_version_info import installed_ops as i; assert i['cpu_adam'] and i['fused_adam'], i" && \
    python -c "import torchao; from torchao.float8 import convert_to_float8_training; print(f'torchao: {torchao.__version__}')" && \
    python -c "from osgeo import gdal; print(f'GDAL: {gdal.__version__}')" &&\
    python -c "import numpy" && \
    python -c "import rasterio; from rasterio.crs import CRS; print(f'Rasterio GDAL: {rasterio.__gdal_version__}'); CRS.from_epsg(4326)" && \
    python -c "import cv2" && \
    python -c "import fastapi, uvicorn, httpx; print(f'FastAPI: {fastapi.__version__}, uvicorn: {uvicorn.__version__}')" && \
    gcc --version && \
    pytest --version && \
    sonar-scanner --version && \
    jq --version

RUN ldconfig
ENTRYPOINT [ "/bin/bash" ]