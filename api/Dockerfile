# REST API image: the mmrotate dev image + this repo + the API's few extra packages.
# Self-contained for airgapped use: Swagger UI / ReDoc assets are in api/static, the model's
# pretrained backbone weights are never fetched (everything comes from the checkpoint).
#
#   docker build -f api/Dockerfile -t mmrotate-api .            # from the repo root
#   docker run --gpus all -p 8200:8200 -v /path/to/model:/model:ro mmrotate-api \
#       --config /model/config.yaml --weights /model/weights.pth --gsd 0.5
ARG BASE_IMAGE=artifactory.project.dso/srsep-docker/voltronix-mmrotate-dev:2026.11.19
FROM ${BASE_IMAGE}

COPY api/requirements.txt /tmp/api-requirements.txt
RUN pip install --no-cache-dir -r /tmp/api-requirements.txt && rm /tmp/api-requirements.txt

COPY . /mmrotate
# HF_HUB_OFFLINE / NO_ALBUMENTATIONS_UPDATE: libraries in the base image phone home at import
# (bitsandbytes kernels, albumentations version check); never on an airgapped box.
ENV PYTHONPATH=/mmrotate \
    HOME=/tmp \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    NO_ALBUMENTATIONS_UPDATE=1
WORKDIR /mmrotate
EXPOSE 8200
HEALTHCHECK --interval=60s --timeout=5s --start-period=600s \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/' % os.environ.get('MMROTATE_API_PORT', '8200'), timeout=4)"
# Flags or MMROTATE_API_* environment variables (see `python -m api --help`).
ENTRYPOINT ["python", "-m", "api"]
CMD []
