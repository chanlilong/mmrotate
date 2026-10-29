#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.
# One assigner_search.py trial: tools/train.py in the dev container, on the GPU
# slurm allocates. Use it as the search's --cmd, e.g.
#
#   --cmd 'sbatch --wait -o {work_dir}/slurm-%j.out tools/assigner_tuning/slurm_train.sbatch {config} {work_dir}'
#
# Env: IMAGE (docker image), DOCKER_ARGS (extra `docker run` arguments).
#SBATCH --partition=sep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --job-name=assigner-search
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
IMAGE=${IMAGE:-artifactory.project.dso/srsep-docker/voltronix-mmrotate-dev:2026.10.19}
# This image's entrypoint is bash and mmrotate is not pip-installed in it.
read -r -a DOCKER_EXTRA <<< "${DOCKER_ARGS:---entrypoint= -e PYTHONPATH=/mmrotate}"
echo "job ${SLURM_JOB_ID:-local} on $(hostname): $* ($(date))"
# Docker does not see slurm's GPU cgroup: pin the container to the allocated card by UUID.
GPU=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1)
docker run --rm --gpus "device=$GPU" --ipc=host --shm-size=16g \
    --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$PWD":/mmrotate -v /mnt:/mnt -w /mmrotate "${DOCKER_EXTRA[@]}" "$IMAGE" \
    python tools/train.py "$1" --work-dir "$2"
echo "job ${SLURM_JOB_ID:-local} done ($(date))"
