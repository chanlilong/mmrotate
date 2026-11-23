#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.
# Run one command in a docker image on the GPU slurm allocates (tests, the suggest tool, a
# short-lived service):
#
#   IMAGE=mmrotate-api:dev sbatch --nodelist=hammond api/slurm_run.sbatch \
#       python -m pytest tests/test_api -q
#   IMAGE=... sbatch api/slurm_run.sbatch \
#       python -m tools.similarity.suggest_api_hyperparameters <config> <weights>
#
# Env: IMAGE (required), DOCKER_ARGS (extra `docker run` arguments).
#SBATCH --partition=sep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=work_dirs/%x-%j.out
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
: "${IMAGE:?set IMAGE to the docker image to run in}"
echo "job ${SLURM_JOB_ID:-} on $(hostname): $* ($(date))"
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "FATAL: no image $IMAGE on $(hostname)"; exit 1; }
# Docker does not see slurm's GPU cgroup: pin the container to the allocated card by UUID.
GPU=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1)
echo "GPU: $GPU"
# shellcheck disable=SC2086
docker run --rm --gpus "\"device=$GPU\"" --ipc host \
    --user "$(id -u):$(id -g)" -e HOME=/tmp -e PYTHONPATH=/mmrotate \
    -v "$PWD:/mmrotate" -v /mnt:/mnt -w /mmrotate --entrypoint= ${DOCKER_ARGS:-} \
    "$IMAGE" "$@"
echo "job ${SLURM_JOB_ID:-} done ($(date))"
