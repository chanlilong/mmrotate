#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.
# Run one command (e.g. tools/tensorrt/run_matrix.sh) on one slurm GPU, inside the dev image:
#
#   sbatch --nodelist=may --gres=gpu:1 tools/tensorrt/slurm_run.sbatch tools/tensorrt/run_matrix.sh ...
#
# Env: IMAGE (default: the dev image with TensorRT 11.4 / ModelOpt 0.47), DOCKER_ARGS.
#SBATCH --partition=sep
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
IMAGE=${IMAGE:-artifactory.project.dso/srsep-docker/voltronix-mmrotate-dev:2026.11.05}
echo "job ${SLURM_JOB_ID:-} on $(hostname): $* ($(date))"
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "FATAL: no image $IMAGE on $(hostname)"; exit 1; }
# Docker does not see slurm's GPU cgroup: pin the container to the allocated card by UUID.
GPU=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1)
echo "GPU: $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1) ($GPU)"
# scancel only signals this script: stop the container with it.
NAME=trt-${SLURM_JOB_ID:-$$}
trap 'docker stop -t 10 "$NAME" >/dev/null 2>&1' TERM INT
# shellcheck disable=SC2086
docker run --rm --name "$NAME" --gpus "\"device=$GPU\"" --ipc host --user "$(id -u):$(id -g)" \
    -e HOME=/tmp -e PYTHONPATH=/mmrotate -v "$PWD:/mmrotate" -v /mnt:/mnt -w /mmrotate \
    --entrypoint= ${DOCKER_ARGS:-} "$IMAGE" "$@" &
wait $! || status=$?
[[ ${status:-0} -eq 0 ]] || { echo "job ${SLURM_JOB_ID:-} failed ($status)"; exit "$status"; }
echo "job ${SLURM_JOB_ID:-} done ($(date))"
