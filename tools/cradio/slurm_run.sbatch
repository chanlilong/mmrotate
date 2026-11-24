#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.
# Run one command inside the dev image on the GPU(s) slurm allocated: the
# benchmarks, GPU tests and smoke trainings of the C-RADIOv4 CoMer configs.
#
#   mkdir -p work_dirs/cradio_lowmem/slurm
#   sbatch --constraint=vram45 --gres=gpu:1 -J bench -o work_dirs/cradio_lowmem/slurm/%x-%j.out \
#       tools/cradio/slurm_run.sbatch python tools/cradio/bench_train_step.py CONFIG --batch 2 4
#   sbatch --constraint=vram45 --gres=gpu:1 -J tests -o work_dirs/cradio_lowmem/slurm/%x-%j.out \
#       tools/cradio/slurm_run.sbatch pytest -p no:cacheprovider tests/test_models/test_necks
#
# --constraint=vram45 picks the 46 GB RTX 6000 Ada cards, vram96 the 96 GB ones.
# When those are taken, the spare card of a node can be used at low priority:
#   sbatch --qos=low --nodelist=hammond --gres=sparegpu:1 ... tools/cradio/slurm_run.sbatch ...
# (one process only: the spare card is in exclusive-process mode).
# Env: IMAGE (default below), DOCKER_ARGS (extra `docker run` arguments).
#SBATCH --partition=sep
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=02:00:00
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
IMAGE=${IMAGE:-artifactory.project.dso/srsep-docker/voltronix-mmrotate-dev:2026.11.23}
echo "job ${SLURM_JOB_ID:-} on $(hostname): $* ($(date))"
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "FATAL: no image $IMAGE on $(hostname)"; exit 1; }

# Docker does not see slurm's GPU cgroup: pass the allocated cards by UUID.
uuids=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | paste -sd, -)
echo "GPUs: $uuids"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader

# shellcheck disable=SC2086
docker run --rm --gpus "\"device=$uuids\"" --ipc=host --shm-size=16g --network host \
    --user "$(id -u):$(id -g)" -e HOME=/tmp -e PYTHONPATH=/mmrotate \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -v "$PWD":/mmrotate -v /mnt:/mnt -w /mmrotate --entrypoint= ${DOCKER_ARGS:-} "$IMAGE" "$@"
echo "job ${SLURM_JOB_ID:-} done ($(date))"
