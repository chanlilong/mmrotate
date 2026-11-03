#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.
# Cost of the CoMer-lite variants against the CoMer baseline, on one GPU:
#   - tools/soda_dinov3/bench.py: real train step (dataloader, loss, AdamW) and
#     test throughput, with the configs' gradient checkpointing and without;
#   - tools/comer_lite/profile_modules.py: per-module GPU time and GFLOP.
# Writes $WORK/bench.json (read by collect_results.py), bench_nocp.json,
# profile_cp.{json,md}, profile_nocp.{json,md} and bench_gpu_clocks.csv.
# Run it on an otherwise idle machine: drift was large with two trainings on
# the neighbouring cards (power-capped Max-Q GPUs).
#
#   sbatch [...] tools/comer_lite/slurm_bench.sbatch   # via slurm_submit.sh bench
#   GPU=0 bash tools/comer_lite/slurm_bench.sbatch     # directly on a free local GPU
#
# Env: IMAGE, WORK, DATA_ROOT, DOCKER_ARGS (as slurm_submit.sh sets them).
#SBATCH --partition=sep
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
echo "bench on $(hostname) ($(date))"
if [[ -n ${SLURM_JOB_GPUS:-} ]]; then
    # Docker does not see slurm's GPU cgroup: pin the container to the allocated card by UUID.
    GPU=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1)
fi
GPU=${GPU:-0}
echo "GPU: $GPU"
read -r -a DOCKER_EXTRA <<< "${DOCKER_ARGS:---entrypoint= -e PYTHONPATH=/mmrotate}"

CONFIGS=(
    configs/soda_dinov3/vits-dinov3-comer_rtmdet-p2-6e-soda.yaml
    configs/soda_dinov3/vits-dinov3-comer-noinj_rtmdet-p2-6e-soda.yaml
    configs/comer_lite/vits-dinov3-comer-extra0_rtmdet-p2-6e-soda.yaml
    configs/comer_lite/vits-dinov3-comer-ratio05_rtmdet-p2-6e-soda.yaml
    configs/comer_lite/vits-dinov3-comer-cfinter-late_rtmdet-p2-6e-soda.yaml
    configs/comer_lite/vits-dinov3-comer-slim_rtmdet-p2-6e-soda.yaml
    configs/comer_lite/vits-dinov3-comer-lite_rtmdet-p2-6e-soda.yaml
)
DATA=(
    "train_dataloader.dataset.data_root=$DATA_ROOT"
    "val_dataloader.dataset.data_root=$DATA_ROOT"
    "test_dataloader.dataset.data_root=$DATA_ROOT"
    "model.backbone.checkpoint=None"
)
NOCP=("model.backbone.with_cp=False" "model.backbone.vit_with_cp=False")

in_docker() {
    docker run --rm --gpus "device=$GPU" --ipc=host --shm-size=16g \
        --user "$(id -u):$(id -g)" -e HOME=/tmp \
        -v "$PWD":/mmrotate -v /mnt:/mnt -w /mmrotate "${DOCKER_EXTRA[@]}" "$IMAGE" "$@"
}

mkdir -p "$WORK"
# Clock / power / throttle log of the benchmarked card, to explain any drift.
nvidia-smi -i "$GPU" --format=csv -l 2 \
    --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu,utilization.gpu,clocks_event_reasons.active \
    > "$WORK/bench_gpu_clocks.csv" &
SMI=$!
trap 'kill $SMI' EXIT
echo "other GPUs at start:"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
in_docker python tools/comer_lite/profile_modules.py "${CONFIGS[@]}" --out "$WORK/profile_cp.json"
in_docker python tools/comer_lite/profile_modules.py "${CONFIGS[@]}" --no-cp \
    --out "$WORK/profile_nocp.json"
in_docker python tools/soda_dinov3/bench.py "${CONFIGS[@]}" --out "$WORK/bench.json" \
    --cfg-options "${DATA[@]}"
in_docker python tools/soda_dinov3/bench.py "${CONFIGS[@]}" --out "$WORK/bench_nocp.json" \
    --cfg-options "${DATA[@]}" "${NOCP[@]}"
echo "bench done ($(date))"
