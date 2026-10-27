#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.
# One slurm job = one run_experiments.sh invocation, on the GPU slurm gives us.
# Submitted by slurm_submit.sh (which sets the job options and the env below):
#
#   sbatch [...] slurm_run.sbatch one <config> [seed] | bench | collect
#
# Env: IMAGE, WORK (as run_experiments.sh), DATA_ROOT (split SODA-A root),
# WEIGHTS (folder with the timm checkpoints), DOCKER_ARGS.
#SBATCH --partition=sep
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
echo "job $SLURM_JOB_ID on $(hostname): $* ($(date))"
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "FATAL: no image $IMAGE on $(hostname)"; exit 1; }

# Docker does not see slurm's GPU cgroup: pin the container to the allocated card by UUID.
if [[ -n ${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}} ]]; then
    GPU=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1)
else
    GPU=0  # CPU-only step (collect); the container still needs a --gpus value
fi
export GPU
echo "GPU: $GPU"

# Local data / weight paths instead of the /mnt/4TB ones in the configs.
opts=(
    "train_dataloader.dataset.data_root=$DATA_ROOT"
    "val_dataloader.dataset.data_root=$DATA_ROOT"
    "test_dataloader.dataset.data_root=$DATA_ROOT"
)
case ${1:-} in
    one)
        if grep -q convnext <<< "$2"; then ckpt=convnext_tiny.dinov3_lvd1689m
        else ckpt=vit_small_patch16_dinov3.lvd1689m; fi
        opts+=("model.backbone.checkpoint=$WEIGHTS/$ckpt")
        ;;
    bench)
        # Speed / memory only, and the configs mix backbones: random init.
        opts+=("model.backbone.checkpoint=None")
        ;;
esac
export CFG_OPTIONS="${opts[*]}"

tools/soda_dinov3/run_experiments.sh "$@"
echo "job $SLURM_JOB_ID done ($(date))"
