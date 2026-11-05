#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.
# Multi-GPU / multi-node training (or testing) with HF accelerate under slurm:
# one slurm task per node, each running `accelerate launch` for its GPUs.
#
#   sbatch --nodes=1 --gres=gpu:2 tools/accelerate/slurm_train.sbatch <config> [train.py args]
#   sbatch --nodes=2 --gres=gpu:4 tools/accelerate/slurm_train.sbatch <config> --cfg-options ...
#   SCRIPT=tools/accelerate/test.py sbatch --gres=gpu:2 tools/accelerate/slurm_train.sbatch <config> <ckpt>
#
# Env:
#   IMAGE        run inside this docker image (unset: run with the host python)
#   SCRIPT       tools/accelerate/train.py (default) or tools/accelerate/test.py
#   GPUS_PER_NODE  default: the GPUs slurm gave this node
#   MAIN_PORT    rendezvous port on the first node (default 29500 + job id % 1000)
#   DOCKER_ARGS  extra `docker run` arguments
# The precision / DeepSpeed recipe is in the config's `accelerator:` block; the launch
# here only sets the topology. The work_dir must be on a filesystem all nodes share.
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
SCRIPT=${SCRIPT:-tools/accelerate/train.py}
NNODES=${SLURM_NNODES:-1}
MAIN_IP=$(scontrol show hostnames "${SLURM_JOB_NODELIST:-$(hostname)}" | head -1)
MAIN_PORT=${MAIN_PORT:-$((29500 + ${SLURM_JOB_ID:-0} % 1000))}
export SCRIPT NNODES MAIN_IP MAIN_PORT
echo "job ${SLURM_JOB_ID:-} nodes=$NNODES main=$MAIN_IP:$MAIN_PORT: $SCRIPT $* ($(date))"

# Runs once per node (srun task); SLURM_NODEID is the machine rank.
per_node() {
    local gpus uuids
    gpus=${GPUS_PER_NODE:-$(nvidia-smi -L | wc -l)}
    local launch=(accelerate launch --num_machines "$NNODES" --machine_rank "${SLURM_NODEID:-0}"
        --num_processes $((NNODES * gpus)) --main_process_ip "$MAIN_IP"
        --main_process_port "$MAIN_PORT" --mixed_precision no --dynamo_backend no
        "$SCRIPT" "$@")
    if [[ -z ${IMAGE:-} ]]; then
        exec "${launch[@]}"
    fi
    # Docker does not see slurm's GPU cgroup: pass the allocated cards by UUID.
    uuids=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | paste -sd, -)
    echo "node ${SLURM_NODEID:-0} ($(hostname)): GPUs $uuids"
    # shellcheck disable=SC2086
    exec docker run --rm --gpus "\"device=$uuids\"" --network host --ipc host \
        --user "$(id -u):$(id -g)" -e HOME=/tmp -e PYTHONPATH=/mmrotate \
        -v "$PWD:/mmrotate" -v /mnt:/mnt -w /mmrotate --entrypoint= ${DOCKER_ARGS:-} \
        "$IMAGE" "${launch[@]}"
}
export -f per_node

if [[ -n ${SLURM_JOB_ID:-} ]]; then
    srun --ntasks-per-node=1 --kill-on-bad-exit=1 bash -c 'per_node "$@"' _ "$@"
else
    per_node "$@"
fi
echo "job ${SLURM_JOB_ID:-} done ($(date))"
