#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.
# Queue the SODA-A DINOv3 experiments of run_experiments.sh as separate slurm
# jobs (one GPU each), so they run in parallel on whatever GPUs are free, plus
# a `collect` job that runs once all of them have ended. Finished runs are
# skipped inside the jobs, so re-submitting after a failure is safe.
#
#   tools/soda_dinov3/slurm_submit.sh [all|bench|phase1|noinj|seeds|lora|dcfl|phase2] ...
#
# Env (defaults for the voltronix cluster): PARTITION, NODELIST (nodes that
# have IMAGE), IMAGE, WORK, DATA_ROOT, WEIGHTS, TIME (per-run limit),
# COLLECT=0 to skip the collect job (e.g. when adding runs to a queued batch).
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
CFG=configs/soda_dinov3
export IMAGE=${IMAGE:-artifactory.project.dso/srsep-docker/voltronix-mmrotate-dev:2026.10.19}
export WORK=${WORK:-work_dirs/soda_dinov3}
export DATA_ROOT=${DATA_ROOT:-/mnt/zfs/datasets/eo/soda-a/split_800/}
export WEIGHTS=${WEIGHTS:-/mnt/zfs/models/timm}
# This image's entrypoint is bash and mmrotate is not pip-installed in it.
export DOCKER_ARGS=${DOCKER_ARGS:---entrypoint= -e PYTHONPATH=/mmrotate}
PARTITION=${PARTITION:-sep}
NODELIST=${NODELIST:-may}
TIME=${TIME:-12:00:00}

mkdir -p "$WORK/slurm"
common=(--partition="$PARTITION" --export=ALL --output="$WORK/slurm/%x-%j.out")
[[ -n $NODELIST ]] && common+=(--nodelist="$NODELIST")
jobs=()

# submit <job name> <slurm_run.sbatch args...>
submit() {
    local name=$1; shift
    local id
    id=$(sbatch --parsable "${common[@]}" --job-name="$name" --gres=gpu:1 --time="$TIME" \
        tools/soda_dinov3/slurm_run.sbatch "$@")
    echo "$id  $name"
    jobs+=("$id")
}
one() { submit "soda-$(basename "$1" .yaml | sed 's/_rtmdet-p2//; s/-6e-soda//')${2:+-s$2}" one "$@"; }

groups=("$@")
[[ ${#groups[@]} == 0 || ${groups[0]} == all ]] && groups=(bench phase1 phase2 lora dcfl noinj seeds)
for g in "${groups[@]}"; do
    case $g in
        bench) submit soda-bench bench ;;
        phase1)
            one $CFG/vits-dinov3-comer_rtmdet-p2-6e-soda.yaml
            one $CFG/vits-dinov3-vitdet_rtmdet-p2-6e-soda.yaml
            one $CFG/convnext-t-dinov3_rtmdet-p2-6e-soda.yaml
            ;;
        phase2)
            one $CFG/vits-dinov3-comer_rtmdet-p2-nwd-rka-6e-soda.yaml
            one $CFG/vits-dinov3-comer_rtmdet-p2-nwd-dsl-rka-6e-soda.yaml
            ;;
        noinj) one $CFG/vits-dinov3-comer-noinj_rtmdet-p2-6e-soda.yaml ;;
        dcfl)
            one $CFG/vits-dinov3-comer_rtmdet-p2-dcfl-6e-soda.yaml
            one $CFG/convnext-t-dinov3_rtmdet-p2-dcfl-6e-soda.yaml
            ;;
        lora)
            one $CFG/vits-dinov3-comer-lora_rtmdet-p2-6e-soda.yaml
            one $CFG/vits-dinov3-vitdet-lora_rtmdet-p2-6e-soda.yaml
            ;;
        seeds)
            one $CFG/vits-dinov3-vitdet_rtmdet-p2-6e-soda.yaml 1
            one $CFG/convnext-t-dinov3_rtmdet-p2-6e-soda.yaml 1
            ;;
        *) echo "unknown group $g"; exit 1 ;;
    esac
done

dep=$(IFS=:; echo "${jobs[*]}")
[[ ${COLLECT:-1} == 0 ]] && exit 0
id=$(sbatch --parsable "${common[@]}" --job-name=soda-collect --time=01:00:00 \
    --dependency="afterany:$dep" tools/soda_dinov3/slurm_run.sbatch collect)
echo "$id  soda-collect (after $dep)"
