#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.
# Queue the CoMer-lite sensitivity runs (configs/comer_lite) as slurm jobs, one
# GPU each, on this PC (node `may`), reusing the SODA-A DINOv3 job script:
# train 6 epochs, test the EMA checkpoint and dump test predictions. Then a
# `falsealarms` and a `collect` job run once all of them have ended. Finished
# runs are skipped inside the jobs, so re-submitting after a failure is safe.
#
#   tools/comer_lite/slurm_submit.sh [round1|round2|bench] ...
#
#   round1  A2-A5, one change each: extra0, ratio05, cfinter-late, slim
#   round2  lite (all changes; set extra_num in its config from the extra0 result)
#   bench   cost: bench.py + profile_modules.py on every variant and the baseline
#
# The finished CoMer and CoMer-noinj runs are symlinked from work_dirs/soda_dinov3,
# so the results and false-alarm tables include them without retraining.
#
# Env (defaults as tools/soda_dinov3/slurm_submit.sh): PARTITION, NODELIST, IMAGE,
# WORK, DATA_ROOT, WEIGHTS, TIME, COLLECT=0 to skip the falsealarms/collect jobs.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
CFG=configs/comer_lite
# Same image as the baseline runs.
export IMAGE=${IMAGE:-artifactory.project.dso/srsep-docker/voltronix-mmrotate-dev:2026.10.19}
export WORK=${WORK:-work_dirs/comer_lite}
export DATA_ROOT=${DATA_ROOT:-/mnt/zfs/datasets/eo/soda-a/split_800/}
export WEIGHTS=${WEIGHTS:-/mnt/zfs/models/timm}
export DOCKER_ARGS=${DOCKER_ARGS:---entrypoint= -e PYTHONPATH=/mmrotate}
PARTITION=${PARTITION:-sep}
NODELIST=${NODELIST:-may}
TIME=${TIME:-12:00:00}

mkdir -p "$WORK/slurm"
for run in vits-dinov3-comer_rtmdet-p2-6e-soda vits-dinov3-comer-noinj_rtmdet-p2-6e-soda; do
    [[ -e $WORK/$run ]] || ln -s "../soda_dinov3/$run" "$WORK/$run"
done

common=(--partition="$PARTITION" --export=ALL --output="$WORK/slurm/%x-%j.out")
[[ -n $NODELIST ]] && common+=(--nodelist="$NODELIST")
jobs=()

# submit <job name> <script> <args...>
submit() {
    local name=$1 script=$2
    shift 2
    local id
    id=$(sbatch --parsable "${common[@]}" --job-name="$name" --gres=gpu:1 --time="$TIME" \
        "$script" "$@")
    echo "$id  $name"
    jobs+=("$id")
}
one() {
    local cfg=$CFG/vits-dinov3-comer-$1_rtmdet-p2-6e-soda.yaml
    [[ -f $cfg ]] || { echo "no config $cfg"; exit 1; }
    submit "comer-$1" tools/soda_dinov3/slurm_run.sbatch one "$cfg"
}

for g in "${@:-round1}"; do
    case $g in
        round1) for v in extra0 ratio05 cfinter-late slim; do one $v; done ;;
        round2) one lite ;;
        bench) TIME=02:00:00 submit comer-bench tools/comer_lite/slurm_bench.sbatch ;;
        *) echo "unknown group $g"; exit 1 ;;
    esac
done

[[ ${COLLECT:-1} == 0 ]] && exit 0
dep=$(IFS=:; echo "${jobs[*]}")
id=$(sbatch --parsable "${common[@]}" --job-name=comer-falsealarms --time=04:00:00 \
    --dependency="afterany:$dep" tools/soda_dinov3/slurm_run.sbatch falsealarms)
echo "$id  comer-falsealarms (after $dep)"
id=$(sbatch --parsable "${common[@]}" --job-name=comer-collect --time=01:00:00 \
    --dependency="afterany:$id" tools/soda_dinov3/slurm_run.sbatch collect)
echo "$id  comer-collect"
