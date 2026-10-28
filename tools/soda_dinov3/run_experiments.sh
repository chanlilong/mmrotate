#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.
# Run the SODA-A DINOv3 experiments (configs/soda_dinov3/README.md) inside the
# dev container: train, then evaluate the final (EMA) checkpoint on the test
# split. Finished runs (test metrics present) are skipped, so the script can be
# re-run after an interruption.
#
#   tools/soda_dinov3/run_experiments.sh bench    # cost / params table
#   tools/soda_dinov3/run_experiments.sh phase1   # ViTDet, ConvNeXt-T, CoMer
#   tools/soda_dinov3/run_experiments.sh noinj    # CoMer without CNN->ViT injector
#   tools/soda_dinov3/run_experiments.sh seeds    # seed-1 repeats (noise)
#   tools/soda_dinov3/run_experiments.sh lora     # CoMer / ViTDet with ViT LoRA
#   tools/soda_dinov3/run_experiments.sh dcfl     # CoMer / ConvNeXt-T with DCFL
#   tools/soda_dinov3/run_experiments.sh phase2   # NWD-RKA / NWD-DSL-RKA
#   tools/soda_dinov3/run_experiments.sh collect  # markdown results
#   tools/soda_dinov3/run_experiments.sh one <config> [seed]  # a single run
#   tools/soda_dinov3/run_experiments.sh dump <config> [seed] # test predictions -> <run>/preds_test.pkl
#   tools/soda_dinov3/run_experiments.sh falsealarms  # false-positive analysis of the dumps
#
# Env: IMAGE (docker image), WORK (work dir root), GPU (device id or UUID),
# CFG_OPTIONS (extra --cfg-options for train/test/bench, e.g. local data and
# weight paths), DOCKER_ARGS (extra `docker run` arguments).
# tools/soda_dinov3/slurm_submit.sh queues every run as its own slurm job.
set -euo pipefail

IMAGE=${IMAGE:-mmrotate-dev:2026.10.24}
WORK=${WORK:-work_dirs/soda_dinov3}
GPU=${GPU:-0}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
CFG=configs/soda_dinov3
read -r -a CFG_OPTS <<< "${CFG_OPTIONS:-}"
read -r -a DOCKER_EXTRA <<< "${DOCKER_ARGS:-}"

VITDET=$CFG/vits-dinov3-vitdet_rtmdet-p2-6e-soda.yaml
COMER=$CFG/vits-dinov3-comer_rtmdet-p2-6e-soda.yaml
COMER_NOINJ=$CFG/vits-dinov3-comer-noinj_rtmdet-p2-6e-soda.yaml
CONVNEXT=$CFG/convnext-t-dinov3_rtmdet-p2-6e-soda.yaml
COMER_LORA=$CFG/vits-dinov3-comer-lora_rtmdet-p2-6e-soda.yaml
VITDET_LORA=$CFG/vits-dinov3-vitdet-lora_rtmdet-p2-6e-soda.yaml
COMER_DCFL=$CFG/vits-dinov3-comer_rtmdet-p2-dcfl-6e-soda.yaml
CONVNEXT_DCFL=$CFG/convnext-t-dinov3_rtmdet-p2-dcfl-6e-soda.yaml
NWD_RKA=$CFG/vits-dinov3-comer_rtmdet-p2-nwd-rka-6e-soda.yaml
NWD_DSL_RKA=$CFG/vits-dinov3-comer_rtmdet-p2-nwd-dsl-rka-6e-soda.yaml

in_docker() {
    docker run --rm --gpus "device=$GPU" --ipc=host --shm-size=16g \
        --user "$(id -u):$(id -g)" -e HOME=/tmp \
        -v "$REPO":/mmrotate -v /mnt:/mnt -w /mmrotate "${DOCKER_EXTRA[@]}" "$IMAGE" "$@"
}

# train_test <config> [seed]
train_test() {
    local cfg=$1 seed=${2:-0}
    local name
    name=$(basename "$cfg" .yaml)
    [[ $seed != 0 ]] && name=${name}_seed${seed}
    local wd=$WORK/$name
    if compgen -G "$wd/test/*/*.json" >/dev/null; then
        echo "[skip] $name (test metrics present)"
        return
    fi
    echo "[train] $name ($(date))"
    # Retry a crashed training (e.g. an intermittent native segfault) from the
    # latest checkpoint; --resume starts from scratch when there is none yet.
    local attempt rc resume=()
    for attempt in 1 2 3; do
        in_docker python tools/train.py "$cfg" --work-dir "$wd" "${resume[@]}" \
            --cfg-options randomness.seed="$seed" "${CFG_OPTS[@]}" && break
        rc=$?
        echo "[retry] $name: train exited with $rc (attempt $attempt, $(date))"
        [[ $attempt == 3 ]] && return "$rc"
        resume=(--resume)
    done
    echo "[test] $name ($(date))"
    in_docker python tools/test.py "$cfg" "$wd/epoch_6.pth" --work-dir "$wd/test" \
        --out "$wd/preds_test.pkl" ${CFG_OPTS[@]:+--cfg-options "${CFG_OPTS[@]}"}
}

# dump <config> [seed]: test-split predictions of a finished run (for false_alarms.py),
# evaluated in <run>/dump so collect_results.py does not see a second test run.
dump() {
    local cfg=$1 seed=${2:-0}
    local name
    name=$(basename "$cfg" .yaml)
    [[ $seed != 0 ]] && name=${name}_seed${seed}
    local wd=$WORK/$name
    if [[ -f $wd/preds_test.pkl ]]; then
        echo "[skip] $name (predictions present)"
        return
    fi
    echo "[dump] $name ($(date))"
    in_docker python tools/test.py "$cfg" "$wd/epoch_6.pth" --work-dir "$wd/dump" \
        --out "$wd/preds_test.pkl" ${CFG_OPTS[@]:+--cfg-options "${CFG_OPTS[@]}"}
}

case ${1:-} in
    bench)
        in_docker python tools/soda_dinov3/bench.py "$VITDET" "$CONVNEXT" "$COMER" "$COMER_NOINJ" \
            "$COMER_LORA" "$VITDET_LORA" \
            --out "$WORK/bench.json" ${CFG_OPTS[@]:+--cfg-options "${CFG_OPTS[@]}"}
        ;;
    phase1)
        train_test "$VITDET"
        train_test "$CONVNEXT"
        train_test "$COMER"
        ;;
    noinj)
        train_test "$COMER_NOINJ"
        ;;
    seeds)
        train_test "$VITDET" 1
        train_test "$CONVNEXT" 1
        ;;
    lora)
        train_test "$COMER_LORA"
        train_test "$VITDET_LORA"
        ;;
    dcfl)
        train_test "$COMER_DCFL"
        train_test "$CONVNEXT_DCFL"
        ;;
    phase2)
        train_test "$NWD_RKA"
        train_test "$NWD_DSL_RKA"
        ;;
    one)
        train_test "$2" "${3:-0}"
        ;;
    dump)
        dump "$2" "${3:-0}"
        ;;
    falsealarms)
        in_docker python tools/soda_dinov3/false_alarms.py "$WORK"
        ;;
    collect)
        in_docker python tools/soda_dinov3/collect_results.py "$WORK"
        ;;
    *)
        sed -n '3,23p' "$0"
        exit 1
        ;;
esac
