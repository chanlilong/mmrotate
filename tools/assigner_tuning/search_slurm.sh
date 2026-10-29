#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.
# Run assigner_search.py in the dev container with the host's slurm client
# mounted, so that every trial is its own `sbatch --wait` job
# (slurm_train.sbatch, one GPU). The image has no slurm, and the host has no
# torch. Defaults are for the voltronix cluster (Ubuntu host, configless slurm,
# munge); elsewhere run assigner_search.py wherever both mmrotate and sbatch
# work, with the --cmd below.
#
#   tools/assigner_tuning/search_slurm.sh CONFIG [assigner_search.py args...]
#
# The repo is mounted at its host path, so the trial paths are valid on the
# slurm nodes too. Keep --work-dir relative (it is inside the repo).
# Env: IMAGE, PARTITION, MAX_PARALLEL (trials queued at once), TIME (per trial).
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
export IMAGE=${IMAGE:-artifactory.project.dso/srsep-docker/voltronix-mmrotate-dev:2026.10.19}
PARTITION=${PARTITION:-sep}
MAX_PARALLEL=${MAX_PARALLEL:-4}
TIME=${TIME:-06:00:00}
LIB=/usr/lib/x86_64-linux-gnu
CMD="sbatch --wait --partition=$PARTITION --time=$TIME --job-name=assigner-search"
CMD+=" --output={work_dir}/slurm-%j.out tools/assigner_tuning/slurm_train.sbatch {config} {work_dir}"

docker run --rm -i --entrypoint= --network host --user "$(id -u):$(id -g)" \
    -e HOME=/tmp -e PYTHONPATH="$REPO" -e IMAGE \
    -e SLURM_CONF=/slurmconf/slurm.conf -e LD_LIBRARY_PATH=/hostlib \
    -v /var/spool/slurmd/conf-cache:/slurmconf:ro \
    -v "$(readlink -f $LIB/libmunge.so.2)":/hostlib/libmunge.so.2:ro \
    -v $LIB/slurm:$LIB/slurm:ro -v /run/munge:/run/munge \
    -v /usr/bin/sbatch:/usr/bin/sbatch:ro \
    -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
    -v "$REPO":"$REPO" -v /mnt:/mnt -w "$REPO" "$IMAGE" \
    python tools/assigner_tuning/assigner_search.py "$@" \
    --cmd "$CMD" --max-parallel "$MAX_PARALLEL"
