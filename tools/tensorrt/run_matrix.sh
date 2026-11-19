#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.
# Convert, evaluate and benchmark one trained detector in every TensorRT variant
# (fp16, fp8, INT8 plain / GPTQ / AWQ, Ampere-portable builds), inside the dev image.
# Run through slurm, one GPU (see mmrotate/tensorrt/TENSORRT.md, "Results"):
#
#   sbatch --nodelist=may --gres=gpu:1 tools/tensorrt/slurm_run.sbatch \
#       tools/tensorrt/run_matrix.sh build work_dirs/soda_dinov3/<run> work_dirs/trt_blackwell
#   # on another GPU, with the engines built above:
#   sbatch --nodelist=hammond --gres=gpu:1 tools/tensorrt/slurm_run.sbatch \
#       tools/tensorrt/run_matrix.sh portable work_dirs/soda_dinov3/<run> work_dirs/trt_blackwell
#
# RUN_DIR holds <run>.yaml and epoch_6.pth (CKPT=... to override). Results go to
# OUT_ROOT/<run>/<variant>/ (deploy dir, convert.log, test.log) and OUT_ROOT/<run>/bench.json;
# `portable` writes OUT_ROOT/<run>/<gpu tag>/. Finished steps are skipped on a re-run, and a
# failing variant is logged and skipped.
set -uo pipefail

MODE=$1
RUN_DIR=${2%/}
OUT_ROOT=${3%/}
NAME=$(basename "$RUN_DIR")
CONFIG=$RUN_DIR/$NAME.yaml
CKPT=${CKPT:-$RUN_DIR/epoch_6.pth}
OUT=$OUT_ROOT/$NAME
BATCH=${BATCH:-8}
# Evaluate on val_mini (2588 patches), as the other TensorRT result tables.
EVAL_OPTS=(test_dataloader.dataset.ann_file=val_mini/annfiles/
    test_dataloader.dataset.data_prefix.img_path=val_mini/images/)
mkdir -p "$OUT"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# variant <name> <convert.py args...>
variant() {
    local name=$1; shift
    local dir=$OUT/$name
    if [[ -f $dir/.converted ]]; then
        log "$name: already converted"
    elif [[ -f $dir/.failed ]]; then
        log "$name: failed before (rm $dir/.failed to retry)"
        return
    else
        mkdir -p "$dir"
        log "$name: converting ($*)"
        if python tools/tensorrt/convert.py "$CONFIG" "$CKPT" --out-dir "$dir" \
                --batch-size "$BATCH" "$@" > "$dir/convert.log" 2>&1; then
            touch "$dir/.converted"
        else
            log "$name: convert FAILED, see $dir/convert.log"; tail -5 "$dir/convert.log"
            touch "$dir/.failed"
            return
        fi
    fi
    evaluate "$dir"
}

# evaluate <deploy dir>: val_mini mAP with test_trt.py
evaluate() {
    local dir=$1
    [[ -f $dir/.tested ]] && return
    log "$(basename "$dir"): test_trt"
    if python tools/tensorrt/test_trt.py "$dir" --work-dir "$dir/test" \
            --cfg-options "${EVAL_OPTS[@]}" > "$dir/test.log" 2>&1; then
        touch "$dir/.tested"
        grep -o "[a-z]*/mAP: [0-9.]*" "$dir/test.log" | tail -1
    else
        log "$(basename "$dir"): test_trt FAILED, see $dir/test.log"; tail -5 "$dir/test.log"
    fi
}

converted() {  # deploy dirs of OUT (or $1) that built
    local d
    for d in "${1:-$OUT}"/*/; do [[ -f $d.converted ]] && echo "${d%/}"; done
}

build() {
    nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
    if [[ ! -f $OUT/torch/.tested ]]; then
        log "torch fp32: tools/test.py"
        mkdir -p "$OUT/torch"
        python tools/test.py "$CONFIG" "$CKPT" --work-dir "$OUT/torch" \
            --cfg-options "${EVAL_OPTS[@]}" > "$OUT/torch/test.log" 2>&1 \
            && touch "$OUT/torch/.tested"
        grep -o "[a-z]*/mAP: [0-9.]*" "$OUT/torch/test.log" | tail -1
    fi
    variant fp16 --precision fp16 --keep-onnx
    variant fp16_ampereplus --precision fp16 --hardware-compat ampere+ --keep-onnx
    variant fp16_samecc --precision fp16 --hardware-compat same-cc
    variant fp8 --precision fp8
    variant int8 --precision int8
    variant int8_gptq --precision int8 --quant-algo gptq
    variant int8_awq --precision int8 --quant-algo awq-scale --keep-onnx
    variant int8_awq_gptq --precision int8 --quant-algo awq-scale,gptq
    variant int8_awq_ampereplus --precision int8 --quant-algo awq-scale \
        --hardware-compat ampere+ --keep-onnx
    if [[ $NAME == convnext* ]]; then
        variant int8_awq_mixed --precision int8 --quant-algo awq-scale --quant-auto-exclude 0.05
    fi

    mapfile -t dirs < <(converted)
    if [[ ! -f $OUT/bench.json ]]; then
        log "bench (${#dirs[@]} engines + torch)"
        python tools/tensorrt/bench.py "$CONFIG" "$CKPT" "${dirs[@]}" --e2e \
            --out "$OUT/bench.json" > "$OUT/bench.log" 2>&1 || log "bench FAILED, see $OUT/bench.log"
        grep '^|' "$OUT/bench.log"
    fi
    for v in fp16 fp16_ampereplus; do
        [[ -f $OUT/$v/.converted && ! -f $OUT/$v/profile.json ]] || continue
        log "profile $v"
        python tools/tensorrt/profile_engine.py "$OUT/$v" --out "$OUT/$v/profile.json" \
            > "$OUT/$v/profile.log" 2>&1 || log "profile $v FAILED"
    done
}

# On another GPU: run the portable engines as they are, try a native one, rebuild per GPU.
portable() {
    local cc tag dst
    nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
    tag=sm$cc
    dst=$OUT/$tag
    mkdir -p "$dst"
    # 1. The native engine of another GPU: expected to fail to load.
    log "native fp16 engine of the build GPU on $tag"
    python - "$OUT/fp16" > "$dst/native_fp16_load.log" 2>&1 <<'EOF'
import sys
import torch
from mmrotate.tensorrt import init_trt_detector
model = init_trt_detector(sys.argv[1], 'cuda:0')
shape = model.meta['input_shape']
outs = model._forward(torch.zeros(shape, device='cuda:0'))
print('loaded and ran')
EOF
    echo "exit code $?" >> "$dst/native_fp16_load.log"
    tail -3 "$dst/native_fp16_load.log"
    # 2. The ampere+ engines of the build GPU, as they are (engine, meta and config linked;
    # logs and results of this GPU are written here, not through the links).
    local v f
    for v in fp16_ampereplus int8_awq_ampereplus; do
        [[ -f $OUT/$v/.converted ]] || continue
        mkdir -p "$dst/$v"
        for f in end2end.engine trt_meta.json deploy_cfg.py; do
            ln -sfr "$OUT/$v/$f" "$dst/$v/$f"
        done
        touch "$dst/$v/.converted"
        evaluate "$dst/$v"
    done
    # 3. Native engines for this GPU, rebuilt from the kept ONNX.
    for v in fp16 int8_awq; do
        [[ -f $OUT/$v/.converted && ! -f $dst/${v}_rebuilt/.converted ]] || continue
        log "rebuild $v for $tag"
        if python tools/tensorrt/build_engine.py "$OUT/$v" --out-dir "$dst/${v}_rebuilt" \
                > "$dst/${v}_rebuilt.build.log" 2>&1; then
            touch "$dst/${v}_rebuilt/.converted"
        else
            log "rebuild $v FAILED"; tail -5 "$dst/${v}_rebuilt.build.log"
        fi
        [[ -f $dst/${v}_rebuilt/.converted ]] && evaluate "$dst/${v}_rebuilt"
    done
    mapfile -t dirs < <(converted "$dst")
    if [[ ! -f $dst/bench.json ]]; then
        log "bench on $tag (${#dirs[@]} engines + torch)"
        python tools/tensorrt/bench.py "$CONFIG" "$CKPT" "${dirs[@]}" --e2e \
            --out "$dst/bench.json" > "$dst/bench.log" 2>&1 || log "bench FAILED, see $dst/bench.log"
        grep '^|' "$dst/bench.log"
    fi
}

case $MODE in
    build) build ;;
    portable) portable ;;
    *) echo "usage: $0 build|portable RUN_DIR OUT_ROOT"; exit 2 ;;
esac
log done
